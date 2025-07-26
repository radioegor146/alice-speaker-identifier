import os
import logging
import uuid
import json
import wave
from cachetools import TTLCache
import numpy as np
from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from models.ecapa_tdnn_embedder import create_ecapa_tdnn_embedder_from_env
from models.vosk_embedder import create_vosk_embedding_model_from_env
from utils import f32_samples_to_s16_bytes, s16_bytes_to_f32_samples

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9010"))
AUDIO_CACHE_SIZE = int(os.getenv("AUDIO_CACHE_SIZE", "50"))
AUDIO_CACHE_TTL_SECONDS = int(os.getenv("AUDIO_CACHE_TTL_SECONDS", "600"))
VOICE_ENROLLMENT_CACHE_SIZE = int(os.getenv("VOICE_ENROLLMENT_CACHE_SIZE", "100"))
VOICE_ENROLLMENT_CACHE_TTL_SECONDS = int(os.getenv("VOICE_ENROLLMENT_CACHE_TTL_SECONDS", "1800"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))
SPEAKER_EMBEDDINGS_FILE = os.getenv("SPEAKER_EMBEDDINGS_FILE", "speaker_embeddings.json")
VOICEPRINTS_DIR = os.getenv("VOICEPRINTS_DIR")
GAIN = float(os.getenv("GAIN", "90"))
EMBEDDER = os.getenv("EMBEDDER", "ecapa-tdnn")

sample_rate = 16000

embedders = {
    "ecapa-tdnn": create_ecapa_tdnn_embedder_from_env,
    "vosk": create_vosk_embedding_model_from_env
}

if not EMBEDDER in embedders:
    raise EnvironmentError(f"Embedder {EMBEDDER} does not exist")

model = embedders[EMBEDDER](sample_rate)

audio_cache = TTLCache(maxsize=AUDIO_CACHE_SIZE, ttl=AUDIO_CACHE_TTL_SECONDS)
voice_enrollment_cache = TTLCache(maxsize=VOICE_ENROLLMENT_CACHE_SIZE, ttl=VOICE_ENROLLMENT_CACHE_TTL_SECONDS)

speaker_embeddings = {}

logger = logging.getLogger("uvicorn.access")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

app = FastAPI()


class SessionContext(BaseModel):
    id: str
    metadata: dict


class GetFunctionsOrStatePayload(BaseModel):
    context: SessionContext


class FunctionCallPayload(BaseModel):
    context: SessionContext
    name: str
    parameters: dict


def save_voice_sample_enrollment(context: SessionContext):
    meta = context.metadata or {}
    record_id = meta.get("recordId")
    if record_id is None:
        logger.warning(f"save_voice_sample_enrollment missing recordId: context_id={context.id}, metadata={meta}")
        return
    sample = audio_cache.get(record_id)
    if sample is None:
        logger.warning(
            f"save_voice_sample_enrollment record not found or expired: context_id={context.id}, recordId={record_id}")
        return
    samples = voice_enrollment_cache.get(context.id) or []
    samples.append(sample)
    voice_enrollment_cache[context.id] = samples
    logger.info(
        f"save_voice_sample_enrollment context_id={context.id} recordId={record_id} samples_count={len(samples)}")


def save_voiceprint_wav(context_id: str, samples: np.ndarray):
    if VOICEPRINTS_DIR is None:
        return
    try:
        os.makedirs(VOICEPRINTS_DIR, exist_ok=True)
        path = os.path.join(VOICEPRINTS_DIR, f"{context_id}.wav")
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(f32_samples_to_s16_bytes(samples))
        logger.info(f"voiceprint_saved path={path} context_id={context_id}")
    except Exception as e:
        logger.exception(f"voiceprint_save_failed context_id={context_id}: {e}")


def finish_voice_sample_enrollment(context: SessionContext, comment: str):
    save_voice_sample_enrollment(context)
    samples = voice_enrollment_cache.get(context.id) or []
    if not samples:
        logger.warning(f"finish_voice_sample_enrollment no samples: context_id={context.id} comment={comment}")
        return
    data = np.concatenate(samples)
    save_voiceprint_wav(context.id, data)
    emb = model.extract_embeddings(data)
    if emb is None:
        logger.warning(f"finish_voice_sample_enrollment embedding_failed: context_id={context.id} comment={comment}")
        return
    speaker_embeddings[context.id] = (emb, comment)
    save_speaker_embeddings_to_file()
    voice_enrollment_cache.pop(context.id, None)
    logger.info(
        f"finish_voice_sample_enrollment context_id={context.id} samples_count={len(samples)} comment={comment}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    print(a.shape, b.shape)
    an = np.linalg.norm(a)
    bn = np.linalg.norm(b)
    if an == 0 or bn == 0:
        return 0.0
    return float(np.dot(a, b) / (an * bn))


def match_speaker(embedding: np.ndarray) -> (str, float):
    if not speaker_embeddings:
        return None, None
    best_id = None
    best_sim = -1.0
    for sid, ref in speaker_embeddings.items():
        ref_emb = ref[0]
        sim = cosine_similarity(embedding, ref_emb)
        if sim > best_sim:
            best_sim = sim
            best_id = sid
    if best_sim >= SIMILARITY_THRESHOLD:
        return best_id, best_sim
    return None, best_sim


def save_speaker_embeddings_to_file():
    try:
        data = {sid: {"embedding": val[0].tolist(), "comment": val[1]} for sid, val in speaker_embeddings.items()}
        with open(SPEAKER_EMBEDDINGS_FILE, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.exception(f"failed to save speaker embeddings: {e}")


def load_speaker_embeddings_from_file():
    try:
        with open(SPEAKER_EMBEDDINGS_FILE, "r") as f:
            data = json.load(f)
        for sid, entry in data.items():
            emb_list = entry.get("embedding")
            comment = entry.get("comment")
            if emb_list is None:
                continue
            emb = np.asarray(emb_list, dtype=np.float32)
            speaker_embeddings[sid] = (emb, comment)
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.exception(f"failed to load speaker embeddings: {e}")


load_speaker_embeddings_from_file()


def convert_and_normalize_input(raw: bytes) -> np.ndarray:
    return s16_bytes_to_f32_samples(raw) * GAIN


@app.post("/audio-metadata", response_class=JSONResponse)
async def audio_metadata(request: Request, req_sample_rate: int = Query(..., alias="sample_rate")):
    data = await request.body()
    length = len(data)
    if req_sample_rate != sample_rate:
        logger.info(f"Invalid sample_rate={req_sample_rate}, must be {sample_rate}, body_length={length}")
        return JSONResponse(status_code=500, content={"error": "sample_rate must be 16000"})
    processed = convert_and_normalize_input(data)
    record_id = str(uuid.uuid4())
    audio_cache[record_id] = processed
    speaker_id = None
    similarity = None
    try:
        emb = model.extract_embeddings(processed)
        if emb is not None:
            speaker_id, similarity = match_speaker(emb)
    except Exception as e:
        logger.exception(f"recordId={record_id} speaker processing failed: {e}")
    speaker_label = speaker_id if speaker_id is not None else "unknown"
    similarity_label = f" similarity={similarity:.4f}" if similarity is not None else ""
    logger.info(f"recordId={record_id} speakerId={speaker_label}{similarity_label} body_length={length}")
    return {"recordId": record_id, "speakerId": speaker_id}


@app.post("/functions", response_class=JSONResponse)
async def list_functions():
    return {
        "save_voice_sample_enrollment": {
            "description": "saves voice sample for current voiceprint enrollment session",
            "arguments": {}
        },
        "finish_voice_sample_enrollment": {
            "description": "finishes voice sample enrollment for current voiceprint enrollment session",
            "arguments": {
                "comment": {
                    "description": "comment for voiceprint (name for example)",
                    "constraints": {
                        "type": "string-not-empty",
                        "argumentType": "string"
                    }
                }
            }
        }
    }


@app.patch("/functions")
async def update_function(payload: FunctionCallPayload):
    try:
        name = payload.name
        params = payload.parameters or {}
        if name == "save_voice_sample_enrollment":
            save_voice_sample_enrollment(payload.context)
            return Response(status_code=200)
        if name == "finish_voice_sample_enrollment":
            comment = params.get("comment")
            if comment is None or (isinstance(comment, str) and comment.strip() == ""):
                return JSONResponse(status_code=400, content={"error": "comment is required"})
            finish_voice_sample_enrollment(payload.context, str(comment))
            return Response(status_code=200)
        return JSONResponse(status_code=400, content={"error": "unknown function"})
    except Exception as e:
        logger.exception(f"function call failed: {e}")
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/saved-embeddings", response_class=JSONResponse)
async def get_saved_embeddings():
    items = [{"id": sid, "comment": val[1]} for sid, val in speaker_embeddings.items()]
    return {"embeddings": items}


@app.post("/state", response_class=JSONResponse)
async def get_state(payload: GetFunctionsOrStatePayload):
    meta = payload.context.metadata or {}
    voice_id = meta.get("speakerId")
    if not voice_id:
        value = "voice not saved and unknown"
    else:
        val = speaker_embeddings.get(voice_id)
        if not val:
            value = "voice not saved and unknown"
        else:
            value = val[1]
    return {
        "voiceprint_user_comment": {
            "description": "comment of user voiceprint if they have saved their voice before",
            "value": value
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
