import numpy as np
from models.base import BaseEmbeddingModel
from vosk import Model, KaldiRecognizer, SpkModel
from utils import getenv_required, f32_samples_to_s16_bytes
import json


class VoskEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, sample_rate: int, model_path: str, spk_model_path: str):
        super().__init__(sample_rate)
        self.vosk_model = Model(model_path)
        self.spk_model = SpkModel(spk_model_path)

    def extract_embeddings(self, samples: np.ndarray) -> np.ndarray | None:
        rec = KaldiRecognizer(self.vosk_model, self.sample_rate)
        rec.SetSpkModel(self.spk_model)
        rec.AcceptWaveform(f32_samples_to_s16_bytes(samples))
        res = json.loads(rec.FinalResult())
        spk = res.get("spk")
        if isinstance(spk, dict) and "spk" in spk:
            vec = spk["spk"]
        elif isinstance(spk, list):
            vec = spk
        else:
            return None
        try:
            return np.asarray(vec, dtype=np.float32)
        except:
            return None


def create_vosk_embedding_model_from_env(sample_rate: int) -> BaseEmbeddingModel:
    return VoskEmbeddingModel(sample_rate, getenv_required("VOSK_MODEL_PATH"), getenv_required("VOSK_SPK_MODEL_PATH"))
