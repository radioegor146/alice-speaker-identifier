import json
import os
import sys
import numpy as np
import soundfile

from models.nemo_enc_dec_embedder import create_nemo_enc_dec_embedder_from_env
from models.vosk_embedder import create_vosk_embedding_model_from_env
from models.yandex_ecapa_tdnn_embedder import create_yandex_ecapa_tdnn_embedding_model_from_env
from utils import s16_bytes_to_f32_samples

GAIN = float(os.getenv("GAIN", "200"))
EMBEDDER = os.getenv("EMBEDDER", "yandex-ecapa-tdnn")

sample_rate = 16000

embedders = {
    "nemo-enc-dec": create_nemo_enc_dec_embedder_from_env,
    "vosk": create_vosk_embedding_model_from_env,
    "yandex-ecapa-tdnn": create_yandex_ecapa_tdnn_embedding_model_from_env,
}

if not EMBEDDER in embedders:
    raise EnvironmentError(f"Embedder {EMBEDDER} does not exist")

model = embedders[EMBEDDER](sample_rate)


def convert_and_normalize_input(raw: np.ndarray) -> np.ndarray:
    return raw / np.max(raw) * GAIN


wav_file_name = sys.argv[1]

samples, data_sample_rate = soundfile.read(wav_file_name)
if data_sample_rate != sample_rate:
    raise ValueError(f"Sample rate {sample_rate} does not match sample rate {data_sample_rate}")

print(json.dumps(model.extract_embeddings(convert_and_normalize_input(samples)).tolist()))
