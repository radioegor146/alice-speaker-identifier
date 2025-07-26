import os
import numpy as np

def getenv_required(key: str) -> str:
    value = os.getenv(key)
    if value is None:
        raise EnvironmentError(f"Environment variable {key} not set")
    return value

def f32_samples_to_s16_bytes(samples: np.ndarray) -> bytes:
    return np.clip(samples * 32768, -32768, 32767).astype(np.int16).tobytes()

def s16_bytes_to_f32_samples(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0