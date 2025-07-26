import numpy as np
import nemo.collections.asr as nemo_asr

from models.base import BaseEmbeddingModel
from utils import getenv_required


class EcapaTdnnEmbedder(BaseEmbeddingModel):
    def __init__(self, sample_rate: int, model_path: str):
        super().__init__(sample_rate)
        self.model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path)

    def extract_embeddings(self, samples: np.ndarray) -> np.ndarray | None:
        emb, logits = self.model.infer_segment(samples)
        return emb.detach().numpy()[0]

def create_ecapa_tdnn_embedder_from_env(sample_rate: int) -> BaseEmbeddingModel:
    return EcapaTdnnEmbedder(sample_rate, getenv_required("ECAPA_TDNN_MODEL_PATH"))
