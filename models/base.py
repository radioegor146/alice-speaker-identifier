import abc
import numpy as np


class BaseEmbeddingModel(abc.ABC):
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate
        pass

    @abc.abstractmethod
    def extract_embeddings(self, samples: np.ndarray) -> np.ndarray | None:
        pass
