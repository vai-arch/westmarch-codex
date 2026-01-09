from abc import ABC, abstractmethod
from typing import List, Tuple


class BaseEmbeddingManager(ABC):
    def __init__(self, config_dict: dict):
        self.config = config_dict

    @abstractmethod
    def test_connection(self) -> bool:
        pass

    @abstractmethod
    def get_manager_config(self) -> dict:
        pass

    @abstractmethod
    def embed_chunks(self, texts: List[str], show_progress=True, prefix=None) -> Tuple[List[List[float]], int]:
        """
        Returns:
            embeddings: List[List[float]]
            total_tokens: int
        """
        pass
