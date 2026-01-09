from src.utils.embedding.sentence_transformer_embedding_manager import SentenceTransformerEmbeddingManager

from .ollama_embedding_manager import OllamaEmbeddingManager

EMBEDDING_MANAGERS = {
    "ollama": OllamaEmbeddingManager,
    "sentence_transformer": SentenceTransformerEmbeddingManager,
}


def create_embedding_manager(embedding_manager: str, config_dict: dict):
    if embedding_manager not in EMBEDDING_MANAGERS:
        raise ValueError(f"Unknown embedding backend: {embedding_manager}")

    return EMBEDDING_MANAGERS[embedding_manager](config_dict)
