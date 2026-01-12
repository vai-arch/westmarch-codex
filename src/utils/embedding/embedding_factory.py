from src.utils.embedding.ollama_embedding_manager import OllamaEmbeddingManager
from src.utils.embedding.sentence_transformer_embedding_manager import SentenceTransformerEmbeddingManager

EMBEDDING_MANAGERS = {
    "ollama": OllamaEmbeddingManager,
    "sentence_transformer": SentenceTransformerEmbeddingManager,
}


def create_embedding_manager(config_dict: dict):
    embedding_manager_name = config_dict["EMBEDDING_MANAGER"]

    if embedding_manager_name not in EMBEDDING_MANAGERS:
        raise ValueError(f"Unknown embedding backend: {embedding_manager_name}")

    return EMBEDDING_MANAGERS[embedding_manager_name](config_dict)
