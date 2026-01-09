import re
from typing import List

from scipy.spatial.distance import cosine

from src.config import get_config, get_embedding_manager_config
from src.utils.embedding.embedding_factory import create_embedding_manager

# Global model - loaded once
_embedding_manager = None


def _get_embedding_manager():
    global _embedding_manager
    if _embedding_manager is None:
        config = get_config()
        _embedding_manager = create_embedding_manager(config.EMBEDDING_MANAGER, get_embedding_manager_config(config.EMBEDDING_MANAGER))
    return _embedding_manager


def semantic_chunker_function(text: str, target_tokens: int = 1000, min_tokens: int = 600, overlap_tokens: int = 200, similarity_bonus_threshold: float = 0.82) -> List[str]:
    """
    Size-driven chunking with light semantic protection.
    Fully compatible with embedding_manager returning tuple (embeddings, count).
    """
    if not text.strip():
        return []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # Early returns for short text — no embedding needed
    if len(sentences) == 0:
        return []
    if len(sentences) <= 5:
        return [text.strip()]

    # Get embeddings via manager (no prefix for internal similarity)
    embedding_manager = _get_embedding_manager()  # Assuming global or injectable
    embed_result = embedding_manager.embed_chunks(
        texts=sentences,
        show_progress=False,
        prefix=embedding_manager.get_manager_config()["EMBEDDING_MODEL"]["EMBEDDING_MODEL_RAW_PREFIX"],
    )

    # Safe extraction of embeddings
    if isinstance(embed_result, (tuple, list)):
        embeddings_raw = embed_result[0]
    else:
        embeddings_raw = embed_result

    import numpy as np

    embeddings = np.array(embeddings_raw)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(sentences):
        return [text.strip()]  # Fallback to whole block

    chunks = []
    i = 0
    while i < len(sentences):
        current_sents = []
        current_tokens = 0
        start_i = i

        while i < len(sentences):
            next_sent_tokens = len(sentences[i].split())
            if current_tokens + next_sent_tokens > target_tokens + 50:
                break

            # Semantic guard
            if current_tokens > min_tokens and len(current_sents) > 0 and i > start_i and 1 - cosine(embeddings[i - 1], embeddings[i]) < similarity_bonus_threshold:
                break

            current_sents.append(sentences[i])
            current_tokens += next_sent_tokens
            i += 1

        if not current_sents:
            current_sents = sentences[start_i:]
            i = len(sentences)

        chunk_text = " ".join(current_sents)
        chunks.append(chunk_text)

        # Overlap
        if overlap_tokens > 0 and i < len(sentences):
            words_so_far = 0
            overlap_count = 0
            for sent in reversed(current_sents):
                words_so_far += len(sent.split())
                overlap_count += 1
                if words_so_far >= overlap_tokens:
                    break
            if overlap_count > 0:
                rewind_to = len(sentences) - len(current_sents) + overlap_count
                i = max(i, rewind_to)

    return chunks
