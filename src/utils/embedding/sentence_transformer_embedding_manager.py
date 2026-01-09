from datetime import datetime
from typing import List, Tuple

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils.embedding.base_embedding_manager import BaseEmbeddingManager


class SentenceTransformerEmbeddingManager(BaseEmbeddingManager):
    def __init__(self, config: dict):
        self.config = config

        self.method = config["EMBEDDING_METHOD"]
        self.batch_size = config["EMBEDDING_BATCH_SIZE"]

        self.model_name = config["EMBEDDING_MODEL"]["EMBEDDING_MODEL_NAME"]
        self.embedding_model_dimension = config["EMBEDDING_MODEL"]["EMBEDDING_MODEL_DIMENSION"]
        self.embedding_model_max_tokens = config["EMBEDDING_MODEL"]["EMBEDDING_MODEL_MAX_TOKENS"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.model_name, device=self.device, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        print("✅ SentenceTransformer loaded")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Embedding dim: {self.model.get_sentence_embedding_dimension()}")

        if self.model.get_sentence_embedding_dimension() != self.embedding_model_dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_model_dimension}, got {self.model.get_sentence_embedding_dimension()}")

    def get_manager_config(self) -> dict:
        return self.config

    def test_connection(self) -> bool:
        """
        Local model sanity check
        """
        emb = self.model.encode("test", normalize_embeddings=True)
        if len(emb) != self.embedding_model_dimension:
            raise ValueError("Embedding dimension mismatch during test")
        return True

    # ---------------------------------------------------------
    # Public entry point (mirrors OllamaEmbeddingManager)
    # ---------------------------------------------------------

    def embed_chunks(
        self,
        texts: List[str],
        show_progress: bool = True,
        prefix=None,  # intentionally ignored (kept for interface parity)
    ) -> Tuple[List[List[float]], float, int, float]:
        dispatch = {
            "ONE_BY_ONE": self.embed_one_by_one,
            "BATCH": self.embed_batch,
        }

        try:
            embed_fn = dispatch[self.method]
        except KeyError:
            raise ValueError(f"Unknown embedding method: {self.method}")

        return embed_fn(texts, show_progress)

    # ---------------------------------------------------------
    # ONE BY ONE
    # ---------------------------------------------------------

    def embed_one_by_one(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> Tuple[List[List[float]], float, int, float]:
        start_time = datetime.now()

        embeddings = []
        total_tokens = 0
        max_tokens = 0

        iterator = tqdm(texts, desc="Embedding chunks") if show_progress and len(texts) > 1 else texts

        for text in iterator:
            token_count = len(self.model.tokenize(text)["input_ids"])

            if token_count >= self.embedding_model_max_tokens:
                raise ValueError(f"Maximum number of tokens ({self.embedding_model_max_tokens}) reached. Rechunk input text.")

            emb = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            embeddings.append(emb.tolist())

            total_tokens += token_count
            max_tokens = max(max_tokens, token_count)

            avg_tokens = total_tokens / len(embeddings)

            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(f"MAX: {max_tokens} | AVG: {avg_tokens:.1f}")

        total_time = datetime.now() - start_time

        return embeddings, avg_tokens, max_tokens, total_time

    # ---------------------------------------------------------
    # BATCH MODE
    # ---------------------------------------------------------

    def embed_batch(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> Tuple[List[List[float]], float, int, float]:
        start_time = datetime.now()

        all_embeddings = []
        total_tokens = 0
        max_tokens = 0

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding batches")

        for i in iterator:
            batch_texts = texts[i : i + self.batch_size]

            # Debug: print batch info
            if not batch_texts:
                print(f"⚠️ Empty batch at index {i}")
                continue  # skip empty batches

            for j, t in enumerate(batch_texts):
                if not isinstance(t, str):
                    print(f"⚠️ Non-string chunk at batch {i}, position {j}: {repr(t)}")
                elif t.strip() == "":
                    print(f"⚠️ Empty string chunk at batch {i}, position {j}")

            # token_counts = [len(self.model.tokenize(t)["input_ids"]) for t in batch_texts]
            token_counts = [len(self.tokenizer(t, truncation=False)["input_ids"]) for t in batch_texts]
            for text, count in zip(batch_texts, token_counts):
                if count >= self.embedding_model_max_tokens:
                    raise ValueError(f"Text too long ({count} tokens, max {self.embedding_model_max_tokens}). Snippet: {text[:200]}...")

            embeddings = self.model.encode(
                batch_texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            all_embeddings.extend(embeddings.tolist())

            total_tokens += sum(token_counts)
            max_tokens = max(max_tokens, max(token_counts))
            avg_tokens = total_tokens / len(all_embeddings)

            if hasattr(iterator, "set_postfix_str"):
                iterator.set_postfix_str(f"Proc Items: {len(all_embeddings)} | AVG Tokens: {avg_tokens:.1f}")

        total_time = datetime.now() - start_time

        return all_embeddings, avg_tokens, max_tokens, total_time
