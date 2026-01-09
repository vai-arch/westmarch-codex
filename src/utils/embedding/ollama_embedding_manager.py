# src/embeddings/ollama_manager.py
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Tuple

import requests
from tqdm import tqdm

from src.utils.util_statistics import progress_bar

from .base_embedding_manager import BaseEmbeddingManager


class OllamaEmbeddingManager(BaseEmbeddingManager):
    config = None

    def __init__(self, config: dict):
        self.config = config
        self.ollama_url = config["OLLAMA_BASE_URL"]
        self.method = config["EMBEDDING_METHOD"]
        self.batch_size = config["EMBEDDING_BATCH_SIZE"]
        self.embedding_model_name = config["EMBEDDING_MODEL"]["EMBEDDING_MODEL_NAME"]
        self.embedding_model_dimension = config["EMBEDDING_MODEL"]["EMBEDDING_MODEL_DIMENSION"]
        self.embedding_model_max_tokens = config["EMBEDDING_MODEL"]["EMBEDDING_MODEL_MAX_TOKENS"]
        self.session = requests.Session()
        print(json.dumps(config, indent=2))

    def get_manager_config(self) -> dict:
        return self.config

    def test_connection(self) -> bool:
        """
        Test if Ollama is running and model is available

        Returns:
            bool: True if connection successful
        """
        try:
            response = requests.post(f"{self.ollama_url}/api/embeddings", json={"model": self.embedding_model_name, "prompt": "test"}, timeout=10)

            if response.status_code == 200:
                embedding = response.json().get("embedding", [])
                actual_dim = len(embedding)

                print(f"✅ Ollama connected. Model: {self.embedding_model_name}")
                print(f"   Embedding dim: {actual_dim} (expected: {self.embedding_model_dimension})")

                if actual_dim != self.embedding_model_dimension:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_model_dimension}, got {actual_dim}")

                return True
            else:
                raise ValueError(f"❌ Ollama error: {response.status_code}")

        except Exception as e:
            raise ValueError(f"❌ Connection failed: {e}")

    def embed_chunks(self, texts: List[str], show_progress=True, prefix=None) -> Tuple[List[List[float]], int]:
        """
        Proxy for the embed method. For the moment we are doing everything with embed_batchs but down the road we may want to
        make this configurable, depending enviroment, etc
        """
        dispatch = {
            "ONE_BY_ONE": self.embed_one_by_one,
            "BATCH": self.embed_batch,
            "BATCH_IN_PARALLEL": self.embed_batch_parallel,
        }

        if prefix:
            texts = [f"{prefix}{text}" for text in texts]

        try:
            embed_fn = dispatch[self.method]
        except KeyError:
            raise ValueError(f"Unknown embedding method: {self.method}")

        return embed_fn(texts, show_progress)

    def embed_one_by_one(self, texts: List[str], show_progress: bool = True) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts and return statistics.

        This function processes a list of input texts, generates embeddings
        for each text (using a model such as Ollama), and computes token-usage
        and performance metrics across the full batch.

        Args:
            texts (List[str]):
                List of input text strings to embed.
            batch_size (int):
                Number of items processed per API call.

        Returns:
            tuple:
                embeddings (List[List[float]]):
                    The embedding vector for each input text, in order.

                avg_tokens (float):
                    The average number of tokens consumed per batch request.

                max_tokens (int):
                    The maximum token usage observed across all batch calls.

                total_time (float):
                    Total processing time (in seconds) for generating all embeddings.
        """

        start_time = datetime.now()

        total_tokens = 0
        max_tokens = 0
        count = 0

        use_pbar = len(texts) > 1
        iterator = tqdm(texts, desc="Embedding chunks") if use_pbar else texts

        for text in iterator:
            response = self.session.post(
                f"{self.ollama_url}/api/embed",
                json={
                    "model": self.embedding_model_name,
                    "input": text,  # Send list instead of single string
                },
            )

            response.raise_for_status()
            data = response.json()

            # Returns list of embeddings
            embeddings = data["embeddings"]
            this_chunk_tokens = data.get("prompt_eval_count", 0)

            if this_chunk_tokens == self.embedding_model_max_tokens:
                raise ValueError(f"Maximum number of tokens ({self.embedding_model_max_tokens}) reached!. We need to rechunk everything with safer parameters")

            # update max
            if this_chunk_tokens > max_tokens:
                max_tokens = this_chunk_tokens

            # update totals for average
            total_tokens += this_chunk_tokens
            count += 1

            avg_tokens = total_tokens / count

            if use_pbar:
                iterator.set_postfix_str(f"MAX: {max_tokens} | AVG: {avg_tokens:.1f}")

        total_time = datetime.now() - start_time

        return embeddings, avg_tokens, max_tokens, total_time

    def embed_batch(self, texts: List[str], show_progress: bool = True) -> Tuple[List[List[float]], int]:
        """
        Generate embeddings for a batch of texts and return statistics.

         This function processes a list of input texts, generates embeddings
         for each text (using a model such as Ollama), and computes token-usage
         and performance metrics across the full batch.

         Args:
             texts (List[str]):
                 List of input text strings to embed.

         Returns:
             tuple:
                 embeddings (List[List[float]]):
                     The embedding vector for each input text, in order.

                 avg_tokens (float):
                     The average number of tokens consumed per batch request.

                 max_tokens (int):
                     The maximum token usage observed across all batch calls.

                 total_time (float):
                     Total processing time (in seconds) for generating all embeddings.
        """
        start_time = datetime.now()

        all_embeddings = []

        max_tokens = -1

        total_tokens = 0  # running sum of tokens across all calls
        total_items = 0  # running count of all embedded texts

        # pbar = tqdm(range(0, len(texts), batch_size), desc="Embedding batches")

        pbar = progress_bar(range(0, len(texts), self.batch_size), enable=show_progress, desc="Embedding batches")

        for i in pbar:
            batch_texts = texts[i : i + self.batch_size]

            response = self.session.post(f"{self.ollama_url}/api/embed", json={"model": self.embedding_model_name, "input": batch_texts})
            response.raise_for_status()
            data = response.json()

            batch_embeddings = data["embeddings"]

            # tokens for this API call
            tokens_this_call = data.get("prompt_eval_count", 0)

            # running totals
            total_tokens += tokens_this_call
            total_items += len(batch_embeddings)

            # running average
            avg_tokens = total_tokens / total_items if total_items else 0

            all_embeddings.extend(batch_embeddings)

            if hasattr(pbar, "set_postfix_str"):
                pbar.set_postfix_str(f"Procc Items: {total_items} | AVG Tokens: {avg_tokens:.1f}")

        total_time = datetime.now() - start_time

        return all_embeddings, avg_tokens, max_tokens, total_time

    def embed_batch_parallel(self, texts: List[str], show_progress: bool = True, max_workers: int = 4) -> Tuple[List[List[float]], int]:
        """
        Generate embeddings for a batch of texts and return statistics.

        This function processes a list of input texts, generates embeddings
        for each text (using a model such as Ollama), and computes token-usage
        and performance metrics across the full batch.

        Args:
            texts (List[str]):
                List of input text strings to embed.

        Returns:
            tuple:
                embeddings (List[List[float]]):
                    The embedding vector for each input text, in order.

                avg_tokens (float):
                    The average number of tokens consumed per batch request.

                max_tokens (int):
                    The maximum token usage observed across all batch calls.

                total_time (float):
                    Total processing time (in seconds) for generating all embeddings.
        """

        start_time = datetime.now()

        # Split into batches
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        all_embeddings = []
        avg_tokens = 0
        max_tokens = -1

        # Process batches in parallel
        pbar = tqdm(total=len(batches), desc="Embedding batches (parallel)")

        def process_batch(batch_texts):
            """Process single batch - runs in parallel"""
            response = self.session.post(f"{self.ollama_url}/api/embed", json={"model": self.embedding_model_name, "input": batch_texts}, timeout=300)
            response.raise_for_status()
            data = response.json()

            return data["embeddings"], data.get("prompt_eval_count", 0)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches
            futures = [executor.submit(process_batch, batch) for batch in batches]

            # Collect results as they complete
            for future in futures:
                batch_embeddings, tokens = future.result()
                all_embeddings.extend(batch_embeddings)

                if tokens > avg_tokens:
                    avg_tokens = tokens

                pbar.update(1)
                pbar.set_postfix_str(f"MAX TOKENS: {avg_tokens}")

        pbar.close()

        total_time = datetime.now() - start_time

        return all_embeddings, avg_tokens, max_tokens, total_time
