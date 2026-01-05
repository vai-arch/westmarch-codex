# The Westmarch Codex

A production-quality RAG system built from a single book source (the Hobbit), demonstrating proper architecture and entity extraction.

## Goal

Build the best possible RAG system for The Hobbit with zero manual data curation - everything extracted and processed automatically from the book itself.

## Architecture Principles

- **Hybrid retrieval**: Dense (semantic) + Sparse (keyword/BM25)
- **Entity-first approach**: Bootstrap entity extraction → enhance chunking → improve retrieval
- **Two-stage retrieval**: Broad recall → precise reranking
- **Hierarchical metadata**: Chapter → Scene → Chunk linking
- **Knowledge graph**: Characters, locations, events, relationships extracted automatically
- **Reusability and Modularity** We need to reuse code from previous projects and make efverything reusable and modular

## Technical Stack

- **Environment**: Windows, Python 3.10+, Conda
- **Vector DB**: ChromaDB (local, simple, effective)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2 for speed, can upgrade later)
- **Sparse retrieval**: BM25 via rank-bm25
- **Entity extraction**: spaCy + LLM-assisted refinement
- **LLM**: Local via Ollama
- **Everything Local**

## Project Structure

```
westmarch-codex/
├── raw/                 # The Hobbit .epub. Only true source for the rest of content
├── data/
│   └── chunks/           # Generated chunks
│   └── embeddings/       # Generated embeddings
│   └── processed/        # Generated entities, aliases, indexes, etc
│   └── vector_store/     # Chrome db collections
│   └── bm25/             # Bm25 indexes
├── src/
│   ├── extraction/       # Entity & structure extraction
│   ├── indexing/         # Chunking, embedding, vector store
│   ├── retrieval/        # Query processing, hybrid search
│   └── evaluation/       # Test queries, metrics
├── notebooks/            # Exploration & analysis
├── config.py             # All parameters in one place
├── paths.py              # All files and folder paths in one place
└── runScripts.ps1        # Full pipeline orchestration (powershell)

we also have some util functions from another projects. RESUSE ALWAYS
from utils import statistics
  statistics.total_statistics_logging(stats, total_time, "TRAINING PAIR GENERATOR", "01_generate_training_pairs", tables=False, configuration_section=None)
from utils import files
 def deserialize_object(input_file, log=False):
 def serialize_object(data, output_file, log=False):
 def remove_file(filepath: str):
 def find_files_in_folder(path_folder, extension=".txt", exception=True, recursive=False, sort=True):
 def copy_files(
 def load_json_line_by_line(file, log=True):
 def load_txt_line_by_line(file, log=True):
 def load_text_from_file(file, log=True):
 def load_json_from_file(file, log=True):
 def save_jsonl_to_file(data: List[Dict], output_file, indent: int = None):
 def save_json_to_file(data: List[Dict], output_file, indent: int = None, log=True):
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
  
class BaseVectorStoreManager(ABC):
    @abstractmethod
    def get_collection(self, name: str):
        pass

    @abstractmethod
    def get_or_create_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

```

## Setup

```bash
conda create -n westmarch python=3.11.14
conda activate westmarch
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run Full Pipeline

```
runScript.ps1
```

## Build Phases

1. **Extract entities** (characters, places, timeline markers)
2. **Semantic chunking** with entity metadata
3. **Build indexes** (vector + sparse + KG)
4. **Query classification** using extracted entities
5. **Retrieval pipeline** with reranking

## Learning Goals

- Entity bootstrapping without external data
- Chunking strategies impact on retrieval
- Query understanding and routing
- Evaluation without labeled data
- Iterative refinement patterns

---

*No shortcuts. Best practices only.*
