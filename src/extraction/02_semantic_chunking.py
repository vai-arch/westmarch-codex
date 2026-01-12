"""
Entity Extraction using LLM (Ollama)
=====================================
Extracts all entity types from The Hobbit chapters using local LLM.

Input: data/processed/book_00_structured.json
Output: data/processed/entities_raw.json

Entity Types Extracted:
- Characters (name, aliases, race, gender, role, affiliations, relationships, traits)
- Locations (name, aliases, type, parent_location, inhabitants, significance)
- Objects (name, aliases, type, owner, properties, significance)
- Events (name, type, participants, location, chapter_refs, outcome)
- Groups (name, type, members, purpose, allegiances)
- Concepts (name, description, related_entities)
"""

# Import configuration
import time
from typing import Any, Dict, List

from src.config import get_config
from src.paths import get_paths
from src.utils.chunking.chunking_strategies import semantic_chunker_function
from src.utils.embedding.embedding_factory import create_embedding_manager
from src.utils.util_files_functions import load_json_from_file, save_jsonl_to_file
from src.utils.util_statistics import total_statistics_logging

FILE_BOOKS_CHUNKED = None
CHUNKING_STRATEGY = None
embedding_manager = None


def chunk_book_chapter(chapter: Dict, book_number: int, book_title: str, config: Dict) -> List[Dict]:
    """
    Chunk a single chapter into smaller pieces.

    Args:
        chapter: Chapter dict with content, title, etc.
        book_number: Book number
        book_title: Book title

    Returns:
        List of chunk dicts
    """
    chapter_text = "\n\n".join(chapter["paragraphs"])
    chapter_num = chapter.get("chapter_num", 0)
    chapter_title = chapter.get("chapter_title", "")
    chapter_type = chapter.get("chapter_type", "chapter")

    print(f"Chapter: {chapter_num}, Chapter length: {len(chapter_text)} characters")

    # Semantic chunking (current strategy)
    raw_chunks = semantic_chunker_function(text=chapter_text, chunking_strategy_config=CHUNKING_STRATEGY)

    # === PASS 1: Merge tiny chunks based on tokens only ===
    min_tokens = CHUNKING_STRATEGY["MIN_BOOKS_CHUNKS_SIZE_TOKENS"]

    merged_chunks: list[str] = []

    for chunk_text in raw_chunks:
        tokens = embedding_manager.token_count(chunk_text)

        # Merge tiny chunk into previous one
        if tokens < min_tokens and merged_chunks:
            merged_chunks[-1] = merged_chunks[-1].rstrip() + " " + chunk_text.lstrip()
        else:
            merged_chunks.append(chunk_text)

    # === PASS 2: Add full metadata ===
    filtered_chunks = []
    total_chunks = len(merged_chunks)

    for idx, chunk_text in enumerate(merged_chunks):
        token_count = embedding_manager.token_count(chunk_text)
        chunk = {
            "source": "book",
            "book_number": book_number,
            "book_title": book_title,
            "chapter_number": chapter_num,
            "chapter_title": chapter_title,
            "chapter_type": chapter_type,
            "text": chunk_text,
            "token_count": token_count,
            "temporal_order": book_number,
            "chunk_index": idx + 1,
            "total_chunks_in_chapter": total_chunks,
            "chunk_id": f"book_{book_number:02d}_ch_{chapter_num:02d}_chunk_{idx + 1:03d}",
        }

        filtered_chunks.append(chunk)

        # Debug
        print(f"{chunk['chunk_id']} → {token_count} tokens, {len(chunk_text)} chars")

    return filtered_chunks


def generate_statistics(chapter_chunks: List[Dict]) -> Dict[str, Any]:
    """
    Generate chunk statistics per chapter and total.
    """
    per_chapter_stats = []

    # Group chunks by chapter
    chapters_dict = {}
    for chunk in chapter_chunks:
        ch_num = chunk["chapter_number"]
        if ch_num not in chapters_dict:
            chapters_dict[ch_num] = []
        chapters_dict[ch_num].append(chunk)

    # Calculate per-chapter statistics
    for ch_num in sorted(chapters_dict.keys()):
        chunks = chapters_dict[ch_num]
        chunk_sizes = [len(chunk["text"]) for chunk in chunks]
        token_count = [chunk["token_count"] for chunk in chunks]

        per_chapter_stats.append(
            {
                "name": f"Chapter {ch_num}",
                "metrics": {
                    "num_chunks": len(chunks),
                    "avg_chunk_size": int(sum(chunk_sizes) / len(chunk_sizes)) if chunk_sizes else 0,
                    "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                    "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                    "total_characters": sum(chunk_sizes),
                    "avg_chunk_size_tokens": int(sum(token_count) / len(chunk_sizes)) if chunk_sizes else 0,
                    "min_chunk_size_tokens": min(token_count) if token_count else 0,
                    "max_chunk_size_tokens": max(token_count) if token_count else 0,
                    "total_tokens": sum(token_count),
                },
            }
        )

    # Calculate total statistics
    all_chunk_sizes = [len(chunk["text"]) for chunk in chapter_chunks]
    all_token_count = [chunk["token_count"] for chunk in chapter_chunks]
    per_chapter_stats.append(
        {
            "name": "TOTAL",
            "metrics": {
                "num_chunks": len(chapter_chunks),
                "avg_chunk_size": int(sum(all_chunk_sizes) / len(all_chunk_sizes)) if all_chunk_sizes else 0,
                "min_chunk_size": min(all_chunk_sizes) if all_chunk_sizes else 0,
                "max_chunk_size": max(all_chunk_sizes) if all_chunk_sizes else 0,
                "total_characters": sum(all_chunk_sizes),
                "avg_chunk_size_tokens": int(sum(all_token_count) / len(all_chunk_sizes)) if all_chunk_sizes else 0,
                "min_chunk_size_tokens": min(all_token_count) if all_token_count else 0,
                "max_chunk_size_tokens": max(all_token_count) if all_token_count else 0,
                "total_tokens": sum(all_token_count),
            },
        }
    )

    return per_chapter_stats


def main():
    start_time = time.time()

    config = get_config()
    paths = get_paths()
    global FILE_BOOKS_CHUNKED, CHUNKING_STRATEGY, embedding_manager

    FILE_BOOKS_CHUNKED = paths.FILE_BOOKS_CHUNKED
    CHUNKING_STRATEGY = config.CHUNKING_STRATEGY

    embedding_manager = create_embedding_manager(CHUNKING_STRATEGY["EMBEDDING_MANAGER_CONFIG"])

    chapters = load_json_from_file(paths.FILE_BOOK_00_PROCESSED)

    chapter_chunks = []

    # Chunk chapters
    for chapter in chapters:
        chapter_chunks.extend(chunk_book_chapter(chapter, chapter["book_num"], "The Hobbit", config.CHUNKING_STRATEGY))

    # Print statistics
    statistics = generate_statistics(chapter_chunks)

    # Save results
    save_jsonl_to_file(
        data=chapter_chunks,
        output_file=FILE_BOOKS_CHUNKED,
    )

    total_time = time.time() - start_time
    total_statistics_logging(
        statistics,
        total_time,
        "ENTITIES EXTRACTION",
        "04_draft_chunking",
        tables=True,
        configuration_section=config.CHUNKING_STRATEGY,
    )


if __name__ == "__main__":
    main()
