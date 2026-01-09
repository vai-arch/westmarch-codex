import re
from typing import List

from src.utils.chunking.chunking_strategies import get_chunker
from src.utils.config import get_config

config = get_config()

_books_chunker = get_chunker(config.CHUNKING_STRATEGY["BOOKS_CHUNKING_STRATEGY_NAME"])


TARGET_CHARS = config.TARGET_TOKENS * config.CHARS_PER_TOKEN
MAX_CHARS = config.MAX_CHUNK_SIZE
OVERLAP_CHARS = config.OVERLAP_TOKENS * config.CHARS_PER_TOKEN


def chunk_statistics(chunks: List[dict], name):
    # Chunk sizes
    chunk_sizes = [len(c["text"]) for c in chunks]
    avg_size = sum(chunk_sizes) / len(chunk_sizes)
    min_size = min(chunk_sizes)
    max_size = max(chunk_sizes)

    # # # Raise exception if a chunk exceeds max allowed size
    # if max_size > MAX_CHARS:
    #     raise ValueError(f"❌ Chunk exceeds max allowed size: {max_size:,} > {MAX_CHARS:,}")

    results = {
        "name": name,
        "metrics": {
            "total_chunks": len(chunks),
            "avg_chunks_size": avg_size,
            "min_chunk_size": min_size,
            "max_chunk_size": max_size,
            "target_size": TARGET_CHARS,
            "max_allowed_size": MAX_CHARS,
        },
    }

    return results


def split_oversized_paragraph(paragraph: str, max_size: int) -> List[str]:
    """
    Split a paragraph that exceeds max_size into smaller chunks.
    Tries to split at sentence boundaries first, then hard splits if needed.

    Args:
        paragraph: Oversized paragraph text
        max_size: Maximum chunk size in characters

    Returns:
        List of chunk strings
    """
    if len(paragraph) <= max_size:
        return [paragraph]

    chunks = []

    # Try splitting at sentence boundaries first
    # Match sentence endings: . ! ? followed by space or end of string
    sentences = re.split(r"([.!?]\s+)", paragraph)

    # Rejoin sentences with their punctuation
    sentence_list = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_list.append(sentences[i] + sentences[i + 1])
        else:
            sentence_list.append(sentences[i])

    # If there's a remaining part without punctuation, add it
    if len(sentences) % 2 == 1:
        sentence_list.append(sentences[-1])

    # Group sentences into chunks
    current_chunk = []
    current_size = 0

    for sentence in sentence_list:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_size = len(sentence)

        # If single sentence exceeds max, hard split it
        if sentence_size > max_size:
            # Save current chunk if exists
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Hard split the sentence
            for i in range(0, len(sentence), max_size):
                chunk_part = sentence[i : i + max_size]
                chunks.append(chunk_part)
            continue

        # Check if adding sentence would exceed max
        separator_size = 1 if current_chunk else 0
        new_size = current_size + separator_size + sentence_size

        if new_size <= max_size:
            # Fits - add to current chunk
            current_chunk.append(sentence)
            current_size = new_size
        else:
            # Doesn't fit - start new chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = [sentence]
            current_size = sentence_size

    # Add final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def group_paragraphs_into_chunks(paragraphs: List[str]) -> List[str]:
    """
    Group paragraphs into chunks that approach target size.

    Strategy:
    - Combine paragraphs sequentially until reaching target size
    - Start new chunk if adding next paragraph would exceed max size
    - Split oversized paragraphs at sentence boundaries

    Args:
        paragraphs: List of paragraph strings
        config: Config object with CHUNK_SIZE and MAX_CHUNK_SIZE

    Returns:
        List of chunk strings (grouped paragraphs)
    """
    if not paragraphs:
        return []

    chunks = []
    current_chunk = []
    current_size = 0

    # target_size = config.CHUNK_SIZE  # 5.980 chars (1,328 tokens)
    max_size = config.MAX_CHUNK_SIZE  # 6.956 chars (1,545 tokens)

    for paragraph in paragraphs:
        para_size = len(paragraph)

        # If single paragraph exceeds max, split it further
        if para_size > max_size:
            # Save current chunk if exists
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0

            # Split oversized paragraph at sentence boundaries
            para_chunks = split_oversized_paragraph(paragraph, max_size)
            chunks.extend(para_chunks)
            continue

        # Check if adding this paragraph would exceed max
        # (include \n\n separator in size calculation)
        separator_size = 2 if current_chunk else 0
        new_size = current_size + separator_size + para_size

        if new_size <= max_size:
            # Fits - add to current chunk
            current_chunk.append(paragraph)
            current_size = new_size
        else:
            # Doesn't fit - start new chunk
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

            current_chunk = [paragraph]
            current_size = para_size

    # Add final chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def split_into_paragraphs(text: str) -> List[str]:
    """
    Split text into properly-sized chunks, grouping paragraphs as needed.

    Args:
        text: Text to split
        config: Config object

    Returns:
        List of chunk strings
    """
    if not text:
        return []

    content_length = len(text)

    # If content fits in one chunk, return as-is
    if content_length <= config.MAX_CHUNK_SIZE:
        return [text.strip()]

    # Content too large - split into paragraphs
    paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Group paragraphs into target-sized chunks
    chunks = group_paragraphs_into_chunks(paragraphs)

    return chunks


def split_paragraph_into_chunks(paragraph: str) -> List[str]:
    """
    Split a paragraph into overlapping chunks.
    Handles both small paragraphs (returns as-is) and large ones (splits with overlap).

    Args:
        paragraph: The text to split
        target_chars: Target size for each chunk
        max_chars: Maximum size before forcing a split
        overlap_chars: Number of characters to overlap between chunks

    Returns:
        List of overlapping text chunks (single item if paragraph fits in one chunk)
    """
    # If paragraph fits in target, return as-is
    if len(paragraph) <= TARGET_CHARS:
        return [paragraph]

    chunks = []
    start = 0

    while start < len(paragraph):
        # Determine end position
        end = start + TARGET_CHARS

        # If this is the last chunk, take everything remaining
        if end >= len(paragraph):
            chunks.append(paragraph[start:].strip())
            break

        # Try to break at a sentence boundary (. ! ?)
        # Look a bit beyond target to find a good break point
        search_end = min(end + 200, len(paragraph))
        sentence_end = max(
            paragraph.rfind(". ", start, search_end),
            paragraph.rfind("! ", start, search_end),
            paragraph.rfind("? ", start, search_end),
        )

        if sentence_end > start:
            # Found a sentence boundary
            end = sentence_end + 1  # Include the period/punctuation
        else:
            # No sentence boundary, try word boundary
            space_pos = paragraph.rfind(" ", start, min(start + MAX_CHARS, len(paragraph)))
            if space_pos > start:
                end = space_pos
            else:
                # Worst case: hard cut at max_chars
                end = min(start + MAX_CHARS, len(paragraph))

        chunks.append(paragraph[start:end].strip())

        # Move start position with overlap
        start = end - OVERLAP_CHARS

        # Ensure we're making progress (avoid infinite loop)
        if start <= len(chunks[-1]) - OVERLAP_CHARS:
            start = end

    return chunks
