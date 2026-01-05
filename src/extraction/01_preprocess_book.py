"""
01_preprocess_book.py

Extracts chapters from The Hobbit epub and saves structured JSON.

Input: raw/the_hobbit.epub
Output: data/processed/book_00_structured.json

Each chapter contains:
- book_num: 0 (The Hobbit)
- chapter_num: sequential number
- chapter_title: extracted title
- text: cleaned chapter text
- paragraph_count: number of paragraphs
- char_count: character count
"""

import re
import time
from pathlib import Path
from typing import Dict, List

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

from src.config import get_config
from src.paths import get_paths
from src.utils.util_files_functions import save_json_to_file
from src.utils.util_logging import get_logger
from src.utils.util_statistics import total_statistics_logging

log = get_logger(__name__)


def extract_text_from_html(html_content: str, skip_first_heading: bool = True) -> str:
    """
    Clean HTML content and extract plain text.
    Preserves paragraph breaks.

    Args:
        html_content: Raw HTML content
        skip_first_heading: If True, removes first h1/h2/h3 tag (chapter title)
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Remove first heading if it's the chapter title
    if skip_first_heading:
        for tag in ["h1", "h2", "h3"]:
            heading = soup.find(tag)
            if heading:
                heading.decompose()
                break

    # Get text and preserve paragraph structure
    paragraphs = []
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            paragraphs.append(text)

    # Join paragraphs with double newline
    full_text = "\n\n".join(paragraphs)

    # Clean up whitespace
    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    full_text = re.sub(r" +", " ", full_text)

    return full_text.strip()


def extract_chapter_title(html_content: str) -> str:
    """
    Extract chapter title from HTML content.
    Looks for h1, h2, or first significant text.
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Try heading tags first
    for tag in ["h1", "h2", "h3"]:
        heading = soup.find(tag)
        if heading:
            return heading.get_text(strip=True)

    fallback_p_index = config.EXTRACTION.get("CHAPTER_TITLE_FALLBACK_P_INDEX", 1)
    # Fallback: use first paragraph if it's short (likely a title)
    paragraphs = soup.find_all("p")
    if len(paragraphs) > fallback_p_index:
        text = paragraphs[fallback_p_index].get_text(strip=True)
        return text
    return "Untitled Chapter"


def extract_paragraphs(text: str) -> List[str]:
    """
    Compute paragraph statistics for text separated by double newlines.
    """

    paragraphs: List[str] = [p.strip() for p in text.split("\n\n") if p.strip()]

    return paragraphs


def paragraph_statistics(paragraphs: List[str]) -> Dict[str, float]:
    if not paragraphs:
        return {
            "paragraphs": [],
            "count": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_chars": 0.0,
        }

    lengths = [len(p) for p in paragraphs[2:]]

    avg_chars = int(sum(lengths) / len(lengths))

    threshold = avg_chars * 0.1

    if True:
        short_paragraphs = [(i, p, len(p)) for i, p in enumerate(paragraphs) if len(p) < threshold]

        if short_paragraphs:
            for idx, p, length in short_paragraphs:
                print(f"  [{idx}] {length} chars → {p[:120]!r}")

    return {
        "count": len(paragraphs),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "avg_chars": avg_chars,
    }


def process_epub(epub_path: Path) -> List[Dict]:
    """
    Process epub file and extract structured chapter data.

    Returns list of chapter dictionaries.
    """
    print(f"\n📖 Loading epub from: {epub_path}")

    book = epub.read_epub(str(epub_path))
    chapters = []
    statistics = []

    # Get all document items (chapters)
    items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))

    print(f"Found {len(items)} document items in epub")

    chapter_num = 0
    paragraph_count = 0
    char_count = 0
    for item in items:
        content = item.get_content().decode("utf-8", errors="ignore")

        # Extract text
        chapter = extract_text_from_html(content)

        paragraphs = extract_paragraphs(chapter)
        paragraph_count = len(paragraphs)
        char_count = len(chapter)

        if char_count < config.EXTRACTION["CHAPTER_MIN_CHAR_COUNT"]:
            log.info(f"Chapter with only {char_count} chars")
            continue
        if paragraph_count < config.EXTRACTION["CHAPTER_MIN_PARAGRAPH_COUNT"]:
            log.info(f"Chapter with only {paragraph_count} paragraphs")
            continue

        title = paragraphs[1]
        paragraphs = paragraphs[2:]  # Skip chapter number & title

        paragraphs_stats = paragraph_statistics(paragraphs)

        chapter_num += 1

        chapter_data = {
            "book_num": 0,
            "chapter_num": chapter_num,
            "chapter_title": title,
            "paragraphs": paragraphs,
            "paragraph_count": len(paragraphs),
            "char_count": char_count,
            "paragraph_min_chars": paragraphs_stats["min_chars"],
            "paragraph_max_chars": paragraphs_stats["max_chars"],
            "paragraph_avg_chars": paragraphs_stats["avg_chars"],
        }

        chapters.append(chapter_data)

        statistics.append(
            {
                "name": f"book_num: 0 - chapter_num: {chapter_num}",
                "metrics": {
                    "title": title,
                    "paragraph_count": len(paragraphs),
                    "paragraph_min_chars": paragraphs_stats["min_chars"],
                    "paragraph_max_chars": paragraphs_stats["max_chars"],
                    "paragraph_avg_chars": paragraphs_stats["avg_chars"],
                    "char_count": char_count,
                },
            }
        )

        print(f"  ✓ Chapter {chapter_num}: {title} ({len(chapter):,} chars, {chapter_data['paragraph_count']} paragraphs)")

    return chapters, statistics


def main():
    """Main preprocessing pipeline."""
    start_time = time.time()

    print("=" * 70)
    print("BOOK PREPROCESSING - The Hobbit")
    print("=" * 70)

    paths = get_paths()

    # Validate input file exists
    if not paths.FILE_BOOK_00_RAW.exists():
        raise FileNotFoundError(f"Input file not found: {paths.FILE_BOOK_00_RAW}")

    # Extract chapters
    chapters, statistics = process_epub(paths.FILE_BOOK_00_RAW)

    # Save output
    save_json_to_file(chapters, paths.FILE_BOOK_00_PROCESSED, indent=2, log=True)

    total_time = time.time() - start_time
    total_statistics_logging(
        statistics,
        total_time,
        "BOOK PREPROCESSING",
        "01_preprocess_book",
        tables=True,
        configuration_section=config.EXTRACTION,
    )

    print("\n✅ Preprocessing complete!")


if __name__ == "__main__":
    config = get_config()
    main()
