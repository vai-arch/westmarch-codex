"""
Entity Extraction using LLM (Ollama)
=====================================
Extracts all entity types from The Hobbit chapters using local LLM.

Input: data/processed/book_00_structured.json
Output: data/processed/entities_raw.json

Entity Types Extracted:
- Characters (name, aliases, race, gender, role, affiliations, relationships, traits)
- Locations (name, aliases, type, parent_location, inhabitants, significance)
- Races (name, characteristics, notable_members)
- Objects (name, aliases, type, owner, properties, significance)
- Events (name, type, participants, location, chapter_refs, outcome)
- Groups (name, type, members, purpose, allegiances)
- Concepts (name, description, related_entities)
- Songs (title, performer, context, chapter_reference)
"""

import json

# Import configuration
import time
import traceback
from typing import Any, Dict, List

import requests
from tqdm import tqdm

from src.config import get_config
from src.paths import get_paths
from src.utils.llm_access.ollama_llm import testing_connection
from src.utils.util_files_functions import load_json_from_file, save_json_to_file
from src.utils.util_statistics import total_statistics_logging

config = None
LLM_CONFIG = None
ENTITY_TYPES = None

# =============================================================================
# LLM INTERACTION
# =============================================================================


def call_ollama(prompt: str) -> Dict[str, Any]:
    """
    Call Ollama API with the extraction prompt.

    Args:
        prompt: The extraction prompt with chapter text
        model: Model name to use

    Returns:
        Parsed JSON response from LLM
    """
    url = f"{LLM_CONFIG['base_url']}/api/generate"

    payload = {
        "model": LLM_CONFIG["model"],
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": LLM_CONFIG["temperature"],
            "num_predict": LLM_CONFIG["max_tokens"],
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=LLM_CONFIG["timeout"])
        response.raise_for_status()

        result = response.json()
        response_text = result.get("response", "")

        # Clean response - extract JSON only
        response_text = response_text.strip()

        # Remove everything before first {
        if "{" in response_text:
            response_text = response_text[response_text.find("{") :]
        else:
            raise ValueError("No JSON object found in response")

        # Remove everything after last }
        if "}" in response_text:
            response_text = response_text[: response_text.rfind("}") + 1]
        else:
            raise ValueError("No complete JSON object found in response")

        # Parse JSON
        entities = json.loads(response_text)
        return entities

    except requests.exceptions.RequestException as e:
        print(f"❌ API Error: {e}")
        raise
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parse Error: {e}")
        print(f"Response text: {response_text[:500]}")
        raise


# =============================================================================
# EXTRACTION LOGIC
# =============================================================================


def extract_entities_from_chapter(chapter: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract entities from a single chapter.

    Args:
        chapter: Chapter data with paragraphs

    Returns:
        Extracted entities for this chapter
    """
    # Combine paragraphs into chapter text
    chapter_text = "\n\n".join(chapter["paragraphs"])

    # Build prompt
    chapter_num = chapter["chapter_num"]
    chapter_title = chapter["chapter_title"]

    prompt = config.ENTITIES_EXTRACTION["EXTRACTION_PROMPT"].format(chapter_text=chapter_text, chapter_num=chapter_num, chapter_title=chapter_title)

    # Call LLM
    entities = call_ollama(prompt)

    # Add metadata
    entities["book_num"] = chapter["book_num"]
    entities["chapter_num"] = chapter["chapter_num"]
    entities["chapter_title"] = chapter["chapter_title"]

    return entities


def extract_all_entities(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract entities from all chapters with progress tracking.

    Args:
        chapters: List of chapter data

    Returns:
        List of entity extractions per chapter
    """
    all_entities = []

    for chapter in tqdm(chapters, desc="Processing chapters"):
        try:
            entities = extract_entities_from_chapter(chapter)
            all_entities.append(entities)
        except Exception as e:
            print(f"\n❌ Error processing chapter {chapter['chapter_num']}: {e}")
            traceback.print_exc()
            raise e

    return all_entities


# =============================================================================
# STATISTICS
# =============================================================================


def generate_statistics(all_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate entity statistics per chapter and total.
    """

    total_counts = {entity_type: 0 for entity_type in ENTITY_TYPES}
    per_chapter_stats = []

    for chapter_entities in all_entities:
        chapter_num = chapter_entities.get("chapter_num")

        chapter_counts = {entity_type: 0 for entity_type in ENTITY_TYPES}

        for entity_type in ENTITY_TYPES:
            count = len(chapter_entities.get(entity_type, []))
            chapter_counts[entity_type] = count
            total_counts[entity_type] += count

        per_chapter_stats.append(
            {
                "name": f"book_num: 0 - chapter_num: {chapter_num}",
                "metrics": {
                    **chapter_counts,
                },
            }
        )

    per_chapter_stats.append(
        {
            "name": "TOTAL",
            "metrics": {
                **total_counts,
            },
        }
    )

    return per_chapter_stats


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function."""

    start_time = time.time()

    global config, LLM_CONFIG, paths, ENTITY_TYPES
    config = get_config()
    paths = get_paths()
    LLM_CONFIG = config.LLM_CONFIG
    ENTITY_TYPES = config.ENTITY_TYPES

    chapters = load_json_from_file(paths.FILE_BOOK_00_PROCESSED)

    testing_connection(config.LLM_CONFIG)

    # Extract entities
    all_entities = extract_all_entities(chapters)

    # Print statistics
    statistics = generate_statistics(all_entities)

    # Save results
    save_json_to_file(all_entities, paths.FILE_ENTITIES_RAW, indent=2, log=False)

    total_time = time.time() - start_time
    total_statistics_logging(
        statistics,
        total_time,
        "ENTITIES EXTRACTION",
        "02_extract_entities",
        tables=True,
        configuration_section=config.LLM_CONFIG,
    )


if __name__ == "__main__":
    main()
