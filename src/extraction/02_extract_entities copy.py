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

# Import configuration
import json
import time
from typing import Any, Dict, List

from src.config import get_config
from src.paths import get_paths
from src.utils.llm_access.ollama_llm import prompt_builder, testing_connection
from src.utils.util_files_functions import load_json_from_file, save_json_to_file
from src.utils.util_statistics import total_statistics_logging

LLM_CONFIG = None
ENTITY_TYPES = None
ENTITY_SCHEMAS = None
SINGLE_ENTITY_EXTRACTION_PROMPT = None
FILE_ENTITIES_RAW = None


class _CheckpointManager:
    """Internal class to manage entity extraction checkpoints."""

    def __init__(self):
        self.checkpoint_file = FILE_ENTITIES_RAW.parent / str(FILE_ENTITIES_RAW.stem + ".checkpoint.json")

    def load_checkpoint(self) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, str]]], int]:
        """
        Load checkpoint if it exists.

        Returns:
            Tuple of (all_entities, accumulated_entities, start_chapter)
        """
        checkpoint_file = self.checkpoint_file

        if checkpoint_file.exists():
            checkpoint = load_json_from_file(checkpoint_file)

            all_entities = checkpoint.get("all_entities", [])
            accumulated_entities = checkpoint.get("accumulated_entities", {})
            start_chapter = checkpoint.get("last_chapter", 0) + 1

            print(f"✅ Loaded checkpoint: {len(all_entities)} chapters processed")
            print(f"📍 Resuming from chapter {start_chapter}")

            return all_entities, accumulated_entities, start_chapter

        # No checkpoint - start fresh
        accumulated_entities = {entity_type: [] for entity_type in ENTITY_TYPES}
        return [], accumulated_entities, 1

    def save_checkpoint(self, all_entities: List[Dict[str, Any]], accumulated_entities: Dict[str, List[Dict[str, str]]]):
        """
        Save current progress to checkpoint.

        Args:
            all_entities: List of all chapter extractions so far
            accumulated_entities: Accumulated lightweight entity contexts
        """

        checkpoint = {"all_entities": all_entities, "accumulated_entities": accumulated_entities, "last_chapter": all_entities[-1]["chapter_num"] if all_entities else 0, "timestamp": time.time()}

        save_json_to_file(data=checkpoint, output_file=self.checkpoint_file, indent=2, log=False)

    def finalize_checkpoint(self):
        """Rename checkpoint to final output file."""

        if self.checkpoint_file.exists():
            checkpoint = load_json_from_file(self.checkpoint_file)

            # Save final results
            all_entities = checkpoint.get("all_entities", [])
            save_json_to_file(data=all_entities, output_file=FILE_ENTITIES_RAW, indent=2, log=True)

            # Remove checkpoint
            self.checkpoint_file.unlink()
            print(f"\n✅ Checkpoint finalized to: {FILE_ENTITIES_RAW}")


def get_lightweight_context(entities: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert full entity list to lightweight context (names + aliases only).

    Args:
        entities: Full entity data

    Returns:
        Lightweight context with just names and aliases
    """
    lightweight = []

    for entity in entities:
        context_item = {"name": entity.get("name", "")}

        # Add aliases if present
        aliases = entity.get("aliases", [])
        if aliases:
            context_item["aliases"] = aliases

        lightweight.append(context_item)

    return lightweight


def merge_into_accumulated(accumulated: List[Dict[str, str]], new_entities: List[Dict[str, Any]]):
    """
    Merge new entities into accumulated lightweight context.
    Updates existing entities or adds new ones.

    Args:
        accumulated: Existing accumulated entities (modified in place)
        new_entities: Newly extracted entities to merge
    """
    for new_entity in new_entities:
        name = new_entity.get("name", "")
        aliases = new_entity.get("aliases", [])

        # Check if entity already exists (by name)
        existing = None
        for acc_entity in accumulated:
            if acc_entity["name"] == name:
                existing = acc_entity
                break

        if existing:
            # Merge aliases (avoid duplicates)
            existing_aliases = set(existing.get("aliases", []))
            new_aliases = set(aliases)
            existing["aliases"] = list(existing_aliases | new_aliases)
        else:
            # Add new entity
            accumulated.append({"name": name, "aliases": aliases})


DESCRIPTIVE_KEYS = {
    "description",
    "type",
    "role",
    "location_type",
    "object_type",
    "event_type",
    "group_type",
    "race",
    "lyrics",
    "significance",
}


def validate_extraction_structure(data: Dict[str, Any]) -> List[str]:
    """
    Validates entity extraction structure.
    Returns a list of validation error messages.
    Empty list == valid.
    """

    errors = []

    # ---- top-level checks ----
    for key in ("chapter_num", "chapter_title", "book_num"):
        if key not in data:
            errors.append(f"Missing top-level field: '{key}'")

    # ---- entity checks ----
    for entity_type in ENTITY_TYPES:
        if entity_type not in data:
            errors.append(f"Missing entity type: '{entity_type}'")
            continue

        if not isinstance(data[entity_type], list):
            errors.append(f"'{entity_type}' must be a list")
            continue

        for i, entity in enumerate(data[entity_type]):
            if not isinstance(entity, dict):
                errors.append(f"{entity_type}[{i}] is not an object")
                continue

            if "name" not in entity or not entity["name"]:
                errors.append(f"{entity_type}[{i}] missing 'name'")
                continue

            # Must have at least one descriptive field
            if not any(key in entity and entity[key] for key in DESCRIPTIVE_KEYS):
                errors.append(f"{entity_type}[{i}] ('{entity.get('name')}') has no descriptive fields")

    return errors


# =============================================================================
# ENTITY EXTRACTION WITH CONTEXT
# =============================================================================


def normalize_entity_fields(entities: List[Dict[str, Any]], entity_type: str) -> List[Dict[str, Any]]:
    """
    Normalize entity fields to match expected schema.
    Fix common LLM mistakes like location_name -> name.
    """
    # Field name corrections
    name_corrections = {
        "location_name": "name",
        "event_name": "name",
        "character_name": "name",
        "object_name": "name",
        "group_name": "name",
        "concept_id": "name",
        "song_title": "name",
        "title": "name",
    }

    for entity in entities:
        # Fix name fields
        for wrong_name, correct_name in name_corrections.items():
            if wrong_name in entity and correct_name not in entity:
                entity[correct_name] = entity.pop(wrong_name)

        # Remove empty descriptions (cleanup)
        if "description" in entity and entity["description"] == "":
            entity.pop("description")

    return entities


def build_context_section(previous_entities: List[Dict[str, str]], entity_type: str) -> str:
    """
    Build the context section showing previously extracted entities.

    Args:
        previous_entities: Lightweight context from previous chapters
        entity_type: Current entity type being extracted

    Returns:
        Formatted context string for prompt
    """
    if not previous_entities:
        return ""

    context_lines = [f"\nPREVIOUSLY EXTRACTED {entity_type.upper()} FROM EARLIER CHAPTERS:", "Use these to maintain consistent naming and recognize aliases:\n"]

    for entity in previous_entities[:50]:  # Limit to 50 to avoid token bloat
        name = entity.get("name", "")
        aliases = entity.get("aliases", [])
        if aliases:
            context_lines.append(f"- {name} (also called: {', '.join(aliases)})")
        else:
            context_lines.append(f"- {name}")

    if len(previous_entities) > 50:
        context_lines.append(f"... and {len(previous_entities) - 50} more")

    context_lines.append("")
    return "\n".join(context_lines)


def extract_single_entity_type(entity_type: str, chapter_text: str, chapter_num: int, chapter_title: str, previous_entities: List[Dict[str, str]], max_retries: int = 2) -> List[Dict[str, Any]]:
    """
    Extract a single entity type with retry logic and context from previous chapters.

    Args:
        entity_type: Type of entity to extract
        chapter_text: Full chapter text
        chapter_num: Chapter number
        chapter_title: Chapter title
        previous_entities: Lightweight context from previous chapters (same type only)
        max_retries: Maximum retry attempts for full extraction

    Returns:
        List of extracted entities
    """
    schema = ENTITY_SCHEMAS[entity_type]
    entity_singular = entity_type.rstrip("s")

    # Build context section
    if entity_type in ["characters", "locations", "objects", "groups", "races"]:
        context_section = build_context_section(previous_entities, entity_type)
        print(context_section)
    else:
        context_section = ""

        # Build prompt with context
    prompt = SINGLE_ENTITY_EXTRACTION_PROMPT.format(
        entity_type=entity_type,
        entity_singular=entity_singular,
        description=schema["description"],
        format=json.dumps(schema["format"], indent=2),
        example=json.dumps(schema["example"], indent=2),
        context_section=context_section,
        chapter_text=chapter_text,
        chapter_num=chapter_num,
        chapter_title=chapter_title,
    )

    for attempt in range(max_retries + 1):
        try:
            # Attempt extraction (includes internal fix attempt)
            result = prompt_builder(prompt, LLM_CONFIG)
            extracted = result.get(entity_type, [])

            # Normalize fields
            extracted = normalize_entity_fields(extracted, entity_type)

            return extracted
        except Exception:
            if attempt < max_retries:
                print(f"⚠️  Retry {attempt + 1}/{max_retries}...", end=" ")
                continue
            else:
                print(f"✗ Failed after {max_retries} retries")
                return []

    return []


def extract_entities_from_chapter(chapter: Dict[str, Any], accumulated_entities: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    Extract entities from a single chapter by looping through entity types.
    Uses accumulated context from previous chapters for consistency.

    Args:
        chapter: Chapter data with paragraphs
        accumulated_entities: Accumulated lightweight contexts per entity type

    Returns:
        Extracted entities for this chapter (all types)
    """
    # Combine paragraphs into chapter text
    chapter_text = "\n\n".join(chapter["paragraphs"])
    chapter_num = chapter["chapter_num"]
    chapter_title = chapter["chapter_title"]

    # Initialize result structure
    entities = {"book_num": chapter["book_num"], "chapter_num": chapter_num, "chapter_title": chapter_title}

    print(f"\n📖 Chapter {chapter_num}: {chapter_title}")

    # Loop through each entity type
    for entity_type in ENTITY_TYPES:
        print(f"  {entity_type:12} ", end="")

        # Get context from previous chapters (same type only)
        previous_entities = accumulated_entities.get(entity_type, [])

        # Extract this entity type
        extracted = extract_single_entity_type(
            entity_type=entity_type, chapter_text=chapter_text, chapter_num=chapter_num, chapter_title=chapter_title, previous_entities=previous_entities, max_retries=2
        )

        entities[entity_type] = extracted
        print(f"→ {len(extracted)} found")

        # Merge into accumulated context for next chapter
        merge_into_accumulated(accumulated_entities[entity_type], extracted)

    return entities


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================


def extract_all_entities(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract entities from all chapters with checkpoint support.

    Args:
        chapters: List of chapter data

    Returns:
        List of entity extractions per chapter
    """
    checkpointManager = _CheckpointManager()

    # Load checkpoint if exists
    all_entities, accumulated_entities, start_chapter = checkpointManager.load_checkpoint()

    # Get chapters to process
    chapters_to_process = [ch for ch in chapters if ch["chapter_num"] >= start_chapter]

    if not chapters_to_process:
        print("\n✅ All chapters already processed!")
        return all_entities

    print(f"\n🔍 Processing {len(chapters_to_process)} chapters (starting from {start_chapter})...")
    print(f"Using model: {LLM_CONFIG['model']}")

    # Process chapters
    for chapter in chapters_to_process:
        try:
            start_time = time.time()

            # Extract entities from this chapter
            entities = extract_entities_from_chapter(chapter, accumulated_entities)

            elapsed = time.time() - start_time

            # Add to results
            all_entities.append(entities)

            # Save checkpoint after each chapter
            checkpointManager.save_checkpoint(all_entities, accumulated_entities)

            # Stats
            total_entities = sum(len(entities.get(et, [])) for et in ENTITY_TYPES)
            print(f"  ✅ Completed in {elapsed:.1f}s ({total_entities} entities) - Checkpoint saved")

        except Exception as e:
            print(f"\n❌ Critical error processing chapter {chapter['chapter_num']}: {e}")
            print("Progress saved in checkpoint. Fix issue and re-run to resume.")
            raise

    return all_entities


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
    start_time = time.time()

    config = get_config()
    paths = get_paths()
    global LLM_CONFIG, ENTITY_TYPES, ENTITY_SCHEMAS, SINGLE_ENTITY_EXTRACTION_PROMPT, FILE_ENTITIES_RAW
    LLM_CONFIG = config.LLM_CONFIG
    ENTITY_TYPES = config.ENTITY_TYPES
    ENTITY_SCHEMAS = config.ENTITY_SCHEMAS
    SINGLE_ENTITY_EXTRACTION_PROMPT = config.SINGLE_ENTITY_EXTRACTION_PROMPT
    FILE_ENTITIES_RAW = paths.FILE_ENTITIES_RAW

    chapters = load_json_from_file(paths.FILE_BOOK_00_PROCESSED)

    testing_connection(LLM_CONFIG)

    # Extract entities
    all_entities = extract_all_entities(chapters)

    # Print statistics
    statistics = generate_statistics(all_entities)

    # Save results
    save_json_to_file(all_entities, FILE_ENTITIES_RAW, indent=2, log=False)

    total_time = time.time() - start_time
    total_statistics_logging(
        statistics,
        total_time,
        "ENTITIES EXTRACTION",
        "02_extract_entities",
        tables=True,
        configuration_section=LLM_CONFIG,
    )


if __name__ == "__main__":
    main()
