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
import json
import time
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Counter, Dict, List

from src.config import get_config
from src.paths import get_paths
from src.utils.llm_access.ollama_llm import prompt_builder, testing_connection
from src.utils.util_files_functions import load_json_from_file, load_jsonl_from_file, save_json_to_file
from src.utils.util_statistics import total_statistics_logging

STAGE_1_LLM_CONFIG = None
STAGE_3_LLM_CONFIG = None
ENTITY_TYPES = None
ENTITY_EXTRACTION_CONFIG = None
ACCUMULATED_ENTITY_TYPES = None
ENTITY_SCHEMAS = None
UNIFIED_EXTRACTION_PROMPT = None
VALIDATION_PROMPT = None
FILE_ENTITIES_RAW = None
ENTITY_EXTRACTION = None
DATA_TEMP_PATH = None
FILE_BOOKS_CHUNKED = None
checkpoint_manager = None


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


def build_context_section(accumulated_entities: Dict[str, List[Dict[str, str]]]) -> str:
    """Build context section from accumulated entities across all types."""

    if not any(accumulated_entities.values()):
        return ""

    context_lines = ["\nPREVIOUSLY EXTRACTED ENTITIES FROM EARLIER CHAPTERS:", "Use these for consistent naming and to recognize aliases:\n"]

    for entity_type in ACCUMULATED_ENTITY_TYPES:
        entities = accumulated_entities.get(entity_type, [])
        if not entities:
            continue

        context_lines.append(f"\n{entity_type.upper()}:")
        for entity in entities[:20]:  # Limit per type
            name = entity.get("name", "")
            aliases = entity.get("aliases", [])
            if aliases:
                context_lines.append(f"  - {name} (also: {', '.join(aliases)})")
            else:
                context_lines.append(f"  - {name}")

    context_lines.append("")
    return "\n".join(context_lines)


def update_accumulated_entities(accumulated: Dict[str, List[Dict[str, str]]], new_entities: Dict[str, Any]):
    """Update accumulated entities with new chapter's entities."""

    for entity_type in ACCUMULATED_ENTITY_TYPES:
        entities = new_entities.get(entity_type, [])

        for entity in entities:
            name = entity.get("name", "")
            aliases = entity.get("aliases", [])

            if not name:
                continue

            # Check if already exists
            existing = None
            for acc_entity in accumulated[entity_type]:
                if acc_entity["name"] == name:
                    existing = acc_entity
                    break

            if existing:
                # Merge aliases
                existing_aliases = set(existing.get("aliases", []))
                new_aliases = set(aliases)
                existing["aliases"] = list(existing_aliases | new_aliases)
            else:
                # Add new
                accumulated[entity_type].append({"name": name, "aliases": aliases if aliases else []})


# =============================================================================
# TWO-STAGE EXTRACTION APPROACH
# =============================================================================


def stage1_unified_extraction(chapter_text: str, chapter_num: int, chapter_title: str, accumulated_entities: Dict[str, List[Dict[str, str]]], max_retries: int = 2) -> List[Dict[str, Any]]:
    """
    Stage 1: Extract all entities in unified format.
    Splits long chapters into chunks, extracts from each, merges results.

    Returns:
        List of entities with: entity_type, name, description, aliases
    """

    # Check config for chunking strategy
    use_semantic_chunks = ENTITY_EXTRACTION_CONFIG.get("use_semantic_chunks", True)

    if use_semantic_chunks:
        # Load semantic chunks for this chapter
        all_chunks = load_jsonl_from_file(FILE_BOOKS_CHUNKED)
        chapter_chunks = [c for c in all_chunks if c["chapter_number"] == chapter_num]
        chunks = [c["text"] for c in chapter_chunks]
        print(f"Using {len(chunks)} semantic chunks")
    else:
        # Legacy: character-based splitting
        chunk_size = ENTITY_EXTRACTION_CONFIG.get("context_window_chars", 10000)
        chunks = [chapter_text[i : i + chunk_size] for i in range(0, len(chapter_text), chunk_size)]

    # Build context once (shared across chunks)
    context_section = build_context_section(accumulated_entities)

    all_entities = []

    # Extract from each chunk
    for chunk_idx, chunk in enumerate(chunks):
        print(f"[chunk {chunk_idx + 1}/{len(chunks)}] ", end="")

        prompt = UNIFIED_EXTRACTION_PROMPT.format(context_section=context_section, chapter_text=chunk, chapter_num=chapter_num, chapter_title=chapter_title)

        for attempt in range(max_retries + 1):
            try:
                result = prompt_builder(prompt, STAGE_1_LLM_CONFIG)
                entities = result.get("entities", [])
                all_entities.extend(entities)
                break

            except Exception as e:
                if attempt < max_retries:
                    print(f"⚠️  Retry {attempt + 1}/{max_retries}...", end=" ")
                    continue
                else:
                    print(f"✗ Chunk {chunk_idx + 1} failed: {e}")
                    break

    return all_entities


def similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold


def merge_dicts(*dicts):
    merged = defaultdict(list)
    for d in dicts:
        if not d:
            continue
        for k, v in d.items():
            merged[k].extend(v)
    return dict(merged)


def stage2_merge_entities(entities):
    merged = {}

    for e in entities:
        key = (e["entity_type"], e["name"])  # STRICT match

        if key not in merged:
            merged[key] = {"entity_type": e["entity_type"], "name": e["name"], "description": [e["description"]], "aliases": list(e.get("aliases", []))}
            continue

        m = merged[key]

        # --- merge descriptions (fuzzy) ---
        desc = e["description"]
        if not any(similar(desc, d) for d in m["description"]):
            m["description"].append(desc)

        # --- merge aliases (fuzzy) ---
        for alias in e.get("aliases", []):
            if not any(similar(alias, a) for a in m["aliases"]):
                m["aliases"].append(alias)

    return list(merged.values())


def stage3_validate_and_structure(entity_types: List[str], raw_entities: List[Dict[str, Any]], chapter_num: int, chapter_title: str, max_retries: int = 2) -> Dict[str, List[Dict[str, Any]]]:
    """
    Stage 2: Validate and structure entities with proper schemas.

    Returns:
        Dict with all 8 entity types properly structured
    """

    if not raw_entities:
        # Return empty structure
        return {entity_type: [] for entity_type in entity_types}

    # Build schemas for prompt
    schemas = {
        "characters_schema": json.dumps(ENTITY_SCHEMAS["characters"]["format"], indent=2),
        "locations_schema": json.dumps(ENTITY_SCHEMAS["locations"]["format"], indent=2),
        "objects_schema": json.dumps(ENTITY_SCHEMAS["objects"]["format"], indent=2),
        "events_schema": json.dumps(ENTITY_SCHEMAS["events"]["format"], indent=2),
        "groups_schema": json.dumps(ENTITY_SCHEMAS["groups"]["format"], indent=2),
        "concepts_schema": json.dumps(ENTITY_SCHEMAS["concepts"]["format"], indent=2),
    }

    # Build prompt
    prompt = VALIDATION_PROMPT.format(raw_entities=json.dumps(raw_entities, indent=2), chapter_num=chapter_num, chapter_title=chapter_title, **schemas)

    for attempt in range(max_retries + 1):
        try:
            result = prompt_builder(prompt, STAGE_3_LLM_CONFIG)

            # Ensure all entity types present
            structured = {entity_type: result.get(entity_type, []) for entity_type in ENTITY_TYPES}
            return structured

        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️  Retry {attempt + 1}/{max_retries}...", end=" ")
                continue
            else:
                print(f"✗ Stage 2 failed: {e}")
                return {entity_type: [] for entity_type in ENTITY_TYPES}

    return {entity_type: [] for entity_type in ENTITY_TYPES}


def extract_entities_from_chapter(chapter: Dict[str, Any], accumulated_entities: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
    """
    Two-stage extraction for a single chapter.
    """
    chapter_text = "\n\n".join(chapter["paragraphs"])
    chapter_num = chapter["chapter_num"]
    chapter_title = chapter["chapter_title"]

    print(f"\n📖 Chapter {chapter_num}: {chapter_title}")

    # Stage 1: Unified extraction
    print("  Stage 1: Extracting all entities...", end=" ")
    raw_entities = stage1_unified_extraction(chapter_text, chapter_num, chapter_title, accumulated_entities)
    save_json_to_file(raw_entities, DATA_TEMP_PATH / "extract_entities" / str(FILE_ENTITIES_RAW.stem + f".chapter_{chapter_num}_stage1_TEMP.json"), indent=2)
    print(f"✓ {len(raw_entities)} entities found")

    # Stage 2: Merge entities before validation to reduce size of the prompt
    merged_entities = stage2_merge_entities(raw_entities)
    save_json_to_file(merged_entities, DATA_TEMP_PATH / "extract_entities" / str(FILE_ENTITIES_RAW.stem + f".chapter_{chapter_num}_stage2_TEMP.json"), indent=2)
    print(f"Merged Entities: {dict(Counter(e['entity_type'] for e in merged_entities))}")

    # Stage 3: Validate and structure
    print("  Stage 3: Validating and structuring...", end=" ")
    entity_groups = defaultdict(list)
    for e in merged_entities:
        entity_groups[e["entity_type"]].append(e)
    structured_locations = stage3_validate_and_structure(["locations"], entity_groups.get("location", []), chapter_num, chapter_title)
    structured_objects = stage3_validate_and_structure(["objects"], entity_groups.get("object", []), chapter_num, chapter_title)
    structured_events = stage3_validate_and_structure(["events"], entity_groups.get("event", []), chapter_num, chapter_title)
    structured_concepts = stage3_validate_and_structure(["concepts"], entity_groups.get("concept", []), chapter_num, chapter_title)
    structured_characters_groups = stage3_validate_and_structure(["characters", "groups"], entity_groups.get("character", []) + entity_groups.get("group", []), chapter_num, chapter_title)
    structured_entities = merge_dicts(structured_locations, structured_objects, structured_events, structured_concepts, structured_characters_groups)
    save_json_to_file(merged_entities, DATA_TEMP_PATH / "extract_entities" / str(FILE_ENTITIES_RAW.stem + f".chapter_{chapter_num}_stage3_TEMP.json"), indent=2)

    # Count totals
    total = sum(len(structured_entities.get(et, [])) for et in ENTITY_TYPES)
    print(f"✓ {total} entities structured")

    # Show breakdown
    for entity_type in ENTITY_TYPES:
        count = len(structured_entities.get(entity_type, []))
        if count > 0:
            print(f"    {entity_type:12} {count}")

    # Build final structure
    result = {"book_num": chapter["book_num"], "chapter_num": chapter_num, "chapter_title": chapter_title}
    result.update(structured_entities)

    # Update accumulated entities
    update_accumulated_entities(accumulated_entities, structured_entities)

    return result


# =============================================================================
# MAIN EXTRACTION PIPELINE
# =============================================================================


def extract_all_entities(chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract entities from all chapters with checkpoint support."""

    all_entities, accumulated_entities, start_chapter = checkpoint_manager.load_checkpoint()

    chapters_to_process = [ch for ch in chapters if ch["chapter_num"] >= start_chapter]

    if not chapters_to_process:
        print("\n✅ All chapters already processed!")
        return all_entities

    print(f"\n🔍 Processing {len(chapters_to_process)} chapters (starting from {start_chapter})...")

    for chapter in chapters_to_process:
        try:
            start_time = time.time()

            entities = extract_entities_from_chapter(chapter, accumulated_entities)

            elapsed = time.time() - start_time

            all_entities.append(entities)

            checkpoint_manager.save_checkpoint(all_entities, accumulated_entities)

            print(f"  ✅ Completed in {elapsed:.1f}s - Checkpoint saved\n")

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
    global \
        STAGE_1_LLM_CONFIG, \
        ENTITY_TYPES, \
        ENTITY_SCHEMAS, \
        UNIFIED_EXTRACTION_PROMPT, \
        FILE_ENTITIES_RAW, \
        DATA_TEMP_PATH, \
        FILE_BOOKS_CHUNKED, \
        VALIDATION_PROMPT, \
        checkpoint_manager, \
        ACCUMULATED_ENTITY_TYPES, \
        ENTITY_EXTRACTION_CONFIG, \
        STAGE_3_LLM_CONFIG

    STAGE_1_LLM_CONFIG = config.MEDIUM_MODEL_CONFIG
    STAGE_3_LLM_CONFIG = config.MEDIUM_MODEL_CONFIG
    ENTITY_EXTRACTION_CONFIG = config.ENTITY_EXTRACTION_CONFIG
    ENTITY_TYPES = config.ENTITY_TYPES
    ENTITY_SCHEMAS = config.ENTITY_SCHEMAS
    UNIFIED_EXTRACTION_PROMPT = config.UNIFIED_EXTRACTION_PROMPT
    VALIDATION_PROMPT = config.VALIDATION_PROMPT
    FILE_ENTITIES_RAW = paths.FILE_ENTITIES_RAW
    DATA_TEMP_PATH = paths.DATA_TEMP_PATH
    ACCUMULATED_ENTITY_TYPES = config.ACCUMULATED_ENTITY_TYPES
    FILE_BOOKS_CHUNKED = paths.FILE_BOOKS_CHUNKED

    checkpoint_manager = _CheckpointManager()

    testing_connection(STAGE_1_LLM_CONFIG)
    testing_connection(STAGE_3_LLM_CONFIG)

    chapters = load_json_from_file(paths.FILE_BOOK_00_PROCESSED)

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
        configuration_section=STAGE_1_LLM_CONFIG,
    )


if __name__ == "__main__":
    main()
    # config = get_config()
    # paths = get_paths()

    # REASONING_MODEL_CONFIG = config.MEDIUM_MODEL_CONFIG
    # MEDIUM_MODEL_CONFIG = config.MEDIUM_MODEL_CONFIG
    # FAST_MODEL_CONFIG = config.FAST_MODEL_CONFIG
    # ENTITY_EXTRACTION_CONFIG = config.ENTITY_EXTRACTION_CONFIG
    # ENTITY_TYPES = config.ENTITY_TYPES
    # ENTITY_SCHEMAS = config.ENTITY_SCHEMAS
    # UNIFIED_EXTRACTION_PROMPT = config.UNIFIED_EXTRACTION_PROMPT
    # VALIDATION_PROMPT = config.VALIDATION_PROMPT
    # FILE_ENTITIES_RAW = paths.FILE_ENTITIES_RAW
    # DATA_TEMP_PATH = paths.DATA_TEMP_PATH
    # ACCUMULATED_ENTITY_TYPES = config.ACCUMULATED_ENTITY_TYPES
    # FILE_BOOKS_CHUNKED = paths.FILE_BOOKS_CHUNKED

    # entities = load_json_from_file("C:\\Users\\Usuario\\Documents\\_AI\\westmarch-codex\\data\\temp\\extract_entities\\entities_raw.chapter_1_stage1_TEMP.json")
    # merged_entities = stage2_merge_entities(entities)
    # chapter_num = 1
    # save_json_to_file(merged_entities, DATA_TEMP_PATH / "extract_entities" / str(FILE_ENTITIES_RAW.stem + f".chapter_{chapter_num}_stage2_TEMP.json"), indent=2)
    # print(f"Merged Entities: {dict(Counter(e['entity_type'] for e in merged_entities))}")
    # # Stage 3: Validate and structure
    # print("  Stage 3: Validating and structuring...", end=" ")
    # # structured_entities = stage3_validate_and_structure(merged_entities, chapter_num, chapter_title)

    # entity_groups = defaultdict(list)
    # for e in merged_entities:
    #     entity_groups[e["entity_type"]].append(e)
    # chapter_title = "TITLE"
    # structured_locations = stage3_validate_and_structure(["locations"], entity_groups.get("location", []), chapter_num, chapter_title)
    # structured_objects = stage3_validate_and_structure(["objects"], entity_groups.get("object", []), chapter_num, chapter_title)
    # structured_events = stage3_validate_and_structure(["events"], entity_groups.get("event", []), chapter_num, chapter_title)
    # structured_concepts = stage3_validate_and_structure(["concepts"], entity_groups.get("concept", []), chapter_num, chapter_title)
    # structured_characters_groups = stage3_validate_and_structure(["characters", "groups"], entity_groups.get("character", []) + entity_groups.get("group", []), chapter_num, chapter_title)

    # structured_entities = merge_dicts(structured_locations, structured_objects, structured_events, structured_concepts, structured_characters_groups)
    # save_json_to_file(merged_entities, DATA_TEMP_PATH / "extract_entities" / str(FILE_ENTITIES_RAW.stem + f".chapter_{chapter_num}_stage3_TEMP.json"), indent=2)

    # total = sum(len(structured_entities.get(et, [])) for et in ENTITY_TYPES)
    # print(f"✓ {total} entities structured")

    # # Show breakdown
    # for entity_type in ENTITY_TYPES:
    #     count = len(structured_entities.get(entity_type, []))
    #     if count > 0:
    #         print(f"    {entity_type:12} {count}")
