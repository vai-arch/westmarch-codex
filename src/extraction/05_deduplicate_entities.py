"""
Entity Deduplication & Enrichment
Transforms raw entity extractions into clean, deduplicated knowledge base.

Strategy:
1. Within-chapter deduplication (remove duplicates in same chapter)
2. Cross-chapter rule-based matching (exact name, fuzzy matching, alias overlap)
3. LLM review for ambiguous cases (hybrid mode only)
4. Consolidate and enrich final entities
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Tuple

from fuzzywuzzy import fuzz

from src.config import get_config
from src.paths import get_paths
from src.utils.llm_access.ollama_llm import prompt_builder
from src.utils.util_files_functions import load_json_from_file, load_jsonl_from_file, save_json_to_file
from src.utils.util_statistics import total_statistics_logging

# Global configs
DEDUPLICATION_CONFIG = None
DEDUPLICATION_LLM_CONFIG = None
DEDUPLICATION_CLEANUP_PROMPT = None
REASONING_MODEL_CONFIG = None
FAST_MODEL_CONFIG = None
CHARACTER_STATS_PROMPT = None
ENTITY_TYPES = None
ENTITY_EXTRACTION_CONFIG = None
FILE_BOOKS_CHUNKED = None


def deduplicate_within_chapter(chapter_entities: List[Dict]) -> List[Dict]:
    """Remove duplicate entities within a single chapter."""
    seen = {}

    for entity_type in ENTITY_TYPES:
        if entity_type not in chapter_entities:
            continue

        for entity in chapter_entities[entity_type]:
            name = entity.get("name")
            if not name:
                continue

            key = (entity_type, name.lower().strip())

            if key not in seen:
                seen[key] = entity
            else:
                # Merge aliases and other data
                seen[key] = merge_entities(seen[key], entity)

    # Reconstruct chapter structure
    result = {"book_num": chapter_entities["book_num"], "chapter_num": chapter_entities["chapter_num"], "chapter_title": chapter_entities["chapter_title"]}

    # Group back by type
    for entity_type in ENTITY_TYPES:
        result[entity_type] = [e for k, e in seen.items() if k[0] == entity_type]

    return result


def extract_all_entities_flat(chapters: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract all entities from all chapters, flattened by type."""
    entities_by_type = defaultdict(list)

    for chapter in chapters:
        for entity_type in ENTITY_TYPES:
            if entity_type in chapter:
                for entity in chapter[entity_type]:
                    if entity.get("name"):
                        # Add chapter context
                        entity["source_chapters"] = [chapter["chapter_num"]]
                        entities_by_type[entity_type].append(entity)

    return entities_by_type


def are_entities_similar(e1: Dict, e2: Dict, entity_type: str) -> bool:
    """Determine if two entities are likely the same using fuzzy matching and alias overlap."""
    name1 = e1["name"].lower().strip()
    name2 = e2["name"].lower().strip()

    # Exact match
    if name1 == name2:
        return True

    # Length-aware fuzzy matching threshold
    min_length = min(len(name1), len(name2))
    if min_length <= 5:
        required_threshold = 95
    elif min_length <= 10:
        required_threshold = 90
    else:
        required_threshold = DEDUPLICATION_CONFIG["fuzzy_threshold"]  # 85

    fuzzy_score = fuzz.ratio(name1, name2)
    if fuzzy_score >= required_threshold:
        return True

    # Check alias overlap
    aliases1 = set(a.lower().strip() for a in e1.get("aliases", []))
    aliases2 = set(a.lower().strip() for a in e2.get("aliases", []))

    # If either name appears in the other's aliases
    if name1 in aliases2 or name2 in aliases1:
        return True

    # If they share any aliases
    if aliases1 and aliases2 and aliases1.intersection(aliases2):
        return True

    return False


def consolidate_relationships(entities: List[Dict]) -> List[Dict]:
    """Consolidate multiple relationships per entity into grouped format."""
    for entity in entities:
        if "relationships" in entity and entity["relationships"]:
            # Group by entity
            entity_rels = {}
            for rel in entity["relationships"]:
                target = rel.get("entity")
                relationship = rel.get("relationship")
                if target and relationship:
                    if target not in entity_rels:
                        entity_rels[target] = set()
                    entity_rels[target].add(relationship)

            # Replace with consolidated format
            entity["relationships"] = [{"entity": target, "relationships": sorted(list(rels))} for target, rels in sorted(entity_rels.items())]
    return entities


def merge_entities(e1: Dict, e2: Dict) -> Dict:
    """Merge two entities, combining all information."""
    # Choose canonical name (longest by default)
    if DEDUPLICATION_CONFIG["canonical_name_strategy"] == "longest":
        canonical_name = max([e1["name"], e2["name"]], key=len)
    else:
        canonical_name = e1["name"]  # Default to first

    # Merge aliases
    all_aliases = set(e1.get("aliases", []) + e2.get("aliases", []))
    all_aliases.discard(canonical_name)
    if e1["name"] != canonical_name:
        all_aliases.add(e1["name"])
    if e2["name"] != canonical_name:
        all_aliases.add(e2["name"])

    merged = {"name": canonical_name, "aliases": sorted(list(all_aliases))}

    # Merge source chapters (special handling - always integers)
    chapters1 = e1.get("source_chapters", [])
    chapters2 = e2.get("source_chapters", [])
    if chapters1 or chapters2:
        merged["source_chapters"] = sorted(list(set(chapters1 + chapters2)))

    # Merge other fields - iterate through ALL keys from both entities
    all_keys = set(e1.keys()) | set(e2.keys())
    for key in all_keys:
        if key in ["name", "aliases", "source_chapters"]:
            continue

        val1 = e1.get(key)
        val2 = e2.get(key)

        # Skip if both are None
        if val1 is None and val2 is None:
            continue

        # If only one has value, use it
        if val1 is None:
            merged[key] = val2
            continue
        if val2 is None:
            merged[key] = val1
            continue

        # Both have values - merge based on type
        if isinstance(val1, list) and isinstance(val2, list):
            has_dicts = (val1 and isinstance(val1[0], dict)) or (val2 and isinstance(val2[0], dict))

            if has_dicts:
                # For lists of dicts, use JSON deduplication
                import json

                combined = val1 + val2
                unique_items = {}
                for item in combined:
                    item_json = json.dumps(item, sort_keys=True)
                    unique_items[item_json] = item
                merged[key] = list(unique_items.values())
            else:
                # For simple lists, use set
                merged[key] = sorted(list(set(val1 + val2)))
        elif isinstance(val1, str) and isinstance(val2, str):
            # For strings, prefer longer
            merged[key] = val1 if len(val1) > len(val2) else val2
        else:
            # Default to first
            merged[key] = val1

    return merged


def deduplicate_entity_type_rule_based(entities: List[Dict], entity_type: str) -> List[Dict]:
    """Deduplicate entities of a single type using rule-based matching."""
    if not entities:
        return []

    # Start with all entities as separate clusters
    clusters = [[e] for e in entities]

    # Merge clusters if entities are similar
    merged = True
    while merged:
        merged = False
        new_clusters = []
        used = set()

        for i, cluster1 in enumerate(clusters):
            if i in used:
                continue

            # Try to merge with subsequent clusters
            for j in range(i + 1, len(clusters)):
                if j in used:
                    continue

                # Check if any entity in cluster1 matches any in cluster2
                should_merge = False
                for e1 in cluster1:
                    for e2 in clusters[j]:
                        if are_entities_similar(e1, e2, entity_type):
                            should_merge = True
                            break
                    if should_merge:
                        break

                if should_merge:
                    cluster1.extend(clusters[j])
                    used.add(j)
                    merged = True

            new_clusters.append(cluster1)
            used.add(i)

        clusters = new_clusters

    # Merge all entities within each cluster
    deduplicated = []
    for cluster in clusters:
        merged_entity = cluster[0]
        for entity in cluster[1:]:
            merged_entity = merge_entities(merged_entity, entity)
        deduplicated.append(merged_entity)

    return deduplicated


def deduplicate_entities(raw_chapters: List[Dict]) -> Dict[str, List[Dict]]:
    """Main deduplication pipeline."""
    stats = {"within_chapter_removed": 0, "cross_chapter_merged": {}}

    # Step 1: Within-chapter deduplication
    print("\n=== Step 1: Within-chapter deduplication ===")
    if DEDUPLICATION_CONFIG["within_chapter_dedup"]:
        deduplicated_chapters = []
        for chapter in raw_chapters:
            original_count = sum(len(chapter.get(et, [])) for et in ENTITY_TYPES)
            deduped_chapter = deduplicate_within_chapter(chapter)
            new_count = sum(len(deduped_chapter.get(et, [])) for et in ENTITY_TYPES)
            stats["within_chapter_removed"] += original_count - new_count
            deduplicated_chapters.append(deduped_chapter)
        print(f"Removed {stats['within_chapter_removed']} duplicate entities within chapters")
    else:
        deduplicated_chapters = raw_chapters

    # Step 2: Extract all entities flat
    print("\n=== Step 2: Extracting entities across all chapters ===")
    entities_by_type = extract_all_entities_flat(deduplicated_chapters)

    for entity_type, entities in entities_by_type.items():
        print(f"{entity_type}: {len(entities)} entities")

    # Step 3: Cross-chapter deduplication
    print("\n=== Step 3: Cross-chapter deduplication ===")
    final_entities = {}

    for entity_type, entities in entities_by_type.items():
        print(f"\nProcessing {entity_type}...")
        original_count = len(entities)

        # Rule-based deduplication
        deduplicated = deduplicate_entity_type_rule_based(entities, entity_type)

        # LLM review for hybrid mode (optional)
        if DEDUPLICATION_CONFIG["strategy"] == "hybrid":
            # TODO: Identify ambiguous cases and send to LLM
            pass

        final_entities[entity_type] = deduplicated
        merged_count = original_count - len(deduplicated)
        stats["cross_chapter_merged"][entity_type] = merged_count

        print(f"  {original_count} → {len(deduplicated)} ({merged_count} merged)")

    return final_entities, stats


def generate_statistics(raw_chapters: List[Dict], deduplicated_entities: Dict[str, List[Dict]], dedup_stats: Dict) -> Tuple[List[Dict], List[Dict]]:
    """Generate statistics for the deduplication process."""
    overall_stats = []
    detail_stats = []

    # Overall stats
    raw_total = sum(sum(len(chapter.get(et, [])) for et in ENTITY_TYPES) for chapter in raw_chapters)

    final_total = sum(len(entities) for entities in deduplicated_entities.values())

    overall_stats.append(
        {
            "name": "Overall Summary",
            "metrics": {
                "raw_entities": raw_total,
                "within_chapter_removed": dedup_stats["within_chapter_removed"],
                "cross_chapter_merged": sum(dedup_stats["cross_chapter_merged"].values()),
                "final_entities": final_total,
                "reduction_pct": round((1 - final_total / raw_total) * 100, 2),
            },
        }
    )

    # Per entity type stats
    for entity_type in ENTITY_TYPES:
        entities = deduplicated_entities.get(entity_type, [])
        raw_count = sum(len(chapter.get(entity_type, [])) for chapter in raw_chapters)

        if raw_count == 0:
            continue

        avg_chapters = sum(len(e.get("source_chapters", [])) for e in entities) / len(entities) if entities else 0
        total_aliases = sum(len(e.get("aliases", [])) for e in entities)
        avg_aliases = total_aliases / len(entities) if entities else 0

        detail_stats.append(
            {
                "name": entity_type.capitalize(),
                "metrics": {
                    "raw": raw_count,
                    "final": len(entities),
                    "merged": dedup_stats["cross_chapter_merged"].get(entity_type, 0),
                    "avg_chapters": round(avg_chapters, 2),
                    "total_aliases": total_aliases,
                    "avg_aliases": round(avg_aliases, 2),
                },
            }
        )

    return overall_stats, detail_stats


def llm_validate_and_cleanup(deduplicated_entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
    """Use LLM to validate, reclassify, merge duplicates, and clean up entities."""

    print("\n=== Step 4: LLM Validation & Cleanup ===")

    cleaned_entities = {}

    # Process each entity type separately
    for entity_type in ENTITY_TYPES:
        if entity_type not in deduplicated_entities:
            cleaned_entities[entity_type] = []
            continue

        entities = deduplicated_entities[entity_type]
        if not entities:
            cleaned_entities[entity_type] = []
            continue

        print(f"\nCleaning {entity_type}... ({len(entities)} entities)")

        entities_json = json.dumps({entity_type: entities}, separators=(",", ":"))
        prompt = DEDUPLICATION_CLEANUP_PROMPT[entity_type].format(entities=entities_json)

        try:
            result = prompt_builder(prompt, REASONING_MODEL_CONFIG)

            if isinstance(result, dict) and entity_type in result:
                cleaned_entities[entity_type] = result[entity_type]
                print(f"  ✓ Cleaned: {len(result[entity_type])} entities")
            else:
                print("  ✗ Failed, keeping original")
                cleaned_entities[entity_type] = entities

        except Exception as e:
            print(f"  ✗ Error: {e}, keeping original")
            cleaned_entities[entity_type] = entities

    # Derive races from cleaned characters
    print("\nDeriving races from characters...")
    cleaned_entities["races"] = derive_races_from_characters(cleaned_entities.get("characters", []))
    print(f"  ✓ Derived {len(cleaned_entities['races'])} races")

    return cleaned_entities


def derive_races_from_characters(characters: List[Dict]) -> List[Dict]:
    """Extract unique races from character entities."""

    race_data = {}

    for char in characters:
        race = char.get("race")
        if not race:
            continue

        if race not in race_data:
            race_data[race] = {"name": race, "aliases": [], "notable_members": [], "characteristics": [], "source_chapters": []}

        # Add character as notable member
        race_data[race]["notable_members"].append(char["name"])

        # Merge source chapters
        race_data[race]["source_chapters"].extend(char.get("source_chapters", []))

    # Deduplicate and sort
    for race in race_data.values():
        race["notable_members"] = sorted(list(set(race["notable_members"])))
        race["source_chapters"] = sorted(list(set(race["source_chapters"])))

    return list(race_data.values())


def print_entities_summary(entities: Dict[str, List[Dict]]):
    for entity_type in ENTITY_TYPES:
        if entity_type not in entities:
            continue

        print(f"{'=' * 80}")
        print(f"{entity_type.upper()}")
        print(f"{'=' * 80}")

        entities = entities[entity_type]
        for entity in sorted(entities, key=lambda x: x["name"]):
            name = entity["name"]
            aliases = entity.get("aliases", [])
            chapters = entity.get("source_chapters", [])

            # Format output
            alias_str = f" | Aliases: {', '.join(aliases)}" if aliases else ""
            chapter_str = f" | Chapters: {chapters}"
            chapter_str = ""

            print(f"{name}{alias_str}{chapter_str}")


def enhance_all_characters(characters: List[Dict[str, Any]], chapters: List[Dict[str, Any]], chunk_size: int = 10000, max_retries: int = 2) -> List[Dict[str, Any]]:
    """
    Enhance all characters with prominence statistics.
    Analyzes mentions, dialogues, and actions per chapter (with chunking).

    Args:
        characters: List of character entities
        chapters: List of all book chapters
        chunk_size: Max characters per chunk

    Returns:
        List of enhanced characters with statistics
    """

    enhanced_characters = []

    print(f"\n🔍 Analyzing {len(characters)} characters across {len(chapters)} chapters...")

    for char_idx, character in enumerate(characters, 1):
        character_name = character["name"]
        aliases = character.get("aliases", [])
        aliases_str = ", ".join(aliases) if aliases else "none"
        new_aliases = []
        chapter_stats = []
        justifications = []

        total_mentions = 0
        total_dialogues = 0
        total_actions = 0

        # Build list of all names to search for
        search_terms = [character_name]
        search_terms.extend(aliases)

        print(f"\n[{char_idx}/{len(characters)}] Analyzing '{character_name}'... {search_terms}", end=" ")

        for chapter in chapters:
            chapter_text = "\n\n".join(chapter["paragraphs"])
            chapter_num = chapter["chapter_num"]

            # Check config for chunking strategy
            use_semantic_chunks = ENTITY_EXTRACTION_CONFIG["use_semantic_chunks"]

            if use_semantic_chunks:
                # Load semantic chunks for this chapter
                books_chunks = load_jsonl_from_file(FILE_BOOKS_CHUNKED)
                chapter_chunks = [c for c in books_chunks if c["chapter_number"] == chapter_num]
                all_chunks = [c["text"] for c in chapter_chunks]
            else:
                # Legacy: character-based splitting
                all_chunks = []
                for i in range(0, len(chapter_text), chunk_size):
                    all_chunks.append(chapter_text[i : i + chunk_size])

            filtered_chunks = []
            # Rest stays the same - check if character appears in chunks
            for chunk in all_chunks:
                found = False
                for term in search_terms:
                    if not term:
                        continue
                    if " " in term:
                        if term.lower() in chunk.lower():
                            found = True
                            break
                    else:
                        if re.search(r"\b" + re.escape(term) + r"\b", chunk, re.IGNORECASE):
                            found = True
                            break
                if found:
                    filtered_chunks.append(chunk)

            if not filtered_chunks:
                print("Skipping chapter, no quick mentions found.", end=" ")
                continue

            # Accumulate stats across chunks for this chapter
            chapter_mentions = 0
            chapter_dialogues = 0
            chapter_actions = 0
            for chunk_idx, chunk in enumerate(filtered_chunks):
                prompt = CHARACTER_STATS_PROMPT.format(character_name=character_name, aliases=aliases_str, chapter_text=chunk, chapter_num=chapter_num)

                for attempt in range(max_retries + 1):
                    try:
                        result = prompt_builder(prompt, REASONING_MODEL_CONFIG)

                        chapter_mentions += result.get("mentions_by_others", 0)
                        chapter_dialogues += result.get("dialogues", 0)
                        chapter_actions += result.get("actions", 0)
                        justifications.append(result.get("justification", ""))
                        new_aliases.extend(result.get("new_aliases", []))
                        print(f"---> {result}")
                        break

                    except Exception:
                        if attempt < max_retries:
                            continue
                        else:
                            # Failed chunk - continue to next
                            break

            # Store aggregated stats for this chapter
            chapter_stats.append({"chapter_num": chapter_num, "mentions": chapter_mentions, "dialogues": chapter_dialogues, "actions": chapter_actions})

            total_mentions += chapter_mentions
            total_dialogues += chapter_dialogues
            total_actions += chapter_actions

        print(f"✓ {total_mentions} mentions, {total_dialogues} dialogues, {total_actions} actions")
        if justifications:
            print(justifications)

        # Add statistics to character
        character["statistics"] = {
            "total_mentions": total_mentions,
            "total_dialogues": total_dialogues,
            "total_actions": total_actions,
            "chapters_appeared": len([s for s in chapter_stats if s["mentions"] > 0]),
            "per_chapter": chapter_stats,
        }

        # Normalize (optional but strongly recommended)
        def normalize(a):
            return a.strip().lower()

        normalized_existing = {normalize(a): a for a in aliases}
        normalized_new = {normalize(a): a for a in new_aliases}

        # Merge, preferring existing capitalization if present
        merged_aliases = dict(normalized_existing)
        for key, original in normalized_new.items():
            if key not in merged_aliases:
                merged_aliases[key] = original

        # Write back
        character["aliases"] = sorted(merged_aliases.values())

        enhanced_characters.append(character)

    return enhanced_characters


def main():
    start_time = datetime.now()
    config = get_config()
    paths = get_paths()

    global \
        DEDUPLICATION_CONFIG, \
        DEDUPLICATION_LLM_CONFIG, \
        ENTITY_TYPES, \
        DEDUPLICATION_CLEANUP_PROMPT, \
        REASONING_MODEL_CONFIG, \
        FAST_MODEL_CONFIG, \
        CHARACTER_STATS_PROMPT, \
        FILE_BOOKS_CHUNKED, \
        ENTITY_EXTRACTION_CONFIG
    DEDUPLICATION_CONFIG = config.DEDUPLICATION_CONFIG
    DEDUPLICATION_LLM_CONFIG = config.DEDUPLICATION_LLM_CONFIG
    DEDUPLICATION_CLEANUP_PROMPT = config.DEDUPLICATION_CLEANUP_PROMPT
    REASONING_MODEL_CONFIG = config.REASONING_MODEL_CONFIG
    FAST_MODEL_CONFIG = config.FAST_MODEL_CONFIG
    ENTITY_EXTRACTION_CONFIG = config.ENTITY_EXTRACTION_CONFIG
    FILE_BOOKS_CHUNKED = paths.FILE_BOOKS_CHUNKED
    ENTITY_TYPES = config.ENTITY_TYPES
    CHARACTER_STATS_PROMPT = config.CHARACTER_STATS_PROMPT

    # Load raw entities
    raw_chapters = load_json_from_file(paths.FILE_ENTITIES_RAW)

    # Deduplicate
    deduplicated_entities, dedup_stats = deduplicate_entities(raw_chapters)

    print_entities_summary(deduplicated_entities)

    # Consolidate relationships
    for entity_type in deduplicated_entities:
        deduplicated_entities[entity_type] = consolidate_relationships(deduplicated_entities[entity_type])

    # Load chapters for analysis
    chapters = load_json_from_file(paths.FILE_BOOK_00_PROCESSED)

    # Enhance characters with statistics
    enhanced_characters = enhance_all_characters(deduplicated_entities.get("characters", []), chapters)

    # Update deduplicated entities with enhanced characters
    deduplicated_entities["characters"] = enhanced_characters

    # Save deduplicated entities
    save_json_to_file(deduplicated_entities, paths.FILE_ENTITIES_DEDUPLICATED, indent=2)

    # Generate statistics
    overall_stats, detail_stats = generate_statistics(raw_chapters, deduplicated_entities, dedup_stats)

    # Log statistics
    total_time = datetime.now() - start_time

    print_entities_summary(deduplicated_entities)

    total_statistics_logging(
        detail_stats,
        total_time,
        "ENTITY DEDUPLICATION - DETAILS",
        "03_deduplicate_entities_details",
        tables=True,
        configuration_section=DEDUPLICATION_CONFIG,
    )


def testing_enhance_all_characters():
    config = get_config()
    paths = get_paths()

    global DEDUPLICATION_CONFIG, DEDUPLICATION_LLM_CONFIG, ENTITY_TYPES, DEDUPLICATION_CLEANUP_PROMPT, REASONING_MODEL_CONFIG, FAST_MODEL_CONFIG, CHARACTER_STATS_PROMPT
    DEDUPLICATION_CONFIG = config.DEDUPLICATION_CONFIG
    DEDUPLICATION_LLM_CONFIG = config.DEDUPLICATION_LLM_CONFIG
    DEDUPLICATION_CLEANUP_PROMPT = config.DEDUPLICATION_CLEANUP_PROMPT
    REASONING_MODEL_CONFIG = config.MEDIUM_MODEL_CONFIG
    FAST_MODEL_CONFIG = config.FAST_MODEL_CONFIG
    ENTITY_TYPES = config.ENTITY_TYPES
    CHARACTER_STATS_PROMPT = config.CHARACTER_STATS_PROMPT

    # Pick what you want to test
    TEST_CHARACTER_NAME = "Bard"
    TEST_CHAPTER = 15

    # Load chapters for analysis
    chapters = load_json_from_file(paths.FILE_BOOK_00_PROCESSED)
    characters = load_json_from_file(paths.FILE_ENTITIES_DEDUPLICATED).get("characters", [])

    # Filter characters
    test_characters = [c for c in characters if c.get("name") == TEST_CHARACTER_NAME]

    # Filter chapters
    test_chapters = [ch for ch in chapters if ch.get("chapter_num") == TEST_CHAPTER]

    # Run enrichment only on this small slice
    debug_result = enhance_all_characters(characters=test_characters, chapters=test_chapters, chunk_size=1000, max_retries=2)

    # Inspect output
    from pprint import pprint

    pprint(debug_result)


if __name__ == "__main__":
    main()
    # config = get_config()
    # paths = get_paths()
    # DEDUPLICATION_CONFIG = config.DEDUPLICATION_CONFIG
    # DEDUPLICATION_LLM_CONFIG = config.DEDUPLICATION_LLM_CONFIG
    # DEDUPLICATION_CLEANUP_PROMPT = config.DEDUPLICATION_CLEANUP_PROMPT
    # REASONING_MODEL_CONFIG = config.REASONING_MODEL_CONFIG
    # FAST_MODEL_CONFIG = config.FAST_MODEL_CONFIG
    # ENTITY_EXTRACTION_CONFIG = config.ENTITY_EXTRACTION_CONFIG
    # FILE_BOOKS_CHUNKED = paths.FILE_BOOKS_CHUNKED
    # ENTITY_TYPES = config.ENTITY_TYPES
    # CHARACTER_STATS_PROMPT = config.CHARACTER_STATS_PROMPT
    # testing_enhance_all_characters()
