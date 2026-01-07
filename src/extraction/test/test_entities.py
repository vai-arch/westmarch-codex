import json
from collections import defaultdict
from typing import Any, Dict, List

from tabulate import tabulate

from src.paths import get_paths
from src.utils.util_files_functions import load_json_from_file


def extract_character_chapters(chapters: List[Dict]) -> Dict[str, List[int]]:
    """
    Returns a dict:
    {
        "Bilbo Baggins": [1, 2, 3, ...],
        "Gandalf": [1, 2, ...],
        ...
    }
    """
    character_chapters = defaultdict(set)

    for chapter in chapters:
        chapter_num = chapter.get("chapter_num")
        for character in chapter.get("characters", []):
            name = character.get("name")
            if name:
                character_chapters[name].add(chapter_num)

    # Convert sets to sorted lists
    return {name: sorted(chapters) for name, chapters in character_chapters.items()}


def print_character_chapter_table(chapters):
    # Collect all chapter numbers
    chapter_nums = sorted(ch["chapter_num"] for ch in chapters)

    # Map character → set of chapters
    character_map = {}

    for chapter in chapters:
        chap_num = chapter["chapter_num"]
        for char in chapter.get("characters", []):
            name = char["name"]
            character_map.setdefault(name, set()).add(chap_num)

    # Build table rows
    headers = ["Character"] + [f"Ch {n}" for n in chapter_nums]
    rows = []

    for character in sorted(character_map.keys()):
        row = [character]
        for chap_num in chapter_nums:
            row.append("X" if chap_num in character_map[character] else "")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="grid"))


def extract_all_entities_by_chapter(chapters: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[int]]]:
    """
    Returns:
    {
        entity_type: {
            entity_name: [chapter_nums...]
        }
    }
    """

    entity_map: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for chapter in chapters:
        chapter_num = chapter.get("chapter_num")

        for key, value in chapter.items():
            # We only care about entity lists
            if not isinstance(value, list):
                continue

            for entity in value:
                if not isinstance(entity, dict):
                    continue

                name = entity.get("name")
                if not name:
                    continue

                # Avoid duplicates
                if chapter_num not in entity_map[key][name]:
                    entity_map[key][name].append(chapter_num)

    # Convert defaultdicts → dicts
    return {entity_type: dict(entities) for entity_type, entities in entity_map.items()}


if __name__ == "__main__":
    paths = get_paths()
    entities = load_json_from_file(paths.FILE_ENTITIES_RAW)
    characters = extract_character_chapters(entities)
    print(characters)
    print_character_chapter_table(entities)
    print(json.dumps(extract_all_entities_by_chapter(entities), indent=2))
