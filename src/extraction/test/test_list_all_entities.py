"""
Quick review script to check deduplicated entities
"""

from src.paths import get_paths
from src.utils.util_files_functions import load_json_from_file


def main():
    paths = get_paths()
    data = load_json_from_file(paths.FILE_ENTITIES_DEDUPLICATED)

    for entity_type in ["characters", "locations", "races", "objects", "events", "groups", "concepts"]:
        if entity_type not in data:
            continue

        print(f"\n{'=' * 80}")
        print(f"{entity_type.upper()}")
        print(f"{'=' * 80}")

        entities = data[entity_type]
        for entity in sorted(entities, key=lambda x: x["name"]):
            name = entity["name"]
            aliases = entity.get("aliases", [])
            chapters = entity.get("source_chapters", [])

            # Format output
            alias_str = f" | Aliases: {', '.join(aliases)}" if aliases else ""
            chapter_str = f" | Chapters: {chapters}"
            chapter_str = ""

            print(f"{name}{alias_str}{chapter_str}")


if __name__ == "__main__":
    main()
