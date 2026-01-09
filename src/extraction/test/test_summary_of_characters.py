"""
Quick character analysis script
"""

from src.paths import get_paths
from src.utils.util_files_functions import load_json_from_file


def main():
    paths = get_paths()
    data = load_json_from_file(paths.FILE_ENTITIES_DEDUPLICATED)

    characters = data.get("characters", [])

    print(f"\n{'Name':<30} | Aliases | Chapters | Relationships | Traits | Mentions | Dialogues | Actions")
    print("-" * 120)

    for char in sorted(characters, key=lambda x: x["name"]):
        name = char["name"]
        alias_count = len(char.get("aliases", []))
        chapter_count = len(char.get("source_chapters", []))
        rel_count = len(char.get("relationships", []))
        trait_count = len(char.get("traits", []))
        total_mentions = char.get("statistics", {}).get("total_mentions", [])
        total_dialogues = char.get("statistics", {}).get("total_dialogues", [])
        total_actions = char.get("statistics", {}).get("total_actions", [])

        print(f"{name:<30} | {alias_count:7} | {chapter_count:8} | {rel_count:13} | {trait_count:6} | {total_mentions:8} | {total_dialogues:9} | {total_actions:8}")


if __name__ == "__main__":
    main()
