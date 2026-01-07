import json
from collections import defaultdict

with open("data/processed/entities_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Find entities that appear across chapters
entity_occurrences = defaultdict(list)

for chapter in data:
    chapter_num = chapter["chapter_num"]
    for entity_type in ["characters", "locations", "races", "objects", "events", "groups", "concepts", "songs"]:
        if entity_type in chapter:
            for entity in chapter[entity_type]:
                name = entity.get("name")
                if name:  # Skip entities with None/missing names
                    key = (entity_type, name)
                    entity_occurrences[key].append({"chapter": chapter_num, "aliases": entity.get("aliases", []), "full_entity": entity})

# Show entities that appear in 3+ chapters
print("Entities appearing in 3+ chapters:\n")
multi_chapter = [(k, v) for k, v in entity_occurrences.items() if len(v) >= 3]
multi_chapter.sort(key=lambda x: (x[0][0], x[0][1]))

for (entity_type, name), occurrences in multi_chapter:
    print(f"{entity_type.upper()}: {name}")
    print(f"  Chapters: {[o['chapter'] for o in occurrences]}")
    all_aliases = set()
    for o in occurrences:
        all_aliases.update(o["aliases"])
    print(f"  Aliases: {sorted(all_aliases)}")
    print()
