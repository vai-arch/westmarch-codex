"""
Generic Westmarch Codex Configuration
"""

from dotenv import load_dotenv


class Config:
    def __init__(self, env_file=".env"):
        """Initialize configuration"""

        load_dotenv(override=True)

        self.LOG = {
            "LEVEL": "INFO",
            "MAX_BYTES": 10485760,
            "BACKUP_COUNT": 5,
        }

        # fmt: off
        self.EXTRACTION = {
            "CHAPTER_MIN_CHAR_COUNT": 1000,
            "CHAPTER_MIN_PARAGRAPH_COUNT": 10,
            "CHAPTER_TITLE_FALLBACK_P_INDEX": 1
        }

        self.LLM_CONFIG = {
            "model": "qwen2.5:32b-instruct-q4_K_M",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 32768,
            "timeout": 3600,
        }

        # fmt: on

        self.ENTITY_TYPES = [
            "characters",
            "locations",
            "races",
            "objects",
            "events",
            "groups",
            "concepts",
            "songs",
        ]

        # Entity schemas with examples
        self.ENTITY_SCHEMAS = {
            "characters": {
                "description": "ALL people, creatures, or beings - MAJOR and MINOR. If a character is named even once, extract them. If multiple characters of the same race are mentioned, extract each one separately.",
                "format": {
                    "name": "canonical full name",
                    "aliases": ["every other way they're referred to"],
                    "race": "Hobbit/Dwarf/Elf/Wizard/Dragon/Goblin/etc or null",
                    "gender": "male/female/unknown",
                    "role": "their profession/role or null",
                    "affiliations": ["groups they belong to or empty list"],
                    "relationships": [{"entity": "name", "relationship": "type"}],
                    "traits": ["characteristics mentioned or empty list"],
                },
                "example": {
                    "name": "Gandalf",
                    "aliases": ["the wizard", "the Grey Pilgrim", "Gandalf the Grey"],
                    "race": "Wizard",
                    "gender": "male",
                    "role": "wizard",
                    "affiliations": ["White Council"],
                    "relationships": [{"entity": "Bilbo Baggins", "relationship": "friend"}],
                    "traits": ["wise", "mysterious", "carries a staff"],
                },
            },
            "locations": {
                "description": "places, regions, buildings, or geographical features",
                "format": {
                    "name": "canonical name",
                    "aliases": ["other names or empty list"],
                    "location_type": "city/mountain/forest/cave/building/region/river/etc",
                    "parent_location": "larger location it's part of or null",
                    "inhabitants": ["who lives there or empty list"],
                    "significance": "why it matters or null",
                },
                "example": {
                    "name": "Bag End",
                    "aliases": ["Bilbo's home", "the hobbit-hole"],
                    "location_type": "building",
                    "parent_location": "Hobbiton",
                    "inhabitants": ["Bilbo Baggins"],
                    "significance": "Bilbo's comfortable home where the adventure begins",
                },
            },
            "races": {
                "description": "species or peoples mentioned in the story. It is ok if you just find the name mentioned without detailed characteristics.",
                "format": {"name": "race name", "characteristics": ["traits mentioned or empty list"], "notable_members": ["key characters of this race or empty list"]},
                "example": {"name": "Hobbits", "characteristics": ["small", "love comfort", "live in holes"], "notable_members": ["Bilbo Baggins"]},
            },
            "objects": {
                "description": "ONLY physical inanimate items like weapons, rings, treasure, tools, clothing, books, maps, keys - absolutely NO people, NO creatures, NO places, NO events, NO conversations. If there are no physical items mentioned, return empty array.",
                "format": {
                    "name": "canonical name",
                    "aliases": ["other names or empty list"],
                    "object_type": "weapon/treasure/tool/magical_item/clothing/map/book/key/etc",
                    "owner": "who possesses it or null",
                    "properties": ["special characteristics or empty list"],
                    "significance": "why it matters or null",
                },
                "example": {
                    "name": "Sting",
                    "aliases": ["the elvish blade", "Bilbo's sword"],
                    "object_type": "weapon",
                    "owner": "Bilbo Baggins",
                    "properties": ["glows blue when orcs are near", "made by elves"],
                    "significance": "Becomes Bilbo's primary weapon",
                },
                "additional_examples": [
                    {
                        "name": "Thorin's Map",
                        "aliases": ["the map", "the treasure map"],
                        "object_type": "map",
                        "owner": "Thorin Oakenshield",
                        "properties": ["shows secret entrance to Lonely Mountain", "has moon-letters"],
                        "significance": "Key to finding the hidden door",
                    }
                ],
            },
            "events": {
                "description": "significant happenings, battles, or plot points",
                "format": {
                    "name": "name of the event",
                    "event_type": "battle/journey/meeting/discovery/etc",
                    "participants": ["who was involved or empty list"],
                    "location": "where it happened or null",
                    "outcome": "what happened or null",
                },
                "example": {
                    "name": "The Unexpected Party",
                    "event_type": "meeting",
                    "participants": ["Bilbo Baggins", "Gandalf", "Thorin Oakenshield", "twelve dwarves"],
                    "location": "Bag End",
                    "outcome": "Bilbo agrees to join the quest",
                },
            },
            "groups": {
                "description": "organizations, companies, factions, families, or teams of multiple members - NOT individual people, NOT dialogue, NOT conversations. Examples: Thorin and Company, The White Council, armies, councils, families",
                "format": {
                    "name": "group name",
                    "group_type": "company/army/council/family/etc",
                    "members": ["who belongs or empty list"],
                    "purpose": "their goal or null",
                    "allegiances": ["who they're allied with or empty list"],
                },
                "example": {
                    "name": "Thorin and Company",
                    "group_type": "company",
                    "members": ["Thorin Oakenshield", "Bilbo Baggins", "Gandalf", "twelve dwarves"],
                    "purpose": "reclaim the Lonely Mountain from Smaug",
                    "allegiances": ["Dwarves of Erebor"],
                },
            },
            "concepts": {
                "description": "abstract ideas, themes, philosophies, or intangible story elements ONLY - NOT people, NOT creatures, NOT places, NOT objects. Examples of valid concepts: fate, prophecy, honor, greed, destiny, dragon-sickness, burglary (as a role/idea), quests (as abstract journeys). Invalid examples: Smaug (this is a character), treasure (this is an object), mountain (this is a location)",
                "format": {"name": "concept name", "description": "what it means in the story", "related_entities": ["who/what it affects or empty list"]},
                "example": {
                    "name": "Dragon-sickness",
                    "description": "A mental affliction causing extreme greed and possessiveness over treasure, leading those affected to make irrational and destructive decisions",
                    "related_entities": ["Thorin Oakenshield", "Smaug", "dwarves"],
                },
            },
            "songs": {
                "description": "songs, poems, or verses performed in the story",
                "format": {"name": "song title or first line", "performer": "who sang it or null", "context": "why it was sung or null", "themes": ["main themes or empty list"]},
                "example": {
                    "name": "Far over the misty mountains cold",
                    "performer": "the dwarves",
                    "context": "sung at Bag End to inspire the quest",
                    "themes": ["longing for home", "adventure", "treasure"],
                },
            },
        }

        # Single dynamic extraction prompt
        self.SINGLE_ENTITY_EXTRACTION_PROMPT = """You are an expert literary analyst extracting entities from "The Hobbit" by J.R.R. Tolkien.

            TASK: Extract ALL {entity_type} from this chapter.

            WHAT ARE {entity_type}?
            {description}

            =============================================================================
            MANDATORY SCHEMA - YOU MUST USE THESE EXACT FIELDS (NO OTHER FIELDS ALLOWED)
            =============================================================================

            {format}

            CRITICAL RULES:
            1. Use ONLY the field names shown above - DO NOT invent new fields
            2. DO NOT use: "description", "importance_to_plot", "location_name", "event_name", "concept_name", "concept_id"
            3. Every entity MUST have ALL fields from the schema above
            4. If a field value is unknown: use null (for text) or [] (for arrays)
            5. Extract ONLY {entity_type} - do not extract entities from other categories
            6. If NO {entity_type} exist in this chapter, return empty array []
            7. "ALL" means EVERY SINGLE mention - even minor characters who appear once
            8. Include background characters, unnamed groups, briefly mentioned entities
            9. Better to over-extract than under-extract - we will filter later
            10. If dwarves are mentioned, extract EACH dwarf individually by name
            {context_section}

            EXAMPLE showing EXACT field names and structure:
            {example}

            VERIFY BEFORE RETURNING:
            - Does each entity have ALL required fields?
            - Are field names EXACTLY as shown in schema?
            - Are you extracting the right entity type (not mixing categories)?

            CHAPTER TEXT:
            {chapter_text}

            =============================================================================
            OUTPUT FORMAT (use EXACT field names from schema)
            =============================================================================

            {{
            "chapter_num": {chapter_num},
            "chapter_title": "{chapter_title}",
            "{entity_type}": [
                // Each entity must have ALL fields from schema above
                // Use exact field names - no variations allowed
            ]
            }}

            Return ONLY valid JSON. No text before or after. No markdown blocks.
            """

    def __repr__(self):
        """String representation showing all config attributes"""
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"Config({attrs})"


# Global instance
_config = None


def get_config():
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
    return _config


def print_config():
    """Print current fine-tuning configuration"""
    config = get_config()

    print(config.__repr__)


def configuration_to_string(configuration_section: dict, indent: int = 0) -> str:
    """
    Recursively convert a nested configuration dict to a multi-line string.
    """
    lines = ["Configuration:"]
    prefix = " " * indent
    for key, value in configuration_section.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(configuration_to_string(value, indent=indent + 4))
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Test configuration
    print_config()
    print(get_config().EXTRACTION)
