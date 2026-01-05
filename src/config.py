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
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434",  # Ollama default
            "temperature": 0.1,  # Low for consistent extraction
            "max_tokens": 8192,   # Max output tokens
            "timeout": 300,       # 5 minutes timeout per request
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

        self.ENTITIES_EXTRACTION = {
            "EXTRACTION_PROMPT": """You are an expert literary analyst extracting entities from "The Hobbit" by J.R.R. Tolkien.
                Extract ALL entities mentioned in this chapter and return them in STRICT JSON format.

                ENTITY TYPES TO EXTRACT:

                1. CHARACTERS:
                {{
                    "name": "canonical name",
                    "aliases": ["other names used"],
                    "race": "Hobbit/Dwarf/Elf/Wizard/Dragon/etc",
                    "gender": "male/female/unknown",
                    "role": "profession or role",
                    "affiliations": ["groups they belong to"],
                    "relationships": [{{"entity": "name", "relationship": "type"}}],
                    "traits": ["key characteristics mentioned"]
                }}

                2. LOCATIONS:
                {{
                    "name": "canonical name",
                    "aliases": ["alternate names"],
                    "location_type": "city/mountain/forest/cave/building/region/river/etc",
                    "parent_location": "larger location it's part of",
                    "inhabitants": ["who lives there"],
                    "significance": "why it matters"
                }}

                3. RACES:
                {{
                    "name": "race name",
                    "characteristics": ["traits from text"],
                    "notable_members": ["key characters of this race"]
                }}

                4. OBJECTS:
                {{
                    "name": "canonical name",
                    "aliases": ["other names"],
                    "object_type": "weapon/treasure/tool/magical_item/clothing/etc",
                    "owner": "who possesses it",
                    "properties": ["special characteristics"],
                    "significance": "why it matters"
                }}

                5. EVENTS:
                {{
                    "event_name": "name of event",
                    "event_type": "battle/journey/meeting/discovery/etc",
                    "participants": ["who was involved"],
                    "location": "where it happened",
                    "outcome": "what happened"
                }}

                6. GROUPS:
                {{
                    "name": "group name",
                    "group_type": "company/army/council/family/etc",
                    "members": ["who belongs"],
                    "purpose": "their goal",
                    "allegiances": ["who they're allied with"]
                }}

                7. CONCEPTS:
                {{
                    "name": "concept name",
                    "description": "what it means",
                    "related_entities": ["who/what it affects"]
                }}

                8. SONGS:
                {{
                    "title": "song title or first line",
                    "performer": "who sang it",
                    "context": "why it was sung",
                    "themes": ["main themes"]
                }}

                INSTRUCTIONS:
                - Extract ONLY entities explicitly mentioned in this chapter
                - Use exact names as they appear in the text
                - For quotes, keep under 200 characters
                - If information is unknown, use null
                - Be comprehensive but accurate
                - CRITICAL: Return ONLY the JSON object. Start with {{ and end with }}. No text before or after. No markdown code blocks.

                CHAPTER TEXT:
                {chapter_text}

                Return JSON in this exact format:
                {{
                    "chapter_num": {chapter_num},
                    "chapter_title": "{chapter_title}",
                    "characters": [...],
                    "locations": [...],
                    "races": [...],
                    "objects": [...],
                    "events": [...],
                    "groups": [...],
                    "concepts": [...],
                    "songs": [...]
                }}
                """
        }

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
