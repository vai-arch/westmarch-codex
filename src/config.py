"""
Generic Westmarch Codex Configuration
"""

from enum import Enum

from dotenv import load_dotenv


class EmbeddingManagers(str, Enum):
    OLLAMA = "ollama"
    SENTENCE_TRANSFORMER = "sentence_transformer"


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

        # High-quality model for complex reasoning tasks
        self.REASONING_MODEL_CONFIG = {
            "model": "qwen2.5:32b-instruct-q4_K_M",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 32768,
            "max_prompt_length": 15000,
            "timeout": 3000,
        }

        # Fast model for structured tasks (validation, formatting, simple extraction)
        self.FAST_MODEL_CONFIG = {
            "model": "qwen2.5:7b-instruct",
            "base_url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 32768,
            "max_prompt_length": 15000,
            "timeout": 3000,  # Shorter timeout
        }

        # Keep backward compatibility
        self.LLM_CONFIG = self.REASONING_MODEL_CONFIG  # Default to reasoning model

        self.ENTITY_EXTRACTION_CONFIG = {
            "context_window_chars": 10000,
        }

        self.ENTITY_TYPES = [
            "characters",
            "locations",
            "objects",
            "events",
            "groups",
            "concepts"
        ]

        self.ACCUMULATED_ENTITY_TYPES = [
            "characters",
            "locations",
            "objects",
            "groups",
            "concepts"
        ]
        # fmt: on
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
        }

        self.UNIFIED_EXTRACTION_PROMPT = """You are an expert literary analyst extracting entities from "The Hobbit" by J.R.R. Tolkien.

            TASK: Extract ALL entities from this chapter in a SINGLE pass.

            ENTITY TYPES TO EXTRACT:
            1. character - people, creatures, beings (Bilbo, Gandalf, Smaug, any dwarf, goblins, etc.)
            2. location - places, regions, buildings (Bag End, Shire, Lonely Mountain, rivers, forests, etc.)
            3. object - physical items ONLY (swords, rings, treasure, maps, clothing, tools - NOT people or places)
            4. event - significant happenings (The Unexpected Party, battles, discoveries, meetings)
            5. group - organizations, companies (Thorin and Company, White Council, armies)
            6. concept - abstract ideas ONLY (dragon-sickness, fate, prophecy, honor - NOT people, places, or objects)

            CRITICAL RULES:
            1. Extract EVERYTHING - major and minor entities
            2. For each entity provide: entity_type, name, description, aliases (if any)
            3. Be thorough - if 13 dwarves are mentioned, extract all 13 individually
            4. Concepts = abstract ideas ONLY (NOT characters like "Smaug" or places like "Mountain")
            5. Objects = inanimate items ONLY (NOT people or creatures)
            {context_section}

            CHAPTER TEXT:
            {chapter_text}

            Return ONLY valid JSON in this format:
            {{
            "chapter_num": {chapter_num},
            "chapter_title": "{chapter_title}",
            "entities": [
                {{
                "entity_type": "character",
                "name": "Bilbo Baggins",
                "description": "A hobbit chosen as burglar for the quest",
                "aliases": ["Mr. Baggins", "the hobbit"]
                }},
                {{
                "entity_type": "location",
                "name": "Bag End",
                "description": "Bilbo's comfortable hobbit-hole",
                "aliases": ["Bilbo's home"]
                }}
                // ... all other entities
            ]
            }}
            """
        # Single dynamic extraction prompt
        self.VALIDATION_PROMPT = """You are validating and enriching extracted entities from "The Hobbit".

            TASK: Take the raw entity extraction and produce properly structured output with full schemas.

            RAW EXTRACTED ENTITIES:
            {raw_entities}

            YOUR JOBS:
            1. VALIDATE entity_type - move misclassified entities to correct category
            - Remove characters/locations wrongly classified as "concepts"
            - Remove characters wrongly classified as "objects"
            - Concepts must be abstract ideas only (dragon-sickness, fate, honor)
            
            2. REMOVE exact duplicates (same entity listed multiple times)

            3. ENRICH each entity with proper fields based on its type:

            REQUIRED SCHEMAS BY TYPE:

            CHARACTERS:
            {characters_schema}

            LOCATIONS:
            {locations_schema}

            OBJECTS:
            {objects_schema}

            EVENTS:
            {events_schema}

            GROUPS:
            {groups_schema}

            CONCEPTS:
            {concepts_schema}

            CRITICAL:
            - Use EXACT field names from schemas (e.g., "name" not "character_name" or "location_name")
            - Fill all required fields - use null or [] if unknown
            - Maintain aliases from raw extraction

            Return properly structured entities:
            {{
            "chapter_num": {chapter_num},
            "chapter_title": "{chapter_title}",
            "characters": [...],
            "locations": [...],
            "objects": [...],
            "events": [...],
            "groups": [...],
            "concepts": [...],
            }}
            """

        self.CHARACTER_STATS_PROMPT = """Analyze this chapter for mentions of the following character.

            CHARACTER TO TRACK:
            Name: {character_name}
            Known aliases: {aliases}

            CHAPTER TEXT:
            {chapter_text}

            TASK: Count occurrences for the specific character (search for character name AND all aliases):

            1. MENTIONS: Total times the character is referenced by name or alias
            2. DIALOGUES: Times the character speaks (has dialogue attributed to them)
            3. ACTIONS: Times the character performs significant actions (verbs with character as subject)
            4. JUSTIFICATION your counts briefly.
            5. ALIASES not already listed that were found in the text.

            CRITICAL: Never guess - only count what is explicitly in the text. Always justify your count. If the character doesnt appear, return zeros. Never count a different character. Only add new aliases, not already know ones.

            Return ONLY this JSON (no explanation):
            {{
            "chapter_num": {chapter_num},
            "mentions": <number>,
            "dialogues": <number>,
            "actions": <number>,
            "justification": "brief explanation of counts",
            "new_aliases": [list of any NEW aliases found]
            }}
            """

        # === ENTITY DEDUPLICATION CONFIG ===
        self.DEDUPLICATION_CONFIG = {
            # Fuzzy matching threshold for name similarity (0-100, higher = stricter)
            "fuzzy_threshold": 85,
            # Similarity threshold for semantic comparison (0.0-1.0)
            "semantic_threshold": 0.85,
            # Strategy: "rule_based", "llm", or "hybrid"
            "strategy": "rule_based",  # Start with rule_based, can upgrade to hybrid later
            # For hybrid: how many candidates to send to LLM for final decision
            "llm_review_threshold": 3,
            # Enable within-chapter deduplication (for entities extracted multiple times in same chapter)
            "within_chapter_dedup": True,
            # Canonical name selection: "longest", "most_common", or "llm_choice"
            "canonical_name_strategy": "longest",
            # Batch size for LLM processing (if using LLM)
            "llm_batch_size": 10,
        }

        # TODO: REMOVE THIS AFTER TESTING
        self.DEDUPLICATION_CLEANUP_PROMPT = {
            "characters": """Clean CHARACTERS from The Hobbit.

        DECISION TEST for each entity - Apply in order:

        TEST 1: Is this a SPECIES/RACE name (not an individual)?
        Ask: "Can multiple beings share this name?" 
        - YES → REMOVE (it's a race): "Goblins", "Wargs", "Wood-elves", "Hobbits" (plural/collective)
        - NO → Keep (it's an individual): "Bilbo", "Gandalf", "Gollum"

        TEST 2: Is this a DUPLICATE of another character?
        Check: Do two names refer to the SAME individual person?
        - "Bard" + "Bard the Bowman" → YES, same person → MERGE (keep longer name)
        - "Dori" + "Nori" → NO, different people → Keep separate

        TEST 3: Does it have a proper name?
        - Has name → KEEP
        - Generic descriptor only ("the Master") → Check if already captured elsewhere

        MERGE RULES:
        - Longer name becomes primary: "Bard the Bowman" (alias: "Bard")
        - Combine all: aliases, source_chapters, relationships, traits

        Current characters: {entities}
        Return: {{"characters": [...]}}""",
            "locations": """Clean LOCATIONS from The Hobbit.

        DECISION TEST for each location - Apply in order:

        TEST 1: CAPITALIZATION CHECK
        - Starts with capital letter → Probably named → KEEP
        - All lowercase → Probably generic → Go to TEST 2

        TEST 2: PROPER NAME TEST (for lowercase entries)
        Count how many of this type could exist:
        - "river" → Could be 1000+ rivers → REMOVE
        - "River Running" → One specific river → KEEP
        - "tunnel" → Could be any tunnel → REMOVE  
        - "secret passage" → One specific passage → Keep if appears 3+ chapters, else REMOVE

        Rule: lowercase + appears in 1-2 chapters only → REMOVE

        TEST 3: DUPLICATE CHECK
        Exact matches or one name contains another:
        - "Running River" vs "River Running" → Same? Check source_chapters overlap
        - Merge if 80%+ chapter overlap

        SPECIAL CASES:
        - "The [Name]" variations: "The Hill" is a proper name → KEEP
        - Compound locations: "king's cellars" = specific place → KEEP

        Current locations: {entities}
        Return: {{"locations": [...]}}""",
            "objects": """Clean OBJECTS from The Hobbit.

        DECISION TEST for each object - Apply in order:

        TEST 1: BUNDLE DETECTION
        Check aliases - are they SAME object or DIFFERENT objects?
        Method: Could they exist simultaneously in different places?
        - "ring" | aliases: ["precious", "birthday-present"] → SAME object (different names) → Keep together
        - "Weapons" | aliases: ["Glamdring", "Orcrist", "Sting"] → DIFFERENT objects (3 swords) → SPLIT into 3

        SPLIT RULE: If 3+ aliases that are clearly separate physical items → Create separate entities

        TEST 2: DUPLICATE DETECTION  
        Exact name match or longer name contains shorter:
        - "Arkenstone" + "Arkenstone of Thrain" → Check aliases overlap
        - If aliases match 50%+ → MERGE (keep longer name)

        TEST 3: GENERIC REMOVAL
        Is this a category name or specific named item?
        - "sword" → generic category → REMOVE
        - "Sting" → specific named sword → KEEP  
        - "arrow" → generic → REMOVE
        - "black arrow" → specific important arrow → KEEP

        Test: Would Tolkien use a capital letter for this in prose?
        - No capital → likely generic → REMOVE

        Current objects: {entities}
        Return: {{"objects": [...]}}""",
            "events": """Clean EVENTS from The Hobbit.

        DECISION TEST for each event:

        TEST 1: DUPLICATE EVENT CHECK
        Compare events pairwise:
        - Same participants? (50%+ overlap)
        - Same location?
        - Same chapter range? (overlap 2+ chapters)

        If YES to all 3 → Same event, different description → MERGE
        Example: "Death of Smaug" + "Fall of Smaug" → MERGE (choose more descriptive)

        TEST 2: VAGUE vs SPECIFIC
        Events need clear identity. Must have 2+ of:
        - Specific participants (not "they")
        - Specific location  
        - Specific outcome
        - Appears in 2+ chapters

        If too vague → REMOVE

        MERGE RULE: Keep most descriptive/complete name

        Current events: {entities}
        Return: {{"events": [...]}}""",
            "groups": """Clean GROUPS from The Hobbit.

        DECISION TEST:

        TEST 1: DUPLICATE CHECK
        Are these the same group?
        - "elves" vs "Elves" → Same group, different capitalization → MERGE (capitalize)
        - "Lake-men" variants → Merge under most common form

        TEST 2: PROPER GROUP NAME
        - Has specific name/identifier → KEEP
        - Too generic ("people", "folk") → REMOVE

        Current groups: {entities}
        Return: {{"groups": [...]}}""",
            "concepts": """Clean CONCEPTS from The Hobbit.

        DECISION TEST:

        TEST 1: PHILOSOPHICAL vs CONCRETE
        Concepts should be ABSTRACT ideas, not physical things
        - "dragon-sickness" → abstract affliction → KEEP
        - "treasure" → physical thing → REMOVE (unless clearly abstract greed concept)

        TEST 2: DUPLICATE CHECK  
        - "dragon-sickness" vs "Dragon-sickness" → MERGE (consistent capitalization)

        Current concepts: {entities}
        Return: {{"concepts": [...]}}""",
        }

        # TODO: REMOVE THIS AFTER TESTING
        # Models for LLM deduplication (if strategy includes LLM)
        self.DEDUPLICATION_LLM_CONFIG = self.REASONING_MODEL_CONFIG  # Reuse reasoning model

        self.EMBEDDING_MODEL_NOMIC_2048 = {
            "EMBEDDING_MODEL_NAME": "nomic-embed-text-num_batch-2048:latest",
            "EMBEDDING_MODEL_MAX_TOKENS": 2046,
            "EMBEDDING_MODEL_DIMENSION": 768,
            "EMBEDDING_MODEL_RAW_PREFIX": None,
            "EMBEDDING_MODEL_SEARCH_PREFIX": "search_query",
            "EMBEDDING_MODEL_DOCUMENT_PREFIX": "search_document",
        }

        self.EMBEDDING_MODEL_NOMIC_1_5 = {
            "EMBEDDING_MODEL_NAME": "nomic-ai/nomic-embed-text-v1.5",
            "EMBEDDING_MODEL_MAX_TOKENS": 8000,
            "EMBEDDING_MODEL_DIMENSION": 768,
            "EMBEDDING_MODEL_RAW_PREFIX": None,
            "EMBEDDING_MODEL_SEARCH_PREFIX": "",
            "EMBEDDING_MODEL_DOCUMENT_PREFIX": "",
        }

        self.EMBEDDING_MODEL_INTFLOAT_E5_LARGE_V2 = {
            "EMBEDDING_MODEL_NAME": "intfloat/e5-large-v2",
            "EMBEDDING_MODEL_MAX_TOKENS": 512,
            "EMBEDDING_MODEL_DIMENSION": 1024,
            "EMBEDDING_MODEL_RAW_PREFIX": "passage: ",
            "EMBEDDING_MODEL_SEARCH_PREFIX": "query: ",
            "EMBEDDING_MODEL_DOCUMENT_PREFIX": "passage: ",
        }

        self.EMBEDDING_MODEL = self.EMBEDDING_MODEL_INTFLOAT_E5_LARGE_V2
        self.EMBEDDING_MANAGER = EmbeddingManagers.SENTENCE_TRANSFORMER.value

        self.EMBEDDING_SENTENCE_TRANSFORMER_CONFIG = {
            "EMBEDDING_METHOD": "BATCH",  # ONE_BY_ONE, BATCH, BATCH_IN_PARALLEL
            "EMBEDDING_BATCH_SIZE": 32,
            "EMBEDDING_MODEL": self.EMBEDDING_MODEL,
        }

        self.CHUNKING_STRATEGY = {
            "BOOKS_CHUNKING_STRATEGY_NAME": "semantic",
            "SEMANTIC_MAX_CHUNK_TOKENS": 1000,
            "SEMANTIC_MIN_CHUNK_TOKENS": 300,
            "SEMANTIC_OVERLAP_TOKENS": 0,
            "SEMANTIC_SIMILARITY_THRESHOLD": 0.82,  # cosine similarity breakpoint
            "MIN_BOOKS_CHUNKS_SIZE_CHARACTERS": 300,
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


def get_embedding_manager_config(embedding_manager: str):
    config = get_config()

    if embedding_manager == EmbeddingManagers.OLLAMA.value:
        return config.OLLAMA_CONFIG
    elif embedding_manager == EmbeddingManagers.SENTENCE_TRANSFORMER.value:
        return config.EMBEDDING_SENTENCE_TRANSFORMER_CONFIG
    else:
        raise ValueError(f"Unknown embedding backend: {embedding_manager}")


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
