# Goal 1: Entity Extraction Baseline

**Overall Goal:** Extract entities from The Hobbit and save structured data with spoiler-safe tracking.

**Success Metrics:**
- Capture ≥90% of named characters (manually verify top 20-30)
- Capture ≥85% of named locations
- Each entity has: name, mention_count, first_appearance (book, chapter), all appearances
- Spoiler-safe: can filter entities by book/chapter threshold
- Runtime: <5 minutes on full book

**Final Deliverables:**
- `data/processed/characters.json`
- `data/processed/locations.json`
- `data/processed/aliases.json`

---

## Subgoal 1.1: Book Preprocessing

**Status:** 🟢 Complete

**Deliverable:** `src/extraction/01_preprocess_book.py`

**Input:** `raw/the_hobbit.epub` (paths.FILE_BOOK_00_RAW)

**Output:** `data/processed/book_00_structured.json`

**Contains:**
- Full text split by chapters
- Each chapter: `{book_num: 0, chapter_num, chapter_title, text, paragraph_count, char_count}`

**Success Metric:**
- 19 chapters extracted (The Hobbit has 19 chapters)
- Each chapter tagged with `book_num: 0`
- Can print all chapter titles
- Verify text integrity (no corrupted sections)

### book_00_processed.json structure

{
    "book_num": 0,
    "chapter_num": 1,
    "chapter_title": "An Unexpected Party",
    "text": [
      "In a hole in the ground ....",
      "It had a perfectly round door ..."
    ],
    "paragraph_count": 167,
    "char_count": 46180
  },

### STATISTICS

+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
|                               | title                               |   paragraph_count |   paragraph_min_chars |   paragraph_max_chars |   paragraph_avg_chars |   char_count |
+===============================+=====================================+===================+=======================+=======================+=======================+==============+
| book_num: 0 - chapter_num: 1  | An Unexpected Party                 |               167 |                    10 |                  2529 |                   269 |        46180 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 2  | Roast Mutton                        |               109 |                    13 |                  2285 |                   247 |        28204 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 3  | A Short Rest                        |                73 |                     7 |                  1633 |                   198 |        15271 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 4  | Over Hill and Under Hill            |                51 |                    15 |                  1844 |                   395 |        22008 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 5  | Riddles in the Dark                 |               170 |                    13 |                  1598 |                   217 |        37089 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 6  | Out of the Frying-Pan into the Fire |               103 |                     7 |                  1746 |                   346 |        35779 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 7  | Queer Lodgings                      |               173 |                     6 |                  3093 |                   267 |        47277 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 8  | Flies and Spiders                   |               129 |                    14 |                  1828 |                   404 |        53886 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 9  | Barrels Out of Bond                 |                77 |                    21 |                  2052 |                   366 |        30607 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 10 | A Warm Welcome                      |                56 |                     5 |                  1486 |                   336 |        20922 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 11 | On the Doorstep                     |                28 |                     6 |                  1614 |                   553 |        15930 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 12 | Inside Information                  |                95 |                     1 |                  1741 |                   395 |        37817 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 13 | Not at Home                         |                64 |                    36 |                  1331 |                   321 |        20921 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 14 | Fire and Water                      |                31 |                    70 |                  2312 |                   566 |        17305 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 15 | The Gathering of the Clouds         |                78 |                    30 |                  1394 |                   223 |        17784 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 16 | A Thief in the Night                |                46 |                    28 |                  1199 |                   240 |        11138 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 17 | The Clouds Burst                    |                54 |                    11 |                  1378 |                   388 |        21196 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 18 | The Return Journey                  |                44 |                    20 |                  1916 |                   344 |        15000 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
| book_num: 0 - chapter_num: 19 | The Last Stage                      |                91 |                     7 |                  1299 |                   139 |        13209 |
+-------------------------------+-------------------------------------+-------------------+-----------------------+-----------------------+-----------------------+--------------+
---

## Subgoal 1.2: Raw Entity Extraction (NLP-based)

**Status:** 🟢 Complete

**Deliverable:** `02_extract_entities_nlp.py`

**Input:** `data/processed/book_structured.json`

**Output:** `data/processed/entities_raw.json`

**Method:**
- spaCy NER on full text
- Extract: PERSON, GPE, LOC, FAC tags

**Success Metric:**
- Extract all entity mentions (expect ~200-500 raw mentions with duplicates)
- Each entity has:

```json
  {
    "text": "Bilbo",
    "entity_type": "PERSON",
    "mentions": [
      {"book": 0, "chapter": 1, "sentence_context": "..."},
      {"book": 0, "chapter": 2, "sentence_context": "..."}
    ]
  }
```

- No crashes, processes all 19 chapters

### Statistics

+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
|                               |   characters |   locations |   races |   objects |   events |   groups |   concepts |
+===============================+==============+=============+=========+===========+==========+==========+============+
| book_num: 0 - chapter_num: 1  |           15 |           6 |       2 |         7 |        1 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 2  |           17 |           6 |       2 |         7 |        1 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 3  |            4 |           5 |       3 |         3 |        1 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 4  |           14 |          10 |       3 |        10 |        2 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 5  |            3 |           7 |       2 |         7 |        2 |        0 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 6  |           22 |           4 |       3 |         5 |        3 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 7  |           15 |           6 |       2 |         1 |        1 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 8  |           13 |           4 |       3 |         6 |        2 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 9  |           15 |           9 |       3 |         5 |        1 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 10 |           15 |           8 |       2 |         3 |        2 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 11 |           10 |           8 |       2 |         4 |        1 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 12 |            9 |           4 |       2 |         4 |        1 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 13 |           15 |           3 |       1 |         7 |        1 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 14 |            7 |           4 |       1 |         2 |        1 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 15 |           14 |           3 |       1 |         3 |        2 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 16 |            7 |           3 |       3 |         3 |        0 |        1 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 17 |           17 |           3 |       4 |         3 |        1 |        2 |          2 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 18 |           13 |           5 |       2 |         2 |        2 |        1 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| book_num: 0 - chapter_num: 19 |            4 |           6 |       2 |         5 |        1 |        2 |          1 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
| TOTAL                         |          229 |         104 |      43 |        87 |       26 |       20 |         29 |
+-------------------------------+--------------+-------------+---------+-----------+----------+----------+------------+
Total Duration: 4h 29m 54.3s
Configuration:
model: qwen2.5:32b-instruct-q4_K_M
base_url: <http://localhost:11434>
temperature: 0.1
max_tokens: 32768
timeout: 1800

---

## Subgoal 1.3: Entity Consolidation & Alias Detection

**Status:** 🟡 In Progress  

**Deliverable:** `03_consolidate_entities.py`

**Input:** `data/processed/entities_raw.json`

**Output:**
- `data/processed/characters.json`
- `data/processed/locations.json`
- `data/processed/aliases.json`

**Method:**
1. Group similar entities (fuzzy string matching)
2. Use co-occurrence patterns (entities appearing together)
3. LLM refinement (Ollama): "Are these the same entity?"
4. Build canonical name → aliases mapping with first appearance tracking

**Success Metric:**
- Characters: 20-40 unique (consolidated from 100+ raw mentions)
- Locations: 15-30 unique
- Each canonical entity has:

```json
  {
    "canonical_name": "Bilbo Baggins",
    "entity_type": "character",
    "first_appearance": {"book": 0, "chapter": 1},
    "appearances": [
      {"book": 0, "chapter": 1},
      {"book": 0, "chapter": 2},
      ...
    ],
    "mention_count": 847,
    "aliases": [
      {
        "alias": "Bilbo",
        "first_appearance": {"book": 0, "chapter": 1}
      },
      {
        "alias": "Mr. Baggins",
        "first_appearance": {"book": 0, "chapter": 1}
      },
      {
        "alias": "the burglar",
        "first_appearance": {"book": 0, "chapter": 2}
      }
    ]
  }
```

- Aliases file maps all variations to canonical names with spoiler tracking:

```json
  {
    "Bilbo": {
      "canonical": "Bilbo Baggins",
      "first_appearance": {"book": 0, "chapter": 1}
    },
    "the burglar": {
      "canonical": "Bilbo Baggins",
      "first_appearance": {"book": 0, "chapter": 2}
    }
  }
```

**Validation:**

```bash
python 03_consolidate_entities.py
# Should output: "Consolidated X entities into Y characters, Z locations"
# Should show spoiler-safe tracking enabled
```

Goal 1.4: Semantic chunking with entity metadata

Chunk narrative (skip songs)
Tag chunks with entities present

Goal 1.5: Dense embeddings + vector store

Embed chunks
Build ChromaDB collection

Goal 1.6: Simple retrieval + test queries

Basic semantic search
Test with 10-20 queries
Identify gaps

## Subgoal 1.4: Entity Enrichment & Knowledge Profiles

**Status:** 🔴 Not Started

**Deliverable:** `05_enrich_entities.py`

**Input:**
- `data/processed/entities_deduplicated.json`
- `data/processed/entity_mentions_index.json`
- Full chapter text corpus (processed)

**Output:**
- `data/processed/entity_profiles/`
  - `characters.json`
  - `locations.json`
  - `races.json`
  - `objects.json`
  - `groups.json`
  - `concepts.json`
- Optional: `data/processed/entity_knowledge_graph.json`

**Description:**  
After entity extraction, deduplication, and validation, perform an enrichment pass to build **wiki-style knowledge profiles** for each entity.  
For each entity, gather all textual mentions across the book and use an LLM to generate a consolidated, canonical description capturing its defining traits, context, and narrative significance.

This creates a structured **entity knowledge base** separate from raw narrative chunks.

**Enrichment Strategy:**
- Aggregate all paragraphs mentioning a given entity (canonical name + aliases)
- Prompt the LLM to synthesize:
  - Description / definition
  - Characteristics and attributes
  - Role in the story
  - Relationships to other entities
  - Notable appearances (book/chapter range)
- Output is factual, spoiler-aware, and grounded only in provided text

**Success Metric:**
- Each major entity (e.g., *Hobbits*, *Bilbo Baggins*, *The Shire*) has a single, rich profile
- Profiles are consistent, non-duplicated, and alias-aware
- Entity profiles can answer factual queries without needing raw chapter text
- Profiles improve RAG retrieval quality for:
  - “Tell me about Hobbits”
  - “Who is Thorin Oakenshield?”
  - “What is the significance of the Lonely Mountain?”
- Manual spot-check confirms summaries are accurate, comprehensive, and spoiler-aware

**Why This Matters:**
- Enables **two-tier RAG**:
  1. Retrieve entity profiles for factual grounding  
  2. Retrieve narrative chunks for story context
- Reduces hallucination by anchoring LLM responses to curated knowledge
- Establishes foundation for a knowledge graph and advanced reasoning

---

## Subgoal 1.5: Validation & Stats

**Status:** 🔴 Not Started

**Deliverable:** `04_validate_entities.py`

**Input:**
- `data/processed/characters.json`
- `data/processed/locations.json`
- `data/processed/aliases.json`

**Output:**
- Console report
- `data/processed/entity_stats.txt`

**Success Metric:**
- Top 10 characters by mention count includes: Bilbo, Gandalf, Thorin, Gollum, Smaug, Balin
- Top 10 locations includes: Shire, Rivendell, Lonely Mountain, Mirkwood, Misty Mountains
- Each entity shows first appearance (book, chapter)
- Can filter entities: "Show only entities appearing up to Book 0, Chapter 5"
- Alias mapping works with spoiler awareness ("the burglar" only shows after Chapter 2)
- Manual spot-check confirms all major entities captured

**Validation:**

```bash
python 04_validate_entities.py
# Should output formatted report with:
# - Top entities by mention count
# - First appearance tracking
# - Spoiler-safe filtering demonstration
# - Alias coverage stats
```

---

## Spoiler-Safety Requirements

All entity data must support:
1. **Filtering by progression**: `get_entities(up_to_book=0, up_to_chapter=5)`
2. **Alias appearance tracking**: Don't reveal "the burglar" = Bilbo until Chapter 2
3. **First mention tracking**: Know when each entity/alias first appears
4. **Extensibility**: Structure supports adding LOTR books (1, 2, 3) later

---

## Progress Tracking

| Subgoal | Status | Completion Date | Notes |
| --------- | -------- | --------------- | ------- |
| 1.1 Book Preprocessing | 🔴 Not Started | - | - |
| 1.2 Raw Entity Extraction | 🔴 Not Started | - | - |
| 1.3 Entity Consolidation | 🔴 Not Started | - | - |
| 1.4 Validation & Stats | 🔴 Not Started | - | - |

**Status Legend:**
- 🔴 Not Started
- 🟡 In Progress
- 🟢 Complete
- ⚠️ Blocked/Issues

LEARN ABOUT:

FAISS vs Qdrant config

Hybrid BM25 + E5 setup

How to embed aliases properly

If you also embed entity wiki pages

Use same model but longer passages broken into:

Character profile chunks

Location profile chunks

Event profile chunks

Still 300–500 tokens each.
