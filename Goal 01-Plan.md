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

**Status:** DONE !!!

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

**Status:** 🔴 Not Started

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

**Validation:**

```bash
python 02_extract_entities_nlp.py
# Should output: "Extracted X PERSON, Y LOC, Z GPE entities across Book 0"
```

---

## Subgoal 1.3: Entity Consolidation & Alias Detection

**Status:** 🔴 Not Started

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

---

## Subgoal 1.4: Validation & Stats

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

---

## Next Steps

1. Create project chat for Subgoal 1.1
2. Obtain The Hobbit source file (.txt or .epub)
3. Set up folder structure: `data/raw/` and `data/processed/`
4. Begin implementation of `01_preprocess_book.py`
