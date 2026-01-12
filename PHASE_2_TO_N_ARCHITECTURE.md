# The Westmarch Codex: Complete Architecture Roadmap

## From Basic RAG to Literary-Grade Question Answering

---

## Current State: Phase 1 (Foundation)

**What You're Building:**
- Semantic chunking (800-1000 tokens, similarity-based boundaries)
- Entity extraction (characters, locations, objects, events, groups)
- Hybrid retrieval (dense embeddings + BM25)
- Basic evaluation framework

**What Phase 1 Delivers:**
- ✓ Functional RAG system
- ✓ Entity metadata on chunks
- ✓ Baseline retrieval performance
- ✓ Test harness for measuring improvements

**Expected Limitations:**
- Misses context when answer spans multiple chunks
- Cannot track character/entity evolution over time
- Struggles with "why" questions requiring causal reasoning
- No understanding of narrative structure (chapters → scenes → events)
- Cannot handle queries about relationships between entities
- Weak on thematic or emotional queries

---

## Phase 2: Multi-Granularity Retrieval

**Goal:** Fix "missing context" problem

### Problem Phase 1 Can't Solve

Query: *"Describe the entire Battle of Five Armies"*
- Phase 1 returns: 5-10 disconnected chunks
- Missing: Complete narrative arc, sequence of events, outcome

### Solution: Hierarchical Chunk Architecture

**2.1 Add Scene-Level Chunks**
- Scene = continuous narrative unit (same time, place, characters)
- Detection strategy:
  - Chapter boundaries (explicit)
  - Hard breaks (---, ***, blank lines)
  - Paragraph clustering by semantic similarity
  - Time/location shifts detected via entities
- Target: 1500-3000 tokens per scene
- Metadata: `scene_id`, `characters_present`, `location`, `time_marker`

**2.2 Add Chapter Summaries**
- One chunk = entire chapter (5k-10k tokens)
- Used for: "What happened in Chapter X?", "Why did Y occur?"
- Generated: Automatic summarization via LLM (Ollama)

**2.3 Retrieval Strategy**

```
Query → Classify intent:
  - Factual detail → Micro chunks (Phase 1)
  - Narrative sequence → Scene chunks
  - Causality/motivation → Chapter chunks
  
Retrieve from appropriate layer(s):
  - Top 5 micro chunks
  - Top 3 scene chunks
  - Top 1 chapter chunk
  
Assemble in narrative order
```

**Success Metric:**
- Queries requiring 3+ chunk context show 40%+ improvement in answer quality

**Estimated Effort:** 2-3 weeks
- Scene detection algorithm
- Chapter summarization pipeline
- Multi-index retrieval logic
- Evaluation harness extension

---

## Phase 3: Entity-Centric Knowledge Graph

**Goal:** Track entities across time and relationships

### Problem Phase 2 Can't Solve

Query: *"How does Bilbo's relationship with the dwarves change throughout the story?"*
- Phase 2 returns: Scattered scenes with Bilbo and dwarves
- Missing: Temporal evolution, relationship arc, sentiment progression

### Solution: Entity Graph with Temporal Edges

**3.1 Entity Profile Construction**
For each major entity (character, location, group):
- **Static properties:** Name, aliases, type, physical description
- **Temporal attributes:** First/last appearance, active chapters
- **Relationships:** Other entities, relationship type, sentiment
- **Evolution markers:** Key events that changed the entity

**3.2 Relationship Extraction**
- Entity co-occurrence in chunks → relationship candidate
- LLM-based relationship classification:
  - Type: ally, enemy, neutral, family, mentor, etc.
  - Sentiment: positive, negative, complex
  - Strength: weak, moderate, strong
- Temporal tracking: relationship changes over chapters

**3.3 Entity Timeline Construction**
For each character:

```json
{
  "character": "Bilbo Baggins",
  "arc": [
    {"chapter": 1, "state": "comfortable, risk-averse", "location": "Bag End"},
    {"chapter": 5, "state": "desperate, clever", "event": "Riddle game with Gollum"},
    {"chapter": 12, "state": "confident, heroic", "event": "Enters Smaug's lair alone"},
    {"chapter": 19, "state": "changed, worldly", "location": "Returns to Bag End"}
  ]
}
```

**3.4 Graph-Augmented Retrieval**

```
Query mentions entity → 
  1. Retrieve relevant chunks (Phase 2)
  2. Load entity profile + relationships
  3. Add temporal context markers
  4. Include relationship evolution if query implies time
```

**Success Metric:**
- Entity-focused queries (relationships, change, evolution) show 50%+ improvement
- Can answer: "Who does X trust by the end?" "How does Y feel about Z?"

**Estimated Effort:** 3-4 weeks
- Entity profile schema design
- Relationship extraction pipeline
- Graph storage (NetworkX or simple JSON)
- Graph-enhanced retrieval integration

---

## Phase 4: Query Understanding & Routing

**Goal:** Intelligent query classification and retrieval strategy selection

### Problem Phase 3 Can't Solve

Different queries need different retrieval strategies, but system treats all queries the same.

### Solution: Query Classifier + Dynamic Retrieval Strategy

**4.1 Query Type Taxonomy**

```
Factual:
  - Who/What/When/Where → Micro chunks + entities
  - "What is Bilbo's weapon?" → Sting entity + relevant chunks

Narrative:
  - How did X happen? → Scene + chapter chunks
  - "Describe the troll encounter" → Scene reconstruction

Analytical:
  - Why did X do Y? → Chapter chunks + entity graph + causality
  - "Why did Thorin distrust Bilbo initially?" → Multi-source reasoning

Comparative:
  - How does X compare to Y? → Entity profiles + relevant scenes
  - "Compare Bilbo's courage in Ch1 vs Ch12" → Temporal entity tracking

Thematic:
  - Themes, motifs, patterns → Cross-chapter semantic search + summaries
  - "What role does greed play?" → Thematic entity/event clustering
```

**4.2 Query Classifier**
- Lightweight LLM call (or local model)
- Input: Query
- Output: Type, entities mentioned, temporal scope, expected granularity

**4.3 Dynamic Retrieval Strategy**

```python
def route_query(query, classification):
    if classification.type == "factual":
        return retrieve_micro(query, top_k=10)
    
    elif classification.type == "narrative":
        scenes = retrieve_scenes(query, top_k=5)
        return assemble_narrative_sequence(scenes)
    
    elif classification.type == "analytical":
        chapters = retrieve_chapters(query, top_k=2)
        entities = load_entity_profiles(classification.entities)
        return merge_analytical_context(chapters, entities)
    
    elif classification.type == "comparative":
        entity_timelines = load_temporal_profiles(classification.entities)
        relevant_chunks = retrieve_hybrid(query, top_k=8)
        return build_comparison_context(entity_timelines, relevant_chunks)
    
    elif classification.type == "thematic":
        chapters = retrieve_chapters(query, top_k=5)
        cross_chapter = semantic_clustering(query, all_chunks)
        return aggregate_thematic_evidence(chapters, cross_chapter)
```

**Success Metric:**
- Query routing accuracy > 85%
- Each query type shows appropriate retrieval pattern in logs
- Overall answer quality improves 25%+ across diverse query types

**Estimated Effort:** 2 weeks
- Query classifier (prompt engineering or fine-tuned model)
- Routing logic implementation
- Evaluation dataset with query type labels

---

## Phase 5: Cross-Encoder Reranking

**Goal:** Improve precision of retrieved chunks

### Problem Phase 4 Can't Solve

Bi-encoder (current embeddings) retrieval is fast but imprecise:
- Returns semantically similar chunks that don't actually answer the query
- Ranking quality degrades with diverse chunk types

### Solution: Two-Stage Retrieval with Reranking

**5.1 Architecture**

```
Stage 1: Fast Retrieval (Current)
  - Bi-encoder embeddings (SentenceTransformers)
  - Retrieve top 50 candidates from each index
  - Fast, broad recall

Stage 2: Precise Reranking (New)
  - Cross-encoder model (ms-marco-MiniLM or bge-reranker)
  - Score query-chunk pairs directly
  - Rerank to top 10-15
  - Much slower but much more accurate
```

**5.2 Implementation**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

# Stage 1: Retrieve candidates
candidates = hybrid_retrieve(query, top_k=50)

# Stage 2: Rerank
pairs = [(query, chunk['text']) for chunk in candidates]
scores = reranker.predict(pairs)

# Return top K after reranking
reranked = [candidates[i] for i in np.argsort(scores)[::-1][:10]]
```

**Success Metric:**
- Precision@10 improves by 30%+
- Fewer irrelevant chunks in final context
- Better answer quality on complex queries

**Estimated Effort:** 1 week
- Cross-encoder model integration
- Reranking pipeline
- Performance benchmarking

---

## Phase 6: Contextual Chunk Linking

**Goal:** Preserve narrative continuity across chunk boundaries

### Problem Phase 5 Can't Solve

Query: *"What happened after Bilbo found the ring?"*
- Retrieved chunks are from correct scene
- But narrative flow is broken: missing what happened immediately before/after

### Solution: Chunk Neighborhood Context

**6.1 Bidirectional Linking**
Every chunk stores:

```json
{
  "chunk_id": "book_00_ch_05_chunk_023",
  "text": "...",
  "prev_chunk_id": "book_00_ch_05_chunk_022",
  "next_chunk_id": "book_00_ch_05_chunk_024",
  "scene_id": "ch05_riddle_game"
}
```

**6.2 Context Expansion Strategy**

```python
def expand_context(retrieved_chunks, expand_by=1):
    """
    For each retrieved chunk, optionally include:
    - Previous N chunks (what led here)
    - Next N chunks (what happened next)
    """
    expanded = []
    for chunk in retrieved_chunks:
        if query_needs_preceding_context():
            expanded.append(load_chunk(chunk.prev_chunk_id))
        
        expanded.append(chunk)
        
        if query_needs_following_context():
            expanded.append(load_chunk(chunk.next_chunk_id))
    
    return deduplicate_and_order(expanded)
```

**6.3 Scene Completion**
If multiple chunks from same scene are retrieved:
- Retrieve entire scene
- Highlight relevant chunks within scene context

**Success Metric:**
- "What happened before/after X?" queries improve 40%+
- Narrative continuity maintained in answers
- LLM has better context for reasoning

**Estimated Effort:** 1 week
- Linking metadata generation
- Context expansion logic
- Smart deduplication

---

## Phase 7: Thematic & Emotional Indexing

**Goal:** Answer abstract, literary-analysis queries

### Problem Phase 6 Can't Solve

Query: *"What are the main themes of courage in the story?"*
- Current system: keyword/semantic search for "courage"
- Missing: Passages where courage is *implied* but not stated

### Solution: Thematic Metadata Layer

**7.1 Theme Extraction**
For each chunk, use LLM to tag:

```json
{
  "chunk_id": "...",
  "themes": ["courage", "greed", "home", "transformation"],
  "emotions": ["fear", "determination", "joy"],
  "character_development": ["Bilbo shows unexpected bravery"],
  "symbolic_elements": ["ring as temptation"]
}
```

**7.2 Thematic Index**

```
Theme: "courage"
  - Ch1, Chunk 5: Bilbo decides to join adventure (reluctant courage)
  - Ch5, Chunk 23: Bilbo faces Gollum alone (desperate courage)
  - Ch12, Chunk 8: Bilbo enters Smaug's lair (mature courage)
  
Emotion: "fear"
  - Ch1, Chunk 8: Bilbo's anxiety about adventure
  - Ch5, Chunk 20: Terror during riddle game
  - Ch8, Chunk 15: Dwarves' fear of spiders
```

**7.3 Thematic Retrieval**

```python
if query_is_thematic(query):
    # Extract themes from query
    themes = extract_themes(query)  # e.g., ["courage", "growth"]
    
    # Retrieve chunks tagged with these themes
    thematic_chunks = []
    for theme in themes:
        thematic_chunks += theme_index[theme]
    
    # Cluster by chapter for narrative progression
    return cluster_by_narrative_arc(thematic_chunks)
```

**Success Metric:**
- Can answer: "How does X evolve as a theme?"
- Can answer: "What drives character Y's emotions?"
- Literary analysis queries show 50%+ improvement

**Estimated Effort:** 2-3 weeks
- Theme extraction prompt engineering
- Thematic index structure
- Integration with existing retrieval

---

## Phase 8: Causal Reasoning & Event Chains

**Goal:** Answer "why" questions requiring multi-hop reasoning

### Problem Phase 7 Can't Solve

Query: *"Why did the Battle of Five Armies happen?"*
- Need to trace: Thorin's gold obsession → Bard kills Smaug → Elves + Men claim treasure → Dwarves refuse → Battle

Current system can't link these causally.

### Solution: Event Graph with Causal Edges

**8.1 Event Extraction**
Extract events with:

```json
{
  "event_id": "evt_smaug_death",
  "type": "death",
  "description": "Bard kills Smaug with black arrow",
  "chapter": 14,
  "participants": ["Bard", "Smaug"],
  "location": "Lake-town",
  "causes": ["evt_dwarves_awaken_dragon"],
  "effects": ["evt_lake_town_celebration", "evt_treasure_dispute"]
}
```

**8.2 Causal Chain Construction**

```python
# Build directed graph: Event → Causes → Event
causal_graph = {
    "evt_smaug_death": {
        "causes": ["evt_dwarves_awaken_dragon"],
        "effects": ["evt_treasure_dispute", "evt_battle_of_five_armies"]
    }
}

def trace_causality(query, target_event):
    """
    Find causal path from root causes to target event
    """
    path = find_shortest_path(causal_graph, root, target_event)
    return [load_event_details(evt) for evt in path]
```

**8.3 Why-Question Handler**

```python
if query.startswith("Why"):
    # Extract event mentioned
    event = extract_event_from_query(query)
    
    # Trace causal chain
    causal_chain = trace_causality(query, event)
    
    # Retrieve chunks for each event in chain
    context = []
    for evt in causal_chain:
        context += retrieve_chunks_for_event(evt)
    
    return assemble_causal_narrative(context, causal_chain)
```

**Success Metric:**
- "Why did X happen?" queries show 60%+ improvement
- Multi-hop reasoning (3+ causal steps) works reliably
- LLM receives explicit causal structure to reason over

**Estimated Effort:** 3-4 weeks
- Event extraction pipeline
- Causal relationship detection (LLM-assisted)
- Graph construction and storage
- Causal query handler

---

## Phase 9: Conversational Memory & Multi-Turn RAG

**Goal:** Support follow-up questions and contextual queries

### Problem Phase 8 Can't Solve

```
User: "Who is Gandalf?"
System: [Retrieves and answers]

User: "What does he do in Chapter 5?"
System: [Doesn't know "he" = Gandalf, loses context]
```

### Solution: Conversation State Management

**9.1 Conversation Context**

```json
{
  "conversation_id": "conv_123",
  "turns": [
    {
      "turn": 1,
      "query": "Who is Gandalf?",
      "entities_mentioned": ["Gandalf"],
      "retrieved_chunks": ["ch01_chunk_05", "ch01_chunk_12"],
      "response": "..."
    },
    {
      "turn": 2,
      "query": "What does he do in Chapter 5?",
      "resolved_query": "What does Gandalf do in Chapter 5?",
      "entities_mentioned": ["Gandalf"],
      "retrieved_chunks": ["ch05_chunk_08"],
      "response": "..."
    }
  ],
  "active_entities": ["Gandalf", "Bilbo"],
  "active_chapters": [1, 5],
  "active_themes": ["adventure"]
}
```

**9.2 Coreference Resolution**

```python
def resolve_query(query, conversation_context):
    """
    Resolve pronouns and implicit references
    """
    if has_pronoun(query):  # "he", "she", "it", "they"
        # Look at previous turns
        last_entities = conversation_context["active_entities"]
        query = substitute_pronouns(query, last_entities)
    
    if is_implicit_continuation(query):  # "What about Chapter 5?"
        # Add context from previous query
        query = expand_with_context(query, conversation_context)
    
    return query
```

**9.3 Follow-up Handling**

```python
# User: "Tell me more"
if query_is_followup(query):
    # Retrieve additional chunks from same context
    previous_chunks = conversation_context.turns[-1]["retrieved_chunks"]
    return expand_around_chunks(previous_chunks, expand_by=2)
```

**Success Metric:**
- Multi-turn conversations maintain coherence
- Pronouns resolved correctly 90%+
- Users can naturally explore topics across turns

**Estimated Effort:** 2 weeks
- Conversation state management
- Coreference resolution (rule-based or LLM)
- Follow-up query handling

---

## Phase 10: Evaluation & Metrics Framework

**Goal:** Systematic measurement of RAG quality

### Problem Phases 1-9 Can't Solve

No objective way to measure:
- Which phase actually improved things?
- Where is the system still failing?
- How does it compare to baseline?

### Solution: Comprehensive Evaluation Suite

**10.1 Test Query Dataset**
Create 100+ diverse queries:

```
Factual (20):
  - "What is Bilbo's weapon?"
  - "Where does Gandalf first meet Bilbo?"

Narrative (20):
  - "Describe the riddle game with Gollum"
  - "What happened at the Battle of Five Armies?"

Analytical (20):
  - "Why does Thorin distrust Bilbo?"
  - "How does Bilbo change throughout the story?"

Comparative (15):
  - "Compare Bilbo in Ch1 vs Ch19"
  - "How do dwarves vs elves view treasure?"

Thematic (15):
  - "What role does greed play?"
  - "How is courage portrayed?"

Causal (10):
  - "Why did the battle happen?"
  - "What led to Smaug's death?"
```

**10.2 Ground Truth Annotations**
For each query:
- Expected chunks (which chunks should be retrieved)
- Key facts (what must be in the answer)
- Evaluation rubric (1-5 scale for different aspects)

**10.3 Automated Metrics**

```python
# Retrieval Metrics
- Recall@K: % of relevant chunks retrieved
- Precision@K: % of retrieved chunks that are relevant
- MRR: Mean Reciprocal Rank of first relevant chunk
- NDCG: Normalized Discounted Cumulative Gain

# Answer Quality Metrics (LLM-as-judge)
- Factual correctness (compared to ground truth)
- Completeness (covers all key points)
- Coherence (narrative flow)
- Groundedness (no hallucinations)
```

**10.4 Regression Testing**

```bash
# Run after each phase
python evaluate.py --phase 5 --compare-to phase4

# Output:
Phase 5 vs Phase 4:
  Recall@10:     +15% (0.72 → 0.83)
  Precision@10:  +22% (0.58 → 0.71)
  Answer Quality: +18% (3.2 → 3.8)
  
Query Types Most Improved:
  - Narrative: +35%
  - Analytical: +28%
```

**Success Metric:**
- Can measure impact of each phase quantitatively
- Catch regressions before deployment
- Identify remaining failure modes systematically

**Estimated Effort:** 2-3 weeks
- Query dataset creation (with GPT-4 assistance)
- Ground truth annotation
- Evaluation harness
- Reporting dashboard

---

## Phase 11: LLM Response Optimization

**Goal:** Improve final answer generation quality

### Problem Phase 10 Can't Solve

Even with perfect retrieval, LLM can:
- Hallucinate details not in context
- Miss key information that was retrieved
- Provide unfocused, rambling answers

### Solution: Prompt Engineering + Citation System

**11.1 Structured Prompting**

```python
NOVEL_RAG_PROMPT = """
You are answering questions about "The Hobbit" by J.R.R. Tolkien.

CRITICAL RULES:
1. Only use information from the provided text passages
2. If information is not in passages, say "The text doesn't specify"
3. Maintain narrative voice and literary quality
4. Cite which passage(s) support each claim
5. For "why" questions, trace causality explicitly

CONTEXT:
{retrieved_chunks}

QUERY: {query}

ANSWER (with citations):
"""
```

**11.2 Citation System**

```python
# Each chunk has ID
chunk = {
    "chunk_id": "ch05_chunk_023",
    "text": "...",
    "chapter": 5,
    "page": 78
}

# LLM response format
response = """
Bilbo found the ring in Gollum's cave [ch05_chunk_023]. 
The ring made him invisible [ch05_chunk_028], which helped him 
escape from the goblins [ch05_chunk_031].

Sources:
- Chapter 5, Chunk 23 (page 78)
- Chapter 5, Chunk 28 (page 80)
- Chapter 5, Chunk 31 (page 82)
"""
```

**11.3 Answer Validation**

```python
def validate_answer(answer, retrieved_chunks):
    """
    Check for hallucinations
    """
    # Extract claims from answer
    claims = extract_claims(answer)
    
    # For each claim, verify it's grounded in chunks
    for claim in claims:
        if not is_supported_by(claim, retrieved_chunks):
            flag_hallucination(claim)
    
    # Optionally: regenerate with stricter prompt
```

**Success Metric:**
- Hallucination rate < 5%
- All answers include proper citations
- Answer quality scores improve 15%+

**Estimated Effort:** 1-2 weeks
- Prompt engineering iteration
- Citation system implementation
- Validation logic

---

## Phase 12: Performance Optimization

**Goal:** Make system fast enough for production

### Current Expected Bottlenecks

- Cross-encoder reranking (Phase 5): ~500ms per query
- LLM entity extraction (Phase 3): ~2s per chunk
- Multiple index searches (Phase 4): ~300ms per query

### Solution: Caching & Parallelization

**12.1 Aggressive Caching**

```python
# Cache embeddings (already done)
# Cache LLM calls for entity extraction
# Cache cross-encoder scores for common query-chunk pairs
# Cache query classifications

from functools import lru_cache
import redis  # For persistent cache

@lru_cache(maxsize=10000)
def get_chunk_embedding(chunk_id):
    return embedding_store.get(chunk_id)

# Redis for LLM call caching
def extract_entities_cached(chunk_text):
    cache_key = hash(chunk_text)
    if redis_client.exists(cache_key):
        return redis_client.get(cache_key)
    
    result = llm_extract_entities(chunk_text)
    redis_client.set(cache_key, result, ex=86400)  # 24h TTL
    return result
```

**12.2 Parallel Processing**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_retrieval(query):
    """
    Run multiple retrievals in parallel
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        micro_future = executor.submit(retrieve_micro, query)
        scene_future = executor.submit(retrieve_scenes, query)
        chapter_future = executor.submit(retrieve_chapters, query)
        entity_future = executor.submit(retrieve_entities, query)
        
        results = await asyncio.gather(
            micro_future,
            scene_future,
            chapter_future,
            entity_future
        )
    
    return merge_results(results)
```

**12.3 Index Optimization**

```python
# Use FAISS for faster vector search (if needed)
# Optimize BM25 index structure
# Precompute common query patterns
```

**Success Metric:**
- End-to-end query latency < 2s (p95)
- Retrieval phase < 500ms
- Can handle 10+ concurrent queries

**Estimated Effort:** 1-2 weeks
- Caching implementation
- Parallel processing refactor
- Benchmarking and tuning

---

## Phase 13: User Interface & Visualization

**Goal:** Make system usable and debuggable

### Solution: Interactive Query Interface

**13.1 Web Interface**

```
Streamlit or Gradio app:
- Query input
- Retrieved chunks display (with scores)
- Entity graph visualization
- Timeline view
- Citation links to original text
- Evaluation metrics overlay
```

**13.2 Debugging Tools**

```
- Query classification view
- Retrieval strategy trace
- Chunk relevance heatmap
- Entity extraction review
- Reranking score comparison
```

**Success Metric:**
- Non-technical users can query the system
- Developers can debug retrieval failures visually
- Evaluation becomes interactive

**Estimated Effort:** 1-2 weeks
- UI implementation (Streamlit)
- Visualization components
- Integration with backend

---

## Summary: Complete Roadmap

| Phase | Goal | Key Deliverable | Effort | Cumulative |
|-------|------|----------------|--------|------------|
| 1 | Foundation | Basic RAG working | 2-3 weeks | 3 weeks |
| 2 | Multi-granularity | Scene + chapter chunks | 2-3 weeks | 6 weeks |
| 3 | Entity graph | Character tracking + relationships | 3-4 weeks | 10 weeks |
| 4 | Query routing | Intelligent retrieval strategy | 2 weeks | 12 weeks |
| 5 | Reranking | Cross-encoder precision | 1 week | 13 weeks |
| 6 | Chunk linking | Narrative continuity | 1 week | 14 weeks |
| 7 | Thematic index | Literary analysis queries | 2-3 weeks | 17 weeks |
| 8 | Causal reasoning | Event graphs + "why" questions | 3-4 weeks | 21 weeks |
| 9 | Conversational | Multi-turn dialogue | 2 weeks | 23 weeks |
| 10 | Evaluation | Comprehensive metrics | 2-3 weeks | 26 weeks |
| 11 | LLM optimization | Prompting + citations | 1-2 weeks | 28 weeks |
| 12 | Performance | Caching + parallelization | 1-2 weeks | 30 weeks |
| 13 | UI | Interactive interface | 1-2 weeks | 32 weeks |

**Total Timeline: ~8 months to literary-grade RAG**

---

## Critical Decision Points

### After Phase 1

**Decision:** Is basic retrieval "good enough"?
- If YES → Stop here, you have functional RAG
- If NO → Continue to Phase 2

### After Phase 3

**Decision:** Are entity-focused queries working?
- If YES → Skip Phase 7 (thematic), go to Phase 4
- If NO → Phase 3 needs more work

### After Phase 6

**Decision:** Do you need causal reasoning?
- If YES → Proceed to Phase 8
- If NO → Skip to Phase 10 (evaluation)

### After Phase 10

**Decision:** What's the biggest remaining failure mode?
- Prioritize phases based on evaluation results
- Not all phases may be necessary

---

## Recommended Approach: Iterative with Exit Points

**DON'T try to build all 13 phases.**

Instead:
1. Build Phase 1 → Evaluate
2. Identify top 3 failure modes
3. Pick the phase that addresses #1 failure mode
4. Build → Evaluate → Repeat

**Exit criteria:**
- When evaluation shows 85%+ query satisfaction
- When cost/benefit of next phase is too high
- When system meets your quality bar

---

## Technology Stack Evolution

| Phase | New Tech | Why |
|-------|----------|-----|
| 1 | ChromaDB, SentenceTransformers, BM25 | Foundation |
| 2 | None | Reuse Phase 1 |
| 3 | NetworkX or JSON graph | Entity relationships |
| 5 | CrossEncoder model | Reranking |
| 7 | Ollama for theme extraction | LLM-based tagging |
| 8 | Graph algorithms (shortest path) | Causal chains |
| 9 | Conversation state management | Dialogue context |
| 12 | Redis (optional), FAISS (optional) | Performance |
| 13 | Streamlit or Gradio | UI |

**Philosophy:** Add complexity only when measurement proves it's needed.

---

## Final Architecture Diagram (Phase 13)

```
┌─────────────────────────────────────────────────────────────┐
│                         USER QUERY                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │   Query Classifier       │ ◄── Conversation Context
         │  (Phase 4 + Phase 9)    │
         └─────────┬───────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌─────────┐  ┌──────────┐  ┌──────────┐
│  Micro  │  │  Scene   │  │ Chapter  │
│ Chunks  │  │  Chunks  │  │ Summaries│
│(Phase 1)│  │(Phase 2) │  │(Phase 2) │
└────┬────┘  └────┬─────┘  └────┬─────┘
     │            │             │
     └────────────┼─────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Entity Graph   │ ◄── Character profiles, relationships
         │   (Phase 3)     │
         └────────┬────────┘
                  │
                  ▼
         ┌────────────────┐
         │ Hybrid Search   │ ◄── Dense + BM25
         │   (Phase 1)     │
         └────────┬────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Cross-Encoder  │ ◄── Reranking
         │  (Phase 5)      │
         └────────┬────────┘
                  │
                  ▼
         ┌────────────────┐
         │Context Assembly │ ◄── Chunk linking, narrative order
         │   (Phase 6)     │
         └────────┬────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐
│Thematic│  │  Causal  │  │Event     │
│Context │  │  Chain   │  │Context   │
│(Ph 7)  │  │ (Phase 8)│  │(Phase 8) │
└───┬────┘  └────┬─────┘  └────┬─────┘
    │            │             │
    └────────────┼─────────────┘
                 │
                 ▼
        ┌─────────────────┐
        │  LLM Generation  │ ◄── Structured prompt + citations
        │   (Phase 11)     │
        └─────────┬────────┘
                  │
                  ▼
        ┌─────────────────┐
        │Answer Validation │ ◄── Hallucination check
        │   (Phase 11)     │
        └─────────┬────────┘
                  │
                  ▼
           ┌────────────┐
           │  RESPONSE  │ ◄── With citations + confidence
           └────────────┘
```

---

## Key Principles for Every Phase

1. **Test before building:** Write evaluation queries first
2. **Measure everything:** Before/after metrics for each phase
3. **Fail fast:** If a phase doesn't improve metrics, skip it
4. **Reuse code:** Centralize in utils/, minimize duplication
5. **One phase at a time:** No parallel development
6. **Document decisions:** Why this approach? What did we learn?
7. **Exit points:** Know when "good enough" is actually good enough

---

**This roadmap takes you from basic RAG to a system that understands The Hobbit at a literary level. But remember: Not all phases may be necessary. Build, measure, decide.**
