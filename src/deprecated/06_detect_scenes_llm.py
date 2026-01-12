"""
LLM-Based Scene Detection using Sliding Window
===============================================
Uses an LLM to detect scene boundaries by asking if consecutive chunks
belong to the same narrative scene.

Input: data/chunks/book_chunks.jsonl (small chunks)
Output: data/chunks/llm_scene_chunks.jsonl (scene-level chunks)

Process:
1. Load all chunks grouped by chapter
2. Sliding window (chunks i, i+1, i+2)
3. Ask LLM: "Are chunks i and i+1 part of same scene?"
4. Boundary where answer changes from YES to NO
5. Merge chunks into scenes
6. Visualize boundaries
"""

import time
from typing import Dict, List, Tuple

import numpy as np

from src.config import get_config
from src.paths import get_paths
from src.utils.llm_access.ollama_llm import prompt_builder
from src.utils.util_files_functions import load_jsonl_from_file, save_jsonl_to_file
from src.utils.util_statistics import total_statistics_logging

SCENE_DETECTION_PROMPT = """You are analyzing the narrative structure of "The Hobbit" by J.R.R. Tolkien.

TASK: Determine if two consecutive text chunks belong to the SAME narrative scene.

DEFINITION OF A SCENE:
A scene is a continuous narrative unit where:
✓ Same time period (no time jumps like "next morning", "later that day")
✓ Same location (no shifts like "meanwhile in another place")
✓ Same character focus (no switches to different characters' perspectives)
✓ Continuous action/dialogue (no major topic changes)

CONTEXT - Three consecutive chunks:

--- CHUNK {chunk_num_1} ---
{chunk_text_1}

--- CHUNK {chunk_num_2} ---
{chunk_text_2}

--- CHUNK {chunk_num_3} ---
{chunk_text_3}

QUESTION: Are Chunk {chunk_num_1} and Chunk {chunk_num_2} part of the SAME scene?

IMPORTANT:
- Focus ONLY on chunks {chunk_num_1} and {chunk_num_2}
- Chunk {chunk_num_3} is provided for context only
- Look for: time jumps (next day, one morning), location changes, character shifts, topic breaks
- If unsure, prefer YES (same scene)

Answer with A JSON containing the WORD: YES or NO

{{
   "answer": "YES"
}}

"""


def ask_llm_same_scene(chunk1: Dict, chunk2: Dict, chunk3: Dict, config: Dict) -> bool:
    """
    Ask LLM if chunk1 and chunk2 are part of the same scene.

    Args:
        chunk1, chunk2, chunk3: Consecutive chunk dicts with 'text' and 'chunk_id'
        config: LLM configuration

    Returns:
        True if same scene, False if different scenes
    """
    # Safely get chunk index (fallback to chunk_id if chunk_index missing)
    chunk1_id = chunk1.get("chunk_index", chunk1.get("chunk_id", "?"))
    chunk2_id = chunk2.get("chunk_index", chunk2.get("chunk_id", "?"))
    chunk3_id = chunk3.get("chunk_index", chunk3.get("chunk_id", "?"))

    # Safely truncate text (handle empty chunks)
    chunk1_text = chunk1.get("text", "")[:1000] if chunk1.get("text") else "[Empty chunk]"
    chunk2_text = chunk2.get("text", "")[:1000] if chunk2.get("text") else "[Empty chunk]"
    chunk3_text = chunk3.get("text", "")[:1000] if chunk3.get("text") else "[Empty chunk]"

    prompt = SCENE_DETECTION_PROMPT.format(
        chunk_num_1=chunk1_id,
        chunk_text_1=chunk1_text,
        chunk_num_2=chunk2_id,
        chunk_text_2=chunk2_text,
        chunk_num_3=chunk3_id,
        chunk_text_3=chunk3_text,
    )

    # Get LLM response with retries
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            result = prompt_builder(prompt, config["REASONING_MODEL_CONFIG"])

            if not result or not isinstance(result, Dict):
                print(f"\n    ⚠️ Invalid LLM response type: {type(result)}. Retry")
                if attempt < max_retries:
                    continue
                return True  # Conservative default

            answer = result.get("answer").strip().upper()

            # Parse answer - be lenient with parsing
            if "YES" in answer or "SAME" in answer:
                return True
            elif "NO" in answer or "DIFFERENT" in answer or "NOT" in answer:
                return False
            else:
                # Ambiguous answer
                print(f"\n    ⚠️ Ambiguous LLM answer (attempt {attempt + 1}): '{answer[:50]}...'")
                if attempt < max_retries:
                    continue
                # Default to YES (conservative - assume same scene unless clearly different)
                print(f"    → Defaulting to YES after {max_retries + 1} attempts")
                return True

        except Exception as e:
            print(f"\n    ⚠️ LLM error (attempt {attempt + 1}): {str(e)[:100]}")
            if attempt < max_retries:
                continue
            # After all retries, default to YES
            print("    → Defaulting to YES after error")
            return True

    return True  # Fallback


def detect_llm_scene_boundaries(chunks: List[Dict], chapter_num: int, config: Dict) -> Tuple[List[int], List[bool]]:
    """
    Detect scene boundaries using LLM sliding window.

    Args:
        chunks: List of chunk dicts
        chapter_num: Chapter number
        config: Configuration

    Returns:
        Tuple of (boundary_indices, same_scene_answers)
    """
    if len(chunks) < 3:
        print(f"  ⚠️ Chapter {chapter_num} has only {len(chunks)} chunks, need at least 3")
        return [], []

    same_scene_answers = []
    boundaries = []

    print(f"  🤖 Asking LLM about {len(chunks) - 2} chunk pairs...")

    # Sliding window: (i, i+1, i+2)
    for i in range(len(chunks) - 2):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        chunk3 = chunks[i + 2]

        # print(f"    Window [{i}, {i + 1}, {i + 2}]: Asking if chunks {i} and {i + 1} are same scene...", end=" ")

        is_same_scene = ask_llm_same_scene(chunk1, chunk2, chunk3, config)
        same_scene_answers.append(is_same_scene)

        # print(f"{'YES' if is_same_scene else 'NO'}")

        # CRITICAL FIX: Every NO means a boundary, regardless of previous answer
        # If chunks i and i+1 are different scenes, boundary is at i+1
        if not is_same_scene:
            boundaries.append(i + 1)
            print(f"      🔪 SCENE BOUNDARY detected! New scene starts at chunk {i + 1}")

    return boundaries, same_scene_answers


def merge_chunks_into_scenes(chunks: List[Dict], boundaries: List[int], chapter_num: int, config) -> List[Dict]:
    """
    Merge chunks between boundaries into scene-level chunks.
    """
    scenes = []

    # Add boundaries at start and end
    all_boundaries = [0] + boundaries + [len(chunks)]

    # Remove duplicate boundaries (can happen with consecutive NOs)
    all_boundaries = sorted(set(all_boundaries))

    print(f"\n  📍 LLM Scene boundaries: {boundaries}")
    for boundary_idx in boundaries:
        if boundary_idx < len(chunks):
            chunk_text = chunks[boundary_idx]["text"][:100]
            print(f'     Scene starts at chunk[{boundary_idx}]: "{chunk_text}..."')

    for scene_idx in range(len(all_boundaries) - 1):
        start_idx = all_boundaries[scene_idx]
        end_idx = all_boundaries[scene_idx + 1]

        scene_chunks = chunks[start_idx:end_idx]

        # Safety check: skip empty scenes
        if not scene_chunks:
            print(f"  ⚠️ Warning: Empty scene detected between boundaries {start_idx} and {end_idx}")
            continue

        # Merge text
        scene_text = " ".join([c["text"] for c in scene_chunks])
        chunk_ids = [c["chunk_id"] for c in scene_chunks]

        # More accurate token counting (approximate based on characters)
        # Rough estimate: 1 token ≈ 4 characters for English text
        scene_char_count = sum(len(c.get("text", "")) for c in scene_chunks)
        scene_tokens = scene_char_count // 4  # Approximate token count

        scene = {
            "source": "book",
            "book_number": chunks[0]["book_number"],
            "book_title": chunks[0]["book_title"],
            "chapter_number": chapter_num,
            "chapter_title": chunks[0].get("chapter_title", "Unknown"),
            "chapter_type": chunks[0].get("chapter_type", "chapter"),
            "scene_number": scene_idx + 1,
            "scene_id": f"book_{chunks[0]['book_number']:02d}_ch_{chapter_num:02d}_llm_scene_{scene_idx + 1:03d}",
            "text": scene_text,
            "chunk_ids": chunk_ids,
            "num_chunks": len(scene_chunks),
            "tokens": scene_tokens,
            "characters": scene_char_count,
            "detection_method": "llm",
        }

        scenes.append(scene)

    return scenes


def visualize_llm_boundaries(chapter_num: int, same_scene_answers: List[bool], boundaries: List[int], output_path: str = None):
    """
    Create a visualization of LLM scene detection.
    Shows YES/NO pattern and detected boundaries.
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 6))

        # Convert boolean to numeric (1=YES, 0=NO)
        y_values = [1 if is_same else 0 for is_same in same_scene_answers]
        x = list(range(len(y_values)))

        # Plot as step function
        plt.step(x, y_values, where="post", linewidth=2, color="blue", label="Same Scene (YES=1, NO=0)")

        # Mark boundaries
        for boundary in boundaries:
            if boundary - 1 < len(y_values):
                plt.axvline(x=boundary - 1, color="r", linestyle="--", linewidth=2, alpha=0.7, label="Scene Boundary" if boundary == boundaries[0] else "")

        plt.xlabel("Chunk Pair Index (i, i+1)", fontsize=12)
        plt.ylabel("Same Scene? (1=YES, 0=NO)", fontsize=12)
        plt.title(f"Chapter {chapter_num}: LLM Scene Detection (Sliding Window)", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.1, 1.1)
        plt.yticks([0, 1], ["NO", "YES"])

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"  📊 Saved LLM visualization to {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("  ⚠️ matplotlib not available, skipping visualization")


def process_chapter_scenes_llm(chapter_chunks: List[Dict], chapter_num: int, config: Dict) -> Tuple[List[Dict], Dict]:
    """
    Process one chapter to detect scenes using LLM.
    """
    if len(chapter_chunks) < 3:
        print(f"  ⚠️ Chapter {chapter_num} has only {len(chapter_chunks)} chunks, skipping LLM scene detection")
        return [], {}

    # Detect boundaries
    boundaries, same_scene_answers = detect_llm_scene_boundaries(chapter_chunks, chapter_num, config)

    print(f"  ✓ Found {len(boundaries)} scene boundaries")
    print(f"    → {len(boundaries) + 1} scenes detected")

    # Merge chunks into scenes
    scenes = merge_chunks_into_scenes(chapter_chunks, boundaries, chapter_num, config)

    # Visualize if enabled
    if config.get("VISUALIZE_SCENES", False):
        paths = get_paths()
        viz_path = f"{paths.DATA_TEMP_PATH}/chapter_{chapter_num:02d}_llm_scenes.png"
        visualize_llm_boundaries(chapter_num, same_scene_answers, boundaries, viz_path)

    # Generate statistics
    stats = {
        "chapter": chapter_num,
        "num_chunks": len(chapter_chunks),
        "num_scenes": len(scenes),
        "avg_chunks_per_scene": len(chapter_chunks) / len(scenes) if scenes else 0,
        "num_boundaries": len(boundaries),
        "num_llm_calls": len(same_scene_answers),
    }

    return scenes, stats


def generate_statistics(all_scenes: List[Dict], chapter_stats: List[Dict]) -> List[Dict]:
    """
    Generate comprehensive statistics for LLM scene detection.
    """
    per_chapter_stats = []

    for ch_stat in chapter_stats:
        per_chapter_stats.append(
            {
                "name": f"Chapter {ch_stat['chapter']}",
                "metrics": {
                    "chunks": ch_stat["num_chunks"],
                    "scenes": ch_stat["num_scenes"],
                    "chunks_per_scene": f"{ch_stat['avg_chunks_per_scene']:.1f}",
                    "boundaries": ch_stat["num_boundaries"],
                    "llm_calls": ch_stat["num_llm_calls"],
                },
            }
        )

    # Total stats
    total_chunks = sum(s["num_chunks"] for s in chapter_stats)
    total_scenes = sum(s["num_scenes"] for s in chapter_stats)
    total_llm_calls = sum(s["num_llm_calls"] for s in chapter_stats)

    scene_sizes = [s["num_chunks"] for s in all_scenes]
    scene_tokens = [s["tokens"] for s in all_scenes]

    per_chapter_stats.append(
        {
            "name": "TOTAL",
            "metrics": {
                "total_chunks": total_chunks,
                "total_scenes": total_scenes,
                "total_llm_calls": total_llm_calls,
                "avg_chunks_per_scene": f"{total_chunks / total_scenes:.1f}" if total_scenes else "0",
                "min_scene_chunks": min(scene_sizes) if scene_sizes else 0,
                "max_scene_chunks": max(scene_sizes) if scene_sizes else 0,
                "avg_scene_tokens": f"{np.mean(scene_tokens):.0f}" if scene_tokens else "0",
            },
        }
    )

    return per_chapter_stats


def main():
    start_time = time.time()

    config = get_config()
    paths = get_paths()

    configuration = {"REASONING_MODEL_CONFIG": config.MEDIUM_MODEL_CONFIG, "VISUALIZE_SCENES": True}

    all_chunks = load_jsonl_from_file(paths.FILE_BOOKS_CHUNKED)

    if not all_chunks:
        print("❌ ERROR: No chunks loaded from file")
        return

    # Validate chunk structure
    required_fields = ["chapter_number", "text", "chunk_id"]
    sample_chunk = all_chunks[0]
    missing_fields = [f for f in required_fields if f not in sample_chunk]
    if missing_fields:
        print(f"❌ ERROR: Chunks missing required fields: {missing_fields}")
        print(f"   Sample chunk keys: {list(sample_chunk.keys())}")
        return

    # Group by chapter
    chapters_dict = {}
    for chunk in all_chunks:
        ch_num = chunk["chapter_number"]
        if ch_num not in chapters_dict:
            chapters_dict[ch_num] = []
        chapters_dict[ch_num].append(chunk)

    print(f"  ✓ Loaded {len(all_chunks)} chunks across {len(chapters_dict)} chapters")

    # Process each chapter
    all_scenes = []
    chapter_stats = []

    for ch_num in sorted(chapters_dict.keys()):
        chapter_chunks = chapters_dict[ch_num]
        print(f"\n📚 Chapter {ch_num}: {chapter_chunks[0].get('chapter_title', 'Unknown')}")
        print(f"  Processing {len(chapter_chunks)} chunks with LLM...")

        try:
            scenes, stats = process_chapter_scenes_llm(chapter_chunks, ch_num, configuration)

            all_scenes.extend(scenes)
            if stats:
                chapter_stats.append(stats)
        except Exception as e:
            print(f"  ❌ ERROR processing chapter {ch_num}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_scenes:
        print("\n❌ No scenes detected across any chapters")
        return

    # Generate statistics
    statistics = generate_statistics(all_scenes, chapter_stats)

    save_jsonl_to_file(data=all_scenes, output_file=paths.FILE_LLM_SCENE_CHUNKS)

    total_time = time.time() - start_time
    total_statistics_logging(
        statistics,
        total_time,
        "LLM SCENE DETECTION",
        "06_detect_scenes_llm",
        tables=True,
    )


if __name__ == "__main__":
    main()
