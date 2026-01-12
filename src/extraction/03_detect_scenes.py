"""
Scene Detection using Chunk Similarity Analysis
================================================
Analyzes semantic similarities between consecutive chunks to identify
natural scene boundaries in chapters.

Input: data/chunks/book_chunks.jsonl (small chunks)
Output: data/chunks/scene_chunks.jsonl (scene-level chunks)

Process:
1. Load all chunks grouped by chapter
2. Embed chunks and calculate consecutive similarities
3. Detect valleys (low similarity = scene boundaries)
4. Merge chunks between boundaries into scenes
5. Visualize similarity patterns (optional)
"""

import time
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.distance import cosine

from src.config import get_config
from src.paths import get_paths
from src.utils.embedding.embedding_factory import create_embedding_manager
from src.utils.util_files_functions import load_jsonl_from_file, save_jsonl_to_file
from src.utils.util_statistics import total_statistics_logging

# Global embedding manager
_embedding_manager = None


def analyze_chapter_similarities(similarities: List[float]) -> Dict:
    """
    Analyze similarity distribution to recommend parameters.
    """
    sims = np.array(similarities)

    stats = {
        "mean": np.mean(sims),
        "std": np.std(sims),
        "min": np.min(sims),
        "max": np.max(sims),
        "median": np.median(sims),
        "p25": np.percentile(sims, 25),  # 25th percentile
        "p75": np.percentile(sims, 75),  # 75th percentile
        "range": np.max(sims) - np.min(sims),
    }

    # Adaptive thresholds
    adaptive = {
        # Threshold: mean minus some fraction of std
        "threshold_conservative": stats["mean"] - 0.5 * stats["std"],
        "threshold_moderate": stats["mean"] - 1.0 * stats["std"],
        "threshold_aggressive": stats["p25"],  # 25th percentile
        # Min drop: relative to std
        "min_drop_conservative": 0.5 * stats["std"],
        "min_drop_moderate": 1.0 * stats["std"],
        # Strong drop: outliers
        "strong_drop": 1.5 * stats["std"],
    }

    return {**stats, **adaptive}


def detect_scene_boundaries(similarities: List[float]) -> List[int]:
    """
    Detect scene boundaries with optional adaptive parameters.
    """
    if len(similarities) < 2:
        return []

    # Check if adaptive mode is enabled
    use_adaptive = SCENE_DETECTION_CONFIG.get("USE_ADAPTIVE_PARAMETERS", False)

    if use_adaptive:
        # Calculate adaptive parameters based on chapter's similarity distribution
        adaptive_params = analyze_chapter_similarities(similarities)

        # Choose which adaptive strategy (conservative, moderate, aggressive)
        strategy = SCENE_DETECTION_CONFIG.get("ADAPTIVE_STRATEGY", "moderate")  # conservative, moderate, or aggressive

        if strategy == "conservative":
            threshold = adaptive_params["threshold_conservative"]
            min_drop = adaptive_params["min_drop_conservative"]
            strong_drop = adaptive_params["strong_drop"]
        elif strategy == "aggressive":
            threshold = adaptive_params["threshold_aggressive"]
            min_drop = adaptive_params["min_drop_moderate"]
            strong_drop = adaptive_params["strong_drop"]
        else:  # moderate (default)
            threshold = adaptive_params["threshold_moderate"]
            min_drop = adaptive_params["min_drop_moderate"]
            strong_drop = adaptive_params["strong_drop"]

        recovery_limit = SCENE_DETECTION_CONFIG.get("RECOVERY_LIMIT", 0.05)

        print("  📊 ADAPTIVE PARAMETERS:")
        print(f"     Similarity stats: mean={adaptive_params['mean']:.3f}, std={adaptive_params['std']:.3f}")
        print(f"     Using {strategy} strategy:")
        print(f"     threshold={threshold:.3f}, min_drop={min_drop:.3f}, strong_drop={strong_drop:.3f}")
    else:
        # Use fixed parameters from config
        threshold = SCENE_DETECTION_CONFIG.get("SCENE_SIMILARITY_THRESHOLD", 0.87)
        min_drop = SCENE_DETECTION_CONFIG.get("VALLEY_MIN_DROP", 0.02)
        strong_drop = SCENE_DETECTION_CONFIG.get("STRONG_DROP_THRESHOLD", 0.04)
        recovery_limit = SCENE_DETECTION_CONFIG.get("RECOVERY_LIMIT", 0.05)

    if len(similarities) < 2:
        return []

    boundaries = []

    for i in range(1, len(similarities) - 1):
        curr = similarities[i]
        prev = similarities[i - 1]
        next_sim = similarities[i + 1]

        drop = prev - curr

        # DEBUG: Print every check
        print(f"  Checking i={i}: curr={curr:.3f}, prev={prev:.3f}, next={next_sim:.3f}, drop={drop:.3f}")

        is_local_minimum = curr < prev and curr < next_sim
        is_below_threshold = curr <= threshold
        has_min_drop = drop >= min_drop
        is_strong_drop = drop >= strong_drop

        print(f"    local_min={is_local_minimum}, below_thresh={is_below_threshold}, has_drop={has_min_drop}, strong_drop={is_strong_drop}")

        # Trigger if: (local min + threshold + drop) OR (strong drop + threshold)
        triggers_normal = is_local_minimum and is_below_threshold and has_min_drop
        triggers_strong = is_strong_drop and is_below_threshold

        if triggers_normal or triggers_strong:
            # Noise filter: ensure it doesn't immediately rebound
            if i + 1 < len(similarities):
                recovery = similarities[i + 1] - curr
                if recovery > recovery_limit:
                    print(f"    ⚠️ BLOCKED by noise filter (recovery={recovery:.3f})")
                    continue

            print("    🔪 SCENE CUT detected:")
            print(f"       Local minimum at similarity[{i}] = {curr:.3f}")
            print(f"       Drop: {drop:.3f} | Prev: {prev:.3f}, Next: {next_sim:.3f}")
            print(f"       New scene starts at chunk[{i + 1}]")
            boundaries.append(i + 1)

    return boundaries


def merge_chunks_into_scenes(chunks: List[Dict], boundaries: List[int], similarities: List[float], chapter_num: int) -> List[Dict]:
    """
    Merge chunks between boundaries into scene-level chunks.
    """
    scenes = []

    # Add boundaries at start and end
    all_boundaries = [0] + boundaries + [len(chunks)]

    # DEBUG: Print boundary details
    print(f"\n  📍 Scene boundaries: {boundaries}")
    for boundary_idx in boundaries:
        if boundary_idx < len(chunks):
            chunk_text = chunks[boundary_idx]["text"][:100]
            print(f'     Scene starts at chunk[{boundary_idx}]: "{chunk_text}..."')

    for scene_idx in range(len(all_boundaries) - 1):
        start_idx = all_boundaries[scene_idx]
        end_idx = all_boundaries[scene_idx + 1]

        # Get chunks in this scene
        scene_chunks = chunks[start_idx:end_idx]

        if not scene_chunks:
            continue

        # Merge text
        scene_text = " ".join([c["text"] for c in scene_chunks])

        # Get chunk IDs
        chunk_ids = [c["chunk_id"] for c in scene_chunks]

        # FIX: Correct similarity slicing
        # similarities[k] = similarity between chunk[k] and chunk[k+1]
        # For scene covering chunks [start_idx ... end_idx-1],
        # we want similarities[start_idx ... end_idx-2]
        scene_similarities = []
        if end_idx - start_idx > 1:  # At least 2 chunks in scene
            sim_start = start_idx
            sim_end = min(end_idx - 1, len(similarities))
            scene_similarities = similarities[sim_start:sim_end]

        # FIX: Use proper token counting
        scene_tokens = sum(_embedding_manager.token_count(c["text"]) for c in scene_chunks)

        # Create scene chunk
        scene = {
            "source": "book",
            "book_number": chunks[0]["book_number"],
            "book_title": chunks[0]["book_title"],
            "chapter_number": chapter_num,
            "chapter_title": chunks[0]["chapter_title"],
            "chapter_type": chunks[0]["chapter_type"],
            "scene_number": scene_idx + 1,
            "scene_id": f"book_{chunks[0]['book_number']:02d}_ch_{chapter_num:02d}_scene_{scene_idx + 1:03d}",
            "text": scene_text,
            "chunk_ids": chunk_ids,
            "num_chunks": len(scene_chunks),
            "tokens": scene_tokens,
            "avg_similarity": float(np.mean(scene_similarities)) if scene_similarities else 0.0,
            "min_similarity": float(np.min(scene_similarities)) if scene_similarities else 0.0,
        }

        scenes.append(scene)

    return scenes


def visualize_similarities(chapter_num: int, similarities: List[float], boundaries: List[int], output_path: str = None):
    """
    Create a visualization of similarity scores and detected boundaries.

    Args:
        chapter_num: Chapter number
        similarities: List of similarity scores
        boundaries: Scene boundary indices
        output_path: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(14, 6))

        # Plot similarities
        x = list(range(len(similarities)))
        plt.plot(x, similarities, "b-", linewidth=2, label="Chunk Similarity")

        # Mark boundaries
        for boundary in boundaries:
            if boundary < len(similarities):
                plt.axvline(x=boundary, color="r", linestyle="--", linewidth=2, alpha=0.7)

        # Add threshold line
        threshold = SCENE_DETECTION_CONFIG.get("SCENE_SIMILARITY_THRESHOLD", 0.65)
        plt.axhline(y=threshold, color="g", linestyle=":", linewidth=1, alpha=0.5, label=f"Threshold ({threshold})")

        plt.xlabel("Chunk Index", fontsize=12)
        plt.ylabel("Cosine Similarity", fontsize=12)
        plt.title(f"Chapter {chapter_num}: Semantic Similarity Between Consecutive Chunks", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.7, 1)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"  📊 Saved visualization to {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("  ⚠️ matplotlib not available, skipping visualization")


def process_chapter_scenes(chapter_chunks: List[Dict], chapter_num: int) -> Tuple[List[Dict], Dict]:
    """
    Process one chapter to detect scenes.

    Args:
        chapter_chunks: List of chunk dicts for this chapter
        chapter_num: Chapter number

    Returns:
        Tuple of (scene_chunks, statistics)
    """
    global _embedding_manager

    if len(chapter_chunks) < 2:
        print(f"  ⚠️ Chapter {chapter_num} has only {len(chapter_chunks)} chunks, skipping scene detection")
        return [], {}

    # Extract texts
    texts = [chunk["text"] for chunk in chapter_chunks]

    print(f"  📝 Embedding {len(texts)} chunks...")

    # Embed all chunks
    embed_result = _embedding_manager.embed_chunks(texts, show_progress=False)

    if isinstance(embed_result, (tuple, list)):
        embeddings = embed_result[0]
    else:
        embeddings = embed_result

    embeddings = np.array(embeddings)

    # Calculate consecutive similarities
    print("  🔍 Calculating similarities...")
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = 1 - cosine(embeddings[i], embeddings[i + 1])
        similarities.append(float(sim))

    # Detect scene boundaries
    print("  🎬 Detecting scene boundaries...")
    boundaries = detect_scene_boundaries(similarities)

    print(f"  ✓ Found {len(boundaries)} scene boundaries")
    print(f"    → {len(boundaries) + 1} scenes detected")

    # Merge chunks into scenes
    scenes = merge_chunks_into_scenes(chapter_chunks, boundaries, similarities, chapter_num)

    # Visualize if enabled
    if SCENE_DETECTION_CONFIG.get("VISUALIZE_SCENES", False):
        paths = get_paths()
        viz_path = f"{paths.DATA_TEMP_PATH}/chapter_{chapter_num:02d}_similarities.png"
        visualize_similarities(chapter_num, similarities, boundaries, viz_path)

    # Generate statistics
    stats = {
        "chapter": chapter_num,
        "num_chunks": len(chapter_chunks),
        "num_scenes": len(scenes),
        "scenes_tokens": [scene["tokens"] for scene in scenes],
        "avg_chunks_per_scene": len(chapter_chunks) / len(scenes) if scenes else 0,
        "avg_similarity": float(np.mean(similarities)),
        "min_similarity": float(np.min(similarities)),
        "max_similarity": float(np.max(similarities)),
        "num_boundaries": len(boundaries),
    }

    return scenes, stats


def generate_statistics(all_scenes: List[Dict], chapter_stats: List[Dict]) -> List[Dict]:
    """
    Generate comprehensive statistics for scene detection.
    """
    per_chapter_stats = []

    # Per-chapter stats
    for ch_stat in chapter_stats:
        per_chapter_stats.append(
            {
                "name": f"Chapter {ch_stat['chapter']}",
                "metrics": {
                    "chunks": ch_stat["num_chunks"],
                    "scenes": ch_stat["num_scenes"],
                    "scenes_token_count": ch_stat["scenes_tokens"],
                    "chunks_per_scene": f"{ch_stat['avg_chunks_per_scene']:.1f}",
                    "avg_similarity": f"{ch_stat['avg_similarity']:.3f}",
                    "boundaries": ch_stat["num_boundaries"],
                },
            }
        )

    # Total stats
    total_chunks = sum(s["num_chunks"] for s in chapter_stats)
    total_scenes = sum(s["num_scenes"] for s in chapter_stats)

    # Scene size distribution
    scene_sizes = [s["num_chunks"] for s in all_scenes]
    scene_tokens = [s["tokens"] for s in all_scenes]

    per_chapter_stats.append(
        {
            "name": "TOTAL",
            "metrics": {
                "total_chunks": total_chunks,
                "total_scenes": total_scenes,
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

    global _embedding_manager, SCENE_DETECTION_CONFIG

    SCENE_DETECTION_CONFIG = config.SCENE_DETECTION_CONFIG

    # Initialize embedding manager
    print("🔧 Initializing embedding manager...")
    _embedding_manager = create_embedding_manager(SCENE_DETECTION_CONFIG["EMBEDDING_MANAGER_CONFIG"])

    all_chunks = load_jsonl_from_file(paths.FILE_BOOKS_CHUNKED)

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
        print(f"\n📚 Chapter {ch_num}: {chapter_chunks[0]['chapter_title']}")
        print(f"  Processing {len(chapter_chunks)} chunks...")

        scenes, stats = process_chapter_scenes(chapter_chunks, ch_num)

        all_scenes.extend(scenes)
        if stats:
            chapter_stats.append(stats)

    # Generate statistics
    statistics = generate_statistics(all_scenes, chapter_stats)

    # Save scenes
    print(f"\n💾 Saving {len(all_scenes)} scenes...")
    save_jsonl_to_file(
        data=all_scenes,
        output_file=paths.FILE_SCENE_CHUNKS,
    )

    total_time = time.time() - start_time
    total_statistics_logging(
        statistics,
        total_time,
        "SCENE DETECTION",
        "05_detect_scenes",
        tables=True,
        configuration_section=SCENE_DETECTION_CONFIG,
    )


if __name__ == "__main__":
    main()
