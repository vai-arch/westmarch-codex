import matplotlib.pyplot as plt
import numpy as np

from src.paths import get_paths
from src.utils.util_files_functions import load_json_from_file


def plot_paragraph_length_distribution(paragraphs, min_length_to_show=10, title="Paragraph Lengths - The Hobbit"):
    """
    Plots paragraph lengths across the book with statistical reference lines.

    Parameters:
    - text: str - the full text of the book
    - min_length_to_show: int - ignore very short paragraphs (like empty lines or titles)
    - title: str - plot title
    """
    # Calculate length in words for each paragraph
    lengths = [len(p.split()) for p in paragraphs]

    # Filter out very short "paragraphs" (often chapter titles, separators, etc.)
    indices = np.arange(len(lengths))  # 0,1,2,... for filtered paragraphs

    # Basic statistics
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    p25, p50, p75, p95 = np.percentile(lengths, [25, 50, 75, 95])

    print(f"Number of meaningful paragraphs: {len(lengths)}")
    print(f"Mean length: {mean_len:.1f} words")
    print(f"Std deviation: {std_len:.1f} words")
    print(f"Median: {p50:.0f} words")
    print(f"25th–75th percentile: {p25:.0f} – {p75:.0f} words")
    print(f"95th percentile: {p95:.0f} words")

    # ────────────────────────────────────────────────────────────────
    # Plotting
    # ────────────────────────────────────────────────────────────────
    plt.figure(figsize=(14, 7), dpi=100)

    # Main scatter plot
    plt.scatter(indices, lengths, s=15, alpha=0.6, color="steelblue", label="Paragraph length (words)")

    # Reference lines
    plt.axhline(mean_len, color="darkred", linestyle="--", linewidth=2.2, label=f"Mean = {mean_len:.1f}")

    plt.axhline(mean_len + std_len, color="orange", linestyle=":", linewidth=1.8, label=f"Mean + 1σ = {mean_len + std_len:.1f}")
    plt.axhline(mean_len - std_len, color="orange", linestyle=":", linewidth=1.8, label=f"Mean - 1σ = {mean_len - std_len:.1f}")

    plt.axhline(mean_len + 2 * std_len, color="darkgreen", linestyle="--", linewidth=1.4, alpha=0.7, label=f"Mean + 2σ = {mean_len + 2 * std_len:.1f}")
    plt.axhline(mean_len - 2 * std_len, color="darkgreen", linestyle="--", linewidth=1.4, alpha=0.7)

    # Optional: median line
    plt.axhline(p50, color="purple", linestyle="-", linewidth=1.6, alpha=0.6, label=f"Median = {p50:.0f}")

    plt.title(title, fontsize=14, pad=12)
    plt.xlabel("Paragraph index (filtered)", fontsize=12)
    plt.ylabel("Length in words", fontsize=12)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Limit y-axis if there are crazy outliers
    plt.ylim(0, np.percentile(lengths, 99.5) * 1.3)

    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=10)
    plt.tight_layout()

    plt.show()


# ────────────────────────────────────────────────────────────────
# Example usage:
# ────────────────────────────────────────────────────────────────
# with open("the_hobbit.txt", encoding="utf-8") as f:
#     hobbit_text = f.read()
#
# plot_paragraph_length_distribution(hobbit_text,
#                                    min_length_to_show=8,
#                                    title="Paragraph Lengths across The Hobbit")


def main():
    paths = get_paths()

    chapters = load_json_from_file(paths.FILE_BOOK_00_PROCESSED)

    # for chapter in chapters:
    #     plot_paragraph_length_distribution(paragraphs=chapter.get("paragraphs"))
    chapter = chapters[1]
    print(f"Chapter: {chapter.get('chapter_num')}")
    plot_paragraph_length_distribution(paragraphs=chapter.get("paragraphs"))


if __name__ == "__main__":
    main()
