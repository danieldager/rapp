"""
sWuggy Score Calculation - Analysis Examples

This module provides example code for calculating same-voice and cross-voice
sWuggy scores after running evaluate.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl


def load_results(parquet_path: str):
    """Load the evaluated swuggy metadata with log probabilities."""
    return pl.read_parquet(parquet_path)


def _get_pos_neg_pairs(
    df: pl.DataFrame, prob_column: str, group_by: list
) -> pl.DataFrame:
    """Helper: Extract positive/negative probability pairs."""
    pairs = df.group_by(group_by).agg(
        [
            pl.col(prob_column).filter(pl.col("positive")).first().alias("pos_prob"),
            pl.col(prob_column).filter(~pl.col("positive")).first().alias("neg_prob"),
        ]
    )
    return pairs.filter(
        pl.col("pos_prob").is_not_null() & pl.col("neg_prob").is_not_null()
    )


def calculate_same_voice_accuracy(
    df: pl.DataFrame, prob_column: str = "log_prob"
) -> float:
    """
    Within-voice accuracy: compare pos vs neg for each (word_id, voice) pair.
    Returns proportion where pos > neg.
    """
    pairs = _get_pos_neg_pairs(df, prob_column, ["word_id", "voice"])
    if len(pairs) == 0:
        return 0.0

    # Count pairs where positive beats negative
    correct = (pairs["pos_prob"] > pairs["neg_prob"]).sum()
    total = len(pairs)
    return correct / total


def calculate_cross_voice_accuracy(
    df: pl.DataFrame, prob_column: str = "log_prob"
) -> float:
    """
    Cross-voice accuracy: for each word_id, compare each positive against all negatives.
    Returns average proportion of negatives beaten by each positive.
    """
    word_accuracies = []
    for word_id in df["word_id"].unique():
        word_df = df.filter(pl.col("word_id") == word_id)
        pos_probs = word_df.filter(pl.col("positive"))[prob_column].to_list()
        neg_probs = word_df.filter(~pl.col("positive"))[prob_column].to_list()

        if pos_probs and neg_probs:
            # Average score across all positives for this word
            scores = [sum(p > n for n in neg_probs) / len(neg_probs) for p in pos_probs]
            word_accuracies.append(sum(scores) / len(scores))

    return sum(word_accuracies) / len(word_accuracies) if word_accuracies else 0.0


def calculate_per_voice_accuracy(
    df: pl.DataFrame, prob_column: str = "log_prob"
) -> dict:
    """Calculate same-voice accuracy broken down by individual voices."""
    per_voice = {}
    for voice in df["voice"].unique():
        voice_df = df.filter(pl.col("voice") == voice)
        pairs = _get_pos_neg_pairs(voice_df, prob_column, ["word_id"])
        if len(pairs) > 0:
            per_voice[voice] = (pairs["pos_prob"] > pairs["neg_prob"]).mean()
    return per_voice


def analyze_by_word_length(df: pl.DataFrame, prob_column: str = "log_prob"):
    """Analyze accuracy by phone length bins."""
    df = df.with_columns(pl.col("phones").str.len_chars().alias("phone_length"))
    bins = [(0, 3), (3, 5), (5, 7), (7, 100)]

    for min_len, max_len in bins:
        subset = df.filter(
            (pl.col("phone_length") > min_len) & (pl.col("phone_length") <= max_len)
        )
        if len(subset) > 0:
            acc = calculate_same_voice_accuracy(subset, prob_column)
            print(
                f"  {min_len+1}-{max_len:>3d} phones: {acc:.3f} ({len(subset)} samples)"
            )


def plot_swuggy_scores(results_paths: dict, output_path: Path):
    """
    Create bar graphs comparing same-voice and cross-voice scores across models.
    Shows both unnormalized and length-normalized versions.
    """
    import numpy as np

    models = []
    same_voice_scores = []
    cross_voice_scores = []
    same_voice_norm_scores = []
    cross_voice_norm_scores = []

    for model_name, path in results_paths.items():
        df = pl.read_parquet(path)
        models.append(model_name)
        same_voice_scores.append(calculate_same_voice_accuracy(df, "log_prob"))
        cross_voice_scores.append(calculate_cross_voice_accuracy(df, "log_prob"))
        same_voice_norm_scores.append(
            calculate_same_voice_accuracy(df, "log_prob_norm")
        )
        cross_voice_norm_scores.append(
            calculate_cross_voice_accuracy(df, "log_prob_norm")
        )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.2

    # Plot bars
    ax.bar(
        x - 1.5 * width, same_voice_scores, width, label="Same-Voice", color="#1f77b4"
    )
    ax.bar(
        x - 0.5 * width,
        same_voice_norm_scores,
        width,
        label="Same-Voice (Norm)",
        color="#aec7e8",
    )
    ax.bar(
        x + 0.5 * width, cross_voice_scores, width, label="Cross-Voice", color="#ff7f0e"
    )
    ax.bar(
        x + 1.5 * width,
        cross_voice_norm_scores,
        width,
        label="Cross-Voice (Norm)",
        color="#ffbb78",
    )

    # Add value labels on bars
    for i, (sv, svn, cv, cvn) in enumerate(
        zip(
            same_voice_scores,
            same_voice_norm_scores,
            cross_voice_scores,
            cross_voice_norm_scores,
        )
    ):
        ax.text(
            i - 1.5 * width,
            sv + 0.01,
            f"{sv:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            i - 0.5 * width,
            svn + 0.01,
            f"{svn:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            i + 0.5 * width,
            cv + 0.01,
            f"{cv:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.text(
            i + 1.5 * width,
            cvn + 0.01,
            f"{cvn:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Styling
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "sWuggy Lexical Accuracy (Unnormalized vs Length-Normalized)",
        fontsize=14,
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.0)
    ax.axhline(
        y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance"
    )
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

    plt.tight_layout()

    # Ensure directory exists and save
    if output_path is None:
        output_path = (
            Path(__file__).parent.parent.parent / "figures" / "swuggy_scores.png"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to {output_path}")


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    figures_dir = project_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    path = project_root / "metadata" / "swuggy_evaluated.parquet"
    df = load_results(str(path))

    print("\n" + "=" * 60)
    print("SWUGGY LEXICAL ACCURACY ANALYSIS")
    print("=" * 60)

    # Calculate all metrics
    for label, col in [
        ("UNNORMALIZED", "log_prob"),
        ("LENGTH-NORMALIZED", "log_prob_norm"),
    ]:
        print(f"\n{label}:")
        sv = calculate_same_voice_accuracy(df, col)
        cv = calculate_cross_voice_accuracy(df, col)
        print(f"  Same-Voice:  {sv:.4f}")
        print(f"  Cross-Voice: {cv:.4f}")

        print(f"\n  Per-Voice:")
        for voice, acc in sorted(calculate_per_voice_accuracy(df, col).items()):
            print(f"    {voice:<12s} {acc:.3f}")

    print(f"\n  By Phone Length:")
    analyze_by_word_length(df, "log_prob")

    # Generate plot
    print(f"\nGenerating plot...")
    plot_swuggy_scores(
        {"GPT2": str(path)}, output_path=figures_dir / "swuggy_scores.png"
    )
    print(f"\n{'='*60}\n")
