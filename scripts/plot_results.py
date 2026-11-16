"""Draw baseline vs fine-tuned bar charts from the CSVs in results/.

Reads:
    results/igbo1_vs_nllb1.3b.csv
    results/v1_vs_nllb600m.csv   (uses the 600-1600 sentence range)

Writes:
    results/igbo1_vs_nllb1.3b.png
    results/v1_vs_nllb600m.png

Values marked TBD in a CSV are left out of the chart (the missing bar is
labelled "baseline TBD"). Scores are AfriCOMET system scores.

Usage:
    python scripts/plot_results.py
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Colors: validated categorical slots 1 and 2 on a light surface.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
BASELINE_COLOR = "#2a78d6"   # blue: base NLLB model
FINETUNED_COLOR = "#eb6834"  # orange: LangBridge model

BAR_WIDTH = 0.36
GAP = 0.02  # small surface gap between the two bars of a pair


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None  # "TBD" and blanks


def _fmt4(value):
    """Round the exact CSV string half-up to 4 decimals (same rule as the docs)."""
    try:
        return str(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):
        return None


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def bar_pairs(directions, baseline, finetuned, labels, title, subtitle, out_path):
    """Draw one bar pair per direction.

    baseline / finetuned are lists of the raw CSV strings. Values that are not
    numbers (for example "TBD") are skipped.
    """
    baseline_text = [_fmt4(v) for v in baseline]
    finetuned_text = [_fmt4(v) for v in finetuned]
    baseline = [_to_float(v) for v in baseline]
    finetuned = [_to_float(v) for v in finetuned]
    fig, ax = plt.subplots(figsize=(max(6.4, 1.45 * len(directions) + 2), 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axes(ax)

    xs = range(len(directions))
    top = max(v for v in baseline + finetuned if v is not None)

    for x, b, f, bt, ft in zip(xs, baseline, finetuned, baseline_text, finetuned_text):
        for value, text, offset, color in (
            (b, bt, -(BAR_WIDTH / 2 + GAP / 2), BASELINE_COLOR),
            (f, ft, +(BAR_WIDTH / 2 + GAP / 2), FINETUNED_COLOR),
        ):
            if value is None:
                ax.text(
                    x + offset, 0.01, "baseline\nTBD", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_SECONDARY, style="italic",
                )
                continue
            ax.bar(x + offset, value, width=BAR_WIDTH, color=color, edgecolor=SURFACE, linewidth=1)
            ax.text(
                x + offset, value + top * 0.015, text, ha="center", va="bottom",
                fontsize=8, color=TEXT_PRIMARY,
            )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(directions, fontsize=9, color=TEXT_PRIMARY)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("AfriCOMET score (higher is better)", fontsize=9, color=TEXT_SECONDARY)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=FINETUNED_COLOR),
    ]
    legend = ax.legend(
        handles, labels, loc="upper left", bbox_to_anchor=(0, 1.02), ncol=2,
        frameon=False, fontsize=9,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_PRIMARY)

    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=TEXT_PRIMARY, fontweight="bold")
    fig.text(0.01, 0.935, subtitle, ha="left", va="top", fontsize=8.5, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_igbo1():
    df = pd.read_csv(RESULTS / "igbo1_vs_nllb1.3b.csv", dtype=str)
    bar_pairs(
        directions=list(df["direction"].str.replace(" -> ", " → ")),
        baseline=list(df["baseline_score"]),
        finetuned=list(df["finetuned_score"]),
        labels=["NLLB-200-distilled-1.3B (base)", "LangBridge AI Igbo 1.0"],
        title="LangBridge AI Igbo 1.0 vs its base model",
        subtitle="AfriCOMET, v3 test split (1,693 sentences).\n"
        "Caveat: 89% of test pairs also appear in the training data.",
        out_path=RESULTS / "igbo1_vs_nllb1.3b.png",
    )


def plot_v1():
    df = pd.read_csv(RESULTS / "v1_vs_nllb600m.csv", dtype=str)
    df = df[(df["range_start"] == "600") & (df["range_end"] == "1600")]
    order = [
        "Igbo -> English", "Hausa -> English", "Yoruba -> English",
        "English -> Igbo", "English -> Hausa", "English -> Yoruba",
    ]
    base = df[df["model"] == "NLLB-200-distilled-600M"].set_index("direction")
    v1 = df[df["model"] == "LangBridge v1"].set_index("direction")
    bar_pairs(
        directions=[d.replace(" -> ", " →\n") for d in order],
        baseline=[base.loc[d, "score"] for d in order],
        finetuned=[v1.loc[d, "score"] for d in order],
        labels=["NLLB-200-distilled-600M (base)", "LangBridge v1"],
        title="LangBridge v1 vs its base model (archived)",
        subtitle="AfriCOMET, sentences 600–1600 of a 2,000-sentence sample per direction. "
        "v1 scored below its base model in every direction with a known baseline.",
        out_path=RESULTS / "v1_vs_nllb600m.png",
    )


if __name__ == "__main__":
    plot_igbo1()
    plot_v1()
