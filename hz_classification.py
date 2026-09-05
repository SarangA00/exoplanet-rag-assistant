"""
hz_classification.py
─────────────────────
Module 3 of the exoplanet analysis pipeline: Habitable Zone Classification.

WHAT THIS DOES
--------------
Classifies every exoplanet in the dataset as "Too Hot", "Habitable Zone",
"Too Cold", or "Unknown" based on the Kopparapu et al. (2013) insolation
flux thresholds, then compares how that classification breaks down between
the two dominant detection methods (Transit vs Radial Velocity).

HOW TO RUN IT
-------------
From the command line, inside your project folder:

    python hz_classification.py

By default it looks for the CSV in the same place the notebook did. To
point it at a different file (e.g. a train/test split later on), pass
--csv:

    python hz_classification.py --csv /path/to/other_file.csv --outdir results/

WHAT IT PRODUCES (in --outdir)
-------------------------------
- hz_classification.png   : the grouped bar chart
- hz_summary.csv          : per-method, per-class counts and percentages
- hz_summary.json         : same info, in JSON (easy for a RAG/LLM pipeline
                             to read later)
"""

import argparse          # lets us accept command-line options like --csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")    # so this works with no display (e.g. run from terminal)
import matplotlib.pyplot as plt


# ── Config: the classification thresholds, kept as named constants ──────────
# (Kopparapu et al. 2013 conservative habitable zone bounds, in units of
#  Earth insolation S⊕)
HZ_TOO_HOT_ABOVE = 1.1
HZ_TOO_COLD_BELOW = 0.36

CATEGORIES = ["Too Hot", "Habitable Zone", "Too Cold", "Unknown"]
METHODS_TO_COMPARE = ["Transit", "Radial Velocity"]


def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the raw NASA PSCompPars CSV into a DataFrame."""
    print(f"Loading data from {csv_path} …")
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    print(f"  Loaded {len(df):,} planets")
    return df


def classify_habitable_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an 'hz_class' column to df, labeling each planet as
    Too Hot / Habitable Zone / Too Cold / Unknown based on pl_insol.
    """
    conditions = [
        df["pl_insol"] > HZ_TOO_HOT_ABOVE,
        (df["pl_insol"] >= HZ_TOO_COLD_BELOW) & (df["pl_insol"] <= HZ_TOO_HOT_ABOVE),
        df["pl_insol"] < HZ_TOO_COLD_BELOW,
        df["pl_insol"].isna(),
    ]
    df = df.copy()
    df["hz_class"] = np.select(conditions, CATEGORIES, default="Unknown")
    return df


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter down to the two dominant detection methods and compute,
    for each method, what percentage of its planets fall in each
    hz_class category.
    """
    filtered = df[df["discoverymethod"].isin(METHODS_TO_COMPARE)]

    counts = (
        filtered.groupby(["discoverymethod", "hz_class"])
        .size()
        .reset_index(name="count")
    )
    counts["percentage"] = counts.groupby("discoverymethod")["count"].transform(
        lambda x: x / x.sum() * 100
    )
    return counts


def plot_comparison(counts: pd.DataFrame, out_path: Path) -> None:
    """Save the grouped bar chart comparing Transit vs Radial Velocity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"Transit": "steelblue", "Radial Velocity": "darkorange"}

    x = np.arange(len(CATEGORIES))
    width = 0.35

    for i, method in enumerate(METHODS_TO_COMPARE):
        method_data = counts[counts["discoverymethod"] == method]
        percentages = [
            method_data.loc[method_data["hz_class"] == cat, "percentage"].values[0]
            if cat in method_data["hz_class"].values else 0
            for cat in CATEGORIES
        ]
        ax.bar(x + i * width, percentages, width, label=method,
               color=colors[method], alpha=0.85)

    ax.set_xlabel("Habitable Zone Classification", fontsize=12)
    ax.set_ylabel("Percentage of Planets (%)", fontsize=12)
    ax.set_title("Habitable Zone Classification by Discovery Method",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(CATEGORIES)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def save_results(df: pd.DataFrame, counts: pd.DataFrame, outdir: Path) -> None:
    """Write standardized outputs: a CSV and a JSON summary."""
    counts.to_csv(outdir / "hz_summary.csv", index=False)

    summary = {
        "thresholds": {
            "too_hot_above": HZ_TOO_HOT_ABOVE,
            "too_cold_below": HZ_TOO_COLD_BELOW,
        },
        "total_planets_classified": int(len(df)),
        "overall_counts": df["hz_class"].value_counts().to_dict(),
        "by_method": counts.to_dict(orient="records"),
    }
    with open(outdir / "hz_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  saved hz_summary.csv and hz_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Habitable Zone classification module")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path.home() / "Downloads" / "PSCompPars_2026.06.30_11.22.37.csv",
        help="Path to the PSCompPars CSV file",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("hz_output"),
        help="Directory to save outputs to",
    )
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    df = load_data(args.csv)
    df = classify_habitable_zone(df)

    print("\nOverall classification counts:")
    print(df["hz_class"].value_counts().to_string())

    counts = summarize_by_method(df)
    print("\nBy discovery method:")
    print(counts.to_string(index=False))

    plot_comparison(counts, args.outdir / "hz_classification.png")
    save_results(df, counts, args.outdir)

    print(f"\n✓ Done. Outputs saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
