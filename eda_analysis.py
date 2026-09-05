"""
eda_analysis.py
─────────────────
Module 1 of the exoplanet analysis pipeline: Exploratory Data Analysis.

WHAT THIS DOES
--------------
Data overview, distributions (linear + log), discovery-method breakdown,
discoveries per year, correlation matrix, radius-period diagram, and a
pairplot for the core numeric columns.

HOW TO RUN IT
-------------
    python3 eda_analysis.py
    python3 eda_analysis.py --csv /path/to/other_file.csv --outdir results/

WHAT IT PRODUCES (in --outdir)
-------------------------------
- core_summary.csv              : completeness/summary stats per column
- 01_distributions.png          : histograms of the 9 core columns
- 02_discovery_methods.png      : bar chart of confirmed planets by method
- 03_discoveries_per_year.png   : stacked bar chart over time
- 04_correlation_matrix.png     : Pearson correlation heatmap
- 05_radius_period_transit.png  : radius vs period scatter (Transit planets)
- 06_pairplot.png               : pairwise relationships, core columns
- eda_summary.json              : key numbers in a format a RAG step can read
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["font.family"] = "DejaVu Sans"   # so ⊕, ☉, ₁₀ render correctly

CORE_NUM = [
    "pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt", "pl_insol",
    "pl_orbeccen", "st_teff", "st_rad", "st_mass",
]
LABELS = {
    "pl_orbper":   "Orbital Period (days)",
    "pl_rade":     "Planet Radius (R\u2295)",
    "pl_bmasse":   "Planet Mass (M\u2295)",
    "pl_eqt":      "Equil. Temperature (K)",
    "pl_insol":    "Insolation Flux (S\u2295)",
    "pl_orbeccen": "Eccentricity",
    "st_teff":     "Stellar T_eff (K)",
    "st_rad":      "Stellar Radius (R\u2609)",
    "st_mass":     "Stellar Mass (M\u2609)",
}
# Columns that benefit from log-scale (heavily right-skewed quantities)
LOG_COLS = {"pl_orbper", "pl_rade", "pl_bmasse", "pl_insol", "st_rad", "st_mass"}
PAIR_COLS = ["pl_orbper", "pl_rade", "pl_bmasse", "pl_eqt", "st_teff", "st_mass"]


def load_data(csv_path: Path) -> pd.DataFrame:
    print("Loading data …")
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    print(f"  {df.shape[0]:,} planets  \u00d7  {df.shape[1]} columns")
    return df


def compute_core_summary(df: pd.DataFrame) -> pd.DataFrame:
    print("\n\u2500\u2500 Core column completeness \u2500\u2500")
    summary = pd.DataFrame({
        "n_valid":   df[CORE_NUM].notna().sum(),
        "pct_valid": (df[CORE_NUM].notna().mean() * 100).round(1),
        "median":    df[CORE_NUM].median().round(3),
        "mean":      df[CORE_NUM].mean().round(3),
        "std":       df[CORE_NUM].std().round(3),
        "min":       df[CORE_NUM].min().round(3),
        "max":       df[CORE_NUM].max().round(3),
    })
    print(summary.to_string())
    return summary


def plot_distributions(df: pd.DataFrame, out_path: Path) -> None:
    """
    Histograms of the 9 core columns. For log-scaled columns, only strictly
    positive values are used (rather than skipping the log entirely if any
    value is <= 0) so skewed columns like insolation flux still display on
    a readable log scale instead of being crushed against the axis.
    """
    print("\nPlotting distributions …")
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for ax, col in zip(axes, CORE_NUM):
        data = df[col].dropna()
        use_log = col in LOG_COLS
        if use_log:
            data = data[data > 0]  # drop non-positive values, keep the rest log-scaled
            plot_data = np.log10(data)
        else:
            plot_data = data
            # For non-log columns with extreme outliers (e.g. eccentricity,
            # equilibrium temperature), clip the x-axis to the 1st-99th
            # percentile range so the bulk of the distribution is visible.
            lo, hi = plot_data.quantile(0.01), plot_data.quantile(0.99)

        ax.hist(plot_data, bins=60, color="steelblue", edgecolor="none", alpha=0.85)

        lbl = LABELS[col]
        ax.set_xlabel(f"log\u2081\u2080({lbl})" if use_log else lbl, fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.set_title(lbl, fontsize=10, fontweight="bold")

        if not use_log:
            ax.set_xlim(lo, hi)

        med = data.median()
        ax.axvline(np.log10(med) if use_log else med,
                   color="crimson", linewidth=1.4, linestyle="--",
                   label=f"median={med:.2g}")
        ax.legend(fontsize=8)

    fig.suptitle("Core Column Distributions \u2014 NASA PSCompPars", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_discovery_methods(df: pd.DataFrame, out_path: Path) -> pd.Series:
    method_counts = df["discoverymethod"].value_counts()
    print("\n\u2500\u2500 Discovery method breakdown \u2500\u2500")
    print(method_counts.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("muted", len(method_counts))
    method_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="none")
    ax.set_title("Confirmed Exoplanets by Discovery Method", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Planets")
    ax.tick_params(axis="x", rotation=30)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height()):,}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")
    return method_counts


def plot_discoveries_per_year(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    year_method = (df.groupby(["disc_year", "discoverymethod"])
                     .size()
                     .unstack(fill_value=0))
    keep = year_method.sum().nlargest(6).index
    year_method[keep].plot(kind="bar", stacked=True, ax=ax, width=0.85,
                            colormap="tab10", edgecolor="none")
    ax.set_title("Exoplanet Discoveries per Year (top methods)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Planets Discovered")
    ax.legend(title="Method", fontsize=8, loc="upper left")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_correlation_matrix(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    print("Computing correlation matrix …")
    corr_df = df[CORE_NUM].copy()
    for col in LOG_COLS:
        mask = corr_df[col] > 0
        corr_df.loc[mask, col] = np.log10(corr_df.loc[mask, col])
        corr_df.loc[~mask, col] = np.nan

    corr = corr_df.rename(columns=LABELS).corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, square=True, linewidths=0.5,
        annot_kws={"size": 9}, ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix (log-scaled where appropriate)",
                 fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")
    return corr


def plot_radius_period(df: pd.DataFrame, out_path: Path) -> None:
    transit = df[df["discoverymethod"] == "Transit"].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(
        transit["pl_orbper"], transit["pl_rade"],
        c=transit["pl_eqt"], cmap="plasma",
        s=8, alpha=0.5, linewidths=0,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Orbital Period (days)", fontsize=11)
    ax.set_ylabel("Planet Radius (R\u2295)", fontsize=11)
    ax.set_title("Transit Planets: Radius vs Period (colour = T_eq)", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Equilibrium Temp (K)", fontsize=10)

    ax.axhline(1.0,  color="gray", linewidth=0.8, linestyle=":", alpha=0.7, label="1 R\u2295 (Earth)")
    ax.axhline(3.9,  color="gray", linewidth=0.8, linestyle="--", alpha=0.7, label="~4 R\u2295 (Neptune)")
    ax.axhline(11.2, color="gray", linewidth=0.8, linestyle="-", alpha=0.7, label="~11 R\u2295 (Jupiter)")
    ax.axvline(365,  color="goldenrod", linewidth=0.9, linestyle="--", alpha=0.8, label="1 yr")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


def plot_pairplot(df: pd.DataFrame, out_path: Path) -> None:
    print("Generating pairplot (may take ~30 s) …")
    pair_df = (df[PAIR_COLS + ["discoverymethod"]]
                 .dropna(subset=PAIR_COLS)
                 .copy())

    for col in LOG_COLS:
        if col in pair_df.columns:
            pair_df[col] = np.where(pair_df[col] > 0, np.log10(pair_df[col]), np.nan)

    pair_df = pair_df.sample(min(2000, len(pair_df)), random_state=42)
    top_methods = pair_df["discoverymethod"].value_counts().nlargest(4).index
    pair_df["Method"] = pair_df["discoverymethod"].where(
        pair_df["discoverymethod"].isin(top_methods), other="Other"
    )
    pair_df = pair_df.rename(columns={c: f"log\u2081\u2080({LABELS[c]})" if c in LOG_COLS else LABELS[c]
                                       for c in PAIR_COLS})

    g = sns.pairplot(
        pair_df.drop(columns=["discoverymethod"]),
        hue="Method", plot_kws={"alpha": 0.4, "s": 12, "linewidths": 0},
        diag_kind="kde", corner=True,
    )
    g.figure.suptitle("Pairplot \u2014 Core Exoplanet Parameters (log-scaled)", y=1.01, fontsize=12)
    g.figure.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(g.figure)
    print(f"  saved {out_path.name}")


def save_results(df: pd.DataFrame, summary: pd.DataFrame,
                  method_counts: pd.Series, corr: pd.DataFrame, outdir: Path) -> None:
    output = {
        "n_planets": int(len(df)),
        "n_columns": int(df.shape[1]),
        "column_completeness": {
            col: {
                "pct_valid": float(summary.loc[col, "pct_valid"]),
                "median": float(summary.loc[col, "median"]),
            }
            for col in CORE_NUM
        },
        "discovery_method_counts": method_counts.to_dict(),
        "strongest_correlations": (
            corr.abs().unstack()
            .reset_index()
            .rename(columns={"level_0": "col_a", "level_1": "col_b", 0: "abs_corr"})
            .query("col_a != col_b")               # drop self-correlations (always 1.0)
            .assign(pair=lambda d: d.apply(lambda r: tuple(sorted([r.col_a, r.col_b])), axis=1))
            .drop_duplicates(subset="pair")         # each pair only counted once
            .sort_values("abs_corr", ascending=False)
            .head(6)[["col_a", "col_b", "abs_corr"]]
            .round(3)
            .to_dict(orient="records")
        ),
    }
    with open(outdir / "eda_summary.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("  saved eda_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Exploratory data analysis module")
    parser.add_argument(
        "--csv", type=Path,
        default=Path.home() / "Downloads" / "PSCompPars_2026.06.30_11.22.37.csv",
        help="Path to the PSCompPars CSV file",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("eda_output"),
        help="Directory to save outputs to",
    )
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    df = load_data(args.csv)

    summary = compute_core_summary(df)
    summary.to_csv(args.outdir / "core_summary.csv")

    plot_distributions(df, args.outdir / "01_distributions.png")
    method_counts = plot_discovery_methods(df, args.outdir / "02_discovery_methods.png")
    plot_discoveries_per_year(df, args.outdir / "03_discoveries_per_year.png")
    corr = plot_correlation_matrix(df, args.outdir / "04_correlation_matrix.png")
    plot_radius_period(df, args.outdir / "05_radius_period_transit.png")
    plot_pairplot(df, args.outdir / "06_pairplot.png")

    save_results(df, summary, method_counts, corr, args.outdir)

    print(f"\n\u2713 Done. Outputs saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
