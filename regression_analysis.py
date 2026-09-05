"""
regression_analysis.py
────────────────────────
Module 2 of the exoplanet analysis pipeline: Regression Analysis.

WHAT THIS DOES
--------------
Part A — Fits a power-law relation between planet mass and planet radius
          (in log-log space: log R = alpha * log M + c).
Part B — Fits a multiple linear regression predicting planet radius from
          three stellar properties (temperature, radius, mass), to test
          whether the HOST STAR determines the planet's size.

HOW TO RUN IT
-------------
    python regression_analysis.py
    python regression_analysis.py --csv /path/to/other_file.csv --outdir results/

WHAT IT PRODUCES (in --outdir)
-------------------------------
- A_mass_radius.png          : scatter + fit line + residuals (Part A)
- B_stellar_regression.png   : predicted-vs-actual + feature importance (Part B)
- regression_summary.json    : all the numeric results from both parts,
                                in a format a later RAG step can read directly
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'



def load_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading data from {csv_path} …")
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    print(f"  Loaded {len(df):,} planets")
    return df


# ─────────────────────────────────────────────────────────────────────────
# Part A — Mass–Radius power law
# ─────────────────────────────────────────────────────────────────────────

def fit_mass_radius(df: pd.DataFrame) -> dict:
    """
    Fit R ∝ M^alpha in log-log space using linear regression.
    Returns a dict with the fitted parameters and the data used,
    so the plotting function doesn't have to redo any of the work.
    """
    mr = df[["pl_bmasse", "pl_rade", "discoverymethod"]].dropna()
    mr = mr[(mr["pl_bmasse"] > 0) & (mr["pl_rade"] > 0)]

    log_mass = np.log10(mr["pl_bmasse"])
    log_radius = np.log10(mr["pl_rade"])

    model = LinearRegression()
    model.fit(log_mass.values.reshape(-1, 1), log_radius.values)

    alpha = float(model.coef_[0])
    intercept = float(model.intercept_)
    predictions = model.predict(log_mass.values.reshape(-1, 1))
    r2 = float(r2_score(log_radius, predictions))
    residuals = log_radius.values - predictions

    print(f"\n[Part A] Power-law exponent  alpha = {alpha:.3f}")
    print(f"[Part A] Intercept                c = {intercept:.3f}")
    print(f"[Part A] R²                         = {r2:.3f}")
    print(f"[Part A] Fitted relation: R ∝ M^{alpha:.3f}")

    return {
        "mr_data": mr,
        "log_mass": log_mass,
        "log_radius": log_radius,
        "residuals": residuals,
        "alpha": alpha,
        "intercept": intercept,
        "r2": r2,
    }


def plot_mass_radius(fit: dict, out_path: Path) -> None:
    """Scatter + fit line (left) and residuals vs mass (right)."""
    mr = fit["mr_data"]
    log_mass = fit["log_mass"]
    alpha = fit["alpha"]
    c = fit["intercept"]
    r2 = fit["r2"]
    residuals = fit["residuals"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Scatter + fit ──
    ax = axes[0]
    colors = {"Transit": "steelblue", "Radial Velocity": "darkorange"}
    for method, grp in mr.groupby("discoverymethod"):
        col = colors.get(method, "gray")
        alpha_pt = 0.5 if method in colors else 0.2
        ax.scatter(grp["pl_bmasse"], grp["pl_rade"], s=6, alpha=alpha_pt,
                   color=col, label=method if method in colors else "_nolegend_")

    x_line = np.logspace(log_mass.min(), log_mass.max(), 300)
    y_line = 10 ** (alpha * np.log10(x_line) + c)
    ax.plot(x_line, y_line, color="crimson", linewidth=2,
            label=f"Fit: R \u221d M^{alpha:.2f}  (R\u00b2={r2:.2f})")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Planet Mass (M\u2295)", fontsize=11)
    ax.set_ylabel("Planet Radius (R\u2295)", fontsize=11)
    ax.set_title("Mass\u2013Radius Relation (log\u2013log)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    for mass, radius, label in [(1, 1, "Earth"), (17.1, 3.9, "Neptune"), (317.8, 11.2, "Jupiter")]:
        ax.axvline(mass, color="gray", linewidth=0.7, linestyle=":")
        ax.axhline(radius, color="gray", linewidth=0.7, linestyle=":")
        ax.annotate(label, xy=(mass, radius), xytext=(4, 4),
                    textcoords="offset points", fontsize=7, color="gray")

    # ── Residuals ──
    ax2 = axes[1]
    ax2.scatter(log_mass, residuals, s=5, alpha=0.3, color="steelblue")
    ax2.axhline(0, color="crimson", linewidth=1.5, linestyle="--")
    ax2.set_xlabel("log\u2081\u2080(Planet Mass / M\u2295)", fontsize=11)
    ax2.set_ylabel("Residual (log\u2081\u2080 R)", fontsize=11)
    ax2.set_title("Residuals vs Mass", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────
# Part B — Stellar properties → planet radius
# ─────────────────────────────────────────────────────────────────────────

FEATURES = ["st_teff", "st_rad", "st_mass"]
TARGET = "pl_rade"


def fit_stellar_regression(df: pd.DataFrame, random_state: int = 42) -> dict:
    """
    Multiple linear regression: does host-star temperature/radius/mass
    predict planet radius? (Spoiler from the meeting: not much — R² ~ 0.20)
    """
    reg = df[FEATURES + [TARGET]].dropna()
    reg = reg[reg[TARGET] > 0]

    X = reg[FEATURES].copy()
    X["st_rad"] = np.log10(X["st_rad"])
    X["st_mass"] = np.log10(X["st_mass"])
    y = np.log10(reg[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    coef_df = pd.DataFrame({
        "Feature": ["st_teff", "log st_rad", "log st_mass"],
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", key=abs, ascending=False)

    print(f"\n[Part B] Test R²  = {r2:.3f}")
    print(f"[Part B] Test MAE = {mae:.3f} (log\u2081\u2080 R\u2295 units)")
    print("[Part B] Standardised coefficients:")
    print(coef_df.to_string(index=False))

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "coef_df": coef_df,
        "r2": r2,
        "mae": mae,
    }


def plot_stellar_regression(fit: dict, out_path: Path) -> None:
    """Predicted vs actual (left) and feature-importance bar chart (right)."""
    y_test = fit["y_test"]
    y_pred = fit["y_pred"]
    coef_df = fit["coef_df"]
    r2 = fit["r2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.scatter(y_test, y_pred, s=8, alpha=0.3, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, color="crimson", linewidth=1.5, linestyle="--", label="Perfect fit")
    ax.set_xlabel("Actual log\u2081\u2080(Planet Radius)", fontsize=11)
    ax.set_ylabel("Predicted log\u2081\u2080(Planet Radius)", fontsize=11)
    ax.set_title(f"Predicted vs Actual  (R\u00b2={r2:.2f})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    bar_colors = ["steelblue" if v >= 0 else "salmon" for v in coef_df["Coefficient"]]
    ax2.barh(coef_df["Feature"], coef_df["Coefficient"], color=bar_colors, edgecolor="none")
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Standardised Coefficient", fontsize=11)
    ax2.set_title("Feature Importance\n(stellar \u2192 planet radius)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path.name}")


# ─────────────────────────────────────────────────────────────────────────
# Standardized output + orchestration
# ─────────────────────────────────────────────────────────────────────────

def save_results(mr_fit: dict, stellar_fit: dict, outdir: Path) -> None:
    summary = {
        "part_a_mass_radius": {
            "power_law_exponent_alpha": mr_fit["alpha"],
            "intercept_c": mr_fit["intercept"],
            "r2": mr_fit["r2"],
            "n_planets": int(len(mr_fit["mr_data"])),
            "interpretation": (
                "Planet radius scales with mass to the power of alpha. "
                "Departure from a single power law at high mass reflects "
                "the transition from rocky planets to compressible gas giants."
            ),
        },
        "part_b_stellar_regression": {
            "features": FEATURES,
            "target": TARGET,
            "test_r2": stellar_fit["r2"],
            "test_mae": stellar_fit["mae"],
            "standardized_coefficients": stellar_fit["coef_df"].to_dict(orient="records"),
            "interpretation": (
                "Low R2 indicates host-star properties (temperature, radius, "
                "mass) are weak predictors of planet radius on their own; "
                "other factors (e.g. distance from host star) likely dominate."
            ),
        },
    }
    with open(outdir / "regression_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved regression_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Regression analysis module")
    parser.add_argument(
        "--csv", type=Path,
        default=Path.home() / "Downloads" / "PSCompPars_2026.06.30_11.22.37.csv",
        help="Path to the PSCompPars CSV file",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("regression_output"),
        help="Directory to save outputs to",
    )
    args = parser.parse_args()
    args.outdir.mkdir(exist_ok=True, parents=True)

    df = load_data(args.csv)

    mr_fit = fit_mass_radius(df)
    plot_mass_radius(mr_fit, args.outdir / "A_mass_radius.png")

    stellar_fit = fit_stellar_regression(df)
    plot_stellar_regression(stellar_fit, args.outdir / "B_stellar_regression.png")

    save_results(mr_fit, stellar_fit, args.outdir)

    print(f"\n\u2713 Done. Outputs saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
