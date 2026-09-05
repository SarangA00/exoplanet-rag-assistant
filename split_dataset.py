"""
split_dataset.py
─────────────────
Splits the raw PSCompPars exoplanet CSV into a training set and a held-out
test set.
HOW TO RUN IT
-------------
    python3 split_dataset.py
    python3 split_dataset.py --csv /path/to/other_file.csv --test-size 0.2

WHAT IT PRODUCES
----------------
- train.csv   : ~80% of the planets (default), used for all development
- test.csv    : ~20% of the planets, held out and untouched
- split_summary.json : record of how the split was made (for reproducibility)
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: Path) -> pd.DataFrame:
    print(f"Loading data from {csv_path} …")
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    print(f"  Loaded {len(df):,} planets")
    return df


def split_data(df: pd.DataFrame, test_size: float, random_state: int) -> tuple:
    """
    Randomly splits df into (train_df, test_df).
    random_state is fixed so the split is reproducible — running this
    script twice on the same data always gives the same two groups.
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    return train_df, test_df


def save_outputs(train_df: pd.DataFrame, test_df: pd.DataFrame,
                  outdir: Path, test_size: float, random_state: int) -> None:
    outdir.mkdir(exist_ok=True, parents=True)

    train_path = outdir / "train.csv"
    test_path = outdir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    summary = {
        "total_planets": int(len(train_df) + len(test_df)),
        "train_planets": int(len(train_df)),
        "test_planets": int(len(test_df)),
        "test_size_fraction": test_size,
        "random_state": random_state,
        "note": (
            "test.csv should remain unseen during module development and "
            "tuning. Only use it to validate final analysis/chat behavior."
        ),
    }
    with open(outdir / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  saved {train_path.name} ({len(train_df):,} rows)")
    print(f"  saved {test_path.name} ({len(test_df):,} rows)")
    print(f"  saved split_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Train/test split for exoplanet data")
    parser.add_argument(
        "--csv", type=Path,
        default=Path.home() / "Downloads" / "PSCompPars_2026.06.30_11.22.37.csv",
        help="Path to the raw PSCompPars CSV file",
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("split_data"),
        help="Directory to save train.csv / test.csv to",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to hold out as the test set (default 0.2 = 20%%)",
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed, fixed for reproducibility",
    )
    args = parser.parse_args()

    df = load_data(args.csv)
    train_df, test_df = split_data(df, args.test_size, args.random_state)
    save_outputs(train_df, test_df, args.outdir, args.test_size, args.random_state)

    print(f"\n✓ Done. Outputs saved to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
