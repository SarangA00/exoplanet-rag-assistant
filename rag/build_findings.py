"""
Step 1 of the RAG pipeline.

Reads the *_summary.json files produced by eda_analysis.py, hz_classification.py,
and regression_analysis.py, and turns each into a short human-readable text
document. These .txt files are what gets embedded in step 2.
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FINDINGS_DIR = Path(__file__).resolve().parent / "findings"


def format_eda(data: dict) -> str:
    lines = []
    lines.append(
        f"Exploratory data analysis was run on {data['n_planets']} confirmed "
        f"exoplanets across {data['n_columns']} columns from the NASA "
        f"Planetary Systems Composite Parameters (PSCompPars) dataset."
    )

    completeness = data["column_completeness"]
    completeness_bits = [
        f"{col} ({info['pct_valid']}% valid, median {info['median']})"
        for col, info in completeness.items()
    ]
    lines.append(
        "Column completeness for key fields: " + "; ".join(completeness_bits) + "."
    )

    methods = data["discovery_method_counts"]
    top_methods = sorted(methods.items(), key=lambda kv: kv[1], reverse=True)
    method_bits = [f"{name} ({count})" for name, count in top_methods]
    lines.append(
        "Planets by discovery method, most to least common: "
        + ", ".join(method_bits)
        + "."
    )

    corr_bits = []
    for c in data["strongest_correlations"]:
        corr_bits.append(f"{c['col_a']} vs. {c['col_b']} (|r|={c['abs_corr']})")
    lines.append(
        "The strongest pairwise correlations found were: " + "; ".join(corr_bits) + "."
    )

    return "\n\n".join(lines)


def format_hz(data: dict) -> str:
    lines = []
    thresholds = data["thresholds"]
    lines.append(
        f"Habitable zone classification was run on {data['total_planets_classified']} "
        f"planets, using an insolation flux threshold of {thresholds['too_hot_above']} "
        f"S(earth) for 'too hot' and {thresholds['too_cold_below']} S(earth) for "
        f"'too cold'; planets in between are classified 'Habitable Zone', and "
        f"planets missing the needed data are 'Unknown'."
    )

    overall = data["overall_counts"]
    overall_bits = [f"{cls}: {count}" for cls, count in overall.items()]
    lines.append("Overall classification counts were: " + ", ".join(overall_bits) + ".")

    by_method = data["by_method"]
    method_names = sorted({row["discoverymethod"] for row in by_method})
    for method in method_names:
        rows = [r for r in by_method if r["discoverymethod"] == method]
        bits = [
            f"{r['hz_class']} {r['percentage']:.1f}% ({r['count']})" for r in rows
        ]
        lines.append(
            f"Among planets discovered via {method}: " + ", ".join(bits) + "."
        )

    return "\n\n".join(lines)


def format_regression(data: dict) -> str:
    lines = []

    a = data["part_a_mass_radius"]
    lines.append(
        f"A power-law fit of planet radius to planet mass across {a['n_planets']} "
        f"planets found an exponent (alpha) of {a['power_law_exponent_alpha']:.3f} "
        f"with intercept {a['intercept_c']:.3f} and R-squared of {a['r2']:.3f}. "
        f"{a['interpretation']}"
    )

    b = data["part_b_stellar_regression"]
    coef_bits = [
        f"{c['Feature']} ({c['Coefficient']:.3f})"
        for c in b["standardized_coefficients"]
    ]
    lines.append(
        f"A regression of planet radius ('{b['target']}') on host-star features "
        f"({', '.join(b['features'])}) achieved test R-squared of "
        f"{b['test_r2']:.3f} and test MAE of {b['test_mae']:.3f}. "
        f"Standardized coefficients: " + ", ".join(coef_bits) + ". "
        f"{b['interpretation']}"
    )

    return "\n\n".join(lines)


SOURCES = [
    ("eda_output/eda_summary.json", "eda_findings.txt", format_eda),
    ("hz_output/hz_summary.json", "hz_findings.txt", format_hz),
    ("regression_output/regression_summary.json", "regression_findings.txt", format_regression),
]


def main():
    FINDINGS_DIR.mkdir(exist_ok=True)

    for json_rel_path, out_filename, formatter in SOURCES:
        json_path = BASE_DIR / json_rel_path
        with open(json_path) as f:
            data = json.load(f)

        text = formatter(data)

        out_path = FINDINGS_DIR / out_filename
        out_path.write_text(text)
        print(f"Wrote {out_path.relative_to(BASE_DIR)} ({len(text)} chars)")


if __name__ == "__main__":
    main()
