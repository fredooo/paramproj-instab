#!/usr/bin/env python3
"""Aggregate experimental results and populate LaTeX comparison table."""

import math
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent

MODEL_FILES = {
    "MLP-small": "nn_MLP_h512_n3_nojac.csv",
    "MLP-small+J": "nn_MLP_h512_n3_jac10.0.csv",
    "MLP-large": "nn_MLP_h1024_n6_nojac.csv",
    "MLP-large+J": "nn_MLP_h1024_n6_jac10.0.csv",
}

# (dataset, projection, label) tuples for column ordering
COLUMNS = [
    ("mnist", "umap", "MNIST (UMAP)"),
    ("fmnist", "umap", "Fashion (UMAP)"),
    ("mnist", "tsne", "MNIST (t-SNE)"),
    ("fmnist", "tsne", "Fashion (t-SNE)"),
]

# Metrics for each section
SECTIONS = [
    ("Average MSE Loss (lower is better)", "test_loss"),
    ("Avg. Trustworthiness $T(k)$ with $k \\in \\{2, 4, 8, \\dots, n / 2\\}$ (higher is better)", "trust_p2"),
    ("Avg. Continuity $C(k)$ with $k \\in \\{2, 4, 8, \\dots, n / 2\\}$ (higher is better)", "cont_p2"),
    ("Mean Displacement (lower is better)", "D_dev"),
    ("Displacement Bias (lower is better)", "D_bias"),
    ("Average Nearest-Anchor Assignment Error (lower is better)", "E_NA"),
]


def _fmt3(x):
    """Format x to 3 significant figures, capped at 3 decimal places."""
    if x == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(x)))
    decimals = min(3, max(0, 2 - magnitude))
    return f"{x:.{decimals}f}"


def _strip_and_pad(s):
    """Strip trailing zeros after decimal point, pad with \\phantom{0} for alignment."""
    if "." not in s:
        return s
    stripped = s.rstrip("0")
    n_zeros = len(s) - len(stripped)
    stripped = stripped.rstrip(".")
    # Remove leading "0" from numbers like "0.123" -> ".123"
    if stripped.startswith("0."):
        stripped = stripped[1:]
    return stripped + r"\phantom{0}" * n_zeros


def format_value(mean, std, bold=False, underline=False):
    """Format mean ± std with 3 significant figures, removing trailing zeros and leading zero."""
    if pd.isna(mean) or pd.isna(std):
        return "--"
    mean_str = _strip_and_pad(_fmt3(mean))
    std_str = _strip_and_pad(_fmt3(std))

    result = f"{mean_str} $\\pm$ {std_str}"
    if bold:
        return f"\\textbf{{{result}}}"
    if underline:
        return f"\\underline{{{result}}}"
    return result


def load_and_aggregate():
    """Load CSVs and compute mean ± std for each metric, grouped by (dataset, projection)."""
    aggregated = {}

    metric_cols = ["test_loss", "trust_p2", "cont_p2", "D_dev", "D_bias", "E_NA"]

    for model_name, csv_file in MODEL_FILES.items():
        csv_path = RESULTS_DIR / csv_file
        df = pd.read_csv(csv_path)

        # Convert test_loss to numeric (handles "N/A")
        df["test_loss"] = pd.to_numeric(df["test_loss"], errors="coerce")

        # Group by (dataset, projection)
        stats = df.groupby(["dataset", "projection"])[metric_cols].agg(["mean", "std"])

        aggregated[model_name] = stats

    return aggregated


def generate_latex_table(aggregated):
    """Generate LaTeX table with aggregated results."""

    # Build two-line column headers using \shortstack
    col_headers = []
    for _, _, label in COLUMNS:
        # Split "MNIST (UMAP)" into two lines
        parts = label.split(" (")
        top = parts[0]
        bottom = "(" + parts[1]
        col_headers.append(f"\\shortstack{{\\textbf{{{top}}}\\\\\\textbf{{{bottom}}}}}")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\setlength{\tabcolsep}{1.1mm}")
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\textbf{Model} & " + " & ".join(col_headers) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\hline")
    for section_idx, (section_name, metric_col) in enumerate(SECTIONS):
        if section_idx > 0:
            lines.append(r"\hline")

        lines.append(f"\\multicolumn{{5}}{{c}}{{\\textit{{{section_name}}}}} \\\\\n\\hline")

        # Determine optimization direction
        is_lower_better = "lower is better" in section_name.lower()

        # Find best and worst value for each (dataset, projection) column
        best_models = {}  # (dataset, proj) -> model_name
        worst_models = {}  # (dataset, proj) -> model_name
        for dataset, proj, _ in COLUMNS:
            key = (dataset, proj)
            best_val = None
            best_model = None
            worst_val = None
            worst_model = None
            for model_name in MODEL_FILES.keys():
                stats = aggregated[model_name]
                if key in stats.index:
                    mean = stats.loc[key, (metric_col, "mean")]
                    if pd.notna(mean):
                        if best_val is None:
                            best_val = worst_val = mean
                            best_model = worst_model = model_name
                        else:
                            if (is_lower_better and mean < best_val) or (not is_lower_better and mean > best_val):
                                best_val = mean
                                best_model = model_name
                            if (is_lower_better and mean > worst_val) or (not is_lower_better and mean < worst_val):
                                worst_val = mean
                                worst_model = model_name
            if best_model:
                best_models[key] = best_model
            if worst_model:
                worst_models[key] = worst_model

        for model_name in MODEL_FILES.keys():
            stats = aggregated[model_name]

            row_values = []
            for dataset, proj, _ in COLUMNS:
                key = (dataset, proj)
                if key in stats.index:
                    mean = stats.loc[key, (metric_col, "mean")]
                    std = stats.loc[key, (metric_col, "std")]
                    is_best = best_models.get(key) == model_name
                    is_worst = worst_models.get(key) == model_name and not is_best
                    row_values.append(format_value(mean, std, bold=is_best, underline=is_worst))
                else:
                    row_values.append("--")

            row_parts = [f"\\texttt{{{model_name}}}"] + [f"& {val}" for val in row_values]
            lines.append(" ".join(row_parts) + r" \\")

    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Stability and quality metrics for MLP-based parametric projections "
        r"based on UMAP and t-SNE with different regularization strategies across datasets. "
        r"Bold values indicate best, underlined values worst performance per metric and dataset.}"
    )
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\vspace{-2em}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("Loading CSV files...")
    aggregated = load_and_aggregate()

    print("Generating LaTeX table...")
    latex_table = generate_latex_table(aggregated)

    output_file = OUTPUT_DIR / "00-comparison-table.tex"
    output_file.write_text(latex_table)
    print(f"Created {output_file}")


if __name__ == "__main__":
    main()
