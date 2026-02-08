#!/usr/bin/env python3
"""Aggregate experimental results and populate LaTeX comparison table."""

from pathlib import Path

import pandas as pd

# Model → CSV file mapping
MODEL_FILES = {
    "MLP-small": "nn_MLP_h512_n3_nojac.csv",
    "MLP-large": "nn_MLP_h1024_n6_nojac.csv",
    "MLP-small+J": "nn_MLP_h512_n3_jac10.0.csv",
    "MLP-small+S": "nn_SpecMLP_h512_n3_nojac.csv",
    "MLP-small+JS": "nn_SpecMLP_h512_n3_jac10.0.csv",
}

# Dataset normalization
DATASET_ORDER = ["blobs", "har", "mnist", "fmnist"]
DATASET_LABELS = {
    "blobs": "Blobs",
    "har": "HAR",
    "mnist": "MNIST",
    "fmnist": "Fashion",
}

# Metrics for each section
SECTIONS = [
    ("Average Loss (lower is better)", "test_loss"),
    ("Average Trustworthiness $T(k)$ with $k \\in \\{2, 4, 8, \\dots, n / 2\\}$ (higher is better)", "trust_p2"),
    ("Average Continuity $C(k)$ with $k \\in \\{2, 4, 8, \\dots, n / 2\\}$ (higher is better)", "cont_p2"),
    ("Average Displacement (lower is better)", "D_dev"),
    ("Displacement Bias (lower is better)", "D_bias"),
    ("Average Anchor Assignment Error (lower is better)", "E_NA"),
]


def format_value(mean, std, bold=False):
    """Format mean ± std, removing trailing zeros and leading zero."""
    if pd.isna(mean) or pd.isna(std):
        return "--"
    mean_str = f"{mean:.3f}".rstrip("0").rstrip(".")
    std_str = f"{std:.3f}".rstrip("0").rstrip(".")

    # Remove leading "0" from numbers like "0.123" -> ".123"
    if mean_str.startswith("0."):
        mean_str = mean_str[1:]
    if std_str.startswith("0."):
        std_str = std_str[1:]

    result = f"{mean_str} $\\pm$ {std_str}"
    return f"\\textbf{{{result}}}" if bold else result


def load_and_aggregate():
    """Load CSVs and compute mean ± std for each metric."""
    results_dir = Path("./output/results")
    aggregated = {}

    # Columns to aggregate
    metric_cols = ["test_loss", "trust_p2", "cont_p2", "D_dev", "D_bias", "E_NA"]

    for model_name, csv_file in MODEL_FILES.items():
        csv_path = results_dir / csv_file
        df = pd.read_csv(csv_path)

        # Filter for UMAP projection only
        df_umap = df[df["projection"] == "umap"].copy()

        # Convert test_loss to numeric (handles "N/A")
        df_umap["test_loss"] = pd.to_numeric(df_umap["test_loss"], errors="coerce")

        # Select only metric columns and group by dataset
        stats = df_umap.groupby("dataset")[metric_cols].agg(["mean", "std"])

        aggregated[model_name] = stats

    return aggregated


def generate_latex_table(aggregated):
    """Generate LaTeX table with aggregated results."""

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\setlength{\tabcolsep}{1.1mm}")
    lines.append(r"\small")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcccc}")
    # lines.append(r'\hline')
    lines.append(r"\textbf{Model} & \textbf{Blobs} & \textbf{HAR} & \textbf{MNIST} & \textbf{Fashion-MNIST} \\")
    lines.append(r"\hline")
    for section_idx, (section_name, metric_col) in enumerate(SECTIONS):
        if section_idx > 0:
            lines.append(r"\hline")

        lines.append(f"\\multicolumn{{5}}{{c}}{{\\textit{{{section_name}}}}} \\\\\n\\hline")

        # Determine optimization direction
        is_lower_better = "lower is better" in section_name.lower()

        # Find best value for each dataset
        best_models = {}  # dataset -> model_name
        for dataset in DATASET_ORDER:
            best_val = None
            best_model = None
            for model_name in MODEL_FILES.keys():
                stats = aggregated[model_name]
                if dataset in stats.index:
                    mean = stats.loc[dataset, (metric_col, "mean")]
                    # Skip UMAP test_loss or NaN values
                    if model_name == "UMAP" and metric_col == "test_loss":
                        continue
                    if pd.notna(mean):
                        if best_val is None:
                            best_val = mean
                            best_model = model_name
                        elif (is_lower_better and mean < best_val) or (not is_lower_better and mean > best_val):
                            best_val = mean
                            best_model = model_name
            if best_model:
                best_models[dataset] = best_model

        for model_name in MODEL_FILES.keys():
            stats = aggregated[model_name]

            row_values = []
            for dataset in DATASET_ORDER:
                if dataset in stats.index:
                    # Handle UMAP test_loss specially
                    if model_name == "UMAP" and metric_col == "test_loss":
                        row_values.append("--")
                    else:
                        mean = stats.loc[dataset, (metric_col, "mean")]
                        std = stats.loc[dataset, (metric_col, "std")]
                        # Bold if this is the best model for this dataset
                        is_best = best_models.get(dataset) == model_name
                        row_values.append(format_value(mean, std, bold=is_best))
                else:
                    row_values.append("--")

            row_parts = [f"{model_name}"] + [f"& {val}" for val in row_values]
            lines.append(" ".join(row_parts) + r" \\")

    # lines.append(r'\hline')
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Stability and quality metrics for MLP-based parametric projections "
        r"nased on UMAP with different regularization strategies across datasets. "
        r"Bold values indicate best performance per metric and dataset.}"
    )
    lines.append(r"\label{tab:comparison}")
    lines.append(r"\vspace{-1.5em}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    print("Loading CSV files...")
    aggregated = load_and_aggregate()

    print("Generating LaTeX table...")
    latex_table = generate_latex_table(aggregated)

    output_file = Path("00-comparison-table.tex")
    output_file.write_text(latex_table)
    print(f"Created {output_file}")


if __name__ == "__main__":
    main()
