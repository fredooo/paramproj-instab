#!/usr/bin/env python3
"""Generate UMAP vs tSNE comparison table."""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent

DATASETS = ["mnist", "fmnist", "blobs", "har"]
DATASET_LABELS = {"mnist": "MNIST", "fmnist": "Fashion", "blobs": "Blobs", "har": "HAR"}

MODEL_FILES = {
    "MLP-small": "nn_MLP_h512_n3_nojac.csv",
    "MLP-small+J": "nn_MLP_h512_n3_jac10.0.csv",
    "MLP-large": "nn_MLP_h1024_n6_nojac.csv",
    "MLP-large+J": "nn_MLP_h1024_n6_jac10.0.csv",
}

METRICS = [
    ("Mean Displacement ($\\bar{D}_{\\mathrm{dev}}$)", "D_dev"),
    ("Test Loss (MSE)", "test_loss"),
]


def fmt(mean, std):
    mean_str = f"{mean:.3f}".rstrip("0").rstrip(".")
    std_str = f"{std:.3f}".rstrip("0").rstrip(".")
    if mean_str.startswith("0."):
        mean_str = mean_str[1:]
    if std_str.startswith("0."):
        std_str = std_str[1:]
    return f"${mean_str} \\pm {std_str}$"


def main():
    # Load all data
    data = {}
    for model_name, csv_file in MODEL_FILES.items():
        df = pd.read_csv(RESULTS_DIR / csv_file)
        df["test_loss"] = pd.to_numeric(df["test_loss"], errors="coerce")
        data[model_name] = df

    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{UMAP vs tSNE comparison across models and datasets (mean $\pm$ std, 10 runs). "
        r"Stability improvements from Jacobian regularization are consistent across projection methods.}"
    )
    lines.append(r"\label{tab:tsne-comparison}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\begin{tabular}{@{}l cc cc cc cc@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{2}{c}{MLP-small} & \multicolumn{2}{c}{MLP-small+J} & "
        r"\multicolumn{2}{c}{MLP-large} & \multicolumn{2}{c}{MLP-large+J} \\"
    )
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}")
    lines.append(r"Dataset & UMAP & tSNE & UMAP & tSNE & UMAP & tSNE & UMAP & tSNE \\")
    lines.append(r"\midrule")

    for mi, (metric_name, metric_col) in enumerate(METRICS):
        if mi > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\textit{{{metric_name}}}}} \\\\")

        for ds in DATASETS:
            cells = []
            for model_name in MODEL_FILES:
                df = data[model_name]
                for proj in ["umap", "tsne"]:
                    df_sub = df[(df["dataset"] == ds) & (df["projection"] == proj)]
                    if len(df_sub) == 0:
                        cells.append("--")
                    else:
                        cells.append(fmt(df_sub[metric_col].mean(), df_sub[metric_col].std()))
            lines.append(f"{DATASET_LABELS[ds]} & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    output_file = OUTPUT_DIR / "00-tsne-table.tex"
    output_file.write_text("\n".join(lines))
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
