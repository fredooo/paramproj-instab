#!/usr/bin/env python3
"""Generate expanded training time table."""

from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent

DATASETS = ["mnist", "fmnist", "blobs", "har"]
DATASET_LABELS = {"mnist": "MNIST", "fmnist": "Fashion", "blobs": "Blobs", "har": "HAR"}

MODEL_FILES = {
    "MLP-small": "nn_MLP_h512_n3_nojac.csv",
    "MLP-small+J": "nn_MLP_h512_n3_jac10.0.csv",
    "SpecMLP-small": "nn_SpecMLP_h512_n3_nojac.csv",
    "SpecMLP-small+J": "nn_SpecMLP_h512_n3_jac10.0.csv",
    "MLP-large": "nn_MLP_h1024_n6_nojac.csv",
    "MLP-large+J": "nn_MLP_h1024_n6_jac10.0.csv",
}


def main():
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Training time in seconds (mean $\pm$ std, 10 runs, UMAP projection).}")
    lines.append(r"\label{tab:training-time}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{@{}l cccc@{}}")
    lines.append(r"\toprule")
    lines.append(r"Config & MNIST & Fashion & Blobs & HAR \\")
    lines.append(r"\midrule")

    for model_name, csv_file in MODEL_FILES.items():
        df = pd.read_csv(RESULTS_DIR / csv_file)
        df = df[df["projection"] == "umap"]

        cells = []
        for ds in DATASETS:
            df_ds = df[df["dataset"] == ds]
            m = df_ds["fit_time"].mean()
            s = df_ds["fit_time"].std()
            cells.append(f"${m:.1f} \\pm {s:.1f}$")

        lines.append(f"{model_name} & {' & '.join(cells)} \\\\")
        # Add midrule between small and large groups
        if model_name == "SpecMLP-small+J":
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    output_file = OUTPUT_DIR / "00-training-time-table.tex"
    output_file.write_text("\n".join(lines))
    print(f"Wrote {output_file}")


if __name__ == "__main__":
    main()
