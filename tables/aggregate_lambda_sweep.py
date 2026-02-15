#!/usr/bin/env python3
"""Generate lambda sensitivity tables for MLP-small, MLP-large, SpecMLP-small."""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "output" / "results"
OUTPUT_DIR = Path(__file__).resolve().parent

print(RESULTS_DIR, OUTPUT_DIR)

DATASETS = ["mnist", "fmnist", "blobs", "har"]
DATASET_LABELS = {"mnist": "MNIST", "fmnist": "Fashion", "blobs": "Blobs", "har": "HAR"}

METRICS = [
    ("Test Loss (MSE)", "test_loss", True),
    ("Trustworthiness", "trust_p2", False),
    ("Continuity", "cont_p2", False),
    ("Mean Displacement ($\\bar{D}_{\\mathrm{dev}}$)", "D_dev", True),
    ("Displacement Bias ($\\bar{D}_{\\mathrm{bias}}$)", "D_bias", True),
    ("Nearest-Anchor Error ($\\bar{E}_{\\mathrm{NA}}$)", "E_NA", True),
]

CONFIGS = [
    {
        "name": "MLP-small",
        "prefix": "nn_MLP_h512_n3",
        "lambdas": [0, 1, 10, 20, 40, 80],
        "label": "tab:lambda-sensitivity",
        "caption": "Sensitivity to Jacobian regularization strength $\\lambda$ for MLP-small (512 hidden, 3 layers, UMAP, 10 runs). Best per dataset in \\textbf{bold}.",
    },
    {
        "name": "MLP-large",
        "prefix": "nn_MLP_h1024_n6",
        "lambdas": [0, 10, 20, 40, 80],
        "label": "tab:lambda-sensitivity-large",
        "caption": "Sensitivity to $\\lambda$ for MLP-large (1024 hidden, 6 layers, UMAP, 10 runs). Best per dataset in \\textbf{bold}.",
    },
    {
        "name": "SpecMLP-small",
        "prefix": "nn_SpecMLP_h512_n3",
        "lambdas": [0, 1, 10, 20, 40, 80],
        "label": "tab:lambda-sensitivity-spec",
        "caption": "Sensitivity to $\\lambda$ for SpecMLP-small (512 hidden, 3 layers, spectral norm, UMAP, 10 runs). Best per dataset in \\textbf{bold}.",
    },
]


def format_value(mean, std, bold=False):
    if pd.isna(mean) or pd.isna(std):
        return "--"
    mean_str = f"{mean:.3f}".rstrip("0").rstrip(".")
    std_str = f"{std:.3f}".rstrip("0").rstrip(".")
    if mean_str.startswith("0."):
        mean_str = mean_str[1:]
    if std_str.startswith("0."):
        std_str = std_str[1:]
    result = f"${mean_str} \\pm {std_str}$"
    if bold:
        return f"\\mathbf{{{mean_str} \\pm {std_str}}}".replace(result, "") or f"$\\mathbf{{{mean_str} \\pm {std_str}}}$"
    return result


def csv_filename(prefix, lam):
    if lam == 0:
        return f"{prefix}_nojac.csv"
    lam_str = f"{lam:.1f}" if isinstance(lam, float) else f"{float(lam):.1f}"
    return f"{prefix}_jac{lam_str}.csv"


def generate_table(config):
    lambdas = config["lambdas"]
    prefix = config["prefix"]
    ncols = len(lambdas)

    # Load data
    data = {}
    for lam in lambdas:
        fname = csv_filename(prefix, lam)
        fpath = RESULTS_DIR / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        df = pd.read_csv(fpath)
        df = df[df["projection"] == "umap"]
        df["test_loss"] = pd.to_numeric(df["test_loss"], errors="coerce")
        data[lam] = df

    # Build table
    col_headers = " & ".join([f"$\\lambda{{=}}{l}$" for l in lambdas])
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{config['caption']}}}")
    lines.append(f"\\label{{{config['label']}}}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(f"\\begin{{tabular}}{{@{{}}l {'c' * ncols}@{{}}}}")
    lines.append(r"\toprule")
    lines.append(f"Dataset & {col_headers} \\\\")
    lines.append(r"\midrule")

    for mi, (metric_name, metric_col, lower_better) in enumerate(METRICS):
        if mi > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{{metric_name}}}}} \\\\")

        for ds in DATASETS:
            # Compute means per lambda
            means = {}
            stds = {}
            for lam in lambdas:
                if lam not in data:
                    continue
                df_ds = data[lam][data[lam]["dataset"] == ds]
                if len(df_ds) == 0:
                    continue
                means[lam] = df_ds[metric_col].mean()
                stds[lam] = df_ds[metric_col].std()

            # Find best
            best_lam = None
            if means:
                if lower_better:
                    best_lam = min(means, key=means.get)
                else:
                    best_lam = max(means, key=means.get)

            # Format cells
            cells = []
            for lam in lambdas:
                if lam in means:
                    is_best = (lam == best_lam)
                    m, s = means[lam], stds[lam]
                    mean_str = f"{m:.3f}".rstrip("0").rstrip(".")
                    std_str = f"{s:.3f}".rstrip("0").rstrip(".")
                    if mean_str.startswith("0."):
                        mean_str = mean_str[1:]
                    if std_str.startswith("0."):
                        std_str = std_str[1:]
                    if is_best:
                        cells.append(f"$\\mathbf{{{mean_str} \\pm {std_str}}}$")
                    else:
                        cells.append(f"${mean_str} \\pm {std_str}$")
                else:
                    cells.append("--")

            lines.append(f"{DATASET_LABELS[ds]}   & {' & '.join(cells)} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def main():
    all_tables = []
    for config in CONFIGS:
        print(f"Generating table for {config['name']}...")
        table = generate_table(config)
        all_tables.append(table)
        print(f"  -> {config['label']}")

    output_file = OUTPUT_DIR / "00-lambda-tables.tex"
    output_file.write_text("\n\n".join(all_tables))
    print(f"\nWrote {output_file}")


if __name__ == "__main__":
    main()
