import csv
import os
import time

import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform

from dataset_loaders import load_blobs_split, load_fmnist_split, load_har_split, load_mnist_split
from measures import (
    compute_stability_metrics,
    create_noisy_versions,
    metric_continuity_numba,
    metric_trustworthiness_numba,
    trustworthiness_continuity_powers_of_two,
)
from models import create_model, get_model_prefix, predict
from plotting.plot_all import plot_all
from projection_utils import tsne_setup, umap_setup
from train import evaluate_projection_model, train_projection_model
from typedefs import (
    DatasetConfig,
    DataSplit,
    ModelConfig,
    OutputDirs,
    ProjectionConfig,
    ProjectionContext,
    RunContext,
    TrainData,
    TrainingConfig,
)
from utils import centroid_representative_indices, plot_projection_data, set_seed

SEED = 777
N_RUNS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = "./output/results"
MODELS_DIR = "./output/models"
IMAGES_DIR = "./output/images"
OUTPUT_DIRS = OutputDirs(MODELS_DIR, IMAGES_DIR, RESULTS_DIR)

TRAINING_CONFIG = TrainingConfig(
    max_epochs=100,
    batch_size=256,
    lr=1e-3,
    patience=10,
)

DATASETS = [
    DatasetConfig("mnist", load_mnist_split, 784, (0.0, 1.0), 0.1689, 2000),
    DatasetConfig("fmnist", load_fmnist_split, 784, (0.0, 1.0), 0.1597, 2000),
    DatasetConfig("blobs", load_blobs_split, 10, None, 0.7371, 2000),
    DatasetConfig("har", load_har_split, 561, None, 0.1433, 2000),
]

PROJECTIONS = [
    ProjectionConfig("umap", umap_setup),
    ProjectionConfig("tsne", tsne_setup),
]

# use_spec (False=MLP, True=SpecMLP), hidden_dim, n_hidden, use_jac, lambda_jac
MODELS = [
    ModelConfig(False, 512, 3, False, 0.0),
    ModelConfig(True, 512, 3, False, 0.0),
    
    ModelConfig(False, 512, 3, True, 1.0),
    ModelConfig(True, 512, 3, True, 1.0),
    
    ModelConfig(False, 512, 3, True, 10.0),
    ModelConfig(True, 512, 3, True, 10.0),
    
    ModelConfig(False, 512, 3, True, 20.0),
    ModelConfig(True, 512, 3, True, 20.0),
    
    ModelConfig(False, 512, 3, True, 40.0),
    ModelConfig(True, 512, 3, True, 40.0),
    
    ModelConfig(False, 512, 3, True, 80.0),
    ModelConfig(True, 512, 3, True, 80.0),

    ModelConfig(False, 1024, 6, False, 0.0),
    ModelConfig(False, 1024, 6, True, 10.0),
    ModelConfig(False, 1024, 6, True, 20.0),
    ModelConfig(False, 1024, 6, True, 40.0),
    ModelConfig(False, 1024, 6, True, 80.0),
]


def compute_quality_metrics(D_high, Z_low, k=7):
    """Compute trustworthiness and continuity metrics.

    Parameters
    ----------
    D_high : ndarray, shape (n, n)
        High-dimensional distance matrix.
    Z_low : ndarray, shape (n, 2)
        Low-dimensional embeddings.
    k : int
        Number of neighbors for metrics.

    Returns
    -------
    dict with trust_p2, cont_p2 (aggregated powers-of-two), trust, cont (k=7).
    """
    D_low = squareform(pdist(Z_low, metric="euclidean"))

    # Aggregated powers-of-two metrics
    ks, trust_array, cont_array = trustworthiness_continuity_powers_of_two(D_high, D_low)
    trust_p2 = np.mean(trust_array)
    cont_p2 = np.mean(cont_array)

    # k=7 metrics
    trust = metric_trustworthiness_numba(D_high, D_low, k=k)
    cont = metric_continuity_numba(D_high, D_low, k=k)

    return {"trust_p2": trust_p2, "cont_p2": cont_p2, "trust": trust, "cont": cont}


def compute_projection_stability_metrics(reducer, proj_ctx):
    """Compute stability metrics for the projection method.

    Returns (metrics, Z_clusters, inference_time).
    """
    start = time.time()
    Z_clusters = [reducer.transform(Xn) for Xn in proj_ctx.X_noisy_per_class]
    inference_time = time.time() - start
    metrics = compute_stability_metrics(proj_ctx.Z_base, Z_clusters)
    return metrics, Z_clusters, inference_time


def compute_nn_metrics(model, proj_ctx, device):
    """Compute stability metrics for the neural network model.

    Returns (metrics, Z_clusters, Z_base, inference_time).
    """

    def project_fn(X):
        return predict(model, X, device=device)

    Z_base = project_fn(proj_ctx.X_base)
    start = time.time()
    Z_clusters = [project_fn(Xn) for Xn in proj_ctx.X_noisy_per_class]
    inference_time = time.time() - start
    metrics = compute_stability_metrics(Z_base, Z_clusters)
    return metrics, Z_clusters, Z_base, inference_time


def evaluate_projection(run_ctx, data, D_high_te, output_dirs):
    """Evaluate a projection method (with supports_transform=True) as a first-class model.

    Parameters
    ----------
    run_ctx : RunContext
    data : DataSplit
    D_high_te : ndarray
        Precomputed high-dimensional distance matrix.
    output_dirs : OutputDirs

    Returns
    -------
    tuple (row, ProjectionContext)
        row is None if projection doesn't support transform.
    """
    dataset_cfg = run_ctx.dataset_cfg
    projection_cfg = run_ctx.projection_cfg
    seed = run_ctx.seed
    path_prefix = os.path.join(output_dirs.models, f"{projection_cfg.name}_{dataset_cfg.name}_{seed}")

    # Setup projection
    reducer, Z_tr, Z_val, Z_te, supports_transform, fit_time = projection_cfg.setup(
        data.X_tr, data.y_tr, data.X_val, data.X_te, seed, path_prefix
    )

    # Plot train/val/test projections
    for Z, y, subset in [(Z_tr, data.y_tr, "train"), (Z_val, data.y_val, "val"), (Z_te, data.y_te, "test")]:
        filename = os.path.join(output_dirs.images, f"{projection_cfg.name}_{dataset_cfg.name}_{seed}_{subset}.png")
        subset_base_idxs = centroid_representative_indices(Z, y)
        subset_anchors = Z[subset_base_idxs]
        plot_projection_data(Z, y, filename, anchors=subset_anchors)

    # Compute projection quality metrics (trustworthiness & continuity on clean test data)
    quality = compute_quality_metrics(D_high_te, Z_te, k=7)

    # Select anchor points and create noisy samples
    base_idxs = centroid_representative_indices(Z_te, data.y_te)
    X_base = data.X_te[base_idxs]
    Z_base = Z_te[base_idxs]
    X_noisy_per_class = create_noisy_versions(
        X_base, dataset_cfg.sigma, dataset_cfg.n_samples, clip_bounds=dataset_cfg.clip_bounds
    )

    # Build projection context for NN evaluation
    proj_ctx = ProjectionContext(
        reducer=reducer,
        Z_tr=Z_tr,
        Z_val=Z_val,
        Z_te=Z_te,
        supports_transform=supports_transform,
        X_base=X_base,
        Z_base=Z_base,
        X_noisy_per_class=X_noisy_per_class,
    )

    # Only compute stability metrics and generate plots if projection supports transform
    row = None
    if supports_transform:
        stability, Z_clusters, inference_time = compute_projection_stability_metrics(reducer, proj_ctx)

        # Generate plots
        img_prefix = os.path.join(output_dirs.images, f"proj_{projection_cfg.name}_{dataset_cfg.name}_{seed}")
        plot_all(Z_clusters, Z_base, img_prefix)

        # Build result row
        row = {
            "dataset": dataset_cfg.name,
            "projection": projection_cfg.name,
            "run": seed,
            "test_loss": "N/A",
            "trust_p2": quality["trust_p2"],
            "cont_p2": quality["cont_p2"],
            "trust": quality["trust"],
            "cont": quality["cont"],
            "fit_time": fit_time,
            "inference_time": inference_time,
            **stability,
        }

    return row, proj_ctx


def load_or_train_model(model_cfg, run_ctx, train_data, training_cfg, device, models_dir):
    """Load cached model or train and save."""
    prefix = get_model_prefix(model_cfg)
    base_path = os.path.join(
        models_dir, f"{prefix}_{run_ctx.projection_cfg.name}_{run_ctx.dataset_cfg.name}_{run_ctx.seed}"
    )
    model_path = f"{base_path}.pt"
    metrics_path = f"{base_path}.csv"

    model = create_model(model_cfg, run_ctx.dataset_cfg.input_dim)

    if os.path.exists(model_path):
        print(f"        Loading cached model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
        # Read training_time from metrics CSV
        with open(metrics_path, "r") as f:
            reader = csv.DictReader(f)
            row = next(reader)
            training_time = float(row["training_time"])
    else:
        result = train_projection_model(
            model, train_data, device, training_cfg, use_jacobian=model_cfg.use_jac, lambda_jac=model_cfg.lambda_jac
        )
        model = result.model
        training_time = result.training_time
        torch.save(model.state_dict(), model_path)

        with open(metrics_path, "w", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["best_val_loss", "final_train_loss", "epochs", "early_stopped", "training_time"]
            )
            w.writeheader()
            w.writerow(
                {
                    "best_val_loss": result.best_val_loss,
                    "final_train_loss": result.final_train_loss,
                    "epochs": result.epochs,
                    "early_stopped": result.early_stopped,
                    "training_time": result.training_time,
                }
            )

        print(
            f"        Saved model: {model_path} "
            f"(epochs={result.epochs}, val_loss={result.best_val_loss:.6f}, "
            f"time={result.training_time:.1f}s)"
        )

    return model, training_time


def evaluate_nn_model(run_ctx, model_cfg, data, proj_ctx, D_high_te, training_cfg, device, output_dirs):
    """Evaluate an NN model for one (dataset, projection, model, seed) configuration."""
    train_data = TrainData(data.X_tr, proj_ctx.Z_tr, data.X_val, proj_ctx.Z_val)
    model, fit_time = load_or_train_model(model_cfg, run_ctx, train_data, training_cfg, device, output_dirs.models)
    test_loss = evaluate_projection_model(model, data.X_te, proj_ctx.Z_te, device=device)

    # Compute NN quality metrics (trustworthiness & continuity on clean test data)
    Z_te_nn = predict(model, data.X_te, device=device)
    nn_quality = compute_quality_metrics(D_high_te, Z_te_nn, k=7)

    # Compute NN-side stability metrics
    nn_stability, Z_clusters_nn, Z_base_nn, inference_time = compute_nn_metrics(model, proj_ctx, device)

    # Plot NN test predictions with anchors
    test_img = os.path.join(
        output_dirs.images,
        f"{get_model_prefix(model_cfg)}_{run_ctx.projection_cfg.name}_{run_ctx.dataset_cfg.name}_{run_ctx.seed}_test.png",
    )
    plot_projection_data(Z_te_nn, data.y_te, test_img, anchors=Z_base_nn)

    # Generate plots for NN model
    img_prefix = os.path.join(
        output_dirs.images,
        f"{get_model_prefix(model_cfg)}_{run_ctx.projection_cfg.name}_{run_ctx.dataset_cfg.name}_{run_ctx.seed}",
    )
    plot_all(Z_clusters_nn, Z_base_nn, img_prefix)

    # Build result row
    row = {
        "dataset": run_ctx.dataset_cfg.name,
        "projection": run_ctx.projection_cfg.name,
        "run": run_ctx.seed,
        "test_loss": test_loss,
        "trust_p2": nn_quality["trust_p2"],
        "cont_p2": nn_quality["cont_p2"],
        "trust": nn_quality["trust"],
        "cont": nn_quality["cont"],
        "fit_time": fit_time,
        "inference_time": inference_time,
        **nn_stability,
    }
    return row


def write_results_csv(rows_by_prefix, results_dir):
    """Write CSV files for each model/projection prefix.

    Uses uniform column names - the filename determines the model/method type.
    """
    os.makedirs(results_dir, exist_ok=True)

    fieldnames = [
        "dataset",
        "projection",
        "run_id",
        "run",
        "test_loss",
        "trust_p2",
        "cont_p2",
        "trust",
        "cont",
        "fit_time",
        "inference_time",
        "D_dev",
        "D_bias",
        "E_NA",
    ]

    for prefix, rows in rows_by_prefix.items():
        if not rows:
            continue

        path = os.path.join(results_dir, f"{prefix}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        print(f"Wrote {path} ({len(rows)} rows)")


def run_experiment(datasets, projections, models, seeds, training_cfg, output_dirs=OUTPUT_DIRS, device=DEVICE):
    """Run experiment with given configuration. Returns rows_by_prefix dict."""
    os.makedirs(output_dirs.models, exist_ok=True)
    os.makedirs(output_dirs.images, exist_ok=True)

    # Initialize result storage for both NN models and projections
    rows_by_prefix = {get_model_prefix(m): [] for m in models}
    for p in projections:
        rows_by_prefix[f"proj_{p.name}"] = []

    for run_id, seed in enumerate(seeds):
        set_seed(seed)
        print(f"Run {run_id + 1}/{len(seeds)} (seed={seed})")

        for dataset_cfg in datasets:
            # Load data once, wrap in DataSplit
            data = DataSplit(*dataset_cfg.load_fn(seed))

            # Precompute high-dimensional distance matrix (reused across projections/models)
            print(f"  Computing distance matrix for {dataset_cfg.name}...")
            D_high_te = squareform(pdist(data.X_te, metric="euclidean"))

            for projection_cfg in projections:
                print(f"    Evaluating projection: {projection_cfg.name}")
                run_ctx = RunContext(dataset_cfg, projection_cfg, seed)

                # Evaluate projection (returns None for row if not supports_transform)
                proj_row, proj_ctx = evaluate_projection(run_ctx, data, D_high_te, output_dirs)

                # Store projection results if available
                if proj_row is not None:
                    proj_row["run_id"] = run_id
                    rows_by_prefix[f"proj_{projection_cfg.name}"].append(proj_row)

                # Evaluate each NN model
                for model_cfg in models:
                    print(f"      Evaluating model: {get_model_prefix(model_cfg)}")
                    nn_row = evaluate_nn_model(
                        run_ctx, model_cfg, data, proj_ctx, D_high_te, training_cfg, device, output_dirs
                    )
                    nn_row["run_id"] = run_id
                    rows_by_prefix[get_model_prefix(model_cfg)].append(nn_row)

    return rows_by_prefix


def main():
    seeds = [SEED + r for r in range(N_RUNS)]

    rows = run_experiment(
        datasets=DATASETS,
        projections=PROJECTIONS,
        models=MODELS,
        seeds=seeds,
        training_cfg=TRAINING_CONFIG,
    )
    # Single write at end; incremental writes complex due to per-seed data deps (data, D_high_te, proj_ctx)
    write_results_csv(rows, RESULTS_DIR)


if __name__ == "__main__":
    main()
