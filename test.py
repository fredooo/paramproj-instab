"""Quick smoke test - runs one dataset, one projection, one model."""
from dataset_loaders import load_mnist_split
from main import (
    run_experiment, write_results_csv,
    DatasetConfig, ProjectionConfig, ModelConfig, ExperimentConfig,
)
from typedefs import OutputDirs, TrainingConfig
from projection_utils import umap_setup

# Minimal configuration
TEST_DATASETS = [
    DatasetConfig("mnist", load_mnist_split, 784, (0.0, 1.0)),
]

TEST_PROJECTIONS = [
    ProjectionConfig("umap", umap_setup),
]

TEST_MODELS = [
    ModelConfig(False, 256, 2, False, 0.0),
]

TEST_TRAINING_CONFIG = TrainingConfig(
    max_epochs=10,
    batch_size=256,
    lr=1e-3,
    patience=10,
)

TEST_EXPERIMENT_CONFIG = ExperimentConfig(
    sigma=0.15,
    n_samples=500,       # Reduced for speed
    sigmas_cq=[0.1],     # Fewer sigma values
    n_samples_cq=200,    # Reduced for speed
)


def main():
    print("Running smoke test...")
    rows = run_experiment(
        datasets=TEST_DATASETS,
        projections=TEST_PROJECTIONS,
        models=TEST_MODELS,
        seeds=[777],  # Single seed
        training_cfg=TEST_TRAINING_CONFIG,
        experiment_cfg=TEST_EXPERIMENT_CONFIG,
        output_dirs=OutputDirs("./test_output/models", "./test_output/images", "./test_output/results"),
    )
    write_results_csv(rows, "./test_output/results")
    print("Smoke test complete!")


if __name__ == "__main__":
    main()
