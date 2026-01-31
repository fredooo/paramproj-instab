# **Stability of Parametric Projections under Input Perturbations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Uses: venv](https://img.shields.io/badge/Environment-venv-blue)](https://docs.python.org/3/library/venv.html)
[![Paper](https://img.shields.io/badge/paper-EuroVA%202026-red)](PAPER_LINK_PLACEHOLDER)
[![OSF Project](https://img.shields.io/badge/OSF-View%20Project-lightgrey)](OSF_LINK_PLACEHOLDER)

## Key Features

* Compares parametric projection methods (neural networks) against non-parametric baselines (UMAP, t-SNE) for dimensionality reduction.
* Measures projection stability under Gaussian input perturbations using novel metrics (D_dev, D_bias, Q, C_Q).
* Evaluates projection quality via trustworthiness and continuity metrics.
* Includes MLP and spectrally-normalized MLP (SpecMLP) architectures with optional Jacobian regularization.
* Provides multiple visualization types: scatter plots, KDE contours, local PCA ellipses, anchor lines, and Voronoi tessellations.

## Requirements

* **Python** >= 3.10 ([Python 3.10.x](https://www.python.org/downloads/release/python-3100/))
* **Virtual environment**: [venv](https://docs.python.org/3/library/venv.html)

## How to Run

### 1. Setup Environment

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Experiments

```bash
# Run full experiment pipeline
python3 main.py
```

This will:
- Load datasets (MNIST, FashionMNIST, Blobs, HAR)
- Fit UMAP and t-SNE projections
- Train neural network models to mimic projections
- Compute stability and quality metrics
- Generate visualization outputs

### 3. Smoke Test

```bash
# Verify installation with a quick test
python3 test.py
```

## File Overview

| File Name           | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| `main.py`           | Main experiment pipeline: datasets, projections, models, metrics.           |
| `config.py`         | Configuration dataclasses for datasets, projections, models, experiments.   |
| `models.py`         | MLP and SpecMLP neural network architectures.                               |
| `measures.py`       | Stability metrics (D_dev, D_bias, Q, C_Q) and quality metrics (trust/cont). |
| `train.py`          | Training loop for projection-mimicking neural networks.                     |
| `utils.py`          | Utility functions (seeding, centroid selection, plotting).                  |
| `test.py`           | Smoke test for verifying installation.                                      |
| `dataset_loaders/`  | Dataset loading functions for MNIST, FashionMNIST, HAR, Blobs.              |
| `projection_utils/` | UMAP and t-SNE setup utilities.                                             |
| `plotting/`         | Visualization modules (scatter, KDE, PCA ellipses, Voronoi, anchor lines).  |

## Metrics

### Stability Metrics
- **D_dev**: Mean displacement from anchor (noise-induced drift)
- **D_bias**: Systematic bias (distance between mean noisy projection and anchor)
- **Q**: Variance normalized by noise level (amplification measure)
- **C_Q**: Coefficient of variation of Q across noise levels

### Quality Metrics
- **Trustworthiness**: Measures false neighbors in low-dimensional space
- **Continuity**: Measures missing neighbors in low-dimensional space

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
