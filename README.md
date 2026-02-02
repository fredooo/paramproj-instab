# **Stability of Parametric Projections under Input Perturbations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Uses: venv](https://img.shields.io/badge/Environment-venv-blue)](https://docs.python.org/3/library/venv.html)
[![Paper](https://img.shields.io/badge/paper-EuroVA%202026-red)](PAPER_LINK_PLACEHOLDER)
[![OSF Project](https://img.shields.io/badge/OSF-View%20Project-lightgrey)](OSF_LINK_PLACEHOLDER)

## Key Features

* Compares parametric projection methods (neural networks) against non-parametric baselines (UMAP, t-SNE) for dimensionality reduction.
* Measures projection stability under Gaussian input perturbations using novel metrics ($D_{\text{dev}}$, $D_{\text{bias}}$, $E_{\text{NA}}$).
* Evaluates projection quality via trustworthiness and continuity metrics.
* Includes MLP and spectrally-normalized MLP (SpecMLP) architectures with optional Jacobian regularization.
* Provides multiple visualization types: scatter plots, KDE contours, local PCA ellipses, anchor lines, and Voronoi tessellations.

## Requirements

* **Python** >= 3.11 ([Python 3.11.x](https://www.python.org/downloads/release/python-3110/))
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
| `typedefs.py`       | Configuration namedtuples for datasets, projections, models, experiments.   |
| `models.py`         | MLP and SpecMLP neural network architectures.                               |
| `measures.py`       | Stability metrics ($D_{\text{dev}}$, $D_{\text{bias}}$, $E_{\text{NA}}$) and quality metrics. |
| `train.py`          | Training loop for projection-mimicking neural networks.                     |
| `utils.py`          | Utility functions (seeding, centroid selection, plotting).                  |
| `test.py`           | Smoke test for verifying installation.                                      |
| `dataset_loaders/`  | Dataset loading functions for MNIST, FashionMNIST, HAR, Blobs.              |
| `projection_utils/` | UMAP and t-SNE setup utilities.                                             |
| `plotting/`         | Visualization modules (scatter, KDE, PCA ellipses, Voronoi, anchor lines).  |

## Metrics

### Stability Metrics

Given anchor point $\mathbf{z}_0$ and $N$ noisy projections $\{\mathbf{z}_i\}_{i=1}^N$:

- **$D_{\text{dev}}$** — Mean displacement: $D_{\text{dev}} = \frac{1}{N} \sum_{i=1}^{N} \lVert \mathbf{z}_i - \mathbf{z}_0 \rVert$

- **$D_{\text{bias}}$** — Displacement bias: $D_{\text{bias}} = \lVert \mathbf{\bar{z}} - \mathbf{z}_0 \rVert$ with $\mathbf{\bar{z}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{z}_i$

- **$E_{\text{NA}}$** — Nearest-Anchor Assignment Error: Fraction of noisy projections assigned to wrong anchor via nearest-neighbor.

### Quality Metrics

- **Trustworthiness**: Penalizes false neighbors (points close in low-dim but distant in high-dim)
- **Continuity**: Penalizes missing neighbors (points close in high-dim but distant in low-dim)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
