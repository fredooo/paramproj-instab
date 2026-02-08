# **Stability of Parametric Projections under Input Perturbations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Uses: venv](https://img.shields.io/badge/Environment-venv-blue)](https://docs.python.org/3/library/venv.html)
[![Paper](https://img.shields.io/badge/paper-EuroVA%202026-red)](PAPER_LINK_PLACEHOLDER)
[![OSF Project](https://img.shields.io/badge/OSF-View%20Project-lightgrey)](OSF_LINK_PLACEHOLDER)

## Key Features

* Compares parametric projection methods (neural networks) against non-parametric baselines (UMAP [1], t-SNE [2]) for dimensionality reduction.
* Measures projection stability under Gaussian input perturbations using novel metrics ($D_{\text{dev}}$, $D_{\text{bias}}$, $E_{\text{NA}}$).
* Evaluates projection quality via trustworthiness and continuity [3] metrics.
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
- Fit UMAP [1] and t-SNE [2] projections
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
| `projection_utils/` | UMAP [1] and t-SNE [2] setup utilities.                                             |
| `plotting/`         | Visualization modules (scatter, KDE, PCA ellipses, Voronoi, anchor lines).  |

## Metrics

### Stability Metrics

Given anchor point $z_0$ and $N$ noisy projections $\{z_i\}_{i=1}^N$:

- **$D_{\text{dev}}$** — Mean displacement: $D_{\text{dev}} = \frac{1}{N} \sum_{i=1}^{N} \lVert z_i - z_0 \rVert$

- **$D_{\text{bias}}$** — Displacement bias: $D_{\text{bias}} = \lVert \overline{z} - z_0 \rVert$ with $\overline{z} = \frac{1}{N}\sum_{i=1}^{N} z_i$

- **$E_{\text{NA}}$** — Nearest-Anchor Assignment Error: Fraction of noisy projections assigned to wrong anchor via nearest-neighbor.

### Quality Metrics

- **Trustworthiness** [3]: Penalizes false neighbors (points close in low-dim but distant in high-dim)
- **Continuity** [3]: Penalizes missing neighbors (points close in high-dim but distant in low-dim)

## References

[1] McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*.

[2] van der Maaten, L., & Hinton, G. (2008). Visualizing Data using t-SNE. *Journal of Machine Learning Research*, 9(86), 2579–2605.

[3] Venna, J., & Kaski, S. (2001). Neighborhood Preservation in Nonlinear Projection Methods: An Experimental Study. *30th International Conference on Artificial Neural Networks*, 485–491.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
