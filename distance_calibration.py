"""Calibrate perturbation noise (sigma) across datasets to ensure comparable perturbation magnitudes.

Problem: Different datasets have different dimensionalities and value ranges. A fixed sigma
would result in very different relative perturbation magnitudes across datasets.

Solution: Use MNIST as reference. Find the percentile p at which a given distance threshold
(default 5.0) falls in MNIST's pairwise distance distribution. Then for each dataset, find
the distance d_p at that same percentile, and derive sigma = d_p / sqrt(dim).

This ensures that across all datasets, the perturbation distance sigma*sqrt(dim) corresponds
to the same percentile of the dataset's natural pairwise distance distribution.

Additionally, --effective mode measures the actual perturbation distance after clipping,
since image datasets (mnist, fmnist) are clipped to [0,1] which reduces effective noise.

Result: Effective distance measurement (N=5/class, 1000 noisy samples)
dataset   dim  sigma   d_exp   d_eff  ratio
mnist     784 0.1689  4.7292  3.4577  0.731
fmnist    784 0.1597  4.4716  3.6878  0.825
har       561 0.1433  3.3941  3.3940  1.000
blobs      10 0.7371  2.3309  2.3296  0.999
"""
from dataset_loaders.mnist_loader import load_mnist_split
from dataset_loaders.fmnist_loader import load_fmnist_split
from dataset_loaders.har_loader import load_har_split
from dataset_loaders.blobs_loader import load_blobs_split
from scipy.spatial.distance import pdist
import numpy as np


SEED = 777
MNIST_THRESHOLD = 5.0  # reference distance threshold on MNIST
SUBSAMPLE_SIZE = 7000  # pdist memory grows O(n^2), subsample large datasets

# Per-dataset configs from calibration above
# clip_bounds: image data clipped to [0,1], vector data unclipped
# sigma: noise std dev such that expected perturbation distance = d_p
DATASET_CONFIGS = {
    'mnist': {'clip_bounds': (0.0, 1.0), 'sigma': 0.1689},
    'fmnist': {'clip_bounds': (0.0, 1.0), 'sigma': 0.1597},
    'har': {'clip_bounds': None, 'sigma': 0.1433},
    'blobs': {'clip_bounds': None, 'sigma': 0.7371},
}


def get_all_X(load_fn, subsample=True):
    """Concatenate train/val/test X arrays, optionally subsample."""
    X_tr, _, X_val, _, X_te, _ = load_fn(SEED)
    X = np.vstack([X_tr, X_val, X_te])
    if subsample and len(X) > SUBSAMPLE_SIZE:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X), SUBSAMPLE_SIZE, replace=False)
        X = X[idx]
    return X


def compute_percentile_calibration(threshold=None, percentile=None):
    """Compute calibrated sigma values for each dataset.

    Args:
        threshold: Distance threshold on MNIST to determine percentile p.
                   Default: MNIST_THRESHOLD (5.0)
        percentile: Directly specify percentile p (0-1), overrides threshold.

    Returns:
        dict with 'percentile' and per-dataset dicts containing:
        - distance: d_p, the distance at percentile p
        - dim: input dimensionality
        - sigma: d_p / sqrt(dim), noise std dev for perturbations

    The key insight: if noise ~ N(0, sigma^2 * I_dim), then ||noise|| has
    expected value sigma * sqrt(dim). So sigma = d_p / sqrt(dim) gives
    perturbations with expected distance d_p.
    """
    loaders = {
        'mnist': load_mnist_split,
        'fmnist': load_fmnist_split,
        'har': load_har_split,
        'blobs': load_blobs_split,
    }

    # Determine percentile p from MNIST threshold or use provided percentile
    if percentile is not None:
        p = percentile
    else:
        thresh = threshold if threshold is not None else MNIST_THRESHOLD
        X_mnist = get_all_X(loaders['mnist'])
        D_mnist = pdist(X_mnist, metric='euclidean')
        p = np.mean(D_mnist < thresh)  # fraction of pairs closer than threshold

    # For each dataset, find distance at percentile p and derive sigma
    results = {'percentile': p}
    for name, load_fn in loaders.items():
        X = get_all_X(load_fn)
        D = pdist(X, metric='euclidean')
        d_at_p = np.percentile(D, p * 100)
        dim = X.shape[1]
        sigma = d_at_p / np.sqrt(dim)  # E[||noise||] = sigma * sqrt(dim) = d_at_p
        results[name] = {'distance': d_at_p, 'dim': dim, 'sigma': sigma}

    return results


def measure_effective_distance(n_per_class=5, n_noisy=1000):
    """Measure effective perturbation distance after clipping vs expected.

    For image datasets, values are clipped to [0,1] after adding noise, which
    reduces the effective perturbation magnitude. This function quantifies
    that reduction.

    Args:
        n_per_class: Number of anchor samples to select per class.
        n_noisy: Number of noisy samples to generate per anchor.

    Returns:
        dict with per-dataset results:
        - dim: input dimensionality
        - sigma: calibrated noise std dev
        - d_expected: sigma * sqrt(dim), expected distance without clipping
        - d_eff: sqrt(mean(||x_noisy - x_anchor||^2)), actual distance after clipping
        - ratio: d_eff / d_expected (< 1.0 indicates clipping effect)
        - n_anchors: total number of anchor samples used

    Interpretation:
    - ratio ~= 1.0: no clipping effect (har, blobs)
    - ratio < 1.0: clipping reduces effective noise (mnist ~0.73, fmnist ~0.83)
    """
    loaders = {
        'mnist': load_mnist_split,
        'fmnist': load_fmnist_split,
        'har': load_har_split,
        'blobs': load_blobs_split,
    }

    rng = np.random.default_rng(SEED)
    results = {}

    for name, load_fn in loaders.items():
        cfg = DATASET_CONFIGS[name]
        sigma = cfg['sigma']
        clip_bounds = cfg['clip_bounds']

        # Load data with labels
        X_tr, y_tr, X_val, y_val, X_te, y_te = load_fn(SEED)
        X = np.vstack([X_tr, X_val, X_te])
        y = np.concatenate([y_tr, y_val, y_te])
        dim = X.shape[1]

        # Select n_per_class samples per class
        classes = np.unique(y)
        base_idxs = []
        for c in classes:
            c_idxs = np.where(y == c)[0]
            chosen = rng.choice(c_idxs, min(n_per_class, len(c_idxs)), replace=False)
            base_idxs.extend(chosen)
        X_base = X[base_idxs]

        # Perturb each anchor and measure squared distances
        all_sq_dists = []
        for x0 in X_base:
            # Generate noisy versions: x_noisy = x0 + N(0, sigma^2 * I)
            noise = rng.standard_normal((n_noisy, dim)).astype(np.float32) * sigma
            X_noisy = (x0[None, :] + noise).astype(np.float32)
            if clip_bounds is not None:
                X_noisy = np.clip(X_noisy, clip_bounds[0], clip_bounds[1])
            # ||x_noisy - x0||^2 for each noisy sample
            sq_dists = np.sum((X_noisy - x0) ** 2, axis=1)
            all_sq_dists.extend(sq_dists)

        # d_eff = sqrt(E[||x_noisy - x0||^2])
        d_eff = np.sqrt(np.mean(all_sq_dists))
        d_expected = sigma * np.sqrt(dim)  # E[||noise||] without clipping
        ratio = d_eff / d_expected

        results[name] = {
            'dim': dim,
            'sigma': sigma,
            'd_expected': d_expected,
            'd_eff': d_eff,
            'ratio': ratio,
            'n_anchors': len(X_base),
        }

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--threshold', type=float, help='distance threshold on MNIST')
    group.add_argument('--percentile', type=float, help='direct percentile (0-1)')
    group.add_argument('--effective', action='store_true', help='measure effective distance after perturbation')
    parser.add_argument('--n-per-class', type=int, default=5, help='samples per class for --effective')
    parser.add_argument('--n-noisy', type=int, default=1000, help='noisy samples per anchor for --effective')
    args = parser.parse_args()

    if args.effective:
        res = measure_effective_distance(n_per_class=args.n_per_class, n_noisy=args.n_noisy)
        print(f"Effective distance measurement (N={args.n_per_class}/class, {args.n_noisy} noisy samples)")
        print(f"{'dataset':<8} {'dim':>4} {'sigma':>6} {'d_exp':>7} {'d_eff':>7} {'ratio':>6}")
        for name in ['mnist', 'fmnist', 'har', 'blobs']:
            r = res[name]
            print(f"{name:<8} {r['dim']:>4} {r['sigma']:>6.4f} {r['d_expected']:>7.4f} {r['d_eff']:>7.4f} {r['ratio']:>6.3f}")
    else:
        res = compute_percentile_calibration(
            threshold=args.threshold,
            percentile=args.percentile
        )
        print(f"Calibrated percentile p: {res['percentile']:.6f}")
        for name in ['mnist', 'fmnist', 'har', 'blobs']:
            r = res[name]
            print(f"{name}: distance={r['distance']:.4f}, dim={r['dim']}, sigma={r['sigma']:.4f}")
