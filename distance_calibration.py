from dataset_loaders.mnist_loader import load_mnist_split
from dataset_loaders.fmnist_loader import load_fmnist_split
from dataset_loaders.har_loader import load_har_split
from dataset_loaders.blobs_loader import load_blobs_split
from scipy.spatial.distance import pdist
import numpy as np

SEED = 777
MNIST_THRESHOLD = 5.0
SUBSAMPLE_SIZE = 7000  # pdist memory grows O(n^2)


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
    loaders = {
        'mnist': load_mnist_split,
        'fmnist': load_fmnist_split,
        'har': load_har_split,
        'blobs': load_blobs_split,
    }

    if percentile is not None:
        p = percentile
    else:
        thresh = threshold if threshold is not None else MNIST_THRESHOLD
        X_mnist = get_all_X(loaders['mnist'])
        D_mnist = pdist(X_mnist, metric='euclidean')
        p = np.mean(D_mnist < thresh)

    # Apply to all datasets
    results = {'percentile': p}
    for name, load_fn in loaders.items():
        X = get_all_X(load_fn)
        D = pdist(X, metric='euclidean')
        d_at_p = np.percentile(D, p * 100)
        dim = X.shape[1]
        sigma = d_at_p / np.sqrt(dim)
        results[name] = {'distance': d_at_p, 'dim': dim, 'sigma': sigma}

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--threshold', type=float, help='distance threshold on MNIST')
    group.add_argument('--percentile', type=float, help='direct percentile (0-1)')
    args = parser.parse_args()

    res = compute_percentile_calibration(
        threshold=args.threshold,
        percentile=args.percentile
    )
    print(f"Calibrated percentile p: {res['percentile']:.6f}")
    for name in ['mnist', 'fmnist', 'har', 'blobs']:
        r = res[name]
        print(f"{name}: distance={r['distance']:.4f}, dim={r['dim']}, sigma={r['sigma']:.4f}")
