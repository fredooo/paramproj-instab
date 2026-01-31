import joblib
import numpy as np
import os
from sklearn.manifold import TSNE

from plotting.scatter_plot import ScatterPlot
from utils import labels_to_clusters


def tsne_setup(X_tr, y_tr, X_val, X_te, seed, path_prefix):
    """Setup function for t-SNE projection.

    Returns (reducer, Z_tr, Z_val, Z_te, supports_transform).
    Note: t-SNE does not support transform, so supports_transform=False.
    """
    path = f"{path_prefix}.joblib"
    reducer, Z_tr, Z_val, Z_te = load_or_fit_tsne_splits(
        X_tr, y_tr, X_val, X_te, tsne_path=path, seed=seed
    )
    return reducer, Z_tr, Z_val, Z_te, False


def load_or_fit_tsne(
    X_train,
    y_train,
    tsne_path="tsne_reducer.joblib",
    seed=777,
):
    """Fit t-SNE on training data. Returns (reducer, Z_train).

    Note: t-SNE has no transform(); for val/test embeddings you must use
    load_or_fit_tsne_splits with X_val, X_te.
    """
    if os.path.exists(tsne_path):
        data = joblib.load(tsne_path)
        reducer = data["reducer"]
        Z_train = data["Z_train"]
    else:
        reducer = TSNE(random_state=seed)
        Z_train = reducer.fit_transform(X_train)
        joblib.dump({"reducer": reducer, "Z_train": Z_train}, tsne_path)

    return reducer, Z_train


def load_or_fit_tsne_splits(
    X_train,
    y_train,
    X_val,
    X_te,
    tsne_path="tsne_splits.joblib",
    seed=777,
):
    """Fit t-SNE on train+val+test and return embeddings for each split.

    Use this when you need Z_val, Z_te (t-SNE does not support transform).
    Returns (reducer, Z_train, Z_val, Z_te).
    """
    if os.path.exists(tsne_path):
        data = joblib.load(tsne_path)
        return (
            data["reducer"],
            data["Z_train"],
            data["Z_val"],
            data["Z_te"],
        )

    X_all = np.vstack([X_train, X_val, X_te])
    n_train, n_val = len(X_train), len(X_val)
    reducer = TSNE(random_state=seed)
    Z_all = reducer.fit_transform(X_all)

    Z_train = Z_all[:n_train]
    Z_val = Z_all[n_train : n_train + n_val]
    Z_te = Z_all[n_train + n_val :]

    joblib.dump(
        {"reducer": reducer, "Z_train": Z_train, "Z_val": Z_val, "Z_te": Z_te},
        tsne_path,
    )
    return reducer, Z_train, Z_val, Z_te


