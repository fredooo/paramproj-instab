import os
import time

import joblib
import numpy as np
from sklearn.manifold import TSNE


def tsne_setup(X_tr, y_tr, X_val, X_te, seed, path_prefix):
    """Setup function for t-SNE projection.

    Returns (reducer, Z_tr, Z_val, Z_te, supports_transform, fit_time).
    Note: t-SNE does not support transform, so supports_transform=False.
    fit_time is None if loaded from cache.
    """
    path = f"{path_prefix}.joblib"
    reducer, Z_tr, Z_val, Z_te, fit_time = load_or_fit_tsne_splits(X_tr, y_tr, X_val, X_te, tsne_path=path, seed=seed)
    return reducer, Z_tr, Z_val, Z_te, False, fit_time


def load_or_fit_tsne(
    X_train,
    y_train,
    tsne_path="tsne_reducer.joblib",
    seed=777,
):
    """Fit t-SNE on training data. Returns (reducer, Z_train, fit_time).

    Note: t-SNE has no transform(); for val/test embeddings you must use
    load_or_fit_tsne_splits with X_val, X_te.
    fit_time is None if cached.
    """
    if os.path.exists(tsne_path):
        data = joblib.load(tsne_path)
        reducer = data["reducer"]
        Z_train = data["Z_train"]
        return reducer, Z_train, None

    start = time.time()
    reducer = TSNE(random_state=seed)
    Z_train = reducer.fit_transform(X_train)
    fit_time = time.time() - start
    print(f"        t-SNE fit time: {fit_time:.1f}s")
    joblib.dump({"reducer": reducer, "Z_train": Z_train}, tsne_path)
    return reducer, Z_train, fit_time


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
    Returns (reducer, Z_train, Z_val, Z_te, fit_time). fit_time is None if cached.
    """
    if os.path.exists(tsne_path):
        data = joblib.load(tsne_path)
        return (
            data["reducer"],
            data["Z_train"],
            data["Z_val"],
            data["Z_te"],
            None,
        )

    X_all = np.vstack([X_train, X_val, X_te])
    n_train, n_val = len(X_train), len(X_val)
    start = time.time()
    reducer = TSNE(random_state=seed)
    Z_all = reducer.fit_transform(X_all)
    fit_time = time.time() - start
    print(f"        t-SNE fit time: {fit_time:.1f}s")

    Z_train = Z_all[:n_train]
    Z_val = Z_all[n_train : n_train + n_val]
    Z_te = Z_all[n_train + n_val :]

    joblib.dump(
        {"reducer": reducer, "Z_train": Z_train, "Z_val": Z_val, "Z_te": Z_te},
        tsne_path,
    )
    return reducer, Z_train, Z_val, Z_te, fit_time
