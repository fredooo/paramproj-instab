import joblib
import os
import time
import warnings

import umap

# Suppress UMAP warning about n_jobs override when using random_state
warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*by setting random_state")


def umap_setup(X_tr, y_tr, X_val, X_te, seed, path_prefix):
    """Setup function for UMAP projection.

    Returns (reducer, Z_tr, Z_val, Z_te, supports_transform, fit_time).
    fit_time is None if loaded from cache.
    """
    path = f"{path_prefix}.joblib"
    reducer, Z_tr, fit_time = load_or_fit_umap(X_tr, y_tr, umap_path=path, seed=seed)
    Z_val = reducer.transform(X_val)
    Z_te = reducer.transform(X_te)
    return reducer, Z_tr, Z_val, Z_te, True, fit_time


def load_or_fit_umap(
    X_train,
    y_train,
    umap_path="umap_reducer.joblib",
    seed=777,
):
    """Returns (reducer, Z_train, fit_time). fit_time is None if cached."""
    if os.path.exists(umap_path):
        reducer = joblib.load(umap_path)
        Z_train = reducer.embedding_
        return reducer, Z_train, None

    start = time.time()
    reducer = umap.UMAP(random_state=seed)
    Z_train = reducer.fit_transform(X_train)
    fit_time = time.time() - start
    print(f"        UMAP fit time: {fit_time:.1f}s")
    joblib.dump(reducer, umap_path)
    return reducer, Z_train, fit_time