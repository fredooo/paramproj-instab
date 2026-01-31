import joblib
import os
import warnings

import umap

# Suppress UMAP warning about n_jobs override when using random_state
warnings.filterwarnings("ignore", message="n_jobs value.*overridden.*by setting random_state")


def umap_setup(X_tr, y_tr, X_val, X_te, seed, path_prefix):
    """Setup function for UMAP projection.

    Returns (reducer, Z_tr, Z_val, Z_te, supports_transform).
    """
    path = f"{path_prefix}.joblib"
    reducer, Z_tr = load_or_fit_umap(X_tr, y_tr, umap_path=path, seed=seed)
    Z_val = reducer.transform(X_val)
    Z_te = reducer.transform(X_te)
    return reducer, Z_tr, Z_val, Z_te, True


def load_or_fit_umap(
    X_train,
    y_train,
    umap_path="umap_reducer.joblib",
    seed=777,
):
    if os.path.exists(umap_path):
        reducer = joblib.load(umap_path)
        Z_train = reducer.embedding_
    else:
        reducer = umap.UMAP(random_state=seed)
        Z_train = reducer.fit_transform(X_train)
        joblib.dump(reducer, umap_path)

    return reducer, Z_train