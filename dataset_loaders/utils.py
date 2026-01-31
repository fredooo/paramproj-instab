import numpy as np


def split_data(X, y, seed=777, train_ratio=0.8, val_ratio=0.1):
    """Shuffle and split X, y into train/val/test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n = len(X)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    i_tr, i_val, i_te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
    return X[i_tr], y[i_tr], X[i_val], y[i_val], X[i_te], y[i_te]


def load_torchvision_split(dataset_cls, seed=777):
    """Generic loader for torchvision image datasets (MNIST, FashionMNIST)."""
    train_ds = dataset_cls("./datasets", train=True, download=True)
    test_ds = dataset_cls("./datasets", train=False, download=True)
    X = np.concatenate([train_ds.data.numpy(), test_ds.data.numpy()]).astype("float32") / 255.0
    y = np.concatenate([train_ds.targets.numpy(), test_ds.targets.numpy()])
    X = X.reshape(len(X), -1)
    return split_data(X, y, seed)
