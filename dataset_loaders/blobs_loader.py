import numpy as np
import pandas as pd

from .utils import split_data


def load_blobs_split(seed=777, path="./datasets/blobs.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["labels"]).values.astype(np.float32)
    y = df["labels"].values
    return split_data(X, y, seed)
