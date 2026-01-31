import numpy as np
import pandas as pd

from .utils import split_data


def load_har_split(seed=777, train_path="./datasets/HAR/train.csv", test_path="./datasets/HAR/test.csv"):
    df = pd.concat([pd.read_csv(train_path), pd.read_csv(test_path)], ignore_index=True)
    X = df.drop(columns=["subject", "Activity"]).values.astype(np.float32)
    y = df["Activity"].values
    return split_data(X, y, seed)
