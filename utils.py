import random

import numpy as np
import torch

from plotting.scatter_plot import ScatterPlot


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def aggregate_by_cluster(Z, cluster_ids):
    Z_clusters = []
    for cid in np.unique(cluster_ids):
        Z_clusters.append(Z[cluster_ids == cid])
    return Z_clusters


def centroid_representative_indices(Z, y):
    """For each class, return the index of the sample closest to the class centroid in Z."""
    base_idxs = []
    for c in np.unique(y):
        class_mask = y == c
        Z_c = Z[class_mask]
        centroid = Z_c.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(Z_c - centroid, axis=1)
        idx_in_class = np.argmin(dists)
        global_idx = np.where(class_mask)[0][idx_in_class]
        base_idxs.append(global_idx)
    return np.array(base_idxs)


def labels_to_clusters(Z, y):
    """Group points by label into a list of arrays."""
    clusters = []
    for label in np.unique(y):
        clusters.append(Z[y == label])
    return clusters


def plot_projection_data(Z_data, y_data, filename, anchors=None):
    Z_clusters = labels_to_clusters(Z_data, y_data)
    ScatterPlot(
        Z_clusters=Z_clusters,
        anchors=anchors,
        point_size=2,
        point_alpha=0.6,
        figsize=(6, 6),
        dpi=300,
        filename=filename,
    ).render()
