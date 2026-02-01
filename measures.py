import numba
import numpy as np


# =============================================================================
# Stability Metrics (for perturbation analysis)
# =============================================================================

def D_dev(Z, z0):
    """Typical displacement (noise-induced drift).

    Parameters
    ----------
    Z : ndarray, shape (N, q)
        Noisy projections.
    z0 : ndarray, shape (q,)
        Anchor projection.

    Returns
    -------
    float
        Mean distance to anchor.
    """
    return np.mean(np.linalg.norm(Z - z0[None, :], axis=1))


def D_bias(Z, z0):
    """Mean bias (systematic displacement).

    Parameters
    ----------
    Z : ndarray, shape (N, q)
        Noisy projections.
    z0 : ndarray, shape (q,)
        Anchor projection.

    Returns
    -------
    float
        Distance between mean noisy projection and anchor.
    """
    z_bar = Z.mean(axis=0)
    return np.linalg.norm(z_bar - z0)


def E_NA(Z_clusters, Z_base):
    """Nearest-Anchor Assignment Error.

    Parameters
    ----------
    Z_clusters : list of ndarray
        List of (N, q) arrays - noisy projections per anchor.
    Z_base : ndarray, shape (C, q)
        Anchor projections.

    Returns
    -------
    float
        Misassignment rate in [0, 1].
    """
    C = len(Z_base)
    misassign_rates = []
    for c, Z_c in enumerate(Z_clusters):
        # Distances from each noisy point to all anchors: (N, C)
        dists = np.linalg.norm(Z_c[:, None, :] - Z_base[None, :, :], axis=2)
        # Nearest anchor for each point
        nearest = np.argmin(dists, axis=1)
        # Fraction misassigned
        misassign_rates.append(np.mean(nearest != c))
    return np.mean(misassign_rates)


def create_noisy_versions(X_base, sigma, n_samples, clip_bounds=(0.0, 1.0)):
    """Create n_samples noisy copies of each base vector, adding Gaussian noise with given sigma.
    If clip_bounds is (a, b), values are clipped to [a, b]; if None, no clipping."""
    result = []
    for x0 in X_base:
        noise = np.random.randn(n_samples, x0.shape[0]).astype(np.float32) * sigma
        X_noisy = (x0[None, :] + noise).astype(np.float32)
        if clip_bounds is not None:
            X_noisy = np.clip(X_noisy, clip_bounds[0], clip_bounds[1])
        result.append(X_noisy)
    return result


def compute_stability_metrics(Z_base, Z_clusters):
    """Compute D_dev, D_bias, E_NA stability metrics.

    Args:
        Z_base: anchor points in projection space (n_anchors, 2)
        Z_clusters: list of projected noisy samples per anchor

    Returns:
        dict with D_dev, D_bias, E_NA
    """
    D_dev_val = np.mean([D_dev(Zu, z0) for Zu, z0 in zip(Z_clusters, Z_base)])
    D_bias_val = np.mean([D_bias(Zu, z0) for Zu, z0 in zip(Z_clusters, Z_base)])

    return {
        "D_dev": D_dev_val,
        "D_bias": D_bias_val,
        "E_NA": E_NA(Z_clusters, Z_base),
    }

# =============================================================================
# Projection Quality Metrics (trustworthiness & continuity)
# =============================================================================

@numba.njit(parallel=False)
def argsort_rows(X):
    """Parallel row-wise argsort."""
    n, m = X.shape
    out = np.empty((n, m), dtype=np.int64)
    for i in numba.prange(n):
        out[i] = np.argsort(X[i])
    return out


@numba.njit(parallel=False, fastmath=False)
def _compute_ranks_from_argsort(nn):
    """Convert argsort indices to ranks.

    Parameters
    ----------
    nn : ndarray, shape (n, n)
        nn[i] = argsort indices for row i

    Returns
    -------
    ranks : ndarray, shape (n, n)
        ranks[i, j] = rank of j in row i
    """
    n = nn.shape[0]
    ranks = np.empty((n, n), dtype=np.int64)
    for i in numba.prange(n):
        for r in range(n):
            ranks[i, nn[i, r]] = r
    return ranks


@numba.njit(parallel=False, fastmath=False)
def metric_trustworthiness_numba(D_high, D_low, k=7):
    """Compute trustworthiness metric.

    Parameters
    ----------
    D_high : ndarray, shape (n, n)
        High-dimensional distance matrix.
    D_low : ndarray, shape (n, n)
        Low-dimensional distance matrix.
    k : int
        Number of neighbors.

    Returns
    -------
    float
        Trustworthiness score in [0, 1].
    """
    n = D_high.shape[0]

    nn_orig = argsort_rows(D_high)
    nn_proj = argsort_rows(D_low)
    ranks_orig = _compute_ranks_from_argsort(nn_orig)

    partial = np.zeros(n, dtype=np.float64)

    for i in numba.prange(n):
        s = 0.0
        in_orig = np.zeros(n, dtype=np.uint8)
        for t in range(1, k + 1):
            in_orig[nn_orig[i, t]] = 1

        for t in range(1, k + 1):
            j = nn_proj[i, t]
            if in_orig[j] == 0:
                s += ranks_orig[i, j] - k

        partial[i] = s

    total = partial.sum()
    norm = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 * total) / norm


@numba.njit(parallel=False, fastmath=False)
def metric_continuity_numba(D_high, D_low, k=7):
    """Compute continuity metric.

    Parameters
    ----------
    D_high : ndarray, shape (n, n)
        High-dimensional distance matrix.
    D_low : ndarray, shape (n, n)
        Low-dimensional distance matrix.
    k : int
        Number of neighbors.

    Returns
    -------
    float
        Continuity score in [0, 1].
    """
    n = D_high.shape[0]

    nn_orig = argsort_rows(D_high)
    nn_proj = argsort_rows(D_low)
    ranks_proj = _compute_ranks_from_argsort(nn_proj)

    partial = np.zeros(n, dtype=np.float64)

    for i in numba.prange(n):
        s = 0.0
        in_proj = np.zeros(n, dtype=np.uint8)
        for t in range(1, k + 1):
            in_proj[nn_proj[i, t]] = 1

        for t in range(1, k + 1):
            j = nn_orig[i, t]
            if in_proj[j] == 0:
                s += ranks_proj[i, j] - k

        partial[i] = s

    total = partial.sum()
    norm = n * k * (2 * n - 3 * k - 1)
    return 1.0 - (2.0 * total) / norm


@numba.njit(parallel=False, fastmath=False)
def trustworthiness_continuity_powers_of_two(D_high, D_low):
    """Compute trustworthiness and continuity for k = 2, 4, 8, 16, ... floor(N/2).

    Parameters
    ----------
    D_high : ndarray, shape (n, n)
        High-dimensional distance matrix.
    D_low : ndarray, shape (n, n)
        Low-dimensional distance matrix.

    Returns
    -------
    ks : ndarray
        Array of k values (powers of two).
    trust : ndarray
        Trustworthiness scores for each k.
    cont : ndarray
        Continuity scores for each k.
    """
    n = D_high.shape[0]
    kmax = n // 2

    nn_orig = argsort_rows(D_high)
    nn_proj = argsort_rows(D_low)

    ranks_orig = _compute_ranks_from_argsort(nn_orig)
    ranks_proj = _compute_ranks_from_argsort(nn_proj)

    trust_partial = np.zeros((n, kmax + 1), dtype=np.float64)
    cont_partial = np.zeros((n, kmax + 1), dtype=np.float64)

    for i in numba.prange(n):
        t_sum = 0.0
        c_sum = 0.0

        for k in range(1, kmax + 1):
            j_proj = nn_proj[i, k]
            r_orig = ranks_orig[i, j_proj]
            if r_orig > k:
                t_sum += r_orig - k

            j_orig = nn_orig[i, k]
            r_proj = ranks_proj[i, j_orig]
            if r_proj > k:
                c_sum += r_proj - k

            trust_partial[i, k] = t_sum
            cont_partial[i, k] = c_sum

    count = 0
    k = 2
    while k <= kmax:
        count += 1
        k *= 2

    ks = np.empty(count, dtype=np.int64)
    trust = np.empty(count, dtype=np.float64)
    cont = np.empty(count, dtype=np.float64)

    idx = 0
    k = 2
    while k <= kmax:
        total_t = trust_partial[:, k].sum()
        total_c = cont_partial[:, k].sum()
        norm = n * k * (2 * n - 3 * k - 1)

        ks[idx] = k
        trust[idx] = 1.0 - (2.0 * total_t) / norm
        cont[idx] = 1.0 - (2.0 * total_c) / norm

        idx += 1
        k *= 2

    return ks, trust, cont
