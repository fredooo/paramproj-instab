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


def V_sigma(Z):
    """Local projection variance V(sigma).

    Parameters
    ----------
    Z : ndarray, shape (N, q)
        Noisy projections.

    Returns
    -------
    float
        Mean squared distance to neighborhood mean.
    """
    z_bar = Z.mean(axis=0)
    return np.mean(np.sum((Z - z_bar[None, :]) ** 2, axis=1))

def Q_sigma(Z, sigma):
    """Noise amplification measure Q(sigma).

    Parameters
    ----------
    Z : ndarray, shape (N, q)
        Noisy projections.
    sigma : float
        Noise standard deviation.

    Returns
    -------
    float
        Variance normalized by sigma^2.
    """
    return V_sigma(Z) / (sigma ** 2)


def C_Q(Z_list, sigmas):
    """Coefficient of variation of Q(sigma) across noise levels.

    Parameters
    ----------
    Z_list : list of ndarray
        List of noisy projection arrays, one per sigma.
    sigmas : array-like
        Corresponding noise levels.

    Returns
    -------
    float
        Coefficient of variation of Q(sigma).
    """
    Q_vals = np.array([Q_sigma(Z, sigma) for Z, sigma in zip(Z_list, sigmas)])
    Q_mean = Q_vals.mean()
    Q_std = Q_vals.std(ddof=0)
    return Q_std / Q_mean


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


def compute_stability_metrics(Z_base, Z_clusters, X_base, project_fn, experiment_cfg, clip_bounds):
    """Compute D_dev, D_bias, Q, C_Q for any projection function.

    Args:
        Z_base: anchor points in projection space (n_anchors, 2)
        Z_clusters: list of projected noisy samples per anchor
        X_base: anchor points in input space (n_anchors, input_dim)
        project_fn: callable X -> Z (e.g., reducer.transform or model predict)
        experiment_cfg: ExperimentConfig with sigma, sigmas_cq, n_samples_cq
        clip_bounds: tuple for create_noisy_versions

    Returns:
        dict with D_dev, D_bias, Q, C_Q
    """
    n_anchors = len(Z_base)

    D_dev_val = np.mean([D_dev(Zu, z0) for Zu, z0 in zip(Z_clusters, Z_base)])
    D_bias_val = np.mean([D_bias(Zu, z0) for Zu, z0 in zip(Z_clusters, Z_base)])
    Q_val = np.mean([Q_sigma(Zu, experiment_cfg.sigma) for Zu in Z_clusters])

    cq_vals = []
    for i in range(n_anchors):
        Z_list = [
            project_fn(create_noisy_versions(
                X_base[i:i+1], s, experiment_cfg.n_samples_cq, clip_bounds
            )[0])
            for s in experiment_cfg.sigmas_cq
        ]
        cq_vals.append(C_Q(Z_list, experiment_cfg.sigmas_cq))
    C_Q_val = np.mean(cq_vals)

    return {
        "D_dev": D_dev_val,
        "D_bias": D_bias_val,
        "Q": Q_val,
        "C_Q": C_Q_val,
    }

# TODO 1: 
# Impelment the E_NA (Nearest-Anchor Assignment Error)
# For each class anchor $z_0^{(c)}$ and its corresponding noise-perturbed projections
# $\{z_i^{(c)}\}_{i=1}^N$, we assign each projected point to its nearest anchor in
# the projected space:
# \begin{equation}
# a(z) = \arg\min_{k} \, \| z - z_0^{(k)} \|_2 .
# \end{equation}
# We then define the misassignment rate at noise level $\sigma$ as
# \begin{equation}
# E_{\text{NA}}(\sigma)
# =
# \frac{1}{C}
# \sum_{c=1}^{C}
# \frac{1}{N}
# \sum_{i=1}^{N}
# \mathbf{1}\!\left[ a\!\left(z_i^{(c)}\right) \neq c \right],
# \end{equation}
# where $C$ denotes the number of anchors and $\mathbf{1}[\cdot]$ is the indicator
# function.
# This measure quantifies the probability that noise-induced perturbations cause a
# sample to leave the Voronoi region of its original anchor, providing a direct and
# interpretable notion of projection robustness under sensor noise.

# TODO 2: Include this only in a plan once TODO 1 is done and E_NA is implemente.
# Then replace the computation of Q anc C_Q in the
# codebase, specifically in compute_stability_metrics, with E_NA. This will
# involve effect the dict that is returned by compute_stability_metrics.
# Thus, ceck the calls to compute_stability_metrics in main.py and ensure
# that the returned metrics dict is handled correctly. 

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
