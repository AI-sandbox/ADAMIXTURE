import logging
import time

import numpy as np

from ..src.utils_c import batch_bvls_bpp, tools

log = logging.getLogger(__name__)

def ALS(U: np.ndarray, S: np.ndarray, V: np.ndarray, f: np.ndarray,
        seed: int, M: int, N: int, K: int, max_iter: int, tol: float,
        beta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Alternating Least Squares (ALS) algorithm with exact NNLS (KKT conditions)
    via Block Principal Pivoting and correct biological projection.

    Args:
        U (np.ndarray): Left singular vectors from SVD.
        S (np.ndarray): Singular values from SVD.
        V (np.ndarray): Right singular vectors from SVD.
        f (np.ndarray): Allele frequencies.
        seed (int): Random seed for reproducibility.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        max_iter (int): Maximum number of ALS iterations.
        tol (float): Convergence tolerance for ALS.
        beta (float): Momentum extrapolation factor for Q.

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    f_64 = np.ascontiguousarray(f, dtype=np.float64)

    # The genotype model is Z_half @ V^T + f 1^T. Appending f to Z and a column
    # of ones to V turns it into a single product Z_aug @ V_aug^T.
    k_svd = U.shape[1]
    Z_aug = np.empty((M, k_svd + 1), dtype=np.float64)
    Z_aug[:, :k_svd] = 0.5 * (U * S)
    Z_aug[:, k_svd] = f_64
    Z_aug_T = np.ascontiguousarray(Z_aug.T)
    V_aug = np.ones((N, k_svd + 1), dtype=np.float64)
    V_aug[:, :k_svd] = V
    V_aug_T = np.ascontiguousarray(V_aug.T)

    # Fisher information curvature weights: 1 / max(f * (1 - f), 1e-3)
    w2 = 1.0 / np.maximum(f_64 * (1.0 - f_64), 1e-3)
    w2 /= np.mean(w2)
    w2_col = w2[:, None]

    P = rng.random(size=(M, K), dtype=np.float64)
    tools.mapP_d(P, M, K)
    WP = np.empty((M, K), dtype=np.float64)

    # Initialize Q with Fisher curvature weighting
    np.multiply(P, w2_col, out=WP)
    A_cov_P = np.ascontiguousarray(P.T @ WP, dtype=np.float64)
    pinv_P = np.linalg.pinv(A_cov_P)
    Q = V_aug @ ((Z_aug_T @ WP) @ pinv_P)
    tools.mapQ_d(Q, N, K)

    Q_prev = Q.copy()
    Q_extrap = np.empty_like(Q)
    Q_next = np.empty_like(Q)

    B_target_P = np.empty((M, K), dtype=np.float64)
    B_target_Q = np.empty((N, K), dtype=np.float64)

    for i in range(max_iter):
        if beta > 0.0 and i > 0:
            np.add(Q, beta * (Q - Q_prev), out=Q_extrap)
            tools.mapQ_d(Q_extrap, N, K)
            Q_use = Q_extrap
        else:
            Q_use = Q

        # Update P
        A_cov_Q = np.ascontiguousarray(Q_use.T @ Q_use, dtype=np.float64)
        np.matmul(Z_aug, V_aug_T @ Q_use, out=B_target_P)

        batch_bvls_bpp(A_cov_Q, B_target_P, P, 0.0, 1.0)
        tools.mapP_d(P, M, K)

        # Update Q (weighted by Fisher curvature)
        np.multiply(P, w2_col, out=WP)
        A_cov_P = np.ascontiguousarray(P.T @ WP, dtype=np.float64)
        np.matmul(V_aug, Z_aug_T @ WP, out=B_target_Q)

        batch_bvls_bpp(A_cov_P, B_target_Q, Q_next, 0.0, 1.0)
        tools.mapQ_d(Q_next, N, K)

        rmse_error = tools.rmse_d(Q_next, Q, N, K)
        Q_prev, Q, Q_next = Q, Q_next, Q_prev

        if rmse_error < tol:
            log.info(f"        Convergence reached in iteration {i+1}.")
            break

    total_time = time.time() - t0
    log.info(f"        Total ALS time={total_time:.3f}s\n")

    return P, Q
