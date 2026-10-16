import logging
import time

import numpy as np

from ..src.utils_c import batch_bvls_bpp, tools

log = logging.getLogger(__name__)

def ALS(U: np.ndarray, S: np.ndarray, V: np.ndarray, f: np.ndarray,
        seed: int, M: int, N: int, K: int, max_iter: int, tol: float) -> tuple[np.ndarray, np.ndarray]:
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

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    Z_half = np.ascontiguousarray(0.5 * (U * S), dtype=np.float64)
    Z_half_T = np.ascontiguousarray(Z_half.T)
    V_64 = np.ascontiguousarray(V, dtype=np.float64)
    V_T = np.ascontiguousarray(V_64.T)
    f_64 = np.ascontiguousarray(f, dtype=np.float64)

    P = rng.random(size=(M, K), dtype=np.float64)
    tools.mapP_d(P, M, K)

    # Initialize Q
    A_cov_P = np.ascontiguousarray(P.T @ P, dtype=np.float64)
    pinv_P = np.linalg.pinv(A_cov_P)
    Q = V_64 @ ((Z_half_T @ P) @ pinv_P) + ((f_64 @ P) @ pinv_P)
    tools.mapQ_d(Q, N, K)
    Q0 = np.empty_like(Q)

    B_target_P = np.empty((M, K), dtype=np.float64)
    B_target_Q = np.empty((N, K), dtype=np.float64)

    for i in range(max_iter):
        Q, Q0 = Q0, Q

        # Update P
        A_cov_Q = np.ascontiguousarray(Q0.T @ Q0, dtype=np.float64)
        np.matmul(Z_half, V_T @ Q0, out=B_target_P)
        B_target_P += f_64[:, None] * Q0.sum(axis=0)

        batch_bvls_bpp(A_cov_Q, B_target_P, P, 0.0, 1.0)
        tools.mapP_d(P, M, K)

        # Update Q
        A_cov_P = np.ascontiguousarray(P.T @ P, dtype=np.float64)
        np.matmul(V_64, Z_half_T @ P, out=B_target_Q)
        B_target_Q += f_64 @ P

        batch_bvls_bpp(A_cov_P, B_target_Q, Q, 0.0, 1.0)
        tools.mapQ_d(Q, N, K)

        rmse_error = tools.rmse_d(Q, Q0, N, K)

        if rmse_error < tol:
            log.info(f"        Convergence reached in iteration {i+1}.")
            break

    total_time = time.time() - t0
    log.info(f"        Total ALS time={total_time:.3f}s\n")

    return P, Q
