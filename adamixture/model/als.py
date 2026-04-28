import logging
import time

import numpy as np

from ..src.utils_c import tools

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

    Z = np.ascontiguousarray(U * S).astype(np.float64)
    V_T = V.T.astype(np.float64)
    Z_T = Z.T
    f_64 = f.astype(np.float64)

    P = rng.random(size=(M, K), dtype=np.float64)
    tools.mapP_d(P, M, K)

    A_cov_P = np.ascontiguousarray(P.T @ P, dtype=np.float64)
    I_p = P @ np.linalg.pinv(A_cov_P)

    Q = 0.5 * (V @ (Z.T @ I_p)) + (f_64 @ I_p)
    tools.mapQ_d(Q, N, K)
    Q0 = np.copy(Q)

    P_buffer = np.empty((M, K), dtype=np.float64)
    Q_buffer = np.empty((N, K), dtype=np.float64)

    for i in range(max_iter):

        # UPDATE P
        A_cov_Q = np.ascontiguousarray(Q.T @ Q, dtype=np.float64)
        I_q = Q @ np.linalg.pinv(A_cov_Q)
        P_free = 0.5 * (Z @ (V_T @ I_q)) + np.outer(f_64, I_q.sum(axis=0))
        B_target_P = np.ascontiguousarray(P_free @ A_cov_Q, dtype=np.float64)

        P_buffer.fill(0)
        tools.batch_bvls_bpp(A_cov_Q, B_target_P, P_buffer, 0.0, 1.0)
        tools.mapP_d(P_buffer, M, K)
        memoryview(P.ravel())[:] = memoryview(P_buffer.ravel())

        # UPDATE Q
        A_cov_P = np.ascontiguousarray(P.T @ P, dtype=np.float64)
        I_p = P @ np.linalg.pinv(A_cov_P)
        Q_free = 0.5 * (V @ (Z_T @ I_p)) + (f_64 @ I_p)
        B_target_Q = np.ascontiguousarray(Q_free @ A_cov_P, dtype=np.float64)

        Q_buffer.fill(0)
        tools.batch_bvls_bpp(A_cov_P, B_target_Q, Q_buffer, 0.0, 1.0)
        tools.mapQ_d(Q_buffer, N, K)
        memoryview(Q.ravel())[:] = memoryview(Q_buffer.ravel())

        rmse_error = tools.rmse_d(Q, Q0, N, K)

        if rmse_error < tol:
            log.info(f"        Convergence reached in iteration {i+1}.")
            break
        else:
            memoryview(Q0.ravel())[:] = memoryview(Q.ravel())

    total_time = time.time() - t0
    log.info(f"        Total ALS time={total_time:.3f}s\n")

    return P, Q
