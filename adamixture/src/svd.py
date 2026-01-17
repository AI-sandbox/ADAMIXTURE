import logging
import sys
import time
import numpy as np

from .utils_c import rsvd

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def eigSVD(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Singular Value Decomposition (SVD) of a matrix using eigen-decomposition.

    This function computes the SVD of a matrix S.

    Parameters
    ----------
    S : ndarray, shape (m, n)
        Input matrix.

    Returns
    -------
    U : ndarray, shape (m, r)
        Left singular vectors (orthonormal).
    S : ndarray, shape (r,)
        Singular values in descending order.
    V : ndarray, shape (n, r)
        Right singular vectors (orthonormal).
    """
    D, V = np.linalg.eigh(X.T @ X)
    S = np.sqrt(D)
    U = X @ (V * (1.0 / S))
    return np.ascontiguousarray(U[:, ::-1]), np.ascontiguousarray(S[::-1]), np.ascontiguousarray(V[:, ::-1])

def RSVD(G: np.ndarray, N: int, M: int, f: np.ndarray, k: int, seed: int, 
        power: int, tol: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomized SVD with Dynamic Shifts.

    Based on the paper:
        "dashSVD: Faster Randomized SVD with Dynamic Shifts"
        https://dl.acm.org/doi/10.1145/3660629

    Reference code:
        https://github.com/THU-numbda/dashSVD

    Parameters
    ----------
    G : array-like
        Input matrix in uint8 format.
    N : int
        Number of rows in A.
    M : int
        Number of columns in A.
    f : object
        Extra structure needed by the multiplication routines.
    k : int
        Target rank (number of singular values/vectors).
    seed : int
        Random seed.
    power_iterations : int, optional
        Number of power iterations (default: 10).
    oversampling : int, optional
        Extra dimensions for stability (default: 10).

    Returns
    -------
    U : ndarray, shape (N, k)
        Left singular vectors.
    S : ndarray, shape (k,)
        Singular values.
    V : ndarray, shape (M, k)
        Right singular vectors.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    k_prime = max(k + 10, 20)
    alpha = 0.0

    # Buffers 
    Y = np.zeros((N, k_prime), dtype=np.float32)
    G_small = np.zeros((M, k_prime), dtype=np.float32)

    # Prime iteration:
    log.info("    1) Prime iteration (Y = A @ Omega) ...")
    t_prime = time.time()
    Omega = rng.standard_normal(size=(M, k_prime), dtype=np.float32)
    rsvd.multiply_AT_omega(G, Omega, f, Y)
    Q, _, _ = eigSVD(Y)
    Y.fill(0.0)
    log.info(f"        prime iter time={time.time() - t_prime:.4f}s")

    # Power iterations:
    log.info("    2) Power iterations...")
    t_power = time.time()
    sk = np.zeros(k_prime, dtype=np.float32)
    s=0
    for i in range(power):
        rsvd.multiply_A_omega(G, Q, f, G_small)
        rsvd.multiply_AT_omega(G, G_small, f, Y)
        Y -= alpha * Q
        Q, S_y, _ = eigSVD(Y)
        if i > 0:
            sk_now = S_y + alpha
            pve_all = np.abs(sk_now - sk[:len(sk_now)]) / np.maximum(sk_now, 1e-12)
            ei = np.max(pve_all[s: k+s])
            if ei < tol:
                log.info(f"        Converged at iteration {i}.")
                G_small.fill(0.0)
                Y.fill(0.0)
                break
            sk[:len(sk_now)] = sk_now
        else:
            sk[:len(S_y)] = S_y + alpha

        if alpha < S_y[-1]:
            alpha = (alpha + S_y[-1]) / 2

        # Reset buffers
        G_small.fill(0.0)
        Y.fill(0.0)
            
    log.info(f"        power iterations time={time.time() - t_power:.4f}s")

    # Final SVD:
    log.info("    3) Build small matrix G_small = A^T @ Q ...")
    t_build = time.time()
    rsvd.multiply_A_omega(G, Q, f, G_small)
    log.info(f"        build time={time.time() - t_build:.4f}s")

    log.info("    4) SVD of small matrix A_small ...")
    t_svd = time.time()
    U_small, S_all, V_small = eigSVD(G_small)
    log.info(f"        svd(A_small) time={time.time() - t_svd:.4f}s")

    S = np.ascontiguousarray(S_all[:k])
    U = np.ascontiguousarray(U_small[:, :k])
    V = np.ascontiguousarray(np.dot(Q, V_small[:, :k]))

    total_time = time.time() - t0
    log.info(f"        Total RSVD time={total_time:.4f}s")
    log.info("")

    return U, S, V