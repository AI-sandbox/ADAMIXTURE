import logging
import sys
import time

import numpy as np

from ..src import utils
from ..src.utils_c import tools

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def eigSVD(X: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    Computes the Singular Value Decomposition (SVD) of a matrix X using
    eigen-decomposition of its covariance (X^T @ X).

    Args:
        X (np.ndarray): Input matrix of shape (m, n).
        k (int | None): Optional number of singular components to retain.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (U, S, V) matrices from SVD.
    """
    D, V = np.linalg.eigh(X.T @ X)
    if k is not None:
        D = D[::-1][:k]
        V = np.ascontiguousarray(V[:, ::-1][:, :k])
    else:
        D = D[::-1]
        V = np.ascontiguousarray(V[:, ::-1])

    S = np.sqrt(np.maximum(D, 0.0))
    U = X @ (V * (1.0 / np.maximum(S, 1e-12)))
    return U, S, V

def _RSVD_once(G: np.ndarray, N: int, M: int, f: np.ndarray, k: int, seed: int,
               power: int, tol: float, chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    Randomized Singular Value Decomposition with Dynamic Shifts (dashSVD).
    Optimized for memory-efficient processing of large genotype matrices.

    Args:
        G (np.ndarray): Input genotype matrix (unpacked or packed).
        N (int): Number of individuals (rows).
        M (int): Number of SNPs (columns).
        f (np.ndarray): Allele frequencies buffer.
        k (int): Target rank (number of components).
        seed (int): Random seed for the projection basis.
        power (int): Number of power iterations for subspace refinement.
        tol (float): Relative error tolerance for convergence.
        chunk (int): Number of SNPs per processing batch.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (U matrix, S vector, V matrix).
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # Execution parameters
    n_batches = int(np.ceil(M / chunk))
    alpha = np.float32(0.0)
    k_prime = max(k + 10, 20)

    # Internal buffers
    accum_mat = np.zeros((N, k_prime), dtype=np.float32)
    temp_accum = np.empty((N, k_prime), dtype=np.float32)
    gen_buffer = np.empty((chunk, N), dtype=np.float32)
    proj_chunk = np.empty((chunk, k_prime), dtype=np.float32)
    proj_basis = rng.standard_normal(size=(M, k_prime), dtype=np.float32)

    # Patterson-Price genetic PCA weighting for ancestral individual subspace V
    w_patterson = np.ascontiguousarray(
        1.0 / np.sqrt(2.0 * np.maximum(f * (1.0 - f), 1e-4)),
        dtype=np.float32,
    )

    # 1) Prime iteration
    log.info("    1) Prime iteration...")
    t_prime = time.time()
    for w in range(n_batches):
        start_idx = w * chunk
        n_active = min(chunk, M - start_idx)
        buf = gen_buffer[:n_active]
        tools.decompress_block_scaled(G, buf, f, w_patterson, start_idx)
        np.matmul(buf.T, proj_basis[start_idx : (start_idx + n_active)], out=temp_accum)
        accum_mat += temp_accum

    orth_matrix, _, _ = eigSVD(accum_mat)
    accum_mat.fill(0.0)
    log.info(f"        time={time.time() - t_prime:.4f}s")

    # 2) Power iterations
    log.info("    2) Power iterations...")
    t_power = time.time()
    singular_vals = np.zeros(k_prime, dtype=np.float32)
    for i in range(power):
        for w in range(n_batches):
            start_idx = w * chunk
            n_active = min(chunk, M - start_idx)
            buf = gen_buffer[:n_active]
            p_buf = proj_chunk[:n_active]
            tools.decompress_block_scaled(G, buf, f, w_patterson, start_idx)
            np.matmul(buf, orth_matrix, out=p_buf)
            np.matmul(buf.T, p_buf, out=temp_accum)
            accum_mat += temp_accum

        accum_mat -= alpha * orth_matrix
        orth_matrix, s_vals, _ = eigSVD(accum_mat)
        accum_mat.fill(0.0)

        if i > 0:
            s_current = s_vals + alpha
            rel_diff = np.abs(s_current - singular_vals[:len(s_current)]) / np.maximum(s_current, 1e-12)
            max_err = np.max(rel_diff[:k])
            if max_err < tol:
                log.info(f"        Converged at iteration {i}.")
                break
            singular_vals[:len(s_current)] = s_current
        else:
            singular_vals[:len(s_vals)] = s_vals + alpha

        # Dynamic shift update
        if s_vals[-1] > alpha:
            alpha = np.float32(0.5 * (s_vals[-1] + alpha))

    log.info(f"        time={time.time() - t_power:.4f}s")

    # 3) Build projection
    log.info("    3) Build small matrix...")
    t_build = time.time()
    for w in range(n_batches):
        start_idx = w * chunk
        n_active = min(chunk, M - start_idx)
        buf = gen_buffer[:n_active]
        tools.decompress_block(G, buf, f, start_idx)
        np.matmul(buf, orth_matrix, out=proj_basis[start_idx : (start_idx + n_active)])
    log.info(f"        time={time.time() - t_build:.4f}s")

    # 4) SVD of condensed matrix
    log.info("    4) SVD of small matrix...")
    t_svd = time.time()
    U, S, v_cond = eigSVD(proj_basis, k=k)
    log.info(f"        time={time.time() - t_svd:.4f}s")

    # Rotate right singular vectors
    V = np.matmul(orth_matrix, v_cond)

    total_time = time.time() - t0
    log.info(f"\n    Total time for SVD={total_time:.3f}s")

    return U, S, V


def RSVD(G: np.ndarray, N: int, M: int, f: np.ndarray, k: int, seed: int,
         power: int, tol: float, chunk: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomized SVD on CPU, retrying with smaller chunks on memory errors.
    """
    current_chunk = chunk

    while True:
        try:
            if current_chunk != chunk:
                log.warning(f"    Retrying SVD with chunk_size={current_chunk}.")
            return _RSVD_once(G, N, M, f, k, seed, power, tol, current_chunk)
        except (MemoryError, RuntimeError) as exc:
            if not utils.is_cpu_memory_error(exc) or current_chunk <= utils.MIN_CPU_CHUNK_SIZE:
                raise
            current_chunk = utils.reduce_cpu_chunk_or_raise(current_chunk, exc, "running SVD")
