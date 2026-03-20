import logging
import sys
import time
import numpy as np

from .utils_c import tools

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def eigSVD(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Description:
    Computes the Singular Value Decomposition (SVD) of a matrix X using 
    eigen-decomposition of its covariance (X^T @ X).

    Args:
        X (np.ndarray): Input matrix of shape (m, n).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (U, S, V) matrices from SVD.
    """
    D, V = np.linalg.eigh(X.T @ X)
    S = np.sqrt(D)
    U = X @ (V * (1.0 / S))
    return np.ascontiguousarray(U[:, ::-1]), np.ascontiguousarray(S[::-1]), np.ascontiguousarray(V[:, ::-1])

def RSVD(G: np.ndarray, N: int, M: int, f: np.ndarray, k: int, seed: int, 
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
    alpha = 0.0
    k_prime = max(k + 10, 20)
    
    # Internal buffers
    accum_mat = np.zeros((N, k_prime), dtype=np.float32)
    gen_buffer = np.zeros((chunk, N), dtype=np.float32)
    proj_basis = rng.standard_normal(size=(M, k_prime), dtype=np.float32)

    # 1) Prime iteration:
    log.info("    1) Prime iteration...")
    t_prime = time.time()
    for w in range(n_batches):
        start_idx = w * chunk
        n_active = min(chunk, M - start_idx)
        if gen_buffer.shape[0] != n_active:
            gen_buffer = np.zeros((n_active, N), dtype=np.float32)
        tools.decompress_block(G, gen_buffer, f, start_idx)
        accum_mat += gen_buffer.T @ proj_basis[start_idx : (start_idx + n_active)]
    
    orth_matrix, _, _ = eigSVD(accum_mat)
    accum_mat.fill(0.0)
    log.info(f"        time={time.time() - t_prime:.4f}s")

    # 2) Power iterations:
    log.info("    2) Power iterations...")
    t_power = time.time()
    singular_vals = np.zeros(k_prime, dtype=np.float32)
    shift_offset = 0
    for i in range(power):
        for w in range(n_batches):
            start_idx = w * chunk
            n_active = min(chunk, M - start_idx)
            if gen_buffer.shape[0] != n_active:
                gen_buffer = np.zeros((n_active, N), dtype=np.float32)
            tools.decompress_block(G, gen_buffer, f, start_idx)
            proj_basis[start_idx : (start_idx + n_active)] = gen_buffer @ orth_matrix
            accum_mat += gen_buffer.T @ proj_basis[start_idx : (start_idx + n_active)]
        
        accum_mat -= alpha * orth_matrix
        orth_matrix, s_vals, _ = eigSVD(accum_mat)
        accum_mat.fill(0.0)
        
        if i > 0:
            s_current = s_vals + alpha
            rel_diff = np.abs(s_current - singular_vals[:len(s_current)]) / np.maximum(s_current, 1e-12)
            max_err = np.max(rel_diff[shift_offset : k + shift_offset])
            if max_err < tol:
                log.info(f"        Converged at iteration {i}.")
                break
            singular_vals[:len(s_current)] = s_current
        else:
            singular_vals[:len(s_vals)] = s_vals + alpha

        # Dynamic shift update
        if s_vals[-1] > alpha:
            alpha = 0.5 * (s_vals[-1] + alpha)
            
    log.info(f"        time={time.time() - t_power:.4f}s")

    # 3) Build projection:
    log.info("    3) Build small matrix...")
    t_build = time.time()
    for w in range(n_batches):
        start_idx = w * chunk
        n_active = min(chunk, M - start_idx)
        if gen_buffer.shape[0] != n_active:
            gen_buffer = np.zeros((n_active, N), dtype=np.float32)
        tools.decompress_block(G, gen_buffer, f, start_idx)
        proj_basis[start_idx : (start_idx + n_active)] = gen_buffer @ orth_matrix
    log.info(f"        time={time.time() - t_build:.4f}s")

    # 4) SVD of condensed matrix:
    log.info("    4) SVD of small matrix...")
    t_svd = time.time()
    u_cond, s_all, v_cond = eigSVD(proj_basis)
    log.info(f"        time={time.time() - t_svd:.4f}s")

    # Truncate and rotate to final basis
    S = np.ascontiguousarray(s_all[:k])
    U = np.ascontiguousarray(u_cond[:, :k])
    V = np.ascontiguousarray(np.dot(orth_matrix, v_cond[:, :k]))

    total_time = time.time() - t0
    log.info(f"\n    Total time for SVD={total_time:.3f}s\n")

    return U, S, V
