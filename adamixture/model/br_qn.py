import logging
import sys
import time

import numpy as np

from ..src.utils_c.cython import sqp, tools
from ..src.utils_c.cython.br_qn import qn_extrapolate_ZAL, update_UV_ZAL

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def q_hessian_block_size(K: int) -> int:
    """Individuals per Q-Hessian tile so the per-thread workspace,
    B * (K(K+1)/2 + 3K + 6) doubles, stays near 64 KB."""
    per_individual = K * (K + 1) // 2 + 3 * K + 6
    return min(256, max(16, (8192 // per_individual) // 8 * 8))

def polish_sqp_qn(G: np.ndarray, P_init: np.ndarray, Q_init: np.ndarray,
                  M: int, N: int, K: int,
                  n_iters: int = 3, Q_hist: int = 3,
                  q_block: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Polishes a cross-validation fold with the same SQP + ZAL quasi-Newton
    optimizer as the main BR-QN run. Early stopping is disabled so every fold
    receives exactly ``n_iters`` iterations from its global-fit warm start.
    """
    P = np.array(P_init, dtype=np.float64, copy=True)
    Q = np.array(Q_init, dtype=np.float64, copy=True)
    return optimize_original(
        G, P, Q,
        max_iter=n_iters,
        K=K,
        M=M,
        N=N,
        tol=0.0,
        Q_hist=Q_hist,
        patience=n_iters + 1,
        verbose=False,
        q_block=q_block,
    )


def optimize_original(G: np.ndarray, P: np.ndarray, Q: np.ndarray, max_iter: int,
                      K: int, M: int, N: int, tol: float, Q_hist: int,
                      patience: int, verbose: bool = True,
                      q_block: int | None = None, fused_sqp: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Optimizes the P and Q matrices using the original ADMIXTURE algorithm on CPU:
    Sequential Quadratic Programming (SQP) block updates with ZAL Quasi-Newton acceleration.

    Args:
        G (np.ndarray): Input genotype matrix (M x N, uint8).
        P (np.ndarray): Pre-initialized P matrix (M x K).
        Q (np.ndarray): Pre-initialized Q matrix (N x K).
        max_iter (int): Maximum SQP iterations.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        tol (float): Relative convergence tolerance.
        Q_hist (int): Depth of ZAL Quasi-Newton acceleration history.
        patience (int): Iterations without log-likelihood improvement before stopping.
        verbose (bool): If True, log iteration progress and final likelihood.
        q_block (int | None): Q-Hessian tile size from the outer K run. If None, derived from K.
        copy_back (bool): If True, write the best parameters back into P and Q in-place.

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized (P, Q) matrices.
    """
    # 1. Precompute Vt matrix from SVD of ones(1, K)
    _, _, vt = np.linalg.svd(np.ones((1, K)), full_matrices=True)
    v_kk = np.ascontiguousarray(vt.T, dtype=np.float64)
    if q_block is None:
        q_block = q_hessian_block_size(K)

    # 2. Allocate buffers
    XtX_q = np.empty((N, K, K), dtype=np.float64)
    Xtz_q = np.empty((N, K), dtype=np.float64)
    if not fused_sqp:
        XtX_p = np.empty((M, K, K), dtype=np.float64)
        Xtz_p = np.empty((M, K), dtype=np.float64)

    # Contiguous parameter buffers with zero-copy views
    PQ = np.empty((M + N, K), dtype=np.float64)
    P_cur, Q_cur = PQ[:M], PQ[M:]
    P_cur[:] = P
    Q_cur[:] = Q
    x_buf = PQ.ravel()

    PQ_next = np.empty((M + N, K), dtype=np.float64)
    P_next, Q_next = PQ_next[:M], PQ_next[M:]
    x_next_buf = PQ_next.ravel()

    PQ_next2 = np.empty((M + N, K), dtype=np.float64)
    P_next2, Q_next2 = PQ_next2[:M], PQ_next2[M:]
    x_next2_buf = PQ_next2.ravel()

    PQ_qn = np.empty((M + N, K), dtype=np.float64)
    P_qn, Q_qn = PQ_qn[:M], PQ_qn[M:]
    x_qn = PQ_qn.ravel()

    PQ_best = np.empty((M + N, K), dtype=np.float64)
    P_best, Q_best = PQ_best[:M], PQ_best[M:]
    x_best = PQ_best.ravel()

    # QN history buffers
    dim = (M + N) * K
    U = np.zeros(dim * Q_hist, dtype=np.float64)
    V = np.zeros(dim * Q_hist, dtype=np.float64)
    UtUmV_workspace = np.zeros(Q_hist * (Q_hist + 1), dtype=np.float64)
    coeff_workspace = np.zeros(Q_hist, dtype=np.float64)

    # 3. Initialize log-likelihood tracking
    ll_prev_iter = -float('inf')
    ll_best = -float("inf")
    best_in_cur = False
    wait = 0

    for it in range(1, max_iter + 1):
        it_start = time.time()

        # --- SQP Update 1: (P_cur, Q_cur) -> (P_next, Q_next) ---
        sqp.update_q_sqp(G, Q_cur, Q_next, P_cur, XtX_q, Xtz_q, v_kk, M, N, K, q_block)
        if fused_sqp:
            sqp.update_p_sqp_fused(G, Q_next, P_cur, P_next, M, N, K)
        else:
            sqp.update_p_sqp_unfused(G, Q_next, P_cur, P_next, XtX_p, Xtz_p, M, N, K)

        # --- SQP Update 2: (P_next, Q_next) -> (P_next2, Q_next2) ---
        sqp.update_q_sqp(G, Q_next, Q_next2, P_next, XtX_q, Xtz_q, v_kk, M, N, K, q_block)
        if fused_sqp:
            sqp.update_p_sqp_fused(G, Q_next2, P_next, P_next2, M, N, K)
        else:
            sqp.update_p_sqp_unfused(G, Q_next2, P_next, P_next2, XtX_p, Xtz_p, M, N, K)

        # --- ZAL QN acceleration (Zero-Copy) ---
        update_UV_ZAL(U, V, x_buf, x_next_buf, x_next2_buf, it, Q_hist, dim)

        n_cols = min(it, Q_hist)
        qn_extrapolate_ZAL(x_qn, x_next_buf, x_buf, U, V, n_cols, dim, UtUmV_workspace, coeff_workspace)

        sqp.project_p_box(P_qn, M, K)
        sqp.project_q_simplex(Q_qn, N, K)

        # --- Conditional QN Acceptance (Pointer swapping, 0 memcpy) ---
        ll_qn = tools.loglikelihood(G, P_qn, Q_qn)
        accept_qn = ll_qn > ll_prev_iter
        ll_new = ll_qn if accept_qn else tools.loglikelihood(G, P_next2, Q_next2)

        best_before = ll_best
        if ll_new > ll_best:
            ll_best = ll_new
            best_in_cur = True
        elif best_in_cur:
            PQ, PQ_best = PQ_best, PQ
            P_cur, P_best = P_best, P_cur
            Q_cur, Q_best = Q_best, Q_cur
            x_buf, x_best = x_best, x_buf
            best_in_cur = False

        if accept_qn:
            PQ, PQ_qn = PQ_qn, PQ
            P_cur, P_qn = P_qn, P_cur
            Q_cur, Q_qn = Q_qn, Q_cur
            x_buf, x_qn = x_qn, x_buf
        else:
            PQ, PQ_next2 = PQ_next2, PQ
            P_cur, P_next2 = P_next2, P_cur
            Q_cur, Q_next2 = Q_next2, Q_cur
            x_buf, x_next2_buf = x_next2_buf, x_buf

        if best_before != -float("inf") and ll_new <= best_before + tol:
            wait += 1
        else:
            wait = 0

        if verbose:
            log.info(
                f"    Iteration {it}, "
                f"Log-likelihood: {ll_new:.1f}, "
                f"Time: {time.time() - it_start:.3f}s"
            )

        diff = ll_new - ll_prev_iter
        if abs(diff) < tol:
            if verbose:
                log.info(f"    Converged at iteration {it}.")
            break
        if wait >= patience:
            if verbose:
                log.info(f"    Converged at iteration {it} after {wait} iterations without improvement.")
            break

        ll_prev_iter = ll_new

    if best_in_cur:
        P_best, Q_best = P_cur, Q_cur

    if verbose:
        log.info(f"\n    Final log-likelihood: {ll_best:.1f}")
    P[:] = P_best
    Q[:] = Q_best
    return P_best, Q_best
