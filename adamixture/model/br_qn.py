import logging
import sys
import time

import numpy as np

from ..src.utils_c.cython import em, sqp, tools
from ..src.utils_c.cython.br_qn import qn_extrapolate_ZAL, update_UV_ZAL

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def brStep(G: np.ndarray, P0: np.ndarray, Q0: np.ndarray, T: np.ndarray, P1: np.ndarray,
           Q1: np.ndarray, q_bat: np.ndarray, K: int, M: int, N: int) -> None:
    """
    Description:
    Performs a single block-relaxation step: EM P-step followed by EM Q-step.
    The results are written into P1, Q1. Unlike emStep in em_adam.py, this does
    NOT copy back into P0/Q0 (the caller decides what to keep).

    Args:
        G (np.ndarray): Input genotype matrix (M x N, uint8).
        P0 (np.ndarray): Current P matrix (M x K).
        Q0 (np.ndarray): Current Q matrix (N x K).
        T (np.ndarray): Temporary buffer for EM calculations (N x K).
        P1 (np.ndarray): Buffer for updated P matrix (M x K).
        Q1 (np.ndarray): Buffer for updated Q matrix (N x K).
        q_bat (np.ndarray): Batch-wise normalization buffer (N,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.

    Returns:
        None
    """
    em.P_step(G, P0, P1, Q0, T, q_bat, K, M, N)
    em.Q_step(Q0, Q1, T, q_bat, K, N)

def _flatten_PQ_inplace(P: np.ndarray, Q: np.ndarray, out: np.ndarray) -> None:
    """
    Description:
    Flattens P and Q in-place into a pre-allocated 1-D parameter vector.

    Args:
        P (np.ndarray): P matrix (M x K).
        Q (np.ndarray): Q matrix (N x K).
        out (np.ndarray): Pre-allocated flat output buffer.

    Returns:
        None
    """
    mk = P.size
    memoryview(out[:mk])[:] = P.ravel()
    memoryview(out[mk:])[:] = Q.ravel()

def _unflatten_PQ(x: np.ndarray, P_out: np.ndarray, Q_out: np.ndarray,
                  M: int, K: int) -> None:
    """
    Description:
    Unflattens a 1-D parameter vector back into pre-allocated P and Q matrices
    using memoryview for zero-copy speed.

    Args:
        x (np.ndarray): Flattened parameter vector of length M*K + N*K.
        P_out (np.ndarray): Pre-allocated output P matrix (M x K).
        Q_out (np.ndarray): Pre-allocated output Q matrix (N x K).
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.

    Returns:
        None
    """
    memoryview(P_out.ravel())[:] = memoryview(x[:M * K])
    memoryview(Q_out.ravel())[:] = memoryview(x[M * K:])

def qnStep_ZAL(G: np.ndarray, P: np.ndarray, Q: np.ndarray,
               T: np.ndarray, P1: np.ndarray, Q1: np.ndarray,
               P2: np.ndarray, Q2: np.ndarray,
               q_bat: np.ndarray, K: int, M: int, N: int,
               U: np.ndarray, V: np.ndarray,
               iteration: int, Q_hist: int,
               UtUmV_workspace: np.ndarray, coeff_workspace: np.ndarray,
               x_qn: np.ndarray = None,
               x_buf: np.ndarray = None,
               x_next_buf: np.ndarray = None,
               x_next2_buf: np.ndarray = None) -> None:
    """
    Description:
    Performs one full ZAL quasi-Newton iteration:
      1. Two consecutive block-relaxation EM steps to produce (P1, Q1) and (P2, Q2).
      2. Update U, V history matrices.
      3. Compute QN extrapolation.
      4. Project the QN result back onto feasible sets.

    Args:
        G (np.ndarray): Input genotype matrix (M x N, uint8).
        P (np.ndarray): Current P matrix (M x K). Updated in-place with the best result.
        Q (np.ndarray): Current Q matrix (N x K). Updated in-place with the best result.
        T (np.ndarray): Temporary buffer for EM calculations (N x K).
        P1 (np.ndarray): Buffer for first block-relaxation P result (M x K).
        Q1 (np.ndarray): Buffer for first block-relaxation Q result (N x K).
        P2 (np.ndarray): Buffer for second block-relaxation P result (M x K).
        Q2 (np.ndarray): Buffer for second block-relaxation Q result (N x K).
        q_bat (np.ndarray): Batch-wise normalization buffer (N,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        U (np.ndarray): QN history matrix U, flattened (dim x Q_hist), column-major.
        V (np.ndarray): QN history matrix V, flattened (dim x Q_hist), column-major.
        iteration (int): Current iteration number (1-based).
        Q_hist (int): Number of QN history columns.
        UtUmV_workspace (np.ndarray): Pre-allocated workspace for QN extrapolation.
        coeff_workspace (np.ndarray): Pre-allocated workspace for QN extrapolation.
        x_qn (np.ndarray, optional): Pre-allocated workspace for the extrapolated state.
        x_buf (np.ndarray, optional): Pre-allocated workspace for current state vector.
        x_next_buf (np.ndarray, optional): Pre-allocated workspace for next state vector.
        x_next2_buf (np.ndarray, optional): Pre-allocated workspace for next2 state vector.

    Returns:
        None: P and Q are updated in-place with the best result.
    """
    dim = M * K + N * K
    if x_qn is None:
        x_qn = np.empty(dim, dtype=np.float64)
    if x_buf is None:
        x_buf = np.empty(dim, dtype=np.float64)
    if x_next_buf is None:
        x_next_buf = np.empty(dim, dtype=np.float64)
    if x_next2_buf is None:
        x_next2_buf = np.empty(dim, dtype=np.float64)

    # --- Block-relaxation step 1: (P, Q) → (P1, Q1) ---
    brStep(G, P, Q, T, P1, Q1, q_bat, K, M, N)

    # --- Block-relaxation step 2: (P1, Q1) → (P2, Q2) ---
    brStep(G, P1, Q1, T, P2, Q2, q_bat, K, M, N)

    # --- Flatten for QN ---
    _flatten_PQ_inplace(P, Q, x_buf)
    _flatten_PQ_inplace(P1, Q1, x_next_buf)
    _flatten_PQ_inplace(P2, Q2, x_next2_buf)

    # --- Update UV history ---
    update_UV_ZAL(U, V, x_buf, x_next_buf, x_next2_buf, iteration, Q_hist, dim)

    # --- QN extrapolation ---
    n_cols = min(iteration, Q_hist)

    qn_extrapolate_ZAL(x_qn, x_next_buf, x_buf, U, V, n_cols, dim, UtUmV_workspace, coeff_workspace)

    # Unflatten and project into P, Q
    _unflatten_PQ(x_qn, P, Q, M, K)
    tools.mapP_d(P, M, K)
    tools.mapQ_d(Q, N, K)

def polish_br_qn(G: np.ndarray, P_init: np.ndarray, Q_init: np.ndarray,
                 M: int, N: int, K: int,
                 n_iters: int = 3, Q_hist: int = 3,
                 UtUmV_workspace: np.ndarray = None,
                 coeff_workspace: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Polishes P and Q matrices using block-relaxation with ZAL quasi-Newton
    acceleration. Designed for cross-validation fold polishing, warm-started
    from the global P and Q estimates.

    Args:
        G (np.ndarray): Genotype matrix with held-out entries masked as 3 (M x N, uint8).
        P_init (np.ndarray): Global P matrix (M x K) used as warm-start.
        Q_init (np.ndarray): Global Q matrix (N x K) used as warm-start.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        n_iters (int): Number of QN iterations (default 3).
        Q_hist (int): Number of QN history columns (default 3).
        UtUmV_workspace (np.ndarray, optional): Pre-allocated memory for UtUmV.
        coeff_workspace (np.ndarray, optional): Pre-allocated memory for coeff.

    Returns:
        tuple[np.ndarray, np.ndarray]: Polished (P, Q) matrices.
    """
    P = np.array(P_init, dtype=np.float64, copy=True)
    Q = np.array(Q_init, dtype=np.float64, copy=True)

    # Preallocate workspaces on the fly if they weren't provided
    if UtUmV_workspace is None:
        UtUmV_workspace = np.empty(Q_hist * (Q_hist + 1), dtype=np.float64)
    if coeff_workspace is None:
        coeff_workspace = np.empty(Q_hist, dtype=np.float64)

    # EM buffers
    P1 = np.empty_like(P, dtype=np.float64)
    Q1 = np.empty_like(Q, dtype=np.float64)
    P2 = np.empty_like(P, dtype=np.float64)
    Q2 = np.empty_like(Q, dtype=np.float64)
    T = np.zeros_like(Q, dtype=np.float64)
    q_bat = np.zeros(N, dtype=np.float64)

    # QN history buffers (flattened column-major: dim x Q_hist)
    dim = M * K + N * K
    U = np.zeros(dim * Q_hist, dtype=np.float64)
    V = np.zeros(dim * Q_hist, dtype=np.float64)
    x_qn = np.empty(dim, dtype=np.float64)
    x_buf = np.empty(dim, dtype=np.float64)
    x_next_buf = np.empty(dim, dtype=np.float64)
    x_next2_buf = np.empty(dim, dtype=np.float64)

    for it in range(1, n_iters + 1):
        qnStep_ZAL(
            G, P, Q, T, P1, Q1, P2, Q2,
            q_bat, K, M, N,
            U, V, it, Q_hist,
            UtUmV_workspace, coeff_workspace,
            x_qn, x_buf, x_next_buf, x_next2_buf
        )

    return P, Q


def optimize_original(G: np.ndarray, P: np.ndarray, Q: np.ndarray, max_iter: int,
                      K: int, M: int, N: int, tol: float, Q_hist: int,
                      patience: int) -> tuple[np.ndarray, np.ndarray]:
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

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized (P, Q) matrices.
    """
    # 1. Precompute Vt matrix from SVD of ones(1, K)
    _, _, vt = np.linalg.svd(np.ones((1, K)), full_matrices=True)
    v_kk = np.ascontiguousarray(vt.T, dtype=np.float64)

    # 2. Allocate buffers
    XtX_q = np.empty((N, K, K), dtype=np.float64)
    Xtz_q = np.empty((N, K), dtype=np.float64)
    XtX_p = np.empty((M, K, K), dtype=np.float64)
    Xtz_p = np.empty((M, K), dtype=np.float64)

    P_next = np.empty_like(P, dtype=np.float64)
    Q_next = np.empty_like(Q, dtype=np.float64)
    P_next2 = np.empty_like(P, dtype=np.float64)
    Q_next2 = np.empty_like(Q, dtype=np.float64)

    # QN history buffers
    dim = M * K + N * K
    U = np.zeros(dim * Q_hist, dtype=np.float64)
    V = np.zeros(dim * Q_hist, dtype=np.float64)
    UtUmV_workspace = np.zeros(Q_hist * (Q_hist + 1), dtype=np.float64)
    coeff_workspace = np.zeros(Q_hist, dtype=np.float64)

    # QN extrapolation buffers
    x_qn = np.empty(dim, dtype=np.float64)
    P_qn = np.empty_like(P)
    Q_qn = np.empty_like(Q)

    # Pre-allocated buffers for ZAL QN acceleration
    x_buf = np.empty(dim, dtype=np.float64)
    x_next_buf = np.empty(dim, dtype=np.float64)
    x_next2_buf = np.empty(dim, dtype=np.float64)

    # 3. Initialize log-likelihood
    ll_prev_iter = -float('inf')
    ll_best = -float("inf")
    wait = 0
    P_best = np.empty_like(P)
    Q_best = np.empty_like(Q)

    for it in range(1, max_iter + 1):
        it_start = time.time()

        # --- SQP Update 1: (P, Q) -> (P_next, Q_next) ---
        sqp.update_q_sqp(G, Q, Q_next, P, XtX_q, Xtz_q, v_kk, M, N, K)
        sqp.update_p_sqp(G, Q_next, P, P_next, XtX_p, Xtz_p, M, N, K)

        # --- SQP Update 2: (P_next, Q_next) -> (P_next2, Q_next2) ---
        sqp.update_q_sqp(G, Q_next, Q_next2, P_next, XtX_q, Xtz_q, v_kk, M, N, K)
        sqp.update_p_sqp(G, Q_next2, P_next, P_next2, XtX_p, Xtz_p, M, N, K)

        # --- ZAL QN acceleration ---
        _flatten_PQ_inplace(P, Q, x_buf)
        _flatten_PQ_inplace(P_next, Q_next, x_next_buf)
        _flatten_PQ_inplace(P_next2, Q_next2, x_next2_buf)

        update_UV_ZAL(U, V, x_buf, x_next_buf, x_next2_buf, it, Q_hist, dim)

        n_cols = min(it, Q_hist)
        qn_extrapolate_ZAL(x_qn, x_next_buf, x_buf, U, V, n_cols, dim, UtUmV_workspace, coeff_workspace)

        _unflatten_PQ(x_qn, P_qn, Q_qn, M, K)

        sqp.project_p_box(P_qn, M, K)
        sqp.project_q_simplex(Q_qn, N, K)

        # --- Conditional QN Acceptance ---
        ll_qn = tools.loglikelihood(G, P_qn, Q_qn)

        if ll_qn > ll_prev_iter:
            memoryview(P.ravel())[:] = memoryview(P_qn.ravel())
            memoryview(Q.ravel())[:] = memoryview(Q_qn.ravel())
            ll_new = ll_qn
        else:
            memoryview(P.ravel())[:] = memoryview(P_next2.ravel())
            memoryview(Q.ravel())[:] = memoryview(Q_next2.ravel())
            ll_new = tools.loglikelihood(G, P_next2, Q_next2)

        best_before = ll_best
        if ll_new > ll_best:
            ll_best = ll_new
            memoryview(P_best.ravel())[:] = memoryview(P.ravel())
            memoryview(Q_best.ravel())[:] = memoryview(Q.ravel())
        if best_before != -float("inf") and ll_new <= best_before + tol:
            wait += 1
        else:
            wait = 0

        log.info(
            f"    Iteration {it}, "
            f"Log-likelihood: {ll_new:.1f}, "
            f"Time: {time.time() - it_start:.3f}s"
        )

        diff = ll_new - ll_prev_iter
        if abs(diff) < tol:
            log.info(f"    Converged at iteration {it}.")
            break
        if wait >= patience:
            log.info(f"    Converged at iteration {it} after {wait} iterations without improvement.")
            break

        ll_prev_iter = ll_new

    log.info(f"\n    Final log-likelihood: {ll_best:.1f}")
    return P_best, Q_best
