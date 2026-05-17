import logging
import sys

import numpy as np

from ..src.utils_c.cython import em, tools
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

def _flatten_PQ(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Description:
    Flattens P and Q into a single 1-D parameter vector [P_flat, Q_flat].

    Args:
        P (np.ndarray): P matrix (M x K).
        Q (np.ndarray): Q matrix (N x K).

    Returns:
        np.ndarray: Flattened 1-D vector of length M*K + N*K.
    """
    return np.concatenate([P.ravel(), Q.ravel()])

def _unflatten_PQ(x: np.ndarray, P_out: np.ndarray, Q_out: np.ndarray,
                  M: int, N: int, K: int) -> None:
    """
    Description:
    Unflattens a 1-D parameter vector back into pre-allocated P and Q matrices
    using memoryview for zero-copy speed.

    Args:
        x (np.ndarray): Flattened parameter vector of length M*K + N*K.
        P_out (np.ndarray): Pre-allocated output P matrix (M x K).
        Q_out (np.ndarray): Pre-allocated output Q matrix (N x K).
        M (int): Number of SNPs.
        N (int): Number of individuals.
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
               UtUmV_workspace: np.ndarray, coeff_workspace: np.ndarray) -> None:
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

    Returns:
        None: P and Q are updated in-place with the best result.
    """
    dim = M * K + N * K

    # --- Block-relaxation step 1: (P, Q) → (P1, Q1) ---
    brStep(G, P, Q, T, P1, Q1, q_bat, K, M, N)

    # --- Block-relaxation step 2: (P1, Q1) → (P2, Q2) ---
    brStep(G, P1, Q1, T, P2, Q2, q_bat, K, M, N)

    # --- Flatten for QN ---
    x = _flatten_PQ(P, Q)
    x_next = _flatten_PQ(P1, Q1)
    x_next2 = _flatten_PQ(P2, Q2)

    # --- Update UV history ---
    update_UV_ZAL(U, V, x, x_next, x_next2, iteration, Q_hist, dim)

    # --- QN extrapolation ---
    n_cols = min(iteration, Q_hist)
    x_qn = np.empty(dim, dtype=np.float64)

    qn_extrapolate_ZAL(x_qn, x_next, x, U, V, n_cols, dim, UtUmV_workspace, coeff_workspace)

    # Unflatten and project into P, Q
    _unflatten_PQ(x_qn, P, Q, M, N, K)
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
    P1 = np.zeros_like(P, dtype=np.float64)
    Q1 = np.zeros_like(Q, dtype=np.float64)
    P2 = np.zeros_like(P, dtype=np.float64)
    Q2 = np.zeros_like(Q, dtype=np.float64)
    T = np.zeros_like(Q, dtype=np.float64)
    q_bat = np.zeros(N, dtype=np.float64)

    # QN history buffers (flattened column-major: dim x Q_hist)
    dim = M * K + N * K
    U = np.zeros(dim * Q_hist, dtype=np.float64)
    V = np.zeros(dim * Q_hist, dtype=np.float64)

    for it in range(1, n_iters + 1):
        qnStep_ZAL(
            G, P, Q, T, P1, Q1, P2, Q2,
            q_bat, K, M, N,
            U, V, it, Q_hist,
            UtUmV_workspace, coeff_workspace
        )

    return P, Q
