import time
import sys
import numpy as np
import logging

from ..src.utils_c import tools, em

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def adamStep(G: np.ndarray, P0: np.ndarray, Q0: np.ndarray, T: np.ndarray, P1: np.ndarray, 
            Q1: np.ndarray, q_bat: np.ndarray, K: int, M: int, N: int, m_P: np.ndarray, 
            v_P: np.ndarray, m_Q: np.ndarray, v_Q: np.ndarray, t: int, lr: float, beta1: float, 
            beta2: float, epsilon: np.ndarray):
    """
    Description:
    Performs a single EM step followed by an Adam optimization update for matrices P and Q.

    Args:
        G (np.ndarray): Input genotype matrix.
        P0 (np.ndarray): Current P matrix.
        Q0 (np.ndarray): Current Q matrix.
        T (np.ndarray): Temporary buffer for EM calculations.
        P1 (np.ndarray): Buffer for updated P matrix.
        Q1 (np.ndarray): Buffer for updated Q matrix.
        Q_bat (np.ndarray): Batch-wise normalization buffer.
        K (int): Number of components (clusters).
        M (int): Number of SNPs (rows in G).
        N (int): Number of individuals (columns in G).
        m_P (np.ndarray): First moment vector for P.
        v_P (np.ndarray): Second moment vector for P.
        m_Q (np.ndarray): First moment vector for Q.
        v_Q (np.ndarray): Second moment vector for Q.
        t (list): Current time step as a single-element list for in-place update.
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for the first moment.
        beta2 (float): Exponential decay rate for the second moment.
        epsilon (float): Numerical stability constant.

    Return:
        None
    """
    # EM Step:
    em.P_step(G, P0, P1, Q0, T, q_bat, K, M, N)
    em.Q_step(Q0, Q1, T, q_bat, K, N)
    
    # Adam update using the computed pseudo-gradients:
    t_val = t[0] + 1
    em.adamUpdateP(P0, P1, m_P, v_P, lr, beta1, beta2, epsilon, t_val, M, K)
    em.adamUpdateQ(Q0, Q1, m_Q, v_Q, lr, beta1, beta2, epsilon, t_val, N, K)
    
    # Update the global iteration counter for Adam bias correction:
    t[0] = t_val

def optimize_parameters(G: np.ndarray, P: np.ndarray, Q: np.ndarray, lr: float, 
                        beta1: float, beta2: float, reg_adam: float, max_iter: int,
                        check: int, K: int, M: int, N: int, lr_decay: float, min_lr: float):
    """
    Description:
    Optimizes the P and Q matrices using a hybrid approach of Adam-accelerated EM and 
    Quasi-Newton refinement.

    Args:
        G (np.ndarray): Input genotype matrix.
        P (np.ndarray): Initial P matrix (frequencies).
        Q (np.ndarray): Initial Q matrix (proportions).
        seed (int): Random seed for reproducibility.
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1 parameter.
        beta2 (float): Adam beta2 parameter.
        reg_adam (float): Adam epsilon for numerical stability.
        max_iter (int): Maximum number of Adam-EM iterations.
        check (int): Frequency of log-likelihood evaluation and checkpointing.
        K (int): Number of components (clusters).
        M (int): Number of SNPs (rows in G).
        N (int): Number of individuals (columns in G).
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate value.

    Return:
        Tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    # Adam state variables (first and second moments):
    m_P = np.zeros_like(P, dtype=np.float64)
    v_P = np.zeros_like(P, dtype=np.float64)
    m_Q = np.zeros_like(Q, dtype=np.float64)
    v_Q = np.zeros_like(Q, dtype=np.float64)
    t = [0]

    # Temporary buffers for EM calculations:
    P1 = np.zeros_like(P, dtype=np.float64)
    Q1 = np.zeros_like(Q, dtype=np.float64)
    T = np.zeros_like(Q, dtype=np.float64)
    q_bat = np.zeros(G.shape[1], dtype=np.float64)

    # Variables for tracking the best solution:
    P_best = P.copy()
    Q_best = Q.copy()
    L_best = float('-inf')
    no_improve = 0

    ts = time.time()
    # ---------- Adam-EM ----------
    for it in range(max_iter):
        adamStep(G, P, Q, T, P1, Q1, q_bat,K, M, N,m_P, v_P, m_Q, v_Q,
                t, lr, beta1, beta2, reg_adam)

        if (it + 1) % check == 0:
            L_cur = tools.loglikelihood(G, P, Q)
            log.info(
                f"    Iteration {it+1}, "
                f"Log-likelihood: {L_cur:.1f}, "
                f"Time: {time.time() - ts:.3f}s"
            )
            ts = time.time()
            if abs(L_cur - L_best) < 0.1:
                break
            if L_cur > L_best:
                L_best = L_cur
                np.copyto(P_best, P)
                np.copyto(Q_best, Q)
                no_improve = 0
            else:
                no_improve += 1
                old_lr = lr
                lr = max(lr * lr_decay, min_lr)
                log.info(
                    f"    No improvement ({no_improve}). "
                    f"Reducing lr: {old_lr:.3e} → {lr:.3e}"
                )

            if no_improve >= 2:
                log.info("    Early stopping triggered.")
                break

    L_final = tools.loglikelihood(G, P_best, Q_best)
    log.info(f"    Log-likelihood after final EM: {L_final:.1f}")

    return P_best, Q_best
