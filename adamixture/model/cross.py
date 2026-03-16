import logging
import sys
import time
import numpy as np

from ..src.utils_c import tools, em

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def crossAdamStep(G: np.ndarray, P: np.ndarray, Q: np.ndarray, 
                  P1: np.ndarray, Q1: np.ndarray, Q_T: np.ndarray, 
                  q_bat: np.ndarray, K: int, M: int,
                  m_P: np.ndarray, v_P: np.ndarray, m_Q: np.ndarray, 
                  v_Q: np.ndarray, t: list, lr: float, 
                  beta1: float, beta2: float, epsilon: float, 
                  s_ind: np.ndarray) -> None:
    """
    One Adam-EM iteration on the training subset.
    Updates both P and Q using only the samples in s_ind.

    Args:
        G (np.ndarray): Genotype matrix (M x N), dtype uint8.
        P (np.ndarray): Allele frequency matrix (M x K).
        Q (np.ndarray): Admixture proportions matrix (N x K).
        P1 (np.ndarray): Buffer for updated P matrix (M x K).
        Q1 (np.ndarray): Buffer for updated Q matrix (N x K).
        Q_T (np.ndarray): Temporary accumulator for Q update (N_trn x K).
        q_bat (np.ndarray): Batch normalization buffer (N_trn,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        m_P (np.ndarray): First moment estimate for P (M x K).
        v_P (np.ndarray): Second moment estimate for P (M x K).
        m_Q (np.ndarray): First moment estimate for Q (N x K).
        v_Q (np.ndarray): Second moment estimate for Q (N x K).
        t (list): Single-element list with current Adam timestep.
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for first moment.
        beta2 (float): Exponential decay rate for second moment.
        epsilon (float): Numerical stability constant.
        s_ind (np.ndarray): Training sample indices, dtype uint32.
    """
    N_sub = s_ind.shape[0]
    
    # EM step on subset
    em.P_step_cross(G, P, P1, Q, Q_T, q_bat, K, M, s_ind)
    em.Q_step_cross(Q, Q1, Q_T, q_bat, K, s_ind)
    
    # Adam update using the computed pseudo-gradients
    t_val = t[0] + 1
    em.adamUpdateP(P, P1, m_P, v_P, lr, beta1, beta2, epsilon, t_val, M, K)
    em.adamUpdateQ(Q, Q1, m_Q, v_Q, lr, beta1, beta2, epsilon, t_val, Q.shape[0], K)
    
    t[0] = t_val

def crossProjAdamStep(G: np.ndarray, P: np.ndarray, Q: np.ndarray, 
                      Q1: np.ndarray, Q_T: np.ndarray, q_bat: np.ndarray, 
                      K: int, M: int,
                      m_Q: np.ndarray, v_Q: np.ndarray, t: list, 
                      lr: float, beta1: float, beta2: float, 
                      epsilon: float, s_ind: np.ndarray) -> None:
    """
    One Adam-EM iteration for Q-only projection on the test subset.
    P is fixed; only Q rows for test samples are updated.

    Args:
        G (np.ndarray): Genotype matrix (M x N), dtype uint8.
        P (np.ndarray): Fixed allele frequency matrix (M x K).
        Q (np.ndarray): Admixture proportions matrix (N x K).
        Q1 (np.ndarray): Buffer for updated Q matrix (N x K).
        Q_T (np.ndarray): Temporary accumulator for Q update (N_tst x K).
        q_bat (np.ndarray): Batch normalization buffer (N_tst,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        m_Q (np.ndarray): First moment estimate for Q (N x K).
        v_Q (np.ndarray): Second moment estimate for Q (N x K).
        t (list): Single-element list with current Adam timestep.
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for first moment.
        beta2 (float): Exponential decay rate for second moment.
        epsilon (float): Numerical stability constant.
        s_ind (np.ndarray): Test sample indices, dtype uint32.
    """
    # EM projection step (Q only)
    em.Q_proj_step(G, P, Q, Q1, Q_T, q_bat, K, M, s_ind)
    
    # Adam update for Q only
    t_val = t[0] + 1
    em.adamUpdateQ(Q, Q1, m_Q, v_Q, lr, beta1, beta2, epsilon, t_val, Q.shape[0], K)
    
    t[0] = t_val

def crossSteps(G: np.ndarray, P: np.ndarray, Q: np.ndarray, 
               P1: np.ndarray, Q1: np.ndarray, Q_T: np.ndarray, 
               q_bat: np.ndarray, K: int, M: int,
               s_ind: np.ndarray) -> None:
    """
    One regular EM iteration on the training subset.
    Updates both P and Q using only the samples in s_ind.
    """
    em.P_step_cross(G, P, P1, Q, Q_T, q_bat, K, M, s_ind)
    em.Q_step_cross(Q, Q1, Q_T, q_bat, K, s_ind)
    memoryview(P.ravel())[:] = memoryview(P1.ravel())
    memoryview(Q.ravel())[:] = memoryview(Q1.ravel())

def crossProjSteps(G: np.ndarray, P: np.ndarray, Q: np.ndarray, 
                   Q1: np.ndarray, Q_T: np.ndarray, q_bat: np.ndarray, 
                   K: int, M: int, s_ind: np.ndarray) -> None:
    """
    One regular EM iteration for Q-only projection on the test subset.
    P is fixed; only Q rows for test samples are updated.
    """
    em.Q_proj_step(G, P, Q, Q1, Q_T, q_bat, K, M, s_ind)
    memoryview(Q.ravel())[:] = memoryview(Q1.ravel())

def crossRun(G: np.ndarray, P: np.ndarray, Q: np.ndarray, K: int, 
             seed: int, lr: float, beta1: float, beta2: float, 
             reg_adam: float, max_iter: int, check: int, cross: int, 
             lr_decay: float, min_lr: float, cv_tole: float,
             patience_adam: int, tol_adam: float) -> dict:
    """
    Perform K-fold cross-validation.
    
    Args:
        G (np.ndarray): Genotype matrix (M x N), dtype uint8.
        P (np.ndarray): Trained allele frequency matrix (M x K).
        Q (np.ndarray): Trained admixture proportions matrix (N x K).
        K (int): Number of ancestral populations.
        seed (int): Random seed for index shuffling.
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1 (1st moment decay).
        beta2 (float): Adam beta2 (2nd moment decay).
        reg_adam (float): Adam epsilon for numerical stability.
        max_iter (int): Maximum iterations per fold phase.
        check (int): Frequency of convergence checks.
        cross (int): Number of folds.
        lr_decay (float): Learning rate decay factor on no-improvement.
        min_lr (float): Minimum learning rate value.
        cv_tole (float): Convergence tolerance for log-likelihood.
        patience_adam (int): Early stopping patience.
        tol_adam (float): Convergence tolerance for Adam-EM.
    
    Returns:
        dict: {'avg': float, 'std': float} with cross-validation deviance error.
    """
    M, N = G.shape
    rng = np.random.default_rng(seed)
    v_crv = np.zeros(cross)
    
    # Shuffled sample indices
    s_all = rng.permutation(N).astype(np.uint32)
    N_crv = N // cross
    
    ts = time.time()
    log.info(f"\n    Performing cross-validation using {cross} folds.")
    
    for c in range(cross):
        log.info(f"\n    Fold {c + 1}/{cross}\n")
        
        # Split indices into train and test
        s_beg = c * N_crv
        s_end = N if c == (cross - 1) else (c + 1) * N_crv
        s_trn = np.ascontiguousarray(np.concatenate((s_all[:s_beg], s_all[s_end:])), dtype=np.uint32)
        s_tst = np.ascontiguousarray(s_all[s_beg:s_end], dtype=np.uint32)
        N_trn = s_trn.shape[0]
        N_tst = s_tst.shape[0]
        
        # Copy original solutions
        P_crv = np.empty_like(P)
        Q_crv = np.empty_like(Q)
        memoryview(P_crv.ravel())[:] = memoryview(P.ravel())
        memoryview(Q_crv.ravel())[:] = memoryview(Q.ravel())
        
        # ----- Phase 1: Train on training set (P and Q) -----
        log.info(f"        1) Training on training set (P and Q)\n")
        # Adam state variables
        m_P = np.zeros_like(P_crv)
        v_P = np.zeros_like(P_crv)
        m_Q = np.zeros_like(Q_crv)
        v_Q = np.zeros_like(Q_crv)
        t = [0]
        
        # EM buffers
        P1 = np.zeros_like(P_crv)
        Q1 = np.zeros_like(Q_crv)
        Q_T = np.zeros((N_trn, K), dtype=np.float64)
        q_bat = np.zeros(N_trn, dtype=np.float64)
        
        # Best-solution tracking for training
        P_best = np.empty_like(P_crv)
        Q_best = np.empty_like(Q_crv)
        L_best_trn = float('-inf')
        no_improve_trn = 0
        fold_lr = lr
        ts_fold = time.time()

        # Initial EM step for training
        crossSteps(G, P_crv, Q_crv, P1, Q1, Q_T, q_bat, K, M, s_trn)
        L_best_trn = tools.loglike_cross(G, P_crv, Q_crv, s_trn, M, N_trn, K)
        memoryview(P_best.ravel())[:] = memoryview(P_crv.ravel())
        memoryview(Q_best.ravel())[:] = memoryview(Q_crv.ravel())
        
        for it in range(max_iter):
            crossAdamStep(G, P_crv, Q_crv, P1, Q1, Q_T, q_bat, K, M,
                          m_P, v_P, m_Q, v_Q, t, fold_lr, beta1, beta2, reg_adam, s_trn)
            crossSteps(G, P_crv, Q_crv, P1, Q1, Q_T, q_bat, K, M, s_trn)
            
            if (it + 1) % check == 0:
                L_cur = tools.loglike_cross(G, P_crv, Q_crv, s_trn, M, N_trn, K)
                log.info(
                    f"        Iteration {it+1}, "
                    f"Log-likelihood: {L_cur:.1f}, "
                    f"Time: {time.time() - ts_fold:.3f}s"
                )
                ts_fold = time.time()
                diff = abs(L_cur - L_best_trn)
                if L_cur > L_best_trn:
                    L_best_trn = L_cur
                    memoryview(P_best.ravel())[:] = memoryview(P_crv.ravel())
                    memoryview(Q_best.ravel())[:] = memoryview(Q_crv.ravel())
                    no_improve_trn = 0
                else:
                    no_improve_trn += 1
                    old_lr = fold_lr
                    fold_lr = max(fold_lr * lr_decay, min_lr)
                    log.info(
                        f"        No improvement. "
                        f"Reducing lr: {old_lr:.3e} → {fold_lr:.3e}"
                    )

                if diff < tol_adam:
                    break

                if no_improve_trn >= patience_adam:
                    log.info("\n        Early stopping triggered.")
                    break
        
        log.info(f"\n        Final training log-likelihood: {L_best_trn:.1f}\n")
        
        # Revert to best training solution
        memoryview(P_crv.ravel())[:] = memoryview(P_best.ravel())
        memoryview(Q_crv.ravel())[:] = memoryview(Q_best.ravel())
        
        # ----- Phase 2: Project Q on test set (P fixed) -----
        log.info(f"\n        2) Projecting Q on test set (P fixed)\n")
        # Reset Adam state for Q projection
        m_Q_proj = np.zeros_like(Q_crv)
        v_Q_proj = np.zeros_like(Q_crv)
        Q1_proj = np.zeros_like(Q_crv)
        Q_T_proj = np.zeros((N_tst, K), dtype=np.float64)
        q_bat_proj = np.zeros(N_tst, dtype=np.float64)
        t_proj = [0]
        
        # Best-solution tracking for projection
        Q_best_proj = np.empty_like(Q_crv)
        L_best_proj = float('-inf')
        no_improve_proj = 0
        proj_lr = lr

        # Initial EM step for projection
        crossProjSteps(G, P_crv, Q_crv, Q1_proj, Q_T_proj, q_bat_proj, K, M, s_tst)
        L_best_proj = tools.loglike_cross(G, P_crv, Q_crv, s_tst, M, N_tst, K)
        memoryview(Q_best_proj.ravel())[:] = memoryview(Q_crv.ravel())
        
        for it in range(max_iter):
            crossProjAdamStep(G, P_crv, Q_crv, Q1_proj, Q_T_proj, q_bat_proj, K, M,
                              m_Q_proj, v_Q_proj, t_proj, proj_lr, beta1, beta2, reg_adam, s_tst)
            crossProjSteps(G, P_crv, Q_crv, Q1_proj, Q_T_proj, q_bat_proj, K, M, s_tst)
            
            if (it + 1) % check == 0:
                L_cur_tst = tools.loglike_cross(G, P_crv, Q_crv, s_tst, M, N_tst, K)
                log.info(
                    f"        Iteration {it+1}, "
                    f"Log-likelihood: {L_cur_tst:.1f}, "
                    f"Time: {time.time() - ts_fold:.3f}s"
                )
                ts_fold = time.time()
                diff = abs(L_cur_tst - L_best_proj)
                if L_cur_tst > L_best_proj:
                    L_best_proj = L_cur_tst
                    memoryview(Q_best_proj.ravel())[:] = memoryview(Q_crv.ravel())
                    no_improve_proj = 0
                else:
                    no_improve_proj += 1
                    old_lr = proj_lr
                    proj_lr = max(proj_lr * lr_decay, min_lr)
                    log.info(
                        f"        No improvement."
                        f"Reducing lr: {old_lr:.3e} → {proj_lr:.3e}"
                    )

                if diff < tol_adam:
                    break

                if no_improve_proj >= patience_adam:
                    log.info("\n        Early stopping triggered.")
                    break

        log.info(f"\n        Final projection log-likelihood: {L_best_proj:.1f}")
        
        # Revert to best projection solution
        memoryview(Q_crv.ravel())[:] = memoryview(Q_best_proj.ravel())
        
        # ----- Score: deviance on test set -----
        L_nrm_tst = tools.cross_norm(G, s_tst, M, N_tst)
        v_crv[c] = tools.deviance(G, P_crv, Q_crv, s_tst, M, N_tst, K) / L_nrm_tst
    
    elapsed = time.time() - ts
    log.info(f"\n    Cross-validation completed in {elapsed:.1f}s: error (SD) {np.mean(v_crv):.5f} ({np.std(v_crv):.5f})\n")
    
    res = {"avg": np.mean(v_crv), "std": np.std(v_crv)}
    return res
