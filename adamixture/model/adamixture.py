import logging
import numpy as np
import sys
import time

from .em_adam import optimize_parameters
from ..src.svd import RSVD
from ..src.utils_c import tools

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def ALS(G: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray, f: np.ndarray, 
        seed: int, M: int, N: int, K: int, max_iter: int, tole: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Alternating Least Squares (ALS) algorithm with exact NNLS (KKT conditions) 
    via Block Principal Pivoting and correct biological projection.

    Args:
        G (np.ndarray): Input genotype matrix.
        U (np.ndarray): Left singular vectors from SVD.
        S (np.ndarray): Singular values from SVD.
        V (np.ndarray): Right singular vectors from SVD.
        f (np.ndarray): Allele frequencies.
        seed (int): Random seed for reproducibility.
        M (int): Number of individuals.
        N (int): Number of SNPs.
        K (int): Number of ancestral populations.
        max_iter (int): Maximum number of ALS iterations.
        tole (float): Convergence tolerance for ALS.

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
        tools.batch_nnls_bpp(A_cov_Q, B_target_P, P_buffer)
        tools.mapP_d(P_buffer, M, K)
        memoryview(P.ravel())[:] = memoryview(P_buffer.ravel())
        
        # UPDATE Q
        A_cov_P = np.ascontiguousarray(P.T @ P, dtype=np.float64)
        I_p = P @ np.linalg.pinv(A_cov_P)
        Q_free = 0.5 * (V @ (Z_T @ I_p)) + (f_64 @ I_p)
        B_target_Q = np.ascontiguousarray(Q_free @ A_cov_P, dtype=np.float64)
        
        Q_buffer.fill(0)
        tools.batch_nnls_bpp(A_cov_P, B_target_Q, Q_buffer)
        tools.mapQ_d(Q_buffer, N, K)
        memoryview(Q.ravel())[:] = memoryview(Q_buffer.ravel())
        
        rmse_error = tools.rmse_d(Q, Q0, N, K) 
        
        if rmse_error < tole:
            log.info(f"        Convergence reached in iteration {i+1}.")
            break
        else:
            memoryview(Q0.ravel())[:] = memoryview(Q.ravel())
    
    total_time = time.time() - t0
    log.info(f"        Total ALS time={total_time:.3f}s")
    logl = tools.loglikelihood(G, P, Q)
    log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.") 
    
    return P, Q

def train(G: np.ndarray, K: int, seed: int, lr: float, beta1: float, 
        beta2: float, reg_adam: float, max_iter: int, check: int,
        max_als: int, tole_als: float, power: int, tole_svd: float,
        lr_decay: float, min_lr: float, chunk_size: int,
        patience_adam: int, tol_adam: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Initializes P and Q matrices and trains the ADAMIXTURE model.

    Args:
        G (np.ndarray): Input genotype matrix.
        K (int): Number of ancestral populations.
        seed (int): Random seed for reproducibility.
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1 parameter.
        beta2 (float): Adam beta2 parameter.
        reg_adam (float): Adam epsilon for numerical stability.
        max_iter (int): Maximum number of Adam-EM iterations.
        check (int): Frequency of log-likelihood evaluation.
        max_als (int): Maximum number of ALS iterations.
        tole_als (float): Convergence tolerance for ALS.
        power (int): Number of power iterations for RSVD.
        tole_svd (float): Convergence tolerance for SVD.
        reg_als (float): Regularization parameter for ALS.
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate value.
        chunk_size (int): Number of SNPs in chunk operations for RSVD.
        correlation_als (float): Correlation threshold for ALS high-correlation check.
        stall_als (int): Maximum stall iterations for ALS.
        patience_adam (int): Early stopping patience for Adam-EM.
        tol_adam (float): Convergence tolerance for Adam-EM.

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    log.info("    Running initialization...\n")
    M, N = G.shape

    log.info("    Frequencies calculated...\n")
    f = np.zeros(M, dtype=np.float32)
    tools.alleleFrequency(G, f, M, N)
    
    # SVD + ALS:
    log.info("    Running RSVD...\n")
    U, S, V = RSVD(G, N, M, f, K, seed, power, tole_svd, chunk_size)
    log.info("    Running ALS...")
    P, Q = ALS(G, U, S, V, f, seed, M, N, K, max_als, tole_als)
    del U, S, V, f
    
    # ADAM EM:
    log.info("    Adam-EM running...\n")
    P, Q = optimize_parameters(G, P, Q, lr, beta1, beta2, reg_adam, max_iter, 
                            check, K, M, N, lr_decay, min_lr, patience_adam, tol_adam)
    del G

    return P, Q
