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
        seed: int, M: int, N: int, K: int, max_iter: int, tole: float, reg: float,
        stall_max=20) -> tuple[np.ndarray, np.ndarray]:
    """
    Alternating Least Squares (ALS) algorithm with HighCorr check.

    Args:
        G (np.ndarray): Input genotype matrix.
        U (np.ndarray): Left singular vectors from RSVD.
        S (np.ndarray): Singular values from RSVD.
        V (np.ndarray): Right singular vectors from RSVD.
        f (np.ndarray): Allele frequencies.
        seed (int): Random seed for reproducibility.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        max_iter (int): Maximum number of ALS iterations.
        tole (float): Convergence tolerance for ALS.
        reg (float): Regularization parameter for ALS.
        stall_max (int, optional): Maximum number of iterations without improvement before stopping. Defaults to 20.

    Returns:
        tuple[np.ndarray, np.ndarray]: Initialized P and Q matrices.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    Z = np.ascontiguousarray(U * S)
    reg_eye = (reg * np.eye(K)).astype(np.float32)

    # Init P:
    P = rng.random(size=(M, K), dtype=np.float32).clip(min=1e-5, max=1-(1e-5))
    I = P @ np.linalg.pinv(P.T @ P + reg_eye)
    
    # Init Q:
    Q = 0.5 * (V @ (Z.T @ I)) + (I * f[:, None]).sum(axis=0)
    tools.mapQ(Q, N, K)
    Q0 = Q.copy()

    rmse_best = np.inf
    stall_counter = 0
    high_corr = False
    P_best = P.copy()
    Q_best = Q.copy()
    
    for i in range(max_iter):
        # Update P
        I = Q @ np.linalg.pinv(Q.T @ Q + reg_eye)
        P = 0.5 * (Z @ (V.T @ I)) + np.outer(f, I.sum(axis=0))
        tools.mapP(P, M, K)

        # Update Q
        G_p = P.T @ P
        I = P @ np.linalg.pinv(G_p + reg_eye)
        Q = 0.5 * (V @ (Z.T @ I)) + (I * f[:, None]).sum(axis=0)
        tools.mapQ(Q, N, K)
        rmse_error = tools.rmse(Q, Q0, N, K)        

        if not high_corr:
            v = np.sqrt(np.diag(G_p))
            denom = np.outer(v, v)
            denom[denom == 0] = 1e-10 
            C = G_p / denom
            np.fill_diagonal(C, 0)
            max_corr = np.max(np.abs(C))
            
            if max_corr > 0.95:
                high_corr = True
                
        if high_corr:
            if rmse_error < rmse_best:
                rmse_best = rmse_error
                P_best = P.copy()
                Q_best = Q.copy()
                stall_counter = 0
            else:
                stall_counter += 1
            
            if stall_counter >= stall_max:
                log.info(f"        Stall limit reached ({stall_max}) at iter {i+1}. Reverting to best P, Q.")
                P = P_best
                Q = Q_best
                break
        if rmse_error < tole:
            log.info(f"        Convergence reached in iteration {i+1}.")
            break
        else:
            np.copyto(Q0, Q)
    
    total_time = time.time() - t0
    log.info(f"        Total ALS time={total_time:.4f}s")
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)
    logl = tools.loglikelihood(G, P, Q)
    log.info(f"    Initial log-likelihood for K={K}: {logl:2f}.") 
    return P, Q

def train(G: np.ndarray, K: int, seed: int, lr: float, beta1: float, 
        beta2: float, reg_adam: float, max_iter: int, check: int,
        max_als: int, tole_als: float, power: int, tole_svd: float,
        reg_als: float, lr_decay: float, min_lr: float) -> tuple[np.ndarray, np.ndarray]:
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

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    log.info("    Running initialization...")
    log.info("\n")
    M, N = G.shape

    log.info("    Frequencies calculated...")
    f = np.zeros(M, dtype=np.float32)
    tools.alleleFrequency(G, f, M, N)
    
    # SVD + ALS:
    log.info("    Running RSVD...")
    log.info("\n")
    U, S, V = RSVD(G, N, M, f, K, seed, power, tole_svd)
    log.info("    Running ALS...")
    P, Q = ALS(G, U, S, V, f, seed, M, N, K, max_als, tole_als, reg_als)
    del U, S, V, f
    
    # ADAM EM:
    log.info("    Adam expectation maximization running...")
    log.info("")
    P, Q = optimize_parameters(G, P, Q, lr, beta1, beta2, reg_adam, max_iter, 
                            check, K, M, N, lr_decay, min_lr)
    del G

    return P, Q