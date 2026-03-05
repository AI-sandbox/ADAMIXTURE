import logging
import numpy as np
import sys
import time

from .em_adam import optimize_parameters
from ..src.svd import RSVD
from ..src.utils_c import tools

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

import time
import numpy as np
import logging

log = logging.getLogger(__name__)

def ALS(G: np.ndarray, U: np.ndarray, S: np.ndarray, V: np.ndarray, f: np.ndarray, 
        seed: int, M: int, N: int, K: int, max_iter: int, tole: float, reg: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Alternating Least Squares (ALS) algorithm with Tikhonov regularization 
    and proximal smoothing.

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

    Returns:
        tuple[np.ndarray, np.ndarray]: Final P and Q matrices.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    
    Z = np.ascontiguousarray(U * S)
    
    reg_eye = reg * np.eye(K, dtype=np.float32)
    mu = 0.05
    mu_eye = mu * np.eye(K, dtype=np.float32)

    # Init P:
    P = rng.random(size=(M, K), dtype=np.float32)
    tools.mapP(P, M, K)

    # Init Q:
    G_mat = P.T @ P
    G_reg = G_mat + reg_eye
    inv_G = np.linalg.pinv(G_reg)
    I = P @ inv_G
    Q = 0.5 * (V @ (Z.T @ I)) + (I * f[:, None]).sum(axis=0)
    tools.mapQ(Q, N, K)
    
    Q0 = np.empty_like(Q)
    memoryview(Q0.ravel())[:] = memoryview(Q.ravel())

    for i in range(max_iter):
        P0 = np.empty_like(P)
        memoryview(P0.ravel())[:] = memoryview(P.ravel())
        
        # === Update P ===
        G_mat = Q.T @ Q
        G_reg = G_mat + reg_eye + mu_eye
        inv_G = np.linalg.pinv(G_reg)
        I = Q @ inv_G
        
        P = 0.5 * (Z @ (V.T @ I)) + np.outer(f, I.sum(axis=0)) + (P0 @ inv_G * mu)
        tools.mapP(P, M, K)

        # === Update Q ===
        G_mat = P.T @ P
        G_reg = G_mat + reg_eye + mu_eye
        inv_G = np.linalg.pinv(G_reg)
        I = P @ inv_G
        
        Q = 0.5 * (V @ (Z.T @ I)) + (I * f[:, None]).sum(axis=0) + (Q0 @ inv_G * mu)
        tools.mapQ(Q, N, K)

        # === Check convergence ===
        rmse_error = tools.rmse(Q, Q0, N, K)        

        if rmse_error < tole:
            log.info(f"        Convergence at iteration {i+1}.")
            break
            
        memoryview(Q0.ravel())[:] = memoryview(Q.ravel())
    
    total_time = time.time() - t0
    log.info(f"    Total time for ALS={total_time:.4f}s\n")
    
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)
    
    logl = tools.loglikelihood(G, P, Q)
    log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.\n") 
    
    return P, Q

def train(G: np.ndarray, K: int, seed: int, lr: float, beta1: float, 
        beta2: float, reg_adam: float, max_iter: int, check: int,
        max_als: int, tole_als: float, power: int, tole_svd: float,
        reg_als: float, lr_decay: float, min_lr: float, chunk_size: int) -> tuple[np.ndarray, np.ndarray]:
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
    P, Q = ALS(G, U, S, V, f, seed, M, N, K, max_als, tole_als, reg_als)
    del U, S, V, f
    
    # ADAM EM:
    log.info("    Adam-EM running...\n")
    P, Q = optimize_parameters(G, P, Q, lr, beta1, beta2, reg_adam, max_iter, 
                            check, K, M, N, lr_decay, min_lr)
    del G

    return P, Q
