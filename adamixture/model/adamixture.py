import logging
import numpy as np
import sys
import torch

from .em_adam import optimize_parameters
from .em_adam_gpu import optimize_parameters_gpu, loglikelihood_gpu
from .als import ALS
from .als_gpu import ALS_gpu
from ..src.svd import RSVD
from ..src.svd_gpu import SVD_gpu
from ..src.utils_c import tools
from ..src import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def train(G: torch.Tensor | np.ndarray, N: int, M: int, K: int, seed: int, lr: float, beta1: float, 
          beta2: float, reg_adam: float, max_iter: int, check: int,
          max_als: int, tol_als: float, power: int, tol_svd: float,
          lr_decay: float, min_lr: float, chunk_size: int,
          patience_adam: int, tol_adam: float, device: str = 'cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Initializes P and Q matrices and trains the ADAMIXTURE model.

    Args:
        G (torch.Tensor | np.ndarray): Input genotype matrix.
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.
        seed (int): Random seed for reproducibility.
        lr (float): Learning rate for Adam-EM.
        beta1 (float): Beta1 parameter for Adam.
        beta2 (float): Beta2 parameter for Adam.
        reg_adam (float): Regularization term for Adam.
        max_iter (int): Maximum number of iterations for Adam-EM.
        check (int): Frequency of log-likelihood checks.
        max_als (int): Maximum iterations for ALS.
        tol_als (float): Tolerance for ALS convergence.
        power (int): Power iterations for Randomized SVD.
        tol_svd (float): Tolerance for SVD convergence.
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate.
        chunk_size (int): Size of chunks for GPU/memory processing.
        patience_adam (int): Patience for Adam-EM convergence.
        tol_adam (float): Tolerance for Adam-EM convergence.
        device (str, optional): Target device ('cpu', 'gpu', 'mps'). Defaults to 'cpu'.

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    device_obj = torch.device(device)
    log.info(f"    Running on {str(device_obj).upper()}.\n")
    utils.load_extensions(device_obj)
    threads_per_block = utils.get_tuning_params(device_obj)

    if device_obj.type == 'cpu':
        log.info("    Frequencies calculated...\n")
        f = np.zeros(M, dtype=np.float32)
        tools.alleleFrequency(G, f, M, N)
        
        # SVD + ALS:
        log.info("    Running SVD...\n")
        U, S, V = RSVD(G, N, M, f, K, seed, power, tol_svd, chunk_size)
        log.info("    Running ALS...")
        P, Q = ALS(G, U, S, V, f, seed, M, N, K, max_als, tol_als)
        del U, S, V, f

        # Initial Log-Likelihood
        logl = tools.loglikelihood(G, P, Q)
        log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.") 
        
        # ADAM EM:
        log.info("    Adam-EM running on CPU...\n")
        P, Q = optimize_parameters(G, P, Q, lr, beta1, beta2, reg_adam, max_iter, 
                                 check, K, M, N, lr_decay, min_lr, patience_adam, tol_adam)
    else:
        # GPU Path
        G = utils.manage_gpu_memory(G, device_obj, M, N, K, chunk_size)
        
        # Calculate frequencies in chunks
        f_torch = utils.calculate_frequencies_gpu(G, M, chunk_size, device_obj, threads_per_block)
        log.info("    Frequencies calculated.\n")
        
        # SVD + ALS:
        log.info("    Running SVD on GPU...\n")
        U, S, V = SVD_gpu(G, N, M, f_torch, K, seed, power, tol_svd, chunk_size, device_obj, threads_per_block)
        
        if device_obj.type == 'mps':
            log.info("    Running ALS on CPU (device is MPS)...")
            U_cpu, S_cpu, V_cpu, f_cpu = U.cpu().numpy(), S.cpu().numpy(), V.cpu().numpy(), f_torch.cpu().numpy()
            G_cpu = G.cpu().numpy() if isinstance(G, torch.Tensor) else G
            P_np, Q_np = ALS(G_cpu, U_cpu, S_cpu, V_cpu, f_cpu, seed, M, N, K, max_als, tol_als)
            P = torch.from_numpy(P_np).to(device_obj, dtype=torch.float32)
            Q = torch.from_numpy(Q_np).to(device_obj, dtype=torch.float32)
        else:
            log.info("    Running ALS on GPU...")
            P, Q = ALS_gpu(G, U, S, V, f_torch, seed, M, N, K, max_als, tol_als, device_obj, threads_per_block, chunk_size)
        
        # Initial Log-Likelihood
        logl_calc = utils.get_logl_calculator(device_obj)
        logl = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
        
        log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.") 
        
        torch.set_float32_matmul_precision('medium')
        torch.set_flush_denormal(True)

        # ADAM EM:
        log.info(f"    Adam-EM running on GPU ({device})...\n")
        P, Q = optimize_parameters_gpu(G, P, Q, lr, beta1, beta2, reg_adam, max_iter, 
                                check, K, M, N, lr_decay, min_lr, patience_adam, tol_adam, 
                                device_obj, chunk_size, threads_per_block)
    
    return P, Q
