import logging
import sys

import numpy as np
import torch

from ..model.als import ALS
from ..model.als_gpu import ALS_gpu
from ..model.em_adam import optimize_parameters
from ..model.em_adam_gpu import optimize_parameters_gpu
from ..model.svd import RSVD
from ..model.svd_gpu import SVD_gpu
from . import utils
from .utils_c import tools

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def setup(G: torch.Tensor | np.ndarray, N: int, M: int, K_max: int, seed: int, power: int,
          tol_svd: float, chunk_size: int, device: str) -> tuple:
    """
    Description:
    One-time initialisation shared across all K values in a sweep:
    device setup, CUDA extension loading, GPU memory management,
    allele frequency calculation, and Randomized SVD with K_max.

    Args:
        G (torch.Tensor | np.ndarray): Input genotype matrix.
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K_max (int): Maximum K in the sweep (used for SVD rank and GPU memory estimate).
        seed (int): Random seed.
        power (int): Power iterations for Randomized SVD.
        tol_svd (float): Convergence tolerance for SVD.
        chunk_size (int): Chunk size for batched operations.
        device (str): Target device string ('cpu', 'cuda', 'mps').

    Returns:
        tuple: (device_obj, threads_per_block, f, U, S, V, G) where G may
               have been moved to GPU.
    """
    device_obj = torch.device(device)
    if device_obj.type == 'mps':
        try:
            import torch._inductor.config as inductor_config
            inductor_config.max_autotune_gemm = False
        except (ImportError, AttributeError):
            pass

    log.info(f"    Running on {str(device_obj).upper()}.\n")
    utils.load_extensions(device_obj)
    threads_per_block = utils.get_tuning_params(device_obj)

    if device_obj.type == 'cpu':
        f = np.zeros(M, dtype=np.float32)
        tools.alleleFrequency(G, f, M, N)
        log.info("    Frequencies calculated...\n")

        log.info("    Running SVD...\n")
        U, S, V = RSVD(G, N, M, f, K_max, seed, power, tol_svd, chunk_size)
    else:
        G = utils.manage_gpu_memory(G, device_obj, M, N, K_max, chunk_size)

        f = utils.calculate_frequencies_gpu(G, M, chunk_size, device_obj, threads_per_block)
        log.info("    Frequencies calculated.\n")

        log.info("    Running SVD on GPU...\n")
        U, S, V = SVD_gpu(G, N, M, f, K_max, seed, power, tol_svd,
                          chunk_size, device_obj, threads_per_block)

    return device_obj, threads_per_block, f, U, S, V, G


def train_k(G: torch.Tensor | np.ndarray, N: int, M: int, K: int, U_max: np.ndarray | torch.Tensor, S_max: np.ndarray | torch.Tensor,
        V_max: np.ndarray | torch.Tensor, f: np.ndarray | torch.Tensor, seed: int, lr: float, beta1: float, beta2: float, reg_adam: float,
        max_iter: int, check: int, max_als: int, tol_als: float, lr_decay: float, min_lr: float, chunk_size: int, patience_adam: int, tol_adam: float,
        device_obj: torch.device, threads_per_block: int) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Trains ADAMIXTURE for a single K value, using pre-computed SVD results.
    Slices U, S, V to rank K, runs ALS initialisation, then Adam-EM.

    Args:
        G (torch.Tensor | np.ndarray): Genotype matrix.
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations for this run.
        U_max (np.ndarray | torch.Tensor): Left singular vectors from setup (M x K_max).
        S_max (np.ndarray | torch.Tensor): Singular values from setup (K_max,).
        V_max (np.ndarray | torch.Tensor): Right singular vectors from setup (N x K_max).
        f (np.ndarray | torch.Tensor): Allele frequencies from setup.
        seed (int): Random seed.
        lr (float): Learning rate for Adam-EM.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        reg_adam (float): Adam epsilon.
        max_iter (int): Maximum Adam-EM iterations.
        check (int): Log-likelihood check frequency.
        max_als (int): Maximum ALS iterations.
        tol_als (float): ALS convergence tolerance.
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate.
        chunk_size (int): Chunk size for batched operations.
        patience_adam (int): Patience for Adam-EM plateau detection.
        tol_adam (float): Adam-EM convergence tolerance.
        device_obj (torch.device): Computation device.
        threads_per_block (int): CUDA threads per block.

    Returns:
        tuple: (P, Q) — numpy arrays on CPU, GPU tensors on CUDA/MPS.
    """
    U = U_max[:, :K]
    S = S_max[:K]
    V = V_max[:, :K]

    if device_obj.type == 'cpu':
        log.info("    Running ALS...")
        P, Q = ALS(U, S, V, f, seed, M, N, K, max_als, tol_als)

        logl = tools.loglikelihood(G, P, Q)
        log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.")

        log.info("    Adam-EM running on CPU...\n")
        P, Q = optimize_parameters(G, P, Q, lr, beta1, beta2, reg_adam, max_iter,
                                   check, K, M, N, lr_decay, min_lr, patience_adam, tol_adam)
    else:
        if device_obj.type == 'mps':
            log.info("    Running ALS on CPU (device is MPS)...")
            U_cpu = U.cpu().numpy()
            S_cpu = S.cpu().numpy()
            V_cpu = V.cpu().numpy()
            f_cpu = f.cpu().numpy()
            G_cpu = G.cpu().numpy() if isinstance(G, torch.Tensor) else G
            P_np, Q_np = ALS(U_cpu, S_cpu, V_cpu, f_cpu, seed, M, N, K, max_als, tol_als)
            P = torch.from_numpy(P_np).to(device_obj, dtype=torch.float32)
            Q = torch.from_numpy(Q_np).to(device_obj, dtype=torch.float32)
            del U_cpu, S_cpu, V_cpu, f_cpu, G_cpu, P_np, Q_np
        else:
            log.info("    Running ALS on GPU...")
            U_k = U.contiguous()
            S_k = S.contiguous()
            V_k = V.contiguous()
            P, Q = ALS_gpu(U_k, S_k, V_k, f, seed, M, K, max_als, tol_als, device_obj)

        if device_obj.type == 'cuda':
            torch.cuda.empty_cache()

        logl_calc = utils.get_logl_calculator(device_obj)
        logl = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
        log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.")

        log.info(f"    Adam-EM running on GPU ({device_obj})...\n")
        P, Q = optimize_parameters_gpu(G, P, Q, lr, beta1, beta2, reg_adam, max_iter,
                                       check, M, N, lr_decay, min_lr, patience_adam, tol_adam,
                                       device_obj, chunk_size, threads_per_block)
    return P, Q
