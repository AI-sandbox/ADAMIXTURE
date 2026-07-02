import logging
import sys
import time

import numpy as np
import torch

from ..model.als import ALS
from ..model.als_gpu import ALS_gpu
from ..model.br_qn import optimize_original
from ..model.br_qn_gpu import optimize_original_gpu
from ..model.em_adam import emStep, optimize_parameters
from ..model.em_adam_gpu import EMAdamOptimizer, optimize_parameters_gpu
from ..model.svd import RSVD
from ..model.svd_gpu import SVD_gpu
from . import utils
from .utils_c import tools

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def setup(G: torch.Tensor | np.ndarray, N: int, M: int, K_max: int, seed: int, power: int,
          tol_svd: float, chunk_size: int, device: str, original: bool = False,
          init_original: str = 'em', q_hist: int = 3) -> tuple:
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
        original (bool): If True, run original ADMIXTURE after initialization.
        init_original (str): Initialization used by --original ('em' or 'als').
        q_hist (int): ZAL-QN history depth used when original is True.

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

    if original and init_original == 'em':
        if device_obj.type != 'cpu':
            G = utils.manage_gpu_memory(
                G,
                device_obj,
                M,
                N,
                K_max,
                chunk_size,
                'brqn',
                q_hist,
                include_initialization=False,
            )
        log.info("    Skipping SVD; --original uses random + EM initialization.\n")
        return device_obj, threads_per_block, None, None, None, None, G

    if device_obj.type == 'cpu':
        f = utils.calculate_frequencies_cpu(G, M, N, chunk_size)
        log.info("    Frequencies calculated...\n")
        log.info("    Running SVD...\n")
        U, S, V = RSVD(G, N, M, f, K_max, seed, power, tol_svd, chunk_size)
    else:
        algorithm = 'brqn' if original else 'adamem'
        G = utils.manage_gpu_memory(G, device_obj, M, N, K_max, chunk_size, algorithm, q_hist)

        f = utils.calculate_frequencies_gpu(G, M, chunk_size, device_obj, threads_per_block)
        log.info("    Frequencies calculated.\n")
        log.info("    Running SVD on GPU...\n")
        U, S, V = SVD_gpu(G, N, M, f, K_max, seed, power, tol_svd,
                          chunk_size, device_obj, threads_per_block)

    return device_obj, threads_per_block, f, U, S, V, G


def initialize_em_cpu(G: np.ndarray, seed: int, M: int, N: int, K: int,
                      n_steps: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Initializes P and Q randomly on CPU and runs a few EM priming steps.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        seed (int): Random seed.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        n_steps (int): Number of EM priming steps. Defaults to 5.

    Returns:
        tuple[np.ndarray, np.ndarray]: Initialized (P, Q) matrices.
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    P = rng.random(size=(M, K), dtype=np.float64)
    Q = rng.random(size=(N, K), dtype=np.float64)
    tools.mapP_d(P, M, K)
    tools.mapQ_d(Q, N, K)

    P_next = np.empty_like(P)
    Q_next = np.empty_like(Q)
    T = np.zeros_like(Q)
    q_bat = np.zeros(N, dtype=np.float64)

    log.info(f"    Random initialization + {n_steps} EM initial steps...")
    for _ in range(n_steps):
        emStep(G, P, Q, T, P_next, Q_next, q_bat, K, M, N)

    log.info(f"        Total EM initialization time={time.time() - t0:.3f}s\n")
    return P, Q


def initialize_em_gpu(G: torch.Tensor, seed: int, M: int, N: int, K: int,
                      device_obj: torch.device, chunk_size: int, threads_per_block: int,
                      n_steps: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Initializes P and Q randomly on GPU and runs a few EM priming steps.

    Args:
        G (torch.Tensor): Packed genotype tensor.
        seed (int): Random seed.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        device_obj (torch.device): Computation device.
        chunk_size (int): SNP chunk size for batched GPU work.
        threads_per_block (int): CUDA tuning parameter.
        n_steps (int): Number of EM priming steps. Defaults to 5.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Initialized (P, Q) tensors.
    """
    t0 = time.time()
    dtype = utils.get_dtype(device_obj)
    rng = np.random.default_rng(seed)
    P_np = rng.random(size=(M, K), dtype=np.float64)
    Q_np = rng.random(size=(N, K), dtype=np.float64)
    np.clip(P_np, 1e-5, 1.0 - 1e-5, out=P_np)
    np.clip(Q_np, 1e-5, 1.0 - 1e-5, out=Q_np)
    Q_np /= Q_np.sum(axis=1, keepdims=True)
    P = torch.from_numpy(P_np).to(device=device_obj, dtype=dtype)
    Q = torch.from_numpy(Q_np).to(device=device_obj, dtype=dtype)

    optimizer = EMAdamOptimizer(P.shape, Q.shape, 0.0, 0.0, 0.0, 0.0, device_obj)
    unpacker = utils.get_unpacker(device_obj, threads_per_block)

    log.info(f"    Random initialization + {n_steps} EM initial steps...")
    for _ in range(n_steps):
        optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)
        P.copy_(optimizer.P_EM)
        Q.copy_(optimizer.Q_EM)

    log.info(f"        Total EM initialization time={time.time() - t0:.3f}s\n")
    return P, Q


def train_k(G: torch.Tensor | np.ndarray, N: int, M: int, K: int, U_max: np.ndarray | torch.Tensor, S_max: np.ndarray | torch.Tensor,
        V_max: np.ndarray | torch.Tensor, f: np.ndarray | torch.Tensor, seed: int, lr: float, beta1: float, beta2: float, reg_adam: float,
        max_iter: int, check: int, max_als: int, tol_als: float, lr_decay: float, min_lr: float, chunk_size: int, patience: int, tol_adam: float,
        device_obj: torch.device, threads_per_block: int, original: bool = False, rtol: float = 1e-7, Q_hist: int = 3,
        init_original: str = 'em', em_init_steps: int = 5) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Trains ADAMIXTURE for a single K value, using pre-computed SVD results.
    Slices U, S, V to rank K, runs ALS initialisation, then Adam-EM or SQP.

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
        patience (int): Patience for Adam-EM learning-rate decay and BR-QN convergence.
        tol_adam (float): Adam-EM convergence tolerance.
        device_obj (torch.device): Computation device.
        threads_per_block (int): CUDA threads per block.
        original (bool): If True, run original ADMIXTURE algorithm.
        rtol (float): Convergence tolerance for original ADMIXTURE.
        Q_hist (int): History depth for original ADMIXTURE.
        init_original (str): Initialization used by --original ('em' or 'als').

    Returns:
        tuple: (P, Q) — numpy arrays on CPU, GPU tensors on CUDA/MPS.
    """
    U = U_max[:, :K] if U_max is not None else None
    S = S_max[:K] if S_max is not None else None
    V = V_max[:, :K] if V_max is not None else None

    if device_obj.type == 'cpu':
        if original and init_original == 'em':
            P, Q = initialize_em_cpu(G, seed, M, N, K, em_init_steps)
        else:
            log.info("    Running ALS...")
            P, Q = ALS(U, S, V, f, seed, M, N, K, max_als, tol_als)
        logl = tools.loglikelihood(G, P, Q)
        log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.")

        if original:
            log.info("    SQP + ZAL QN running on CPU...\n")
            P, Q = optimize_original(G, P, Q, max_iter, K, M, N, rtol, Q_hist, patience)
        else:
            log.info("    Adam-EM running on CPU...\n")
            P, Q = optimize_parameters(G, P, Q, lr, beta1, beta2, reg_adam, max_iter,
                                       check, K, M, N, lr_decay, min_lr, patience, tol_adam)
    else:
        if original and init_original == 'em':
            P, Q = initialize_em_gpu(G, seed, M, N, K, device_obj, chunk_size, threads_per_block, em_init_steps)
        else:
            log.info(f"    Running ALS on GPU ({device_obj})...")
            U_k = U.contiguous()
            S_k = S.contiguous()
            V_k = V.contiguous()
            P, Q = ALS_gpu(U_k, S_k, V_k, f, seed, M, K, max_als, tol_als, device_obj)

        if device_obj.type == 'cuda':
            torch.cuda.empty_cache()

        logl_calc = utils.get_logl_calculator(device_obj)
        logl = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
        log.info(f"    Initial log-likelihood for K={K}: {logl:.1f}.")

        if original:
            log.info(f"    SQP + ZAL QN running on GPU ({device_obj})...\n")
            P, Q = optimize_original_gpu(G, P, Q, max_iter, K, M, N, rtol, Q_hist,
                                         patience, device_obj, chunk_size, threads_per_block)
        else:
            log.info(f"    Adam-EM running on GPU ({device_obj})...\n")
            P, Q = optimize_parameters_gpu(G, P, Q, lr, beta1, beta2, reg_adam, max_iter,
                                           check, K, M, N, lr_decay, min_lr, patience, tol_adam,
                                           device_obj, chunk_size, threads_per_block)
    return P, Q
