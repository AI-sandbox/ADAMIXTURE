import logging
import sys
from argparse import Namespace

import numpy as np
import torch

from ..model.em_adam import adamStep
from . import utils
from .utils_c import tools

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def _build_folds(non_missing_flat: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """
    Description:
    Partitions non-missing genotype flat indices into v roughly equal folds
    using a random shuffle.

    Args:
        non_missing_flat (np.ndarray): 1-D array of flat indices where genotype != 3.
        n_folds (int): Number of cross-validation folds.
        seed (int): Random seed for reproducibility.

    Returns:
        list[np.ndarray]: List of v arrays, each containing the flat indices for one fold.
    """
    rng = np.random.default_rng(seed)
    shuffled = non_missing_flat.copy()
    rng.shuffle(shuffled)

    fold_ids = np.arange(shuffled.size, dtype=np.int32) % n_folds
    folds = [shuffled[fold_ids == fold] for fold in range(n_folds)]
    return folds

def _polish_fold(G: np.ndarray, P_init: np.ndarray, Q_init: np.ndarray,
                args: Namespace, M: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Runs a fixed number of Adam-EM polishing iterations on CPU,
    warm-started from the global P and Q estimates.

    Args:
        G (np.ndarray): Genotype matrix with held-out entries masked as 3.
        P_init (np.ndarray): Global P matrix (M x K) used as warm-start.
        Q_init (np.ndarray): Global Q matrix (N x K) used as warm-start.
        args (Namespace): Parsed command-line arguments (lr, beta1, beta2, reg_adam).
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.

    Returns:
        tuple[np.ndarray, np.ndarray]: Polished (P, Q) matrices after 2 iterations.
    """
    P = np.array(P_init, dtype=np.float64, copy=True)
    Q = np.array(Q_init, dtype=np.float64, copy=True)

    m_P = np.zeros_like(P, dtype=np.float64)
    v_P = np.zeros_like(P, dtype=np.float64)
    m_Q = np.zeros_like(Q, dtype=np.float64)
    v_Q = np.zeros_like(Q, dtype=np.float64)
    t = [0]

    P1 = np.zeros_like(P, dtype=np.float64)
    Q1 = np.zeros_like(Q, dtype=np.float64)
    T = np.zeros_like(Q, dtype=np.float64)
    q_bat = np.zeros(N, dtype=np.float64)

    for _ in range(2):
        adamStep(
            G, P, Q, T, P1, Q1, q_bat, K, M, N,
            m_P, v_P, m_Q, v_Q, t,
            float(args.lr), float(args.beta1),
            float(args.beta2), float(args.reg_adam),
        )

    return P, Q

def _polish_fold_gpu(G: torch.Tensor, P_init: torch.Tensor, Q_init: torch.Tensor,
                    args: Namespace, M: int, N: int, K: int,
                    device: torch.device, threads_per_block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Runs a fixed number of Adam-EM polishing iterations on GPU,
    warm-started from the global P and Q estimates. Operates on
    packed genotype data using chunked unpacking.

    Args:
        G (torch.Tensor): Packed genotype tensor on GPU (M_bytes x N, uint8).
        P_init (torch.Tensor): Global P tensor (M x K) on GPU, used as warm-start.
        Q_init (torch.Tensor): Global Q tensor (N x K) on GPU, used as warm-start.
        args (Namespace): Parsed command-line arguments.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        device (torch.device): CUDA device.
        threads_per_block (int): CUDA threads per block for unpacking.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Polished (P, Q) tensors on GPU.
    """
    from ..model.em_adam_gpu import EMAdamOptimizer

    P = P_init.clone()
    Q = Q_init.clone()

    optimizer = EMAdamOptimizer(
        P.shape, Q.shape,
        float(args.lr), float(args.beta1),
        float(args.beta2), float(args.reg_adam), device,
    )
    unpacker = utils.get_unpacker(device, threads_per_block)
    chunk_size = int(args.chunk_size)

    for _ in range(2):
        P_target, Q_target = optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)
        optimizer.step(P, Q, P_target, Q_target)

    del optimizer
    return P, Q

def _deviance_squared_sum_gpu(saved_values: torch.Tensor, held_out_entries: torch.Tensor,
                            P: torch.Tensor, Q: torch.Tensor, N: int,
                            chunk_size: int = 1_000_000) -> float:
    """
    Description:
    Computes the sum of squared binomial deviance residuals entirely on GPU.
    Processes entries in chunks to limit peak memory.

    Args:
        saved_values (torch.Tensor): 1-D uint8 tensor of original genotype values on GPU.
        held_out_entries (torch.Tensor): 1-D int64 tensor of flat indices on GPU.
        P (torch.Tensor): Polished P tensor (M x K) on GPU.
        Q (torch.Tensor): Polished Q tensor (N x K) on GPU.
        N (int): Number of individuals.
        chunk_size (int): Entries per chunk to bound memory usage.

    Returns:
        float: Sum of squared deviance residuals.
    """
    eps = 1e-10
    total = torch.tensor(0.0, dtype=torch.float64, device=P.device)
    P64 = P.to(torch.float64)
    Q64 = Q.to(torch.float64)
    n_entries = held_out_entries.size(0)

    for start in range(0, n_entries, chunk_size):
        end = min(start + chunk_size, n_entries)
        idx = held_out_entries[start:end]
        rows = idx // N
        cols = idx % N
        g = saved_values[start:end].to(torch.float64)

        mu = 2.0 * (P64[rows] * Q64[cols]).sum(dim=1)
        torch.clamp_(mu, eps, 2.0 - eps)

        term_a = torch.where(g > 0.0, g * torch.log(g / mu), torch.zeros_like(g))
        rem = 2.0 - g
        term_b = torch.where(rem > 0.0, rem * torch.log(rem / (2.0 - mu)), torch.zeros_like(g))

        dev = term_a + term_b
        total += (dev * dev).sum()

    return total.item()

def _find_non_missing_packed(G: torch.Tensor, M: int, N: int, device: torch.device,
                            threads_per_block: int, chunk_size: int) -> np.ndarray:
    """
    Description:
    Scans a packed genotype tensor in chunks to find all non-missing
    (value != 3) flat indices in the M x N unpacked space.

    Args:
        G (torch.Tensor): Packed genotype tensor (M_bytes x N, uint8).
        M (int): Number of SNPs.
        N (int): Number of individuals.
        device (torch.device): CUDA device.
        threads_per_block (int): CUDA threads per block for unpacking.
        chunk_size (int): Number of SNPs per unpacking chunk.

    Returns:
        np.ndarray: Sorted 1-D int64 array of flat indices where genotype != 3.
    """
    unpacker = utils.get_unpacker(device, threads_per_block)
    scan_chunk = min(chunk_size, 512)
    parts: list[np.ndarray] = []

    for m_start in range(0, M, scan_chunk):
        actual = min(scan_chunk, M - m_start)
        chunk_gpu = unpacker(G, m_start, actual, M)
        chunk_cpu = chunk_gpu.cpu().numpy()
        del chunk_gpu
        local_flat = np.flatnonzero(chunk_cpu.ravel() != 3)
        if local_flat.size > 0:
            parts.append((local_flat + m_start * N).astype(np.int64))

    return np.concatenate(parts) if parts else np.array([], dtype=np.int64)

def run_cross_validation(args: Namespace, G: np.ndarray, N: int, M: int, K: int,
                        P_global: np.ndarray | torch.Tensor, Q_global: np.ndarray | torch.Tensor) -> float:
    """
    Description:
    Performs v-fold cross-validation on genotype entries. Each fold masks a random
    subset of non-missing entries, polishes P and Q from the global warm-start,
    and scores via squared binomial deviance residuals. Always runs on CPU.

    ``P_global`` / ``Q_global`` may be ``torch.Tensor`` (e.g. MPS/CUDA fits); they are
    converted to NumPy on CPU inside this function.

    Args:
        args (Namespace): Parsed command-line arguments (cv, seed, lr, etc.).
        G (np.ndarray): Unpacked genotype matrix (M x N, uint8).
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.
        P_global (ndarray | Tensor): Global P matrix (M x K) from the main fit.
        Q_global (ndarray | Tensor): Global Q matrix (N x K) from the main fit.

    Returns:
        float: CV index (average squared deviance residual across all held-out entries).
    """
    if isinstance(P_global, torch.Tensor):
        P_global = P_global.detach().cpu().numpy()
    if isinstance(Q_global, torch.Tensor):
        Q_global = Q_global.detach().cpu().numpy()
    if G.shape != (M, N):
        raise ValueError(f"CV requires an unpacked genotype matrix with shape {(M, N)}, got {G.shape}.")

    non_missing_flat = np.flatnonzero(G != 3)
    n_non_missing = int(non_missing_flat.size)
    if n_non_missing == 0:
        raise ValueError("No non-missing genotypes available for cross-validation.")

    n_folds = int(args.cv)
    folds = _build_folds(non_missing_flat, n_folds, int(args.seed))

    cv_sum = 0.0
    for held_out_entries in folds:
        if held_out_entries.size == 0:
            continue

        rows = held_out_entries // N
        cols = held_out_entries % N

        saved_values = G[rows, cols].copy()
        G[rows, cols] = 3

        P_cv, Q_cv = _polish_fold(G, P_global, Q_global, args, M, N, K)

        G[rows, cols] = saved_values

        fold_sum = tools.deviance_squared_sum_direct(
            np.ascontiguousarray(saved_values, dtype=np.uint8),
            np.ascontiguousarray(held_out_entries, dtype=np.int64),
            np.ascontiguousarray(P_cv, dtype=np.float64),
            np.ascontiguousarray(Q_cv, dtype=np.float64),
            N,
        )
        cv_sum += fold_sum

    cv_index = cv_sum / float(n_non_missing)
    log.info(f"    CV index for K={K}: {cv_index:.4f}")
    return cv_index

def run_cross_validation_gpu(args: Namespace, G: torch.Tensor, N: int, M: int, K: int,
                            P_global: torch.Tensor, Q_global: torch.Tensor,
                            device: torch.device, threads_per_block: int) -> float:
    """
    Description:
    Performs v-fold cross-validation directly on the packed GPU genotype tensor.
    Held-out entries are masked/restored in-place via CUDA kernels operating on
    2-bit packed bytes, avoiding any full unpacked copy of G. All computation
    (masking, polishing, scoring) stays on GPU.

    Args:
        args (Namespace): Parsed command-line arguments.
        G (torch.Tensor): Packed genotype tensor on GPU (M_bytes x N, uint8).
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.
        P_global (torch.Tensor): Global P tensor (M x K) on GPU from the main fit.
        Q_global (torch.Tensor): Global Q tensor (N x K) on GPU from the main fit.
        device (torch.device): CUDA device.
        threads_per_block (int): CUDA threads per block for unpacking.

    Returns:
        float: CV index (average squared deviance residual across all held-out entries).
    """
    chunk_size = int(args.chunk_size)

    non_missing_flat = _find_non_missing_packed(G, M, N, device, threads_per_block, chunk_size)
    n_non_missing = int(non_missing_flat.size)
    if n_non_missing == 0:
        raise ValueError("No non-missing genotypes available for cross-validation.")

    n_folds = int(args.cv)
    folds = _build_folds(non_missing_flat, n_folds, int(args.seed))
    del non_missing_flat

    cv_sum = 0.0
    for held_out_entries in folds:
        if held_out_entries.size == 0:
            continue

        held_out_gpu = torch.from_numpy(
            np.ascontiguousarray(held_out_entries, dtype=np.int64)
        ).to(device)

        saved_values_gpu = torch.ops.cv_mask_kernel.mask_entries_packed_cuda(
            G, held_out_gpu, N,
        )

        try:
            P_cv, Q_cv = _polish_fold_gpu(
                G, P_global, Q_global, args, M, N, K, device, threads_per_block,
            )
        finally:
            torch.ops.cv_mask_kernel.restore_entries_packed_cuda(
                G, held_out_gpu, saved_values_gpu, N,
            )

        fold_sum = _deviance_squared_sum_gpu(
            saved_values_gpu, held_out_gpu, P_cv, Q_cv, N,
        )
        del saved_values_gpu, held_out_gpu, P_cv, Q_cv
        cv_sum += fold_sum

    cv_index = cv_sum / float(n_non_missing)
    log.info(f"    CV index for K={K}: {cv_index:.4f}")
    return cv_index
