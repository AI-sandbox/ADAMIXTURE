import logging
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from . import utils
from .utils_c import batch_bvls_bpp, tools

if TYPE_CHECKING:
    import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── CPU (NumPy / Cython) Chunked ALS Projection ───────────────────────────────

def optimize_projection_cpu(
    G: np.ndarray,
    P: np.ndarray,
    chunk_size: int,
    Q: np.ndarray | None = None,
    K: int | None = None,
    M: int | None = None,
    N: int | None = None,
) -> np.ndarray:
    """
    Description:
    Projects target samples onto pre-trained allele frequencies P using
    chunked Alternating Least Squares (ALS) with exact bounded variable
    least squares (BVLS) via Block Principal Pivoting on CPU.

    Evaluates the target covariance in a single memory-efficient streaming pass
    over genotype chunks (uint8), followed by a closed-form BVLS solve.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P (np.ndarray): Fixed allele-frequency matrix (M x K).
        chunk_size (int): SNP batch size for streaming accumulation (required).
        Q (np.ndarray | None): Optional initial ancestry matrix (N x K). Updated in-place if provided.
        K (int | None): Number of ancestral populations (inferred if None).
        M (int | None): Number of SNPs (inferred if None).
        N (int | None): Number of individuals (inferred if None).

    Returns:
        np.ndarray: Optimised ancestry-proportion matrix Q (N x K).
    """
    t0 = time.time()

    M_snps, N_samples = G.shape
    if M is None:
        M = M_snps
    if N is None:
        N = N_samples
    if K is None:
        K = P.shape[1]

    # 1. Covariance matrix A = P^T P (K x K)
    A_cov_P = np.ascontiguousarray(P.T @ P, dtype=np.float64)

    # 2. Accumulate target B = 0.5 * G^T P in streaming chunks of SNPs
    B = np.zeros((N, K), dtype=np.float64)
    actual_chunk = min(chunk_size, M)
    chunk_buf = np.empty((actual_chunk, N), dtype=np.float32)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        n_chunk = end - start
        g_slice = G[start:end]
        buf = chunk_buf[:n_chunk]
        np.copyto(buf, g_slice)

        # Impute missing genotypes (value 3) to 0.0
        missing = (g_slice == 3)
        if np.any(missing):
            buf[missing] = 0.0

        B += 0.5 * (buf.T @ P[start:end])

    B = np.ascontiguousarray(B, dtype=np.float64)

    # 3. Exact BVLS solve on simplex
    Q_out = np.empty((N, K), dtype=np.float64)
    batch_bvls_bpp(A_cov_P, B, Q_out, 0.0, 1.0)
    tools.mapQ_d(Q_out, N, K)

    if Q is not None:
        np.copyto(Q, Q_out)
        return Q

    log.info(f"    Projection finished in {time.time() - t0:.3f}s.")
    return Q_out


# ── GPU / MPS Chunked ALS Projection ──────────────────────────────────────────

def optimize_projection_gpu(
    G: "torch.Tensor",
    P: "torch.Tensor",
    chunk_size: int,
    Q: "torch.Tensor | None" = None,
    K: int | None = None,
    M: int | None = None,
    N: int | None = None,
    device: "torch.device | None" = None,
    threads_per_block: int | None = None,
) -> "torch.Tensor":
    """
    Description:
    Projects target samples onto pre-trained allele frequencies P using
    chunked Alternating Least Squares (ALS) with exact bounded variable
    least squares (BVLS) on GPU / MPS.

    Accumulates the target covariance via cuBLAS/MPS Tensor Cores in a single
    streaming pass over packed 2-bit genotype chunks, followed by a parallel
    BVLS solve.

    Args:
        G (torch.Tensor): Genotype tensor (packed 2-bit format on CPU or device).
        P (torch.Tensor): Fixed allele-frequency tensor (M x K).
        chunk_size (int): SNP batch size for streaming accumulation (required).
        Q (torch.Tensor | None): Optional initial ancestry tensor (N x K). Updated in-place if provided.
        K (int | None): Number of ancestral populations (inferred if None).
        M (int | None): Number of SNPs (inferred if None).
        N (int | None): Number of individuals (inferred if None).
        device (torch.device | None): Computation device (CUDA or MPS).
        threads_per_block (int | None): Threads per block for kernels.

    Returns:
        torch.Tensor: Optimised ancestry-proportion tensor Q on the target device.
    """
    import torch

    t0 = time.time()

    if device is None:
        device = P.device
    if threads_per_block is None:
        threads_per_block = utils.get_tuning_params(device)

    dtype = utils.get_dtype(device)
    if M is None:
        M = P.shape[0]
    if K is None:
        K = P.shape[1]
    if N is None:
        N = Q.shape[0] if Q is not None else G.shape[1]

    P_work = P.to(dtype=torch.float32, device=device)

    # 1. Covariance matrix A = P^T P (K x K)
    A_cov_P = torch.matmul(P_work.t(), P_work).to(dtype=dtype).contiguous()

    # 2. Accumulate target B = 0.5 * G^T P in streaming chunks of SNPs
    B = torch.zeros((N, K), dtype=torch.float32, device=device)
    unpacker = utils.get_unpacker(device, threads_per_block)

    n_batches = int(np.ceil(M / chunk_size))
    for w in range(n_batches):
        start = w * chunk_size
        end = min(start + chunk_size, M)
        n_active = end - start

        g_unpacked = unpacker(G, start, n_active, M)
        buf = g_unpacked.to(dtype=torch.float32)

        # Impute missing genotypes (value 3) to 0.0
        missing = (g_unpacked == 3)
        if missing.any():
            buf[missing] = 0.0

        B += 0.5 * torch.matmul(buf.t(), P_work[start:end])

    # 3. Exact BVLS solve on target device
    B_solve = B.to(dtype=dtype).contiguous()
    if device.type == "mps":
        from .utils_c import metal
        Q_out = metal.batch_bvls_bpp_mps(A_cov_P, B_solve, 0.0, 1.0)
    else:
        Q_out = torch.ops.bvls_kernel.batch_bvls_bpp_cuda(A_cov_P, B_solve, 0.0, 1.0)

    # Project onto simplex (clamp and row normalize)
    Q_out.clamp_(min=1e-5, max=1.0 - 1e-5)
    row_sums = Q_out.sum(dim=1, keepdim=True)
    Q_out.div_(row_sums)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    if Q is not None:
        Q.copy_(Q_out)
        return Q

    log.info(f"    Projection finished in {time.time() - t0:.3f}s.")
    return Q_out


# Aliases for backward compatibility
optimize_projection = optimize_projection_cpu
optimize_projection_als = optimize_projection_cpu
optimize_projection_als_gpu = optimize_projection_gpu
