import logging
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from . import utils
from .utils_c import sqp

if TYPE_CHECKING:
    import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── CPU (NumPy / Cython) Exact Newton Projection ──────────────────────────────

def optimize_projection_cpu(
    G: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    max_iter: int,
    K: int,
    M: int,
    N: int,
    tol: float,
) -> np.ndarray:
    """
    Description:
    Projects target samples onto pre-trained allele frequencies P using exact
    analytical Newton-Raphson steps on the probability simplex (SQP) on CPU.
    Performs 1 single analytical Hessian evaluation and simplex QP solve per iteration.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P (np.ndarray): Fixed allele-frequency matrix (M x K).
        Q (np.ndarray): Initial ancestry-proportion matrix (N x K). Updated in-place.
        max_iter (int): Maximum number of Newton iterations.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        tol (float): Convergence tolerance for mean change in Q (delta_q).

    Returns:
        np.ndarray: Optimised Q matrix.
    """
    from ..model.br_qn import q_hessian_block_size

    # 1. Precompute Vt matrix from SVD of ones(1, K)
    _, _, vt = np.linalg.svd(np.ones((1, K)), full_matrices=True)
    v_kk = np.ascontiguousarray(vt.T, dtype=np.float64)
    q_block = q_hessian_block_size(K)

    # 2. Allocate buffers
    XtX_q = np.empty((N, K, K), dtype=np.float64)
    Xtz_q = np.empty((N, K), dtype=np.float64)
    Q_next = np.empty_like(Q, dtype=np.float64)

    for it in range(1, max_iter + 1):
        it_start = time.time()
        sqp.update_q_sqp(G, Q, Q_next, P, XtX_q, Xtz_q, v_kk, M, N, K, q_block)
        delta_q = float(np.mean(np.abs(Q_next - Q)))
        np.copyto(Q, Q_next)

        log.info(
            f"    Iteration {it}, "
            f"Mean dQ: {delta_q:.2e}, "
            f"Time: {time.time() - it_start:.3f}s"
        )

        if delta_q < tol:
            log.info(f"    Converged at iteration {it} with Mean dQ < {tol:.2e}.")
            break

    return Q


# ── GPU / MPS Exact Newton Projection ─────────────────────────────────────────

def optimize_projection_gpu(
    G: "torch.Tensor",
    P: "torch.Tensor",
    Q: "torch.Tensor",
    max_iter: int,
    K: int,
    M: int,
    N: int,
    tol: float,
    device: "torch.device",
    chunk_size: int,
    threads_per_block: int,
) -> "torch.Tensor":
    """
    Description:
    Projects target samples onto pre-trained allele frequencies P using exact
    analytical Newton-Raphson steps on the probability simplex (SQP) on GPU / MPS.
    Performs 1 single analytical Hessian evaluation and simplex QP solve per iteration.

    Args:
        G (torch.Tensor): Genotype tensor (packed 2-bit format).
        P (torch.Tensor): Fixed allele-frequency tensor (M x K).
        Q (torch.Tensor): Initial ancestry-proportion tensor (N x K). Updated in-place.
        max_iter (int): Maximum number of Newton iterations.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        tol (float): Convergence tolerance for mean change in Q (delta_q).
        device (torch.device): CUDA or MPS device.
        chunk_size (int): SNP batch size for batched operations.
        threads_per_block (int): Threads per block for kernels.

    Returns:
        torch.Tensor: Optimised Q tensor on the target device.
    """
    import torch
    from ..model.br_qn_gpu import compute_grad_hess_Q_gpu

    dtype = utils.get_dtype(device)

    # 1. Precompute Vt matrix from SVD of ones(1, K) on CPU
    ones_K = torch.ones((1, K), dtype=torch.float64)
    _, _, vt = torch.linalg.svd(ones_K, full_matrices=True)
    v_kk = vt.t().to(device=device, dtype=dtype).contiguous()

    # 2. Buffers
    XtX_q = torch.zeros((N, K, K), dtype=dtype, device=device)
    Xtz_q = torch.zeros((N, K), dtype=dtype, device=device)

    unpacker = utils.get_unpacker(device, threads_per_block)
    if device.type == "mps":
        from .utils_c import metal
        sqp_solve_q = metal.sqp_solve_q_mps
    else:
        sqp_solve_q = torch.ops.sqp_kernel.sqp_solve_q_cuda

    for it in range(1, max_iter + 1):
        it_start = time.time()
        compute_grad_hess_Q_gpu(G, Q, P, XtX_q, Xtz_q, M, chunk_size, unpacker, dtype)
        Xtz_q.neg_()
        Q_next = sqp_solve_q(XtX_q, Xtz_q, Q, v_kk, N, K)

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        delta_q = float(torch.mean(torch.abs(Q_next - Q)).item())
        Q.copy_(Q_next)

        log.info(
            f"    Iteration {it}, "
            f"Mean dQ: {delta_q:.2e}, "
            f"Time: {time.time() - it_start:.3f}s"
        )

        if delta_q < tol:
            log.info(f"    Converged at iteration {it} with Mean dQ < {tol:.2e}.")
            break

    return Q


# Aliases for backward compatibility
optimize_projection = optimize_projection_cpu
optimize_projection_newton = optimize_projection_cpu
optimize_projection_newton_gpu = optimize_projection_gpu
