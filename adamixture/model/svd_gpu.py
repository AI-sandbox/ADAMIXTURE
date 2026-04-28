import logging
import math
import time

import torch

from ..src import utils

log = logging.getLogger(__name__)

def eigSVD_gpu(C: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Description:
    GPU-accelerated Singular Value Decomposition (SVD) via eigendecomposition
    of the covariance matrix (C @ C^T).

    Args:
        C (torch.Tensor): Input matrix of shape (L, N) or (L, M).

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (U, S, V) matrices from SVD.
    """
    device = C.device
    cov = C @ C.transpose(-2, -1)

    # Fallback to CPU for eigh if device is MPS (operation not yet implemented)
    if device.type == 'mps':
        D, V = torch.linalg.eigh(cov.cpu())
        D, V = D.to(device), V.to(device)
    else:
        D, V = torch.linalg.eigh(cov)

    S = torch.sqrt(D)
    U = (C.transpose(-2, -1) @ V) / S

    idx = torch.arange(V.shape[1]-1, -1, -1, device=device)
    U, S, V = U[:, idx], S.flip(0), V[:, idx]
    return U, S, V

def SVD_gpu(G: torch.Tensor, N: int, M: int, f: torch.Tensor, k: int, seed: int,
            power: int, tol: float, chunk_size: int, device: torch.device,
            threads_per_block: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Description:
    Randomized SVD with Dynamic Shifts optimized for GPU. Processes 2-bit
    packed genotypes in chunks to manage VRAM usage.

    Args:
        G (torch.Tensor): Packed 2-bit genotype matrix (uint8).
        N (int): Number of individuals.
        M (int): Number of SNPs.
        f (torch.Tensor): 1D tensor of allele frequencies.
        k (int): Target rank (number of components).
        seed (int): Random seed for the projection basis.
        power (int): Number of power iterations.
        tol (float): Relative error tolerance for convergence.
        chunk_size (int): Number of individuals per batch.
        device (torch.device): GPU computation device.
        threads_per_block (int, optional): CUDA thread block size. Defaults to 256.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (U matrix, S vector, V matrix).
    """
    t0 = time.time()
    k_prime = max(k + 10, 20)
    L = k_prime

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    proj_basis = torch.randn((M, L), dtype=torch.float32, generator=gen, device=device)
    accum_mat = torch.zeros((N, L), dtype=torch.float32, device=device)

    n_batches = math.ceil(M / chunk_size)
    unpacker = utils.get_centering_unpacker(device, threads_per_block)

    # 1) Prime iteration
    log.info("    1) Prime iteration...")
    ts = time.time()
    for w in range(n_batches):
        start_idx = w * chunk_size
        end_idx = min(start_idx + chunk_size, M)
        actual_chunk_size = end_idx - start_idx
        X = unpacker(G, f, start_idx, actual_chunk_size, M)

        accum_mat.addmm_(X.T, proj_basis[start_idx:end_idx])

    orth_matrix, _, _ = eigSVD_gpu(accum_mat.T)
    accum_mat.zero_()
    log.info(f"        time={time.time() - ts:.4f}s")

    # 2) Power iterations
    log.info("    2) Power iterations...")
    ts_power = time.time()
    alpha = torch.tensor(0.0, device=device)
    singular_vals = torch.zeros(L, device=device)

    for i in range(power):
        for w in range(n_batches):
            start_idx = w * chunk_size
            end_idx = min(start_idx + chunk_size, M)
            actual_chunk_size = end_idx - start_idx

            X = unpacker(G, f, start_idx, actual_chunk_size, M)

            proj_basis[start_idx:end_idx] = X @ orth_matrix
            accum_mat.addmm_(X.T, proj_basis[start_idx:end_idx])

        Q, S, _ = eigSVD_gpu(accum_mat.sub(orth_matrix, alpha=alpha.item()).T)
        accum_mat.zero_()

        if i > 0:
            s_current = S + alpha
            rel_diff = torch.abs(s_current - singular_vals) / torch.maximum(s_current, torch.tensor(1e-12, device=device))
            max_err = torch.max(rel_diff[:k])
            if max_err < tol:
                log.info(f"        Converged at iteration {i}.")
                break
            singular_vals.copy_(s_current)
        else:
            singular_vals.copy_(S + alpha)

        if S[-1] > alpha:
            alpha = 0.5 * (S[-1] + alpha)

        orth_matrix.copy_(Q)

    log.info(f"        time={time.time() - ts_power:.4f}s")

    # 3) Build projection
    log.info("    3) Build small matrix...")
    ts_build = time.time()
    for w in range(n_batches):
        start_idx = w * chunk_size
        end_idx = min(start_idx + chunk_size, M)
        actual_chunk_size = end_idx - start_idx
        X = unpacker(G, f, start_idx, actual_chunk_size, M)
        proj_basis[start_idx:end_idx] = X @ orth_matrix
    log.info(f"        time={time.time() - ts_build:.4f}s")

    # 4) SVD of condensed matrix
    log.info("    4) SVD of small matrix...")
    ts_svd = time.time()
    u_cond, s_all, v_cond = eigSVD_gpu(proj_basis.T)
    log.info(f"        time={time.time() - ts_svd:.4f}s")

    S_final = s_all[:k].contiguous()
    U_final = u_cond[:, :k].contiguous()
    V_final = (orth_matrix @ v_cond[:, :k]).contiguous()

    total_time = time.time() - t0
    log.info(f"\n    Total time for SVD={total_time:.3f}s")

    return U_final, S_final, V_final
