import logging
import time

import torch

from ..src import utils

log = logging.getLogger(__name__)

def mapP(P_in: torch.Tensor) -> None:
    """
    Description:
    Clamps the P matrix values to a stable range.

    Args:
        P_in (torch.Tensor): Input frequency matrix to be updated in-place.

    Returns:
        None
    """
    P_in.clamp_(min=1e-5, max=1.0 - 1e-5)

def mapQ(Q_in: torch.Tensor) -> None:
    """
    Description:
    Clamps and normalizes the Q matrix row-wise.

    Args:
        Q_in (torch.Tensor): Input proportions matrix to be updated in-place.

    Returns:
        None
    """
    Q_in.clamp_(min=1e-5, max=1.0 - 1e-5)
    row_sums = Q_in.sum(dim=1, keepdim=True)
    Q_in.div_(row_sums)


def _cov_inverse(A: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "mps":
        return torch.linalg.inv(A)
    return torch.linalg.pinv(A)


def ALS_gpu(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, f: torch.Tensor,
            seed: int, M: int, K: int, max_iter: int, tol: float,
            device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Alternating Least Squares (ALS) implementation in PyTorch using Block Principal Pivoting.
    U, S, V, f are assumed to be on the target device.

    Args:
        U (torch.Tensor): SVD U matrix.
        S (torch.Tensor): SVD singular values.
        V (torch.Tensor): SVD V matrix.
        f (torch.Tensor): Allele frequencies.
        seed (int): Random seed.
        M (int): Number of SNPs.
        K (int): Number of populations.
        max_iter (int): Maximum ALS iterations.
        tol (float): RMSE convergence tolerance.
        device (torch.device): Computation device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Optimized P and Q matrices.
    """
    t0 = time.time()
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    dtype = utils.get_dtype(device)
    if device.type == "mps":
        from ..src.utils_c import metal

        bvls_solver = metal.batch_bvls_bpp_mps
    else:
        bvls_solver = torch.ops.bvls_kernel.batch_bvls_bpp_cuda

    Z = (U * S).to(dtype)
    V = torch.as_tensor(V, device=device, dtype=dtype)
    V_T = V.T
    Z_T = Z.T
    f_work = f.to(dtype)

    # Init P
    P = torch.rand((M, K), dtype=dtype, generator=gen, device=device)
    mapP(P)

    # Initial Q
    A_cov_P = P.T @ P
    I_p = P @ _cov_inverse(A_cov_P, device)
    f_p = (f_work.reshape(1, -1) @ I_p)
    Q = 0.5 * (V.to(dtype) @ (Z_T @ I_p)) + f_p

    mapQ(Q)
    Q0 = Q.clone()

    for i in range(max_iter):
        # UPDATE P
        A_cov_Q = Q.T @ Q
        I_q = Q @ _cov_inverse(A_cov_Q, device)
        P_free = 0.5 * (Z @ (V_T @ I_q)) + torch.outer(f_work, I_q.sum(dim=0))
        B_target_P = P_free @ A_cov_Q

        P = bvls_solver(A_cov_Q, B_target_P, 0.0, 1.0)
        mapP(P)

        # UPDATE Q
        A_cov_P = P.T @ P
        I_p = P @ _cov_inverse(A_cov_P, device)
        f_p = (f_work.reshape(1, -1) @ I_p)
        Q_free = 0.5 * (V.to(dtype) @ (Z_T @ I_p)) + f_p
        B_target_Q = Q_free @ A_cov_P

        Q = bvls_solver(A_cov_P, B_target_Q, 0.0, 1.0)
        mapQ(Q)

        rmse_error = torch.sqrt(torch.mean((Q - Q0)**2))
        if rmse_error < tol:
            log.info(f"        Convergence reached in iteration {i+1}.")
            break
        else:
            Q0.copy_(Q)

    total_time = time.time() - t0
    log.info(f"        Total ALS time={total_time:.3f}s\n")
    return P, Q
