import logging
import time

import torch

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

def ALS_gpu(U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, f: torch.Tensor,
            seed: int, M: int, K: int, max_iter: int, tol: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
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

    Z = (U * S).to(torch.float64)
    V = torch.as_tensor(V, device=device, dtype=torch.float64)
    V_T = V.T
    Z_T = Z.T
    f_64 = f.to(torch.float64)

    # Init P
    P = torch.rand((M, K), dtype=torch.float64, generator=gen, device=device)
    mapP(P)

    # Initial Q
    A_cov_P = P.T @ P
    I_p = P @ torch.linalg.pinv(A_cov_P)
    f_p = (f_64.reshape(1, -1) @ I_p)
    Q = 0.5 * (V.to(torch.float64) @ (Z_T @ I_p)) + f_p

    mapQ(Q)
    Q0 = Q.clone()

    for i in range(max_iter):
        # UPDATE P
        A_cov_Q = Q.T @ Q
        I_q = Q @ torch.linalg.pinv(A_cov_Q)
        P_free = 0.5 * (Z @ (V_T @ I_q)) + torch.outer(f_64, I_q.sum(dim=0))
        B_target_P = P_free @ A_cov_Q

        P = torch.ops.bvls_kernel.batch_bvls_bpp_cuda(A_cov_Q, B_target_P, 0.0, 1.0)
        mapP(P)

        # UPDATE Q
        A_cov_P = P.T @ P
        I_p = P @ torch.linalg.pinv(A_cov_P)
        f_p = (f_64.reshape(1, -1) @ I_p)
        Q_free = 0.5 * (V.to(torch.float64) @ (Z_T @ I_p)) + f_p
        B_target_Q = Q_free @ A_cov_P

        Q = torch.ops.bvls_kernel.batch_bvls_bpp_cuda(A_cov_P, B_target_Q, 0.0, 1.0)
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
