import torch
import time
import logging

log = logging.getLogger(__name__)

def nnls_bpp_gpu(A: torch.Tensor, B: torch.Tensor, max_iter: int = 50) -> torch.Tensor:
    """
    Description:
    Solves Non-Negative Least Squares (min 1/2 x^T A x - b^T x s.t. x >= 0) using Block Principal Pivoting.
    Vectorized across the first dimension of B.

    Args:
        A (torch.Tensor): Input matrix A (K x K).
        B (torch.Tensor): Input matrix B (M x K).
        max_iter (int): Maximum number of BPP iterations.

    Returns:
        torch.Tensor: Matrix X (M x K) containing the non-negative solution.
    """
    M, K = B.shape
    device = A.device
    dtype = A.dtype
    
    F = torch.ones((M, K), dtype=torch.bool, device=device)
    X = torch.zeros((M, K), dtype=dtype, device=device)
    indices = torch.arange(K, device=device)
    
    for _ in range(max_iter):
        A_full = A.unsqueeze(0).expand(M, K, K).clone()
        mask = ~F
        
        A_full.masked_fill_(mask.unsqueeze(2), 0.0)
        A_full.masked_fill_(mask.unsqueeze(1), 0.0)
        A_full[:, indices, indices] = A_full[:, indices, indices].masked_fill(mask, 1.0)
        
        B_curr = B.clone()
        B_curr.masked_fill_(mask, 0.0)
        
        X = torch.linalg.solve(A_full, B_curr)
        y = X @ A - B
        
        to_active = F & (X < -1e-8)
        to_passive = (~F) & (y < -1e-8)
        
        if not to_active.any() and not to_passive.any():
            break
            
        F = F ^ (to_active | to_passive)
    return X

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

def batch_nnls_bpp(A: torch.Tensor, B: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Description:
    Processes NNLS in batches to avoid GPU memory overflow.

    Args:
        A (torch.Tensor): Input matrix A (K x K).
        B (torch.Tensor): Target matrix B (M x K).
        chunk_size (int): Batch size.

    Returns:
        torch.Tensor: Non-negative solution matrix X.
    """
    m = B.shape[0]
    x = torch.zeros_like(B)
    for i in range(0, m, chunk_size):
        end = min(i + chunk_size, m)
        x[i:end] = nnls_bpp_gpu(A, B[i:end])
    return x

def ALS_gpu(G: torch.Tensor, U: torch.Tensor, S: torch.Tensor, V: torch.Tensor, f: torch.Tensor, 
            seed: int, M: int, N: int, K: int, max_iter: int, tol: float, device: torch.device, 
            threads_per_block: int, chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Alternating Least Squares (ALS) implementation in PyTorch using Block Principal Pivoting.
    G, U, S, V, f are assumed to be on the target device.

    Args:
        G (torch.Tensor): Packed genotype matrix.
        U (torch.Tensor): SVD U matrix.
        S (torch.Tensor): SVD singular values.
        V (torch.Tensor): SVD V matrix.
        f (torch.Tensor): Allele frequencies.
        seed (int): Random seed.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of populations.
        max_iter (int): Maximum ALS iterations.
        tol (float): RMSE convergence tolerance.
        device (torch.device): Computation device.
        threads_per_block (int): GPU thread scaling.
        chunk_size (int): Batch size for NNLS.

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
        
        P = batch_nnls_bpp(A_cov_Q, B_target_P, chunk_size)
        mapP(P)
        
        # UPDATE Q
        A_cov_P = P.T @ P
        I_p = P @ torch.linalg.pinv(A_cov_P)
        f_p = (f_64.reshape(1, -1) @ I_p)
        Q_free = 0.5 * (V.to(torch.float64) @ (Z_T @ I_p)) + f_p
        B_target_Q = Q_free @ A_cov_P
        
        Q = batch_nnls_bpp(A_cov_P, B_target_Q, chunk_size)
        mapQ(Q)
        
        rmse_error = torch.sqrt(torch.mean((Q - Q0)**2))
        if rmse_error < tol:
            log.info(f"        Convergence reached in iteration {i+1}.")
            break
        else:
            Q0.copy_(Q)
            
    total_time = time.time() - t0
    log.info(f"        Total ALS time={total_time:.3f}s")
    return P, Q
