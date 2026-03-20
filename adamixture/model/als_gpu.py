import torch
import time
import logging

log = logging.getLogger(__name__)

def bvls_bpp_gpu(A: torch.Tensor, B: torch.Tensor, lower: float = 0.0, upper: float = 1.0, max_iter: int = 50) -> torch.Tensor:
    """
    Description:
    Solves Bounded Variable Least Squares (min 1/2 x^T A x - b^T x s.t. lower <= x <= upper) 
    using a 3-state Block Principal Pivoting algorithm. Vectorized across the first dimension of B.

    Args:
        A (torch.Tensor): Input matrix A (K x K), usually a covariance matrix.
        B (torch.Tensor): Input matrix B (M x K).
        lower (float): Lower bound for variables.
        upper (float): Upper bound for variables.
        max_iter (int): Maximum number of BPP iterations.

    Returns:
        torch.Tensor: Matrix X (M x K) containing the bounded solution.
    """
    M, K = B.shape
    device = A.device
    dtype = A.dtype
    
    # States: F (Free), U (At Upper Bound), L (At Lower Bound implied)
    F = torch.ones((M, K), dtype=torch.bool, device=device)
    U = torch.zeros((M, K), dtype=torch.bool, device=device)
    indices = torch.arange(K, device=device)
    
    for _ in range(max_iter):
        A_full = A.unsqueeze(0).expand(M, K, K).clone()
        not_F = ~F
        
        # Mask A to solve only for F variables
        A_full.masked_fill_(not_F.unsqueeze(2), 0.0)
        A_full.masked_fill_(not_F.unsqueeze(1), 0.0)
        A_full[:, indices, indices] = A_full[:, indices, indices].masked_fill(not_F, 1.0)
        
        # Contribution from fixed variables (Upper and Lower bounds)
        X_fixed = torch.zeros((M, K), dtype=dtype, device=device)
        X_fixed.masked_fill_(U, upper)
        X_fixed.masked_fill_(~(F | U), lower)
        
        B_curr = B - (X_fixed @ A)
        B_curr.masked_fill_(not_F, 0.0)
        
        # Solve for Free variables
        X = torch.linalg.solve(A_full, B_curr)
        
        # Combine with fixed values
        X.masked_fill_(U, upper)
        X.masked_fill_(~(F | U), lower)
        
        # Gradient y = X @ A - B
        y = X @ A - B
        
        # Pivoting rules
        to_L = F & (X < lower - 1e-8)
        to_U = F & (X > upper + 1e-8)
        from_L = (~(F | U)) & (y < -1e-8)
        from_U = U & (y > 1e-8)
        
        if not (to_L.any() or to_U.any() or from_L.any() or from_U.any()):
            break
            
        F = (F & ~to_L & ~to_U) | from_L | from_U
        U = (U | to_U) & ~from_U
        
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

def batch_bvls_bpp(A: torch.Tensor, B: torch.Tensor, chunk_size: int, lower: float = 0.0, upper: float = 1.0) -> torch.Tensor:
    """
    Description:
    Processes BVLS in batches to avoid GPU memory overflow.

    Args:
        A (torch.Tensor): Input matrix A (K x K).
        B (torch.Tensor): Target matrix B (M x K).
        chunk_size (int): Batch size.
        lower (float): Lower bound.
        upper (float): Upper bound.

    Returns:
        torch.Tensor: Bounded solution matrix X.
    """
    m = B.shape[0]
    x = torch.zeros_like(B)
    for i in range(0, m, chunk_size):
        end = min(i + chunk_size, m)
        x[i:end] = bvls_bpp_gpu(A, B[i:end], lower=lower, upper=upper)
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
        
        P = batch_bvls_bpp(A_cov_Q, B_target_P, chunk_size, lower=0.0, upper=1.0)
        mapP(P)
        
        # UPDATE Q
        A_cov_P = P.T @ P
        I_p = P @ torch.linalg.pinv(A_cov_P)
        f_p = (f_64.reshape(1, -1) @ I_p)
        Q_free = 0.5 * (V.to(torch.float64) @ (Z_T @ I_p)) + f_p
        B_target_Q = Q_free @ A_cov_P
        
        Q = batch_bvls_bpp(A_cov_P, B_target_Q, chunk_size, lower=0.0, upper=1.0)
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
