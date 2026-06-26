import logging
import sys
import time
import torch

from ..src import utils
from .em_adam_gpu import EMAdamOptimizer

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def _flatten_PQ_gpu_inplace(P: torch.Tensor, Q: torch.Tensor, out: torch.Tensor) -> None:
    """
    Description:
    Flattens P and Q in-place into a pre-allocated 1-D parameter vector.
    """
    mk = P.numel()
    out[:mk].copy_(P.ravel())
    out[mk:].copy_(Q.ravel())

def _unflatten_PQ_gpu(x: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, K: int) -> None:
    """
    Description:
    Unflattens a 1-D parameter vector back into pre-allocated P and Q matrices.

    Args:
        x (torch.Tensor): Flattened parameter vector of length M*K + N*K.
        P (torch.Tensor): Pre-allocated output P matrix (M x K) on GPU.
        Q (torch.Tensor): Pre-allocated output Q matrix (N x K) on GPU.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.

    Returns:
        None
    """
    P.copy_(x[:M * K].view(M, K))
    Q.copy_(x[M * K:].view(-1, K))

def _mapPQ_gpu(P: torch.Tensor, Q: torch.Tensor) -> None:
    """
    Description:
    Projects P and Q back onto the feasible set (non-negative, Q rows sum to 1).

    Args:
        P (torch.Tensor): P matrix (M x K) on GPU.
        Q (torch.Tensor): Q matrix (N x K) on GPU.

    Returns:
        None
    """
    torch.clamp_(P, 1e-5, 1.0 - 1e-5)
    torch.clamp_(Q, 1e-5, 1.0 - 1e-5)
    Q.div_(Q.sum(dim=1, keepdim=True))

# Compilation
_mapPQ_compiled = torch.compile(_mapPQ_gpu, disable=not hasattr(torch, "compile"))


def _grad_hess_q_chunk(
    G_chunk: torch.Tensor,
    P_chunk: torch.Tensor,
    Q: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    qp = torch.matmul(P_chunk, Q.T)
    qp = torch.clamp(qp, min=1e-10, max=1.0 - 1e-10)

    mask = (G_chunk != 3).to(dtype)
    g = G_chunk.to(dtype)

    coeff_a = (g / qp) * mask
    coeff_b = ((2.0 - g) / (1.0 - qp)) * mask
    one_minus_p = 1.0 - P_chunk

    Xtz_a = torch.matmul(coeff_a.T, P_chunk)
    Xtz_b = torch.matmul(coeff_b.T, one_minus_p)

    H_coeff_a = (g / (qp * qp)) * mask
    H_coeff_b = ((2.0 - g) / ((1.0 - qp) * (1.0 - qp))) * mask

    XtX_a = torch.einsum('ji,jk,jl->ikl', H_coeff_a, P_chunk, P_chunk)
    XtX_b = torch.einsum('ji,jk,jl->ikl', H_coeff_b, one_minus_p, one_minus_p)
    return Xtz_a, Xtz_b, XtX_a, XtX_b


def _grad_hess_p_chunk(
    G_chunk: torch.Tensor,
    P_chunk: torch.Tensor,
    Q: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    qp = torch.matmul(P_chunk, Q.T)
    qp = torch.clamp(qp, min=1e-10, max=1.0 - 1e-10)

    mask = (G_chunk != 3).to(dtype)
    g = G_chunk.to(dtype)

    coeff_a = (g / qp) * mask
    coeff_b = ((2.0 - g) / (1.0 - qp)) * mask
    Xtz = torch.matmul(coeff_a - coeff_b, Q)

    H_coeff = (g / (qp * qp) + (2.0 - g) / ((1.0 - qp) * (1.0 - qp))) * mask
    XtX = torch.einsum('ji,ik,il->jkl', H_coeff, Q, Q)
    return Xtz, XtX


_grad_hess_q_chunk_compiled = torch.compile(_grad_hess_q_chunk, disable=not hasattr(torch, "compile"))
_grad_hess_p_chunk_compiled = torch.compile(_grad_hess_p_chunk, disable=not hasattr(torch, "compile"))


def polish_br_qn_gpu(G: torch.Tensor, P_init: torch.Tensor, Q_init: torch.Tensor,
                    M: int, N: int, K: int, args,
                    device: torch.device, threads_per_block: int,
                    n_iters: int = 3, Q_hist: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Polishes P and Q tensors on GPU using block-relaxation with ZAL quasi-Newton
    acceleration. Designed for cross-validation fold polishing.

    Args:
        G (torch.Tensor): Genotype matrix with held-out entries masked as 3 (M_bytes x N, uint8).
        P_init (torch.Tensor): Global P matrix (M x K) used as warm-start.
        Q_init (torch.Tensor): Global Q matrix (N x K) used as warm-start.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        args (Namespace): Parsed command-line arguments.
        device (torch.device): Target computation device.
        threads_per_block (int): Threads per block for GPU kernels.
        n_iters (int): Number of QN iterations (default 3).
        Q_hist (int): Number of QN history columns (default 3).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Polished (P, Q) tensors.
    """
    P = P_init.clone()
    Q = Q_init.clone()

    # We use EMAdamOptimizer to reuse its accumulation buffers and EM logic,
    # but we skip the Adam step and use pure EM updates.
    optimizer = EMAdamOptimizer(P.shape, Q.shape, 0.0, 0.0, 0.0, 0.0, device)
    unpacker = utils.get_unpacker(device, threads_per_block)
    chunk_size = int(args.chunk_size)

    dim = M * K + N * K
    U = torch.zeros((dim, Q_hist), dtype=P.dtype, device=device)
    V = torch.zeros((dim, Q_hist), dtype=P.dtype, device=device)
    UV_diff = torch.zeros((dim, Q_hist), dtype=P.dtype, device=device)
    x_buf = torch.empty(dim, dtype=P.dtype, device=device)
    x_next_buf = torch.empty(dim, dtype=P.dtype, device=device)
    x_next2_buf = torch.empty(dim, dtype=P.dtype, device=device)
    x_qn = torch.empty(dim, dtype=P.dtype, device=device)

    for it in range(1, n_iters + 1):
        # --- Block-relaxation step 1: (P, Q) -> (P1, Q1) ---
        P1_EM, Q1_EM = optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)
        P1 = P1_EM.clone()
        Q1 = Q1_EM.clone()

        # --- Block-relaxation step 2: (P1, Q1) -> (P2, Q2) ---
        P2_EM, Q2_EM = optimizer.run_em_step(G, P1, Q1, M, chunk_size, unpacker)
        P2 = P2_EM.clone()
        Q2 = Q2_EM.clone()

        # --- Flatten for QN ---
        _flatten_PQ_gpu_inplace(P, Q, x_buf)
        _flatten_PQ_gpu_inplace(P1, Q1, x_next_buf)
        _flatten_PQ_gpu_inplace(P2, Q2, x_next2_buf)

        # --- Update UV history ---
        col = (it - 1) % Q_hist
        torch.sub(x_next_buf, x_buf, out=U[:, col])
        torch.sub(x_next2_buf, x_next_buf, out=V[:, col])

        # --- QN extrapolation ---
        n_cols = min(it, Q_hist)
        U_sub = U[:, :n_cols]
        V_sub = V[:, :n_cols]

        # Linear system: (U^T @ (U - V)) alpha = U^T @ (x - x_next)
        torch.sub(U_sub, V_sub, out=UV_diff[:, :n_cols])
        LHS = U_sub.T @ UV_diff[:, :n_cols]

        torch.sub(x_buf, x_next_buf, out=x_qn)
        RHS = U_sub.T @ x_qn

        try:
            # Solve the tiny (Q_hist x Q_hist) system
            alpha = torch.linalg.solve(LHS, RHS)
        except RuntimeError:
            # Fallback if the system is singular/ill-conditioned
            alpha = torch.linalg.lstsq(LHS, RHS).solution

        # x_qn = x_next_buf - V_sub @ alpha
        torch.matmul(V_sub, alpha, out=x_qn)
        torch.sub(x_next_buf, x_qn, out=x_qn)

        # --- Unflatten and project ---
        _unflatten_PQ_gpu(x_qn, P, Q, M, K)
        _mapPQ_compiled(P, Q)

    del optimizer, U, V, UV_diff, x_buf, x_next_buf, x_next2_buf, x_qn
    return P, Q


def compute_grad_hess_Q_gpu(G: torch.Tensor, Q: torch.Tensor, P: torch.Tensor,
                            XtX_q: torch.Tensor, Xtz_q: torch.Tensor,
                            M: int, chunk_size: int,
                            unpacker, dtype: torch.dtype) -> None:
    """
    Description:
    Computes the gradient and Hessian of the log-likelihood with respect to Q on GPU.

    Args:
        G (torch.Tensor): Packed genotype matrix (M_bytes x N, uint8).
        Q (torch.Tensor): Current Q matrix (N x K).
        P (torch.Tensor): Current P matrix (M x K).
        XtX_q (torch.Tensor): Pre-allocated buffer for Hessian (N x K x K).
        Xtz_q (torch.Tensor): Pre-allocated buffer for Gradient (N x K).
        M (int): Number of SNPs.
        chunk_size (int): Size of chunks to process at once.
        unpacker (function): Unpacker function for packed genotype bytes.
        dtype (torch.dtype): Target computation data type.

    Returns:
        None
    """
    XtX_q.zero_()
    Xtz_q.zero_()
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        actual_chunk_size = end - i
        G_chunk = unpacker(G, i, actual_chunk_size, M) # (C, N)
        P_chunk = P[i:end] # (C, K)
        Xtz_a, Xtz_b, XtX_a, XtX_b = _grad_hess_q_chunk_compiled(G_chunk, P_chunk, Q, dtype)

        Xtz_q.add_(Xtz_a)
        Xtz_q.add_(Xtz_b)
        XtX_q.add_(XtX_a)
        XtX_q.add_(XtX_b)


def compute_grad_hess_P_gpu(G: torch.Tensor, Q: torch.Tensor, P: torch.Tensor,
                            XtX_p: torch.Tensor, Xtz_p: torch.Tensor,
                            M: int, chunk_size: int,
                            unpacker, dtype: torch.dtype) -> None:
    """
    Description:
    Computes the gradient and Hessian of the log-likelihood with respect to P on GPU.

    Args:
        G (torch.Tensor): Packed genotype matrix (M_bytes x N, uint8).
        Q (torch.Tensor): Current Q matrix (N x K).
        P (torch.Tensor): Current P matrix (M x K).
        XtX_p (torch.Tensor): Pre-allocated buffer for Hessian (M x K x K).
        Xtz_p (torch.Tensor): Pre-allocated buffer for Gradient (M x K).
        M (int): Number of SNPs.
        chunk_size (int): Size of chunks to process at once.
        unpacker (function): Unpacker function for packed genotype bytes.
        dtype (torch.dtype): Target computation data type.

    Returns:
        None
    """
    XtX_p.zero_()
    Xtz_p.zero_()
    for i in range(0, M, chunk_size):
        end = min(i + chunk_size, M)
        actual_chunk_size = end - i
        G_chunk = unpacker(G, i, actual_chunk_size, M) # (C, N)
        P_chunk = P[i:end]
        Xtz_chunk, XtX_chunk = _grad_hess_p_chunk_compiled(G_chunk, P_chunk, Q, dtype)

        Xtz_p[i:end] = Xtz_chunk
        XtX_p[i:end] = XtX_chunk


def optimize_original_gpu(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, max_iter: int,
                          K: int, M: int, N: int, tol: float, Q_hist: int,
                          device: torch.device, chunk_size: int, threads_per_block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Optimizes the P and Q matrices using the original ADMIXTURE algorithm on GPU:
    Sequential Quadratic Programming (SQP) block updates with ZAL Quasi-Newton acceleration.

    Args:
        G (torch.Tensor): Packed genotype matrix (M_bytes x N, uint8) on GPU.
        P (torch.Tensor): Pre-initialized P matrix (M x K) on GPU.
        Q (torch.Tensor): Pre-initialized Q matrix (N x K) on GPU.
        max_iter (int): Maximum SQP iterations.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        tol (float): Relative convergence tolerance.
        Q_hist (int): Depth of ZAL Quasi-Newton acceleration history.
        device (torch.device): Computation device.
        chunk_size (int): Size of chunks to process at once.
        threads_per_block (int): Threads per block for GPU kernels.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Optimized (P, Q) tensors.
    """
    # 1. Precompute Vt matrix from SVD of ones(1, K) on GPU
    ones_K = torch.ones((1, K), dtype=torch.float64, device=device)
    _, _, vt = torch.linalg.svd(ones_K, full_matrices=True)
    v_kk = vt.t().contiguous()
    
    # 2. Allocate buffers on GPU
    dtype = utils.get_dtype(device)
    XtX_q = torch.zeros((N, K, K), dtype=torch.float64, device=device)
    Xtz_q = torch.zeros((N, K), dtype=torch.float64, device=device)
    XtX_p = torch.zeros((M, K, K), dtype=torch.float64, device=device)
    Xtz_p = torch.zeros((M, K), dtype=torch.float64, device=device)
    
    # QN history buffers
    dim = M * K + N * K
    U = torch.zeros((dim, Q_hist), dtype=torch.float64, device=device)
    V = torch.zeros((dim, Q_hist), dtype=torch.float64, device=device)
    UV_diff = torch.zeros((dim, Q_hist), dtype=torch.float64, device=device)
    P_qn = torch.empty_like(P)
    Q_qn = torch.empty_like(Q)
    
    # Pre-allocated buffers for ZAL QN acceleration
    x_qn = torch.empty(dim, dtype=torch.float64, device=device)
    x_buf = torch.empty(dim, dtype=torch.float64, device=device)
    x_next_buf = torch.empty(dim, dtype=torch.float64, device=device)
    x_next2_buf = torch.empty(dim, dtype=torch.float64, device=device)
    
    unpacker = utils.get_unpacker(device, threads_per_block)
    logl_calc = utils.get_logl_calculator(device)
    
    # --- Initialize log-likelihood ---
    ll_prev_iter = -float('inf')
    ll_best = -float("inf")
    P_best = torch.empty_like(P)
    Q_best = torch.empty_like(Q)
    
    for it in range(1, max_iter + 1):
        it_start = time.time()
        
        # --- SQP Update 1: (P, Q) -> (P_next, Q_next) ---
        compute_grad_hess_Q_gpu(G, Q, P, XtX_q, Xtz_q, M, chunk_size, unpacker, dtype)
        Xtz_q.neg_()
        Q_next = torch.ops.sqp_kernel.sqp_solve_q_cuda(XtX_q, Xtz_q, Q, v_kk, N, K)
        
        compute_grad_hess_P_gpu(G, Q_next, P, XtX_p, Xtz_p, M, chunk_size, unpacker, dtype)
        Xtz_p.neg_()
        P_next = torch.ops.sqp_kernel.sqp_solve_p_cuda(XtX_p, Xtz_p, P, M, K)
        
        # --- SQP Update 2: (P_next, Q_next) -> (P_next2, Q_next2) ---
        compute_grad_hess_Q_gpu(G, Q_next, P_next, XtX_q, Xtz_q, M, chunk_size, unpacker, dtype)
        Xtz_q.neg_()
        Q_next2 = torch.ops.sqp_kernel.sqp_solve_q_cuda(XtX_q, Xtz_q, Q_next, v_kk, N, K)
        
        compute_grad_hess_P_gpu(G, Q_next2, P_next, XtX_p, Xtz_p, M, chunk_size, unpacker, dtype)
        Xtz_p.neg_()
        P_next2 = torch.ops.sqp_kernel.sqp_solve_p_cuda(XtX_p, Xtz_p, P_next, M, K)
        
        # --- ZAL QN acceleration ---
        _flatten_PQ_gpu_inplace(P, Q, x_buf)
        _flatten_PQ_gpu_inplace(P_next, Q_next, x_next_buf)
        _flatten_PQ_gpu_inplace(P_next2, Q_next2, x_next2_buf)
        
        col = (it - 1) % Q_hist
        torch.sub(x_next_buf, x_buf, out=U[:, col])
        torch.sub(x_next2_buf, x_next_buf, out=V[:, col])
        
        n_cols = min(it, Q_hist)
        U_sub = U[:, :n_cols]
        V_sub = V[:, :n_cols]
        
        torch.sub(U_sub, V_sub, out=UV_diff[:, :n_cols])
        LHS = U_sub.T @ UV_diff[:, :n_cols]
        
        torch.sub(x_buf, x_next_buf, out=x_qn)
        RHS = U_sub.T @ x_qn
        
        try:
            alpha = torch.linalg.solve(LHS, RHS)
        except RuntimeError:
            alpha = torch.linalg.lstsq(LHS, RHS).solution
            
        # x_qn = x_next_buf - V_sub @ alpha
        torch.matmul(V_sub, alpha, out=x_qn)
        torch.sub(x_next_buf, x_qn, out=x_qn)
        
        _unflatten_PQ_gpu(x_qn, P_qn, Q_qn, M, K)
        
        _mapPQ_gpu(P_qn, Q_qn)
        
        # --- Conditional QN Acceptance ---
        ll_qn = logl_calc(G, P_qn, Q_qn, M, N, chunk_size, threads_per_block)
        
        if ll_qn > ll_prev_iter:
            P.copy_(P_qn)
            Q.copy_(Q_qn)
            ll_new = ll_qn
        else:
            P.copy_(P_next2)
            Q.copy_(Q_next2)
            ll_new = logl_calc(G, P_next2, Q_next2, M, N, chunk_size, threads_per_block)

        if ll_new > ll_best:
            ll_best = ll_new
            P_best.copy_(P)
            Q_best.copy_(Q)

        log.info(
            f"    Iteration {it}, "
            f"Log-likelihood: {ll_new:.1f}, "
            f"Time: {time.time() - it_start:.3f}s"
        )
        
        diff = ll_new - ll_prev_iter
        if 0 <= diff < tol:
            log.info(f"    Converged at iteration {it}.")
            break
            
        ll_prev_iter = ll_new
        
    log.info(f"\n    Final log-likelihood: {ll_best:.1f}")
    return P_best, Q_best
