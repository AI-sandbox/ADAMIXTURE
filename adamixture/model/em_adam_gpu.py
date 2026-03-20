import torch
import logging
import time
import numpy as np
from typing import Tuple

log = logging.getLogger(__name__)

def adam_update(param: torch.Tensor, param_target: torch.Tensor, m: torch.Tensor, v: torch.Tensor, 
                t_tensor: torch.Tensor, lr: float, beta1: float, beta2: float, reg_adam: float) -> torch.Tensor:
    """
    Description:
    Performs an Adam optimizer step on a parameter tensor.

    Args:
        param (torch.Tensor): Current parameter values.
        param_target (torch.Tensor): Target gradient/update values.
        m (torch.Tensor): First moment estimate.
        v (torch.Tensor): Second moment estimate.
        t_tensor (torch.Tensor): Current timestep.
        lr (float): Learning rate.
        beta1 (float): Exponential decay rate for first moment.
        beta2 (float): Exponential decay rate for second moment.
        reg_adam (float): Regularization term (epsilon).

    Returns:
        torch.Tensor: Updated parameter tensor.
    """
    delta = param_target - param
    m.mul_(beta1).add_(delta, alpha=1 - beta1)
    v.mul_(beta2).addcmul_(delta, delta, value=1 - beta2)
    
    bias_correction1 = 1 - beta1 ** t_tensor
    bias_correction2 = 1 - beta2 ** t_tensor
    
    m_hat = m / bias_correction1
    v_hat = v / bias_correction2
    
    step_val = lr * m_hat / (torch.sqrt(v_hat) + reg_adam)
    param.add_(step_val)
    return param

adam_update_compiled = torch.compile(adam_update, disable=not hasattr(torch, "compile"))

class EMAdamOptimizer:
    """
    Manages Adam optimization states for Expectation Minimization process.
    """
    def __init__(self, P_shape: torch.Size, Q_shape: torch.Size, lr: float, beta1: float, beta2: float, 
                reg_adam: float, device: torch.device) -> None:
        """
        Description:
        Initializes the EMAdamOptimizer with provided hyperparameters and shapes.

        Args:
            P_shape (torch.Size): Shape of the P matrix.
            Q_shape (torch.Size): Shape of the Q matrix.
            lr (float): Learning rate.
            beta1 (float): Adam beta1 parameter.
            beta2 (float): Adam beta2 parameter.
            reg_adam (float): Regularization term (epsilon).
            device (torch.device): Target computation device.

        Returns:
            None
        """
        self.device = device
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.reg_adam = reg_adam
        self.t = torch.tensor(0.0, device=device, dtype=torch.float32)
        
        self.m_P = torch.zeros(P_shape, dtype=torch.float32, device=device)
        self.v_P = torch.zeros(P_shape, dtype=torch.float32, device=device)
        self.m_Q = torch.zeros(Q_shape, dtype=torch.float32, device=device)
        self.v_Q = torch.zeros(Q_shape, dtype=torch.float32, device=device)

    def step(self, P: torch.Tensor, Q: torch.Tensor, P_target: torch.Tensor, Q_target: torch.Tensor) -> None:
        """
        Description:
        Performs a single Adam optimization step for P and Q matrices.

        Args:
            P (torch.Tensor): Current P matrix.
            Q (torch.Tensor): Current Q matrix.
            P_target (torch.Tensor): Target values/gradients for P.
            Q_target (torch.Tensor): Target values/gradients for Q.

        Returns:
            None
        """
        self.t.add_(1.0)
        
        adam_update_compiled(P, P_target, self.m_P, self.v_P, self.t, 
                             self.lr, self.beta1, self.beta2, self.reg_adam)
        
        adam_update_compiled(Q, Q_target, self.m_Q, self.v_Q, self.t, 
                             self.lr, self.beta1, self.beta2, self.reg_adam)
        
        torch.clamp_(P, 1e-5, 1.0 - 1e-5)
        torch.clamp_(Q, 1e-5, 1.0 - 1e-5)
        Q.div_(Q.sum(dim=1, keepdim=True))

def em_batch_math(G_chunk: torch.Tensor, p_batch: torch.Tensor, Q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Description:
    Computes intermediate EM quantities for a batch of genotypes.

    Args:
        G_chunk (torch.Tensor): Unpacked genotype chunk.
        p_batch (torch.Tensor): Corresponding subset of P matrix.
        Q (torch.Tensor): Current Q matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Intermediate updates (A_part, B_part, T_part, T_sum_part, q_bat_part).
    """
    mask = (G_chunk != 3)
    g_val = G_chunk.float()
    
    rec = torch.matmul(p_batch, Q.T)
    rec = torch.clamp(rec, 1e-5, 1.0-1e-5)

    term_a = (g_val / rec) * mask
    denom_b = 1.0 - rec
    term_b = ((2.0 - g_val) / denom_b) * mask
    
    A_part = torch.matmul(term_a, Q)
    B_part = torch.matmul(term_b, Q)
    
    diff = term_a - term_b
    T_part = torch.matmul(diff.T, p_batch)
    T_sum_part = term_b.sum(dim=0, keepdim=True).T
    
    q_bat_part = mask.float().sum(dim=0) * 2.0
    
    return A_part, B_part, T_part, T_sum_part, q_bat_part

em_batch_compiled = torch.compile(em_batch_math, disable=not hasattr(torch, "compile"))

def em_final_update(P: torch.Tensor, Q: torch.Tensor, A_accum: torch.Tensor, B_accum: torch.Tensor, 
                    T_accum: torch.Tensor, q_bat: torch.Tensor, 
                    P_target: torch.Tensor, Q_target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Finalizes the EM update by combining processed batch results.

    Args:
        P (torch.Tensor): Current P matrix.
        Q (torch.Tensor): Current Q matrix.
        A_accum (torch.Tensor): Accumulated positive updates.
        B_accum (torch.Tensor): Accumulated negative updates.
        T_accum (torch.Tensor): Accumulated Q updates.
        q_bat (torch.Tensor): Accumulated normalization factors.
        P_target (torch.Tensor): Buffer to store updated P.
        Q_target (torch.Tensor): Buffer to store updated Q.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated P_target and Q_target.
    """
    torch.sub(A_accum, B_accum, out=P_target)
    P_target.mul_(P)
    P_target.add_(B_accum)
    torch.clamp(P_target, min=1e-5, out=P_target)
    P_target.reciprocal_()
    P_target.mul_(P)
    P_target.mul_(A_accum)
    
    scale_Q = torch.clamp(q_bat, min=1e-5).reciprocal_().unsqueeze(1)
    torch.mul(Q, T_accum, out=Q_target)
    Q_target.mul_(scale_Q)
    
    sum_Q = Q_target.sum(dim=1, keepdim=True)
    torch.clamp(sum_Q, min=1e-5, out=sum_Q)
    Q_target.div_(sum_Q)
    
    return P_target, Q_target

em_final_compiled = torch.compile(em_final_update, disable=not hasattr(torch, "compile"))

def logl_batch_math(g_chunk: torch.Tensor, p_batch: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Description:
    Computes log-likelihood for a batch of genotypes.

    Args:
        g_chunk (torch.Tensor): Genotype chunk.
        p_batch (torch.Tensor): Subset of P matrix (float64).
        Q (torch.Tensor): Q matrix (float64).

    Returns:
        torch.Tensor: Log-likelihood scalar (float64).
    """
    rec = torch.matmul(p_batch, Q.T)
    rec = torch.clamp(rec, 1e-10, 1.0 - 1e-10)
    
    mask = (g_chunk != 3)
    g_val = g_chunk.to(torch.float64)
    
    ll_chunk = g_val * torch.log(rec) + (2.0 - g_val) * torch.log1p(-rec)
    return (ll_chunk * mask).sum()

logl_batch_compiled = torch.compile(logl_batch_math, disable=not hasattr(torch, "compile"))

def _unpack_chunk_uint8(G: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int, device: torch.device, threads_per_block: int) -> torch.Tensor:
    """
    Description:
    Unpacks a 2-bit genotype chunk into uint8 on the device.

    Args:
        G (torch.Tensor): Packed genotypes.
        start_idx (int): Starting individual index.
        actual_chunk_size (int): Chunk size to unpack.
        M (int): Total number of SNPs.
        device (torch.device): Computation device.
        threads_per_block (int): CUDA threads configuration.

    Returns:
        torch.Tensor: Unpacked uint8 tensor.
    """
    if G.device.type == 'cpu':
        byte_start = start_idx // 4
        byte_end = (start_idx + actual_chunk_size + 3) // 4
        G_sub = G[byte_start:byte_end, :].to(device, non_blocking=True)
        return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G_sub, start_idx, actual_chunk_size, M, byte_start, threads_per_block)
    else:
        return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G, start_idx, actual_chunk_size, M, 0, threads_per_block)

def run_em_step(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, N: int, M: int, device: torch.device, 
                batch_size: int, A_accum: torch.Tensor, B_accum: torch.Tensor, 
                T_accum: torch.Tensor, q_bat: torch.Tensor, P_target: torch.Tensor, Q_target: torch.Tensor, threads_per_block: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Executes a complete EM step iterating over genotype batches on the GPU.

    Args:
        G (torch.Tensor): Packed genotypes.
        P (torch.Tensor): Current frequency matrix.
        Q (torch.Tensor): Current proportion matrix.
        N (int): Number of individuals.
        M (int): Number of SNPs.
        device (torch.device): Computation device.
        batch_size (int): Batch size for processing.
        A_accum (torch.Tensor): Buffer for accumulated A quantities.
        B_accum (torch.Tensor): Buffer for accumulated B quantities.
        T_accum (torch.Tensor): Buffer for accumulated Q updates.
        q_bat (torch.Tensor): Buffer for normalization factors.
        P_target (torch.Tensor): Buffer to store new frequencies.
        Q_target (torch.Tensor): Buffer to store new proportions.
        threads_per_block (int): GPU thread configuration.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Updated P_target and Q_target.
    """
    A_accum.zero_()
    B_accum.zero_()
    T_accum.zero_()
    q_bat.zero_()

    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        actual_chunk_size = end - i
        G_chunk = _unpack_chunk_uint8(G, i, actual_chunk_size, M, device, threads_per_block)
        p_batch = P[i:end]
        A_p, B_p, T_p, T_sum_p, q_p = em_batch_compiled(G_chunk, p_batch, Q)
        A_accum[i:end] = A_p
        B_accum[i:end] = B_p
        T_accum.add_(T_p)
        T_accum.add_(T_sum_p)
        q_bat.add_(q_p)
    em_final_compiled(P, Q, A_accum, B_accum, T_accum, q_bat, P_target, Q_target)
    return P_target, Q_target

def loglikelihood_gpu(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, N: int, batch_size: int, device: torch.device, threads_per_block: int) -> float:
    """
    Description:
    Computes total log-likelihood iterating over chunks in float64.

    Args:
        G (torch.Tensor): Packed genotypes.
        P (torch.Tensor): P matrix.
        Q (torch.Tensor): Q matrix.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        batch_size (int): Processing chunk size.
        device (torch.device): Computation device.
        threads_per_block (int): CUDA threads.

    Returns:
        float: Total log-likelihood.
    """
    ll_tensor = torch.tensor(0.0, dtype=torch.float64, device=device)
    Q_64 = Q.to(torch.float64)
    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        actual_chunk_size = end - i
        G_chunk = _unpack_chunk_uint8(G, i, actual_chunk_size, M, device, threads_per_block)
        p_batch = P[i:end].to(torch.float64)
        ll_tensor.add_(logl_batch_compiled(G_chunk, p_batch, Q_64))
    return ll_tensor.item()

def optimize_parameters_gpu(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, lr: float, beta1: float, beta2: float, 
                  reg_adam: float, max_iter: int, check: int, K: int, M: int, N: int, lr_decay: float, min_lr: float, 
                  patience_adam: int, tol_adam: float, device: torch.device, chunk_size: int, threads_per_block: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Adam-EM optimization on the GPU (float32).

    Args:
        G (torch.Tensor): Packed genotype matrix.
        P (torch.Tensor): Initial allele frequencies.
        Q (torch.Tensor): Initial ancestry proportions.
        lr (float): Learning rate.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        reg_adam (float): Adam epsilon.
        max_iter (int): Maximum iterations.
        check (int): Log-likelihood check frequency.
        K (int): Number of populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        lr_decay (float): LR decay factor.
        min_lr (float): Minimum LR.
        patience_adam (int): Convergence patience.
        tol_adam (float): Convergence tolerance.
        device (torch.device): Computation device.
        chunk_size (int): Batch size.
        threads_per_block (int): GPU thread scaling.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices on CPU.
    """
    optimizer = EMAdamOptimizer(P.shape, Q.shape, lr, beta1, beta2, reg_adam, device)

    wait_lr = 0

    A_accum = torch.zeros_like(P)
    B_accum = torch.zeros_like(P)
    T_accum = torch.zeros_like(Q)
    q_bat = torch.zeros(N, device=device, dtype=torch.float32)
    
    P_EM = torch.zeros_like(P)
    Q_EM = torch.zeros_like(Q)
    
    # Accelerated priming iteration
    ts_priming = time.time()
    run_em_step(G, P, Q, N, M, device, chunk_size, A_accum, B_accum, T_accum, q_bat, P_EM, Q_EM, threads_per_block)
    optimizer.step(P, Q, P_EM, Q_EM)
    run_em_step(G, P, Q, N, M, device, chunk_size, A_accum, B_accum, T_accum, q_bat, P_EM, Q_EM, threads_per_block)
    log.info(f"    Performed priming iteration... ({time.time() - ts_priming:.1f}s)\n")
    
    # Initial log-likelihood
    L_best = loglikelihood_gpu(G, P, Q, M, N, chunk_size, device, threads_per_block)

    P_best = P.clone()
    Q_best = Q.clone()

    ts = time.time()

    for it in range(max_iter):
        with torch.no_grad():
            run_em_step(G, P, Q, N, M, device, chunk_size,
                        A_accum, B_accum, T_accum, q_bat, P_EM, Q_EM, threads_per_block)
            optimizer.step(P, Q, P_EM, Q_EM)

            if (it + 1) % check == 0:
                L_cur = loglikelihood_gpu(G, P, Q, M, N, chunk_size, device, threads_per_block)
                
                log.info(
                    f"    Iteration {it+1}, "
                    f"Log-likelihood: {L_cur:.1f}, "
                    f"Time: {time.time() - ts:.3f}s"
                )
                ts = time.time()

                if L_cur > L_best + tol_adam:
                    L_best = L_cur
                    P_best.copy_(P)
                    Q_best.copy_(Q)
                    wait_lr = 0 
                else:
                    wait_lr += 1
                    
                    if wait_lr >= patience_adam:
                        old_lr = optimizer.lr
                        optimizer.lr = max(optimizer.lr * lr_decay, min_lr)
                        log.info(
                            f"    Plateau reached ({wait_lr} checks without beating best). "
                            f"Reducing lr: {old_lr:.3e} → {optimizer.lr:.3e}"
                        )
                        if optimizer.lr <= min_lr:
                            log.info("        Convergence reached.")
                            break
                        wait_lr = 0

    log.info(f"\n    Final log-likelihood: {L_best:.1f}")

    return P_best.cpu().numpy(), Q_best.cpu().numpy()