import logging
import time
from collections.abc import Callable

import torch

from ..src import utils

log = logging.getLogger(__name__)

def adam_update(param: torch.Tensor, param_target: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                t_tensor: torch.Tensor, lr: float, beta1: float, beta2: float, reg_adam: float) -> torch.Tensor:
    """
    Apply one in-place Adam update toward an EM target tensor.

    Args:
        param: Tensor to update in place.
        param_target: EM target tensor for the same parameter.
        m: First-moment buffer.
        v: Second-moment buffer.
        t_tensor: Adam step counter tensor.
        lr: Adam learning rate.
        beta1: First-moment decay.
        beta2: Second-moment decay.
        reg_adam: Epsilon used in the Adam denominator.

    Returns:
        The updated parameter tensor.
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

def em_batch_math(G_chunk: torch.Tensor, p_batch: torch.Tensor, Q: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute EM sufficient-statistic contributions for one SNP chunk.

    Args:
        G_chunk: Unpacked genotype chunk with shape ``chunk_size x N``.
        p_batch: P rows matching ``G_chunk`` with shape ``chunk_size x K``.
        Q: Current ancestry-proportion matrix with shape ``N x K``.
        dtype: Floating-point dtype used for the computation.

    Returns:
        Per-chunk P numerator terms, Q accumulator terms, and genotype counts.
    """
    mask = (G_chunk != 3).to(dtype)
    g_val = G_chunk.to(dtype)

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

    q_bat_part = mask.sum(dim=0) * 2.0

    return A_part, B_part, T_part, T_sum_part, q_bat_part

em_batch_compiled = torch.compile(em_batch_math, disable=not hasattr(torch, "compile"))

def em_final_update(P: torch.Tensor, Q: torch.Tensor, A_accum: torch.Tensor, B_accum: torch.Tensor,
                    T_accum: torch.Tensor, q_bat: torch.Tensor,
                    P_target: torch.Tensor, Q_target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert accumulated EM statistics into P and Q target tensors.

    Args:
        P: Current allele-frequency matrix.
        Q: Current ancestry-proportion matrix.
        A_accum: Accumulated P numerator term.
        B_accum: Accumulated P denominator complement term.
        T_accum: Accumulated Q update term.
        q_bat: Non-missing genotype counts per individual.
        P_target: Preallocated output buffer for the EM P target.
        Q_target: Preallocated output buffer for the EM Q target.

    Returns:
        ``(P_target, Q_target)`` after in-place normalization/clipping.
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

class EMAdamOptimizer:
    """
    Manages Adam optimization states for Expectation Minimization process.
    """
    def __init__(self, P_shape: torch.Size, Q_shape: torch.Size, lr: float, beta1: float, beta2: float,
                reg_adam: float, device: torch.device) -> None:
        """
        Description:
        Initializes Adam-EM optimizer state tensors and EM target buffers.

        Args:
            P_shape (torch.Size): Shape of the P tensor.
            Q_shape (torch.Size): Shape of the Q tensor.
            lr (float): Adam learning rate.
            beta1 (float): Adam beta1.
            beta2 (float): Adam beta2.
            reg_adam (float): Adam epsilon.
            device (torch.device): Device for optimizer buffers.

        Returns:
            None
        """
        self.device = device
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.reg_adam = reg_adam
        self.dtype = utils.get_dtype(device)
        self.t = torch.tensor(0.0, device=device, dtype=self.dtype)

        self.m_P = torch.zeros(P_shape, dtype=self.dtype, device=device)
        self.v_P = torch.zeros(P_shape, dtype=self.dtype, device=device)
        self.m_Q = torch.zeros(Q_shape, dtype=self.dtype, device=device)
        self.v_Q = torch.zeros(Q_shape, dtype=self.dtype, device=device)

        # Accumulation buffers
        self.A_accum = torch.zeros(P_shape, dtype=self.dtype, device=device)
        self.B_accum = torch.zeros(P_shape, dtype=self.dtype, device=device)
        self.T_accum = torch.zeros(Q_shape, dtype=self.dtype, device=device)
        self.q_bat = torch.zeros(Q_shape[0], dtype=self.dtype, device=device)

        # Target buffers (P_EM, Q_EM)
        self.P_EM = torch.zeros(P_shape, dtype=self.dtype, device=device)
        self.Q_EM = torch.zeros(Q_shape, dtype=self.dtype, device=device)

    def run_em_step(self, G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor,
                   M: int, chunk_size: int, unpacker: Callable) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run one full EM sweep over the genotype matrix.

        Args:
            G: Packed genotype matrix.
            P: Current allele-frequency matrix.
            Q: Current ancestry-proportion matrix.
            M: Number of SNPs.
            chunk_size: Number of SNP rows processed per batch.
            unpacker: Callable that unpacks a genotype chunk onto the target device.

        Returns:
            EM target tensors ``(P_EM, Q_EM)``.
        """
        self.A_accum.zero_()
        self.B_accum.zero_()
        self.T_accum.zero_()
        self.q_bat.zero_()

        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            actual_chunk_size = end - i
            G_chunk = unpacker(G, i, actual_chunk_size, M)
            p_batch = P[i:end]
            A_p, B_p, T_p, T_sum_p, q_p = em_batch_compiled(G_chunk, p_batch, Q, self.dtype)
            self.A_accum[i:end] = A_p
            self.B_accum[i:end] = B_p
            self.T_accum.add_(T_p)
            self.T_accum.add_(T_sum_p)
            self.q_bat.add_(q_p)

        return em_final_compiled(P, Q, self.A_accum, self.B_accum, self.T_accum, self.q_bat, self.P_EM, self.Q_EM)

    def step(self, P: torch.Tensor, Q: torch.Tensor, P_target: torch.Tensor, Q_target: torch.Tensor) -> None:
        """
        Apply one Adam-accelerated EM update to P and Q in place.

        Args:
            P: Allele-frequency matrix updated in place.
            Q: Ancestry-proportion matrix updated in place.
            P_target: EM target for P.
            Q_target: EM target for Q.

        Returns:
            None.
        """
        self.t.add_(1.0)

        adam_update_compiled(P, P_target, self.m_P, self.v_P, self.t,
                             self.lr, self.beta1, self.beta2, self.reg_adam)

        adam_update_compiled(Q, Q_target, self.m_Q, self.v_Q, self.t,
                             self.lr, self.beta1, self.beta2, self.reg_adam)

        torch.clamp_(P, 1e-5, 1.0 - 1e-5)
        torch.clamp_(Q, 1e-5, 1.0 - 1e-5)
        Q.div_(Q.sum(dim=1, keepdim=True))

def logl_batch_math(g_chunk: torch.Tensor, p_batch: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Compute log-likelihood contribution for one unpacked genotype chunk.

    Args:
        g_chunk: Unpacked genotype chunk with missing values encoded as 3.
        p_batch: P rows matching the chunk.
        Q: Current ancestry-proportion matrix.

    Returns:
        Scalar tensor containing the chunk log-likelihood contribution.
    """
    rec = torch.matmul(p_batch, Q.T)
    rec = torch.clamp(rec, 1e-10, 1.0 - 1e-10)

    mask = (g_chunk != 3)
    g_val = g_chunk.to(torch.float64)
    rec_64 = rec.to(torch.float64)

    ll_chunk = g_val * torch.log(rec_64) + (2.0 - g_val) * torch.log1p(-rec_64)
    return (ll_chunk * mask).sum()

logl_batch_compiled = torch.compile(logl_batch_math, disable=not hasattr(torch, "compile"))

def loglikelihood_gpu(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, N: int, batch_size: int, device: torch.device, threads_per_block: int) -> float:
    """
    Compute the full genotype log-likelihood on GPU/MPS in SNP batches.

    Args:
        G: Packed genotype matrix.
        P: Allele-frequency matrix.
        Q: Ancestry-proportion matrix.
        M: Number of SNPs.
        N: Number of individuals.
        batch_size: Number of SNP rows per batch.
        device: Computation device.
        threads_per_block: CUDA kernel tuning parameter.

    Returns:
        Total log-likelihood as a Python float.
    """
    if device.type == 'mps':
        return utils.loglikelihood_cpu_chunked(G, P, Q, M, batch_size)

    ll_tensor = torch.tensor(0.0, dtype=torch.float64, device=device)
    Q_64 = Q.to(torch.float64)
    unpacker = utils.get_unpacker(device, threads_per_block)
    for i in range(0, M, batch_size):
        end = min(i + batch_size, M)
        actual_chunk_size = end - i
        G_chunk = unpacker(G, i, actual_chunk_size, M)
        p_batch = P[i:end].to(torch.float64)
        ll_tensor.add_(logl_batch_compiled(G_chunk, p_batch, Q_64))
    return ll_tensor.item()

def _optimize_parameters_gpu_once(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, lr: float, beta1: float, beta2: float,
                  reg_adam: float, max_iter: int, check: int, K: int, M: int, N: int, lr_decay: float, min_lr: float,
                  patience: int, tol_adam: float, device: torch.device, chunk_size: int, threads_per_block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Optimize P and Q on GPU using Adam-accelerated EM with learning-rate decay.

    Args:
        G (torch.Tensor): Input genotype matrix.
        P (torch.Tensor): Initial P matrix (frequencies).
        Q (torch.Tensor): Initial Q matrix (proportions).
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1 parameter.
        beta2 (float): Adam beta2 parameter.
        reg_adam (float): Adam epsilon for numerical stability.
        max_iter (int): Maximum number of Adam-EM iterations.
        check (int): Frequency of log-likelihood evaluation and checkpointing.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs (rows in G).
        N (int): Number of individuals (columns in G).
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate value.
        patience (int): Number of checks without improvement before reducing the learning rate.
        tol_adam (float): Convergence tolerance for log-likelihood.
        device (torch.device): GPU computation device.
        chunk_size (int): Batch size to process genotypes.
        threads_per_block (int): CUDA thread scaling factor.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Optimized P and Q matrices.
    """
    optimizer = EMAdamOptimizer(P.shape, Q.shape, lr, beta1, beta2, reg_adam, device)

    wait_lr = 0
    unpacker = utils.get_unpacker(device, threads_per_block)
    logl_calc = utils.get_logl_calculator(device)

    # Accelerated priming iteration
    ts_priming = time.time()
    optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)
    optimizer.step(P, Q, optimizer.P_EM, optimizer.Q_EM)
    optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)
    log.info(f"    Performed priming iteration... ({time.time() - ts_priming:.1f}s)\n")

    L_best = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
    P_best = P.clone()
    Q_best = Q.clone()
    ts = time.time()

    for it in range(max_iter):
        optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)
        optimizer.step(P, Q, optimizer.P_EM, optimizer.Q_EM)

        if (it + 1) % check == 0:
            L_cur = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
            log.info(f"    Iteration {it+1}, Log-likelihood: {L_cur:.1f}, Time: {time.time() - ts:.3f}s")
            ts = time.time()

            if L_cur > L_best + tol_adam:
                L_best = L_cur
                P_best.copy_(P)
                Q_best.copy_(Q)
                wait_lr = 0
            else:
                wait_lr += 1
                if wait_lr >= patience:
                    old_lr = optimizer.lr
                    optimizer.lr = max(optimizer.lr * lr_decay, min_lr)
                    log.info(f"    Plateau reached ({wait_lr} checks without beating best). Reducing lr: {old_lr:.3e} → {optimizer.lr:.3e}")
                    if optimizer.lr <= min_lr:
                        log.info("        Convergence reached.")
                        break
                    wait_lr = 0

    log.info(f"\n    Final log-likelihood: {L_best:.1f}")
    return P_best, Q_best


def optimize_parameters_gpu(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, lr: float, beta1: float, beta2: float,
                  reg_adam: float, max_iter: int, check: int, K: int, M: int, N: int, lr_decay: float, min_lr: float,
                  patience: int, tol_adam: float, device: torch.device, chunk_size: int, threads_per_block: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run Adam-EM on GPU, retrying the full optimizer with smaller chunks on CUDA OOM.
    """
    current_chunk_size = chunk_size
    P_attempt = torch.empty_like(P)
    Q_attempt = torch.empty_like(Q)

    while True:
        P_attempt.copy_(P)
        Q_attempt.copy_(Q)
        try:
            if current_chunk_size != chunk_size:
                log.warning(f"    Retrying Adam-EM with chunk_size={current_chunk_size}.")
            return _optimize_parameters_gpu_once(
                G, P_attempt, Q_attempt, lr, beta1, beta2, reg_adam, max_iter,
                check, K, M, N, lr_decay, min_lr, patience, tol_adam, device,
                current_chunk_size, threads_per_block
            )
        except RuntimeError as exc:
            if not utils.is_gpu_oom(exc) or current_chunk_size <= utils.MIN_GPU_CHUNK_SIZE:
                raise
            current_chunk_size = utils.reduce_gpu_chunk_or_raise(
                current_chunk_size, exc, "running Adam-EM"
            )
