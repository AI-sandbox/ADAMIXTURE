import logging
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from . import utils
from .utils_c import em, sqp, tools

if TYPE_CHECKING:
    import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── CPU (numpy) implementation ────────────────────────────────────────────────

def _q_adam_step_cpu(G: np.ndarray, P: np.ndarray, Q0: np.ndarray, T: np.ndarray,
                    Q1: np.ndarray, q_bat: np.ndarray, K: int, M: int, N: int,
                    m_Q: np.ndarray, v_Q: np.ndarray, t: list,
                    lr: float, beta1: float, beta2: float, epsilon: float) -> None:
    """
    Description:
    Single EM step for Q only (P fixed), followed by an Adam update of Q.
    A temporary P_dummy buffer captures the P-side EM output so it can be
    discarded — only the accumulated Q statistics are used.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P (np.ndarray): Fixed allele-frequency matrix (M x K). NOT modified.
        Q0 (np.ndarray): Current Q matrix (N x K). Updated in-place via Adam.
        T (np.ndarray): Temporary accumulator for Q terms (N x K).
        Q1 (np.ndarray): Buffer for EM-updated Q (N x K).
        q_bat (np.ndarray): Per-sample genotype-count accumulator (N,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of samples.
        m_Q (np.ndarray): First Adam moment for Q (N x K).
        v_Q (np.ndarray): Second Adam moment for Q (N x K).
        t (list): Single-element list holding the Adam time-step counter.
        lr (float): Learning rate.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        epsilon (float): Adam epsilon for numerical stability.

    Returns:
        None: Q0 is updated in-place.
    """
    P_dummy = np.empty_like(P)
    em.P_step(G, P, P_dummy, Q0, T, q_bat, K, M, N)
    em.Q_step(Q0, Q1, T, q_bat, K, N)

    t_val = t[0] + 1
    em.adamUpdateQ(Q0, Q1, m_Q, v_Q, lr, beta1, beta2, epsilon, t_val, N, K)
    t[0] = t_val


def _q_em_step_cpu(G: np.ndarray, P: np.ndarray, Q0: np.ndarray, T: np.ndarray,
                Q1: np.ndarray, q_bat: np.ndarray, K: int, M: int, N: int) -> None:
    """
    Description:
    Plain EM step for Q only (P fixed). Used for priming iterations before
    the main Adam-EM loop starts.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P (np.ndarray): Fixed allele-frequency matrix (M x K). NOT modified.
        Q0 (np.ndarray): Current Q matrix (N x K). Updated in-place.
        T (np.ndarray): Temporary accumulator for Q terms (N x K).
        Q1 (np.ndarray): Buffer for EM-updated Q (N x K).
        q_bat (np.ndarray): Per-sample genotype-count accumulator (N,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of samples.

    Returns:
        None: Q0 is updated in-place.
    """
    P_dummy = np.empty_like(P)
    em.P_step(G, P, P_dummy, Q0, T, q_bat, K, M, N)
    em.Q_step(Q0, Q1, T, q_bat, K, N)
    Q0[:] = Q1


def optimize_projection(G: np.ndarray, P: np.ndarray, Q: np.ndarray,
                    lr: float, beta1: float, beta2: float, reg_adam: float,
                    max_iter: int, check: int, K: int, M: int, N: int,
                    lr_decay: float, min_lr: float, patience: int, tol_adam: float) -> np.ndarray:
    """
    Description:
    Projects target samples onto pre-trained allele frequencies P using
    Adam-EM on the CPU (numpy path).  P is kept fixed throughout — only
    Q (ancestry proportions) is optimised.

    Args:
        G (np.ndarray): Genotype matrix for the target samples (M x N, uint8).
        P (np.ndarray): Pre-trained, fixed allele-frequency matrix (M x K).
        Q (np.ndarray): Initial Q matrix (N x K). Updated in-place.
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        reg_adam (float): Adam epsilon.
        max_iter (int): Maximum Adam-EM iterations.
        check (int): Frequency of log-likelihood evaluation.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of samples.
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate.
        patience (int): Checks without improvement before decaying lr.
        tol_adam (float): Convergence tolerance on log-likelihood.

    Returns:
        np.ndarray: Optimised Q matrix (N x K).
    """
    m_Q = np.zeros_like(Q, dtype=np.float64)
    v_Q = np.zeros_like(Q, dtype=np.float64)
    t = [0]

    Q1 = np.zeros_like(Q, dtype=np.float64)
    T = np.zeros_like(Q, dtype=np.float64)
    q_bat = np.zeros(N, dtype=np.float64)

    Q_best = np.empty_like(Q)
    L_best = float("-inf")
    wait_lr = 0

    ts = time.time()

    log.info("    Performing priming iteration...")
    ts_p = time.time()
    _q_em_step_cpu(G, P, Q, T, Q1, q_bat, K, M, N)
    _q_adam_step_cpu(G, P, Q, T, Q1, q_bat, K, M, N, m_Q, v_Q, t, lr, beta1, beta2, reg_adam)
    _q_em_step_cpu(G, P, Q, T, Q1, q_bat, K, M, N)
    log.info(f"    Priming done. ({time.time() - ts_p:.1f}s)\n")

    L_best = tools.loglikelihood(G, P, Q)
    Q_best[:] = Q

    for it in range(max_iter):
        _q_adam_step_cpu(G, P, Q, T, Q1, q_bat, K, M, N, m_Q, v_Q, t, lr, beta1, beta2, reg_adam)

        if (it + 1) % check == 0:
            L_cur = tools.loglikelihood(G, P, Q)
            log.info(
                f"    Iteration {it + 1}, "
                f"Log-likelihood: {L_cur:.1f}, "
                f"Time: {time.time() - ts:.3f}s"
            )
            ts = time.time()

            if L_cur > L_best + tol_adam:
                L_best = L_cur
                Q_best[:] = Q
                wait_lr = 0
            else:
                wait_lr += 1
                if wait_lr >= patience:
                    old_lr = lr
                    lr = max(lr * lr_decay, min_lr)
                    log.info(
                        f"    Plateau ({wait_lr} checks). "
                        f"Reducing lr: {old_lr:.3e} → {lr:.3e}"
                    )
                    if lr <= min_lr:
                        log.info("    Convergence reached.")
                        break
                    wait_lr = 0

    log.info(f"\n    Final log-likelihood (projection): {L_best:.1f}")
    return Q_best


# ── GPU (torch) implementation ────────────────────────────────────────────────

def optimize_projection_gpu(G: "torch.Tensor", P: "torch.Tensor", Q: "torch.Tensor",
                        lr: float, beta1: float, beta2: float, reg_adam: float,
                        max_iter: int, check: int, M: int,
                        lr_decay: float, min_lr: float, patience: int, tol_adam: float,
                        device: "torch.device", chunk_size: int, threads_per_block: int) -> "torch.Tensor":
    """
    Description:
    GPU (torch) projection mode: fixes P and optimises Q using Adam on the
    accumulated EM statistics. Updates only Q; P-side EM output is discarded.

    Args:
        G (torch.Tensor): Genotype tensor (packed or unpacked, on CPU or GPU).
        P (torch.Tensor): Pre-trained, fixed allele-frequency tensor (M x K). NOT modified.
        Q (torch.Tensor): Initial Q tensor (N x K). Updated in-place.
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        reg_adam (float): Adam epsilon.
        max_iter (int): Maximum iterations.
        check (int): Log-likelihood check frequency.
        M (int): Number of SNPs.
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate.
        patience (int): Patience before lr decay.
        tol_adam (float): Convergence tolerance.
        device (torch.device): Computation device.
        chunk_size (int): SNP chunk size for batched EM.
        threads_per_block (int): CUDA tuning parameter.

    Returns:
        torch.Tensor: Optimised Q tensor (N x K).
    """
    import torch

    from ..model.em_adam_gpu import adam_update_compiled, em_batch_compiled, em_final_compiled

    N = Q.shape[0]
    dtype = utils.get_dtype(device)
    P = P.to(dtype)
    Q = Q.to(dtype)

    # Adam state for Q only
    m_Q = torch.zeros_like(Q)
    v_Q = torch.zeros_like(Q)
    t_tensor = torch.tensor(0.0, device=device, dtype=dtype)

    # Accumulation buffers
    A_accum = torch.zeros_like(P)
    B_accum = torch.zeros_like(P)
    T_accum = torch.zeros_like(Q)
    q_bat = torch.zeros(N, dtype=dtype, device=device)
    P_EM = torch.zeros_like(P)
    Q_EM = torch.zeros_like(Q)

    unpacker = utils.get_unpacker(device, threads_per_block)
    logl_calc = utils.get_logl_calculator(device)

    def run_em_q_step() -> "torch.Tensor":
        """
        Description:
        Full EM step accumulating Q statistics; P-side output is discarded.
        Iterates over SNP chunks, accumulates EM sufficient statistics, and
        returns the EM-updated Q tensor.

        Args:
            None.

        Returns:
            torch.Tensor: EM-updated Q tensor (N x K).
        """
        A_accum.zero_()
        B_accum.zero_()
        T_accum.zero_()
        q_bat.zero_()
        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            actual = end - i
            G_chunk = unpacker(G, i, actual, M)
            p_batch = P[i:end]
            A_p, B_p, T_p, T_sum_p, q_p = em_batch_compiled(G_chunk, p_batch, Q, dtype)
            A_accum[i:end] = A_p
            B_accum[i:end] = B_p
            T_accum.add_(T_p)
            T_accum.add_(T_sum_p)
            q_bat.add_(q_p)
        # Only finalise Q (discard P_EM)
        _, Q_out = em_final_compiled(P, Q, A_accum, B_accum, T_accum, q_bat, P_EM, Q_EM)
        return Q_out

    def q_adam_step(Q_target: "torch.Tensor") -> None:
        """
        Description:
        Applies one Adam update to Q using the EM target Q_target, then
        clamps and row-normalises Q to keep it on the probability simplex.

        Args:
            Q_target (torch.Tensor): EM-updated Q tensor used as the Adam target (N x K).

        Returns:
            None: Q is updated in-place.
        """
        t_tensor.add_(1.0)
        adam_update_compiled(Q, Q_target, m_Q, v_Q, t_tensor, lr, beta1, beta2, reg_adam)
        torch.clamp_(Q, 1e-5, 1.0 - 1e-5)
        Q.div_(Q.sum(dim=1, keepdim=True))

    # Priming
    ts_p = time.time()
    Q_target = run_em_q_step()
    q_adam_step(Q_target)
    run_em_q_step()
    log.info(f"    Priming done. ({time.time() - ts_p:.1f}s)\n")

    L_best = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
    Q_best = Q.clone()
    wait_lr = 0
    ts = time.time()

    for it in range(max_iter):
        Q_target = run_em_q_step()
        q_adam_step(Q_target)

        if (it + 1) % check == 0:
            L_cur = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
            log.info(f"    Iteration {it + 1}, Log-likelihood: {L_cur:.1f}, Time: {time.time() - ts:.3f}s")
            ts = time.time()

            if L_cur > L_best + tol_adam:
                L_best = L_cur
                Q_best.copy_(Q)
                wait_lr = 0
            else:
                wait_lr += 1
                if wait_lr >= patience:
                    old_lr = lr
                    lr = max(lr * lr_decay, min_lr)
                    log.info(f"    Plateau ({wait_lr} checks). Reducing lr: {old_lr:.3e} → {lr:.3e}")
                    if lr <= min_lr:
                        log.info("    Convergence reached.")
                        break
                    wait_lr = 0

    log.info(f"\n    Final log-likelihood (projection): {L_best:.1f}")
    return Q_best


def _flatten_Q_inplace(Q: np.ndarray, out: np.ndarray) -> None:
    """
    Description:
    Flattens Q in-place into a pre-allocated 1-D parameter vector.

    Args:
        Q (np.ndarray): Q matrix (N x K).
        out (np.ndarray): Pre-allocated flat output buffer.

    Returns:
        None
    """
    memoryview(out[:])[:] = Q.ravel()


def _unflatten_Q(x: np.ndarray, Q_out: np.ndarray, N: int, K: int) -> None:
    """
    Description:
    Unflattens a 1-D parameter vector into a pre-allocated Q matrix.

    Args:
        x (np.ndarray): Flat parameter vector of length N*K.
        Q_out (np.ndarray): Pre-allocated output Q matrix (N x K).
        N (int): Number of individuals.
        K (int): Number of ancestral populations.

    Returns:
        None
    """
    memoryview(Q_out.ravel())[:] = memoryview(x[:])


def optimize_projection_original(G: np.ndarray, P: np.ndarray, Q: np.ndarray,
                                 max_iter: int, K: int, M: int, N: int, tol: float,
                                 Q_hist: int, patience: int) -> np.ndarray:
    """
    Description:
    Projects target samples onto pre-trained allele frequencies P using the
    original ADMIXTURE algorithm (SQP updates + ZAL QN acceleration on Q) on CPU.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P (np.ndarray): Fixed allele-frequency matrix (M x K).
        Q (np.ndarray): Initial ancestry-proportion matrix (N x K). Updated in-place.
        max_iter (int): Maximum number of SQP + ZAL-QN iterations.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        tol (float): Convergence tolerance for log-likelihood improvement.
        Q_hist (int): ZAL-QN history depth.
        patience (int): Iterations without best log-likelihood improvement before stopping.

    Returns:
        np.ndarray: Optimised Q matrix.
    """
    from .utils_c.cython.br_qn import qn_extrapolate_ZAL, update_UV_ZAL

    # 1. Precompute Vt matrix from SVD of ones(1, K)
    _, _, vt = np.linalg.svd(np.ones((1, K)), full_matrices=True)
    v_kk = np.ascontiguousarray(vt.T, dtype=np.float64)

    # 2. Allocate buffers
    XtX_q = np.empty((N, K, K), dtype=np.float64)
    Xtz_q = np.empty((N, K), dtype=np.float64)

    Q_next = np.empty_like(Q, dtype=np.float64)
    Q_next2 = np.empty_like(Q, dtype=np.float64)

    # QN history buffers on Q only (dim = N * K)
    dim = N * K
    U = np.zeros(dim * Q_hist, dtype=np.float64)
    V = np.zeros(dim * Q_hist, dtype=np.float64)
    UtUmV_workspace = np.zeros(Q_hist * (Q_hist + 1), dtype=np.float64)
    coeff_workspace = np.zeros(Q_hist, dtype=np.float64)

    # QN extrapolation buffers
    x_qn = np.empty(dim, dtype=np.float64)
    Q_qn = np.empty_like(Q)

    # Pre-allocated buffers for ZAL QN acceleration
    x_buf = np.empty(dim, dtype=np.float64)
    x_next_buf = np.empty(dim, dtype=np.float64)
    x_next2_buf = np.empty(dim, dtype=np.float64)

    # 3. Initialize log-likelihood
    ll_prev_iter = -float('inf')
    ll_best = -float("inf")
    wait = 0
    Q_best = np.empty_like(Q)

    for it in range(1, max_iter + 1):
        it_start = time.time()

        # --- SQP Update 1: Q -> Q_next ---
        sqp.update_q_sqp(G, Q, Q_next, P, XtX_q, Xtz_q, v_kk, M, N, K)

        # --- SQP Update 2: Q_next -> Q_next2 ---
        sqp.update_q_sqp(G, Q_next, Q_next2, P, XtX_q, Xtz_q, v_kk, M, N, K)

        # --- ZAL QN acceleration ---
        _flatten_Q_inplace(Q, x_buf)
        _flatten_Q_inplace(Q_next, x_next_buf)
        _flatten_Q_inplace(Q_next2, x_next2_buf)

        update_UV_ZAL(U, V, x_buf, x_next_buf, x_next2_buf, it, Q_hist, dim)

        n_cols = min(it, Q_hist)
        qn_extrapolate_ZAL(x_qn, x_next_buf, x_buf, U, V, n_cols, dim, UtUmV_workspace, coeff_workspace)

        _unflatten_Q(x_qn, Q_qn, N, K)
        sqp.project_q_simplex(Q_qn, N, K)

        # --- Conditional QN Acceptance ---
        ll_qn = tools.loglikelihood(G, P, Q_qn)

        if ll_qn > ll_prev_iter:
            memoryview(Q.ravel())[:] = memoryview(Q_qn.ravel())
            ll_new = ll_qn
        else:
            memoryview(Q.ravel())[:] = memoryview(Q_next2.ravel())
            ll_new = tools.loglikelihood(G, P, Q_next2)

        best_before = ll_best
        if ll_new > ll_best:
            ll_best = ll_new
            memoryview(Q_best.ravel())[:] = memoryview(Q.ravel())
        if best_before != -float("inf") and ll_new <= best_before + tol:
            wait += 1
        else:
            wait = 0

        log.info(
            f"    Iteration {it}, "
            f"Log-likelihood: {ll_new:.1f}, "
            f"Time: {time.time() - it_start:.3f}s"
        )

        diff = ll_new - ll_prev_iter
        if abs(diff) < tol:
            log.info(f"    Converged at iteration {it}.")
            break
        if wait >= patience:
            log.info(f"    Converged at iteration {it} after {wait} iterations without improvement.")
            break

        ll_prev_iter = ll_new

    log.info(f"\n    Final log-likelihood (projection): {ll_best:.1f}")
    return Q_best


def _flatten_Q_gpu_inplace(Q: "torch.Tensor", out: "torch.Tensor") -> None:
    """
    Description:
    Flattens Q in-place into a pre-allocated GPU parameter vector.

    Args:
        Q (torch.Tensor): Q tensor (N x K).
        out (torch.Tensor): Pre-allocated flat output tensor.

    Returns:
        None
    """
    out[:].copy_(Q.ravel())


def _unflatten_Q_gpu(x: "torch.Tensor", Q: "torch.Tensor") -> None:
    """
    Description:
    Unflattens a GPU parameter vector into a pre-allocated Q tensor.

    Args:
        x (torch.Tensor): Flat parameter tensor.
        Q (torch.Tensor): Pre-allocated output Q tensor.

    Returns:
        None
    """
    Q.copy_(x.view(-1, Q.shape[1]))


def optimize_projection_original_gpu(G: "torch.Tensor", P: "torch.Tensor", Q: "torch.Tensor",
                                     max_iter: int, K: int, M: int, N: int, tol: float, Q_hist: int,
                                     patience: int,
                                     device: "torch.device", chunk_size: int, threads_per_block: int) -> "torch.Tensor":
    """
    Description:
    Projects target samples onto pre-trained allele frequencies P using the
    original ADMIXTURE algorithm (SQP updates + ZAL QN acceleration on Q) on GPU.

    Args:
        G (torch.Tensor): Packed genotype tensor.
        P (torch.Tensor): Fixed allele-frequency tensor (M x K).
        Q (torch.Tensor): Initial ancestry-proportion tensor (N x K). Updated in-place.
        max_iter (int): Maximum number of SQP + ZAL-QN iterations.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        tol (float): Convergence tolerance for log-likelihood improvement.
        Q_hist (int): ZAL-QN history depth.
        patience (int): Iterations without best log-likelihood improvement before stopping.
        device (torch.device): CUDA device.
        chunk_size (int): SNP chunk size for batched GPU work.
        threads_per_block (int): CUDA tuning parameter.

    Returns:
        torch.Tensor: Optimised Q tensor.
    """
    import torch

    from ..model.br_qn_gpu import compute_grad_hess_Q_gpu

    dtype = utils.get_dtype(device)

    # 1. Precompute Vt matrix from SVD of ones(1, K). Keep this on CPU first
    # so the MPS path never requests float64 linalg kernels on-device.
    ones_K = torch.ones((1, K), dtype=torch.float64)
    _, _, vt = torch.linalg.svd(ones_K, full_matrices=True)
    v_kk = vt.t().to(device=device, dtype=dtype).contiguous()

    # 2. Allocate buffers on GPU
    XtX_q = torch.zeros((N, K, K), dtype=dtype, device=device)
    Xtz_q = torch.zeros((N, K), dtype=dtype, device=device)

    # QN history buffers on Q only (dim = N * K)
    dim = N * K
    U = torch.zeros((dim, Q_hist), dtype=dtype, device=device)
    V = torch.zeros((dim, Q_hist), dtype=dtype, device=device)
    UV_diff = torch.zeros((dim, Q_hist), dtype=dtype, device=device)
    Q_qn = torch.empty_like(Q)

    # Pre-allocated buffers for ZAL QN acceleration
    x_qn = torch.empty(dim, dtype=dtype, device=device)
    x_buf = torch.empty(dim, dtype=dtype, device=device)
    x_next_buf = torch.empty(dim, dtype=dtype, device=device)
    x_next2_buf = torch.empty(dim, dtype=dtype, device=device)

    unpacker = utils.get_unpacker(device, threads_per_block)
    logl_calc = utils.get_logl_calculator(device)
    if device.type == "mps":
        from .utils_c import metal

        sqp_solve_q = metal.sqp_solve_q_mps
    else:
        sqp_solve_q = torch.ops.sqp_kernel.sqp_solve_q_cuda

    # --- Initialize log-likelihood ---
    ll_prev_iter = -float('inf')
    ll_best = -float("inf")
    wait = 0
    Q_best = torch.empty_like(Q)

    for it in range(1, max_iter + 1):
        it_start = time.time()

        # --- SQP Update 1: Q -> Q_next ---
        compute_grad_hess_Q_gpu(G, Q, P, XtX_q, Xtz_q, M, chunk_size, unpacker, dtype)
        Xtz_q.neg_()
        Q_next = sqp_solve_q(XtX_q, Xtz_q, Q, v_kk, N, K)

        # --- SQP Update 2: Q_next -> Q_next2 ---
        compute_grad_hess_Q_gpu(G, Q_next, P, XtX_q, Xtz_q, M, chunk_size, unpacker, dtype)
        Xtz_q.neg_()
        Q_next2 = sqp_solve_q(XtX_q, Xtz_q, Q_next, v_kk, N, K)

        # --- ZAL QN acceleration ---
        _flatten_Q_gpu_inplace(Q, x_buf)
        _flatten_Q_gpu_inplace(Q_next, x_next_buf)
        _flatten_Q_gpu_inplace(Q_next2, x_next2_buf)

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

        _unflatten_Q_gpu(x_qn, Q_qn)

        # Project Q to simplex
        torch.clamp_(Q_qn, 1e-5, 1.0 - 1e-5)
        Q_qn.div_(Q_qn.sum(dim=1, keepdim=True))

        # --- Conditional QN Acceptance ---
        ll_qn = logl_calc(G, P, Q_qn, M, N, chunk_size, threads_per_block)

        if ll_qn > ll_prev_iter:
            Q.copy_(Q_qn)
            ll_new = ll_qn
        else:
            Q.copy_(Q_next2)
            ll_new = logl_calc(G, P, Q_next2, M, N, chunk_size, threads_per_block)

        best_before = ll_best
        if ll_new > ll_best:
            ll_best = ll_new
            Q_best.copy_(Q)
        if best_before != -float("inf") and ll_new <= best_before + tol:
            wait += 1
        else:
            wait = 0

        log.info(
            f"    Iteration {it}, "
            f"Log-likelihood: {ll_new:.1f}, "
            f"Time: {time.time() - it_start:.3f}s"
        )

        diff = ll_new - ll_prev_iter
        if abs(diff) < tol:
            log.info(f"    Converged at iteration {it}.")
            break
        if wait >= patience:
            log.info(f"    Converged at iteration {it} after {wait} iterations without improvement.")
            break

        ll_prev_iter = ll_new

    log.info(f"\n    Final log-likelihood (projection): {ll_best:.1f}")
    return Q_best
