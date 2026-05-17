import logging
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from . import utils
from .utils_c import em, tools

if TYPE_CHECKING:
    import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


# ── Shared initialisation helpers (CPU/GPU agnostic) ─────────────────────────

def init_q_supervised(Q: np.ndarray, y: np.ndarray, K: int, eps: float = 1e-5) -> None:
    """
    Description:
    Initialises Q for labeled samples.  Labeled individuals (y > 0) get a
    near-one-hot Q row corresponding to their known ancestry
    (1 − eps on the assigned component, eps/(K−1) on all others).
    Unlabeled samples (y == 0) keep their random initialisation, which is
    then row-normalised.

    Args:
        Q (np.ndarray): Q matrix to initialise in-place (N x K).
        y (np.ndarray): Population assignment vector (N,), dtype int.
                        0 = unlabeled, 1..K = labeled population.
        K (int): Number of ancestral populations.
        eps (float): Small value for numerical stability.

    Returns:
        None: Q is updated in-place.
    """
    fill_minor = eps / max(K - 1, 1)
    for i, pop in enumerate(y):
        if pop > 0:
            Q[i, :] = fill_minor
            Q[i, pop - 1] = 1.0 - eps
    unlabeled = y == 0
    row_sums = Q[unlabeled].sum(axis=1, keepdims=True)
    Q[unlabeled] /= row_sums


def init_p_supervised(G: np.ndarray, y: np.ndarray, K: int, M: int, eps: float = 1e-5) -> np.ndarray:
    """
    Description:
    Initialises P from the labeled samples' genotype frequencies.
    For each population k (1..K), allele frequencies are computed from
    genotypes of samples labeled k.  Populations with no labeled samples
    receive uniform frequencies (0.5).

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).  Missing coded as 3.
        y (np.ndarray): Population assignment vector (N,), dtype int.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        eps (float): Clipping bound for numerical stability.

    Returns:
        np.ndarray: Initialized P matrix (M x K, float64), clipped to [eps, 1−eps].
    """
    P = np.full((M, K), 0.5, dtype=np.float64)

    for k in range(K):
        idx = np.where(y == k + 1)[0]
        if len(idx) == 0:
            continue
        G_k = G[:, idx].astype(np.float64)
        missing = G_k == 3
        G_k[missing] = 0.0
        counts = G_k.sum(axis=1)
        denom = 2.0 * (G_k.shape[1] - missing.sum(axis=1))
        valid = denom > 0
        P[valid, k] = counts[valid] / denom[valid]

    return P.clip(eps, 1.0 - eps)


# ── CPU (numpy) implementation ────────────────────────────────────────────────

def _snap_q_cpu(Q: np.ndarray, y: np.ndarray, K: int, eps: float = 1e-5) -> None:
    """
    Description:
    Snaps the Q rows of labeled samples back to their known near-one-hot
    ancestry after each Adam-EM update.  Unlabeled rows are left untouched.

    Args:
        Q (np.ndarray): Q matrix updated in-place (N x K).
        y (np.ndarray): Population assignment vector (N,), int.
        K (int): Number of ancestral populations.
        eps (float): Small value used to anchor labeled rows.

    Returns:
        None: Q is updated in-place.
    """
    fill_minor = eps / max(K - 1, 1)
    for i, pop in enumerate(y):
        if pop > 0:
            Q[i, :] = fill_minor
            Q[i, pop - 1] = 1.0 - eps


def _supervised_adam_step_cpu(G: np.ndarray, P0: np.ndarray, Q0: np.ndarray, T: np.ndarray,
                            P1: np.ndarray, Q1: np.ndarray, q_bat: np.ndarray, K: int, M: int, N: int,
                            m_P: np.ndarray, v_P: np.ndarray, m_Q: np.ndarray, v_Q: np.ndarray,
                            t: list, lr: float, beta1: float, beta2: float, epsilon: float,
                            y: np.ndarray) -> None:
    """
    Description:
    Single Adam-EM step in supervised mode: updates both P and Q, then
    snaps labeled samples' Q rows back to their known one-hot ancestry.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P0 (np.ndarray): Current P matrix (M x K). Updated via Adam.
        Q0 (np.ndarray): Current Q matrix (N x K). Updated via Adam + snap.
        T (np.ndarray): Temporary accumulator for Q terms (N x K).
        P1 (np.ndarray): Buffer for EM-updated P (M x K).
        Q1 (np.ndarray): Buffer for EM-updated Q (N x K).
        q_bat (np.ndarray): Per-sample genotype-count accumulator (N,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of samples.
        m_P (np.ndarray): First Adam moment for P.
        v_P (np.ndarray): Second Adam moment for P.
        m_Q (np.ndarray): First Adam moment for Q.
        v_Q (np.ndarray): Second Adam moment for Q.
        t (list): Single-element list holding the Adam time-step counter.
        lr (float): Learning rate.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        epsilon (float): Adam epsilon.
        y (np.ndarray): Population assignment vector (N,), int.

    Returns:
        None: P0 and Q0 are updated in-place.
    """
    em.P_step(G, P0, P1, Q0, T, q_bat, K, M, N)
    em.Q_step(Q0, Q1, T, q_bat, K, N)

    t_val = t[0] + 1
    em.adamUpdateP(P0, P1, m_P, v_P, lr, beta1, beta2, epsilon, t_val, M, K)
    em.adamUpdateQ(Q0, Q1, m_Q, v_Q, lr, beta1, beta2, epsilon, t_val, N, K)
    t[0] = t_val

    _snap_q_cpu(Q0, y, K)


def _supervised_em_step_cpu(G: np.ndarray, P0: np.ndarray, Q0: np.ndarray, T: np.ndarray,
                        P1: np.ndarray, Q1: np.ndarray, q_bat: np.ndarray,
                        K: int, M: int, N: int, y: np.ndarray) -> None:
    """
    Description:
    Plain EM step in supervised mode. Used for priming iterations before the
    main Adam-EM loop starts. Updates both P and Q, then snaps labeled rows.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P0 (np.ndarray): Current P matrix (M x K). Updated in-place.
        Q0 (np.ndarray): Current Q matrix (N x K). Updated in-place.
        T (np.ndarray): Temporary accumulator for Q terms (N x K).
        P1 (np.ndarray): Buffer for EM-updated P (M x K).
        Q1 (np.ndarray): Buffer for EM-updated Q (N x K).
        q_bat (np.ndarray): Per-sample genotype-count accumulator (N,).
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of samples.
        y (np.ndarray): Population assignment vector (N,), int.

    Returns:
        None: P0 and Q0 are updated in-place.
    """
    em.P_step(G, P0, P1, Q0, T, q_bat, K, M, N)
    em.Q_step(Q0, Q1, T, q_bat, K, N)
    P0[:] = P1
    Q0[:] = Q1
    _snap_q_cpu(Q0, y, K)


def optimize_supervised(G: np.ndarray, P: np.ndarray, Q: np.ndarray, y: np.ndarray,
    lr: float, beta1: float, beta2: float, reg_adam: float,
    max_iter: int, check: int, K: int, M: int, N: int,
    lr_decay: float, min_lr: float, patience_adam: int, tol_adam: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Optimises P and Q in supervised mode using Adam-EM on the CPU (numpy path).
    Samples with known population assignments (y > 0) have their Q rows
    snapped back to a near-one-hot encoding after every update step,
    constraining the allele-frequency learning from the labeled data.

    Args:
        G (np.ndarray): Genotype matrix (M x N, uint8).
        P (np.ndarray): Initial P matrix (M x K, float64).
        Q (np.ndarray): Initial Q matrix (N x K, float64).
        y (np.ndarray): Population assignment vector (N,), int.
                        0 = unlabeled, 1..K = labeled population.
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        reg_adam (float): Adam epsilon.
        max_iter (int): Maximum iterations.
        check (int): Log-likelihood check frequency.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of samples.
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate.
        patience_adam (int): Patience before lr decay.
        tol_adam (float): Convergence tolerance.

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimised (P, Q) matrices.
    """
    m_P = np.zeros_like(P, dtype=np.float64)
    v_P = np.zeros_like(P, dtype=np.float64)
    m_Q = np.zeros_like(Q, dtype=np.float64)
    v_Q = np.zeros_like(Q, dtype=np.float64)
    t = [0]

    P1 = np.zeros_like(P, dtype=np.float64)
    Q1 = np.zeros_like(Q, dtype=np.float64)
    T = np.zeros_like(Q, dtype=np.float64)
    q_bat = np.zeros(N, dtype=np.float64)

    P_best = np.empty_like(P)
    Q_best = np.empty_like(Q)
    L_best = float("-inf")
    wait_lr = 0

    ts = time.time()

    log.info("    Performing priming iteration...")
    ts_p = time.time()
    _supervised_em_step_cpu(G, P, Q, T, P1, Q1, q_bat, K, M, N, y)
    _supervised_adam_step_cpu(G, P, Q, T, P1, Q1, q_bat, K, M, N,
                               m_P, v_P, m_Q, v_Q, t, lr, beta1, beta2, reg_adam, y)
    _supervised_em_step_cpu(G, P, Q, T, P1, Q1, q_bat, K, M, N, y)
    log.info(f"    Priming done. ({time.time() - ts_p:.1f}s)\n")

    L_best = tools.loglikelihood(G, P, Q)
    P_best[:] = P
    Q_best[:] = Q

    for it in range(max_iter):
        _supervised_adam_step_cpu(G, P, Q, T, P1, Q1, q_bat, K, M, N,
                                  m_P, v_P, m_Q, v_Q, t, lr, beta1, beta2, reg_adam, y)
        if (it + 1) % check == 0:
            L_cur = tools.loglikelihood(G, P, Q)
            log.info(f"    Iteration {it + 1}, "
                     f"Log-likelihood: {L_cur:.1f}, "
                     f"Time: {time.time() - ts:.3f}s")
            ts = time.time()

            if L_cur > L_best + tol_adam:
                L_best = L_cur
                P_best[:] = P
                Q_best[:] = Q
                wait_lr = 0
            else:
                wait_lr += 1
                if wait_lr >= patience_adam:
                    old_lr = lr
                    lr = max(lr * lr_decay, min_lr)
                    log.info(f"    Plateau ({wait_lr} checks). "
                            f"Reducing lr: {old_lr:.3e} → {lr:.3e}")
                    if lr <= min_lr:
                        log.info("    Convergence reached.")
                        break
                    wait_lr = 0

    log.info(f"\n    Final log-likelihood (supervised): {L_best:.1f}")
    return P_best, Q_best


# ── GPU (torch) implementation ────────────────────────────────────────────────

def _snap_q_gpu(Q: "torch.Tensor", y: np.ndarray, K: int, eps: float = 1e-5) -> None:
    """
    Description:
    GPU version of the Q-snapping step. Writes near-one-hot rows for labeled
    samples directly into the Q tensor on the device.

    Args:
        Q (torch.Tensor): Q tensor (N x K) on the target device. Updated in-place.
        y (np.ndarray): Population assignment vector (N,), int (on CPU).
        K (int): Number of ancestral populations.
        eps (float): Small value for anchoring labeled rows.

    Returns:
        None: Q is updated in-place.
    """
    import torch
    fill_minor = eps / max(K - 1, 1)
    device = Q.device
    dtype = Q.dtype

    labeled_idx = np.where(y > 0)[0]
    if len(labeled_idx) == 0:
        return

    # Build replacement rows on CPU, then move to device in one go
    rows = np.full((len(labeled_idx), K), fill_minor, dtype=np.float64)
    for r, i in enumerate(labeled_idx):
        rows[r, y[i] - 1] = 1.0 - eps

    rows_t = torch.tensor(rows, dtype=dtype, device=device)
    idx_t = torch.tensor(labeled_idx, dtype=torch.long, device=device)
    Q[idx_t] = rows_t


def optimize_supervised_gpu(
    G: "torch.Tensor",
    P: "torch.Tensor",
    Q: "torch.Tensor",
    y: np.ndarray,
    lr: float, beta1: float, beta2: float, reg_adam: float,
    max_iter: int, check: int, M: int,
    lr_decay: float, min_lr: float, patience_adam: int, tol_adam: float,
    device: "torch.device", chunk_size: int, threads_per_block: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    Description:
    GPU (torch) supervised mode: optimises both P and Q via Adam-EM while
    snapping labeled samples' Q rows back to their known ancestry after
    every update.

    Args:
        G (torch.Tensor): Genotype tensor (packed or unpacked, on CPU or GPU).
        P (torch.Tensor): Initial allele-frequency tensor (M x K). Updated in-place.
        Q (torch.Tensor): Initial ancestry-proportion tensor (N x K). Updated in-place.
        y (np.ndarray): Population assignment vector (N,), int (on CPU).
                        0 = unlabeled, 1..K = labeled.
        lr (float): Adam learning rate.
        beta1 (float): Adam beta1.
        beta2 (float): Adam beta2.
        reg_adam (float): Adam epsilon.
        max_iter (int): Maximum iterations.
        check (int): Log-likelihood check frequency.
        M (int): Number of SNPs.
        lr_decay (float): Learning rate decay factor.
        min_lr (float): Minimum learning rate.
        patience_adam (int): Patience before lr decay.
        tol_adam (float): Convergence tolerance.
        device (torch.device): Computation device.
        chunk_size (int): SNP chunk size for batched EM.
        threads_per_block (int): CUDA tuning parameter.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Optimised (P, Q) tensors.
    """
    from ..model.em_adam_gpu import EMAdamOptimizer

    N = Q.shape[0]
    dtype = utils.get_dtype(device)
    P = P.to(dtype)
    Q = Q.to(dtype)

    optimizer = EMAdamOptimizer(P.shape, Q.shape, lr, beta1, beta2, reg_adam, device)
    unpacker = utils.get_unpacker(device, threads_per_block)
    logl_calc = utils.get_logl_calculator(device)

    def run_em_step() -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Description:
        Runs one full EM step over all SNP chunks, accumulating P and Q
        sufficient statistics, and returns EM-updated (P_target, Q_target).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: EM-updated (P_target, Q_target).
        """
        return optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)

    def supervised_step(P_target: "torch.Tensor", Q_target: "torch.Tensor") -> None:
        """
        Description:
        Applies one Adam update to P and Q using the EM targets, then snaps
        labeled samples' Q rows back to their known near-one-hot ancestry.

        Args:
            P_target (torch.Tensor): EM-updated P tensor used as the Adam target (M x K).
            Q_target (torch.Tensor): EM-updated Q tensor used as the Adam target (N x K).

        Returns:
            None: P and Q are updated in-place.
        """
        optimizer.step(P, Q, P_target, Q_target)
        _snap_q_gpu(Q, y, Q.shape[1])

    # Priming
    ts_p = time.time()
    P_target, Q_target = run_em_step()
    supervised_step(P_target, Q_target)
    run_em_step()
    log.info(f"    Priming done. ({time.time() - ts_p:.1f}s)\n")

    L_best = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
    P_best = P.clone()
    Q_best = Q.clone()
    wait_lr = 0
    ts = time.time()

    for it in range(max_iter):
        P_target, Q_target = run_em_step()
        supervised_step(P_target, Q_target)

        if (it + 1) % check == 0:
            L_cur = logl_calc(G, P, Q, M, N, chunk_size, threads_per_block)
            log.info(f"    Iteration {it + 1}, Log-likelihood: {L_cur:.1f}, Time: {time.time() - ts:.3f}s")
            ts = time.time()

            if L_cur > L_best + tol_adam:
                L_best = L_cur
                P_best.copy_(P)
                Q_best.copy_(Q)
                wait_lr = 0
            else:
                wait_lr += 1
                if wait_lr >= patience_adam:
                    old_lr = lr
                    lr = max(lr * lr_decay, min_lr)
                    log.info(f"    Plateau ({wait_lr} checks). Reducing lr: {old_lr:.3e} → {lr:.3e}")
                    if lr <= min_lr:
                        log.info("    Convergence reached.")
                        break
                    wait_lr = 0

    log.info(f"\n    Final log-likelihood (supervised): {L_best:.1f}")
    return P_best, Q_best
