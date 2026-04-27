import logging
import sys
from argparse import Namespace

import numpy as np

from ..model.em_adam import adamStep

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _build_folds(non_missing_flat: np.ndarray, n_folds: int, seed: int) -> list[np.ndarray]:
    """
    Description:
    Partitions non-missing genotype flat indices into v roughly equal folds
    using a random shuffle.

    Args:
        non_missing_flat (np.ndarray): 1-D array of flat indices where genotype != 3.
        n_folds (int): Number of cross-validation folds.
        seed (int): Random seed for reproducibility.

    Returns:
        list[np.ndarray]: List of v arrays, each containing the flat indices for one fold.
    """
    rng = np.random.default_rng(seed)
    shuffled = non_missing_flat.copy()
    rng.shuffle(shuffled)

    fold_ids = np.arange(shuffled.size, dtype=np.int32) % n_folds
    folds = [shuffled[fold_ids == fold] for fold in range(n_folds)]
    return folds


def _polish_fold(
    G: np.ndarray,
    P_init: np.ndarray,
    Q_init: np.ndarray,
    args: Namespace,
    M: int,
    N: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Runs a fixed number of Adam-EM polishing iterations on CPU,
    warm-started from the global P and Q estimates.

    Args:
        G (np.ndarray): Genotype matrix with held-out entries masked as 3.
        P_init (np.ndarray): Global P matrix (M x K) used as warm-start.
        Q_init (np.ndarray): Global Q matrix (N x K) used as warm-start.
        args (Namespace): Parsed command-line arguments (lr, beta1, beta2, reg_adam, cv_max_iter).
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.

    Returns:
        tuple[np.ndarray, np.ndarray]: Polished (P, Q) matrices after cv_max_iter iterations.
    """
    P = np.array(P_init, dtype=np.float64, copy=True)
    Q = np.array(Q_init, dtype=np.float64, copy=True)

    m_P = np.zeros_like(P, dtype=np.float64)
    v_P = np.zeros_like(P, dtype=np.float64)
    m_Q = np.zeros_like(Q, dtype=np.float64)
    v_Q = np.zeros_like(Q, dtype=np.float64)
    t = [0]

    P1 = np.zeros_like(P, dtype=np.float64)
    Q1 = np.zeros_like(Q, dtype=np.float64)
    T = np.zeros_like(Q, dtype=np.float64)
    q_bat = np.zeros(N, dtype=np.float64)

    for _ in range(int(args.cv_max_iter)):
        adamStep(
            G, P, Q, T, P1, Q1, q_bat, K, M, N,
            m_P, v_P, m_Q, v_Q, t,
            float(args.lr), float(args.beta1),
            float(args.beta2), float(args.reg_adam),
        )

    return P, Q


def _deviance_squared_sum_for_entries(
    flat_entries: np.ndarray,
    G_flat: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    N: int,
    score_chunk_size: int = 1_000_000,
) -> float:
    """
    Description:
    Computes the sum of squared binomial deviance residuals on held-out entries.
    Processes entries in chunks to avoid large intermediate allocations.

    Args:
        flat_entries (np.ndarray): 1-D array of flat indices of held-out genotype entries.
        G_flat (np.ndarray): Flattened genotype matrix with true (restored) values.
        P (np.ndarray): Polished P matrix (M x K).
        Q (np.ndarray): Polished Q matrix (N x K).
        N (int): Number of individuals (used to map flat index to row/col).
        score_chunk_size (int): Number of entries to process per chunk.

    Returns:
        float: Sum of squared deviance residuals over all held-out entries.
    """
    total = 0.0
    two = 2.0
    eps = 1e-10

    for start in range(0, flat_entries.size, score_chunk_size):
        stop = min(start + score_chunk_size, flat_entries.size)
        idx = flat_entries[start:stop]

        rows = idx // N
        cols = idx % N
        g = G_flat[idx].astype(np.float64, copy=False)

        mu = two * np.sum(Q[cols] * P[rows], axis=1)
        np.clip(mu, eps, two - eps, out=mu)

        term_a = np.zeros_like(g, dtype=np.float64)
        mask_a = g > 0.0
        term_a[mask_a] = g[mask_a] * np.log(g[mask_a] / mu[mask_a])

        rem = two - g
        term_b = np.zeros_like(g, dtype=np.float64)
        mask_b = rem > 0.0
        term_b[mask_b] = rem[mask_b] * np.log(rem[mask_b] / (two - mu[mask_b]))

        dev = term_a + term_b
        total += float(np.sum(dev * dev))

    return total


def run_cross_validation(
    args: Namespace,
    G: np.ndarray,
    N: int,
    M: int,
    K: int,
    P_global: np.ndarray,
    Q_global: np.ndarray,
) -> float:
    """
    Description:
    Performs v-fold cross-validation on genotype entries. Each fold masks a random
    subset of non-missing entries, polishes P and Q from the global warm-start,
    and scores via squared binomial deviance residuals. Always runs on CPU.

    Args:
        args (Namespace): Parsed command-line arguments (cv, cv_max_iter, seed, lr, etc.).
        G (np.ndarray): Unpacked genotype matrix (M x N, uint8). Modified in-place per fold
                        and restored before scoring.
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.
        P_global (np.ndarray): Global P matrix (M x K) from the main fit.
        Q_global (np.ndarray): Global Q matrix (N x K) from the main fit.

    Returns:
        float: CV index (average squared deviance residual across all held-out entries).
    """
    if G.shape != (M, N):
        raise ValueError(
            f"CV requires an unpacked genotype matrix with shape {(M, N)}, got {G.shape}."
        )

    G_flat = G.reshape(-1)
    non_missing_flat = np.flatnonzero(G_flat != 3)
    n_non_missing = int(non_missing_flat.size)
    if n_non_missing == 0:
        raise ValueError("No non-missing genotypes available for cross-validation.")

    n_folds = int(args.cv)
    folds = _build_folds(non_missing_flat, n_folds, int(args.seed))

    cv_sum = 0.0
    for fold_idx, held_out_entries in enumerate(folds, start=1):
        if held_out_entries.size == 0:
            continue

        saved_values = G_flat[held_out_entries].copy()
        G_flat[held_out_entries] = 3

        P_cv, Q_cv = _polish_fold(G, P_global, Q_global, args, M, N, K)

        G_flat[held_out_entries] = saved_values

        fold_sum = _deviance_squared_sum_for_entries(
            held_out_entries, G_flat, P_cv, Q_cv, N
        )
        cv_sum += fold_sum

    cv_index = cv_sum / float(n_non_missing)
    log.info(f"    CV index for K={K}: {cv_index:.4f}")
    return cv_index
