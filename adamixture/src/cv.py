import gc
import logging
import sys
from argparse import Namespace

import numpy as np
import torch

from ..model.br_qn import polish_sqp_qn
from .utils_c import (
    deviance_squared_sum,
    deviance_squared_sum_i32,
    mask_entries_i32,
    mask_entries_i64,
    restore_entries_i32,
    restore_entries_i64,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

_STREAMING_CV_TARGET_ENTRIES = 10_000_000
_HASH_CONST_1 = np.uint64(0x9E3779B97F4A7C15)
_HASH_CONST_2 = np.uint64(0xBF58476D1CE4E5B9)
_HASH_CONST_3 = np.uint64(0x94D049BB133111EB)


def _shuffle_non_missing(non_missing_flat: np.ndarray, seed: int) -> np.ndarray:
    """
    Description:
    Returns a shuffled copy of non-missing genotype flat indices.

    Args:
        non_missing_flat (np.ndarray): 1-D array of flat indices where genotype != 3.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Shuffled flat indices.
    """
    rng = np.random.default_rng(seed)
    shuffled = non_missing_flat.copy()
    rng.shuffle(shuffled)
    return shuffled

def _streaming_rows_per_chunk(N: int) -> int:
    """
    Description:
    Returns the number of genotype rows to scan per streaming CV chunk.

    Args:
        N (int): Number of individuals.

    Returns:
        int: Number of SNP rows per chunk.
    """
    return max(1, _STREAMING_CV_TARGET_ENTRIES // N)

def _hash_fold_mask(flat_entries: np.ndarray, seed: int, n_folds: int, fold: int) -> np.ndarray:
    """
    Description:
    Returns a boolean mask selecting entries assigned to one CV fold by a deterministic hash.

    Args:
        flat_entries (np.ndarray): 1-D array of flat genotype indices.
        seed (int): Random seed used to change the deterministic fold assignment.
        n_folds (int): Number of cross-validation folds.
        fold (int): Fold index to select.

    Returns:
        np.ndarray: Boolean mask indicating which entries belong to the requested fold.
    """
    hashed = flat_entries.astype(np.uint64, copy=False) + np.uint64(seed) + _HASH_CONST_1
    hashed ^= hashed >> np.uint64(30)
    hashed *= _HASH_CONST_2
    hashed ^= hashed >> np.uint64(27)
    hashed *= _HASH_CONST_3
    hashed ^= hashed >> np.uint64(31)
    return (hashed % np.uint64(n_folds)) == np.uint64(fold)

def _iter_non_missing_flat_chunks(G: np.ndarray, N: int):
    """
    Description:
    Yields chunks of non-missing genotype flat indices without materializing all entries.

    Args:
        G (np.ndarray): Unpacked genotype matrix (M x N, uint8).
        N (int): Number of individuals.

    Returns:
        Iterator[np.ndarray]: Chunks of flat indices where genotype != 3.
    """
    rows_per_chunk = _streaming_rows_per_chunk(N)
    for row_start in range(0, G.shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, G.shape[0])
        local = np.flatnonzero(G[row_start:row_end].ravel() != 3)
        if local.size:
            yield local.astype(np.int64, copy=False) + row_start * N

def _count_non_missing_streaming(G: np.ndarray, N: int) -> int:
    """
    Description:
    Counts non-missing genotype entries by scanning the matrix in chunks.

    Args:
        G (np.ndarray): Unpacked genotype matrix (M x N, uint8).
        N (int): Number of individuals.

    Returns:
        int: Number of genotype entries where genotype != 3.
    """
    rows_per_chunk = _streaming_rows_per_chunk(N)
    total = 0
    for row_start in range(0, G.shape[0], rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, G.shape[0])
        total += int(np.count_nonzero(G[row_start:row_end] != 3))
    return total

def _build_hashed_fold_entries(G: np.ndarray, N: int, n_folds: int, fold: int, seed: int) -> np.ndarray:
    """
    Description:
    Builds the held-out flat indices for one CV fold using streaming deterministic hashing.

    Args:
        G (np.ndarray): Unpacked genotype matrix (M x N, uint8).
        N (int): Number of individuals.
        n_folds (int): Number of cross-validation folds.
        fold (int): Fold index to build.
        seed (int): Random seed used to change the deterministic fold assignment.

    Returns:
        np.ndarray: Flat genotype indices assigned to the requested fold.
    """
    n_fold_entries = 0
    for flat_entries in _iter_non_missing_flat_chunks(G, N):
        n_fold_entries += int(np.count_nonzero(_hash_fold_mask(flat_entries, seed, n_folds, fold)))

    held_out_entries = np.empty(n_fold_entries, dtype=np.int64)
    offset = 0
    for flat_entries in _iter_non_missing_flat_chunks(G, N):
        selected = flat_entries[_hash_fold_mask(flat_entries, seed, n_folds, fold)]
        next_offset = offset + selected.size
        held_out_entries[offset:next_offset] = selected
        offset = next_offset

    return held_out_entries

def _polish_fold(G: np.ndarray, P_init: np.ndarray, Q_init: np.ndarray,
                M: int, N: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Runs three SQP + ZAL quasi-Newton polishing iterations on CPU,
    warm-started from the global P and Q estimates.

    Args:
        G (np.ndarray): Genotype matrix with held-out entries masked as 3.
        P_init (np.ndarray): Global P matrix (M x K) used as warm-start.
        Q_init (np.ndarray): Global Q matrix (N x K) used as warm-start.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.

    Returns:
        tuple[np.ndarray, np.ndarray]: Polished (P, Q) matrices after 3 BR-QN iterations.
    """
    return polish_sqp_qn(
        G, P_init, Q_init, M, N, K,
        n_iters=3,
        Q_hist=3,
    )

def run_cross_validation(args: Namespace, G: np.ndarray, N: int, M: int, K: int,
                        P_global: np.ndarray | torch.Tensor, Q_global: np.ndarray | torch.Tensor) -> float:
    """
    Description:
    Performs v-fold cross-validation on genotype entries. Each fold masks a random
    subset of non-missing entries, polishes P and Q from the global warm-start,
    and scores via squared binomial deviance residuals. Always runs on CPU.

    ``P_global`` / ``Q_global`` may be ``torch.Tensor`` (e.g. MPS/CUDA fits); they are
    converted to NumPy on CPU inside this function.

    Args:
        args (Namespace): Parsed command-line arguments (cv, seed, lr, etc.).
        G (np.ndarray): Unpacked genotype matrix (M x N, uint8).
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.
        P_global (ndarray | Tensor): Global P matrix (M x K) from the main fit.
        Q_global (ndarray | Tensor): Global Q matrix (N x K) from the main fit.

    Returns:
        float: CV index (average squared deviance residual across all held-out entries).
    """
    if isinstance(P_global, torch.Tensor):
        P_global = P_global.detach().cpu().numpy()
    if isinstance(Q_global, torch.Tensor):
        Q_global = Q_global.detach().cpu().numpy()
    if G.shape != (M, N):
        raise ValueError(f"CV requires an unpacked genotype matrix with shape {(M, N)}, got {G.shape}.")

    n_folds = int(args.cv)
    use_streaming_folds = M * N > np.iinfo(np.int32).max
    if use_streaming_folds:
        n_non_missing = _count_non_missing_streaming(G, N)
        shuffled = None
    else:
        non_missing_flat = np.flatnonzero(G != 3)
        n_non_missing = int(non_missing_flat.size)
        non_missing_flat = non_missing_flat.astype(np.int32, copy=False)
        shuffled = _shuffle_non_missing(non_missing_flat, int(args.seed))
        del non_missing_flat
        gc.collect()

    if n_non_missing == 0:
        raise ValueError("No non-missing genotypes available for cross-validation.")

    cv_sum = 0.0
    for fold in range(n_folds):
        if use_streaming_folds:
            held_out_entries = _build_hashed_fold_entries(G, N, n_folds, fold, int(args.seed))
        else:
            held_out_entries = shuffled[fold::n_folds]
        if held_out_entries.size == 0:
            if use_streaming_folds:
                del held_out_entries
            continue

        if held_out_entries.dtype == np.int32:
            saved_values = np.empty(held_out_entries.size, dtype=np.uint8)
            mask_entries_i32(G, held_out_entries, saved_values, N)
            try:
                P_cv, Q_cv = _polish_fold(G, P_global, Q_global, M, N, K)
            finally:
                restore_entries_i32(G, held_out_entries, saved_values, N)

            fold_sum = deviance_squared_sum_i32(
                saved_values,
                held_out_entries,
                np.ascontiguousarray(P_cv, dtype=np.float64),
                np.ascontiguousarray(Q_cv, dtype=np.float64),
                N,
            )
            cv_sum += fold_sum
            del saved_values, P_cv, Q_cv
        else:
            saved_values = np.empty(held_out_entries.size, dtype=np.uint8)
            mask_entries_i64(G, held_out_entries, saved_values, N)
            try:
                P_cv, Q_cv = _polish_fold(G, P_global, Q_global, M, N, K)
            finally:
                restore_entries_i64(G, held_out_entries, saved_values, N)

            fold_sum = deviance_squared_sum(
                saved_values,
                held_out_entries,
                np.ascontiguousarray(P_cv, dtype=np.float64),
                np.ascontiguousarray(Q_cv, dtype=np.float64),
                N,
            )
            cv_sum += fold_sum
            del saved_values, P_cv, Q_cv
        if use_streaming_folds:
            del held_out_entries
        gc.collect()

    cv_index = cv_sum / float(n_non_missing)
    return cv_index
