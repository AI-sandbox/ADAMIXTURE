import argparse
import itertools
import logging
import sys

import numpy as np

# Global logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("ClusterStability")

def load_proportions(filepath: str) -> np.ndarray:
    """
    Loads ancestry proportions (Q matrix) from a file, ensures correct orientation,
    and normalizes rows to sum to 1.
    """
    try:
        if filepath.endswith(".npy"):
            mat = np.load(filepath)
        else:
            mat = np.loadtxt(filepath, dtype=float)
    except Exception as e:
        log.error(f"Error loading {filepath}: {e}")
        sys.exit(1)

    # Ensure shape is (Samples x K)
    # Common convention: Samples > K
    if mat.shape[1] > mat.shape[0]:
        mat = np.ascontiguousarray(mat.T)

    # Row-wise normalization
    row_sums = np.sum(mat, axis=1, keepdims=True)
    # To avoid division by zero
    row_sums[row_sums == 0] = 1.0
    mat /= row_sums
    return mat

def align_matrices(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """
    Matches columns of Q2 to Q1 using a greedy assignment based on
    squared Euclidean distance (minimizing Frobenius distance).
    """
    K = Q1.shape[1]
    cost_matrix = np.zeros((K, K))

    # Calculate pairwise costs (squared euclidean distance per column)
    for i in range(K):
        col1 = Q1[:, i]
        for j in range(K):
            col2 = Q2[:, j]
            diff = col1 - col2
            cost_matrix[i, j] = np.dot(diff, diff)

    # Greedy assignment
    q1_indices, q2_indices = [], []
    sorted_costs_idx = np.argsort(cost_matrix.flatten())

    for idx in sorted_costs_idx:
        r, c = np.unravel_index(idx, (K, K))
        if r not in q1_indices and c not in q2_indices:
            q1_indices.append(r)
            q2_indices.append(c)
        if len(q1_indices) == K:
            break

    # Reorder according to Q1's original order [0, 1, ..., K-1]
    # To do this, we sort pairs by q1_indices
    alignment = sorted(zip(q1_indices, q2_indices, strict=False))
    final_q2_order = [p[1] for p in alignment]

    return np.ascontiguousarray(Q2[:, final_q2_order])

def calculate_correlations(Q1: np.ndarray, Q2_aligned: np.ndarray) -> float:
    """
    Calculates the mean Pearson correlation across aligned columns.
    """
    K = Q1.shape[1]
    corrs = []
    for k in range(K):
        # Pearson correlation
        c = np.corrcoef(Q1[:, k], Q2_aligned[:, k])[0, 1]
        corrs.append(c)
    return np.mean(corrs)

def calculate_frobenius(Q1: np.ndarray, Q2_aligned: np.ndarray) -> float:
    """
    Calculates the Frobenius distance between two matrices.
    """
    return np.sqrt(np.sum((Q1 - Q2_aligned) ** 2))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate cluster stability across multiple runs (seeds).")
    parser.add_argument("qfiles", nargs="+", help="Space-separated list of Q matrix files (e.g., *.Q or matrix1.npy matrix2.npy)")
    return parser.parse_args()

def main():
    args = parse_args()

    if len(args.qfiles) < 2:
        log.error("Error: At least two Q files are required to calculate stability.")
        sys.exit(1)

    log.info(f"Loading {len(args.qfiles)} Q matrices...")

    matrices = []
    for f in args.qfiles:
        matrices.append(load_proportions(f))

    # Verify same shape
    reference_shape = matrices[0].shape
    for i, mat in enumerate(matrices):
        if mat.shape != reference_shape:
            log.error(f"Shape mismatch: '{args.qfiles[i]}' has shape {mat.shape}, "
                      f"but '{args.qfiles[0]}' has shape {reference_shape}.")
            sys.exit(1)

    num_runs = len(matrices)
    log.info(f"Shape verified: {reference_shape[0]} samples, {reference_shape[1]} clusters (K).")
    log.info("-" * 50)

    all_corrs = []
    all_frobs = []

    # Pairwise comparison
    pairs = list(itertools.combinations(range(num_runs), 2))
    log.info(f"Performing {len(pairs)} pairwise comparisons...")

    for i, j in pairs:
        Q_ref = matrices[i]
        Q_target = matrices[j]

        # Align target to ref
        Q_target_aligned = align_matrices(Q_ref, Q_target)

        corr = calculate_correlations(Q_ref, Q_target_aligned)
        frob = calculate_frobenius(Q_ref, Q_target_aligned)

        all_corrs.append(corr)
        all_frobs.append(frob)

        log.info(f"Pair ({args.qfiles[i]}, {args.qfiles[j]}): "
                 f"Mean Corr = {corr:.6f}, Frobenius Dist = {frob:.6f}")

    log.info("-" * 50)
    log.info("FINAL SUMMARY:")
    log.info(f"Average Pairwise Correlation: {np.mean(all_corrs):.6f} (± {np.std(all_corrs):.6f})")
    log.info(f"Average Frobenius Distance:     {np.mean(all_frobs):.6f} (± {np.std(all_frobs):.6f})")
    log.info("-" * 50)

if __name__ == "__main__":
    main()
