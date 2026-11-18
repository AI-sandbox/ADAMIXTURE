import argparse
import itertools
import logging
import sys

import numpy as np

from ..entry import _fix_macos_libomp

_fix_macos_libomp()

from ..src.utils_c.cython.clumppling import solve_lap  # noqa: E402

# Global logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("ClusterStability")


def _load_matrix_2d(filepath: str) -> np.ndarray:
    """Load a 2D (or 1D K=1) finite float64 matrix from a .npy or text file."""
    if filepath.endswith(".npy"):
        mat = np.array(np.load(filepath), dtype=np.float64, copy=True)
    else:
        mat = np.array(np.loadtxt(filepath, dtype=np.float64), dtype=np.float64, copy=True)

    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    if mat.ndim != 2 or mat.size == 0:
        raise ValueError(f"Expected a non-empty 2D matrix in '{filepath}', got shape {mat.shape}.")
    if not np.all(np.isfinite(mat)):
        raise ValueError(f"Matrix in '{filepath}' contains non-finite values.")
    return mat


def _orient_and_normalize_q(
    mat: np.ndarray,
    bound: float = 0.0,
    expected_samples: int | None = None,
    transpose_if_wide: bool = True,
) -> np.ndarray:
    """Orient an ancestry proportion matrix as (N, K), clip if bounded, and row-normalize."""
    mat = np.array(mat, dtype=np.float64, copy=True)
    if expected_samples is not None:
        if mat.shape[0] != expected_samples and mat.shape[1] == expected_samples:
            mat = np.ascontiguousarray(mat.T)
    elif transpose_if_wide and mat.shape[1] > mat.shape[0]:
        mat = np.ascontiguousarray(mat.T)

    if bound > 0.0:
        np.clip(mat, bound, 1.0 - bound, out=mat)
    else:
        np.clip(mat, 0.0, 1.0, out=mat)

    row_sums = np.sum(mat, axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    mat /= row_sums
    return np.ascontiguousarray(mat, dtype=np.float64)


def _optimal_column_permutation(ref_props: np.ndarray, query_props: np.ndarray) -> np.ndarray:
    """Compute exact minimum squared-Euclidean column permutation aligning query_props to ref_props."""
    if ref_props.ndim != 2 or query_props.ndim != 2 or ref_props.shape != query_props.shape:
        raise ValueError(
            f"Shape mismatch for cluster alignment: {ref_props.shape} vs {query_props.shape}."
        )
    K = ref_props.shape[1]
    cost_matrix = np.empty((K, K), dtype=np.float64)
    for i in range(K):
        diff = query_props - ref_props[:, i : i + 1]
        cost_matrix[i] = np.einsum("ij,ij->j", diff, diff)
    return np.asarray(solve_lap(np.ascontiguousarray(cost_matrix, dtype=np.float64)), dtype=np.intp)


def load_proportions(filepath: str, transpose_if_wide: bool = True) -> np.ndarray:
    """
    Load ancestry proportions (Q matrix) from a text or .npy file, orient to
    (samples x K), clip negatives, and normalize rows to sum to 1.
    """
    mat = _load_matrix_2d(filepath)
    return _orient_and_normalize_q(mat, bound=0.0, transpose_if_wide=transpose_if_wide)


def align_matrices(Q1: np.ndarray, Q2: np.ndarray) -> np.ndarray:
    """
    Match columns of Q2 to Q1 using exact linear sum assignment (Jonker-Volgenant)
    minimizing total squared Euclidean / Frobenius distance.
    """
    q1 = np.ascontiguousarray(Q1, dtype=np.float64)
    q2 = np.ascontiguousarray(Q2, dtype=np.float64)
    perm = _optimal_column_permutation(q1, q2)
    return np.ascontiguousarray(q2[:, perm], dtype=np.float64)


def calculate_correlations(Q1: np.ndarray, Q2_aligned: np.ndarray) -> float:
    """
    Calculate mean Pearson correlation across aligned cluster columns,
    safely handling collapsed/constant columns without producing NaN.
    """
    K = Q1.shape[1]
    corrs = np.empty(K, dtype=np.float64)
    for k in range(K):
        col1 = Q1[:, k]
        col2 = Q2_aligned[:, k]
        c1_const = float(np.ptp(col1)) <= 1e-7
        c2_const = float(np.ptp(col2)) <= 1e-7
        if c1_const and c2_const:
            corrs[k] = 1.0 if np.allclose(col1, col2, atol=1e-5, rtol=0.0) else 0.0
        elif c1_const or c2_const:
            corrs[k] = 0.0
        else:
            x = col1 - np.mean(col1)
            y = col2 - np.mean(col2)
            nx = float(np.linalg.norm(x))
            ny = float(np.linalg.norm(y))
            corrs[k] = float(np.clip(np.dot(x, y) / (nx * ny), -1.0, 1.0)) if nx > 0.0 and ny > 0.0 else 0.0
    return float(np.mean(corrs))


def calculate_frobenius(Q1: np.ndarray, Q2_aligned: np.ndarray) -> float:
    """Calculate the Frobenius distance between two aligned Q matrices."""
    return float(np.sqrt(np.sum((Q1 - Q2_aligned) ** 2)))


def calculate_gprime(Q1: np.ndarray, Q2_aligned: np.ndarray) -> float:
    """Calculate the Clumppling G' similarity score between two aligned Q matrices."""
    cost = float(np.sum((Q1 - Q2_aligned) ** 2)) / (2.0 * Q1.shape[0])
    return float(1.0 - np.sqrt(min(1.0, cost)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for cluster stability evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate cluster stability across multiple runs (seeds).")
    parser.add_argument(
        "qfiles",
        nargs="+",
        help="Space-separated list of Q matrix files (e.g., *.Q or matrix1.npy matrix2.npy)",
    )
    parser.add_argument(
        "--no_auto_transpose",
        action="store_true",
        help="Do not automatically transpose Q matrices when columns exceed rows (e.g. N < K).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for cluster stability evaluation across Q matrix files."""
    args = parse_args(argv)

    if len(args.qfiles) < 2:
        log.error("Error: At least two Q files are required to calculate stability.")
        sys.exit(1)

    log.info(f"Loading {len(args.qfiles)} Q matrices...")

    matrices: list[np.ndarray] = []
    try:
        for f in args.qfiles:
            matrices.append(load_proportions(f, transpose_if_wide=not args.no_auto_transpose))
    except (ValueError, OSError) as exc:
        log.error(f"Error loading Q matrix: {exc}")
        sys.exit(1)

    reference_shape = matrices[0].shape
    for i, mat in enumerate(matrices):
        if mat.shape != reference_shape:
            log.error(
                f"Shape mismatch: '{args.qfiles[i]}' has shape {mat.shape}, "
                f"but '{args.qfiles[0]}' has shape {reference_shape}."
            )
            sys.exit(1)

    num_runs = len(matrices)
    log.info(f"Shape verified: {reference_shape[0]} samples, {reference_shape[1]} clusters (K).")
    log.info("-" * 50)

    all_corrs: list[float] = []
    all_frobs: list[float] = []
    all_gprimes: list[float] = []

    pairs = list(itertools.combinations(range(num_runs), 2))
    log.info(f"Performing {len(pairs)} pairwise comparisons...")

    for i, j in pairs:
        Q_ref = matrices[i]
        Q_target = matrices[j]

        Q_target_aligned = align_matrices(Q_ref, Q_target)

        corr = calculate_correlations(Q_ref, Q_target_aligned)
        frob = calculate_frobenius(Q_ref, Q_target_aligned)
        gprime = calculate_gprime(Q_ref, Q_target_aligned)

        all_corrs.append(corr)
        all_frobs.append(frob)
        all_gprimes.append(gprime)

        log.info(
            f"Pair ({args.qfiles[i]}, {args.qfiles[j]}): "
            f"Mean Corr = {corr:.6f}, Frobenius Dist = {frob:.6f}, G' = {gprime:.6f}"
        )

    log.info("-" * 50)
    log.info("FINAL SUMMARY:")
    log.info(f"Average Pairwise Correlation: {np.mean(all_corrs):.6f} (± {np.std(all_corrs):.6f})")
    log.info(f"Average Frobenius Distance:     {np.mean(all_frobs):.6f} (± {np.std(all_frobs):.6f})")
    log.info(f"Average Pairwise G':            {np.mean(all_gprimes):.6f} (± {np.std(all_gprimes):.6f})")
    log.info("-" * 50)


if __name__ == "__main__":
    main()
