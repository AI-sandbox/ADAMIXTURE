import argparse
import logging
import sys

import numpy as np

from ..entry import _fix_macos_libomp

_fix_macos_libomp()

from ..src import utils  # noqa: E402
from ..src.utils_c import tools  # noqa: E402
from .stability import _load_matrix_2d, _optimal_column_permutation, _orient_and_normalize_q  # noqa: E402

# Global logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("AncestryEval")


def set_numerical_backends(n_threads: int) -> None:
    """Restrict thread usage for OpenMP/BLAS libraries to avoid over-subscription."""
    utils.set_cpu_threads(n_threads)


def parse_input(argv: list[str] | None = None) -> argparse.Namespace:
    """Configure and run the CLI argument parser for the evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluation metrics for ancestry inference.")

    # Input Data Group
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--qfile", required=True, help="Estimated ancestry proportions")
    io_group.add_argument("--data_path", help="PLINK/VCF/PGEN genotype dataset path or prefix")
    io_group.add_argument("--pfile", help="Allele frequency file")
    io_group.add_argument("--tfile", help="Ground truth proportions (for validation)")

    # Configuration Group
    cfg_group = parser.add_argument_group("Configuration")
    cfg_group.add_argument("--threads", type=int, default=1, help="CPU cores to use")
    cfg_group.add_argument("--bound", type=float, default=1e-5, help="Numerical stability clip")
    cfg_group.add_argument("--inverse", action="store_true", help="Invert frequency coding (1-P)")
    cfg_group.add_argument(
        "--no_auto_transpose",
        action="store_true",
        help="Do not automatically transpose Q matrices when columns exceed rows (e.g. N < K).",
    )
    cfg_group.add_argument(
        "--chrom_mode",
        choices=["all", "autosomes"],
        default="all",
        help="Chromosome filter for input variants (default: all).",
    )
    cfg_group.add_argument(
        "--autosomes",
        type=int,
        default=22,
        help="Number of autosomes kept when --chrom_mode=autosomes",
    )
    cfg_group.add_argument(
        "--specific_chrom",
        nargs="+",
        help="List of specific chromosomes to analyze when --chrom_mode=autosomes (overrides --autosomes)",
    )

    # Metrics Group
    metric_group = parser.add_argument_group("Metrics")
    metric_group.add_argument("--loglike", action="store_true", help="Calculate model Log-Likelihood")
    metric_group.add_argument("--rmse", action="store_true", help="Calculate RMSE vs Truth")
    metric_group.add_argument("--jsd", action="store_true", help="Calculate JSD vs Truth")

    args = parser.parse_args(argv)

    if not (args.loglike or args.rmse or args.jsd):
        parser.error("At least one metric (--loglike, --rmse, --jsd) must be specified.")
    if args.threads < 1:
        parser.error("--threads must be at least 1.")
    if not (0.0 <= args.bound < 0.5):
        parser.error("--bound must satisfy 0.0 <= bound < 0.5.")

    is_validation = args.rmse or args.jsd
    if is_validation and not args.tfile:
        parser.error("Validation metrics (--rmse, --jsd) require ground truth file (--tfile).")
    if args.loglike and (not args.data_path or not args.pfile):
        parser.error("Log-likelihood (--loglike) requires input data (--data_path) and frequencies (--pfile).")
    if args.chrom_mode == "autosomes" and not args.specific_chrom and args.autosomes < 1:
        parser.error("--autosomes must be at least 1.")

    return args


def load_proportions(
    filepath: str,
    bound: float = 0.0,
    expected_samples: int | None = None,
    transpose_if_wide: bool = True,
) -> np.ndarray:
    """
    Load ancestry proportions (Q matrix) from a text or .npy file, orient as (N x K),
    apply optional numerical bounds, and normalize rows to sum to 1.
    """
    if not (0.0 <= bound < 0.5):
        raise ValueError(f"bound must satisfy 0.0 <= bound < 0.5, got {bound}.")
    mat = _load_matrix_2d(filepath)
    return _orient_and_normalize_q(
        mat,
        bound=bound,
        expected_samples=expected_samples,
        transpose_if_wide=transpose_if_wide,
    )


def align_latent_factors(
    est_props: np.ndarray,
    true_props: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve label switching by matching estimated columns to true columns using exact
    linear sum assignment (Jonker-Volgenant) minimizing squared Euclidean distance.
    """
    est = np.ascontiguousarray(est_props, dtype=np.float64)
    true = np.ascontiguousarray(true_props, dtype=np.float64)
    perm = _optimal_column_permutation(true, est)
    return np.ascontiguousarray(est[:, perm], dtype=np.float64), true


def run_validation(args: argparse.Namespace, est_props: np.ndarray) -> dict[str, float]:
    """Calculate validation metrics (RMSE and/or JSD) against a ground truth Q matrix."""
    est = np.ascontiguousarray(est_props, dtype=np.float64)
    true_props = load_proportions(
        args.tfile,
        bound=0.0,
        expected_samples=est.shape[0],
        transpose_if_wide=not getattr(args, "no_auto_transpose", False),
    )
    if est.shape != true_props.shape:
        raise ValueError(
            f"Shape mismatch between Q-file {est.shape} and truth file {true_props.shape}."
        )

    aligned_est, aligned_true = align_latent_factors(est, true_props)
    N, K = aligned_true.shape
    results: dict[str, float] = {}

    if args.rmse:
        val = float(tools.rmse_d(aligned_est, aligned_true, N, K))
        results["rmse"] = val
        log.info(f"RMSE: {val:.7f}")

    if args.jsd:
        div_a = tools.KL(aligned_est, aligned_true, N, K)
        div_b = tools.KL(aligned_true, aligned_est, N, K)
        jsd = float(0.5 * (div_a + div_b))
        results["jsd"] = jsd
        log.info(f"JSD: {jsd:.7f}")

    return results


def run_fitting_eval(args: argparse.Namespace, est_props: np.ndarray) -> float:
    """Calculate the model log-likelihood using 2-bit packed genotypes and allele frequencies."""
    genotypes, N, M = utils.read_data(
        args.data_path,
        packed="cpu",
        chunk_size=4096,
        chrom_mode=args.chrom_mode,
        autosomes=args.autosomes,
        specific_chrom=getattr(args, "specific_chrom", None),
    )

    est_ll = np.array(est_props, dtype=np.float64, copy=True)
    if est_ll.shape[0] != N and est_ll.shape[1] == N:
        est_ll = np.ascontiguousarray(est_ll.T)
    if est_ll.shape[0] != N:
        raise ValueError(
            f"Sample count mismatch: Q-file has {est_ll.shape[0]} samples (shape {est_props.shape}), "
            f"but genotype data has {N} samples."
        )
    K = est_ll.shape[1]

    freqs = _load_matrix_2d(args.pfile)
    if freqs.shape != (M, K) and freqs.shape == (K, M):
        freqs = np.ascontiguousarray(freqs.T)
    if freqs.shape != (M, K):
        raise ValueError(f"Mismatch: P-file {freqs.shape} vs Data {(M, K)}")

    if args.inverse:
        freqs = 1.0 - freqs

    if args.bound > 0.0:
        np.clip(freqs, args.bound, 1.0 - args.bound, out=freqs)
        np.clip(est_ll, args.bound, 1.0 - args.bound, out=est_ll)
        row_sums = np.sum(est_ll, axis=1, keepdims=True)
        row_sums[row_sums <= 0.0] = 1.0
        est_ll /= row_sums
    else:
        np.clip(freqs, 0.0, 1.0, out=freqs)

    freqs = np.ascontiguousarray(freqs, dtype=np.float64)
    est_ll = np.ascontiguousarray(est_ll, dtype=np.float64)

    ll = float(tools.loglikelihood_packed(genotypes.ctypes.data, M, N, freqs, est_ll))
    log.info(f"Log-likelihood: {round(ll, 1)}")
    return ll


def main(argv: list[str] | None = None) -> None:
    """Main CLI execution flow for the evaluation module."""
    args = parse_input(argv)
    set_numerical_backends(args.threads)
    transpose_if_wide = not args.no_auto_transpose

    try:
        if args.rmse or args.jsd:
            est_val = load_proportions(args.qfile, bound=0.0, transpose_if_wide=transpose_if_wide)
            run_validation(args, est_val)

        if args.loglike:
            est_ll = load_proportions(args.qfile, bound=args.bound, transpose_if_wide=transpose_if_wide)
            run_fitting_eval(args, est_ll)
    except (ValueError, OSError) as exc:
        log.error(f"Evaluation error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
