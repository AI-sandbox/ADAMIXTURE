import argparse
import os
import sys
import logging
import numpy as np

from src import utils
from src.utils_c import tools

# Global logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger("AncestryEval")

def set_numerical_backends(n_threads):
    """Restricts thread usage for OMP, MKL, and BLAS libraries."""
    n_str = str(n_threads)
    keys = [
        "MKL_NUM_THREADS", "MKL_MAX_THREADS", "OMP_NUM_THREADS", 
        "OMP_MAX_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", 
        "OPENBLAS_NUM_THREADS", "OPENBLAS_MAX_THREADS"
    ]
    for key in keys:
        os.environ[key] = n_str

def parse_input():
    parser = argparse.ArgumentParser(description="Evaluation metrics for ancestry inference.")
    
    # Input Data Group
    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument("--qfile", required=True, help="Estimated ancestry proportions")
    io_group.add_argument("--data_path", help="PLINK dataset prefix")
    io_group.add_argument("--pfile", help="Allele frequency file")
    io_group.add_argument("--tfile", help="Ground truth proportions (for validation)")
    
    # Configuration Group
    cfg_group = parser.add_argument_group("Configuration")
    cfg_group.add_argument("--threads", type=int, default=1, help="CPU cores to use")
    cfg_group.add_argument("--bound", type=float, default=1e-5, help="Numerical stability clip")
    cfg_group.add_argument("--inverse", action="store_true", help="Invert frequency coding (1-P)")

    # Metrics Group
    metric_group = parser.add_argument_group("Metrics")
    metric_group.add_argument("--loglike", action="store_true", help="Calculate model Log-Likelihood")
    metric_group.add_argument("--rmse", action="store_true", help="Calculate RMSE vs Truth")
    metric_group.add_argument("--jsd", action="store_true", help="Calculate JSD vs Truth")

    args = parser.parse_args()

    # Logic validation
    is_validation = args.rmse or args.jsd
    if is_validation and not args.tfile:
        parser.error("Validation metrics (--rmse, --jsd) require ground truth file (--tfile).")
    if not is_validation and (not args.data_path or not args.pfile):
        parser.error("Log-likelihood requires input data (--data_path) and frequencies (--pfile).")

    return args

def load_proportions(filepath, bound):
    """Loads, transposes if needed, and normalizes ancestry matrix."""
    mat = np.loadtxt(filepath, dtype=float)
    # Ensure shape is (Samples x K)
    if mat.shape[1] > mat.shape[0]:
        mat = np.ascontiguousarray(mat.T)
    
    # Numerical stability
    mat.clip(min=bound, max=1.0 - bound, out=mat)
    # Row-wise normalization
    mat /= np.sum(mat, axis=1, keepdims=True)
    return mat

def align_latent_factors(est_props, true_props):
    """
    Solves the label switching problem by matching columns 
    between estimated and true matrices via greedy minimization.
    """
    K = est_props.shape[1]
    cost_matrix = np.zeros((K, K))

    # Calculate pairwise costs (squared euclidean distance)
    for i in range(K):
        for j in range(K):
            diff = est_props[:, i] - true_props[:, j]
            cost_matrix[i, j] = np.dot(diff, diff)

    # Greedy assignment based on lowest cost
    est_indices, true_indices = [], []
    sorted_costs = np.argsort(cost_matrix.flatten())

    for idx in sorted_costs:
        r, c = np.unravel_index(idx, (K, K))
        if r not in est_indices and c not in true_indices:
            est_indices.append(r)
            true_indices.append(c)
        if len(est_indices) == K:
            break

    return (np.ascontiguousarray(est_props[:, est_indices]), 
            np.ascontiguousarray(true_props[:, true_indices]))

def run_validation(args, est_props, N, K):
    """Calculates RMSE or JSD against a ground truth."""
    true_props = np.loadtxt(args.tfile, dtype=float)
    N, K = true_props.shape
    
    # Align the latent components (clusters)
    
    aligned_est, aligned_true = align_latent_factors(est_props, true_props)

    if args.rmse:
        val = tools.rmse_d(aligned_est, aligned_true, N, K)
        log.info(f"RMSE: {val:.7f}")

    if args.jsd:
        # Symmetric Jensen-Shannon Divergence
        div_a = tools.KL(aligned_est, aligned_true, N, K)
        div_b = tools.KL(aligned_true, aligned_est, N, K)
        jsd = 0.5 * (div_a + div_b)
        log.info(f"JSD: {jsd:.7f}")

def run_fitting_eval(args, est_props):
    """Calculates Log-Likelihood of the model given the data."""
    K = est_props.shape[1]
    
    # Load Genotypes
    genotypes = utils.read_data(args.data_path)
    M = genotypes.shape[0]

    # Load and prep Frequencies
    freqs = np.loadtxt(args.pfile, dtype=float)
    if args.inverse:
        freqs = 1.0 - freqs
        
    # Sanity checks
    assert freqs.shape == (M, K), f"Mismatch: P-file {freqs.shape} vs Data {(M, K)}"
    freqs.clip(min=args.bound, max=1.0 - args.bound, out=freqs)

    ll = tools.loglikelihood(genotypes, freqs, est_props)
    log.info(f"Log-likelihood: {round(ll, 1)}")

def main():
    args = parse_input()
    set_numerical_backends(args.threads)

    # Load main estimation file
    est_props = load_proportions(args.qfile, args.bound)

    if args.rmse or args.jsd:
        run_validation(args, est_props)
    elif args.loglike:
        run_fitting_eval(args, est_props)

if __name__ == "__main__":
    main()