import argparse
import logging
import sys
import time
from pathlib import Path
from argparse import ArgumentError, ArgumentTypeError
import numpy as np
import torch

from . import utils
from ..model.adamixture import train
from ..plot import plot_q_matrix

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def fit_model(args: argparse.Namespace, G: torch.Tensor | np.ndarray, N: int, M: int, K: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Description:
    Trains the ADAMIXTURE model for a specific number of ancestral populations (K).
    Resulting matrices are saved to the specified directory.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        G (torch.Tensor | np.ndarray): Input genotype matrix.
        N (int): Number of individuals.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations to fit.

    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    (save_dir, name, seed, lr, beta1, 
    beta2, reg_adam, max_iter, 
    check, max_als, tol_als, power, tol_svd,
    lr_decay, min_lr, chunk_size, patience_adam, tol_adam, device) = (args.save_dir, args.name, int(args.seed), float(args.lr),
                float(args.beta1), float(args.beta2), float(args.reg_adam), int(args.max_iter),
                int(args.check), int(args.max_als), float(args.tol_als), int(args.power), 
                float(args.tol_svd), float(args.lr_decay), float(args.min_lr),
                int(args.chunk_size), int(args.patience_adam), float(args.tol_adam), args.device)
            
    P, Q = train(G, N, M, K, seed, lr, beta1, beta2, reg_adam, max_iter, 
                check, max_als, tol_als, power, tol_svd, lr_decay, min_lr, chunk_size,
                patience_adam, tol_adam, device)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    utils.write_outputs(Q, name, K, save_dir, P=None if args.no_freqs else P)
    
    if args.plot is not None:
        labels = None
        if args.labels and Path(args.labels).exists():
            with open(args.labels, 'r') as f:
                labels = [line.strip() for line in f if line.strip()]
        
        colors = None
        if args.colors and Path(args.colors).exists():
            with open(args.colors, 'r') as f:
                colors = [line.strip() for line in f if line.strip()]
            if len(colors) != K:
                log.warning(f"    Number of colors in {args.colors} ({len(colors)}) does not match K={K}. Using default colors.")
                colors = None
        
        plot_path = Path(save_dir) / f"{name}.{K}.{args.plot_format}"
        log.info(f"    Generating plot: {plot_path}")
        plot_q_matrix(Q, plot_path, dpi=args.plot_dpi, format=args.plot_format, 
                     labels=labels, custom_colors=colors)

    return P, Q


def main(args: argparse.Namespace, t0: float) -> None:
    """
    Description:
    The core training loop coordinator. It handles data loading once and then 
    iterates over the requested K values (single or sweep) to perform training.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        t0 (float): Program start time for total execution measurement.

    Returns:
        None
    """
    try:
        # Build list of K values
        if args.min_k is not None and args.max_k is not None:
            k_values = list(range(args.min_k, args.max_k + 1))
            log.info(f"\n    Running from {args.min_k} to {args.max_k}.\n")
        else:
            k_values = [int(args.k)]
        
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
        # READ DATA (once):
        is_gpu = 'cuda' in args.device or 'mps' in args.device
        packed = 'cuda' in args.device # Only packed for CUDA
        G, N, M = utils.read_data(args.data_path, packed=packed, chunk_size=args.chunk_size)

        trained: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for K in k_values:
            log.info(f"\n    Running on K = {K}.\n")
            
            t_k = time.time()
            
            # TRAIN MODEL:
            P, Q = fit_model(args, G, N, M, K)
            if args.cv:
                trained[K] = (P, Q)
            
            log.info(f"\n    K={K} completed in {time.time() - t_k:.2f} seconds.\n")
        
        # CROSS-VALIDATION (after all training, on CPU with unpacked G):
        cv_results: dict[int, float] = {}
        if args.cv and trained:
            if not isinstance(G, np.ndarray):
                del G
                G, _, _ = utils.read_data(args.data_path, packed=False, chunk_size=args.chunk_size)

            from .cv import run_cross_validation
            for K, (P, Q) in sorted(trained.items()):
                log.info(f"\n    Running {int(args.cv)}-fold CV on genotype entries for K={K}...")
                cv_results[K] = run_cross_validation(args, G, N, M, K, P, Q)

            log.info("")
            log.info("    ---- Cross-validation summary ----")
            for k_val, idx in sorted(cv_results.items()):
                log.info(f"    K={k_val}: CV index = {idx:.4f}")
            best_k = min(cv_results, key=cv_results.get)
            log.info(f"    Best K = {best_k} (CV index = {cv_results[best_k]:.4f})")
            log.info("    ----------------------------------")

        t1 = time.time()
        log.info("")
        log.info(f"    Total elapsed time: {t1-t0:.2f} seconds.")
        log.info("")
        
        logging.shutdown()

    except (ArgumentError, ArgumentTypeError) as e:
        log.error(f"    Error parsing arguments")
        logging.shutdown()
        raise e
        
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        logging.shutdown()
