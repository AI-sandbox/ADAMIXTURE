import argparse
import logging
import sys
import time
from pathlib import Path
from argparse import ArgumentError, ArgumentTypeError
import numpy as np

from . import utils
from ..model.adamixture import train

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def fit_model(args: argparse.Namespace, G: np.ndarray, K: int) -> tuple:
    """Train the ADAMIXTURE model for a given K.
    
    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        G (np.ndarray): Genotype matrix (M x N).
        K (int): Number of ancestral populations.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Optimized P and Q matrices.
    """
    (save_dir, name, seed, lr, beta1, 
    beta2, reg_adam, max_iter, 
    check, max_als, tole_als, power, tole_svd,
    reg_als, lr_decay, min_lr, chunk_size,
    correlation_als, stall_als, patience_adam, tol_adam) = (args.save_dir, args.name, int(args.seed), float(args.lr),
                float(args.beta1), float(args.beta2), float(args.reg_adam), int(args.max_iter),
                int(args.check), int(args.max_als), float(args.tole_als), int(args.power), 
                float(args.tole_svd), float(args.reg_als), float(args.lr_decay), float(args.min_lr),
                int(args.chunk_size), float(args.correlation_als), int(args.stall_als),
                int(args.patience_adam), float(args.tol_adam))
            
    P, Q = train(G, K, seed, lr, beta1, beta2, reg_adam, max_iter, 
                check, max_als, tole_als, power, tole_svd, reg_als, lr_decay, min_lr, chunk_size,
                correlation_als, stall_als, patience_adam, tol_adam)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    utils.write_outputs(Q, name, K, save_dir, P=None if args.no_freqs else P)

    return P, Q


def main(args: argparse.Namespace, t0: float):
    """Training entry point. Supports single K (--k) or multi-K sweep (--min_k/--max_k).
    """
    try:
        # Build list of K values
        if args.min_k is not None and args.max_k is not None:
            k_values = list(range(args.min_k, args.max_k + 1))
        else:
            k_values = [int(args.k)]
        
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
        # READ DATA (once):
        G = utils.read_data(args.data_path)
        
        for K in k_values:
            log.info(f"\n    Running on K = {K}.\n")
            
            t_k = time.time()
            
            # TRAIN MODEL:
            P, Q = fit_model(args, G, K)
            
            log.info(f"\n    K={K} completed in {time.time() - t_k:.2f} seconds.\n")
        
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
