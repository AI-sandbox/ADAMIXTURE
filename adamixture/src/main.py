import argparse
import logging
import sys
import time
from pathlib import Path
from argparse import ArgumentError, ArgumentTypeError
import numpy as np

from . import utils
from ..model.adamixture import train
from ..model.cross import crossRun

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
    reg_als, lr_decay, min_lr, chunk_size) = (args.save_dir, args.name, int(args.seed), float(args.lr),
                float(args.beta1), float(args.beta2), float(args.reg_adam), int(args.max_iter),
                int(args.check), int(args.max_als), float(args.tole_als), int(args.power), 
                float(args.tole_svd), float(args.reg_als), float(args.lr_decay), float(args.min_lr),
                int(args.chunk_size))
            
    P, Q = train(G, K, seed, lr, beta1, beta2, reg_adam, max_iter, 
                check, max_als, tole_als, power, tole_svd, reg_als, lr_decay, min_lr, chunk_size)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    utils.write_outputs(Q, name, K, save_dir, P=None if args.no_freqs else P)

    return P, Q

def run_cross_validation(args: argparse.Namespace, G: np.ndarray, 
                         P: np.ndarray, Q: np.ndarray, K: int) -> dict:
    """Run cross-validation after training for a given K.
    
    Args:
        args (argparse.Namespace): Parsed CLI arguments.
        G (np.ndarray): Genotype matrix (M x N).
        P (np.ndarray): Trained allele frequency matrix (M x K).
        Q (np.ndarray): Trained admixture proportions matrix (N x K).
        K (int): Number of ancestral populations.
    
    Returns:
        dict: {'avg': float, 'std': float} with cross-validation deviance error.
    """
    cross = int(args.cv)
    
    res = crossRun(G, P, Q, K, seed=int(args.seed), lr=float(args.lr), beta1=float(args.beta1), beta2=float(args.beta2),
                reg_adam=float(args.reg_adam), max_iter=int(args.max_iter), check=int(args.check), cross=cross,
                lr_decay=float(args.lr_decay), min_lr=float(args.min_lr), cv_tole=float(args.cv_tole))
    
    return res

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
        
        # Collect CV results across all K values
        cv_results = {}
        
        for K in k_values:
            log.info(f"\n    Running on K = {K}.\n")
            
            t_k = time.time()
            
            # TRAIN MODEL:
            P, Q = fit_model(args, G, K)
            
            # CROSS-VALIDATION (if requested):
            if args.cv > 0:
                cv_results[K] = run_cross_validation(args, G, P, Q, K)
            
            log.info(f"\n    K={K} completed in {time.time() - t_k:.2f} seconds.\n")
        
        # Write single CV report file (all K values)
        if cv_results:
            out_path = Path(args.save_dir)
            cv_file = out_path / f"{args.name}.cv"
            with open(cv_file, "w") as f:
                f.write("K\tavg\tstd\n")
                for K in sorted(cv_results.keys()):
                    res = cv_results[K]
                    f.write(f"{K}\t{res['avg']:.6f}\t{res['std']:.6f}\n")
            log.info(f"    Cross-validation results saved to {cv_file}")
            
            # Print summary table
            log.info("")
            log.info(f"    {'K':>4}  {'CV Error':>12}  {'SD':>12}")
            log.info(f"    {'─'*4}  {'─'*12}  {'─'*12}")
            for K in sorted(cv_results.keys()):
                res = cv_results[K]
                log.info(f"    {K:>4d}  {res['avg']:>12.5f}  {res['std']:>12.5f}")
            log.info("")
            
            # Suggest optimal K
            best_k = min(cv_results, key=lambda k: cv_results[k]['avg'])
            log.info(f"\n    Suggested optimal K = {best_k} (lowest CV error: {cv_results[best_k]['avg']:.5f})\n")
            # Generate CV plot only for multi-K
            if len(cv_results) > 1:
                plot_file = out_path / f"{args.name}.cv.png"
                utils.plot_cv_results(cv_results, plot_file)
                log.info(f"    Cross-validation plot saved to {plot_file}\n")
        
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
