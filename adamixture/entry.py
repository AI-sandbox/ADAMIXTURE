import logging
import sys
from typing import List
import configargparse
import time
import os
import platform
import torch

from ._version import __version__

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def parse_args(argv: List[str]) -> configargparse.Namespace:
    """
    Description:
    Parses command-line arguments for the ADAMIXTURE training script.

    Args:
        argv (List[str]): List of command-line arguments.

    Returns:
        configargparse.Namespace: Parsed arguments object.
    """
    parser = configargparse.ArgumentParser(
        prog='adamixture',
        description='Population clustering using ADAM-EM.',
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.80, help='Adam beta1 (1st moment decay)')
    parser.add_argument('--beta2', type=float, default=0.88, help='Adam beta2 (2nd moment decay)')
    parser.add_argument('--reg_adam', type=float, default=1e-8, help='Adam epsilon for numerical stability')
    
    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate value')
    parser.add_argument('--patience_adam', type=int, default=3, help='Patience for reducing the learning rate in Adam-EM')
    parser.add_argument('--tol_adam', type=float, default=0.1, help='Tolerance for stopping the Adam-EM algorithm')

    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    parser.add_argument('--k', required=False, type=int, help='Number of populations/clusters (single run).')
    parser.add_argument('--min_k', required=False, type=int, help='Minimum K for multi-K sweep (inclusive).')
    parser.add_argument('--max_k', required=False, type=int, help='Maximum K for multi-K sweep (inclusive).')
    
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name')
    parser.add_argument('--threads', required=False, default=1, type=int, help='Number of threads to be used in the execution.')
    parser.add_argument('--device', required=False, default='cpu', choices=['cpu', 'gpu', 'mps'], help='Device to use (cpu, gpu, mps)')
    
    parser.add_argument('--max_iter', type=int, default=10000, help='Maximum number of iterations for Adam EM')
    parser.add_argument('--check', type=int, default=5, help='Frequency of log-likelihood checks')
    parser.add_argument('--no_freqs', action='store_true', default=False, help='Do not save the P (allele frequencies) matrix')
    
    parser.add_argument('--max_als', type=int, default=1000, help='Maximum number of iterations for ALS')
    parser.add_argument('--tol_als', type=float, default=1e-4, help='Convergence tolerance for ALS')
    parser.add_argument('--power', type=int, default=5, help='Number of power iterations for SVD')
    parser.add_argument('--tol_svd', type=float, default=1e-1, help='Convergence tolerance for SVD')
    parser.add_argument('--chunk_size', type=int, default=4096, help='Number of SNPs in chunk operations for SVD')
    
    args = parser.parse_args(argv)
    
    # Validation: need either --k or both --min_k and --max_k
    has_single = args.k is not None
    has_range = args.min_k is not None and args.max_k is not None
    if not has_single and not has_range:
        parser.error("Must specify either --k or both --min_k and --max_k.")
    if has_range and args.min_k > args.max_k:
        parser.error("--min_k must be <= --max_k.")
    
    return args
    
def print_adamixture_banner(version: str = "1.0") -> None:
    """
    Description:
    Displays the ADAMIXTURE ASCII banner along with version and author information.

    Args:
        version (str): The software version to display. Defaults to "1.0".

    Returns:
        None
    """
    banner = r"""
      ___  ____   ___  __  __ _____       _______ _    _ _____  ______
     / _ \|  _ \ / _ \|  \/  |_   _\ \ / /__   __| |  | |  __ \|  ____|
    / /_\ | | | / /_\ | \  / | | |  \ V /   | |  | |  | | |__) | |__   
    |  _  | | | |  _  | |\/| | | |   > <    | |  | |  | |  _  /|  __|  
    | | | | |_| | | | | |  | |_| |_ / . \   | |  | |__| | | \ \| |____ 
    \_| |_/____/\_| |_|_|  |_|_____/_/ \_\  |_|   \____/|_|  \_\______|
    """

    info = f"""
    Version: {version}
    Authors: Joan Saurina-i-Ricos, Daniel Mas Montserrat and 
             Alexander G. Ioannidis.
    Pre-print: https://www.biorxiv.org/content/10.64898/2026.02.13.700171v1
    """

    log.info("\n" + banner + info)


def main() -> None:
    """
    Description:
    Main entry point for the ADAMIXTURE command-line interface. 
    Handles application setup, environment configuration, and execution flow.

    Args:
        None

    Returns:
        None
    """
    print_adamixture_banner(__version__)
    arg_list = tuple(sys.argv)
    args = parse_args(arg_list[1:])
    
    # CONTROL THREADS:
    th = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = th
    os.environ["MKL_MAX_THREADS"] = th
    os.environ["OMP_NUM_THREADS"] = th
    os.environ["OMP_MAX_THREADS"] = th
    os.environ["NUMEXPR_NUM_THREADS"] = th
    os.environ["NUMEXPR_MAX_THREADS"] = th
    os.environ["OPENBLAS_NUM_THREADS"] = th
    os.environ["OPENBLAS_MAX_THREADS"] = th

    # VALIDATE PARAMETERS:
    assert args.lr > 0, "Learning rate (lr) must be positive."
    assert 0 <= args.beta1 < 1, "Adam beta1 must be in [0, 1)."
    assert 0 <= args.beta2 < 1, "Adam beta2 must be in [0, 1)."
    assert 0 < args.lr_decay <= 1, "Learning rate decay (lr_decay) must be in (0, 1]."
    assert args.min_lr > 0, "Minimum learning rate (min_lr) must be positive."
    assert args.patience_adam >= 1, "Patience (patience_adam) must be at least 1."
    assert args.seed >= 0, "Seed must be non-negative."
    if args.k is not None:
        assert args.k >= 2, "Number of clusters (k) must be at least 2."
    if args.min_k is not None:
        assert args.min_k >= 2, "Minimum K (min_k) must be at least 2."
    assert args.max_iter >= 1, "Maximum iterations (max_iter) must be at least 1."
    assert args.check >= 1, "Check frequency (check) must be at least 1."
    assert args.max_als >= 1, "Maximum ALS iterations (max_als) must be at least 1."
    assert args.chunk_size >= 1, "Chunk size must be at least 1."
    assert args.tol_adam > 0, "Adam tolerance (tol_adam) must be positive."
    assert args.tol_als > 0, "ALS tolerance (tol_als) must be positive."
    assert args.tol_svd > 0, "SVD tolerance (tol_svd) must be positive."
    assert args.reg_adam >= 0, "Adam regularization (reg_adam) must be non-negative."

    # CONTROL TIME:
    t0 = time.time()
    
    #CONTROL OS:
    system = platform.system()
    if system == "Linux":
        log.info("    Operating system is Linux!")
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"
    elif system == "Darwin":
        log.info("    Operating system is Darwin (Mac OS)!")
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
    elif system == "Windows":
        log.info("    Operating system is Windows!")
        pass
    else:
        log.info(f"System not recognized: {system}")
        sys.exit(1)
    
    if args.device == 'gpu':
        if not torch.cuda.is_available():
            log.error("    GPU requested via --device gpu but CUDA is not available.")
            sys.exit(1)
        args.device = 'cuda'
    elif args.device == 'mps':
        if not torch.backends.mps.is_available():
            log.error("    MPS requested via --device mps but MPS is not available.")
            sys.exit(1)
    
    # Final check: can we actually create a device object?
    try:
        torch.device(args.device)
    except Exception as e:
        log.error(f"    Invalid or unavailable device '{args.device}': {e}")
        sys.exit(1)
    
    # CONTROL SEED:
    from .src import utils
    utils.set_seed(args.seed)

    log.info(f"    Using {th} threads...")
    
    from .src import main
    sys.exit(main.main(args, t0))
