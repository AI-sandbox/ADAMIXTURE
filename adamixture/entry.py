import logging
import sys
from typing import List
import configargparse
import time
import os
import platform

from ._version import __version__
from .src import utils

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def parse_args(argv: List[str]):
    """Training arguments parser"""
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
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate value')

    parser.add_argument('--seed', required=False, type=int, default=42, help='Seed')
    parser.add_argument('--k', required=False, type=int, help='Number of populations/clusters.')
    
    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name')
    parser.add_argument('--threads', required=False, default=1, type=int, help='Number of threads to be used in the execution.')
    
    parser.add_argument('--max_iter', type=int, default=1500, help='Maximum number of iterations for Adam EM')
    parser.add_argument('--check', type=int, default=5, help='Frequency of log-likelihood checks')
    
    parser.add_argument('--max_als', type=int, default=1000, help='Maximum number of iterations for ALS')
    parser.add_argument('--tole_als', type=float, default=1e-4, help='Convergence tolerance for ALS')
    parser.add_argument('--reg_als', type=float, default=1e-5, help='Regularization parameter for ALS')
    parser.add_argument('--power', type=int, default=5, help='Number of power iterations for RSVD')
    parser.add_argument('--tole_svd', type=float, default=1e-1, help='Convergence tolerance for SVD')
    
    return parser.parse_args(argv)
    
def print_adamixture_banner(version: str="1.0") -> None:
    """
    Display the Neural Admixture banner with version and author information.
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
    Authors: Joan Saurina Ricós, Daniel Mas Montserrat and 
             Alexander G. Ioannidis.
    """

    log.info("\n" + banner + info)


def main():
    print_adamixture_banner(__version__)
    arg_list = tuple(sys.argv)
    args = parse_args(arg_list[1:])
    
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
    
    # CONTROL SEED:
    utils.set_seed(args.seed)

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
    
    log.info(f"    Using {th} threads...")
    
    from .src import main
    sys.exit(main.main(args, t0))