import logging
import os
import platform
import sys
import time

import configargparse

from ._version import __version__

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def parse_args(argv: list[str]) -> configargparse.Namespace:
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

    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate (default: 0.005).')
    parser.add_argument('--beta1', type=float, default=0.80, help='Adam beta1 (1st moment decay) (default: 0.80).')
    parser.add_argument('--beta2', type=float, default=0.88, help='Adam beta2 (2nd moment decay) (default: 0.88).')
    parser.add_argument('--reg_adam', type=float, default=1e-8, help='Adam epsilon for numerical stability (default: 1e-8).')

    parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay factor (default: 0.5).')
    parser.add_argument('--min_lr', type=float, default=1e-4, help='Minimum learning rate value (default: 1e-4).')
    parser.add_argument('--patience_adam', type=int, default=3, help='Patience for reducing the learning rate in Adam-EM (default: 3).')
    parser.add_argument('--tol_adam', type=float, default=0.1, help='Tolerance for stopping the Adam-EM algorithm (default: 0.1).')

    parser.add_argument('-s', '--seed', required=False, type=int, default=42, help='Seed (default: 42).')
    parser.add_argument('-k', '--k', required=False, type=int, help='Number of populations/clusters (single run).')
    parser.add_argument('--min_k', required=False, type=int, help='Minimum K for multi-K sweep (inclusive).')
    parser.add_argument('--max_k', required=False, type=int, help='Maximum K for multi-K sweep (inclusive).')

    parser.add_argument('--save_dir', required=True, type=str, help='Save model in this directory.')
    parser.add_argument('--data_path', required=True, type=str, help='Path containing the main data.')
    parser.add_argument('--name', required=True, type=str, help='Experiment/model name.')
    parser.add_argument('-t', '--threads', required=False, default=1, type=int, help='Number of threads to be used in the execution (default: 1).')
    parser.add_argument('--device', required=False, default='cpu', choices=['cpu', 'gpu', 'mps'], help='Device to use (cpu, gpu, mps) (default: cpu).')

    parser.add_argument('--max_iter', type=int, default=10000, help='Maximum number of iterations for Adam EM (default: 10000).')
    parser.add_argument('--check', type=int, default=5, help='Frequency of log-likelihood checks (default: 5).')
    parser.add_argument('--no_freqs', action='store_true', default=False, help='Do not save the P (allele frequencies) matrix (default: False).')

    parser.add_argument('--max_als', type=int, default=1000, help='Maximum number of iterations for ALS (default: 1000).')
    parser.add_argument('--tol_als', type=float, default=1e-4, help='Convergence tolerance for ALS (default: 1e-4).')
    parser.add_argument('--power', type=int, default=5, help='Number of power iterations for SVD (default: 5).')
    parser.add_argument('--tol_svd', type=float, default=1e-1, help='Convergence tolerance for SVD (default: 1e-1).')
    parser.add_argument('--chunk_size', type=int, default=4096, help='Number of SNPs in chunk operations for SVD (default: 4096).')
    parser.add_argument('--cv', nargs='?', const=5, default=0, type=int, help='Enable v-fold cross-validation on genotype entries (default: 5).')

    parser.add_argument('--plot', nargs='*', help='Generate a single combined plot of all Q matrices across the K sweep (Optional: [format] [resolution]) (default: png 300).')
    parser.add_argument('--plot_single', nargs='*', help='Generate individual plots for each K in the sweep (Optional: [format] [resolution]) (default: png 300).')
    parser.add_argument('--labels', type=str, help='Path to population labels file (level 1, one label per sample).')
    parser.add_argument('--labels2', type=str, help='Path to level-2 population grouping file (one label per sample).')
    parser.add_argument('--labels3', type=str, help='Path to level-3 population grouping file (one label per sample).')
    parser.add_argument('--colors', type=str, help='Path to custom colors file (one color per line).')

    args = parser.parse_args(argv)

    # Process plotting arguments (swapping internal roles):
    # args.plot (internal variable for individual plots) represents CLI --plot_single
    # args.plot_single (internal variable for combined plot) represents CLI --plot
    cli_plot_combined = args.plot
    cli_plot_individual = args.plot_single

    # If --plot_single (individual plots) is requested:
    #   disable the combined single plot by default unless --plot was also passed explicitly.
    if cli_plot_individual is not None:
        if '--plot' not in argv:
            cli_plot_combined = None
    else:
        # If --plot_single is NOT requested, the combined single plot is enabled by default.
        if cli_plot_combined is None:
            cli_plot_combined = []

    # Assign to internal variables expected by main.py
    args.plot = cli_plot_individual
    args.plot_single = cli_plot_combined

    args.plot_format = 'png'
    args.plot_dpi = 300

    active_plot_val = None
    if args.plot is not None:
        active_plot_val = args.plot
    elif args.plot_single is not None:
        active_plot_val = args.plot_single

    if active_plot_val is not None:
        if len(active_plot_val) > 0:
            args.plot_format = active_plot_val[0]
        if len(active_plot_val) > 1:
            try:
                args.plot_dpi = int(active_plot_val[1])
            except ValueError:
                parser.error(f"Invalid resolution/DPI value: {active_plot_val[1]}. Must be an integer.")

        # Validation:
        assert args.plot_format in ['pdf', 'png', 'jpg'], f"Invalid plot format: {args.plot_format}. Must be pdf, png or jpg."
        assert 50 <= args.plot_dpi <= 1200, f"Invalid resolution: {args.plot_dpi}. Must be between 50 and 1200."

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
    Preprint: https://www.biorxiv.org/content/10.64898/2026.02.13.700171
    """

    log.info("\n" + banner + info)


def _fix_macos_libomp() -> None:
    """
    On macOS, PyTorch ships its own libomp.dylib which conflicts with
    the Homebrew libomp used by our Cython extensions. Two different OpenMP
    runtimes loaded simultaneously cause segfaults with multiple threads.

    Fix: replace torch's libomp with a symlink to Homebrew's so both
    torch and our extensions use exactly the same OpenMP runtime.
    """
    if platform.system() != "Darwin":
        return

    # Find Homebrew libomp
    brew_omp = None
    for p in ["/opt/homebrew/opt/libomp/lib/libomp.dylib",
              "/usr/local/opt/libomp/lib/libomp.dylib"]:
        if os.path.exists(p):
            brew_omp = p
            break

    if brew_omp is None:
        return

    # Find torch's libomp
    try:
        import torch as _torch
        torch_omp = os.path.join(os.path.dirname(_torch.__file__), "lib", "libomp.dylib")
    except ImportError:
        return

    if not os.path.exists(torch_omp):
        return

    # Already a symlink pointing to the right place? Nothing to do.
    if os.path.islink(torch_omp) and os.path.realpath(torch_omp) == os.path.realpath(brew_omp):
        return

    # Not a symlink (or points elsewhere) → fix it
    try:
        backup = torch_omp + ".bak"
        if not os.path.exists(backup):
            os.rename(torch_omp, backup)
        else:
            os.remove(torch_omp)
        os.symlink(brew_omp, torch_omp)
        log.info(f"    Fixed OpenMP conflict: linked torch's libomp → {brew_omp}")
    except OSError as e:
        log.warning(f"    Could not fix OpenMP conflict automatically: {e}")
        log.warning("    To fix manually, run:")
        log.warning(f"      mv {torch_omp} {torch_omp}.bak")
        log.warning(f"      ln -s {brew_omp} {torch_omp}")

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
    _fix_macos_libomp()
    import torch

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
    assert args.plot_format in ['pdf', 'png', 'jpg'], "Plot format must be pdf, png or jpg."
    assert 50 <= args.plot_dpi <= 1200, "Plot resolution must be between 50 and 1200."
    assert args.cv >= 0, "CV folds (cv) must be >= 0 (0 disables CV)."
    if args.cv:
        assert args.cv >= 2, "CV folds (cv) must be at least 2 when CV is enabled."

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
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
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

        # GPU LIMITS (MAX_K=64 in CUDA kernels):
        if args.k is not None:
            assert args.k <= 64, f"    Error: K={args.k} exceeds the current GPU limit (MAX_K=64)."
        if args.max_k is not None:
            assert args.max_k <= 64, f"    Error: max_k={args.max_k} exceeds the current GPU limit (MAX_K=64)."

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


    # CONTROL INDUCTOR CONFIG:
    if args.device == 'mps':
        try:
            import torch._inductor.config as inductor_config
            inductor_config.max_autotune_gemm = False
        except (ImportError, AttributeError):
            pass

    # CONTROL SEED:
    from .src import utils
    utils.set_seed(args.seed)

    log.info(f"    Using {th} threads...")

    from .src import main
    sys.exit(main.main(args, t0))
