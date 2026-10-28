import argparse
import logging
import os
import platform
import sys
import time

import configargparse

from ._version import __version__

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_SUPPRESS_ERRORS", "1")

class AdamHelpAction(argparse.Action):
    """
    Description:
    Prints the full help including the Adam-EM tuning options, then exits.
    ``builder`` is the parser factory, called as ``builder(show_adam=True)``.
    """
    def __init__(self, option_strings, builder, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings=option_strings, dest=dest, default=default, nargs=0, help=help)
        self.builder = builder

    def __call__(self, parser, namespace, values, option_string=None):
        self.builder(show_adam=True).print_help()
        parser.exit()

def build_parser(show_adam: bool = False) -> configargparse.ArgumentParser:
    """
    Description:
    Builds the ADAMIXTURE argument parser.

    Args:
        show_adam (bool): Whether the Adam-EM tuning options are listed in the help.
            They are always accepted on the command line.

    Returns:
        configargparse.ArgumentParser: Configured parser.
    """
    parser = configargparse.ArgumentParser(
        prog='adamixture',
        description='Fast Biobank-Scale Population Genetics Clustering.',
        epilog=None if show_adam else 'Adam-EM tuning options (--algorithm adamem) are hidden; run "adamixture --help-adam" to list them.',
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument('--help-adam', action=AdamHelpAction, builder=build_parser, help='Show help including the Adam-EM tuning options and exit.')

    required = parser.add_argument_group('required arguments')
    required.add_argument('--data_path', required=True, type=str, help='Input genotype file (BED, PGEN, VCF or BCF).')
    required.add_argument('--save_dir', required=True, type=str, help='Output directory.')
    required.add_argument('--name', required=True, type=str, help='Experiment/model name used as output file prefix.')

    k_group = parser.add_argument_group('number of populations (use -k, or --min_k and --max_k)')
    k_group.add_argument('-k', '--k', required=False, type=int, help='Number of populations/clusters (single run). K=1 is the no-structure null model.')
    k_group.add_argument('--min_k', required=False, type=int, help='Minimum K for multi-K sweep (inclusive). K=1 is the no-structure null model.')
    k_group.add_argument('--max_k', required=False, type=int, help='Maximum K for multi-K sweep (inclusive).')

    general = parser.add_argument_group('general')
    general.add_argument('-t', '--threads', required=False, default=1, type=int, help='Number of CPU threads (default: 1).')
    general.add_argument('-s', '--seed', required=False, type=int, default=42, help='Random seed (default: 42).')
    general.add_argument('--device', required=False, default='cpu', choices=['cpu', 'gpu', 'mps'], help='Device to use (default: cpu).')
    general.add_argument('--algorithm', choices=['brqn', 'adamem'], default='brqn', help='Solver: brqn (SQP + ZAL quasi-Newton) or adamem (Adam-EM) (default: brqn).')
    general.add_argument('--n_inits', type=int, default=1, help='Number of independent initializations; the run with the best log-likelihood is kept (default: 1).')
    general.add_argument('--cv', nargs='?', const=5, default=0, type=int, help='Enable V-fold cross-validation on genotype entries; V is 5 if no value is given (default: disabled).')
    general.add_argument('--no_freqs', action='store_true', default=False, help='Do not save the P (allele frequencies) matrix.')

    data = parser.add_argument_group('input filtering')
    data.add_argument('--chrom_mode', choices=['all', 'autosomes'], default='autosomes', help='Chromosome filter for input variants (default: autosomes).')
    data.add_argument('--autosomes', type=int, default=22, help='Number of autosomes kept when --chrom_mode=autosomes (default: 22).')
    data.add_argument('--specific_chrom', nargs='+', help='List of specific chromosomes to analyze when --chrom_mode=autosomes (overrides --autosomes).')

    optim = parser.add_argument_group('optimization')
    optim.add_argument('--max_iter', type=int, default=10000, help='Maximum number of solver iterations (default: 10000).')
    optim.add_argument('--tol', type=float, default=0.1, help='Convergence tolerance on the log-likelihood change (default: 0.1).')
    optim.add_argument('--patience', type=int, default=3, help='Iterations without log-likelihood improvement before stopping; with adamem, checks before decaying the learning rate (default: 3).')
    optim.add_argument('--Q_hist', type=int, default=3, help=argparse.SUPPRESS)

    init = parser.add_argument_group('initialization (SVD + ALS)')
    init.add_argument('--init', choices=['em', 'als'], default='als', help='Initialization method: SVD+ALS (als) or random EM priming (em) (default: als).')
    init.add_argument('--em_init_steps', type=int, default=5, help=argparse.SUPPRESS)
    init.add_argument('--max_als', type=int, default=1000, help='Maximum number of ALS iterations (default: 1000).')
    init.add_argument('--tol_als', type=float, default=1e-4, help='Convergence tolerance for ALS (default: 1e-4).')
    init.add_argument('--beta_als', type=float, default=0.8, help='Momentum extrapolation factor for ALS, in [0, 1) (default: 0.8).')
    init.add_argument('--power', type=int, default=5, help='Number of power iterations for SVD (default: 5).')
    init.add_argument('--tol_svd', type=float, default=1e-1, help='Convergence tolerance for SVD (default: 1e-1).')
    init.add_argument('--chunk_size', type=int, default=8192, help='Number of SNPs per chunk in SVD (default: 8192).')

    plot = parser.add_argument_group('plotting')
    plot.add_argument('--plot', nargs='*', help='Plot format and resolution: single plot for -k, combined sweep plot for multi-K (optional: [format] [dpi]) (default: png 300).')
    plot.add_argument('--plot_single', nargs='*', help='[multi-K only] Individual plot for each K in the sweep (optional: [format] [dpi]) (default: png 300).')
    plot.add_argument('--labels', type=str, help='Population labels file (level 1, one label per sample).')
    plot.add_argument('--labels2', type=str, help='Level-2 population grouping file (one label per sample).')
    plot.add_argument('--labels3', type=str, help='Level-3 population grouping file (one label per sample).')
    plot.add_argument('--colors', type=str, help='Custom colors file (one color per line).')

    adam = parser.add_argument_group('Adam-EM tuning (only with --algorithm adamem)') if show_adam else parser
    def adam_help(text: str) -> str:
        return text if show_adam else argparse.SUPPRESS
    adam.add_argument('--lr', type=float, default=0.005, help=adam_help('Learning rate (default: 0.005).'))
    adam.add_argument('--beta1', type=float, default=0.80, help=adam_help('Adam beta1, 1st moment decay (default: 0.80).'))
    adam.add_argument('--beta2', type=float, default=0.88, help=adam_help('Adam beta2, 2nd moment decay (default: 0.88).'))
    adam.add_argument('--reg_adam', type=float, default=1e-8, help=adam_help('Adam epsilon for numerical stability (default: 1e-8).'))
    adam.add_argument('--lr_decay', type=float, default=0.5, help=adam_help('Learning rate decay factor (default: 0.5).'))
    adam.add_argument('--min_lr', type=float, default=1e-4, help=adam_help('Minimum learning rate (default: 1e-4).'))
    adam.add_argument('--check', type=int, default=5, help=adam_help('Frequency of log-likelihood checks (default: 5).'))

    return parser

def parse_args(argv: list[str]) -> configargparse.Namespace:
    """
    Description:
    Parses command-line arguments for the ADAMIXTURE training script.

    Args:
        argv (List[str]): List of command-line arguments.

    Returns:
        configargparse.Namespace: Parsed arguments object.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validation: need either --k or both --min_k and --max_k
    has_single = args.k is not None
    has_range = args.min_k is not None and args.max_k is not None
    if not has_single and not has_range:
        parser.error("Must specify either --k or both --min_k and --max_k.")
    if has_range and args.min_k > args.max_k:
        parser.error("--min_k must be <= --max_k.")
    if args.k is not None and args.k < 1:
        parser.error("Number of clusters (k) must be at least 1.")
    if args.min_k is not None and args.min_k < 1:
        parser.error("Minimum K (min_k) must be at least 1.")
    if args.max_k is not None and args.max_k < 1:
        parser.error("Maximum K (max_k) must be at least 1.")

    # Configure Plotting Arguments:
    if has_single:
        if args.plot is None and args.plot_single is not None:
            args.plot = args.plot_single
        args.plot_single = None

        if args.plot is None:
            args.plot = []
    else:
        if args.plot is None and args.plot_single is None:
            args.plot = []

    args.plot_format = 'png'
    args.plot_dpi = 300

    active_plot_val = args.plot if args.plot is not None else args.plot_single

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

    if args.tol <= 0.0:
        parser.error("--tol must be greater than 0.")
    if not (0.0 <= args.beta_als < 1.0):
        parser.error("--beta_als must be in [0, 1).")
    if args.Q_hist < 1:
        parser.error("--Q_hist must be at least 1.")
    if args.chrom_mode == "autosomes" and not args.specific_chrom and args.autosomes < 1:
        parser.error("--autosomes must be at least 1.")

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
      ___  ____   ___  __  __ _____ _   _ _______ _    _ _____  ______
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
    Paper: https://doi.org/10.1093/bioinformatics/btag236
    """

    log.info("\n" + banner + info)


def _fix_macos_libomp() -> None:
    """
    Description:
    On macOS, PyTorch ships its own libomp.dylib. If the package was built
    as a wheel or with a local libomp (e.g. ~/.local/libomp/lib/libomp.dylib),
    loading two different OpenMP runtimes simultaneously causes segfaults
    with multiple threads.

    Fix: point any separate libomp to PyTorch's libomp via a symlink,
    ensuring only a single OpenMP runtime is loaded.
    """
    if platform.system() != "Darwin":
        return

    candidate_paths = [
        os.path.join(os.path.dirname(__file__), ".dylibs", "libomp.dylib"),
        os.path.expanduser("~/.local/libomp/lib/libomp.dylib"),
    ]

    try:
        import importlib.util
        torch_spec = importlib.util.find_spec("torch")
        if not torch_spec or not torch_spec.origin:
            return
        torch_omp = os.path.join(os.path.dirname(torch_spec.origin), "lib", "libomp.dylib")
    except Exception:
        return

    if not os.path.exists(torch_omp):
        return

    torch_omp_real = os.path.realpath(torch_omp)

    for omp_path in candidate_paths:
        if not os.path.exists(omp_path):
            continue
        if os.path.islink(omp_path) and os.path.realpath(omp_path) == torch_omp_real:
            continue
        if os.path.realpath(omp_path) == torch_omp_real:
            continue

        try:
            backup = omp_path + ".bak"
            if not os.path.exists(backup):
                os.rename(omp_path, backup)
            else:
                os.remove(omp_path)
            os.symlink(torch_omp_real, omp_path)
            log.info(f"    Fixed OpenMP conflict: linked {omp_path} → torch's libomp")
        except OSError as e:
            log.warning(f"    Could not fix OpenMP conflict for {omp_path}: {e}")

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
    assert args.threads >= 1, "Threads (threads) must be at least 1."
    assert args.lr > 0, "Learning rate (lr) must be positive."
    assert 0 <= args.beta1 < 1, "Adam beta1 must be in [0, 1)."
    assert 0 <= args.beta2 < 1, "Adam beta2 must be in [0, 1)."
    assert 0 < args.lr_decay <= 1, "Learning rate decay (lr_decay) must be in (0, 1]."
    assert args.min_lr > 0, "Minimum learning rate (min_lr) must be positive."
    assert args.patience >= 1, "Patience must be at least 1."
    assert args.seed >= 0, "Seed must be non-negative."
    if args.k is not None:
        assert args.k >= 1, "Number of clusters (k) must be at least 1."
    if args.min_k is not None:
        assert args.min_k >= 1, "Minimum K (min_k) must be at least 1."
    if args.max_k is not None:
        assert args.max_k >= 1, "Maximum K (max_k) must be at least 1."
    if args.max_k is not None and args.min_k is not None:
        assert args.max_k >= args.min_k, "Maximum K (max_k) must be >= minimum K (min_k)."
    assert args.max_iter >= 1, "Maximum iterations (max_iter) must be at least 1."
    assert args.check >= 1, "Check frequency (check) must be at least 1."
    assert args.max_als >= 1, "Maximum ALS iterations (max_als) must be at least 1."
    assert args.em_init_steps >= 0, "EM initialization steps (em_init_steps) must be non-negative."
    assert args.chunk_size >= 1, "Chunk size must be at least 1."
    assert args.power >= 1, "Power iterations (power) must be at least 1."
    assert args.Q_hist >= 1, "Q history (Q_hist) must be at least 1."
    assert args.autosomes >= 1, "Autosome count (autosomes) must be at least 1."
    assert args.tol > 0, "Tolerance (tol) must be positive."
    assert args.tol_als > 0, "ALS tolerance (tol_als) must be positive."
    assert 0 <= args.beta_als < 1, "ALS beta (beta_als) must be in [0, 1)."
    assert args.tol_svd > 0, "SVD tolerance (tol_svd) must be positive."
    assert args.reg_adam >= 0, "Adam regularization (reg_adam) must be non-negative."
    assert args.plot_format in ['pdf', 'png', 'jpg'], "Plot format must be pdf, png or jpg."
    assert 50 <= args.plot_dpi <= 1200, "Plot resolution must be between 50 and 1200."
    assert 1 <= args.n_inits <= 10, "Number of initializations (n_inits) must be between 1 and 10."
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

    from .compat import Device, require_torch

    if args.device in ('gpu', 'cuda'):
        try:
            torch = require_torch()
        except ImportError as exc:
            log.error(f"    {exc}")
            sys.exit(1)

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
        try:
            torch = require_torch()
        except ImportError as exc:
            log.error(f"    {exc}")
            sys.exit(1)

        if not torch.backends.mps.is_available():
            log.error("    MPS requested via --device mps but MPS is not available.")
            sys.exit(1)
        if args.algorithm == 'brqn':
            if args.k is not None and args.k > 32:
                log.error(f"    Error: K={args.k} exceeds the current MPS BR-QN limit (MAX_K=32).")
                sys.exit(1)
            if args.max_k is not None and args.max_k > 32:
                log.error(f"    Error: max_k={args.max_k} exceeds the current MPS BR-QN limit (MAX_K=32).")
                sys.exit(1)
    elif args.device == 'cpu':
        pass
    else:
        log.error(f"    Invalid or unavailable device '{args.device}'.")
        sys.exit(1)

    # Final check: can we actually create a device object?
    try:
        Device(args.device)
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
