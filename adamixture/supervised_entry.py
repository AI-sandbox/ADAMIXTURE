import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path

import configargparse
import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHDYNAMO_SUPPRESS_ERRORS", "1")

import torch

from ._version import __version__
from .entry import print_adamixture_banner

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_args(argv: list[str]) -> configargparse.Namespace:
    """
    Description:
    Parses command-line arguments for the adamixture-supervised command.
    Supervised mode fixes the ancestry assignments of labeled individuals and
    uses their genotype data to anchor the P matrix, while estimating Q for
    all samples (labeled ones are clamped after every update).

    The labels file used for supervision is selected by ``--level`` (default 1):
    level 1 → ``--labels``, level 2 → ``--labels2``, level 3 → ``--labels3``.
    The file format is one label per line: a population name for labeled samples
    or ``-`` for unlabeled samples.  Integer assignments are derived automatically
    from the order of first appearance of each distinct population name.

    Args:
        argv (list[str]): Command-line arguments (excluding the program name).

    Returns:
        configargparse.Namespace: Parsed arguments.
    """
    # Assert that K, min_k, or max_k are not passed in command-line arguments
    for arg in argv:
        assert arg not in ("-k", "--k", "--min_k", "--max_k"), "K, min_k, and max_k are not allowed in supervised mode. K is automatically derived from the labels."

    parser = configargparse.ArgumentParser(
        prog="adamixture-supervised",
        description=(
            "ADAMIXTURE supervised mode. "
            "Uses known population assignments for a subset of samples to "
            "anchor the allele-frequency model while estimating ancestry "
            "proportions for all samples."
        ),
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_path", required=True, type=str,
        help="Path to the genotype data (BED, VCF or PGEN).",
    )
    parser.add_argument(
        "--save_dir", required=True, type=str,
        help="Directory where output files will be saved.",
    )
    parser.add_argument(
        "--name", required=True, type=str,
        help="Experiment/run name used as prefix for output files.",
    )


    # ── Adam-EM hyperparameters ────────────────────────────────────────────────
    parser.add_argument("--lr",            type=float, default=0.005,  help="[only with --algorithm adamem] Learning rate (default: 0.005).")
    parser.add_argument("--beta1",         type=float, default=0.80,   help="[only with --algorithm adamem] Adam beta1 (default: 0.80).")
    parser.add_argument("--beta2",         type=float, default=0.88,   help="[only with --algorithm adamem] Adam beta2 (default: 0.88).")
    parser.add_argument("--reg_adam",      type=float, default=1e-8,   help="[only with --algorithm adamem] Adam epsilon (default: 1e-8).")
    parser.add_argument("--lr_decay",      type=float, default=0.5,    help="[only with --algorithm adamem] Learning rate decay factor (default: 0.5).")
    parser.add_argument("--min_lr",        type=float, default=1e-4,   help="[only with --algorithm adamem] Minimum learning rate (default: 1e-4).")
    parser.add_argument("--patience",      type=int,   default=3,      help="Patience for lr reduction (default: 3).")
    parser.add_argument("--tol",           type=float, default=0.1,    help="Convergence tolerance (default: 0.1).")
    parser.add_argument("--max_iter",      type=int,   default=10000,  help="Maximum Adam-EM iterations (default: 10000).")
    parser.add_argument("--check",         type=int,   default=5,      help="[only with --algorithm adamem] Log-likelihood check frequency (default: 5).")
    parser.add_argument('--algorithm', choices=['brqn', 'adamem'], default='brqn', help='Algorithm to use (brqn for SQP+ZAL QN, adamem for Adam-EM) (default: brqn).')
    parser.add_argument('--Q_hist', type=int, default=3, help=argparse.SUPPRESS)

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("-s", "--seed",    type=int,   default=42,     help="Random seed (default: 42).")
    parser.add_argument("-t", "--threads", type=int,   default=1,      help="Number of CPU threads (default: 1).")
    parser.add_argument("--chunk_size",    type=int,   default=4096,   help="SNP chunk size for I/O (default: 4096).")
    parser.add_argument("--device",        type=str,   default="cpu",  help="Computation device: cpu, cuda, or mps (default: cpu).")
    parser.add_argument("--no_freqs",      action="store_true",        help="Do not save the P matrix.")
    parser.add_argument(
        "--chromosome_mode",
        choices=["all", "autosomes"],
        default="autosomes",
        help="Chromosome filter for input variants: all or autosomes (default: autosomes).",
    )
    parser.add_argument(
        "--autosome_count",
        type=int,
        default=22,
        help="Number of autosomes kept when --chromosome_mode=autosomes (default: 22).",
    )

    # ── Labels & plotting ──────────────────────────────────────────────────────
    parser.add_argument(
        "--labels", required=True, type=str,
        help=(
            "Population labels file (one label per sample, one per line). "
            "Use a population name for labeled samples and '-' for unlabeled. "
            "Also used as level-1 plot labels."
        ),
    )
    parser.add_argument("--labels2", type=str, help="Level-2 grouping labels file (one per sample).")
    parser.add_argument("--labels3", type=str, help="Level-3 grouping labels file (one per sample).")
    parser.add_argument("--colors",  type=str, help="Custom colors file (one per line).")
    parser.add_argument(
        "--level", type=int, default=1, choices=[1, 2, 3],
        help=(
            "Labels level to use for supervision (default: 1). "
            "1 = --labels, 2 = --labels2, 3 = --labels3."
        ),
    )
    parser.add_argument("--plot", nargs="*", default=[], help="Generate a plot after training. Optional: [format] [dpi].")

    args = parser.parse_args(argv)

    # Validate that the labels file for the chosen level was provided.
    level_arg = {1: "labels", 2: "labels2", 3: "labels3"}[args.level]
    if getattr(args, level_arg) is None:
        parser.error(f"--{level_arg} is required when --level {args.level} is selected.")

    # Process --plot
    args.plot_format = "png"
    args.plot_dpi = 300
    if args.plot is not None:
        if len(args.plot) > 0:
            args.plot_format = args.plot[0]
        if len(args.plot) > 1:
            try:
                args.plot_dpi = int(args.plot[1])
            except ValueError:
                parser.error(f"Invalid DPI value: {args.plot[1]}")
        assert args.plot_format in ["pdf", "png", "jpg"], "Plot format must be pdf, png or jpg."
        assert 50 <= args.plot_dpi <= 1200, "DPI must be between 50 and 1200."
    if args.autosome_count < 1:
        parser.error("--autosome_count must be at least 1.")



    return args


def _parse_labels_file(path: "Path") -> list[str]:
    """
    Description:
    Reads a labels file and returns its entries as a list of strings, one per
    sample. Empty lines are skipped.

    Args:
        path (Path): Path to the labels file.

    Returns:
        list[str]: List of label strings (population names or '-').
    """
    return [line.strip() for line in path.open() if line.strip()]


def _labels_to_y(raw_labels: list[str]) -> tuple[np.ndarray, dict[str, int]]:
    """
    Description:
    Converts a list of named labels into an integer assignment vector.
    '-' maps to 0 (unlabeled); all other distinct names are assigned
    consecutive integers (1, 2, ...) in order of first appearance.

    Args:
        raw_labels (list[str]): Label strings from the labels file.

    Returns:
        tuple[np.ndarray, dict[str, int]]:
            - y: Integer assignment array (N,), dtype int32.
            - pop_to_int: Mapping from population name to integer id.
    """
    pop_to_int: dict[str, int] = {}
    next_id = 1
    y_list: list[int] = []
    for lbl in raw_labels:
        if lbl == "-":
            y_list.append(0)
        else:
            if lbl not in pop_to_int:
                pop_to_int[lbl] = next_id
                next_id += 1
            y_list.append(pop_to_int[lbl])
    return np.array(y_list, dtype=np.int32), pop_to_int


def main() -> None:
    """
    Description:
    Entry point for the ``adamixture-supervised`` command.

    Reads named population labels from the labels file selected by ``--level``
    (default level 1 → ``--labels``; level 2 → ``--labels2``; level 3 →
    ``--labels3``).  The format is one name per line, ``-`` for unlabeled.
    Integer assignments are derived automatically.

    Then initialises P from labeled genotype frequencies and Q as a near-one-hot
    matrix, and runs the Adam-EM loop while snapping labeled Q rows back after
    every update (fastmixture ``superQ`` strategy).

    For plotting, all three label levels are passed through as usual; the
    supervision labels (from ``--labels``, ``--labels2``, or ``--labels3``)
    serve simultaneously as plot labels at their corresponding level.

    Args:
        None.

    Returns:
        None
    """

    print_adamixture_banner(__version__)
    log.info("    Supervised Mode\n")
    arg_list = tuple(sys.argv)
    args = parse_args(arg_list[1:])

    # VALIDATE PARAMETERS:
    assert args.lr > 0, "Learning rate (lr) must be positive."
    assert 0 <= args.beta1 < 1, "Adam beta1 must be in [0, 1)."
    assert 0 <= args.beta2 < 1, "Adam beta2 must be in [0, 1)."
    assert 0 < args.lr_decay <= 1, "Learning rate decay (lr_decay) must be in (0, 1]."
    assert args.min_lr > 0, "Minimum learning rate (min_lr) must be positive."
    assert args.patience >= 1, "Patience must be at least 1."
    assert args.seed >= 0, "Seed must be non-negative."
    assert args.max_iter >= 1, "Maximum iterations (max_iter) must be at least 1."
    assert args.check >= 1, "Check frequency (check) must be at least 1."
    assert args.chunk_size >= 1, "Chunk size must be at least 1."
    assert args.tol > 0, "Tolerance (tol) must be positive."
    assert args.reg_adam >= 0, "Adam regularization (reg_adam) must be non-negative."
    assert args.plot_format in ['pdf', 'png', 'jpg'], "Plot format must be pdf, png or jpg."
    assert 50 <= args.plot_dpi <= 1200, "Plot resolution must be between 50 and 1200."
    assert args.Q_hist >= 1, "Q_hist must be at least 1."

    # Thread control
    th = str(args.threads)
    for env_var in [
        "MKL_NUM_THREADS", "MKL_MAX_THREADS", "OMP_NUM_THREADS", "OMP_MAX_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OPENBLAS_MAX_THREADS",
    ]:
        os.environ[env_var] = th

    system = platform.system()
    if system == "Linux":
        os.environ["CC"] = "gcc"
        os.environ["CXX"] = "g++"
    elif system == "Darwin":
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    from .src import utils
    from .src.supervised import (
        init_p_supervised,
        init_q_supervised,
        optimize_supervised,
        optimize_supervised_gpu,
        optimize_supervised_original,
        optimize_supervised_original_gpu,
    )

    device_str = args.device
    if args.algorithm == 'brqn' and device_str == 'mps':
        log.info("    Supervised SQP + ZAL QN (brqn) is not supported on MPS. Running on CPU.")
        device_str = 'cpu'
        args.device = 'cpu'
    use_gpu = device_str in ("cuda", "mps")
    utils.set_seed(args.seed)

    t0 = time.time()

    # ── Load population assignments from the selected level ───────────────────
    level_arg = {1: "labels", 2: "labels2", 3: "labels3"}[args.level]
    sup_path = Path(getattr(args, level_arg))
    if not sup_path.exists():
        log.error(f"    Error: Labels file not found: {sup_path}")
        sys.exit(1)

    raw_labels = _parse_labels_file(sup_path)
    y, pop_to_int = _labels_to_y(raw_labels)

    K = int(y.max())
    assert K >= 2, f"Number of populations K (derived from labels) must be at least 2, but got {K}."

    # ── Load genotype data ────────────────────────────────────────────────────
    G, N, M = utils.read_data(
        args.data_path,
        packed=False,
        chunk_size=args.chunk_size,
        chromosome_mode=args.chromosome_mode,
        autosome_count=args.autosome_count,
    )

    if N != len(y):
        log.error(
            f"    Error: Sample count mismatch — genotype data has {N} samples "
            f"but labels file has {len(y)}."
        )
        sys.exit(1)

    # ── Initialise P and Q ────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    P = init_p_supervised(G, y, K, M)

    Q = rng.random(size=(N, K)).astype(np.float64)
    Q /= Q.sum(axis=1, keepdims=True)
    init_q_supervised(Q, y, K)

    if args.algorithm == 'brqn':
        log.info("    Running supervised SQP + ZAL QN...\n")
    else:
        log.info("    Running supervised Adam-EM...\n")

    # ── Run supervised optimisation ──────────────────────────────────────────────
    if use_gpu:
        device_obj = torch.device(device_str)
        threads_per_block = utils.get_tuning_params(device_obj)
        utils.load_extensions(device_obj)
        if device_obj.type == 'cuda':
            previous_disable = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                G_t, N_t, M_t = utils.read_data(
                    args.data_path,
                    packed=True,
                    chunk_size=args.chunk_size,
                    chromosome_mode=args.chromosome_mode,
                    autosome_count=args.autosome_count,
                    verbose=False,
                )
            finally:
                logging.disable(previous_disable)
            if N_t != N or M_t != M:
                raise ValueError(
                    f"GPU reread shape mismatch: supervised data was N={N}, M={M}; "
                    f"packed data is N={N_t}, M={M_t}."
                )
        else:
            G_t = torch.from_numpy(G) if not isinstance(G, torch.Tensor) else G
        P_t = torch.tensor(P, dtype=utils.get_dtype(device_obj), device=device_obj)
        Q_t = torch.tensor(Q, dtype=utils.get_dtype(device_obj), device=device_obj)
        G_t = utils.manage_gpu_memory(
            G_t,
            device_obj,
            M,
            N,
            K,
            args.chunk_size,
            args.algorithm,
            args.Q_hist,
            include_initialization=False,
        )
        if args.algorithm == 'brqn':
            P_gpu, Q_gpu = optimize_supervised_original_gpu(
                G=G_t, P=P_t, Q=Q_t, y=y,
                max_iter=args.max_iter, K=K, M=M, N=N, tol=args.tol, Q_hist=args.Q_hist,
                device=device_obj, chunk_size=args.chunk_size, threads_per_block=threads_per_block,
            )
        else:
            P_gpu, Q_gpu = optimize_supervised_gpu(
                G=G_t, P=P_t, Q=Q_t, y=y,
                lr=args.lr, beta1=args.beta1, beta2=args.beta2, reg_adam=args.reg_adam,
                max_iter=args.max_iter, check=args.check, M=M,
                lr_decay=args.lr_decay, min_lr=args.min_lr,
                patience=args.patience, tol_adam=args.tol,
                device=device_obj, chunk_size=args.chunk_size, threads_per_block=threads_per_block,
            )
        P_opt = P_gpu.cpu().numpy()
        Q_opt = Q_gpu.cpu().numpy()
    else:
        if args.algorithm == 'brqn':
            P_opt, Q_opt = optimize_supervised_original(
                G=G, P=P, Q=Q, y=y,
                max_iter=args.max_iter, K=K, M=M, N=N, tol=args.tol, Q_hist=args.Q_hist,
            )
        else:
            P_opt, Q_opt = optimize_supervised(
                G=G, P=P, Q=Q, y=y,
                lr=args.lr, beta1=args.beta1, beta2=args.beta2, reg_adam=args.reg_adam,
                max_iter=args.max_iter, check=args.check, K=K, M=M, N=N,
                lr_decay=args.lr_decay, min_lr=args.min_lr,
                patience=args.patience, tol_adam=args.tol,
            )

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_path = Path(args.save_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    q_file = out_path / f"{args.name}.{K}.Q"
    np.savetxt(str(q_file), Q_opt, delimiter=" ", fmt="%.6f")
    log.info(f"    Q matrix saved to: {q_file}")

    if not args.no_freqs:
        p_file = out_path / f"{args.name}.{K}.P"
        np.savetxt(str(p_file), P_opt, delimiter=" ", fmt="%.6f")
        log.info(f"    P matrix saved to: {p_file}")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot is not None:
        from .src.plot import plot_q_matrix

        def _load(path_str: str, flag_name: str) -> list[str] | None:
            """
            Description:
            Loads labels or colors from an optional plot metadata file.

            Args:
                path_str (str): Path to the metadata file.
                flag_name (str): CLI flag name used in warning messages.

            Returns:
                list[str] | None: Loaded strings, or None if the file is absent.
            """
            p = Path(path_str)
            if not p.exists():
                log.warning(f"    Warning: File specified in {flag_name} not found: {path_str}")
                return None
            return _parse_labels_file(p)

        labels  = _load(args.labels, "--labels")
        labels2 = _load(args.labels2, "--labels2") if args.labels2 else None
        labels3 = _load(args.labels3, "--labels3") if args.labels3 else None
        colors  = _load(args.colors, "--colors") if args.colors else None

        plot_path = out_path / f"{args.name}.{K}.{args.plot_format}"
        log.info(f"    Generating plot: {plot_path.name}")
        plot_q_matrix(
            Q_opt, plot_path,
            dpi=args.plot_dpi, format=args.plot_format,
            labels=labels, labels2=labels2, labels3=labels3,
            custom_colors=colors,
        )

    t_tot = time.time() - t0
    log.info(f"\n    Total elapsed time: {t_tot:.2f}s\n")


if __name__ == "__main__":
    main()
