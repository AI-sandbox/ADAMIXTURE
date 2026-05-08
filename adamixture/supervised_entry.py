import logging
import os
import platform
import sys
import time

import configargparse
import numpy as np

from ._version import __version__

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
    parser.add_argument(
        "-k", "--k", required=True, type=int,
        help="Number of ancestral populations K.",
    )

    # ── Adam-EM hyperparameters ────────────────────────────────────────────────
    parser.add_argument("--lr",            type=float, default=0.005,  help="Learning rate (default: 0.005).")
    parser.add_argument("--beta1",         type=float, default=0.80,   help="Adam beta1 (default: 0.80).")
    parser.add_argument("--beta2",         type=float, default=0.88,   help="Adam beta2 (default: 0.88).")
    parser.add_argument("--reg_adam",      type=float, default=1e-8,   help="Adam epsilon (default: 1e-8).")
    parser.add_argument("--lr_decay",      type=float, default=0.5,    help="Learning rate decay factor (default: 0.5).")
    parser.add_argument("--min_lr",        type=float, default=1e-4,   help="Minimum learning rate (default: 1e-4).")
    parser.add_argument("--patience_adam", type=int,   default=3,      help="Patience for lr reduction (default: 3).")
    parser.add_argument("--tol_adam",      type=float, default=0.1,    help="Convergence tolerance (default: 0.1).")
    parser.add_argument("--max_iter",      type=int,   default=10000,  help="Maximum Adam-EM iterations (default: 10000).")
    parser.add_argument("--check",         type=int,   default=5,      help="Log-likelihood check frequency (default: 5).")

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument("-s", "--seed",    type=int,   default=42,     help="Random seed (default: 42).")
    parser.add_argument("-t", "--threads", type=int,   default=1,      help="Number of CPU threads (default: 1).")
    parser.add_argument("--chunk_size",    type=int,   default=4096,   help="SNP chunk size for I/O (default: 4096).")
    parser.add_argument("--device",        type=str,   default="cpu",  help="Computation device: cpu, cuda, or mps (default: cpu).")
    parser.add_argument("--no_freqs",      action="store_true",        help="Do not save the P matrix.")

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
    parser.add_argument("--plot", nargs="*", help="Generate a plot after training. Optional: [format] [dpi].")

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

    assert args.k >= 2, "K must be at least 2."

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
    """
    import torch

    log.info("\n    ADAMIXTURE — Supervised Mode\n")
    arg_list = tuple(sys.argv)
    args = parse_args(arg_list[1:])

    K = args.k

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

    from pathlib import Path
    from .src import utils
    from .src.supervised import (
        init_p_supervised, init_q_supervised,
        optimize_supervised, optimize_supervised_gpu,
    )

    device_str = args.device
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
    log.info(f"    Supervision from level {args.level} ({sup_path.name}): {len(y)} samples.")
    log.info(f"    Detected populations: {list(pop_to_int.keys())}")

    if y.max() > K:
        log.error(
            f"    Error: Assignment file contains {len(pop_to_int)} distinct populations "
            f"but K={K}. Add more populations or increase K."
        )
        sys.exit(1)

    n_labeled = int((y > 0).sum())
    log.info(f"    Labeled samples: {n_labeled}/{len(y)}\n")

    # ── Load genotype data ────────────────────────────────────────────────────
    G, N, M = utils.read_data(args.data_path, packed=False, chunk_size=args.chunk_size)

    if N != len(y):
        log.error(
            f"    Error: Sample count mismatch — genotype data has {N} samples "
            f"but labels file has {len(y)}."
        )
        sys.exit(1)

    # ── Initialise P and Q ────────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)

    log.info("    Initialising P from labeled genotype frequencies...")
    P = init_p_supervised(G, y, K, M)

    log.info("    Initialising Q (one-hot for labeled, random for unlabeled)...")
    Q = rng.random(size=(N, K)).astype(np.float64)
    Q /= Q.sum(axis=1, keepdims=True)
    init_q_supervised(Q, y, K)

    log.info("    Running supervised Adam-EM...\n")

    # ── Run supervised optimisation ──────────────────────────────────────────────
    if use_gpu:
        import torch
        device_obj = torch.device(device_str)
        threads_per_block = utils.get_tuning_params(device_obj)
        utils.load_extensions(device_obj)
        G_t = torch.from_numpy(G) if not isinstance(G, torch.Tensor) else G
        P_t = torch.tensor(P, dtype=utils.get_dtype(device_obj), device=device_obj)
        Q_t = torch.tensor(Q, dtype=utils.get_dtype(device_obj), device=device_obj)
        G_t = utils.manage_gpu_memory(G_t, device_obj, M, N, K, args.chunk_size)
        P_gpu, Q_gpu = optimize_supervised_gpu(
            G=G_t, P=P_t, Q=Q_t, y=y,
            lr=args.lr, beta1=args.beta1, beta2=args.beta2, reg_adam=args.reg_adam,
            max_iter=args.max_iter, check=args.check, M=M,
            lr_decay=args.lr_decay, min_lr=args.min_lr,
            patience_adam=args.patience_adam, tol_adam=args.tol_adam,
            device=device_obj, chunk_size=args.chunk_size, threads_per_block=threads_per_block,
        )
        P_opt = P_gpu.cpu().numpy()
        Q_opt = Q_gpu.cpu().numpy()
    else:
        P_opt, Q_opt = optimize_supervised(
            G=G, P=P, Q=Q, y=y,
            lr=args.lr, beta1=args.beta1, beta2=args.beta2, reg_adam=args.reg_adam,
            max_iter=args.max_iter, check=args.check, K=K, M=M, N=N,
            lr_decay=args.lr_decay, min_lr=args.min_lr,
            patience_adam=args.patience_adam, tol_adam=args.tol_adam,
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

        def _load(path_str: str) -> list[str] | None:
            p = Path(path_str)
            return _parse_labels_file(p) if p.exists() else None

        labels  = _load(args.labels)
        labels2 = _load(args.labels2) if args.labels2 else None
        labels3 = _load(args.labels3) if args.labels3 else None
        colors  = _load(args.colors)  if args.colors  else None

        plot_path = out_path / f"{args.name}.{K}.{args.plot_format}"
        log.info(f"    Generating plot: {plot_path}")
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
