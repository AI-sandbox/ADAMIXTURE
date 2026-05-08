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
    Parses command-line arguments for the adamixture-project command.
    Projection mode fixes a pre-trained P matrix (allele frequencies) and
    only estimates Q (ancestry proportions) for a set of target samples.

    Args:
        argv (list[str]): Command-line arguments (excluding the program name).

    Returns:
        configargparse.Namespace: Parsed arguments.
    """
    parser = configargparse.ArgumentParser(
        prog="adamixture-project",
        description=(
            "ADAMIXTURE projection mode. "
            "Estimates ancestry proportions Q for target samples using a "
            "fixed, pre-trained allele-frequency matrix P."
        ),
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--data_path", required=True, type=str,
        help="Path to the target genotype data (BED, VCF or PGEN).",
    )
    parser.add_argument(
        "--p_path", required=True, type=str,
        help=(
            "Path to the pre-trained P matrix (.P file). "
            "Must be a whitespace-delimited file with M rows and K columns."
        ),
    )
    parser.add_argument(
        "--save_dir", required=True, type=str,
        help="Directory where the output Q file will be saved.",
    )
    parser.add_argument(
        "--name", required=True, type=str,
        help="Experiment/run name used as prefix for output files.",
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

    # ── Plotting ──────────────────────────────────────────────────────────────
    parser.add_argument("--plot",    nargs="*",  help="Generate a plot after projection. Optional: [format] [dpi].")
    parser.add_argument("--labels",  type=str,   help="Population labels file (one per sample).")
    parser.add_argument("--labels2", type=str,   help="Level-2 grouping labels file (one per sample).")
    parser.add_argument("--labels3", type=str,   help="Level-3 grouping labels file (one per sample).")
    parser.add_argument("--colors",  type=str,   help="Custom colors file (one per line).")

    args = parser.parse_args(argv)

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

    return args


def main() -> None:
    """
    Description:
    Entry point for the ``adamixture-project`` command.

    Loads a pre-trained P matrix and a target genotype dataset, then runs
    the Adam-EM projection loop (Q-only updates) to estimate ancestry
    proportions for the target samples.
    """
    import torch

    log.info("\n    ADAMIXTURE — Projection Mode\n")
    arg_list = tuple(sys.argv)
    args = parse_args(arg_list[1:])

    # Thread control
    th = str(args.threads)
    for env_var in [
        "MKL_NUM_THREADS", "MKL_MAX_THREADS", "OMP_NUM_THREADS", "OMP_MAX_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS", "OPENBLAS_NUM_THREADS", "OPENBLAS_MAX_THREADS",
    ]:
        os.environ[env_var] = th

    # OS-specific compiler settings
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
    from .src.projection import optimize_projection, optimize_projection_gpu

    device_str = args.device
    use_gpu = device_str in ("cuda", "mps")
    utils.set_seed(args.seed)

    t0 = time.time()

    # ── Load P matrix ─────────────────────────────────────────────────────────
    p_path = Path(args.p_path)
    if not p_path.exists():
        log.error(f"    Error: P matrix file not found: {p_path}")
        sys.exit(1)

    log.info(f"    Loading pre-trained P matrix from: {p_path}")
    P = np.loadtxt(str(p_path), dtype=np.float64)
    if P.ndim == 1:
        P = P.reshape(-1, 1)
    M_p, K = P.shape
    P = P.clip(1e-5, 1.0 - 1e-5)
    log.info(f"    P matrix loaded: {M_p} SNPs, K={K} populations.\n")

    # ── Load target genotype data ─────────────────────────────────────────────
    G, N, M = utils.read_data(args.data_path, packed=False, chunk_size=args.chunk_size)

    if M != M_p:
        log.error(
            f"    Error: SNP count mismatch — genotype data has {M} SNPs "
            f"but P matrix has {M_p} SNPs."
        )
        sys.exit(1)

    log.info(f"    Target samples: {N}.\n")

    # ── Initialise Q randomly ─────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    Q = rng.random(size=(N, K)).astype(np.float64)
    Q /= Q.sum(axis=1, keepdims=True)

    log.info("    Running Adam-EM projection (P fixed)...\n")

    # ── Run projection ────────────────────────────────────────────────────────
    if use_gpu:
        import torch
        device_obj = torch.device(device_str)
        threads_per_block = utils.get_tuning_params(device_obj)
        utils.load_extensions(device_obj)
        G_t = torch.from_numpy(G) if not isinstance(G, torch.Tensor) else G
        P_t = torch.tensor(P, dtype=utils.get_dtype(device_obj), device=device_obj)
        Q_t = torch.tensor(Q, dtype=utils.get_dtype(device_obj), device=device_obj)
        G_t = utils.manage_gpu_memory(G_t, device_obj, M, N, K, args.chunk_size)
        Q_gpu = optimize_projection_gpu(
            G=G_t, P=P_t, Q=Q_t,
            lr=args.lr, beta1=args.beta1, beta2=args.beta2, reg_adam=args.reg_adam,
            max_iter=args.max_iter, check=args.check, M=M,
            lr_decay=args.lr_decay, min_lr=args.min_lr,
            patience_adam=args.patience_adam, tol_adam=args.tol_adam,
            device=device_obj, chunk_size=args.chunk_size, threads_per_block=threads_per_block,
        )
        Q_opt = Q_gpu.cpu().numpy()
    else:
        Q_opt = optimize_projection(
            G=G, P=P, Q=Q,
            lr=args.lr, beta1=args.beta1, beta2=args.beta2, reg_adam=args.reg_adam,
            max_iter=args.max_iter, check=args.check, K=K, M=M, N=N,
            lr_decay=args.lr_decay, min_lr=args.min_lr,
            patience_adam=args.patience_adam, tol_adam=args.tol_adam,
        )

    # ── Save output ───────────────────────────────────────────────────────────
    out_path = Path(args.save_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    q_file = out_path / f"{args.name}.{K}.Q"
    np.savetxt(str(q_file), Q_opt, delimiter=" ", fmt="%.6f")
    log.info(f"    Q matrix saved to: {q_file}")

    # ── Optional plot ─────────────────────────────────────────────────────────
    if args.plot is not None:
        from .src.plot import plot_q_matrix

        def _load(path_str):
            p = Path(path_str)
            return [l.strip() for l in p.open() if l.strip()] if p.exists() else None

        labels  = _load(args.labels)  if args.labels  else None
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
