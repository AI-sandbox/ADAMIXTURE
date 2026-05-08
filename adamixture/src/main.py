import argparse
import logging
import sys
import time
from argparse import ArgumentError, ArgumentTypeError
from pathlib import Path

import numpy as np
import torch

from . import utils
from .adamixture import setup, train_k
from .plot import plot_q_matrix

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def main(args: argparse.Namespace, t0: float) -> int:
    """
    Description:
    The core training loop coordinator. It reads data once, performs one-time
    initialisation (device, frequencies, SVD) with K_max, then iterates over
    the requested K values for ALS + Adam-EM training.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        t0 (float): Program start time for total execution measurement.

    Returns:
        int: Exit code (0 for success).
    """
    try:
        if args.min_k is not None and args.max_k is not None:
            k_values = list(range(args.min_k, args.max_k + 1))
            log.info(f"\n    Running from {args.min_k} to {args.max_k}.\n")
        else:
            k_values = [int(args.k)]

        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        G, N, M = utils.read_data(args.data_path, packed='cuda' in args.device, chunk_size=args.chunk_size)

        K_max = max(k_values)
        device_obj, threads_per_block, f, U, S, V, G = setup(
            G, N, M, K_max,
            int(args.seed), int(args.power), float(args.tol_svd),
            int(args.chunk_size), args.device,
        )

        trained: dict[int, tuple] = {}

        for K in k_values:
            log.info(f"\n    Running on K = {K}.\n")
            t_k = time.time()

            P, Q = train_k(
                G, N, M, K, U, S, V, f,
                int(args.seed), float(args.lr), float(args.beta1), float(args.beta2),
                float(args.reg_adam), int(args.max_iter), int(args.check),
                int(args.max_als), float(args.tol_als),
                float(args.lr_decay), float(args.min_lr), int(args.chunk_size),
                int(args.patience_adam), float(args.tol_adam),
                device_obj, threads_per_block,
            )

            P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else P
            Q_np = Q.cpu().numpy() if isinstance(Q, torch.Tensor) else Q

            utils.write_outputs(Q_np, args.name, K, args.save_dir,
                                P=None if args.no_freqs else P_np)

            if args.plot is not None:
                labels = None
                if args.labels and Path(args.labels).exists():
                    with open(args.labels) as fh:
                        labels = [line.strip() for line in fh if line.strip()]

                labels2 = None
                if hasattr(args, 'labels2') and args.labels2 and Path(args.labels2).exists():
                    with open(args.labels2) as fh:
                        labels2 = [line.strip() for line in fh if line.strip()]

                labels3 = None
                if hasattr(args, 'labels3') and args.labels3 and Path(args.labels3).exists():
                    with open(args.labels3) as fh:
                        labels3 = [line.strip() for line in fh if line.strip()]

                colors = None
                if args.colors and Path(args.colors).exists():
                    with open(args.colors) as fh:
                        colors = [line.strip() for line in fh if line.strip()]
                    if len(colors) != K:
                        log.warning(f"    Number of colors in {args.colors} ({len(colors)}) does not match K={K}. Using default colors.")
                        colors = None

                plot_path = Path(args.save_dir) / f"{args.name}.{K}.{args.plot_format}"
                log.info(f"    Generating plot: {plot_path}")
                plot_q_matrix(Q_np, plot_path, dpi=args.plot_dpi, format=args.plot_format,
                              labels=labels, labels2=labels2, labels3=labels3, custom_colors=colors)

            if args.cv:
                trained[K] = (P, Q)

            log.info(f"\n    K={K} completed in {time.time() - t_k:.2f} seconds.")

        del U, S, V, f

        # CROSS-VALIDATION (after all training):
        cv_results: dict[int, float] = {}
        if args.cv and trained:
            use_gpu_cv = isinstance(G, torch.Tensor) and G.device.type == 'cuda'

            if use_gpu_cv:
                from .cv import run_cross_validation_gpu
                for K, (P, Q) in sorted(trained.items()):
                    log.info(f"\n    Running {int(args.cv)}-fold CV on genotype entries for K={K}...")
                    cv_results[K] = run_cross_validation_gpu(args, G, N, M, P, Q, device_obj, threads_per_block)
            else:
                if isinstance(G, torch.Tensor):
                    G = np.ascontiguousarray(G.numpy())
                from .cv import run_cross_validation
                for K, (P, Q) in sorted(trained.items()):
                    log.info(f"\n    Running {int(args.cv)}-fold CV on genotype entries for K={K}...")
                    cv_results[K] = run_cross_validation(args, G, N, M, K, P, Q)

            log.info("")
            log.info("    ---- Cross-validation summary ----")
            for k_val, idx in sorted(cv_results.items()):
                log.info(f"    K={k_val}: CV index = {idx:.4f}")
            log.info("    ----------------------------------")

        t1 = time.time()
        log.info(f"\n    Total elapsed time: {t1 - t0:.2f} seconds.\n")

        logging.shutdown()
        return 0

    except (ArgumentError, ArgumentTypeError) as e:
        log.error(f"    Error parsing arguments: {e}")
        logging.shutdown()
        return 1

    except Exception as e:
        log.error(f"    Unexpected error: {e}")
        logging.shutdown()
        return 1
