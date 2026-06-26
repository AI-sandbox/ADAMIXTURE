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
            original=args.original,
        )

        trained: dict[int, tuple] = {}
        trained_plot: dict[int, tuple] = {}
        previous_Q = None

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
                original=args.original, rtol=args.rtol, Q_hist=args.Q_hist,
            )

            P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else P
            Q_np = Q.cpu().numpy() if isinstance(Q, torch.Tensor) else Q

            if previous_Q is not None:
                from .plot import align_clusters_greedy
                perm = align_clusters_greedy(previous_Q, Q_np)
                Q_np = Q_np[:, perm]
                P_np = P_np[:, perm]

            previous_Q = Q_np
            trained_plot[K] = (P_np, Q_np)

            utils.write_outputs(Q_np, args.name, K, args.save_dir,
                                P=None if args.no_freqs else P_np)

            if args.plot is not None:
                utils.plot_single_k(args, K, Q_np)

            if args.cv:
                trained[K] = (P, Q)

            log.info(f"\n    K={K} completed in {time.time() - t_k:.2f} seconds.")

        # Combined single plot for all K sweep values
        if hasattr(args, 'plot_single') and args.plot_single is not None and len(k_values) > 1:
            utils.plot_combined(args, k_values, trained_plot)

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
