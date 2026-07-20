import argparse
import gc
import logging
import sys
import time
from argparse import ArgumentError, ArgumentTypeError
from pathlib import Path

import numpy as np
import torch

from . import utils
from .adamixture import setup, train_k, initialize_k

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

        n_inits = int(args.n_inits)
        K_max = max(k_values)

        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

        training_packed = args.device in ("gpu", "cuda", "mps") or 'cuda' in args.device
        G, N, M = utils.read_data(
            args.data_path,
            packed=training_packed,
            chunk_size=args.chunk_size,
            chrom_mode=args.chrom_mode,
            autosomes=args.autosomes,
        )

        trained: dict[int, tuple] = {}
        trained_plot: dict[int, tuple] = {}
        previous_Q = None

        if n_inits == 1:
            device_obj, threads_per_block, f, U, S, V, G = setup(
                G, N, M, K_max,
                int(args.seed), int(args.power), float(args.tol_svd),
                int(args.chunk_size), args.device,
                original=(args.algorithm == 'brqn'), init_original=args.init,
                q_hist=args.Q_hist,
            )

            for K in k_values:
                log.info(f"\n    Running on K = {K}.\n")
                t_k = time.time()

                P, Q = train_k(
                    G, N, M, K, U, S, V, f,
                    int(args.seed), float(args.lr), float(args.beta1), float(args.beta2),
                    float(args.reg_adam), int(args.max_iter), int(args.check),
                    int(args.max_als), float(args.tol_als),
                    float(args.lr_decay), float(args.min_lr), int(args.chunk_size),
                    int(args.patience), float(args.tol),
                    device_obj, threads_per_block,
                    original=(args.algorithm == 'brqn'), rtol=float(args.tol), Q_hist=args.Q_hist,
                    init_original=args.init,
                    em_init_steps=int(args.em_init_steps),
                )

                P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else P
                Q_np = Q.cpu().numpy() if isinstance(Q, torch.Tensor) else Q

                if previous_Q is not None:
                    from .plot import align_clusters_clumppling
                    perm = align_clusters_clumppling(previous_Q, Q_np)
                    Q_np = Q_np[:, perm]
                    P_np = P_np[:, perm]

                previous_Q = Q_np
                trained_plot[K] = (P_np, Q_np)

                utils.write_outputs(Q_np, args.name, K, args.save_dir,
                                    P=None if args.no_freqs else P_np)

                should_plot_k = (len(k_values) == 1 and args.plot is not None) or (len(k_values) > 1 and getattr(args, 'plot_single', None) is not None)
                if should_plot_k:
                    from .plot import plot_single_k
                    plot_single_k(args, K, Q_np)

                if args.cv:
                    trained[K] = (P, Q)

                log.info(f"\n    K={K} completed in {time.time() - t_k:.2f} seconds.")

            del U, S, V, f

        else:
            device_obj, threads_per_block, f, _, _, _, G = setup(
                G, N, M, K_max,
                int(args.seed), int(args.power), float(args.tol_svd),
                int(args.chunk_size), args.device,
                original=(args.algorithm == 'brqn'), init_original=args.init,
                q_hist=args.Q_hist,
                compute_svd=False,
            )

            for K in k_values:
                log.info(f"\n    Running on K = {K} with {n_inits} initialization(s).\n")
                t_k = time.time()

                best_P, best_Q, best_logl = None, None, -np.inf
                for init_idx in range(n_inits):
                    init_seed = int(args.seed) + init_idx
                    log.info(f"\n    --- Initialization {init_idx + 1}/{n_inits} (seed={init_seed}) ---\n")

                    P, Q, logl = initialize_k(
                        G, N, M, K, f, init_seed,
                        int(args.power), float(args.tol_svd),
                        int(args.max_als), float(args.tol_als),
                        int(args.chunk_size), device_obj, threads_per_block,
                        original=(args.algorithm == 'brqn'),
                        init_original=args.init,
                        em_init_steps=int(args.em_init_steps),
                    )

                    if logl > best_logl:
                        best_logl = logl
                        best_P, best_Q = P, Q

                log.info(f"\n    Best initialization log-likelihood for K={K}: {best_logl:.1f}")
                P, Q = train_k(
                    G, N, M, K, None, None, None, f,
                    int(args.seed), float(args.lr), float(args.beta1), float(args.beta2),
                    float(args.reg_adam), int(args.max_iter), int(args.check),
                    int(args.max_als), float(args.tol_als),
                    float(args.lr_decay), float(args.min_lr), int(args.chunk_size),
                    int(args.patience), float(args.tol),
                    device_obj, threads_per_block,
                    original=(args.algorithm == 'brqn'), rtol=float(args.tol), Q_hist=args.Q_hist,
                    init_original=args.init,
                    em_init_steps=int(args.em_init_steps),
                    P_init=best_P, Q_init=best_Q,
                )

                P_np = P.cpu().numpy() if isinstance(P, torch.Tensor) else P
                Q_np = Q.cpu().numpy() if isinstance(Q, torch.Tensor) else Q

                if previous_Q is not None:
                    from .plot import align_clusters_clumppling
                    perm = align_clusters_clumppling(previous_Q, Q_np)
                    Q_np = Q_np[:, perm]
                    P_np = P_np[:, perm]

                previous_Q = Q_np
                trained_plot[K] = (P_np, Q_np)

                utils.write_outputs(Q_np, args.name, K, args.save_dir,
                                    P=None if args.no_freqs else P_np)

                should_plot_k = (len(k_values) == 1 and args.plot is not None) or (len(k_values) > 1 and getattr(args, 'plot_single', None) is not None)
                if should_plot_k:
                    from .plot import plot_single_k
                    plot_single_k(args, K, Q_np)

                if args.cv:
                    trained[K] = (P, Q)

                log.info(f"\n    K={K} completed in {time.time() - t_k:.2f} seconds.")

            del f

        # Combined single plot for all K sweep values
        if len(k_values) > 1 and getattr(args, 'plot', None) is not None:
            from .plot import plot_combined
            plot_combined(args, k_values, trained_plot)

        # CROSS-VALIDATION (after all training):
        cv_results: dict[int, float] = {}
        if args.cv and trained:
            from .cv import run_cross_validation

            if training_packed:
                del G
                gc.collect()
                if device_obj.type == 'cuda':
                    torch.cuda.empty_cache()

                previous_disable = logging.root.manager.disable
                logging.disable(logging.CRITICAL)
                try:
                    G_cv, N_cv, M_cv = utils.read_data(
                        args.data_path,
                        packed=False,
                        chunk_size=args.chunk_size,
                        chrom_mode=args.chrom_mode,
                        autosomes=args.autosomes,
                        verbose=False,
                    )
                finally:
                    logging.disable(previous_disable)
                if N_cv != N or M_cv != M:
                    raise ValueError(
                        f"CV reread shape mismatch: training data was N={N}, M={M}; "
                        f"CV data is N={N_cv}, M={M_cv}."
                    )
            else:
                G_cv = G if isinstance(G, np.ndarray) else np.ascontiguousarray(G.detach().cpu().numpy(), dtype=np.uint8)

            for K, (P, Q) in sorted(trained.items()):
                log.info(f"\n    Running {int(args.cv)}-fold CV on genotype entries for K={K}...")
                cv_results[K] = run_cross_validation(args, G_cv, N, M, K, P, Q)
            del G_cv
            gc.collect()

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
