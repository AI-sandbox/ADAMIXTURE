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

        # Combined single plot for all K sweep values
        if hasattr(args, 'plot_single') and args.plot_single is not None and len(k_values) > 1:
            log.info("\n    Generating combined single plot for all K sweep values...")
            import matplotlib.pyplot as plt
            from .plot import _draw_brackets

            # Load labels if available
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

            colors_list = None
            if args.colors and Path(args.colors).exists():
                with open(args.colors) as fh:
                    colors_list = [line.strip() for line in fh if line.strip()]

            num_runs = len(k_values)
            fig, axes = plt.subplots(nrows=num_runs, ncols=1, figsize=(15, 3 * num_runs), squeeze=False)
            axes = axes.flatten()

            first_K = k_values[0]
            first_Q = trained_plot[first_K][1]
            n_samples_global = first_Q.shape[0]

            if labels is not None and len(labels) == n_samples_global:
                if labels3 is not None and labels2 is not None:
                    sort_idx = np.lexsort((labels, labels2, labels3))
                elif labels2 is not None:
                    sort_idx = np.lexsort((labels, labels2))
                else:
                    dominant_cluster = np.argmax(first_Q, axis=1)
                    sort_idx = np.lexsort((np.max(first_Q, axis=1), dominant_cluster, labels))
            else:
                dominant_cluster = np.argmax(first_Q, axis=1)
                sort_idx = np.lexsort((np.max(first_Q, axis=1), dominant_cluster))

            labels_sorted = [labels[i] for i in sort_idx] if labels is not None and len(labels) == n_samples_global else None
            labels2_sorted = [labels2[i] for i in sort_idx] if labels2 is not None else None
            labels3_sorted = [labels3[i] for i in sort_idx] if labels3 is not None else None

            pop_boundaries: list[int] = []
            pop_tick_positions: list[float] = []
            pop_tick_labels: list[str] = []
            if labels_sorted is not None:
                current_label = labels_sorted[0]
                start_idx = 0
                for idx, lbl in enumerate(labels_sorted):
                    if lbl != current_label:
                        pop_boundaries.append(idx)
                        pop_tick_positions.append((start_idx + idx) / 2)
                        pop_tick_labels.append(str(current_label).title())
                        start_idx = idx
                        current_label = lbl
                pop_tick_positions.append((start_idx + n_samples_global) / 2)
                pop_tick_labels.append(str(current_label).title())

            def _build_brackets_list(sorted_lbls):
                if sorted_lbls is None:
                    return []
                items = []
                current_name = sorted_lbls[0]
                seg_start = 0
                for idx, name in enumerate(sorted_lbls):
                    if name != current_name:
                        items.append({'name': current_name, 'start': seg_start, 'end': idx})
                        seg_start = idx
                        current_name = name
                items.append({'name': current_name, 'start': seg_start, 'end': n_samples_global})
                return items

            i2_items = _build_brackets_list(labels2_sorted)
            i3_items = _build_brackets_list(labels3_sorted)

            max_l1_len = max((len(str(l)) for l in pop_tick_labels), default=0)
            max_l2_len = max((len(item['name']) for item in i2_items), default=0)
            max_l3_len = max((len(item['name']) for item in i3_items), default=0)

            bottom_margin = 0.15
            if labels_sorted:
                bottom_margin += max_l1_len * 0.012
            if i2_items:
                bottom_margin += 0.15 + max_l2_len * 0.012
            if i3_items:
                bottom_margin += 0.15 + max_l3_len * 0.012
            bottom_margin = max(0.15, min(bottom_margin, 0.80))

            for i, K_val in enumerate(k_values):
                ax = axes[i]
                P_np, Q_np = trained_plot[K_val]
                Q_sorted = Q_np[sort_idx]
                n_samples, K_curr = Q_sorted.shape

                if colors_list is not None and len(colors_list) >= K_curr:
                    colors = colors_list[:K_curr]
                else:
                    cmap = plt.colormaps.get_cmap('tab20')
                    colors = cmap(np.arange(K_curr) % 20)

                Q_cum = np.cumsum(Q_sorted, axis=1)
                x = np.arange(n_samples)
                zeros = np.zeros(n_samples)

                for j in range(K_curr):
                    lower = Q_cum[:, j - 1] if j > 0 else zeros
                    upper = Q_cum[:, j]
                    ax.fill_between(x, lower, upper, facecolor=colors[j], edgecolor='none', linewidth=0, rasterized=True)

                for boundary in pop_boundaries:
                    ax.axvline(x=boundary, color='black', linestyle='-', linewidth=0.5)

                ax.set_xlim(0, n_samples)
                ax.set_ylim(0, 1)
                ax.set_ylabel(f"K={K_curr}", rotation=0, ha='right', va='center', labelpad=10)

                is_bottom = (i == num_runs - 1)
                if is_bottom and labels_sorted is not None:
                    ax.set_xticks(pop_tick_positions)
                    ax.set_xticklabels(pop_tick_labels, rotation=90, ha='center', fontsize=6)
                    ax.tick_params(axis='x', which='both', length=0, pad=5)

                    y_i2 = -0.3 - max_l1_len * 0.015
                    y_i3 = y_i2 - 0.15 - max_l2_len * 0.015

                    if i2_items:
                        _draw_brackets(ax, i2_items, y_bracket=y_i2, fontsize=6)
                    if i3_items:
                        _draw_brackets(ax, i3_items, y_bracket=y_i3, fontsize=6)
                else:
                    ax.set_xticks([])

                if is_bottom and labels_sorted is None:
                    ax.set_xlabel("Samples")

            plt.subplots_adjust(bottom=bottom_margin, hspace=0.25)
            fig.suptitle(f"ADAMIXTURE (K={k_values[0]} to K={k_values[-1]})", fontsize=12, y=0.95)

            single_plot_path = Path(args.save_dir) / f"{args.name}.{k_values[0]}_{k_values[-1]}.{args.plot_format}"
            fig.savefig(single_plot_path, dpi=args.plot_dpi, format=args.plot_format, bbox_inches='tight')
            log.info(f"    Combined single plot saved to: {single_plot_path}")
            plt.close(fig)

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
