import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .src.plot import align_clusters_greedy

# Global logging configuration
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def parse_filemap(filemap_path: str) -> list[dict]:
    """
    Description:
    Parses a tab-delimited filemap containing run definitions for multi-run plotting.
    Each line must have: run_id, K, path_to_Q_matrix.

    Args:
        filemap_path (str): Path to the filemap file.

    Returns:
        list[dict]: List of dicts with keys 'id' (str), 'K' (int), 'path' (str).
    """
    runs = []
    filemap_path_obj = Path(filemap_path)
    if not filemap_path_obj.exists():
        log.error(f"    Error: Filemap not found: {filemap_path}")
        sys.exit(1)

    filemap_dir = filemap_path_obj.parent
    with open(filemap_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                run_id = parts[0]
                if not any(c.isalpha() for c in run_id):
                    log.error(f"    Error: Run ID '{run_id}' must contain at least one letter.")
                    sys.exit(1)
                if '#' in run_id or '.' in run_id:
                    log.error(f"    Error: Run ID '{run_id}' cannot contain '#' or '.'.")
                    sys.exit(1)

                try:
                    K = int(parts[1])
                except ValueError:
                    log.error(f"    Error: K value '{parts[1]}' must be an integer.")
                    sys.exit(1)

                q_path = parts[2]
                full_q_path = filemap_dir / q_path
                runs.append({'id': run_id, 'K': K, 'path': str(full_q_path)})
            else:
                log.error(f"    Error: Filemap line must be tab-delimited with 3 columns: {line}")
                sys.exit(1)
    return runs


def load_labels(labels_path: str | None) -> list[str] | None:
    """
    Description:
    Loads population labels from a file (one label per line).

    Args:
        labels_path (str | None): Path to the labels file, or None to skip.

    Returns:
        list[str] | None: List of label strings, or None if unavailable.
    """
    if not labels_path:
        return None
    labels_path_obj = Path(labels_path)
    if not labels_path_obj.exists():
        return None
    with open(labels_path) as f:
        labels = [line.strip() for line in f if line.strip()]
    return labels


def _draw_brackets(ax, items: list[dict], y_bracket: float, fontsize: int = 6) -> None:
    """
    Description:
    Draws elegant bracket annotations below the x-axis for a given grouping level.

    Args:
        ax: Matplotlib axes object.
        items (list[dict]): List of dicts with 'name', 'start', 'end' keys (in sample-index space).
        y_bracket (float): Y position in axes-transform space for the bracket line.
        fontsize (int): Font size for the bracket labels.
    """
    trans = ax.get_xaxis_transform()
    y_text = y_bracket - 0.05

    for item in items:
        x0, x1 = item['start'], item['end']
        gap = min((x1 - x0) * 0.01, 10)
        x0_br = x0 + gap if (x0 + gap) < x1 else x0
        x1_br = x1 - gap if (x1 - gap) > x0 else x1

        # Horizontal line
        ax.plot([x0_br, x1_br], [y_bracket, y_bracket],
                color='#222222', lw=0.8, transform=trans, clip_on=False)
        # Vertical ticks
        ax.plot([x0_br, x0_br], [y_bracket, y_bracket + 0.08],
                color='#222222', lw=0.8, transform=trans, clip_on=False)
        ax.plot([x1_br, x1_br], [y_bracket, y_bracket + 0.08],
                color='#222222', lw=0.8, transform=trans, clip_on=False)
        # Label
        ax.text((x0 + x1) / 2, y_text, str(item['name']).title(),
                ha='center', va='top', rotation=90, fontsize=fontsize,
                color='#222222', transform=trans, clip_on=False)


def main() -> None:
    """
    Description:
    Entry point for the ADAMIXTURE multi-run plotting CLI. Loads Q matrices
    from a filemap, optionally aligns clusters across runs of the same K,
    and produces a combined stacked bar chart plot.

    Supports up to three levels of hierarchical population labels via
    --labels (level 1 / finest), --labels2 (level 2), and --labels3 (level 3 / coarsest).
    When multiple levels are provided, bracket annotations are drawn under the
    bottom subplot to represent each grouping tier.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='ADAMIXTURE multi-run plotting tool.')
    parser.add_argument('-m', '--filemap', required=True, help='Path to filemap (run_id\\tK\\tpath)')
    parser.add_argument('-l', '--labels', help='Path to population labels file (level 1, one per sample)')
    parser.add_argument('--labels2', help='Path to level-2 population grouping file (one per sample)')
    parser.add_argument('--labels3', help='Path to level-3 population grouping file (one per sample)')
    parser.add_argument('-c', '--colors', help='Path to custom colors file (one color per line)')
    parser.add_argument('-o', '--output', default='adamixture_plots.png', help='Output file name')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for the output plot')
    parser.add_argument('--format', type=str, choices=['png', 'pdf', 'jpg'], default='png', help='Output format')

    args = parser.parse_args()

    runs_info = parse_filemap(args.filemap)
    if not runs_info:
        log.error("    Error: No valid runs found in filemap.")
        sys.exit(1)

    labels = load_labels(args.labels)
    labels2 = load_labels(args.labels2)
    labels3 = load_labels(args.labels3)

    # Validate label lengths consistency
    ref_len = None
    for name, lbl in [('--labels', labels), ('--labels2', labels2), ('--labels3', labels3)]:
        if lbl is not None:
            if ref_len is None:
                ref_len = len(lbl)
            elif len(lbl) != ref_len:
                log.error(f"    Error: {name} has {len(lbl)} entries but expected {ref_len}.")
                sys.exit(1)

    # Load all Q matrices
    all_qs: list[dict] = []
    for run in runs_info:
        Q = np.loadtxt(run['path'])
        all_qs.append({'id': run['id'], 'K': run['K'], 'Q': Q})

    custom_colors = None
    max_k = max(run['K'] for run in all_qs)
    if args.colors:
        colors_path = Path(args.colors)
        if colors_path.exists():
            with open(colors_path) as f:
                custom_colors = [line.strip() for line in f if line.strip()]
            if len(custom_colors) < max_k:
                log.error(f"    Error: Provided colors file has {len(custom_colors)} colors, but highest K in filemap is {max_k}.")
                sys.exit(1)

    num_runs = len(all_qs)
    all_qs.sort(key=lambda x: x['K'])

    # Align clusters across runs of the same K
    for i in range(1, num_runs):
        if all_qs[i]['K'] == all_qs[i - 1]['K']:
            ref_Q = all_qs[i - 1]['Q']
            curr_Q = all_qs[i]['Q']
            perm = align_clusters_greedy(ref_Q, curr_Q)
            all_qs[i]['Q'] = curr_Q[:, perm]

    fig, axes = plt.subplots(nrows=num_runs, ncols=1, figsize=(15, 3 * num_runs), squeeze=False)
    axes = axes.flatten()

    # ── Pre-compute the sorted order once (from the first run / first Q) ──────
    # All runs share the same samples so we derive a single sort order.
    first_Q = all_qs[0]['Q']
    n_samples_global = first_Q.shape[0]

    # Build sorted indices based on the available label hierarchy
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

    # Derive sorted label lists and pre-compute boundary/bracket data
    labels_sorted = [labels[i] for i in sort_idx] if labels is not None and len(labels) == n_samples_global else None
    labels2_sorted = [labels2[i] for i in sort_idx] if labels2 is not None else None
    labels3_sorted = [labels3[i] for i in sort_idx] if labels3 is not None else None

    # Compute population boundaries and tick positions from sorted labels (level 1)
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

    # Build bracket items for level 2
    i2_items: list[dict] = []
    if labels2_sorted is not None:
        current_name = labels2_sorted[0]
        seg_start = 0
        for idx, name in enumerate(labels2_sorted):
            if name != current_name:
                i2_items.append({'name': current_name, 'start': seg_start, 'end': idx})
                seg_start = idx
                current_name = name
        i2_items.append({'name': current_name, 'start': seg_start, 'end': n_samples_global})

    # Build bracket items for level 3
    i3_items: list[dict] = []
    if labels3_sorted is not None:
        current_name = labels3_sorted[0]
        seg_start = 0
        for idx, name in enumerate(labels3_sorted):
            if name != current_name:
                i3_items.append({'name': current_name, 'start': seg_start, 'end': idx})
                seg_start = idx
                current_name = name
        i3_items.append({'name': current_name, 'start': seg_start, 'end': n_samples_global})

    # ── Dynamic bottom margin based on longest labels ─────────────────────────
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

    # ── Draw each subplot ─────────────────────────────────────────────────────
    for i, run in enumerate(all_qs):
        ax = axes[i]
        Q = run['Q'][sort_idx]
        n_samples, K = Q.shape

        if custom_colors is not None and len(custom_colors) >= K:
            colors = custom_colors[:K]
        else:
            cmap = plt.colormaps.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, K))

        Q_cum = np.cumsum(Q, axis=1)
        x = np.arange(n_samples)
        zeros = np.zeros(n_samples)

        for j in range(K):
            lower = Q_cum[:, j - 1] if j > 0 else zeros
            upper = Q_cum[:, j]
            ax.fill_between(x, lower, upper, facecolor=colors[j], edgecolor='none', linewidth=0, rasterized=True)

        # Draw population boundaries (level 1)
        for boundary in pop_boundaries:
            ax.axvline(x=boundary, color='black', linestyle='-', linewidth=0.5)

        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{run['id']}\n(K={K})", rotation=0, ha='right', va='center')

        is_bottom = (i == num_runs - 1)

        if is_bottom and labels_sorted is not None:
            ax.set_xticks(pop_tick_positions)
            ax.set_xticklabels(pop_tick_labels, rotation=90, ha='center', fontsize=6)
            ax.tick_params(axis='x', which='both', length=0, pad=5)

            # Dynamic y positions for bracket tiers
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

    plt.subplots_adjust(bottom=bottom_margin, hspace=0.3)

    output_path = Path(args.output)
    if output_path.suffix != f".{args.format}":
        output_path = output_path.with_suffix(f".{args.format}")

    fig.savefig(output_path, dpi=args.dpi, format=args.format, bbox_inches='tight')
    log.info(f"    Multi-run plot saved to: {output_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()
