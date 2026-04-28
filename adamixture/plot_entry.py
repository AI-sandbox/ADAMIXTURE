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

def main() -> None:
    """
    Description:
    Entry point for the ADAMIXTURE multi-run plotting CLI. Loads Q matrices
    from a filemap, optionally aligns clusters across runs of the same K,
    and produces a combined stacked bar chart plot.

    Args:
        None

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='ADAMIXTURE multi-run plotting tool.')
    parser.add_argument('-m', '--filemap', required=True, help='Path to filemap (run_id\tK\tpath)')
    parser.add_argument('-l', '--labels', help='Path to population labels file')
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
        if all_qs[i]['K'] == all_qs[i-1]['K']:
            ref_Q = all_qs[i-1]['Q']
            curr_Q = all_qs[i]['Q']
            perm = align_clusters_greedy(ref_Q, curr_Q)
            all_qs[i]['Q'] = curr_Q[:, perm]

    fig, axes = plt.subplots(nrows=num_runs, ncols=1, figsize=(15, 3 * num_runs), squeeze=False)
    axes = axes.flatten()

    for i, run in enumerate(all_qs):
        ax = axes[i]
        Q = run['Q']
        n_samples, K = Q.shape

        if labels is not None:
            if len(labels) == n_samples:
                dominant_cluster = np.argmax(Q, axis=1)
                sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster, labels))
                Q_plot = Q[sort_idx]
                labels_plot = [labels[idx] for idx in sort_idx]
            else:
                log.warning(f"    Warning: Label count mismatch for run {run['id']}. Skipping sort.")
                Q_plot = Q
                labels_plot = None
        else:
            dominant_cluster = np.argmax(Q, axis=1)
            sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster))
            Q_plot = Q[sort_idx]
            labels_plot = None

        if custom_colors is not None and len(custom_colors) >= K:
            colors = custom_colors[:K]
        else:
            cmap = plt.colormaps.get_cmap('tab20')
            colors = cmap(np.linspace(0, 1, K))

        Q_cum = np.cumsum(Q_plot, axis=1)
        x = np.arange(n_samples)
        zeros = np.zeros(n_samples)

        for j in range(K):
            lower = Q_cum[:, j - 1] if j > 0 else zeros
            upper = Q_cum[:, j]
            ax.fill_between(x, lower, upper, facecolor=colors[j], edgecolor='none', linewidth=0, rasterized=True)

        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{run['id']}\n(K={K})", rotation=0, ha='right', va='center')

        if labels_plot is not None:
            current_label = labels_plot[0]
            start_idx = 0
            label_positions = []
            unique_labels = []
            for idx, lbl in enumerate(labels_plot):
                if lbl != current_label:
                    unique_labels.append(str(current_label))
                    label_positions.append((start_idx + idx) / 2)
                    ax.axvline(x=idx, color='black', linestyle='-', linewidth=0.5)
                    start_idx = idx
                    current_label = lbl
            unique_labels.append(str(current_label))
            label_positions.append((start_idx + n_samples) / 2)

            if i == num_runs - 1:
                ax.set_xticks(label_positions)
                ax.set_xticklabels(unique_labels, rotation=45, ha='right')
            else:
                ax.set_xticks([])
        else:
            ax.set_xticks([])

        if i == num_runs - 1 and labels_plot is None:
            ax.set_xlabel("Samples")

    plt.tight_layout()
    output_path = Path(args.output)
    if output_path.suffix != f".{args.format}":
        output_path = output_path.with_suffix(f".{args.format}")

    fig.savefig(output_path, dpi=args.dpi, format=args.format, bbox_inches='tight')
    log.info(f"    Multi-run plot saved to: {output_path}")
    plt.close(fig)

if __name__ == '__main__':
    main()
