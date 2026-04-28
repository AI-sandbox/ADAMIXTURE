import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def plot_q_matrix(Q: np.ndarray, output_path: str | Path, dpi: int = 300, format: str = 'png',
                  labels: list | None = None, custom_colors: list | None = None) -> None:
    """
    Description:
    Plots the Q matrix (ancestry proportions) as a stacked bar chart.
    Optimized for large datasets by using rasterization.
    If labels are provided, sorts samples by population and adds labels to the plot.

    Args:
        Q (np.ndarray): Ancestry proportion matrix (N x K).
        output_path (str | Path): File path to save the plot.
        dpi (int): Resolution in dots per inch. Defaults to 300.
        format (str): Output format (e.g. 'png', 'pdf'). Defaults to 'png'.
        labels (list | None): Population labels, one per sample. Defaults to None.
        custom_colors (list | None): Custom colors, one per cluster. Defaults to None.

    Returns:
        None
    """
    n_samples, K = Q.shape

    if labels is not None:
        if len(labels) != n_samples:
            log.warning(f"    Warning: Number of labels ({len(labels)}) does not match number of samples ({n_samples}). Ignoring labels.")
            labels = None

    if labels is not None:
        dominant_cluster = np.argmax(Q, axis=1)
        sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster, labels))
        Q_sorted = Q[sort_idx]
        labels_sorted = [labels[i] for i in sort_idx]
    else:
        dominant_cluster = np.argmax(Q, axis=1)
        sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster))
        Q_sorted = Q[sort_idx]
        labels_sorted = None

    fig, ax = plt.subplots(figsize=(15, 5))

    Q_cum = np.cumsum(Q_sorted, axis=1)
    x = np.arange(n_samples)
    zeros = np.zeros(n_samples)

    if custom_colors is not None and len(custom_colors) >= K:
        colors = custom_colors[:K]
    else:
        cmap = plt.colormaps.get_cmap('tab20')
        colors = cmap(np.linspace(0, 1, K))

    for j in range(K):
        lower = Q_cum[:, j - 1] if j > 0 else zeros
        upper = Q_cum[:, j]

        ax.fill_between(
            x, lower, upper,
            facecolor=colors[j],
            edgecolor='none',
            linewidth=0,
            rasterized=True
        )

    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Samples")
    ax.set_ylabel("Ancestry Proportions")
    ax.set_title(f"ADAMIXTURE Q-matrix (K={K})")

    if labels_sorted is not None:
        unique_labels = []
        label_positions = []
        current_label = labels_sorted[0]
        start_idx = 0
        for i, lbl in enumerate(labels_sorted):
            if lbl != current_label:
                unique_labels.append(str(current_label))
                label_positions.append((start_idx + i) / 2)
                ax.axvline(x=i, color='black', linestyle='-', linewidth=0.5)
                start_idx = i
                current_label = lbl
        unique_labels.append(str(current_label))
        label_positions.append((start_idx + n_samples) / 2)

        ax.set_xticks(label_positions)
        ax.set_xticklabels(unique_labels, rotation=45, ha='right')

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    plt.close(fig)

def align_clusters_greedy(ref_Q: np.ndarray, query_Q: np.ndarray) -> np.ndarray:
    """
    Description:
    Aligns clusters of query_Q to ref_Q using a greedy minimum-cost approach
    based on squared Euclidean distance between columns.

    Args:
        ref_Q (np.ndarray): Reference Q matrix (N x K).
        query_Q (np.ndarray): Query Q matrix (N x K) to be aligned.

    Returns:
        np.ndarray: Permutation array mapping ref cluster indices to query cluster indices.
    """
    K = ref_Q.shape[1]
    assert query_Q.shape[1] == K

    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            diff = ref_Q[:, i] - query_Q[:, j]
            cost_matrix[i, j] = np.dot(diff, diff)

    permutation = np.zeros(K, dtype=int)
    ref_indices, query_indices = [], []
    sorted_costs = np.argsort(cost_matrix.flatten())

    for idx in sorted_costs:
        r, c = np.unravel_index(idx, (K, K))
        if r not in ref_indices and c not in query_indices:
            ref_indices.append(r)
            query_indices.append(c)
            permutation[r] = c
        if len(ref_indices) == K:
            break

    return permutation
