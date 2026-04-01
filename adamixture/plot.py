import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_q_matrix(Q: np.ndarray, output_path: str | Path, dpi: int = 300, format: str = 'png', labels: list | None = None, custom_colors: list | None = None):
    """
    Plots the Q matrix (ancestry proportions) as a stacked bar chart.
    Optimized for large datasets by using rasterization.
    If labels are provided, sorts samples by population and adds labels to the plot.
    """
    n_samples, K = Q.shape
    
    if labels is not None:
        if len(labels) != n_samples:
            print(f"Warning: Number of labels ({len(labels)}) does not match number of samples ({n_samples}). Ignoring labels.")
            labels = None
            
    if labels is not None:
        # Sort by population label, then by dominant cluster within population
        dominant_cluster = np.argmax(Q, axis=1)
        # Create a combined sort key
        sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster, labels))
        Q_sorted = Q[sort_idx]
        labels_sorted = [labels[i] for i in sort_idx]
    else:
        # Sort samples by their dominant cluster to make the plot cleaner
        dominant_cluster = np.argmax(Q, axis=1)
        sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster))
        Q_sorted = Q[sort_idx]
        labels_sorted = None
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # Cumulative sum for stacking
    Q_cum = np.cumsum(Q_sorted, axis=1)
    x = np.arange(n_samples)
    zeros = np.zeros(n_samples)
    
    # Standard colors (extending if needed) or custom colors
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
        # Add population labels on the x-axis
        # Find boundaries of populations
        unique_labels = []
        label_positions = []
        current_label = labels_sorted[0]
        start_idx = 0
        for i, lbl in enumerate(labels_sorted):
            if lbl != current_label:
                unique_labels.append(str(current_label))
                label_positions.append((start_idx + i) / 2)
                # Add vertical line for population boundary
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
    Aligns clusters of query_Q to ref_Q using a greedy maximum overlap approach.
    Returns the permutation of columns for query_Q.
    """
    K = ref_Q.shape[1]
    assert query_Q.shape[1] == K
    
    # Calculate pairwise costs (squared euclidean distance)
    cost_matrix = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            diff = ref_Q[:, i] - query_Q[:, j]
            cost_matrix[i, j] = np.dot(diff, diff)

    # Greedy assignment based on lowest cost
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
