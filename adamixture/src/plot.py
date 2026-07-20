import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

_MAX_LABEL_LEN = 25


def _draw_brackets(ax, items: list[dict], y_bracket: float, fontsize: int = 6) -> None:
    """
    Description:
    Draws elegant bracket annotations below the x-axis for a given grouping level.

    Args:
        ax: Matplotlib axes object.
        items (list[dict]): List of dicts with 'name', 'start', 'end' keys (in sample-index space).
        y_bracket (float): Y position in axes-transform space for the bracket line.
        fontsize (int): Font size for the bracket labels.

    Returns:
        None
    """
    trans = ax.get_xaxis_transform()
    y_text = y_bracket - 0.05

    for item in items:
        x0, x1 = item['start'], item['end']

        ax.plot([x0, x1], [y_bracket, y_bracket],
                color='#222222', lw=0.8, transform=trans, clip_on=False)
        ax.plot([x0, x0], [y_bracket, y_bracket + 0.08],
                color='#222222', lw=0.8, transform=trans, clip_on=False)
        ax.plot([x1, x1], [y_bracket, y_bracket + 0.08],
                color='#222222', lw=0.8, transform=trans, clip_on=False)
        label_text = str(item['name']).title()
        if len(label_text) > _MAX_LABEL_LEN:
            label_text = label_text[:_MAX_LABEL_LEN - 1] + '…'
        ax.text((x0 + x1) / 2, y_text, label_text,
                ha='center', va='top', rotation=90, fontsize=fontsize,
                color='#222222', transform=trans, clip_on=False)


def plot_q_matrix(
    Q: np.ndarray,
    output_path: str | Path,
    dpi: int = 300,
    format: str = 'png',
    labels: list | None = None,
    labels2: list | None = None,
    labels3: list | None = None,
    custom_colors: list | None = None,
) -> None:
    """
    Description:
    Plots the Q matrix (ancestry proportions) as a stacked bar chart.
    Optimized for large datasets by using rasterization.

    Supports up to three levels of hierarchical population labels.
    When ``labels`` is provided samples are grouped by population and sorted
    by dominant ancestry within each group.  Additional grouping tiers
    (``labels2``, ``labels3``) are drawn as elegant bracket annotations below
    the x-axis.

    Args:
        Q (np.ndarray): Ancestry proportion matrix (N x K).
        output_path (str | Path): File path to save the plot.
        dpi (int): Resolution in dots per inch. Defaults to 300.
        format (str): Output format (e.g. 'png', 'pdf'). Defaults to 'png'.
        labels (list | None): Level-1 population labels, one per sample. Defaults to None.
        labels2 (list | None): Level-2 grouping labels, one per sample. Defaults to None.
        labels3 (list | None): Level-3 grouping labels, one per sample. Defaults to None.
        custom_colors (list | None): Custom colors, one per cluster. Defaults to None.

    Returns:
        None
    """
    n_samples, K = Q.shape

    # Validate label lengths
    for name, lbl in [('labels', labels), ('labels2', labels2), ('labels3', labels3)]:
        if lbl is not None and len(lbl) != n_samples:
            log.warning(
                f"    Warning: Number of {name} ({len(lbl)}) does not match number of "
                f"samples ({n_samples}). Ignoring {name}."
            )
            if name == 'labels':
                labels = None
            elif name == 'labels2':
                labels2 = None
            else:
                labels3 = None

    # Validate hierarchical consistency: each lower-level label must belong to
    # exactly one higher-level group (e.g. "Barcelona" → only "Spain", not also "France").
    def _check_hierarchy(child_lbls, parent_lbls, child_name, parent_name):
        """
        Description:
        Checks that each child label maps to exactly one parent label.

        Args:
            child_lbls (list): Lower-level labels.
            parent_lbls (list): Higher-level labels.
            child_name (str): Name of the child label source for warning messages.
            parent_name (str): Name of the parent label source for warning messages.

        Returns:
            bool: True when the hierarchy is consistent, otherwise False.
        """
        mapping: dict = {}
        conflicts: list[str] = []
        for child, parent in zip(child_lbls, parent_lbls, strict=False):
            if child in mapping:
                if mapping[child] != parent:
                    conflicts.append(child)
            else:
                mapping[child] = parent
        if conflicts:
            log.warning(
                f"    Warning: Some {child_name} labels appear in more than one "
                f"{parent_name} group. Ignoring {parent_name}."
            )
            return False
        return True

    if labels is not None and labels2 is not None:
        if not _check_hierarchy(labels, labels2, '--labels', '--labels2'):
            labels2 = None
    if labels2 is not None and labels3 is not None:
        if not _check_hierarchy(labels2, labels3, '--labels2', '--labels3'):
            labels3 = None

    # ── Sort samples ──────────────────────────────────────────────────────────
    if labels is not None:
        if labels3 is not None and labels2 is not None:
            sort_idx = np.lexsort((labels, labels2, labels3))
        elif labels2 is not None:
            sort_idx = np.lexsort((labels, labels2))
        else:
            dominant_cluster = np.argmax(Q, axis=1)
            sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster, labels))
    else:
        dominant_cluster = np.argmax(Q, axis=1)
        sort_idx = np.lexsort((np.max(Q, axis=1), dominant_cluster))

    Q_sorted = Q[sort_idx]
    labels_sorted = [labels[i] for i in sort_idx] if labels is not None else None
    labels2_sorted = [labels2[i] for i in sort_idx] if labels2 is not None else None
    labels3_sorted = [labels3[i] for i in sort_idx] if labels3 is not None else None

    # ── Build level-1 boundary / tick data ───────────────────────────────────
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
                tick_text = str(current_label).title()
                if len(tick_text) > _MAX_LABEL_LEN:
                    tick_text = tick_text[:_MAX_LABEL_LEN - 1] + '…'
                pop_tick_labels.append(tick_text)
                start_idx = idx
                current_label = lbl
        tick_text = str(current_label).title()
        if len(tick_text) > _MAX_LABEL_LEN:
            tick_text = tick_text[:_MAX_LABEL_LEN - 1] + '…'
        pop_tick_positions.append((start_idx + n_samples) / 2)
        pop_tick_labels.append(tick_text)

    # ── Build bracket items for levels 2 and 3 ───────────────────────────────
    def _build_bracket_items(sorted_lbl_list: list[str]) -> list[dict]:
        """
        Description:
        Builds contiguous bracket spans from sorted labels.

        Args:
            sorted_lbl_list (list[str]): Sorted label list for one hierarchy level.

        Returns:
            list[dict]: Bracket items with name, start, and end keys.
        """
        items: list[dict] = []
        current_name = sorted_lbl_list[0]
        seg_start = 0
        for idx, name in enumerate(sorted_lbl_list):
            if name != current_name:
                items.append({'name': current_name, 'start': seg_start, 'end': idx})
                seg_start = idx
                current_name = name
        items.append({'name': current_name, 'start': seg_start, 'end': n_samples})
        return items

    i2_items = _build_bracket_items(labels2_sorted) if labels2_sorted is not None else []
    i3_items = _build_bracket_items(labels3_sorted) if labels3_sorted is not None else []

    # ── Dynamic figure height and bottom margin ───────────────────────────────
    # The height of the core bar plot area (ax) remains exactly 3.2 inches.
    # We dynamically calculate the extra height needed for each label level in inches.
    max_l1_len = min(max((len(str(lbl)) for lbl in pop_tick_labels), default=0), _MAX_LABEL_LEN)
    max_l2_len = min(max((len(item['name']) for item in i2_items), default=0), _MAX_LABEL_LEN)
    max_l3_len = min(max((len(item['name']) for item in i3_items), default=0), _MAX_LABEL_LEN)

    plot_height_in = 3.2
    l1_height_in = 0.5 + max_l1_len * 0.08 if labels_sorted else 0.0
    l2_height_in = 0.8 + max_l2_len * 0.08 if i2_items else 0.0
    l3_height_in = 0.8 + max_l3_len * 0.08 if i3_items else 0.0

    total_labels_height_in = l1_height_in + l2_height_in + l3_height_in
    if total_labels_height_in == 0:
        total_labels_height_in = 0.6

    fig_height = plot_height_in + total_labels_height_in
    bottom_margin = total_labels_height_in / fig_height

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(15, fig_height))

    Q_cum = np.cumsum(Q_sorted, axis=1)
    x = np.arange(n_samples + 1)
    zeros = np.zeros(n_samples + 1)

    if custom_colors is not None and len(custom_colors) >= K:
        colors = custom_colors[:K]
    else:
        cmap = plt.colormaps.get_cmap('tab20')
        colors = cmap(np.arange(K) % 20)

    for j in range(K):
        lower = np.r_[Q_cum[:, j - 1], Q_cum[-1, j - 1]] if j > 0 else zeros
        upper = np.r_[Q_cum[:, j], Q_cum[-1, j]]
        ax.fill_between(x, lower, upper, facecolor=colors[j], edgecolor='none', linewidth=0, rasterized=True)

    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_ylabel(f"K={K}", rotation=0, ha='right', va='center', labelpad=10, fontweight='bold')
    ax.set_yticks([0.0, 0.5, 1.0])

    # Draw level-1 boundaries and tick labels
    for boundary in pop_boundaries:
        ax.axvline(x=boundary, color='black', linestyle='--', linewidth=0.5)

    if labels_sorted is not None:
        ax.set_xticks(pop_tick_positions)
        ax.set_xticklabels(pop_tick_labels, rotation=90, ha='center', fontsize=6)
        ax.tick_params(axis='x', which='both', length=0, pad=5)

        # ── Bracket positions: convert physical inches to axes coordinates ───
        # 1.0 axes unit = plot_height_in (3.2) inches.
        _CHAR_INCH = 0.08
        _GAP_INCH = 0.35
        _TICK_PAD_INCH = 0.15

        y_l1_bottom_in = -(_TICK_PAD_INCH + max_l1_len * _CHAR_INCH)
        y_i2_in = y_l1_bottom_in - _GAP_INCH
        y_i2 = y_i2_in / plot_height_in

        y_l2_bottom_in = y_i2_in - 0.15 - max_l2_len * _CHAR_INCH
        y_i3_in = y_l2_bottom_in - _GAP_INCH
        y_i3 = y_i3_in / plot_height_in

        if i2_items:
            _draw_brackets(ax, i2_items, y_bracket=y_i2, fontsize=6)
        if i3_items:
            _draw_brackets(ax, i3_items, y_bracket=y_i3, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_xlabel("Samples")

    plt.subplots_adjust(bottom=bottom_margin)
    fig.savefig(output_path, dpi=dpi, format=format, bbox_inches='tight')
    plt.close(fig)


def plot_single_k(args: argparse.Namespace, K: int, Q_np: np.ndarray) -> None:
    """
    Description:
    Generates a Q-matrix plot for a single value of K.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        K (int): The number of populations.
        Q_np (np.ndarray): The inferred ancestry proportions matrix.

    Returns:
        None
    """
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
    log.info(f"    Generating plot: {plot_path.name}")
    plot_q_matrix(Q_np, plot_path, dpi=args.plot_dpi, format=args.plot_format,
                  labels=labels, labels2=labels2, labels3=labels3, custom_colors=colors)


def plot_combined(args: argparse.Namespace, k_values: list[int], trained_plot: dict[int, tuple]) -> None:
    """
    Description:
    Generates a combined single plot for all K sweep values.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        k_values (list[int]): List of K values to include in the plot.
        trained_plot (dict[int, tuple]): Dictionary mapping each K to its (P_np, Q_np) matrices.

    Returns:
        None
    """
    log.info("\n    Generating combined single plot for all K sweep values...")

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

    # Validate hierarchical consistency: each lower-level label must belong to
    # exactly one higher-level group (e.g. "Barcelona" → only "Spain", not also "France").
    def _check_hierarchy(child_lbls, parent_lbls, child_name, parent_name):
        """
        Description:
        Checks that each child label maps to exactly one parent label.

        Args:
            child_lbls (list): Lower-level labels.
            parent_lbls (list): Higher-level labels.
            child_name (str): Name of the child label source for warning messages.
            parent_name (str): Name of the parent label source for warning messages.

        Returns:
            bool: True when the hierarchy is consistent, otherwise False.
        """
        mapping: dict = {}
        conflicts: list[str] = []
        for child, parent in zip(child_lbls, parent_lbls, strict=False):
            if child in mapping:
                if mapping[child] != parent:
                    conflicts.append(child)
            else:
                mapping[child] = parent
        if conflicts:
            log.warning(
                f"    Warning: Some {child_name} labels appear in more than one "
                f"{parent_name} group. Ignoring {parent_name}."
            )
            return False
        return True

    if labels is not None and labels2 is not None:
        if not _check_hierarchy(labels, labels2, '--labels', '--labels2'):
            labels2 = None
    if labels2 is not None and labels3 is not None:
        if not _check_hierarchy(labels2, labels3, '--labels2', '--labels3'):
            labels3 = None

    num_runs = len(k_values)

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
                tick_text = str(current_label).title()
                if len(tick_text) > _MAX_LABEL_LEN:
                    tick_text = tick_text[:_MAX_LABEL_LEN - 1] + '…'
                pop_tick_labels.append(tick_text)
                start_idx = idx
                current_label = lbl
        tick_text = str(current_label).title()
        if len(tick_text) > _MAX_LABEL_LEN:
            tick_text = tick_text[:_MAX_LABEL_LEN - 1] + '…'
        pop_tick_positions.append((start_idx + n_samples_global) / 2)
        pop_tick_labels.append(tick_text)

    def _build_brackets_list(sorted_lbls):
        """
        Description:
        Builds contiguous bracket spans from sorted labels.

        Args:
            sorted_lbls (list | None): Sorted label list for one hierarchy level.

        Returns:
            list[dict]: Bracket items with name, start, and end keys.
        """
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

    max_l1_len = min(max((len(str(lbl)) for lbl in pop_tick_labels), default=0), _MAX_LABEL_LEN)
    max_l2_len = min(max((len(item['name']) for item in i2_items), default=0), _MAX_LABEL_LEN)
    max_l3_len = min(max((len(item['name']) for item in i3_items), default=0), _MAX_LABEL_LEN)

    plot_height_in = 2.5 * num_runs
    l1_height_in = 0.5 + max_l1_len * 0.08 if labels_sorted else 0.0
    l2_height_in = 0.8 + max_l2_len * 0.08 if i2_items else 0.0
    l3_height_in = 0.8 + max_l3_len * 0.08 if i3_items else 0.0

    total_labels_height_in = l1_height_in + l2_height_in + l3_height_in
    if total_labels_height_in == 0:
        total_labels_height_in = 0.6

    fig_height = plot_height_in + total_labels_height_in
    bottom_margin = total_labels_height_in / fig_height

    fig, axes = plt.subplots(nrows=num_runs, ncols=1, figsize=(15, fig_height), squeeze=False)
    axes = axes.flatten()

    for i, K_val in enumerate(k_values):
        ax = axes[i]
        _P_np, Q_np = trained_plot[K_val]
        Q_sorted = Q_np[sort_idx]
        n_samples, K_curr = Q_sorted.shape

        if colors_list is not None and len(colors_list) >= K_curr:
            colors = colors_list[:K_curr]
        else:
            cmap = plt.colormaps.get_cmap('tab20')
            colors = cmap(np.arange(K_curr) % 20)

        Q_cum = np.cumsum(Q_sorted, axis=1)
        x = np.arange(n_samples + 1)
        zeros = np.zeros(n_samples + 1)

        for j in range(K_curr):
            lower = np.r_[Q_cum[:, j - 1], Q_cum[-1, j - 1]] if j > 0 else zeros
            upper = np.r_[Q_cum[:, j], Q_cum[-1, j]]
            ax.fill_between(x, lower, upper, facecolor=colors[j], edgecolor='none', linewidth=0, rasterized=True)

        for boundary in pop_boundaries:
            ax.axvline(x=boundary, color='black', linestyle='--', linewidth=0.5)

        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"K={K_curr}", rotation=0, ha='right', va='center', labelpad=10, fontweight='bold')
        ax.set_yticks([0.0, 0.5, 1.0])

        is_bottom = (i == num_runs - 1)
        if is_bottom and labels_sorted is not None:
            ax.set_xticks(pop_tick_positions)
            ax.set_xticklabels(pop_tick_labels, rotation=90, ha='center', fontsize=6)
            ax.tick_params(axis='x', which='both', length=0, pad=5)

            _CHAR_INCH = 0.08
            _GAP_INCH = 0.35
            _TICK_PAD_INCH = 0.15
            _SUBPLOT_HEIGHT = 2.5

            y_l1_bottom_in = -(_TICK_PAD_INCH + max_l1_len * _CHAR_INCH)
            y_i2_in = y_l1_bottom_in - _GAP_INCH
            y_i2 = y_i2_in / _SUBPLOT_HEIGHT

            y_l2_bottom_in = y_i2_in - 0.15 - max_l2_len * _CHAR_INCH
            y_i3_in = y_l2_bottom_in - _GAP_INCH
            y_i3 = y_i3_in / _SUBPLOT_HEIGHT

            if i2_items:
                _draw_brackets(ax, i2_items, y_bracket=y_i2, fontsize=6)
            if i3_items:
                _draw_brackets(ax, i3_items, y_bracket=y_i3, fontsize=6)
        else:
            ax.set_xticks([])

        if is_bottom and labels_sorted is None:
            ax.set_xlabel("Samples")

    plt.subplots_adjust(bottom=bottom_margin, hspace=0.25)

    single_plot_path = Path(args.save_dir) / f"{args.name}.{k_values[0]}_{k_values[-1]}.{args.plot_format}"
    fig.savefig(single_plot_path, dpi=args.plot_dpi, format=args.plot_format, bbox_inches='tight')
    log.info(f"    Generating plot: {single_plot_path.name}")
    plt.close(fig)


def align_clusters_clumppling(ref_Q: np.ndarray, query_Q: np.ndarray) -> np.ndarray:
    """
    Description:
    Aligns query cluster columns to reference cluster columns using Clumppling's
    Integer Linear Programming (ILP) optimization method.

    Args:
        ref_Q (np.ndarray): Reference Q matrix.
        query_Q (np.ndarray): Query Q matrix to align.

    Returns:
        np.ndarray: Permutation array for the query_Q columns.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from clumppling.core import align_ILP

    K_ref = ref_Q.shape[1]
    K_query = query_Q.shape[1]

    if K_ref <= K_query:
        opt_obj, idxQ2P = align_ILP(ref_Q, query_Q)
        if idxQ2P is not None and len(idxQ2P) > 0:
            perm_map: dict = {}
            for q_col, r_col in enumerate(idxQ2P):
                perm_map.setdefault(r_col, []).append(q_col)

            perm = []
            used_q = set()
            for r in range(K_ref):
                if r in perm_map:
                    q_cols = perm_map[r]
                    if len(q_cols) > 1:
                        q_cols.sort(key=lambda j: float(np.dot(ref_Q[:, r] - query_Q[:, j], ref_Q[:, r] - query_Q[:, j])))
                    perm.append(q_cols[0])
                    used_q.add(q_cols[0])

            for j in range(K_query):
                if j not in used_q:
                    perm.append(j)

            return np.array(perm, dtype=int)
    else:
        opt_obj, idxP2Q = align_ILP(query_Q, ref_Q)
        if idxP2Q is not None and len(idxP2Q) > 0:
            perm = []
            used_q = set()
            for i in range(K_ref):
                q_j = idxP2Q[i]
                if q_j not in used_q:
                    perm.append(q_j)
                    used_q.add(q_j)

            for j in range(K_query):
                if j not in used_q:
                    perm.append(j)

            return np.array(perm, dtype=int)

    return np.arange(K_query, dtype=int)


def plot_multirun_clumppling(
    all_qs: list[dict],
    output_path: str | Path,
    labels: list[str] | None,
    custom_colors: list[str] | None,
    dpi: int,
    format: str,
) -> None:
    """
    Description:
    Generates a multi-run stacked bar chart using Clumppling's native plotting functions.

    Args:
        all_qs (list[dict]): List of dicts with 'id', 'K', 'Q' keys for each run.
        output_path (str | Path): File path to save the plot.
        labels (list[str] | None): Level-1 population labels, one per sample.
        custom_colors (list[str] | None): Custom colors, one per cluster.
        dpi (int): Resolution in dots per inch.
        format (str): Output format (e.g. 'png', 'pdf').

    Returns:
        None
    """
    from clumppling.plot import load_default_cmap, parse_custom_cmap, plot_memberships_list

    max_k = max(run['K'] for run in all_qs)
    if custom_colors and len(custom_colors) >= max_k:
        cmap = parse_custom_cmap(custom_colors, max_k)
    else:
        cmap = load_default_cmap(max_k)

    Q_list = [run['Q'] for run in all_qs]
    names = [f"{run['id']} (K={run['K']})" for run in all_qs]
    ind_labels = labels if labels is not None else []

    fig = plot_memberships_list(
        Q_list=Q_list,
        cmap=cmap,
        names=names,
        ind_labels=ind_labels,
        fontsize=12,
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches='tight', dpi=dpi, transparent=False)
    log.info(f"    Clumppling multi-run plot saved to: {out_path}")
    plt.close(fig)


def plot_clumppling_mode_graph(
    all_qs: list[dict],
    output_path: str | Path,
    labels: list[str] | None,
    custom_colors: list[str] | None,
    dpi: int,
    format: str,
    comm_min: float,
    comm_max: float,
    test_comm: bool,
    cd_res: float,
    cd_method: str,
) -> None:
    """
    Description:
    Generates Clumppling's mode-detection and hierarchical alignment graph
    visualization across K values and runs (producing the mode tree diagram
    and the membership alignment graph with connection cost lines).

    Args:
        all_qs (list[dict]): List of dicts with 'id', 'K', 'Q' keys for each run.
        output_path (str | Path): File path to save the combined plot.
        labels (list[str] | None): Level-1 population labels, one per sample.
        custom_colors (list[str] | None): Custom colors, one per cluster.
        dpi (int): Resolution in dots per inch.
        format (str): Output format (e.g. 'png', 'pdf').
        comm_min (float): Minimum cost threshold for mode separation.
        comm_max (float): Maximum cost threshold for mode separation.
        test_comm (bool): Whether to perform statistical test for community structure.
        cd_res (float): Resolution parameter for Louvain community detection.
        cd_method (str): Community detection method ('louvain', 'leiden', 'custom').

    Returns:
        None
    """
    import contextlib
    import io
    import tempfile

    # Suppress cdlib "Note: to be able to use all crisp methods..." warnings
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from clumppling.core import (
            align_across_k,
            align_within_k_all_K,
            detect_modes_all_K,
            extract_modes_all_K,
            reorderQ_across_k,
        )
        from clumppling.parseInput import extract_meta_input
        from clumppling.plot import (
            load_default_cmap,
            parse_custom_cmap,
            plot_alignment_graph,
            plot_graph,
        )
        from clumppling.utils import get_mode_prop, get_modes_all_K

    # Silence Clumppling loggers
    for _name in list(logging.Logger.manager.loggerDict):
        if _name.startswith("clumppling"):
            logging.getLogger(_name).setLevel(logging.WARNING)

    out_path_obj = Path(output_path)
    out_dir = out_path_obj.parent
    base_name = out_path_obj.stem

    if labels is not None and len(labels) > 0:
        label_order_map = {lbl: idx for idx, lbl in enumerate(dict.fromkeys(labels))}
        label_keys = [label_order_map[lbl] for lbl in labels]
        sort_idx = np.argsort(label_keys, kind='stable')
        labels = [labels[i] for i in sort_idx]
        all_qs = [{'id': item['id'], 'K': item['K'], 'Q': item['Q'][sort_idx, :]} for item in all_qs]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        processed_input_dir = tmp_path / "input"
        processed_input_dir.mkdir(parents=True, exist_ok=True)

        grouped_by_k: dict[int, list[np.ndarray]] = {}
        for item in all_qs:
            K = item['K']
            grouped_by_k.setdefault(K, []).append(item['Q'])

        K_range = sorted(grouped_by_k.keys())
        K_max = max(K_range)

        global_id = 1
        metadata = []
        for K in K_range:
            q_list = grouped_by_k[K]
            for rep_idx, Q in enumerate(q_list, 1):
                q_filename = f"{global_id}_K{K}R{rep_idx}.Q"
                out_q_path = processed_input_dir / q_filename
                np.savetxt(out_q_path, Q, delimiter=' ')
                metadata.append(f"{out_q_path},{out_q_path},{K}")
                global_id += 1

        meta_out_path = processed_input_dir / "input_meta.txt"
        with open(meta_out_path, "w") as f:
            f.write("\n".join(metadata))

        Q_names, _, K2IDs = extract_meta_input(str(processed_input_dir))
        n_runs_per_K = [len(K2IDs[K]) for K in K_range]

        clump_out_dir = tmp_path / "clump_out"
        clump_out_dir.mkdir(parents=True, exist_ok=True)

        alignment_withinK_list, cost_withinK_list = align_within_k_all_K(
            Q_names, K_range, K2IDs, qfile_dir=str(processed_input_dir), output_dir=str(clump_out_dir)
        )

        modes_all_K_list, cost_matrices_list = detect_modes_all_K(
            K_range, cost_withinK_list, Q_names, K2IDs,
            comm_min=comm_min, comm_max=comm_max, test_comm=test_comm,
            res=cd_res, method=cd_method
        )

        modes_dir = clump_out_dir / "modes"
        extract_res = extract_modes_all_K(
            K_range, K2IDs, Q_names, cost_matrices_list, modes_all_K_list,
            alignment_withinK_list, str(processed_input_dir), str(modes_dir)
        )

        if isinstance(extract_res, tuple):
            cd_res_list = extract_res[0]
        else:
            cd_res_list = extract_res

        mode_names_list, Q_rep_modes_list, Q_avg_modes_list = get_modes_all_K(K_range, cd_res_list)
        mode_sizes = get_mode_prop(cd_res_list, 'Size')
        mode_sims = get_mode_prop(cd_res_list, 'Performance')

        alignment_acrossK, cost_acrossK, best_acrossK_out, major_acrossK_out = align_across_k(
            K_range, Q_rep_modes_list, mode_names_list, merge=True
        )

        anchor_pairs = best_acrossK_out["Best Pair"].tolist()
        aligned_Qs_allK, all_modes_alignment = reorderQ_across_k(
            K_range, Q_rep_modes_list, mode_names_list, alignment_acrossK, anchor_pairs
        )

        if custom_colors and len(custom_colors) >= K_max:
            cmap = parse_custom_cmap(custom_colors, K_max)
        else:
            cmap = load_default_cmap(K_max)



        y_aspect = 3 if K_max < 5 else 3 + K_max / 5
        wspace_padding = 1.3 if K_max < 8 else 1.15

        fig_tree = plot_alignment_graph(
            K_range, names_list=mode_names_list, cmap=cmap,
            alignment_acrossK=alignment_acrossK, all_modes_alignment=all_modes_alignment,
            anchor_pairs=anchor_pairs, alt_color=True, ls_alt=['-', '--'],
            y_aspect=y_aspect, wspace_padding=wspace_padding
        )

        tree_path = tmp_path / "mode_tree.png"
        fig_tree.savefig(str(tree_path), bbox_inches='tight', dpi=dpi, transparent=False)
        plt.close(fig_tree)

        ind_labels = labels if labels is not None else []
        mode_labels_list = [
            [
                f"{mode_name.title().replace('_', ' ')} ({mode_sizes[mode_name]}/{n_runs_per_K[i_K]})"
                for mode_name in mode_names
            ]
            for i_K, mode_names in enumerate(mode_names_list)
        ]
        right_labels_list = [
            [f"sim {mode_sims[mode_name]:.3f}" for mode_name in mode_names]
            for i_K, mode_names in enumerate(mode_names_list)
        ]
        Q_modes_reordered_list = [
            [aligned_Qs_allK[mode_name] for mode_name in mode_names]
            for mode_names in mode_names_list
        ]

        width_scale = 3 / K_max + 0.7 if K_max > 10 else 1.0
        height_scale = 5 / len(K_range) if len(K_range) < 5 else 1.0

        order_cls_by_label = True if (ind_labels and len(ind_labels) > 0) else False

        fig_graph = plot_graph(
            K_range, Q_modes_reordered_list, cmap,
            names_list=mode_names_list, labels_list=mode_labels_list,
            right_labels_list=right_labels_list, cost_acrossK=cost_acrossK,
            ind_labels=ind_labels, fontsize=14, alt_color=True,
            order_refQ=None, order_cls_by_label=order_cls_by_label,
            width_scale=width_scale, height_scale=height_scale
        )

        graph_path = tmp_path / "mode_graph.png"
        fig_graph.savefig(str(graph_path), bbox_inches='tight', dpi=dpi, transparent=False)
        plt.close(fig_graph)

        img_tree = Image.open(tree_path)
        img_graph = Image.open(graph_path)

        total_width = max(img_tree.width, img_graph.width)
        total_height = img_tree.height + img_graph.height + 40

        combined_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
        offset_x_tree = (total_width - img_tree.width) // 2
        combined_img.paste(img_tree, (offset_x_tree, 0))
        offset_x_graph = (total_width - img_graph.width) // 2
        combined_img.paste(img_graph, (offset_x_graph, img_tree.height + 40))

        if format.lower() == 'pdf':
            combined_img.save(out_path_obj, "PDF", resolution=float(dpi))
        else:
            combined_img.save(out_path_obj, dpi=(dpi, dpi))

    log.info(f"    Clumppling multi-run mode graph saved to: {out_path_obj}")
