import argparse
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
logging.getLogger("matplotlib").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

_MAX_LABEL_LEN = 25


def _require_pillow_image():
    """Return PIL.Image with an actionable error if Pillow is unavailable."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to compose Clumppling mode graphs and should "
            "be installed transitively with matplotlib."
        ) from exc
    return Image


def _format_label(text: object) -> str:
    """Format a label in title case and truncate to _MAX_LABEL_LEN."""
    s = str(text).title()
    return s[: _MAX_LABEL_LEN - 1] + "…" if len(s) > _MAX_LABEL_LEN else s


def _load_nonempty_lines(file_path: str | Path | None) -> list[str] | None:
    """Read non-empty stripped lines from a file if it exists."""
    if not file_path:
        return None
    p = Path(file_path)
    if not p.exists():
        return None
    with open(p) as fh:
        return [s for line in fh if (s := line.strip())]


def _check_hierarchy_codes(
    child_codes: np.ndarray,
    n_unique_children: int,
    parent_codes: np.ndarray,
    child_name: str,
    parent_name: str,
) -> bool:
    """
    Vectorized O(N) check that each child label maps to a single parent label.
    """
    expected_parent = np.empty(n_unique_children, dtype=parent_codes.dtype)
    expected_parent[child_codes] = parent_codes
    if not np.array_equal(expected_parent[child_codes], parent_codes):
        log.warning(
            f"    Warning: Some {child_name} labels appear in more than one "
            f"{parent_name} group. Ignoring {parent_name}."
        )
        return False
    return True


def _build_contiguous_spans(
    sorted_codes: np.ndarray, unique_vals: np.ndarray
) -> list[dict]:
    """
    Vectorized run-length encoding of sorted integer label codes into
    [{'name': ..., 'start': ..., 'end': ...}] spans.
    """
    n = len(sorted_codes)
    if n == 0:
        return []
    change_idx = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
    starts = np.r_[0, change_idx].tolist()
    ends = np.r_[change_idx, n].tolist()
    names = unique_vals[sorted_codes[starts]].tolist()
    return [
        {"name": name, "start": s, "end": e}
        for name, s, e in zip(names, starts, ends)
    ]


def _prepare_sorting_and_labels(
    Q_ref: np.ndarray,
    labels: list | None,
    labels2: list | None,
    labels3: list | None,
    warn_on_len_mismatch: bool = True,
):
    """
    Validates, encodes, sorts, and computes contiguous spans for up to 3 label tiers
    using integer-encoded arrays instead of Python string loops.
    """
    n_samples = Q_ref.shape[0]

    for name, lbl in [("labels", labels), ("labels2", labels2), ("labels3", labels3)]:
        if lbl is not None and len(lbl) != n_samples:
            if warn_on_len_mismatch:
                log.warning(
                    f"    Warning: Number of {name} ({len(lbl)}) does not match number of "
                    f"samples ({n_samples}). Ignoring {name}."
                )
            if name == "labels":
                labels = None
            elif name == "labels2":
                labels2 = None
            else:
                labels3 = None

    u1, inv1 = (np.unique(labels, return_inverse=True) if labels is not None else (None, None))
    u2, inv2 = (np.unique(labels2, return_inverse=True) if labels2 is not None else (None, None))
    u3, inv3 = (np.unique(labels3, return_inverse=True) if labels3 is not None else (None, None))

    if inv1 is not None and inv2 is not None:
        if not _check_hierarchy_codes(inv1, len(u1), inv2, "--labels", "--labels2"):
            u2, inv2 = None, None
    if inv2 is not None and inv3 is not None:
        if not _check_hierarchy_codes(inv2, len(u2), inv3, "--labels2", "--labels3"):
            u3, inv3 = None, None

    # Sort samples using integer codes and single-pass max extraction
    if inv1 is not None:
        if inv3 is not None and inv2 is not None:
            sort_idx = np.lexsort((inv1, inv2, inv3))
        elif inv2 is not None:
            sort_idx = np.lexsort((inv1, inv2))
        else:
            dominant_cluster = np.argmax(Q_ref, axis=1)
            max_q = Q_ref[np.arange(n_samples), dominant_cluster]
            sort_idx = np.lexsort((max_q, dominant_cluster, inv1))
    else:
        dominant_cluster = np.argmax(Q_ref, axis=1)
        max_q = Q_ref[np.arange(n_samples), dominant_cluster]
        sort_idx = np.lexsort((max_q, dominant_cluster))

    # Build level-1 boundaries and ticks via vectorized change detection
    pop_boundaries: list[int] = []
    pop_tick_positions: list[float] = []
    pop_tick_labels: list[str] = []
    if inv1 is not None:
        l1_spans = _build_contiguous_spans(inv1[sort_idx], u1)
        pop_boundaries = [item["end"] for item in l1_spans[:-1]]
        pop_tick_positions = [(item["start"] + item["end"]) / 2 for item in l1_spans]
        pop_tick_labels = [_format_label(item["name"]) for item in l1_spans]

    i2_items = _build_contiguous_spans(inv2[sort_idx], u2) if inv2 is not None else []
    i3_items = _build_contiguous_spans(inv3[sort_idx], u3) if inv3 is not None else []

    has_l1 = inv1 is not None
    return sort_idx, has_l1, pop_boundaries, pop_tick_positions, pop_tick_labels, i2_items, i3_items


def _draw_brackets(ax, items: list[dict], y_bracket: float, fontsize: int = 6) -> None:
    """
    Draws bracket annotations below the x-axis for a given grouping level.
    Combines all bracket geometry into a single Line2D call for speed.
    """
    if not items:
        return

    trans = ax.get_xaxis_transform()
    y_text = y_bracket - 0.05
    y_top = y_bracket + 0.08

    # Construct a single NaN-separated polyline for all brackets in this tier
    n_items = len(items)
    xs = np.empty((n_items, 5), dtype=float)
    ys = np.empty((n_items, 5), dtype=float)
    ys[:, 0] = y_top
    ys[:, 1] = y_bracket
    ys[:, 2] = y_bracket
    ys[:, 3] = y_top
    ys[:, 4] = np.nan

    for idx, item in enumerate(items):
        x0, x1 = item["start"], item["end"]
        xs[idx, 0] = x0
        xs[idx, 1] = x0
        xs[idx, 2] = x1
        xs[idx, 3] = x1
        xs[idx, 4] = np.nan

        ax.text(
            (x0 + x1) / 2,
            y_text,
            _format_label(item["name"]),
            ha="center",
            va="top",
            rotation=90,
            fontsize=fontsize,
            color="#222222",
            transform=trans,
            clip_on=False,
        )

    ax.plot(
        xs.ravel(),
        ys.ravel(),
        color="#222222",
        lw=0.8,
        transform=trans,
        clip_on=False,
    )


def _render_stacked_q(
    ax,
    Q_sorted: np.ndarray,
    colors,
    pop_boundaries: list[int],
) -> None:
    """
    Renders the rasterized stacked area chart and boundary lines onto ``ax``
    using preallocated 1D buffers and contiguous memory access.
    """
    n_samples, K = Q_sorted.shape
    x = np.arange(n_samples + 1)

    # Transpose once to C-contiguous (K, N) so each cluster slice is contiguous in RAM
    Q_cols = np.ascontiguousarray(Q_sorted.T)
    lower = np.zeros(n_samples + 1, dtype=Q_sorted.dtype)
    upper = np.empty(n_samples + 1, dtype=Q_sorted.dtype)

    for j in range(K):
        np.add(lower[:-1], Q_cols[j], out=upper[:-1])
        upper[-1] = upper[-2]
        ax.fill_between(
            x,
            lower,
            upper,
            facecolor=colors[j],
            edgecolor="none",
            linewidth=0,
            rasterized=True,
        )
        lower, upper = upper, lower

    if pop_boundaries:
        ax.vlines(
            pop_boundaries,
            0.0,
            1.0,
            colors="black",
            linestyles="--",
            linewidth=0.5,
        )

    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, 1)
    ax.set_ylabel(
        f"K={K}", rotation=0, ha="right", va="center", labelpad=10, fontweight="bold"
    )
    ax.set_yticks([0.0, 0.5, 1.0])


def plot_q_matrix(
    Q: np.ndarray,
    output_path: str | Path,
    dpi: int = 300,
    format: str = "png",
    labels: list | None = None,
    labels2: list | None = None,
    labels3: list | None = None,
    custom_colors: list | None = None,
) -> None:
    """
    Plots the Q matrix (ancestry proportions) as a stacked bar chart.
    Optimized for large datasets by using rasterization and vectorized preprocessing.
    """
    _, K = Q.shape

    (
        sort_idx,
        has_l1,
        pop_boundaries,
        pop_tick_positions,
        pop_tick_labels,
        i2_items,
        i3_items,
    ) = _prepare_sorting_and_labels(Q, labels, labels2, labels3, warn_on_len_mismatch=True)

    Q_sorted = Q[sort_idx]

    max_l1_len = min(max((len(lbl) for lbl in pop_tick_labels), default=0), _MAX_LABEL_LEN)
    max_l2_len = min(max((len(str(item["name"])) for item in i2_items), default=0), _MAX_LABEL_LEN)
    max_l3_len = min(max((len(str(item["name"])) for item in i3_items), default=0), _MAX_LABEL_LEN)

    plot_height_in = 3.2
    l1_height_in = 0.5 + max_l1_len * 0.08 if has_l1 else 0.0
    l2_height_in = 0.8 + max_l2_len * 0.08 if i2_items else 0.0
    l3_height_in = 0.8 + max_l3_len * 0.08 if i3_items else 0.0

    total_labels_height_in = l1_height_in + l2_height_in + l3_height_in
    if total_labels_height_in == 0:
        total_labels_height_in = 0.6

    fig_height = plot_height_in + total_labels_height_in
    bottom_margin = total_labels_height_in / fig_height

    fig, ax = plt.subplots(figsize=(15, fig_height))

    if custom_colors is not None and len(custom_colors) >= K:
        colors = custom_colors[:K]
    else:
        cmap = plt.colormaps.get_cmap("tab20")
        colors = cmap(np.arange(K) % 20)

    _render_stacked_q(ax, Q_sorted, colors, pop_boundaries)

    if has_l1:
        ax.set_xticks(pop_tick_positions)
        ax.set_xticklabels(pop_tick_labels, rotation=90, ha="center", fontsize=6)
        ax.tick_params(axis="x", which="both", length=0, pad=5)

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
    fig.savefig(output_path, dpi=dpi, format=format, bbox_inches="tight")
    plt.close(fig)


def plot_single_k(args: argparse.Namespace, K: int, Q_np: np.ndarray) -> None:
    """Generates a Q-matrix plot for a single value of K."""
    labels = _load_nonempty_lines(getattr(args, "labels", None))
    labels2 = _load_nonempty_lines(getattr(args, "labels2", None))
    labels3 = _load_nonempty_lines(getattr(args, "labels3", None))
    colors = _load_nonempty_lines(getattr(args, "colors", None))

    if colors is not None and len(colors) != K:
        log.warning(
            f"    Number of colors in {args.colors} ({len(colors)}) does not match K={K}. Using default colors."
        )
        colors = None

    plot_path = Path(args.save_dir) / f"{args.name}.{K}.{args.plot_format}"
    log.info(f"    Generating plot: {plot_path.name}")
    plot_q_matrix(
        Q_np,
        plot_path,
        dpi=args.plot_dpi,
        format=args.plot_format,
        labels=labels,
        labels2=labels2,
        labels3=labels3,
        custom_colors=colors,
    )


def plot_combined(
    args: argparse.Namespace, k_values: list[int], trained_plot: dict[int, tuple]
) -> None:
    """Generates a combined single plot for all K sweep values."""
    log.info("\n    Generating combined single plot for all K sweep values...")

    labels = _load_nonempty_lines(getattr(args, "labels", None))
    labels2 = _load_nonempty_lines(getattr(args, "labels2", None))
    labels3 = _load_nonempty_lines(getattr(args, "labels3", None))
    colors_list = _load_nonempty_lines(getattr(args, "colors", None))

    num_runs = len(k_values)
    first_K = k_values[0]
    first_Q = trained_plot[first_K][1]

    (
        sort_idx,
        has_l1,
        pop_boundaries,
        pop_tick_positions,
        pop_tick_labels,
        i2_items,
        i3_items,
    ) = _prepare_sorting_and_labels(
        first_Q, labels, labels2, labels3, warn_on_len_mismatch=False
    )

    max_l1_len = min(max((len(lbl) for lbl in pop_tick_labels), default=0), _MAX_LABEL_LEN)
    max_l2_len = min(max((len(str(item["name"])) for item in i2_items), default=0), _MAX_LABEL_LEN)
    max_l3_len = min(max((len(str(item["name"])) for item in i3_items), default=0), _MAX_LABEL_LEN)

    plot_height_in = 2.5 * num_runs
    l1_height_in = 0.5 + max_l1_len * 0.08 if has_l1 else 0.0
    l2_height_in = 0.8 + max_l2_len * 0.08 if i2_items else 0.0
    l3_height_in = 0.8 + max_l3_len * 0.08 if i3_items else 0.0

    total_labels_height_in = l1_height_in + l2_height_in + l3_height_in
    if total_labels_height_in == 0:
        total_labels_height_in = 0.6

    fig_height = plot_height_in + total_labels_height_in
    bottom_margin = total_labels_height_in / fig_height

    fig, axes = plt.subplots(
        nrows=num_runs, ncols=1, figsize=(15, fig_height), squeeze=False
    )
    axes = axes.flatten()
    default_cmap = plt.colormaps.get_cmap("tab20")

    for i, K_val in enumerate(k_values):
        ax = axes[i]
        _P_np, Q_np = trained_plot[K_val]
        Q_sorted = Q_np[sort_idx]
        K_curr = Q_sorted.shape[1]

        if colors_list is not None and len(colors_list) >= K_curr:
            colors = colors_list[:K_curr]
        else:
            colors = default_cmap(np.arange(K_curr) % 20)

        _render_stacked_q(ax, Q_sorted, colors, pop_boundaries)

        is_bottom = i == num_runs - 1
        if is_bottom and has_l1:
            ax.set_xticks(pop_tick_positions)
            ax.set_xticklabels(pop_tick_labels, rotation=90, ha="center", fontsize=6)
            ax.tick_params(axis="x", which="both", length=0, pad=5)

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

        if is_bottom and not has_l1:
            ax.set_xlabel("Samples")

    plt.subplots_adjust(bottom=bottom_margin, hspace=0.25)

    single_plot_path = (
        Path(args.save_dir)
        / f"{args.name}.{k_values[0]}_{k_values[-1]}.{args.plot_format}"
    )
    fig.savefig(
        single_plot_path, dpi=args.plot_dpi, format=args.plot_format, bbox_inches="tight"
    )
    log.info(f"    Generating plot: {single_plot_path.name}")
    plt.close(fig)


def align_clusters_greedy(ref_Q: np.ndarray, query_Q: np.ndarray) -> np.ndarray:
    """
    Aligns query cluster columns to reference cluster columns using a greedy
    minimum-cost matching. Computes the pairwise squared Euclidean distance matrix
    via BLAS matrix multiplication instead of nested Python loops.
    """
    K_ref = ref_Q.shape[1]
    K_query = query_Q.shape[1]

    # ||u - v||_2^2 = ||u||_2^2 + ||v||_2^2 - 2 * u^T v
    ref_norms = np.einsum("ij,ij->j", ref_Q, ref_Q)[:, None]
    query_norms = np.einsum("ij,ij->j", query_Q, query_Q)[None, :]
    cost_matrix = ref_norms + query_norms - 2.0 * (ref_Q.T @ query_Q)

    ref_indices = set()
    query_indices = set()
    matches = {}

    target_matches = min(K_ref, K_query)
    flat_order = np.argsort(cost_matrix, axis=None)
    rows, cols = np.unravel_index(flat_order, (K_ref, K_query))

    for r, c in zip(rows.tolist(), cols.tolist()):
        if r not in ref_indices and c not in query_indices:
            ref_indices.add(r)
            query_indices.add(c)
            matches[r] = c
            if len(ref_indices) == target_matches:
                break

    perm = [matches[r] for r in range(K_ref) if r in matches]
    perm.extend(c for c in range(K_query) if c not in query_indices)
    return np.array(perm, dtype=int)


def align_clusters_clumppling(ref_Q: np.ndarray, query_Q: np.ndarray) -> np.ndarray:
    """
    Aligns query cluster columns to reference cluster columns using Clumppling's
    optimal cluster-matching formulation, falling back to greedy matching.
    """
    from .clumppling import align_ILP

    K_ref = ref_Q.shape[1]
    K_query = query_Q.shape[1]

    try:
        if K_ref <= K_query:
            opt_obj, idxQ2P = align_ILP(ref_Q, query_Q)
            if idxQ2P is not None and len(idxQ2P) > 0:
                perm_map: dict[int, list[int]] = {}
                for q_col, r_col in enumerate(idxQ2P):
                    perm_map.setdefault(r_col, []).append(q_col)

                perm = []
                used_q = set()
                for r in range(K_ref):
                    if r in perm_map:
                        q_cols = perm_map[r]
                        if len(q_cols) > 1:
                            diffs = query_Q[:, q_cols] - ref_Q[:, r : r + 1]
                            sq_dists = np.einsum("ij,ij->j", diffs, diffs).tolist()
                            q_cols = [
                                c for _, c in sorted(zip(sq_dists, q_cols))
                            ]
                        perm.append(q_cols[0])
                        used_q.add(q_cols[0])

                perm.extend(j for j in range(K_query) if j not in used_q)
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

                perm.extend(j for j in range(K_query) if j not in used_q)
                return np.array(perm, dtype=int)
    except Exception:
        return align_clusters_greedy(ref_Q, query_Q)

    return align_clusters_greedy(ref_Q, query_Q)


def plot_multirun_clumppling(
    all_qs: list[dict],
    output_path: str | Path,
    labels: list[str] | None,
    custom_colors: list[str] | None,
    dpi: int,
    format: str,
) -> None:
    """Generates a multi-run stacked bar chart using Clumppling's native plotting functions."""
    from .clumppling import load_default_cmap, parse_custom_cmap, plot_memberships_list

    max_k = max(run["K"] for run in all_qs)
    if custom_colors and len(custom_colors) >= max_k:
        cmap = parse_custom_cmap(custom_colors, max_k)
    else:
        cmap = load_default_cmap(max_k)

    Q_list = [run["Q"] for run in all_qs]
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
    fig.savefig(str(out_path), bbox_inches="tight", dpi=dpi, transparent=False)
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
) -> None:
    """
    Generates Clumppling's mode-detection and hierarchical alignment graph
    visualization across K values and runs, composing both figures in RAM.
    """
    from .clumppling import (
        align_across_k,
        align_within_k_all_K,
        detect_modes_all_K,
        extract_modes_all_K,
        get_mode_prop,
        get_modes_all_K,
        load_default_cmap,
        parse_custom_cmap,
        plot_alignment_graph,
        plot_graph,
        reorderQ_across_k,
    )

    Image = _require_pillow_image()
    out_path_obj = Path(output_path)

    if labels is not None and len(labels) > 0:
        label_order_map = {lbl: idx for idx, lbl in enumerate(dict.fromkeys(labels))}
        label_keys = np.fromiter(
            (label_order_map[lbl] for lbl in labels), dtype=np.int32, count=len(labels)
        )
        sort_idx = np.argsort(label_keys, kind="stable")
        labels = [labels[i] for i in sort_idx]
        all_qs = [
            {"id": item["id"], "K": item["K"], "Q": item["Q"][sort_idx, :]}
            for item in all_qs
        ]

    grouped_by_k: dict[int, list[np.ndarray]] = {}
    for item in all_qs:
        grouped_by_k.setdefault(item["K"], []).append(item["Q"])

    K_range = sorted(grouped_by_k.keys())
    K_max = K_range[-1]
    n_runs_per_K = [len(grouped_by_k[K]) for K in K_range]

    alignment_withinK_list, cost_withinK_list = align_within_k_all_K(
        K_range, grouped_by_k
    )

    modes_all_K_list, cost_matrices_list = detect_modes_all_K(
        K_range,
        cost_withinK_list,
        n_runs_per_K,
        comm_min=comm_min,
        comm_max=comm_max,
        test_comm=test_comm,
        res=cd_res,
    )

    cd_res_list = extract_modes_all_K(
        K_range,
        grouped_by_k,
        cost_matrices_list,
        modes_all_K_list,
        alignment_withinK_list,
    )

    mode_names_list, Q_rep_modes_list, _ = get_modes_all_K(K_range, cd_res_list)
    mode_sizes = get_mode_prop(cd_res_list, "Size")
    mode_sims = get_mode_prop(cd_res_list, "Performance")

    alignment_acrossK, cost_acrossK, anchor_pairs = align_across_k(
        K_range, Q_rep_modes_list, mode_names_list, merge=True
    )

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
        K_range,
        names_list=mode_names_list,
        cmap=cmap,
        alignment_acrossK=alignment_acrossK,
        all_modes_alignment=all_modes_alignment,
        anchor_pairs=anchor_pairs,
        alt_color=True,
        ls_alt=["-", "--"],
        y_aspect=y_aspect,
        wspace_padding=wspace_padding,
    )

    buf_tree = io.BytesIO()
    fig_tree.savefig(
        buf_tree, format="png", bbox_inches="tight", dpi=dpi, transparent=False
    )
    plt.close(fig_tree)
    buf_tree.seek(0)

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
    order_cls_by_label = bool(ind_labels)

    fig_graph = plot_graph(
        K_range,
        Q_modes_reordered_list,
        cmap,
        names_list=mode_names_list,
        labels_list=mode_labels_list,
        right_labels_list=right_labels_list,
        cost_acrossK=cost_acrossK,
        ind_labels=ind_labels,
        fontsize=14,
        alt_color=True,
        order_refQ=None,
        order_cls_by_label=order_cls_by_label,
        width_scale=width_scale,
        height_scale=height_scale,
    )

    buf_graph = io.BytesIO()
    fig_graph.savefig(
        buf_graph, format="png", bbox_inches="tight", dpi=dpi, transparent=False
    )
    plt.close(fig_graph)
    buf_graph.seek(0)

    with Image.open(buf_tree) as img_tree, Image.open(buf_graph) as img_graph:
        total_width = max(img_tree.width, img_graph.width)
        total_height = img_tree.height + img_graph.height + 40

        combined_img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        offset_x_tree = (total_width - img_tree.width) // 2
        combined_img.paste(img_tree, (offset_x_tree, 0))
        offset_x_graph = (total_width - img_graph.width) // 2
        combined_img.paste(img_graph, (offset_x_graph, img_tree.height + 40))

        out_path_obj.parent.mkdir(parents=True, exist_ok=True)
        if format.lower() == "pdf":
            combined_img.save(out_path_obj, "PDF", resolution=float(dpi))
        else:
            combined_img.save(out_path_obj, dpi=(dpi, dpi))

    log.info(f"    Clumppling multi-run mode graph saved to: {out_path_obj}")
