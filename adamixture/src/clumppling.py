"""
In-tree port of the Clumppling routines used by ADAMIXTURE.

Adapted from Clumppling (https://github.com/PopGenClustering/Clumppling), MIT licensed,
by Xiran Liu et al. Only the cluster alignment, mode detection and mode-graph plotting
steps that ADAMIXTURE actually calls are reproduced here, rewritten on top of NumPy and
matplotlib so that the package no longer depends on cvxpy, cdlib, networkx, pandas,
scipy or TracyWidom. Runs are held in memory instead of being round-tripped through
.Q files. The two steps that do not vectorise, the alignment assignment problem and
Louvain community detection, live in adamixture/src/utils_c/cython/clumppling.pyx.
"""

import colorsys
import logging
from collections import Counter, defaultdict
from itertools import combinations, product

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch, Rectangle

from .utils_c.cython import clumppling as _kernels

log = logging.getLogger(__name__)

# Cluster palette shipped with Clumppling (clumppling/files/default_colors.txt).
DEFAULT_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#999999", "#FF7F00", "#8F33FF",
    "#33CC33", "#FF33CC", "#33FFFF", "#6600CC", "#FF6666",
    "#00CC99", "#CC6600", "#3366FF", "#CCFF33", "#990033",
    "#FFB266", "#FF9933", "#FF8000", "#CC6600", "#B24D00",
    "#993300", "#996633", "#CC9966", "#FFCC99", "#FFFFCC",
    "#FFFF99", "#CCFF66", "#99FF33", "#66CC33", "#339900",
    "#339966", "#33CCCC", "#33FFFF", "#66CCFF", "#3399FF",
    "#3366FF", "#3333FF", "#6633FF", "#9933FF", "#CC33FF",
    "#FF33CC", "#FF3399", "#FF6666", "#FF9999", "#FFCCCC",
]

# Colours and line styles cycled through so that crossing edges stay distinguishable.
ALT_EDGE_COLORS = ['rosybrown', 'steelblue', 'goldenrod', 'darkseagreen']
ALT_LINE_STYLES = ['-', '--']

# Shifted-gamma fits (shape, scale, shift) of the Tracy-Widom laws, from Chiani (2014).
_TRACY_WIDOM_FITS = {
    1: (46.446, 0.186054, 9.848),
    2: (79.6595, 0.101037, 9.81961),
    4: (146.021, 0.0595445, 9.84713),
}


# ---------------------------------------------------------------------------
# Membership costs
# ---------------------------------------------------------------------------

def cost_membership(P: np.ndarray, Q: np.ndarray) -> float:
    """
    Description:
    Computes the dissimilarity between two membership matrices as the squared Frobenius
    norm of their difference, normalised by twice the number of individuals.

    Args:
        P (np.ndarray): Membership matrix of shape (N_ind, K).
        Q (np.ndarray): Membership matrix of the same shape as P.

    Returns:
        float: Cost of the pair, in [0, 1].
    """
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same size.")
    return float(np.linalg.norm(P - Q, ord='fro') ** 2 / (2 * P.shape[0]))


def C2Gprime(C: np.ndarray) -> np.ndarray:
    """
    Description:
    Converts an alignment cost into the G' similarity score used to report performance.

    Args:
        C (np.ndarray): Cost values in [0, 1].

    Returns:
        np.ndarray: Similarity values in [0, 1].
    """
    return 1 - np.sqrt(C)


def avg_tril(A: np.ndarray) -> float:
    """
    Description:
    Averages the strictly lower-triangular part of a square matrix, i.e. the mean over
    all unordered pairs.

    Args:
        A (np.ndarray): Square matrix.

    Returns:
        float: Mean of the off-diagonal pairs.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input matrix must be square.")
    return float(np.sum(np.tril(A, k=-1)) / (A.shape[0] * (A.shape[0] - 1) / 2))


def reverse_alignment_same_K(idxQ2P: list[int]) -> list[int]:
    """
    Description:
    Inverts a cluster mapping between two matrices that have the same number of clusters.

    Args:
        idxQ2P (list[int]): Cluster of P that each cluster of Q maps to.

    Returns:
        list[int]: Cluster of Q that each cluster of P maps to.
    """
    idxP2Q = np.zeros(len(idxQ2P), dtype=int)
    for i, idx in enumerate(idxQ2P):
        idxP2Q[idx] = i
    return [int(i) for i in idxP2Q]


def validate_membership_matrix(matrix: np.ndarray, tolerance: float = 1e-6) -> None:
    """
    Description:
    Checks that a membership matrix holds proportions in [0, 1] whose rows sum to one.

    Args:
        matrix (np.ndarray): Membership matrix to validate.
        tolerance (float): Absolute tolerance allowed on the row sums.

    Returns:
        None
    """
    if not ((0 <= matrix) & (matrix <= 1)).all():
        raise ValueError("Matrix contains values outside [0, 1].")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        bad_rows = np.where(~np.isclose(row_sums, 1.0, atol=tolerance))[0]
        raise ValueError(f"Rows {bad_rows.tolist()} do not sum to 1 (+/-{tolerance}).")


# ---------------------------------------------------------------------------
# Cluster alignment
# ---------------------------------------------------------------------------

def _cluster_distances(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Description:
    Computes the mean squared distance between every cluster of Q and every cluster of P.

    Args:
        P (np.ndarray): Membership matrix of shape (N_ind, K_P).
        Q (np.ndarray): Membership matrix of shape (N_ind, K_Q).

    Returns:
        np.ndarray: Distance matrix of shape (K_Q, K_P).
    """
    n_ind = Q.shape[0]
    D = np.empty((Q.shape[1], P.shape[1]), dtype=np.float64)
    for p_idx in range(P.shape[1]):
        D[:, p_idx] = np.square(Q - P[:, p_idx:p_idx + 1]).sum(axis=0) / n_ind
    return D


def align_ILP(P: np.ndarray, Q: np.ndarray) -> tuple[float, list[int]]:
    """
    Description:
    Matches the clusters of Q to the clusters of P at minimum total distance, with every
    cluster of Q assigned once and every cluster of P receiving at least one cluster.

    Clumppling states this as an integer linear program and hands it to a MIP solver.
    The program is solved here in closed form instead: pick one distinct cluster of Q as
    the representative of each cluster of P, after which every surplus cluster of Q is
    free to join its own nearest cluster of P. Subtracting each row's minimum turns the
    choice of representatives into a minimum-cost assignment, which the Jonker-Volgenant
    solver settles exactly, so the returned matching is optimal for the original program.

    Args:
        P (np.ndarray): Membership matrix of shape (N_ind, K_P).
        Q (np.ndarray): Membership matrix of shape (N_ind, K_Q), with K_Q >= K_P.

    Returns:
        tuple[float, list[int]]: Optimal objective value, and the cluster of P that each
        cluster of Q maps to.
    """
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("Both P and Q must be 2D matrices.")
    if P.shape[0] != Q.shape[0]:
        raise ValueError("P and Q must have the same number of rows (individuals).")
    if P.shape[1] > Q.shape[1]:
        raise ValueError("Q must have no fewer columns (clusters) than P.")

    K_P = P.shape[1]
    D = _cluster_distances(P, Q)

    reduced = np.ascontiguousarray((D - D.min(axis=1, keepdims=True)).T)
    representatives = _kernels.solve_lap(reduced)

    idxQ2P = np.argmin(D, axis=1)
    idxQ2P[representatives] = np.arange(K_P)
    opt_obj = float(D[np.arange(D.shape[0]), idxQ2P].sum())

    return opt_obj, [int(i) for i in idxQ2P]


def align_ILP_diff1(P: np.ndarray, Q: np.ndarray) -> tuple[float, list[int]]:
    """
    Description:
    Aligns Q to P when Q has exactly one cluster more than P, by enumerating every pair
    of Q clusters that could be merged and keeping the best alignment.

    Args:
        P (np.ndarray): Membership matrix of shape (N_ind, K_P).
        Q (np.ndarray): Membership matrix of shape (N_ind, K_P + 1).

    Returns:
        tuple[float, list[int]]: Best objective value, and the cluster of P that each
        cluster of Q maps to.
    """
    K_Q = Q.shape[1]
    K_P = P.shape[1]
    if K_Q - K_P != 1:
        raise ValueError("K of Q must be exactly one more than K of P.")

    best_obj = np.inf
    idxQ2P: list[int] = []

    for pair in combinations(np.arange(K_Q), 2):
        old_idx = np.delete(np.arange(K_Q), [pair[0], pair[1]])
        merged_idx = K_Q - 2
        Q_comb = np.hstack([Q[:, old_idx], np.expand_dims(Q[:, pair[0]] + Q[:, pair[1]], 1)])

        opt_obj, idxQ2P_comb = align_ILP(P, Q_comb)
        idxQ2P_comb = np.array(idxQ2P_comb)

        if opt_obj < best_obj:
            best_obj = opt_obj
            mapping = np.zeros(K_Q, dtype=int)
            mapping[old_idx] = idxQ2P_comb[np.arange(len(old_idx))]
            mapping[pair[0]] = idxQ2P_comb[merged_idx]
            mapping[pair[1]] = idxQ2P_comb[merged_idx]
            idxQ2P = [int(i) for i in mapping]

    return float(best_obj), idxQ2P


def alignQ_wrtP(P: np.ndarray, Q: np.ndarray, idxQ2P: list[int], merge: bool = True) -> tuple[np.ndarray, list[int]]:
    """
    Description:
    Reorders the clusters of Q to follow the clusters of P. Merging sums every cluster of
    Q that maps to the same cluster of P; otherwise only the closest one keeps that slot
    and the rest are appended as extra columns.

    Args:
        P (np.ndarray): Reference membership matrix of shape (N_ind, K_P).
        Q (np.ndarray): Membership matrix to reorder, of shape (N_ind, K_Q).
        idxQ2P (list[int]): Cluster of P that each cluster of Q maps to.
        merge (bool): Whether clusters mapping to the same slot are summed.

    Returns:
        tuple[np.ndarray, list[int]]: Reordered matrix, and the column each cluster of Q
        ended up in.
    """
    if P.shape[0] != Q.shape[0]:
        raise ValueError("P and Q must have the same number of rows (individuals).")
    if P.shape[1] > Q.shape[1]:
        raise ValueError("Q must have no fewer columns (clusters) than P.")

    if merge:
        aligned_Q = np.zeros_like(P)
        new_pattern = list(idxQ2P)
        for q_idx in range(Q.shape[1]):
            aligned_Q[:, idxQ2P[q_idx]] += Q[:, q_idx]
    else:
        aligned_Q = np.zeros_like(Q)
        dups = np.unique([i for i in idxQ2P if idxQ2P.count(i) > 1])
        extras: list[int] = []
        dups_min: dict[int, tuple[float, int]] = defaultdict(lambda: (float('inf'), -1))

        new_pattern = [0 for _ in range(Q.shape[1])]
        for q_idx in range(Q.shape[1]):
            p_idx = idxQ2P[q_idx]
            if p_idx not in dups:
                new_pattern[q_idx] = p_idx
                aligned_Q[:, p_idx] = Q[:, q_idx]
            else:
                diff = np.linalg.norm(Q[:, q_idx] - P[:, p_idx])
                if dups_min[p_idx][0] > diff:
                    dups_min[p_idx] = (float(diff), q_idx)

        extra_cnt = P.shape[1]
        for q_idx in range(Q.shape[1]):
            p_idx = idxQ2P[q_idx]
            if p_idx in dups:
                if q_idx == dups_min[p_idx][1]:
                    new_pattern[q_idx] = p_idx
                    aligned_Q[:, p_idx] = Q[:, q_idx]
                else:
                    new_pattern[q_idx] = extra_cnt
                    extras.append(q_idx)
                    extra_cnt += 1

        for ie, e in enumerate(extras):
            aligned_Q[:, P.shape[1] + ie] = Q[:, e]

    validate_membership_matrix(aligned_Q)
    return aligned_Q, new_pattern


def align_within_k(Q_list: list[np.ndarray]) -> list[list]:
    """
    Description:
    Aligns every pair of runs that share the same number of clusters and records the cost
    of each pairwise alignment.

    Args:
        Q_list (list[np.ndarray]): Membership matrices of the runs for a single K.

    Returns:
        list[list]: One entry per pair, holding the pair indices, the cost and the mapping.
    """
    out_summary: list[list] = []
    n_ids = len(Q_list)
    if n_ids < 2:
        log.debug("Only one run available; no alignment within K is performed.")
        return out_summary

    for i in range(n_ids - 1):
        P = Q_list[i]
        for j in range(i + 1, n_ids):
            Q = Q_list[j]
            _, idxQ2P = align_ILP(P, Q)
            aligned_Q, _ = alignQ_wrtP(P, Q, idxQ2P, merge=True)
            out_summary.append([(i, j), cost_membership(P, aligned_Q), idxQ2P])

    return out_summary


def align_within_k_all_K(K_range: list[int], Q_by_K: dict[int, list[np.ndarray]]) -> tuple[list[dict], list[dict]]:
    """
    Description:
    Aligns the runs of every K value against each other.

    Args:
        K_range (list[int]): K values to process, in ascending order.
        Q_by_K (dict[int, list[np.ndarray]]): Membership matrices grouped by K.

    Returns:
        tuple[list[dict], list[dict]]: Per K, the mapping of each ordered pair of runs and
        the cost of each unordered pair.
    """
    alignment_withinK_list: list[dict] = []
    cost_withinK_list: list[dict] = []

    for K in K_range:
        alignment: dict[tuple[int, int], list[int]] = {}
        cost: dict[tuple[int, int], float] = {}
        for (i, j), pair_cost, pattern in align_within_k(Q_by_K[K]):
            alignment[(i, j)] = pattern
            alignment[(j, i)] = reverse_alignment_same_K(pattern)
            cost[(i, j)] = pair_cost
        alignment_withinK_list.append(alignment)
        cost_withinK_list.append(cost)
        log.debug(f"Completed {len(cost)} pairwise alignments within K={K}.")

    return alignment_withinK_list, cost_withinK_list


# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

def construct_cost_mat(cost_withinK: dict, n_runs: int) -> np.ndarray:
    """
    Description:
    Assembles the symmetric matrix of pairwise alignment costs between the runs of one K.

    Args:
        cost_withinK (dict): Cost of each unordered pair of runs.
        n_runs (int): Number of runs for this K.

    Returns:
        np.ndarray: Square cost matrix with a zero diagonal.
    """
    cost_mat = np.zeros((n_runs, n_runs))
    for (i, j), value in cost_withinK.items():
        cost_mat[i, j] = value
        cost_mat[j, i] = value

    if not np.all(np.isfinite(cost_mat)):
        raise ValueError("Cost matrix contains non-finite values (NaN or Inf).")
    if np.any(cost_mat < 0):
        raise ValueError("Cost matrix contains negative values.")
    if np.any(cost_mat > 1):
        raise ValueError("Cost matrix contains values above 1.")
    return cost_mat


def cost_to_adj(cost_mat: np.ndarray, norm: bool) -> np.ndarray:
    """
    Description:
    Turns pairwise costs into edge weights, so that the cheapest pair becomes the
    strongest edge.

    Args:
        cost_mat (np.ndarray): Square cost matrix.
        norm (bool): Whether to rescale the off-diagonal costs to span [0, 1].

    Returns:
        np.ndarray: Adjacency matrix with unit self-weights.
    """
    if norm:
        mask = ~np.eye(cost_mat.shape[0], dtype=bool)
        cost_min = np.min(cost_mat[mask])
        cost_max = np.max(cost_mat[mask])
        adj_mat = 1 - (cost_mat - cost_min) / (cost_max - cost_min)
    else:
        adj_mat = 1 - cost_mat
    np.fill_diagonal(adj_mat, 1)
    return adj_mat


def standardize_matrix(W: np.ndarray) -> np.ndarray:
    """
    Description:
    Centres and scales the off-diagonal entries of a square matrix, leaving zeros on the
    diagonal.

    Args:
        W (np.ndarray): Square matrix.

    Returns:
        np.ndarray: Standardized matrix.
    """
    non_diag_idx = np.where(~np.eye(W.shape[0], dtype=bool))
    non_diag_w = W[non_diag_idx]
    W_standardized = np.zeros(W.shape)
    W_standardized[non_diag_idx] = (non_diag_w - non_diag_w.mean()) / non_diag_w.std()
    return W_standardized


def tracy_widom_cdf(x: np.ndarray, beta: int = 1) -> np.ndarray:
    """
    Description:
    Evaluates the Tracy-Widom cumulative distribution function of the requested index.

    Args:
        x (np.ndarray): Points at which the distribution is evaluated.
        beta (int): Index of the law, one of 1, 2 or 4.

    Returns:
        np.ndarray: Cumulative probabilities at the requested points.
    """
    if beta not in _TRACY_WIDOM_FITS:
        raise ValueError(f"Unsupported Tracy-Widom index beta={beta}; expected 1, 2 or 4.")
    shape, scale, shift = _TRACY_WIDOM_FITS[beta]
    return _kernels.tracy_widom_cdf(np.ascontiguousarray(x, dtype=np.float64), shape, scale, shift)


def test_comm_struc(W: np.ndarray, alpha: float = 0.01) -> bool:
    """
    Description:
    Tests whether an adjacency matrix carries community structure, by checking its extreme
    eigenvalues against the Tracy-Widom bulk expected under no structure.

    Args:
        W (np.ndarray): Square adjacency matrix.
        alpha (float): Significance level of the test.

    Returns:
        bool: True when the eigenvalues fall outside the null interval.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("Input matrix W must be square (2D NumPy array).")

    def normalize(A: np.ndarray) -> np.ndarray:
        return A / np.sqrt(A.shape[0])

    T = normalize(standardize_matrix(W))

    non_diag_idx = np.where(~np.eye(W.shape[0], dtype=bool))
    W_exp = np.zeros(W.shape)
    W_exp[non_diag_idx] = np.exp(W[non_diag_idx] * 0.5)
    Te = normalize(standardize_matrix(W_exp))

    x = np.linspace(-10, 10, 1001)
    cdf = tracy_widom_cdf(x, beta=1)
    CI_max = x[np.where(cdf > (1 - alpha / 4))[0][0]]
    CI_min = x[np.where(cdf < alpha / 4)[0][-1]]

    eig_T = np.linalg.eigvalsh(T)
    eig_Te = np.linalg.eigvalsh(Te)

    inside = (int(eig_T.max() < CI_max) + int(eig_T.min() > CI_min)
              + int(eig_Te.max() < CI_max) + int(eig_Te.min() > CI_min))
    return inside != 4


def _renumber(labels: np.ndarray) -> np.ndarray:
    """
    Description:
    Relabels a partition so that community identifiers are consecutive from zero.

    Args:
        labels (np.ndarray): Community label of each node.

    Returns:
        np.ndarray: Relabelled partition.
    """
    _, renumbered = np.unique(labels, return_inverse=True)
    return renumbered.astype(np.intp).reshape(-1)


def _aggregate(W: np.ndarray, partition: np.ndarray, n_com: int) -> np.ndarray:
    """
    Description:
    Collapses every community into a single node, keeping the internal edge weight as a
    self-loop so that the total weight of the graph is preserved.

    Args:
        W (np.ndarray): Weight matrix of the current level.
        partition (np.ndarray): Community label of each node.
        n_com (int): Number of communities.

    Returns:
        np.ndarray: Weight matrix of the aggregated graph.
    """
    indicator = np.zeros((W.shape[0], n_com))
    indicator[np.arange(W.shape[0]), partition] = 1.0
    aggregated = indicator.T @ W @ indicator
    loops = indicator.T @ np.diag(W)
    np.fill_diagonal(aggregated, (np.diag(aggregated) + loops) / 2.0)
    return np.ascontiguousarray(aggregated)


def louvain_communities(adj_mat: np.ndarray, resolution: float = 1.0, seed: int = 0) -> list[int]:
    """
    Description:
    Partitions a weighted graph with the Louvain method, alternating local moves that
    raise modularity with aggregation of the communities found so far.

    The original calls cdlib, which shuffles nodes with NumPy's global generator; here the
    visiting order comes from a seeded generator, so repeated runs on the same input give
    the same partition.

    Args:
        adj_mat (np.ndarray): Symmetric adjacency matrix.
        resolution (float): Resolution parameter; larger values favour smaller communities.
        seed (int): Seed for the node visiting order.

    Returns:
        list[int]: Community label of each node.
    """
    W = np.array(adj_mat, dtype=np.float64, order='C')
    np.fill_diagonal(W, 0.0)

    labels, modularity = _kernels.one_level(W, resolution, seed)
    partition = _renumber(labels)
    result = partition
    W = _aggregate(W, partition, int(partition.max()) + 1)

    while True:
        labels, new_modularity = _kernels.one_level(W, resolution, seed)
        if new_modularity - modularity < 1e-7:
            break
        partition = _renumber(labels)
        modularity = new_modularity
        result = partition[result]
        W = _aggregate(W, partition, int(partition.max()) + 1)

    return [int(c) for c in result]


def detect_communities(cost_mat: np.ndarray, test_comm: bool = True,
                       res: float = 1.0, min_threshold: float = 1e-4,
                       max_threshold: float = 1e-2) -> list[int]:
    """
    Description:
    Groups the runs of one K into modes, short-circuiting to a single mode when all runs
    agree and to singleton modes when none of them do.

    Args:
        cost_mat (np.ndarray): Square matrix of pairwise alignment costs.
        test_comm (bool): Whether to test for community structure before partitioning.
        res (float): Resolution parameter for community detection.
        min_threshold (float): Cost below which two runs are considered identical.
        max_threshold (float): Cost above which two runs are considered unrelated.

    Returns:
        list[int]: Community label of each run.
    """
    n_nodes = cost_mat.shape[0]
    if n_nodes == 0:
        raise ValueError("Cost matrix is empty.")
    if n_nodes == 1:
        return [0]

    mask = ~np.eye(n_nodes, dtype=bool)
    cost_min = np.min(cost_mat[mask])
    cost_max = np.max(cost_mat[mask])

    if n_nodes == 2:
        return [0, 0] if cost_mat[0, 1] < min_threshold else [0, 1]
    if cost_max < min_threshold:
        return [0] * n_nodes
    if cost_min > max_threshold:
        return list(range(n_nodes))

    adj_mat = cost_to_adj(cost_mat, norm=True)
    if test_comm and not test_comm_struc(adj_mat, alpha=0.01):
        log.debug("No significant community structure detected; returning a single mode.")
        return [0] * n_nodes

    communities = louvain_communities(adj_mat, resolution=res)
    if not communities:
        return list(range(n_nodes))

    # Number the communities from the largest to the smallest.
    counts = Counter(communities)
    order = sorted(counts.keys(), key=lambda c: (-counts[c], communities.index(c)))
    mapping = {old: new for new, old in enumerate(order)}
    return [mapping[c] for c in communities]


def community_labels_to_modes(communities: list[int]) -> list[list[int]]:
    """
    Description:
    Converts per-run community labels into the list of runs belonging to each mode.

    Args:
        communities (list[int]): Community label of each run.

    Returns:
        list[list[int]]: Run indices grouped by mode.
    """
    if not communities:
        return []
    return [[j for j, c in enumerate(communities) if c == i] for i in range(max(communities) + 1)]


def detect_modes_all_K(K_range: list[int], cost_withinK_list: list[dict], n_runs_per_K: list[int],
                       test_comm: bool = True, res: float = 1.0,
                       comm_min: float = 1e-4, comm_max: float = 1e-2) -> tuple[list, list]:
    """
    Description:
    Detects the modes of every K value from the pairwise alignment costs of its runs.

    Args:
        K_range (list[int]): K values to process.
        cost_withinK_list (list[dict]): Per K, the cost of each unordered pair of runs.
        n_runs_per_K (list[int]): Number of runs available for each K.
        test_comm (bool): Whether to test for community structure before partitioning.
        res (float): Resolution parameter for community detection.
        comm_min (float): Cost below which two runs are considered identical.
        comm_max (float): Cost above which two runs are considered unrelated.

    Returns:
        tuple[list, list]: Modes of each K, and the cost matrix of each K.
    """
    modes_all_K = []
    cost_matrices = []

    for i_K, K in enumerate(K_range):
        cost_mat = construct_cost_mat(cost_withinK_list[i_K], n_runs_per_K[i_K])
        cost_matrices.append(cost_mat)
        communities = detect_communities(cost_mat, test_comm=test_comm, res=res,
                                         min_threshold=comm_min, max_threshold=comm_max)
        modes = community_labels_to_modes(communities)
        modes_all_K.append(modes)
        log.debug(f"Detected {len(modes)} modes for K={K}.")

    return modes_all_K, cost_matrices


def find_repr_modes(modes: list[list[int]], cost_mat: np.ndarray) -> list[int]:
    """
    Description:
    Picks the run that best represents each mode, namely the one closest to all the other
    runs of that mode.

    Args:
        modes (list[list[int]]): Run indices grouped by mode.
        cost_mat (np.ndarray): Square matrix of pairwise alignment costs.

    Returns:
        list[int]: Representative run of each mode.
    """
    repr_modes = []
    for all_indices in modes:
        if len(all_indices) == 1:
            repr_modes.append(all_indices[0])
        else:
            idx = np.array(all_indices)
            comm_cost_mat = cost_mat[idx, :][:, idx]
            repr_modes.append(all_indices[int(np.argmin(comm_cost_mat.sum(axis=0)))])
    return repr_modes


def extract_modes_and_stats(modes: list[list[int]], cost_mat: np.ndarray, Q_list: list[np.ndarray],
                            alignment_withinK: dict, label_prefix: str = "M") -> dict:
    """
    Description:
    Summarises each mode, keeping its representative run, the average of its runs once
    aligned to that representative, and how tightly its runs agree.

    Args:
        modes (list[list[int]]): Run indices grouped by mode.
        cost_mat (np.ndarray): Square matrix of pairwise alignment costs.
        Q_list (list[np.ndarray]): Membership matrices of the runs for this K.
        alignment_withinK (dict): Mapping of each ordered pair of runs.
        label_prefix (str): Prefix used to name the modes.

    Returns:
        dict: Modes, representatives, per-mode statistics and membership matrices.
    """
    repr_modes = find_repr_modes(modes, cost_mat)

    repQ_modes = []
    avgQ_modes = []
    mode_stats = []

    for mode_idx, all_indices in enumerate(modes):
        mode_label = f"{label_prefix}{mode_idx + 1}"
        i_ref = repr_modes[mode_idx]
        Q_ref = Q_list[i_ref]
        repQ_modes.append(Q_ref)

        if len(all_indices) == 1:
            mode_stats.append({'Mode': mode_label, 'Representative': i_ref,
                               'Size': 1, 'Cost': 0.0, 'Performance': 1.0})
            avgQ_modes.append(Q_ref)
            continue

        idx = np.array(all_indices)
        comm_cost_mat = cost_mat[idx, :][:, idx]
        mode_stats.append({
            'Mode': mode_label,
            'Representative': i_ref,
            'Size': len(all_indices),
            'Cost': avg_tril(comm_cost_mat),
            'Performance': avg_tril(C2Gprime(comm_cost_mat)),
        })

        Q_sum = np.zeros_like(Q_ref)
        for r in all_indices:
            if r == i_ref:
                Q_sum += Q_ref
            else:
                aligned_Q, _ = alignQ_wrtP(Q_ref, Q_list[r], alignment_withinK[(i_ref, r)], merge=True)
                Q_sum += aligned_Q
        avgQ_modes.append(Q_sum / len(all_indices))

    return {'modes': modes, 'repr_modes': repr_modes, 'mode_stats': mode_stats,
            'repQ_modes': repQ_modes, 'avgQ_modes': avgQ_modes}


def extract_modes_all_K(K_range: list[int], Q_by_K: dict[int, list[np.ndarray]],
                        cost_matrices_list: list[np.ndarray], modes_all_K_list: list[list],
                        alignment_withinK_list: list[dict]) -> list[dict]:
    """
    Description:
    Summarises the modes of every K value.

    Args:
        K_range (list[int]): K values to process.
        Q_by_K (dict[int, list[np.ndarray]]): Membership matrices grouped by K.
        cost_matrices_list (list[np.ndarray]): Cost matrix of each K.
        modes_all_K_list (list[list]): Modes of each K.
        alignment_withinK_list (list[dict]): Per K, the mapping of each ordered pair of runs.

    Returns:
        list[dict]: Mode summary of each K.
    """
    return [
        extract_modes_and_stats(modes_all_K_list[i_K], cost_matrices_list[i_K], Q_by_K[K],
                                alignment_withinK_list[i_K], label_prefix=f"K{K}M")
        for i_K, K in enumerate(K_range)
    ]


def get_modes_all_K(K_range: list[int], cd_res: list[dict]) -> tuple[list, list, list]:
    """
    Description:
    Splits the mode summaries into the names, representative matrices and average matrices
    of each K value.

    Args:
        K_range (list[int]): K values to process.
        cd_res (list[dict]): Mode summary of each K.

    Returns:
        tuple[list, list, list]: Mode names, representative matrices and average matrices.
    """
    mode_names_list = [[stat['Mode'] for stat in cd_res[i_K]['mode_stats']] for i_K in range(len(K_range))]
    Q_rep_modes_list = [cd_res[i_K]['repQ_modes'] for i_K in range(len(K_range))]
    Q_avg_modes_list = [cd_res[i_K]['avgQ_modes'] for i_K in range(len(K_range))]
    return mode_names_list, Q_rep_modes_list, Q_avg_modes_list


def get_mode_prop(cd_res: list[dict], key: str) -> dict:
    """
    Description:
    Collects one statistic of every mode across all K values, keyed by mode name.

    Args:
        cd_res (list[dict]): Mode summary of each K.
        key (str): Statistic to collect, such as 'Size' or 'Performance'.

    Returns:
        dict: Requested statistic of each mode.
    """
    return {stat['Mode']: stat[key] for cd_res_K in cd_res for stat in cd_res_K['mode_stats']}


# ---------------------------------------------------------------------------
# Alignment across K
# ---------------------------------------------------------------------------

def align_across_k(K_range: list[int], Q_list_list: list[list[np.ndarray]],
                   Q_labels_list: list[list[str]], merge: bool = True) -> tuple[dict, dict, list[str]]:
    """
    Description:
    Aligns the modes of consecutive K values, and of the modes within each K, recording the
    cost of every pairing and the cheapest pair joining each pair of K values.

    Args:
        K_range (list[int]): K values to process, in ascending order.
        Q_list_list (list[list[np.ndarray]]): Membership matrix of each mode, grouped by K.
        Q_labels_list (list[list[str]]): Name of each mode, grouped by K.
        merge (bool): Whether a single extra cluster is aligned by enumerating merges.

    Returns:
        tuple[dict, dict, list[str]]: Mapping of each mode pair, cost of each mode pair, and
        the cheapest pair joining each pair of consecutive K values.
    """
    if len(K_range) != len(Q_list_list) or len(K_range) != len(Q_labels_list):
        raise ValueError("K_range does not match the length of the provided mode lists.")

    K_range_sorted = sorted(K_range, reverse=True)
    K_comb = [(K_range_sorted[i], K_range_sorted[i + 1]) for i in range(len(K_range_sorted) - 1)]
    K_comb.extend([(K, K) for K in K_range_sorted])

    alignment_acrossK: dict[str, list[int]] = {}
    cost_acrossK: dict[str, float] = {}
    best_alignments: dict[tuple[int, int], tuple[int, int]] = {}

    for K1, K2 in K_comb:
        i_K1 = K_range.index(K1)
        i_K2 = K_range.index(K2)

        best_alignment_idx = 0
        best_alignment_obj = np.inf

        rijs = list(product(range(len(Q_list_list[i_K1])), range(len(Q_list_list[i_K2]))))
        for i, (ri, rj) in enumerate(rijs):
            Q1 = Q_list_list[i_K1][ri]
            Q2 = Q_list_list[i_K2][rj]

            if K1 == K2 and rj == ri:
                opt_obj = 0.0
                idxQ2P = list(range(K1))
            elif merge and (K1 - K2) == 1:
                opt_obj, idxQ2P = align_ILP_diff1(Q2, Q1)
            else:
                opt_obj, idxQ2P = align_ILP(Q2, Q1)

            if best_alignment_obj > opt_obj:
                best_alignment_obj = opt_obj
                best_alignment_idx = i

            aligned_Q, _ = alignQ_wrtP(Q2, Q1, idxQ2P, merge=True)
            pair_label = f"{Q_labels_list[i_K2][rj]}-{Q_labels_list[i_K1][ri]}"
            cost_acrossK[pair_label] = cost_membership(Q2, aligned_Q)
            alignment_acrossK[pair_label] = idxQ2P

        best_alignments[(K1, K2)] = rijs[best_alignment_idx]

    best_pairs = []
    for ii in range(len(K_range_sorted) - 1):
        K1, K2 = K_range_sorted[ii], K_range_sorted[ii + 1]
        ri, rj = best_alignments[(K1, K2)]
        best_pairs.append(f"{Q_labels_list[K_range.index(K2)][rj]}-{Q_labels_list[K_range.index(K1)][ri]}")

    return alignment_acrossK, cost_acrossK, best_pairs


def reorderQ_across_k(K_range: list[int], Q_modes_list: list[list[np.ndarray]],
                      mode_names_list: list[list[str]], alignment_acrossK: dict,
                      anchor_pairs: list[str]) -> tuple[dict, dict]:
    """
    Description:
    Propagates a single cluster ordering through every mode of every K, following the chain
    of cheapest pairs so that a cluster keeps its colour as K grows.

    Args:
        K_range (list[int]): K values to process, in ascending order.
        Q_modes_list (list[list[np.ndarray]]): Membership matrix of each mode, grouped by K.
        mode_names_list (list[list[str]]): Name of each mode, grouped by K.
        alignment_acrossK (dict): Mapping of each mode pair.
        anchor_pairs (list[str]): Cheapest pair joining each pair of consecutive K values.

    Returns:
        tuple[dict, dict]: Reordered matrix of each mode, and the ordering applied to it.
    """
    anchor_pairs_rev = anchor_pairs[::-1]
    all_modes_alignment = {lb: [] for i_K in range(len(K_range)) for lb in mode_names_list[i_K]}
    aligned_Qs_allK = {}
    base_patterns = {}

    i_K = 0
    K = K_range[i_K]
    mode_names = mode_names_list[i_K]
    if len(anchor_pairs) > 0:
        m1 = anchor_pairs_rev[0].split("-")[0]
        m_m1 = mode_names.index(m1)
        base_Q = Q_modes_list[i_K][m_m1]
        base_patterns[m1] = list(range(K))
        all_modes_alignment[m1] = list(range(base_Q.shape[1]))
    else:
        m_m1 = 0
        m1 = mode_names[m_m1]
        base_Q = Q_modes_list[i_K][0]
        all_modes_alignment[m1] = list(range(base_Q.shape[1]))
    aligned_Qs_allK[m1] = base_Q

    for m in range(len(Q_modes_list[i_K])):
        if m == m_m1:
            continue
        m2 = mode_names_list[i_K][m]
        ali_pat = alignment_acrossK[f"{m1}-{m2}"]
        Q = Q_modes_list[i_K][m]
        aligned_Q = np.zeros_like(Q)
        for q_idx in range(Q.shape[1]):
            aligned_Q[:, ali_pat[q_idx]] += Q[:, q_idx]
        base_patterns[m2] = ali_pat
        all_modes_alignment[m2] = ali_pat
        aligned_Qs_allK[m2] = aligned_Q

    for i_pair, pair in enumerate(anchor_pairs_rev):
        m1, m2 = pair.split("-")
        i_K = i_pair + 1
        m_m2 = mode_names_list[i_K].index(m2)
        Q = Q_modes_list[i_K][m_m2]
        m1_K = K_range[i_pair]
        P = Q_modes_list[i_pair][mode_names_list[i_pair].index(m1)]

        _, new_pattern = alignQ_wrtP(P, Q, alignment_acrossK[pair], merge=False)
        pat = [base_patterns[m1][i] if i < m1_K else i for i in new_pattern]
        aligned_Q = np.zeros_like(Q)
        for q_idx in range(Q.shape[1]):
            aligned_Q[:, pat[q_idx]] += Q[:, q_idx]
        base_patterns[m2] = pat
        all_modes_alignment[m2] = pat
        aligned_Qs_allK[m2] = aligned_Q

        for m in range(len(Q_modes_list[i_K])):
            if m == m_m2:
                continue
            m3 = mode_names_list[i_K][m]
            ali_pat = alignment_acrossK[f"{m2}-{m3}"]
            Q = Q_modes_list[i_K][m]
            _, new_pattern = alignQ_wrtP(Q_modes_list[i_K][m_m2], Q, ali_pat, merge=False)
            pat = [base_patterns[m2][i] for i in new_pattern]
            aligned_Q = np.zeros_like(Q)
            for q_idx in range(Q.shape[1]):
                aligned_Q[:, pat[q_idx]] += Q[:, q_idx]
            base_patterns[m3] = pat
            all_modes_alignment[m3] = pat
            aligned_Qs_allK[m3] = aligned_Q

    for m, orig in all_modes_alignment.items():
        all_modes_alignment[m] = [orig.index(ii) for ii in range(len(orig))]

    return aligned_Qs_allK, all_modes_alignment


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def load_default_cmap(K: int) -> list[tuple[float, float, float]]:
    """
    Description:
    Returns the first K colours of the default cluster palette.

    Args:
        K (int): Number of colours requested.

    Returns:
        list[tuple[float, float, float]]: Colours as RGB triples.
    """
    return [mcolors.to_rgb(c) for c in DEFAULT_COLORS[:K]]


def parse_custom_cmap(colors: list, K: int) -> list[tuple[float, float, float]]:
    """
    Description:
    Validates a user-supplied palette and returns its first K colours.

    Args:
        colors (list): Colour specifications accepted by matplotlib.
        K (int): Number of colours requested.

    Returns:
        list[tuple[float, float, float]]: Colours as RGB triples.
    """
    if len(colors) < K:
        raise ValueError(f"Custom color map has only {len(colors)} colors, but K={K} is requested.")
    if not all(mcolors.is_color_like(c) for c in colors):
        raise ValueError("Invalid color code(s) found in the custom colors provided.")
    return [mcolors.to_rgb(c) for c in colors[:K]]


def labels_are_grouped(labels: list, uniq_labels: list) -> bool:
    """
    Description:
    Checks that all individuals sharing a label sit next to each other.

    Args:
        labels (list): Label of each individual.
        uniq_labels (list): Distinct labels.

    Returns:
        bool: True when every label occupies a contiguous block.
    """
    for lb in uniq_labels:
        positions = [i for i, val in enumerate(labels) if val == lb]
        if positions and (max(positions) - min(positions) + 1 != len(positions)):
            return False
    return True


def get_uniq_lb_sep(labels: list) -> tuple[list, list, list]:
    """
    Description:
    Locates where each label group starts and ends along the x axis, and where its tick
    should sit.

    Args:
        labels (list): Label of each individual.

    Returns:
        tuple[list, list, list]: Distinct labels, their tick positions and the boundaries
        between groups.
    """
    uniq_lbs = list(np.unique(labels))
    if not labels_are_grouped(labels, uniq_lbs):
        raise ValueError("Labels are not grouped together.")

    uniq_lbs_indices = []
    uniq_lbs_sep_idx = [-0.5]
    for lb in uniq_lbs:
        indices = [i for i, val in enumerate(labels) if val == lb]
        uniq_lbs_indices.append(float(np.mean(indices)) + 0.5)
        uniq_lbs_sep_idx.append(indices[-1] + 0.5)
    return uniq_lbs, uniq_lbs_indices, uniq_lbs_sep_idx


def reorder_ind_within_group(Q: np.ndarray, lbs: list) -> tuple[np.ndarray, dict]:
    """
    Description:
    Sorts the individuals of each label group by their dominant ancestries, and records the
    cluster order used within each group.

    Args:
        Q (np.ndarray): Membership matrix of shape (N_ind, K).
        lbs (list): Label of each individual.

    Returns:
        tuple[np.ndarray, dict]: Reordered individual indices, and the cluster order of each
        group.
    """
    uniq_lbs = list(np.unique(lbs))
    if not labels_are_grouped(lbs, uniq_lbs):
        raise ValueError("Labels are not grouped together.")

    mbsp_sort_indices = {}
    ind_sorted_indices = np.zeros(len(lbs), dtype=int)
    for lb in uniq_lbs:
        lb_indices = np.array([i for i, val in enumerate(lbs) if val == lb])
        lb_Q = Q[lb_indices, :]
        mbsp_sortidx = np.argsort(-lb_Q.sum(axis=0))
        lb_Q = lb_Q[:, mbsp_sortidx]
        ind_sorted_indices[lb_indices] = lb_indices[np.lexsort(-lb_Q[:, ::-1].T)]
        mbsp_sort_indices[lb] = mbsp_sortidx
    return ind_sorted_indices, mbsp_sort_indices


def plot_membership(Q: np.ndarray, cmap: list, ax=None, ylab: str = "", title: str = "",
                    fontsize: float = 14):
    """
    Description:
    Draws one run as a stacked bar chart, one bar per individual.

    Args:
        Q (np.ndarray): Membership matrix of shape (N_ind, K).
        cmap (list): Colour of each cluster.
        ax: Axes to draw on; a new figure is created when omitted.
        ylab (str): Label placed to the left of the panel.
        title (str): Title placed above the panel.
        fontsize (float): Font size of the label and title.

    Returns:
        matplotlib.axes.Axes: Axes holding the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    N, K = Q.shape
    Q_aug = np.hstack((np.zeros((N, 1)), Q))
    for k in range(K):
        ax.bar(range(N), Q_aug[:, k + 1], bottom=np.sum(Q_aug[:, 0:k + 1], axis=1),
               width=1.0, edgecolor='w', linewidth=0, facecolor=cmap[k])
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("\n".join(ylab.split()) if ylab else "", rotation=0, fontsize=fontsize,
                  labelpad=30, va="center")
    if title:
        ax.set_title(title, fontsize=fontsize, loc="left", pad=5)
    return ax


def plot_membership_reordered(Q: np.ndarray, lbs: list, mbsp_sort_indices: dict, cmap: list,
                              ax=None, ylab: str = "", title: str = "", fontsize: float = 14):
    """
    Description:
    Draws one run as a stacked bar chart with the clusters of each label group stacked from
    the most to the least represented.

    Args:
        Q (np.ndarray): Membership matrix of shape (N_ind, K).
        lbs (list): Label of each individual.
        mbsp_sort_indices (dict): Cluster order of each group.
        cmap (list): Colour of each cluster.
        ax: Axes to draw on; a new figure is created when omitted.
        ylab (str): Label placed to the left of the panel.
        title (str): Title placed above the panel.
        fontsize (float): Font size of the label and title.

    Returns:
        matplotlib.axes.Axes: Axes holding the plot.
    """
    if ax is None:
        _, ax = plt.subplots()

    N, K = Q.shape
    for lb in np.unique(lbs):
        lb_indices = [i for i, val in enumerate(lbs) if val == lb]
        mbsp_sortidx = mbsp_sort_indices[lb]
        lb_P_sorted = Q[lb_indices, :][:, mbsp_sortidx]
        lb_P_aug = np.hstack((np.zeros((lb_P_sorted.shape[0], 1)), lb_P_sorted))
        for k in range(K):
            ax.bar(lb_indices, lb_P_aug[:, k + 1], bottom=np.sum(lb_P_aug[:, 0:k + 1], axis=1),
                   width=1.0, edgecolor='w', linewidth=0, facecolor=cmap[mbsp_sortidx[k]])
    ax.set_xlim(-0.5, N - 0.5)
    ax.set_xticks([])
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("\n".join(ylab.split()) if ylab else "", rotation=0, fontsize=fontsize,
                  labelpad=30, va="center")
    if title:
        ax.set_title(title, fontsize=fontsize, loc="left", pad=5)
    return ax


def _draw_panel(ax, Q: np.ndarray, cmap: list, ind_labels: list, order_refQ, order_cls_by_label: bool,
                ref_ind_sorted_indices, ylab: str = "", title: str = "", fontsize: float = 14) -> None:
    """
    Description:
    Draws a single run panel, optionally sorting individuals like a reference run and
    stacking clusters by how much of each group they explain.

    Args:
        ax: Axes to draw on.
        Q (np.ndarray): Membership matrix of shape (N_ind, K).
        cmap (list): Colour of each cluster.
        ind_labels (list): Label of each individual.
        order_refQ: Reference membership matrix used to sort individuals, or None.
        order_cls_by_label (bool): Whether clusters are stacked per label group.
        ref_ind_sorted_indices: Individual order derived from the reference matrix.
        ylab (str): Label placed to the left of the panel.
        title (str): Title placed above the panel.
        fontsize (float): Font size of the label and title.

    Returns:
        None
    """
    if order_refQ is None or len(ind_labels) == 0:
        if order_cls_by_label:
            _, mbsp_sort_indices = reorder_ind_within_group(Q, ind_labels)
            plot_membership_reordered(Q, ind_labels, mbsp_sort_indices, cmap, ax=ax,
                                      ylab=ylab, title=title, fontsize=fontsize)
        else:
            plot_membership(Q, cmap, ax=ax, ylab=ylab, title=title, fontsize=fontsize)
        return

    if order_cls_by_label:
        _, mbsp_sort_indices = reorder_ind_within_group(Q, ind_labels)
        reordered_ind_labels = [ind_labels[i] for i in ref_ind_sorted_indices]
        plot_membership_reordered(Q[ref_ind_sorted_indices, :], reordered_ind_labels,
                                  mbsp_sort_indices, cmap, ax=ax, ylab=ylab, title=title,
                                  fontsize=fontsize)
    else:
        Q_reordered = Q[ref_ind_sorted_indices, :]
        mbsp_sortidx = np.argsort(-Q_reordered.sum(axis=0))
        plot_membership(Q_reordered[:, mbsp_sortidx], [cmap[i] for i in mbsp_sortidx], ax=ax,
                        ylab=ylab, title=title, fontsize=fontsize)


def _annotate_label_groups(ax, uniq_lbs: list, uniq_lbs_indices: list, uniq_lbs_sep_idx: list,
                           with_ticks: bool) -> None:
    """
    Description:
    Marks the boundaries between label groups and, on the bottom row, writes their names.

    Args:
        ax: Axes to annotate.
        uniq_lbs (list): Distinct labels.
        uniq_lbs_indices (list): Tick position of each label.
        uniq_lbs_sep_idx (list): Boundaries between groups.
        with_ticks (bool): Whether label names are written under the axis.

    Returns:
        None
    """
    for v in uniq_lbs_sep_idx:
        ax.axvline(v, ymin=-0.2, ymax=1, color='k', ls='--', lw=0.5, clip_on=False)
    if with_ticks:
        ax.set_xticks(uniq_lbs_indices)
        ax.tick_params(axis='x', which='both', length=0)
        ax.set_xticklabels(uniq_lbs, rotation=90, ha='center',
                           fontsize=14 if len(uniq_lbs) < 10 else 10)


def plot_memberships_list(Q_list: list[np.ndarray], cmap: list, names: list[str],
                          ind_labels: list | None = None, right_labels: list[str] | None = None,
                          order_refQ: np.ndarray | None = None, order_cls_by_label: bool = True,
                          width_scale: float = 1.0, fontsize: float = 14):
    """
    Description:
    Stacks several runs as one bar chart per row, sharing a common colour scheme.

    Args:
        Q_list (list[np.ndarray]): Membership matrix of each run.
        cmap (list): Colour of each cluster.
        names (list[str]): Label placed to the left of each run.
        ind_labels (list | None): Label of each individual.
        right_labels (list[str] | None): Text placed to the right of each run.
        order_refQ (np.ndarray | None): Reference matrix used to sort individuals.
        order_cls_by_label (bool): Whether clusters are stacked per label group.
        width_scale (float): Horizontal scaling of the figure.
        fontsize (float): Font size of the run labels.

    Returns:
        matplotlib.figure.Figure: Figure holding the plot.
    """
    ind_labels = ind_labels if ind_labels is not None else []
    ref_ind_sorted_indices = None
    if len(ind_labels) > 0:
        uniq_lbs, uniq_lbs_indices, uniq_lbs_sep_idx = get_uniq_lb_sep(ind_labels)
        if order_refQ is not None:
            ref_ind_sorted_indices, _ = reorder_ind_within_group(order_refQ, ind_labels)

    fig, axes = plt.subplots(len(Q_list), 1, figsize=(15 * width_scale, 1.5 * len(Q_list)),
                             facecolor='white')

    for i, Q in enumerate(Q_list):
        ax = axes[i] if len(Q_list) > 1 else axes
        _draw_panel(ax, Q, cmap, ind_labels, order_refQ, order_cls_by_label,
                    ref_ind_sorted_indices, ylab=names[i], fontsize=fontsize)
        right_lb = right_labels[i] if right_labels and i < len(right_labels) else ""
        ax.set_title(right_lb, fontsize=14, loc="right")

        if len(ind_labels) > 0:
            if len(ind_labels) != Q.shape[0]:
                log.warning(f"    {len(ind_labels)} individual labels for {Q.shape[0]} individuals.")
            _annotate_label_groups(ax, uniq_lbs, uniq_lbs_indices, uniq_lbs_sep_idx,
                                   with_ticks=(i == len(Q_list) - 1))

    fig.subplots_adjust(hspace=0.3)
    return fig


def adjust_lightness(color, factor: float = 1.0) -> tuple[float, float, float]:
    """
    Description:
    Lightens or darkens a colour by scaling its lightness channel.

    Args:
        color: Colour specification accepted by matplotlib.
        factor (float): Values above one lighten the colour, below one darken it.

    Returns:
        tuple[float, float, float]: Adjusted colour as an RGB triple.
    """
    h, lightness, s = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    return colorsys.hls_to_rgb(h, max(0, min(1, lightness * factor)), s)


def create_single_cmap(color, name: str = 'custom_cmap', light: float = 1.5, dark: float = 0.6):
    """
    Description:
    Builds a colormap running from a light to a dark shade of one colour.

    Args:
        color: Base colour of the ramp.
        name (str): Name given to the colormap.
        light (float): Lightening factor of the first stop.
        dark (float): Darkening factor of the last stop.

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The colormap.
    """
    return mcolors.LinearSegmentedColormap.from_list(
        name, [adjust_lightness(color, light), mcolors.to_rgb(color), adjust_lightness(color, dark)]
    )


def plot_graph(K_range: list[int], Q_list_list: list[list[np.ndarray]], cmap: list,
               names_list: list[list[str]], labels_list: list[list[str]] | None = None,
               right_labels_list: list[list[str]] | None = None, cost_acrossK: dict | None = None,
               ind_labels: list | None = None, fontsize: float = 14, alt_color: bool = True,
               line_cmap=None, color_alt: list[str] | None = None,
               order_refQ: np.ndarray | None = None, order_cls_by_label: bool = True,
               width_scale: float = 1.0, height_scale: float = 1.0):
    """
    Description:
    Lays out every mode as a bar chart, one row per K value, and connects the modes of
    consecutive K values with edges whose shade reflects their alignment cost.

    Args:
        K_range (list[int]): K values to draw, in ascending order.
        Q_list_list (list[list[np.ndarray]]): Membership matrix of each mode, grouped by K.
        cmap (list): Colour of each cluster.
        names_list (list[list[str]]): Name of each mode, grouped by K.
        labels_list (list[list[str]] | None): Title of each mode panel.
        right_labels_list (list[list[str]] | None): Text placed to the right of each panel.
        cost_acrossK (dict | None): Cost of each mode pair; edges are omitted when None.
        ind_labels (list | None): Label of each individual.
        fontsize (float): Font size of the panel titles.
        alt_color (bool): Whether edge colours alternate to keep crossings readable.
        line_cmap: Colormap used for the edges when they do not alternate.
        color_alt (list[str] | None): Colours cycled through when edges alternate.
        order_refQ (np.ndarray | None): Reference matrix used to sort individuals.
        order_cls_by_label (bool): Whether clusters are stacked per label group.
        width_scale (float): Horizontal scaling of the figure.
        height_scale (float): Vertical scaling of the figure.

    Returns:
        matplotlib.figure.Figure: Figure holding the plot.
    """
    ind_labels = ind_labels if ind_labels is not None else []
    color_alt = color_alt if color_alt is not None else ALT_EDGE_COLORS
    mode_numbers = [len(names) for names in names_list]
    n_row = len(K_range)
    n_col = max(mode_numbers)
    N = Q_list_list[0][0].shape[0]

    ref_ind_sorted_indices = None
    if len(ind_labels) > 0:
        uniq_lbs, uniq_lbs_indices, uniq_lbs_sep_idx = get_uniq_lb_sep(ind_labels)
        if order_refQ is not None:
            ref_ind_sorted_indices, _ = reorder_ind_within_group(order_refQ, ind_labels)

    if labels_list is None:
        labels_list = names_list

    fig = plt.figure(figsize=(7 * n_col * width_scale, 2.5 * n_row * height_scale), dpi=150)
    gs = fig.add_gridspec(n_row, n_col, wspace=0.2, hspace=1)

    axes_handles = {}
    for i_K, K in enumerate(K_range):
        for i_mode in range(mode_numbers[i_K]):
            ax = fig.add_subplot(gs[i_K, i_mode])
            axes_handles[(K, i_mode)] = ax
            _draw_panel(ax, Q_list_list[i_K][i_mode], cmap, ind_labels, order_refQ,
                        order_cls_by_label, ref_ind_sorted_indices,
                        title=labels_list[i_K][i_mode], fontsize=fontsize)
            right_label = right_labels_list[i_K][i_mode] if right_labels_list is not None else ""
            ax.set_title(right_label, fontsize=14, loc="right")
            ax.set_zorder(9)

            if len(ind_labels) > 0:
                if len(ind_labels) != N:
                    log.warning(f"    {len(ind_labels)} individual labels for {N} individuals.")
                _annotate_label_groups(ax, uniq_lbs, uniq_lbs_indices, uniq_lbs_sep_idx,
                                       with_ticks=(i_K == len(K_range) - 1))

    for K in K_range:
        axes_handles[(K, 0)].set_ylabel(f"K={K}", fontsize=18, rotation=0, labelpad=10,
                                        va="center", ha="right", weight='bold')

    if cost_acrossK is None:
        return fig

    textbox_props = {'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.6,
                     'edgecolor': 'none', 'pad': 0.1}
    cmaps_alt = [create_single_cmap(c, c) for c in color_alt] if alt_color else []
    if not alt_color and line_cmap is None:
        line_cmap = mpl.colormaps["Greys"]

    cost_max = max(cost_acrossK.values())
    cost_min = min(cost_acrossK.values())

    for i_K, K in enumerate(K_range[:-1]):
        for i_mode in range(mode_numbers[i_K]):
            for i_mode2 in range(mode_numbers[i_K + 1]):
                pair_label = f"{names_list[i_K][i_mode]}-{names_list[i_K + 1][i_mode2]}"
                if pair_label not in cost_acrossK:
                    continue
                if alt_color:
                    line_cmap = cmaps_alt[(i_mode + i_mode2 + 2 * i_K + 1) % len(cmaps_alt)]

                ax = axes_handles[(K, i_mode)]
                ax2 = axes_handles[(K_range[i_K + 1], i_mode2)]
                cost = float(cost_acrossK[pair_label])
                edge_w = 0.85 - (cost - cost_min) / (cost_max - cost_min) * 0.8

                con = ConnectionPatch(xyA=(0.5, -0.05), coordsA='axes fraction', axesA=ax,
                                      xyB=(0.5, 1.05), coordsB='axes fraction', axesB=ax2,
                                      color=line_cmap(edge_w), lw=4, alpha=1, zorder=0)
                ax2.add_artist(con)

                xyA_fig = fig.transFigure.inverted().transform(ax.transData.transform((N // 2, -0.05)))
                xyB_fig = fig.transFigure.inverted().transform(ax2.transData.transform((N // 2, 1.05)))
                frac = 0.3 if (i_mode + i_mode2) % 2 == 1 else 0.7
                if mode_numbers[i_K] == 1 or mode_numbers[i_K + 1] == 1:
                    frac = 0.5
                mid = ((xyB_fig[0] - xyA_fig[0]) * frac + xyA_fig[0],
                       (xyB_fig[1] - xyA_fig[1]) * frac + xyA_fig[1])
                fig.text(mid[0], mid[1], f"{cost:.3f}", fontsize=14, bbox=textbox_props,
                         ha='center', va='center', clip_on=False, zorder=1000)
                con.set_in_layout(False)

    return fig


def plot_alignment_graph(K_range: list[int], names_list: list[list[str]], cmap: list,
                         alignment_acrossK: dict, all_modes_alignment: dict,
                         anchor_pairs: list[str] | None = None, wspace_padding: float = 1.3,
                         y_aspect: float = 3, alt_color: bool = True,
                         color_alt: list[str] | None = None, ls_alt: list[str] | None = None,
                         marker_size: float = 200, separate_labels: bool = False):
    """
    Description:
    Draws the clusters of every mode as a row of coloured dots per K value, joined by lines
    that trace how each cluster of one K maps onto the clusters of the next.

    Args:
        K_range (list[int]): K values to draw, in ascending order.
        names_list (list[list[str]]): Name of each mode, grouped by K.
        cmap (list): Colour of each cluster.
        alignment_acrossK (dict): Mapping of each mode pair.
        all_modes_alignment (dict): Cluster ordering applied to each mode.
        anchor_pairs (list[str] | None): Pairs drawn with the solid line style.
        wspace_padding (float): Horizontal space left between modes of the same K.
        y_aspect (float): Aspect ratio of the axes.
        alt_color (bool): Whether line colours alternate to keep crossings readable.
        color_alt (list[str] | None): Colours cycled through when lines alternate.
        ls_alt (list[str] | None): Line styles for anchor pairs and for the rest.
        marker_size (float): Size of the cluster markers.
        separate_labels (bool): Whether mode names are written next to each row.

    Returns:
        matplotlib.figure.Figure: Figure holding the plot.
    """
    color_alt = color_alt if color_alt is not None else ALT_EDGE_COLORS
    ls_alt = ls_alt if ls_alt is not None else ALT_LINE_STYLES
    mode_numbers = [len(names) for names in names_list]
    n_row = len(K_range)
    n_col = max(mode_numbers)
    K_max = int(np.max(K_range))
    anchor_pairs_tuple = [tuple(pr.split('-')) for pr in anchor_pairs] if anchor_pairs else []

    start_positions = np.zeros((n_row, n_col), dtype=int)
    for i_K in range(n_row):
        for i_mode in range(n_col):
            start_positions[i_K, i_mode] = i_mode * int(K_max * wspace_padding)

    fig, ax = plt.subplots(1, 1, figsize=(n_col * K_max * 0.3, n_row * 1.5), dpi=150)

    def scatter_modes(i_K: int, K: int) -> None:
        for i_mode in range(mode_numbers[i_K]):
            ordering = all_modes_alignment[names_list[i_K][i_mode]]
            ax.scatter(ordering + start_positions[i_K, i_mode], np.ones(K) * i_K,
                       s=marker_size, linewidths=0.5, edgecolors='k',
                       c=[cmap[ordering[i]] for i in range(K)], zorder=6)

    def highlight_mode(i_K: int, i_mode: int, K: int, mode_name: str) -> None:
        if separate_labels:
            ax.text(start_positions[i_K, i_mode], i_K + 0.2, mode_name.replace('_', ' '),
                    c='gray', fontsize=9, ha='left', va='center')
        ax.add_patch(Rectangle((start_positions[i_K, i_mode] - 0.5, i_K - 0.15), K, 0.3,
                               linewidth=0.5, edgecolor='lightgrey', facecolor='lightgrey',
                               alpha=0.1, joinstyle='round', capstyle='round', zorder=1))

    for i_K, K in enumerate(K_range[:-1]):
        i_K2, K2 = i_K + 1, K_range[i_K + 1]
        scatter_modes(i_K, K)
        for i_mode in range(mode_numbers[i_K]):
            mode_name = names_list[i_K][i_mode]
            for i_mode2 in range(mode_numbers[i_K2]):
                mode_name2 = names_list[i_K2][i_mode2]
                mode_pair = f"{mode_name}-{mode_name2}"
                if mode_pair not in alignment_acrossK:
                    log.warning(f"    Alignment for mode pair {mode_pair} not found.")
                    continue

                mapping = alignment_acrossK[mode_pair]
                reordering_cur = all_modes_alignment[mode_name]
                reordering_next = all_modes_alignment[mode_name2]
                color = color_alt[(i_mode + i_mode2 + i_K + i_K2) % len(color_alt)] if alt_color else 'k'
                ls = ls_alt[0] if (mode_name, mode_name2) in anchor_pairs_tuple else ls_alt[1]

                for kp1 in range(len(mapping)):
                    x0 = reordering_cur.index(mapping[kp1]) + start_positions[i_K, i_mode]
                    x1 = reordering_next.index(kp1) + start_positions[i_K2, i_mode2]
                    if reordering_cur.index(mapping[kp1]) != reordering_next.index(kp1):
                        ax.plot([x0, x1], [i_K + 0.1, i_K2 - 0.1], c=color, ls=ls, lw=0.8, zorder=2)
                    else:
                        ax.plot([x0, x1], [i_K + 0.1, i_K2 - 0.1], c='lightgrey', ls=':', lw=0.3, zorder=2)

                highlight_mode(i_K2, i_mode2, K2, mode_name2)

    scatter_modes(n_row - 1, K_range[-1])
    for i_mode in range(mode_numbers[0]):
        highlight_mode(0, i_mode, K_range[0], names_list[0][i_mode])

    ax.set_xticks([start_positions[0, i_mode] + (K_max - 1) / 2.0 for i_mode in range(n_col)])
    if not separate_labels:
        ax.set_xticklabels([f"M{i_m + 1}" for i_m in range(n_col)], fontsize=14)
    ax.set_xlabel("Modes", fontsize=14)
    ax.set_xlim(-1, (n_col - 1) * int(K_max * wspace_padding) + K_max)

    ax.set_yticks(np.arange(n_row))
    if not separate_labels:
        ax.set_yticklabels([f"K={K}" for K in K_range], rotation=0, fontsize=14, va='center',
                           ha='right', weight='bold')
    ax.set_ylim(-0.5, n_row - 0.5)
    ax.invert_yaxis()
    ax.set_aspect(y_aspect, adjustable='box')
    return fig
