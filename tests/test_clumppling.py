from itertools import product

import numpy as np
import pytest

from adamixture.src import clumppling
from adamixture.src.plot import align_clusters_clumppling
from adamixture.src.utils_c.cython import clumppling as kernels

# Published moments of the Tracy-Widom law of index 1.
TW1_MEAN = -1.2065335745820
TW1_STD = 1.2679320503004


def _random_memberships(rng: np.random.Generator, n_ind: int, K: int) -> np.ndarray:
    return rng.dirichlet(np.ones(K) * 0.4, size=n_ind)


def _brute_force_alignment(P: np.ndarray, Q: np.ndarray) -> float:
    """Minimum cost over every mapping that uses each cluster of P at least once."""
    K_P, K_Q = P.shape[1], Q.shape[1]
    D = np.array([[np.sum((Q[:, q] - P[:, p]) ** 2) / Q.shape[0] for p in range(K_P)]
                  for q in range(K_Q)])
    best = np.inf
    for mapping in product(range(K_P), repeat=K_Q):
        if len(set(mapping)) != K_P:
            continue
        best = min(best, sum(D[q, p] for q, p in enumerate(mapping)))
    return best


@pytest.mark.parametrize("K_P,K_Q", [(2, 2), (2, 4), (3, 3), (3, 5), (4, 6)])
def test_align_ilp_reaches_the_brute_force_optimum(K_P: int, K_Q: int) -> None:
    rng = np.random.default_rng(K_P * 10 + K_Q)
    for _ in range(5):
        P = _random_memberships(rng, 40, K_P)
        Q = _random_memberships(rng, 40, K_Q)

        opt_obj, idxQ2P = clumppling.align_ILP(P, Q)

        assert opt_obj == pytest.approx(_brute_force_alignment(P, Q), abs=1e-12)
        assert len(idxQ2P) == K_Q
        assert set(idxQ2P) == set(range(K_P))


def test_align_ilp_recovers_a_permuted_copy() -> None:
    rng = np.random.default_rng(0)
    P = _random_memberships(rng, 50, 4)
    perm = np.array([2, 0, 3, 1])

    opt_obj, idxQ2P = clumppling.align_ILP(P, P[:, perm])

    assert opt_obj == pytest.approx(0.0, abs=1e-12)
    assert idxQ2P == list(perm)


def test_align_clusters_clumppling_undoes_a_column_permutation() -> None:
    rng = np.random.default_rng(1)
    ref_Q = _random_memberships(rng, 60, 5)
    perm = np.array([3, 1, 4, 0, 2])

    recovered = align_clusters_clumppling(ref_Q, ref_Q[:, perm])

    np.testing.assert_array_equal(ref_Q[:, perm][:, recovered], ref_Q)


def test_solve_lap_matches_exhaustive_assignment() -> None:
    rng = np.random.default_rng(2)
    for _ in range(20):
        n, m = 4, 6
        cost = np.ascontiguousarray(rng.random((n, m)))

        assignment = kernels.solve_lap(cost)

        best = min(sum(cost[i, cols[i]] for i in range(n))
                   for cols in product(range(m), repeat=n) if len(set(cols)) == n)
        assert len(set(assignment)) == n
        assert sum(cost[i, assignment[i]] for i in range(n)) == pytest.approx(best)


def test_louvain_recovers_planted_communities() -> None:
    block = np.full((6, 6), 0.9)
    adjacency = np.full((12, 12), 0.05)
    adjacency[:6, :6] = block
    adjacency[6:, 6:] = block
    np.fill_diagonal(adjacency, 1.0)

    communities = clumppling.louvain_communities(adjacency)

    assert len(set(communities)) == 2
    assert len(set(communities[:6])) == 1
    assert len(set(communities[6:])) == 1
    assert communities[0] != communities[6]


def test_tracy_widom_fit_reproduces_the_published_moments() -> None:
    step = 1e-3
    x = np.arange(-12.0, 8.0, step)
    cdf = clumppling.tracy_widom_cdf(x, beta=1)

    assert np.all(np.diff(cdf) >= 0)
    assert cdf[0] == pytest.approx(0.0, abs=1e-6)
    assert cdf[-1] == pytest.approx(1.0, abs=1e-6)

    density = np.diff(cdf) / step
    midpoints = x[:-1] + step / 2
    mean = np.sum(midpoints * density) * step
    variance = np.sum((midpoints - mean) ** 2 * density) * step

    assert mean == pytest.approx(TW1_MEAN, abs=1e-3)
    assert np.sqrt(variance) == pytest.approx(TW1_STD, abs=1e-3)


def _replicate_runs(rng: np.random.Generator, n_ind: int, K: int, n_modes: int,
                    n_reps: int) -> list[np.ndarray]:
    """Runs that fall into n_modes well-separated groups, with shuffled cluster labels."""
    bases = [_random_memberships(rng, n_ind, K) for _ in range(n_modes)]
    runs = []
    for i in range(n_modes * n_reps):
        Q = np.clip(bases[i % n_modes] + rng.normal(0, 1e-3, size=(n_ind, K)), 1e-9, None)
        Q /= Q.sum(axis=1, keepdims=True)
        runs.append(Q[:, rng.permutation(K)])
    return runs


def test_mode_detection_groups_replicates_of_the_same_solution() -> None:
    rng = np.random.default_rng(3)
    K_range = [2, 3]
    Q_by_K = {K: _replicate_runs(rng, 80, K, n_modes=2, n_reps=5) for K in K_range}
    n_runs_per_K = [len(Q_by_K[K]) for K in K_range]

    alignment_withinK, cost_withinK = clumppling.align_within_k_all_K(K_range, Q_by_K)
    modes, cost_matrices = clumppling.detect_modes_all_K(K_range, cost_withinK, n_runs_per_K)

    # Replicates alternate between the two solutions, so each mode collects every other run.
    for modes_K in modes:
        assert modes_K == [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]

    cd_res = clumppling.extract_modes_all_K(K_range, Q_by_K, cost_matrices, modes, alignment_withinK)
    mode_names_list, Q_rep_modes_list, _ = clumppling.get_modes_all_K(K_range, cd_res)

    assert mode_names_list == [["K2M1", "K2M2"], ["K3M1", "K3M2"]]
    assert clumppling.get_mode_prop(cd_res, 'Size') == {"K2M1": 5, "K2M2": 5, "K3M1": 5, "K3M2": 5}

    alignment_acrossK, cost_acrossK, anchor_pairs = clumppling.align_across_k(
        K_range, Q_rep_modes_list, mode_names_list, merge=True
    )
    assert len(anchor_pairs) == len(K_range) - 1

    aligned_Qs, orderings = clumppling.reorderQ_across_k(
        K_range, Q_rep_modes_list, mode_names_list, alignment_acrossK, anchor_pairs
    )
    for i_K, K in enumerate(K_range):
        for mode_name in mode_names_list[i_K]:
            assert sorted(orderings[mode_name]) == list(range(K))
            np.testing.assert_allclose(aligned_Qs[mode_name].sum(axis=1), 1.0, atol=1e-9)


def test_mode_graph_is_written_for_a_multi_k_filemap(tmp_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from adamixture.src.plot import plot_clumppling_mode_graph

    rng = np.random.default_rng(4)
    all_qs = []
    for K in (2, 3):
        for rep, Q in enumerate(_replicate_runs(rng, 40, K, n_modes=1, n_reps=2), 1):
            all_qs.append({'id': f"run{K}_{rep}", 'K': K, 'Q': Q})

    output_path = tmp_path / "modes.png"
    plot_clumppling_mode_graph(
        all_qs=all_qs,
        output_path=output_path,
        labels=None,
        custom_colors=None,
        dpi=80,
        format='png',
        comm_min=1e-4,
        comm_max=1e-2,
        test_comm=True,
        cd_res=1.0,
        cd_method='louvain',
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
