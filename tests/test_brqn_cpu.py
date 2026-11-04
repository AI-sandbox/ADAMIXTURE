import numpy as np
import pytest

from adamixture.model.als import ALS
from adamixture.model.br_qn import optimize_original
from adamixture.src.adamixture import run_als
from adamixture.src.compat import Device
from adamixture.src.utils_c import tools

from tests.config import (BETA_ALS, CHUNK_SIZE, CPU_BRQN_EXPECTED_DIR, EXPECTED_ATOL, EXPECTED_RTOL, K,
                          MAX_ALS, SEED, TOL_ALS)
from tests.helpers import assert_matches_expected, canonicalize_svd

MAX_ITER = 10000
TOL = 0.1
Q_HIST = 3
PATIENCE = 3


def test_svd_matches_expected(svd_step) -> None:
    _, U, S, V = svd_step
    U, V = canonicalize_svd(U, V)

    assert U.shape == (8451, K)
    assert S.shape == (K,)
    assert V.shape == (105, K)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.svd.U.expected", U, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.svd.S.expected", S, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.svd.V.expected", V, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)


def test_als_matches_expected(als_step) -> None:
    P, Q = als_step

    assert P.shape == (8451, K)
    assert Q.shape == (105, K)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.als.P.expected", P, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.als.Q.expected", Q, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)


def test_run_als_keeps_best_start(demo_bed_data, svd_step) -> None:
    G, N, M = demo_bed_data
    f, U, S, V = svd_step
    starts = [ALS(U, S, V, f, SEED + 1000 * i, M, N, K, MAX_ALS, TOL_ALS, beta=BETA_ALS) for i in range(3)]
    best = max(tools.loglikelihood(G, P, Q) for P, Q in starts)

    _, _, logl = run_als(G, U, S, V, f, SEED, M, N, K, MAX_ALS, TOL_ALS, BETA_ALS,
                         3, Device("cpu"), CHUNK_SIZE, 0)

    assert logl == pytest.approx(best, rel=1e-12)


def test_brqn_optimization_matches_expected(demo_bed_data, als_step) -> None:
    G, N, M = demo_bed_data
    P, Q = als_step

    P_opt, Q_opt = optimize_original(
        G,
        P.copy(),
        Q.copy(),
        MAX_ITER,
        K,
        M,
        N,
        TOL,
        Q_HIST,
        PATIENCE,
    )

    assert P_opt.shape == (8451, K)
    assert Q_opt.shape == (105, K)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.brqn.P.expected", P_opt, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(CPU_BRQN_EXPECTED_DIR, "demo_run.brqn.Q.expected", Q_opt, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)


def test_qn_cannot_discard_improving_sqp_steps(monkeypatch):
    import adamixture.model.br_qn as brqn
    rng = np.random.default_rng(31)
    K, M, N = 3, 101, 24
    true_p = rng.uniform(.1, .9, (M,K))
    true_q = rng.dirichlet(np.ones(K), N)
    G = rng.binomial(2, true_p @ true_q.T).astype(np.uint8)
    P = rng.uniform(.1, .9, (M,K))
    Q = rng.dirichlet(np.ones(K), N)
    initial_ll = tools.loglikelihood(G, P, Q)

    # A finite but unhelpful extrapolation returns the initial point. The
    # available SQP improvement must survive even on the first iteration,
    # when comparison with a previous LL of -inf would accept this candidate.
    def no_acceleration(output, x_next, x, *args):
        output[:] = x

    monkeypatch.setattr(brqn, 'qn_extrapolate_ZAL', no_acceleration)
    p, q = brqn.optimize_original(G, P.copy(), Q.copy(), 1, K, M, N,
                                  0., 3, 2, verbose=False)
    assert tools.loglikelihood(G, p, q) > initial_ll + 1.
