"""Equivalence and regression tests for optimized CPU EM and Adam-EM kernels."""

import numpy as np
import pytest

from adamixture.model import em_adam
from adamixture.src import adamixture, utils
from adamixture.src.utils_c import em, tools


def _numpy_em_reference(
    G_unp: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact scalar/dense FP64 reference matching original Cython EM equations."""
    m = (G_unp != 3).astype(np.float64)
    g = np.where(G_unp != 3, G_unp, 0).astype(np.float64)
    r = P @ Q.T
    a = np.where(m > 0, g / r, 0.0)
    b = np.where(m > 0, (2.0 - g) / (1.0 - r), 0.0)

    A_mat = a @ Q
    B_mat = b @ Q
    denom = P * (A_mat - B_mat) + B_mat
    P_ref = np.divide(A_mat * P, denom, out=np.zeros_like(P), where=denom > 0.0)
    P_ref = np.clip(P_ref, 1e-5, 1.0 - 1e-5)

    T_mat = (a - b).T @ P + b.sum(axis=0)[:, None]
    q_cnt = 2.0 * m.sum(axis=0)
    inv_q = np.divide(1.0, q_cnt, out=np.zeros_like(q_cnt), where=q_cnt > 0.0)
    Q_ref = np.clip(Q * T_mat * inv_q[:, None], 1e-5, 1.0 - 1e-5)
    Q_ref /= Q_ref.sum(axis=1, keepdims=True)
    return P_ref, Q_ref


@pytest.mark.parametrize("M,N,K", [(1, 1, 2), (5, 17, 3), (131, 1100, 7)])
def test_em_tiled_p_q_step_matches_reference_with_missing_rows_and_cols(M: int, N: int, K: int) -> None:
    rng = np.random.default_rng(1000 + M + N + K)
    G_unp = rng.integers(0, 4, size=(M, N), dtype=np.uint8)
    if M >= 5:
        G_unp[2, :] = 3  # all-missing SNP
    if N >= 5:
        G_unp[:, 3] = 3  # all-missing sample

    P = rng.uniform(0.05, 0.95, size=(M, K)).astype(np.float64)
    Q = rng.dirichlet(np.ones(K), size=N).astype(np.float64)
    G = utils.ensure_packed(G_unp, M)

    P_ref, Q_ref = _numpy_em_reference(G_unp, P, Q)

    P_em = np.empty_like(P)
    Q_em = np.empty_like(Q)
    T = np.zeros_like(Q)
    q_bat = np.zeros(N, dtype=np.float64)

    em.P_step(G, P.copy(), P_em, Q, T, q_bat, K, M, N)
    em.Q_step(Q, Q_em, T, q_bat, K, N)

    np.testing.assert_allclose(P_em, P_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Q_em, Q_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(T, 0.0)
    np.testing.assert_array_equal(q_bat, 0.0)


@pytest.mark.parametrize("use_scratch", [True, False])
def test_fused_em_and_adam_em_kernels_match_reference(use_scratch: bool) -> None:
    rng = np.random.default_rng(2025)
    M, N, K = 133, 61, 5
    G_unp = rng.integers(0, 4, size=(M, N), dtype=np.uint8)
    G_unp[1, :] = 3
    G_unp[:, 0] = 3
    G = utils.ensure_packed(G_unp, M)

    P0 = rng.uniform(0.05, 0.95, size=(M, K))
    Q0 = rng.dirichlet(np.ones(K), size=N)
    inv_q_bat = np.empty(N, dtype=np.float64)
    em.precompute_inv_q_bat_packed(G, inv_q_bat, M, N)

    m_cnt = (G_unp != 3).sum(axis=0).astype(np.float64)
    expected_inv = np.divide(1.0, 2.0 * m_cnt, out=np.zeros_like(m_cnt), where=m_cnt > 0)
    np.testing.assert_allclose(inv_q_bat, expected_inv, rtol=1e-15, atol=1e-15)

    scratch = np.empty(em.em_workspace_bytes(M, N, K, False), dtype=np.uint8) if use_scratch else None

    # 1. em_step_inplace vs emStep
    P_a, Q_a = P0.copy(), Q0.copy()
    T_a = np.zeros_like(Q0)
    P1_a, Q1_a = np.empty_like(P0), np.empty_like(Q0)
    q_bat_a = np.zeros(N, dtype=np.float64)
    em_adam.emStep(G, P_a, Q_a, T_a, P1_a, Q1_a, q_bat_a, K, M, N)

    P_b, Q_b = P0.copy(), Q0.copy()
    T_b = np.empty_like(Q0)
    em.em_step_inplace(G, P_b, Q_b, T_b, inv_q_bat, K, M, N, scratch)
    np.testing.assert_allclose(P_b, P_a, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(Q_b, Q_a, rtol=1e-13, atol=1e-13)

    # 2. em_targets_with_ll vs P_step/Q_step + loglikelihood_packed
    P_t, Q_t = np.empty_like(P0), np.empty_like(Q0)
    ll_fused = em.em_targets_with_ll(G, P_b.copy(), P_t, Q_b, Q_t, T_b, inv_q_bat, K, M, N, scratch)
    ll_ref = tools.loglikelihood_packed(G.ctypes.data, M, N, P_b, Q_b)
    P_ref, Q_ref = _numpy_em_reference(G_unp, P_b, Q_b)
    assert ll_fused == pytest.approx(ll_ref, rel=1e-12, abs=1e-10)
    np.testing.assert_allclose(P_t, P_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Q_t, Q_ref, rtol=1e-12, atol=1e-12)

    # 3. adam_em_step_inplace vs adamStep
    mP_a, vP_a = np.zeros_like(P0), np.zeros_like(P0)
    mQ_a, vQ_a = np.zeros_like(Q0), np.zeros_like(Q0)
    P_aa, Q_aa = P0.copy(), Q0.copy()
    t_list = [0]
    for _ in range(3):
        em_adam.adamStep(G, P_aa, Q_aa, T_a, P1_a, Q1_a, q_bat_a, K, M, N, mP_a, vP_a, mQ_a, vQ_a, t_list, 0.005, 0.8, 0.88, 1e-8)

    mP_b, vP_b = np.zeros_like(P0), np.zeros_like(P0)
    mQ_b, vQ_b = np.zeros_like(Q0), np.zeros_like(Q0)
    P_ab, Q_ab = P0.copy(), Q0.copy()
    for step_idx in range(1, 4):
        em.adam_em_step_inplace(
            G, P_ab, Q_ab, mP_b, vP_b, mQ_b, vQ_b, T_b, inv_q_bat,
            0.005, 0.8, 0.88, 1e-8, step_idx, K, M, N, scratch,
        )

    np.testing.assert_allclose(P_ab, P_aa, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Q_ab, Q_aa, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("check", [1, 5])
def test_optimize_parameters_lookahead_matches_step_by_step(check: int) -> None:
    rng = np.random.default_rng(707 + check)
    M, N, K = 77, 45, 4
    G_unp = rng.integers(0, 4, size=(M, N), dtype=np.uint8)
    G = utils.ensure_packed(G_unp, M)
    P0 = rng.uniform(0.1, 0.9, size=(M, K))
    Q0 = rng.dirichlet(np.ones(K), size=N)

    kwargs = dict(
        lr=0.005, beta1=0.80, beta2=0.88, reg_adam=1e-8,
        max_iter=12, check=check, K=K, M=M, N=N,
        lr_decay=0.5, min_lr=1e-4, patience=3, tol_adam=0.1,
    )
    P_fast, Q_fast = em_adam.optimize_parameters(G, P0.copy(), Q0.copy(), **kwargs)

    # Naive reference loop using standalone emStep / adamStep + loglikelihood_packed
    P_ref, Q_ref = P0.copy(), Q0.copy()
    m_P, v_P = np.zeros_like(P_ref), np.zeros_like(P_ref)
    m_Q, v_Q = np.zeros_like(Q_ref), np.zeros_like(Q_ref)
    P1, Q1 = np.empty_like(P_ref), np.empty_like(Q_ref)
    T = np.zeros_like(Q_ref)
    q_bat = np.zeros(N, dtype=np.float64)
    t = [0]

    em_adam.emStep(G, P_ref, Q_ref, T, P1, Q1, q_bat, K, M, N)
    em_adam.adamStep(G, P_ref, Q_ref, T, P1, Q1, q_bat, K, M, N, m_P, v_P, m_Q, v_Q, t, 0.005, 0.8, 0.88, 1e-8)
    em_adam.emStep(G, P_ref, Q_ref, T, P1, Q1, q_bat, K, M, N)

    L_best = tools.loglikelihood_packed(G.ctypes.data, M, N, P_ref, Q_ref)
    P_best, Q_best = P_ref.copy(), Q_ref.copy()
    wait_lr = 0
    lr = 0.005
    for it in range(12):
        em_adam.adamStep(G, P_ref, Q_ref, T, P1, Q1, q_bat, K, M, N, m_P, v_P, m_Q, v_Q, t, lr, 0.8, 0.88, 1e-8)
        if (it + 1) % check == 0:
            L_cur = tools.loglikelihood_packed(G.ctypes.data, M, N, P_ref, Q_ref)
            if L_cur > L_best + 0.1:
                L_best = L_cur
                P_best[:] = P_ref
                Q_best[:] = Q_ref
                wait_lr = 0
            else:
                wait_lr += 1
                if wait_lr >= 3:
                    lr = max(lr * 0.5, 1e-4)
                    if lr <= 1e-4:
                        break
                    wait_lr = 0

    np.testing.assert_allclose(P_fast, P_best, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(Q_fast, Q_best, rtol=1e-11, atol=1e-11)

    P_init, Q_init = adamixture.initialize_em_cpu(G, 42, M, N, K, n_steps=4, verbose=False)
    assert P_init.shape == (M, K) and Q_init.shape == (N, K)
    assert np.isfinite(P_init).all() and np.isfinite(Q_init).all()
    np.testing.assert_allclose(Q_init.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)

