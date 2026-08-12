import numpy as np

import adamixture.model.br_qn as br_qn
import adamixture.src.cv as cv


def test_sqp_polish_reuses_main_optimizer_without_early_stopping(monkeypatch) -> None:
    G = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    P_init = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float64)
    Q_init = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float64)
    captured = {}

    def fake_optimize(G_arg, P, Q, max_iter, K, M, N, tol, Q_hist, patience, verbose):
        captured.update(
            G=G_arg,
            P=P,
            Q=Q,
            max_iter=max_iter,
            K=K,
            M=M,
            N=N,
            tol=tol,
            Q_hist=Q_hist,
            patience=patience,
            verbose=verbose,
        )
        return P, Q

    monkeypatch.setattr(br_qn, "optimize_original", fake_optimize)

    P_out, Q_out = br_qn.polish_sqp_qn(G, P_init, Q_init, M=2, N=2, K=2)

    assert captured["G"] is G
    assert captured["max_iter"] == 3
    assert captured["Q_hist"] == 3
    assert captured["tol"] == 0.0
    assert captured["patience"] == 4
    assert captured["verbose"] is False
    assert P_out is not P_init
    assert Q_out is not Q_init
    np.testing.assert_array_equal(P_out, P_init)
    np.testing.assert_array_equal(Q_out, Q_init)


def test_cv_fold_runs_three_sqp_qn_iterations(monkeypatch) -> None:
    G = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    P_init = np.array([[0.2, 0.8], [0.6, 0.4]], dtype=np.float64)
    Q_init = np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float64)
    captured = {}

    def fake_polish(G_arg, P_arg, Q_arg, M, N, K, n_iters, Q_hist):
        captured.update(
            G=G_arg,
            P=P_arg,
            Q=Q_arg,
            M=M,
            N=N,
            K=K,
            n_iters=n_iters,
            Q_hist=Q_hist,
        )
        return P_arg, Q_arg

    monkeypatch.setattr(cv, "polish_sqp_qn", fake_polish)

    P_out, Q_out = cv._polish_fold(G, P_init, Q_init, M=2, N=2, K=2)

    assert captured["G"] is G
    assert captured["P"] is P_init
    assert captured["Q"] is Q_init
    assert captured["M"] == 2
    assert captured["N"] == 2
    assert captured["K"] == 2
    assert captured["n_iters"] == 3
    assert captured["Q_hist"] == 3
    assert P_out is P_init
    assert Q_out is Q_init
