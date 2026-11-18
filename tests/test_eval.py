"""Unit and CLI integration tests for adamixture.eval.evaluation and adamixture.eval.stability."""

from argparse import Namespace
import subprocess
import sys

import numpy as np
import pytest

from adamixture.eval import evaluation, stability
from adamixture.src import utils
from adamixture.src.utils_c import tools
from tests.config import DATA_DIR


def _make_greedy_trap_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Return valid simplex (true, est[:, perm]) where greedy column matching is strictly suboptimal."""
    true_q = np.array(
        [
            [0.15, 0.24, 0.61],
            [0.15, 0.24, 0.61],
            [0.00, 0.00, 1.00],
            [0.00, 0.00, 1.00],
        ],
        dtype=np.float64,
    )
    est_base = np.array(
        [
            [0.21, 0.31, 0.48],
            [0.21, 0.31, 0.48],
            [0.00, 0.00, 1.00],
            [0.00, 0.00, 1.00],
        ],
        dtype=np.float64,
    )
    # Permute columns so identity wouldn't accidentally match if LAP was skipped.
    est_q = np.ascontiguousarray(est_base[:, [2, 0, 1]])
    return true_q, est_q


def test_optimal_alignment_beats_greedy_counterexample() -> None:
    true_q, est_q = _make_greedy_trap_matrices()

    aligned_est, aligned_true = evaluation.align_latent_factors(est_q, true_q)
    frob_sq_eval = float(np.sum((aligned_est - aligned_true) ** 2))
    assert frob_sq_eval == pytest.approx(0.0508, abs=1e-12)

    aligned_stab = stability.align_matrices(true_q, est_q)
    frob_sq_stab = float(np.sum((aligned_stab - true_q) ** 2))
    assert frob_sq_stab == pytest.approx(0.0508, abs=1e-12)


def test_kl_handles_exact_zeros_without_nan() -> None:
    q = np.ascontiguousarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.25, 0.75, 0.0],
        ],
        dtype=np.float64,
    )
    val = tools.KL(q, q, q.shape[0], q.shape[1])
    assert np.isfinite(val)
    assert val == pytest.approx(1e-10, abs=1e-12)


def test_evaluation_validation_exact_self_and_bound_zero(tmp_path) -> None:
    q = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.4, 0.6],
            [0.3, 0.7, 0.0],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    qfile = tmp_path / "est.Q"
    tfile = tmp_path / "true.Q"
    np.savetxt(qfile, q[:, [2, 0, 1]], fmt="%.8f")
    np.savetxt(tfile, q, fmt="%.8f")

    args = Namespace(tfile=str(tfile), rmse=True, jsd=True, no_auto_transpose=False)
    est = evaluation.load_proportions(str(qfile), bound=0.0)
    res = evaluation.run_validation(args, est)

    assert res["rmse"] == pytest.approx(0.0, abs=1e-7)
    assert res["jsd"] == pytest.approx(1e-10, abs=1e-12)
    assert np.isfinite(res["jsd"])


def test_evaluation_loglike_and_combined_metrics_on_demo_bed(tmp_path) -> None:
    bed_path = DATA_DIR / "demo_data.bed"
    G_unpacked, N, M = utils.read_data(bed_path, packed=False, verbose=False)
    K = 3
    rng = np.random.default_rng(123)
    P = rng.uniform(0.05, 0.95, size=(M, K))
    Q = rng.dirichlet(np.full(K, 0.7), size=N)
    Q_true = Q[:, [1, 2, 0]].copy()

    qfile = tmp_path / "demo.Q"
    tfile = tmp_path / "demo_true.Q"
    pfile = tmp_path / "demo.P"
    pfile_inv = tmp_path / "demo_inv.P"
    np.savetxt(qfile, Q, fmt="%.10f")
    np.savetxt(tfile, Q_true, fmt="%.10f")
    np.savetxt(pfile, P, fmt="%.10f")
    np.savetxt(pfile_inv, 1.0 - P, fmt="%.10f")

    q_loaded = evaluation.load_proportions(str(qfile), bound=1e-5)
    p_clipped = np.clip(P, 1e-5, 1.0 - 1e-5)
    expected_ll = float(tools.loglikelihood(G_unpacked, p_clipped, q_loaded))

    args_inv = Namespace(
        data_path=str(bed_path),
        pfile=str(pfile_inv),
        bound=1e-5,
        inverse=True,
        chrom_mode="all",
        autosomes=22,
        specific_chrom=None,
    )
    ll_inv = evaluation.run_fitting_eval(args_inv, q_loaded)
    assert ll_inv == pytest.approx(expected_ll, rel=1e-11, abs=1e-7)

    # Verify main() runs both validation metrics and --loglike when combined in one CLI call.
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "adamixture.eval.evaluation",
            "--qfile",
            str(qfile),
            "--tfile",
            str(tfile),
            "--data_path",
            str(bed_path),
            "--pfile",
            str(pfile),
            "--rmse",
            "--jsd",
            "--loglike",
            "--threads",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    combined = res.stdout + res.stderr
    assert res.returncode == 0, combined
    assert "RMSE: 0.0000000" in combined
    assert "JSD: 0.0000000" in combined
    assert f"Log-likelihood: {round(expected_ll, 1)}" in combined


def test_evaluation_cli_errors_and_k1_and_wide_q(tmp_path) -> None:
    qfile = tmp_path / "q1.txt"
    tfile = tmp_path / "t1.txt"
    q_k1 = np.ones((5,), dtype=np.float64)
    np.savetxt(qfile, q_k1)
    np.savetxt(tfile, q_k1)

    # No metrics specified -> argparse error (exit 2).
    no_metric = subprocess.run(
        [sys.executable, "-m", "adamixture.eval.evaluation", "--qfile", str(qfile)],
        capture_output=True,
        text=True,
    )
    assert no_metric.returncode != 0

    # K=1 1D file loads cleanly as (N, 1).
    loaded_k1 = evaluation.load_proportions(str(qfile), bound=0.0)
    assert loaded_k1.shape == (5, 1)
    val = evaluation.run_validation(
        Namespace(tfile=str(tfile), rmse=True, jsd=True, no_auto_transpose=False),
        loaded_k1,
    )
    assert val["rmse"] == pytest.approx(0.0, abs=1e-12)
    assert val["jsd"] == pytest.approx(1e-10, abs=1e-12)

    # N < K with --no_auto_transpose preserves (3, 5) orientation.
    q_wide = np.array(
        [
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.1, 0.2, 0.3, 0.4, 0.0],
            [0.0, 0.0, 0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    wide_file = tmp_path / "wide.npy"
    np.save(wide_file, q_wide)
    loaded_wide = evaluation.load_proportions(str(wide_file), bound=0.0, transpose_if_wide=False)
    assert loaded_wide.shape == (3, 5)

    # Shape mismatch triggers clean exit code 1 via main().
    bad = subprocess.run(
        [
            sys.executable,
            "-m",
            "adamixture.eval.evaluation",
            "--qfile",
            str(wide_file),
            "--tfile",
            str(tfile),
            "--rmse",
            "--no_auto_transpose",
        ],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1


def test_stability_permutations_collapsed_column_mixed_formats_and_k1(tmp_path) -> None:
    rng = np.random.default_rng(7)
    q1 = rng.dirichlet([0.5, 0.5, 0.5], size=20)
    # Force a collapsed constant cluster column + renormalize remaining columns cleanly.
    q_collapsed = np.zeros((20, 4), dtype=np.float64)
    q_collapsed[:, :3] = q1 * 0.999
    q_collapsed[:, 3] = 0.001
    q_collapsed_perm = q_collapsed[:, [2, 3, 0, 1]]

    f_txt = tmp_path / "run1.Q"
    f_npy = tmp_path / "run2.npy"
    np.savetxt(f_txt, q_collapsed, fmt="%.10f")
    np.save(f_npy, q_collapsed_perm)

    m1 = stability.load_proportions(str(f_txt))
    m2 = stability.load_proportions(str(f_npy))
    m2_aligned = stability.align_matrices(m1, m2)

    corr = stability.calculate_correlations(m1, m2_aligned)
    frob = stability.calculate_frobenius(m1, m2_aligned)
    gprime = stability.calculate_gprime(m1, m2_aligned)

    assert np.isfinite(corr)
    assert corr == pytest.approx(1.0, abs=1e-7)
    assert frob == pytest.approx(0.0, abs=1e-7)
    assert gprime == pytest.approx(1.0, abs=1e-7)

    # Test collapsed constant column paired against non-constant column -> 0.0, not NaN.
    q_alt = m1.copy()
    q_alt[:, 3] = np.linspace(0.0, 0.002, 20)
    q_alt /= q_alt.sum(axis=1, keepdims=True)
    corr_mixed = stability.calculate_correlations(m1, q_alt)
    assert np.isfinite(corr_mixed)

    # Test K=1 CLI execution end-to-end.
    f_k1_a = tmp_path / "k1_a.Q"
    f_k1_b = tmp_path / "k1_b.Q"
    np.savetxt(f_k1_a, np.ones(6))
    np.savetxt(f_k1_b, np.ones(6))
    cli = subprocess.run(
        [sys.executable, "-m", "adamixture.eval.stability", str(f_k1_a), str(f_k1_b)],
        capture_output=True,
        text=True,
    )
    out = cli.stdout + cli.stderr
    assert cli.returncode == 0, out
    assert "Average Pairwise Correlation: 1.000000" in out
    assert "Average Pairwise G':            1.000000" in out


def test_eval_modules_do_not_require_torch() -> None:
    code = """
import sys
sys.modules["torch"] = None
from adamixture.eval import evaluation, stability
assert "torch" not in sys.modules or sys.modules["torch"] is None
print("OK_EVAL_NO_TORCH")
"""
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "OK_EVAL_NO_TORCH" in res.stdout

