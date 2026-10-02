import numpy as np
import pytest

from adamixture.project_entry import parse_args
from adamixture.src.projection import optimize_projection_cpu


PROJ_MIN_ARGS = [
    "--save_dir", "out",
    "--data_path", "data.bed",
    "--p_path", "model.P",
    "--name", "test_proj",
]


def test_projection_parse_args() -> None:
    args = parse_args(PROJ_MIN_ARGS)
    assert args.data_path == "data.bed"
    assert args.p_path == "model.P"
    assert args.max_iter == 15
    assert args.tol == 1e-4
    assert args.device == "cpu"


def test_projection_parse_args_rejects_obsolete_flags() -> None:
    with pytest.raises(SystemExit):
        parse_args([*PROJ_MIN_ARGS, "--algorithm", "adamem"])


def test_optimize_projection_cpu_synthetic() -> None:
    rng = np.random.default_rng(42)
    N, M, K = 8, 50, 4

    # Genotype in {0, 1, 2}
    G = rng.integers(0, 3, size=(M, N), dtype=np.uint8)
    # Fixed P in [0.05, 0.95]
    P = rng.uniform(0.05, 0.95, size=(M, K)).astype(np.float64)
    # Initial Q
    Q = rng.random(size=(N, K)).astype(np.float64)
    Q /= Q.sum(axis=1, keepdims=True)

    Q_opt = optimize_projection_cpu(G, P, Q, max_iter=5, K=K, M=M, N=N, tol=1e-4)

    assert Q_opt.shape == (N, K)
    assert np.all(Q_opt >= 0.0)
    assert np.all(Q_opt <= 1.0)
    np.testing.assert_allclose(Q_opt.sum(axis=1), np.ones(N), atol=1e-5)
