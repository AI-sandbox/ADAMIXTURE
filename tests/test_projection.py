import numpy as np

from adamixture.src.projection import optimize_projection_cpu


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

    Q_opt = optimize_projection_cpu(G, P, chunk_size=16, Q=Q, K=K, M=M, N=N)

    assert Q_opt.shape == (N, K)
    assert np.all(Q_opt >= 0.0)
    assert np.all(Q_opt <= 1.0)
    np.testing.assert_allclose(Q_opt.sum(axis=1), np.ones(N), atol=1e-5)
