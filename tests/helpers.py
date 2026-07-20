import os
from pathlib import Path

import numpy as np

from adamixture.src.plot import align_clusters_clumppling
from adamixture.src.utils_c import tools


UPDATE_EXPECTED = os.environ.get("ADAMIXTURE_UPDATE_EXPECTED") == "1"


def save_expected(path: Path, actual: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if np.issubdtype(actual.dtype, np.integer):
        np.savetxt(path, actual, fmt="%d")
    else:
        np.savetxt(path, actual, fmt="%.18e")


def assert_matches_expected(
    expected_dir: Path,
    name: str,
    actual: np.ndarray,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> None:
    path = expected_dir / name
    if UPDATE_EXPECTED:
        save_expected(path, actual)
        return

    assert path.exists(), (
        f"Missing expected fixture: {path}. "
        "Run `ADAMIXTURE_UPDATE_EXPECTED=1 python -m pytest tests` to regenerate fixtures."
    )

    expected = np.loadtxt(path)
    if np.issubdtype(actual.dtype, np.integer):
        expected = expected.astype(actual.dtype, copy=False)
        assert np.array_equal(actual, expected)
    else:
        assert actual.shape == expected.shape
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def canonicalize_svd(U: np.ndarray, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    U_out = np.array(U, copy=True)
    V_out = np.array(V, copy=True)
    for col in range(U_out.shape[1]):
        pivot = int(np.argmax(np.abs(U_out[:, col])))
        if U_out[pivot, col] < 0:
            U_out[:, col] *= -1
            V_out[:, col] *= -1
    return U_out, V_out


def load_expected_model(expected_dir: Path, p_name: str, q_name: str) -> tuple[np.ndarray, np.ndarray]:
    return np.loadtxt(expected_dir / p_name), np.loadtxt(expected_dir / q_name)


def to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def align_model_to_expected(
    expected_q: np.ndarray,
    actual_p: np.ndarray,
    actual_q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    perm = align_clusters_clumppling(expected_q, actual_q)
    return actual_p[:, perm], actual_q[:, perm]


def assert_feasible_model(P: np.ndarray, Q: np.ndarray, *, atol: float = 1e-4) -> None:
    assert np.isfinite(P).all()
    assert np.isfinite(Q).all()
    assert P.min() >= -atol
    assert P.max() <= 1.0 + atol
    assert Q.min() >= -atol
    assert Q.max() <= 1.0 + atol
    np.testing.assert_allclose(Q.sum(axis=1), 1.0, atol=atol, rtol=0.0)


def assert_model_close_to_expected(
    G: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    expected_dir: Path,
    p_name: str,
    q_name: str,
    *,
    matrix_rtol: float,
    matrix_atol: float,
    logl_atol: float,
) -> None:
    expected_p, expected_q = load_expected_model(expected_dir, p_name, q_name)
    P, Q = align_model_to_expected(expected_q, P, Q)

    assert_feasible_model(P, Q)
    np.testing.assert_allclose(P, expected_p, rtol=matrix_rtol, atol=matrix_atol)
    np.testing.assert_allclose(Q, expected_q, rtol=matrix_rtol, atol=matrix_atol)

    actual_logl = tools.loglikelihood(
        G,
        np.ascontiguousarray(P, dtype=np.float64),
        np.ascontiguousarray(Q, dtype=np.float64),
    )
    expected_logl = tools.loglikelihood(
        G,
        np.ascontiguousarray(expected_p, dtype=np.float64),
        np.ascontiguousarray(expected_q, dtype=np.float64),
    )
    assert abs(actual_logl - expected_logl) <= logl_atol
