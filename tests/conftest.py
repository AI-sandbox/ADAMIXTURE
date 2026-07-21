import os

for env_var in (
    "MKL_NUM_THREADS",
    "MKL_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OMP_MAX_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_MAX_THREADS",
):
    os.environ.setdefault(env_var, "1")

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

from adamixture.model.als import ALS
from adamixture.model.svd import RSVD
from adamixture.src import utils
from tests.config import CHUNK_SIZE, DATA_DIR, K, MAX_ALS, POWER, SEED, TOL_ALS, TOL_SVD


@pytest.fixture(scope="session")
def demo_bed_data() -> tuple[np.ndarray, int, int]:
    return utils.read_data(
        str(DATA_DIR / "demo_data.bed"),
        packed=False,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        verbose=False,
    )


@pytest.fixture(scope="session")
def svd_step(demo_bed_data: tuple[np.ndarray, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    G, N, M = demo_bed_data
    f = utils.calculate_frequencies_cpu(G, M, N, CHUNK_SIZE)
    U, S, V = RSVD(G, N, M, f, K, SEED, POWER, TOL_SVD, CHUNK_SIZE)
    return f, U, S, V


@pytest.fixture(scope="session")
def als_step(
    demo_bed_data: tuple[np.ndarray, int, int],
    svd_step: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    _, N, M = demo_bed_data
    f, U, S, V = svd_step
    return ALS(U, S, V, f, SEED, M, N, K, MAX_ALS, TOL_ALS)
