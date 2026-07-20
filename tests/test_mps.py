import os

import pytest

torch = pytest.importorskip("torch")

from adamixture.model.als_gpu import ALS_gpu
from adamixture.model.br_qn_gpu import optimize_original_gpu
from adamixture.model.em_adam_gpu import optimize_parameters_gpu
from adamixture.src import utils
from adamixture.src.adamixture import setup

from tests.config import (
    CHUNK_SIZE,
    DATA_DIR,
    EXPECTED_ATOL,
    EXPECTED_LOGL_ATOL,
    EXPECTED_RTOL,
    K,
    MAX_ALS,
    MPS_ADAMEM_EXPECTED_DIR,
    MPS_BRQN_EXPECTED_DIR,
    POWER,
    SEED,
    TOL_ALS,
    TOL_SVD,
)
from tests.helpers import assert_matches_expected, assert_model_close_to_expected, canonicalize_svd, to_numpy
from tests.test_adamem_cpu import BETA1, BETA2, CHECK, LR, LR_DECAY, MIN_LR, REG_ADAM, TOL_ADAM
from tests.test_brqn_cpu import MAX_ITER, PATIENCE, Q_HIST, TOL


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("ADAMIXTURE_TEST_MPS") != "1",
        reason="Set ADAMIXTURE_TEST_MPS=1 to run MPS tests.",
    ),
    pytest.mark.skipif(
        not torch.backends.mps.is_available(),
        reason="MPS is not available.",
    ),
]


@pytest.fixture(scope="session")
def mps_bed_data():
    return utils.read_data(
        str(DATA_DIR / "demo_data.bed"),
        packed=True,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        verbose=False,
    )


@pytest.fixture(scope="session")
def mps_unpacked_data():
    return utils.read_data(
        str(DATA_DIR / "demo_data.bed"),
        packed=False,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        verbose=False,
    )


def _mps_steps(mps_bed_data, *, algorithm: str):
    G_packed, N, M = mps_bed_data
    is_brqn = algorithm == "brqn"
    device_obj, threads_per_block, f, U, S, V, G_device = setup(
        G_packed,
        N,
        M,
        K,
        SEED,
        POWER,
        TOL_SVD,
        CHUNK_SIZE,
        "mps",
        original=is_brqn,
        init_original="als",
        q_hist=Q_HIST,
    )
    P_als, Q_als = ALS_gpu(
        U.contiguous(),
        S.contiguous(),
        V.contiguous(),
        f,
        SEED,
        M,
        K,
        MAX_ALS,
        TOL_ALS,
        device_obj,
    )
    if is_brqn:
        P_opt, Q_opt = optimize_original_gpu(
            G_device,
            P_als.clone(),
            Q_als.clone(),
            MAX_ITER,
            K,
            M,
            N,
            TOL,
            Q_HIST,
            PATIENCE,
            device_obj,
            CHUNK_SIZE,
            threads_per_block,
        )
    else:
        P_opt, Q_opt = optimize_parameters_gpu(
            G_device,
            P_als.clone(),
            Q_als.clone(),
            LR,
            BETA1,
            BETA2,
            REG_ADAM,
            MAX_ITER,
            CHECK,
            K,
            M,
            N,
            LR_DECAY,
            MIN_LR,
            PATIENCE,
            TOL_ADAM,
            device_obj,
            CHUNK_SIZE,
            threads_per_block,
        )
    return U, S, V, P_als, Q_als, P_opt, Q_opt


@pytest.fixture(scope="session")
def mps_brqn_steps(mps_bed_data):
    return _mps_steps(mps_bed_data, algorithm="brqn")


@pytest.fixture(scope="session")
def mps_adamem_steps(mps_bed_data):
    return _mps_steps(mps_bed_data, algorithm="adamem")


def _assert_mps_steps(expected_dir, steps, algorithm: str, mps_unpacked_data) -> None:
    G, _, _ = mps_unpacked_data
    U, S, V, P_als, Q_als, P_opt, Q_opt = steps
    U, V = canonicalize_svd(to_numpy(U), to_numpy(V))

    assert_matches_expected(expected_dir, "demo_run.svd.U.expected", U, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(expected_dir, "demo_run.svd.S.expected", to_numpy(S), rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(expected_dir, "demo_run.svd.V.expected", V, rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(expected_dir, "demo_run.als.P.expected", to_numpy(P_als), rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_matches_expected(expected_dir, "demo_run.als.Q.expected", to_numpy(Q_als), rtol=EXPECTED_RTOL, atol=EXPECTED_ATOL)
    assert_model_close_to_expected(
        G,
        to_numpy(P_opt),
        to_numpy(Q_opt),
        expected_dir,
        f"demo_run.{algorithm}.P.expected",
        f"demo_run.{algorithm}.Q.expected",
        matrix_rtol=EXPECTED_RTOL,
        matrix_atol=EXPECTED_ATOL,
        logl_atol=EXPECTED_LOGL_ATOL,
    )


def test_mps_brqn_steps_match_expected(mps_brqn_steps, mps_unpacked_data) -> None:
    _assert_mps_steps(MPS_BRQN_EXPECTED_DIR, mps_brqn_steps, "brqn", mps_unpacked_data)


def test_mps_adamem_steps_match_expected(mps_adamem_steps, mps_unpacked_data) -> None:
    _assert_mps_steps(MPS_ADAMEM_EXPECTED_DIR, mps_adamem_steps, "adamem", mps_unpacked_data)
