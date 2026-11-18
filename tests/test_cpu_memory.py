"""Tests for CPU memory budgeting, streaming BED reading, zero-alloc packed kernels, and OOM fallbacks."""

from argparse import Namespace

import numpy as np
import pytest

from adamixture.model import br_qn, svd
from adamixture.src import cv, utils
from adamixture.src.utils_c import alleleFrequency_packed, count_non_missing_packed
from adamixture.src.utils_c.cython import sqp, tools
from tests.config import DATA_DIR


def test_available_cpu_memory_and_chunk_capping(monkeypatch) -> None:
    native_avail = utils.get_available_cpu_memory_bytes()
    assert native_avail > 0

    monkeypatch.setenv("ADAMIXTURE_MAX_CPU_MEM_MB", "4")
    assert utils.get_available_cpu_memory_bytes() == 4 * 1024 * 1024

    capped = utils.cap_cpu_chunk_size(8192, bytes_per_row=4096, reserve_bytes=0, fraction=0.5)
    assert capped == 512

    floor_capped = utils.cap_cpu_chunk_size(8192, bytes_per_row=4096, reserve_bytes=4 * 1024 * 1024)
    assert floor_capped == utils.MIN_CPU_CHUNK_SIZE


def test_rsvd_caps_chunk_size_under_tight_ram(monkeypatch) -> None:
    rng = np.random.default_rng(11)
    M, N, K = 64, 512, 3
    G = utils.ensure_packed(rng.integers(0, 4, size=(M, N), dtype=np.uint8), M)
    f = utils.calculate_frequencies_cpu(G, M, N, chunk_size=32)

    # 0.05 MB -> 26 KB budget; (512 + 20) * 4 = 2128 B/row -> chunk=64 (~136 KB) caps to < 64
    monkeypatch.setenv("ADAMIXTURE_MAX_CPU_MEM_MB", "0.05")
    seen_chunks = []
    orig_cap = utils.cap_cpu_chunk_size

    def spy_cap(chunk_size, bytes_per_row, reserve_bytes=0, fraction=0.5):
        res = orig_cap(chunk_size, bytes_per_row, reserve_bytes=reserve_bytes, fraction=fraction)
        seen_chunks.append(res)
        return res

    monkeypatch.setattr(utils, "cap_cpu_chunk_size", spy_cap)
    U, S, V = svd.RSVD(G, N, M, f, K, seed=42, power=1, tol=0.1, chunk=64)
    assert seen_chunks and seen_chunks[0] < 64
    assert U.shape == (M, K) and S.shape == (K,) and V.shape == (N, K)


@pytest.mark.parametrize("suffix", ["bed", "bed.gz", "bed.zst"])
def test_streaming_bed_reader_matches_unpacked_and_filters(suffix, monkeypatch) -> None:
    path = DATA_DIR / f"demo_data.{suffix}"
    # Force multi-chunk streaming even on demo_data (~220 KB) by tightening budget temporarily.
    unpacked_all, N_a, M_a = utils.read_data(path, packed=False, chrom_mode="all", verbose=False)

    monkeypatch.setenv("ADAMIXTURE_MAX_CPU_MEM_MB", "1")
    packed_all, N_b, M_b = utils.read_data(path, packed="cpu", chunk_size=128, chrom_mode="all", verbose=False)
    assert (N_a, M_a) == (N_b, M_b)
    np.testing.assert_array_equal(unpacked_all, tools.unpack_block(packed_all, 0, M_b))

    unpacked_c1, N_c, M_c = utils.read_data(
        path, packed=False, chrom_mode="autosomes", specific_chrom=["1"], verbose=False
    )
    packed_c1, N_d, M_d = utils.read_data(
        path, packed="cpu", chunk_size=128, chrom_mode="autosomes", specific_chrom=["1"], verbose=False
    )
    assert (N_c, M_c) == (N_d, M_d) and 0 < M_c < M_a
    np.testing.assert_array_equal(unpacked_c1, tools.unpack_block(packed_c1, 0, M_d))


@pytest.mark.parametrize("M,N", [(1, 17), (5, 31), (37, 43)])
def test_packed_allele_frequency_and_non_missing_count_match_unpacked(M: int, N: int) -> None:
    rng = np.random.default_rng(100 + M)
    G = rng.integers(0, 4, size=(M, N), dtype=np.uint8)
    if M >= 5:
        G[2, :] = 3  # all missing SNP
    packed = utils.ensure_packed(G, M)

    f_ref = np.zeros(M, dtype=np.float32)
    tools.alleleFrequency(G, f_ref, M, N)

    f_packed = np.zeros(M, dtype=np.float32)
    alleleFrequency_packed(packed, f_packed, M, N)
    np.testing.assert_array_equal(f_packed, f_ref)

    f_via_utils = utils.calculate_frequencies_cpu(packed, M, N, chunk_size=8)
    np.testing.assert_array_equal(f_via_utils, f_ref)

    cnt_ref = int(np.count_nonzero(G != 3))
    cnt_packed = int(count_non_missing_packed(packed, M, N))
    assert cnt_packed == cnt_ref


@pytest.mark.parametrize("unit_simplex", [True, False])
def test_update_q_sqp_without_external_hessian_buffers_is_identical(unit_simplex: bool) -> None:
    rng = np.random.default_rng(77 if unit_simplex else 88)
    M, N, K = 53, 41, 7
    G = utils.ensure_packed(rng.integers(0, 4, size=(M, N), dtype=np.uint8), M)
    P = rng.uniform(0.05, 0.95, size=(M, K))
    if unit_simplex:
        Q = rng.dirichlet(np.ones(K), size=N)
    else:
        Q = rng.uniform(0.1, 0.9, size=(N, K))
    _, _, vt = np.linalg.svd(np.ones((1, K)), full_matrices=True)
    v_kk = np.ascontiguousarray(vt.T, dtype=np.float64)

    XtX_q = np.empty((N, K, K), dtype=np.float64)
    Xtz_q = np.empty((N, K), dtype=np.float64)
    Q_with = np.empty_like(Q)
    Q_without = np.empty_like(Q)

    sqp.update_q_sqp(G, Q, Q_with, P, XtX_q, Xtz_q, v_kk, M, N, K, 16)
    sqp.update_q_sqp(G, Q, Q_without, P, None, None, v_kk, M, N, K, 16)
    np.testing.assert_array_equal(Q_without, Q_with)


def test_cv_streaming_fallback_under_low_memory(monkeypatch) -> None:
    rng = np.random.default_rng(19)
    M, N, K = 61, 29, 3
    G = utils.ensure_packed(rng.integers(0, 4, size=(M, N), dtype=np.uint8), M)
    P = rng.uniform(0.1, 0.9, size=(M, K))
    Q = rng.dirichlet(np.ones(K), size=N)
    args = Namespace(cv=2, seed=42)

    val_normal = cv.run_cross_validation(args, G, N, M, K, P, Q)
    assert np.isfinite(val_normal)

    # Trigger proactive streaming fallback only inside CV
    monkeypatch.setattr(cv, "get_available_cpu_memory_bytes", lambda: 1024)
    stream_calls = []
    orig_hash = cv._build_hashed_fold_entries

    def spy_hash(*a, **kw):
        stream_calls.append(True)
        return orig_hash(*a, **kw)

    monkeypatch.setattr(cv, "_build_hashed_fold_entries", spy_hash)
    val_stream = cv.run_cross_validation(args, G, N, M, K, P, Q)
    assert len(stream_calls) == 2 and np.isfinite(val_stream)

    # Also test reactive MemoryError fallback inside cv index buffer allocation.
    monkeypatch.undo()
    stream_calls.clear()
    monkeypatch.setattr(cv, "_build_hashed_fold_entries", spy_hash)
    real_empty = np.empty
    state = {"triggered": False}

    def fail_index_alloc(shape, dtype=float, *a, **kw):
        if not state["triggered"] and dtype == np.int32 and isinstance(shape, int):
            state["triggered"] = True
            raise MemoryError("simulated OOM on CV index buffer")
        return real_empty(shape, dtype=dtype, *a, **kw)

    monkeypatch.setattr(cv.np, "empty", fail_index_alloc)
    val_reactive = cv.run_cross_validation(args, G, N, M, K, P, Q)
    assert state["triggered"] is True and len(stream_calls) == 2
    assert val_reactive == pytest.approx(val_stream, rel=1e-12, abs=1e-12)


def test_brqn_preserves_q_hist_and_reports_clean_oom(monkeypatch) -> None:
    rng = np.random.default_rng(55)
    M, N, K = 64, 32, 3
    P = rng.uniform(0.1, 0.9, size=(M, K))
    Q = rng.dirichlet(np.ones(K), size=N)
    G = utils.ensure_packed(rng.binomial(2, P @ Q.T).astype(np.uint8), M)

    seen_q_hist = []
    real_update_uv = br_qn.update_UV_ZAL

    def spy_update_uv(U, V, x, xn, xn2, it, q_hist, dim):
        seen_q_hist.append(q_hist)
        return real_update_uv(U, V, x, xn, xn2, it, q_hist, dim)

    monkeypatch.setattr(br_qn, "update_UV_ZAL", spy_update_uv)

    # Tight CPU RAM budget must NOT downgrade user-requested Q_hist=5
    monkeypatch.setenv("ADAMIXTURE_MAX_CPU_MEM_MB", str(12 / 1024))
    P_out, Q_out = br_qn.optimize_original(G, P.copy(), Q.copy(), 2, K, M, N, 0.0, Q_hist=5, patience=3, verbose=False)
    assert seen_q_hist == [5, 5]
    assert np.isfinite(P_out).all() and np.isfinite(Q_out).all()

    # MemoryError during U/V allocation must propagate cleanly without retrying at lower Q_hist
    monkeypatch.delenv("ADAMIXTURE_MAX_CPU_MEM_MB")
    seen_q_hist.clear()
    real_zeros = np.zeros
    dim = (M + N) * K

    def fail_first_uv(shape, dtype=float, *a, **kw):
        if shape == dim * 3 and dtype == np.float64:
            raise MemoryError("simulated ZAL U/V OOM")
        return real_zeros(shape, dtype=dtype, *a, **kw)

    monkeypatch.setattr(br_qn.np, "zeros", fail_first_uv)
    with pytest.raises(MemoryError, match=r"Out of memory allocating BR-QN ZAL history buffers.*Q_hist=3"):
        br_qn.optimize_original(G, P.copy(), Q.copy(), 2, K, M, N, 0.0, Q_hist=3, patience=3, verbose=False)
    assert seen_q_hist == []

