import argparse
from pathlib import Path

import numpy as np

from adamixture.src.plot import (
    _build_contiguous_spans,
    _check_hierarchy_codes,
    _format_label,
    _require_pillow_image,
    align_clusters_greedy,
    plot_combined,
    plot_q_matrix,
)


def test_pillow_is_available_through_matplotlib_dependencies() -> None:
    image_module = _require_pillow_image()
    assert callable(image_module.open)
    assert callable(image_module.new)


def test_format_label() -> None:
    assert _format_label("barcelona") == "Barcelona"
    long_name = "a" * 30
    formatted = _format_label(long_name)
    assert len(formatted) == 25
    assert formatted.endswith("…")


def test_align_clusters_greedy() -> None:
    np.random.seed(42)
    N = 1000
    K = 4
    raw = np.random.exponential(scale=1.0, size=(N, K))
    Q_ref = raw / raw.sum(axis=1, keepdims=True)

    # Permute columns of Q_ref
    true_perm = [2, 0, 3, 1]
    Q_query = Q_ref[:, true_perm]

    inferred_perm = align_clusters_greedy(Q_ref, Q_query)
    # inferred_perm should map ref cluster r to query column
    assert np.array_equal(inferred_perm, [1, 3, 0, 2])

    # Reconstructed Q_query should match Q_ref
    np.testing.assert_allclose(Q_query[:, inferred_perm], Q_ref, atol=1e-10)


def test_align_clusters_greedy_different_k() -> None:
    np.random.seed(42)
    N = 500
    Q_ref = np.random.dirichlet([1.0] * 3, size=N)
    Q_query = np.random.dirichlet([1.0] * 5, size=N)

    perm = align_clusters_greedy(Q_ref, Q_query)
    assert len(perm) == 5
    assert len(set(perm)) == 5


def test_check_hierarchy_codes() -> None:
    # Valid hierarchy: child 0->parent 0, child 1->parent 0, child 2->parent 1
    child_codes = np.array([0, 1, 0, 1, 2, 2])
    parent_codes = np.array([0, 0, 0, 0, 1, 1])
    assert _check_hierarchy_codes(child_codes, 3, parent_codes, "child", "parent") is True

    # Invalid hierarchy: child 0 maps to parent 0 AND parent 1
    invalid_parent = np.array([0, 0, 1, 0, 1, 1])
    assert _check_hierarchy_codes(child_codes, 3, invalid_parent, "child", "parent") is False


def test_build_contiguous_spans() -> None:
    codes = np.array([0, 0, 0, 1, 1, 2])
    names = np.array(["PopA", "PopB", "PopC"])
    spans = _build_contiguous_spans(codes, names)
    assert len(spans) == 3
    assert spans[0] == {"name": "PopA", "start": 0, "end": 3}
    assert spans[1] == {"name": "PopB", "start": 3, "end": 5}
    assert spans[2] == {"name": "PopC", "start": 5, "end": 6}


def test_plot_q_matrix_and_combined_e2e(tmp_path: Path) -> None:
    np.random.seed(42)
    N, K = 100, 3
    raw = np.random.exponential(size=(N, K))
    Q = raw / raw.sum(axis=1, keepdims=True)

    labels1 = [f"Pop_{i % 4}" for i in range(N)]
    labels2 = [f"Reg_{i % 2}" for i in range(N)]

    out_png = tmp_path / "test.png"
    plot_q_matrix(Q, out_png, dpi=100, format="png", labels=labels1, labels2=labels2)
    assert out_png.exists()
    assert out_png.stat().st_size > 0

    # Test plot_combined
    k_vals = [3, 4]
    trained_plot = {
        3: (None, Q),
        4: (None, np.random.dirichlet([1.0] * 4, size=N)),
    }
    lbl_file = tmp_path / "labels.txt"
    lbl_file.write_text("\n".join(labels1))

    args = argparse.Namespace(
        labels=str(lbl_file),
        labels2=None,
        labels3=None,
        colors=None,
        save_dir=str(tmp_path),
        name="combined_test",
        plot_format="png",
        plot_dpi=100,
    )
    plot_combined(args, k_vals, trained_plot)
    combined_png = tmp_path / "combined_test.3_4.png"
    assert combined_png.exists()
    assert combined_png.stat().st_size > 0
