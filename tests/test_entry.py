import pytest

from adamixture.entry import parse_args


MIN_ARGS = ["--save_dir", "out", "--data_path", "data.bed", "--name", "t"]


def test_k1_is_accepted() -> None:
    args = parse_args(["-k", "1", *MIN_ARGS])
    assert args.k == 1


def test_min_k1_is_accepted() -> None:
    args = parse_args(["--min_k", "1", "--max_k", "6", *MIN_ARGS])
    assert args.min_k == 1
    assert args.max_k == 6


def test_k0_is_rejected() -> None:
    with pytest.raises(SystemExit):
        parse_args(["-k", "0", *MIN_ARGS])
