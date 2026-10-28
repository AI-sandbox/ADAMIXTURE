from pathlib import Path

import numpy as np
import pytest

from adamixture.src.utils_c import pack_genotypes, read_pgen_file, read_pgen_file_packed

PGEN_DIR = Path(__file__).resolve().parent / "fixtures" / "pgen"
EXPECTED = np.load(PGEN_DIR / "expected.npz")

# types_n300: main-track record types 0-4, 6, 7; vblocks_n4: several 2^16-variant
# blocks; ids3_n70000: 3-byte sample IDs in difflists; phased_dosage_n120: phase
# and dosage auxiliary tracks; fixed_mode*: fixed-width storage modes with a
# provisional-REF bitarray; external_index_n300: mode 0x20 with a .pgen.pgi file.
CASES = sorted(EXPECTED.files)


@pytest.mark.parametrize("name", CASES)
def test_pgen_unpacked_matches_expected(name: str) -> None:
    expected = EXPECTED[name]
    G, N, M = read_pgen_file(str(PGEN_DIR / f"{name}.pgen"), 4096, "all", 22)
    assert (M, N) == expected.shape
    np.testing.assert_array_equal(G, expected)


@pytest.mark.parametrize("name", CASES)
def test_pgen_packed_matches_expected(name: str) -> None:
    expected = np.ascontiguousarray(EXPECTED[name])
    M, N = expected.shape
    G_packed, n, m = read_pgen_file_packed(str(PGEN_DIR / f"{name}.pgen"), 100, "all", 22)
    ref = np.zeros(((M + 3) // 4, N), dtype=np.uint8)
    pack_genotypes(expected.ctypes.data, ref.ctypes.data, M, N, ref.shape[0])
    assert (m, n) == (M, N)
    np.testing.assert_array_equal(G_packed, ref)


def test_pgen_truncated_raises(tmp_path: Path) -> None:
    data = (PGEN_DIR / "types_n300.pgen").read_bytes()
    for cut in (2, 11, 40, len(data) // 2, len(data) - 1):
        path = tmp_path / f"cut{cut}.pgen"
        path.write_bytes(data[:cut])
        with pytest.raises(ValueError):
            read_pgen_file(str(path), 4096, "all", 22)


def test_pgen_missing_external_index_raises(tmp_path: Path) -> None:
    path = tmp_path / "noindex.pgen"
    path.write_bytes((PGEN_DIR / "external_index_n300.pgen").read_bytes())
    with pytest.raises(FileNotFoundError):
        read_pgen_file(str(path), 4096, "all", 22)
