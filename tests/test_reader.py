import pytest

from adamixture.src import utils

from tests.config import CHUNK_SIZE, DATA_DIR, READER_EXPECTED_DIR
from tests.helpers import assert_matches_expected, to_numpy


READER_CASES = [
    ("demo_data.bed", "bed"),
    ("demo_data.bed.gz", "bed_gz_same_base"),
    ("demo_data.bed.zst", "bed_zst_same_base"),
    ("demo_data_bed_gz.bed.gz", "bed_gz"),
    ("demo_data_bed_zst.bed.zst", "bed_zst"),
    ("demo_data.pgen", "pgen"),
    ("demo_data.pgen.zst", "pgen_zst_same_base"),
    ("demo_data_pgen_sidecars_gz.pgen", "pgen_sidecars_gz"),
    ("demo_data_pgen_sidecars_zst.pgen", "pgen_sidecars_zst"),
    ("demo_data_pgen_zst.pgen.zst", "pgen_zst"),
    ("demo_data.vcf", "vcf"),
    ("demo_data.vcf.gz", "vcf_gz"),
    ("demo_data.vcf.zst", "vcf_zst"),
]


@pytest.mark.parametrize(
    ("filename", "label"),
    READER_CASES,
)
def test_reader_unpacked_matches_expected(filename: str, label: str) -> None:
    del label

    G, N, M = utils.read_data(
        str(DATA_DIR / filename),
        packed=False,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        verbose=False,
    )

    assert G.shape == (8451, 105)
    assert N == 105
    assert M == 8451
    assert G.dtype.name == "uint8"
    assert_matches_expected(READER_EXPECTED_DIR, "demo_data.G.expected", G)


@pytest.mark.parametrize(
    ("filename", "label"),
    READER_CASES,
)
def test_reader_packed_matches_expected(filename: str, label: str) -> None:
    del label

    G, N, M = utils.read_data(
        str(DATA_DIR / filename),
        packed=True,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        verbose=False,
    )
    G_np = to_numpy(G)

    assert G_np.shape == (2113, 105)
    assert N == 105
    assert M == 8451
    assert G_np.dtype.name == "uint8"
    assert_matches_expected(READER_EXPECTED_DIR, "demo_data.G.packed.expected", G_np)
