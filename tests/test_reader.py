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


@pytest.mark.parametrize(
    "filename",
    ["demo_data.bed", "demo_data.vcf", "demo_data.pgen"],
)
def test_specific_chrom_overrides_autosomes(filename: str) -> None:
    # Single chromosome as int/str list
    G_spec1, N, M_spec1 = utils.read_data(
        str(DATA_DIR / filename),
        packed=False,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        specific_chrom=[1],
        verbose=False,
    )
    assert M_spec1 == 682
    assert N == 105

    # Multiple chromosomes as list
    G_spec12, N, M_spec12 = utils.read_data(
        str(DATA_DIR / filename),
        packed=False,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        specific_chrom=["1", "2"],
        verbose=False,
    )
    assert M_spec12 == 1454

    # Prefix 'chr'
    G_chr1, N, M_chr1 = utils.read_data(
        str(DATA_DIR / filename),
        packed=False,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        specific_chrom=["chr1"],
        verbose=False,
    )
    assert M_chr1 == 682

    # Ignored when chrom_mode="all"
    G_all, N, M_all = utils.read_data(
        str(DATA_DIR / filename),
        packed=False,
        chunk_size=CHUNK_SIZE,
        chrom_mode="all",
        autosomes=22,
        specific_chrom=[1],
        verbose=False,
    )
    assert M_all == 8451


@pytest.mark.parametrize(
    "filename",
    ["demo_data.bed", "demo_data.vcf", "demo_data.pgen"],
)
def test_zero_snps_exits_with_error(filename: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        utils.read_data(
            str(DATA_DIR / filename),
            packed=False,
            chunk_size=CHUNK_SIZE,
            chrom_mode="autosomes",
            autosomes=22,
            specific_chrom=[9999],
            verbose=False,
        )
    assert exc_info.value.code == 1

