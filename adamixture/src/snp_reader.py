import gzip
import logging
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from .utils_c import (
    flip_packed,
    flip_unpacked,
    get_mean_packed,
    get_mean_unpacked,
    pack_genotypes,
    read_bed,
    read_bed_packed,
    read_pgen_file,
    read_pgen_file_packed,
    read_vcf_file,
    read_vcf_file_packed,
    replace_missing_with_three,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class SNPReader:
    """
    Wrapper to read genotype data from several formats.
    """
    _COMPRESSION_EXTENSIONS = (".gz", ".zst")
    _GENOTYPE_EXTENSIONS = (".bed", ".vcf", ".pgen", ".psam", ".pvar", ".fam", ".bim")

    def _open_zst(self, path: Path, mode: str):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "Reading .zst-compressed genotype files requires the 'zstandard' package."
            ) from exc
        return zstd.open(path, mode, encoding="utf-8" if "t" in mode else None)

    def _open_text(self, path: Path):
        if path.suffix == ".gz":
            return gzip.open(path, "rt")
        if path.suffix == ".zst":
            return self._open_zst(path, "rt")
        return open(path)

    @contextmanager
    def _materialize_binary(self, path: Path) -> Iterator[Path]:
        if path.suffix not in self._COMPRESSION_EXTENSIONS:
            yield path
            return

        inner_suffix = path.suffixes[-2] if len(path.suffixes) >= 2 else ""
        tmp = tempfile.NamedTemporaryFile(suffix=inner_suffix, delete=False)
        tmp_path = Path(tmp.name)
        try:
            with tmp:
                if path.suffix == ".gz":
                    source = gzip.open(path, "rb")
                elif path.suffix == ".zst":
                    source = self._open_zst(path, "rb")
                else:
                    source = open(path, "rb")
                with source:
                    shutil.copyfileobj(source, tmp)
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)

    def _compressed_variants(self, extension: str) -> tuple[str, ...]:
        return (extension, *(extension + compression for compression in self._COMPRESSION_EXTENSIONS))

    def _known_extensions(self, extensions: list[str] | tuple[str, ...]) -> tuple[str, ...]:
        expanded = []
        for extension in extensions:
            if extension.endswith(self._COMPRESSION_EXTENSIONS):
                expanded.append(extension)
            else:
                expanded.extend(self._compressed_variants(extension))
        return tuple(dict.fromkeys(expanded))

    def _resolve_existing(
        self,
        base_path: str,
        extensions: list[str] | tuple[str, ...],
        requested_file: str | None = None,
    ) -> Path | None:
        known_extensions = self._known_extensions(extensions)

        if requested_file is not None:
            requested_path = Path(requested_file)
            requested_str = str(requested_path)
            if requested_path.exists() and any(requested_str.endswith(extension) for extension in known_extensions):
                return requested_path

        for extension in known_extensions:
            candidate = Path(base_path + extension)
            if candidate.exists():
                return candidate
        return None

    def _parse_chromosome_number(self, chrom: str) -> int | None:
        """
        Description:
        Parses plain numeric chromosome labels and common chr-prefixed labels.

        Args:
            chrom (str): Chromosome label from the variant metadata.

        Returns:
            int | None: Parsed chromosome number, or None if the label is non-numeric.
        """
        chrom = chrom.strip()
        if chrom.lower().startswith("chr"):
            chrom = chrom[3:]
        if not chrom.isdigit():
            return None
        return int(chrom)

    @staticmethod
    def _normalize_specific_chrom(
        specific_chrom: list | tuple | str | int | set | None
    ) -> list[int | str] | None:
        """
        Description:
        Normalizes specific_chrom parameter into a cleaned list of chromosome identifiers.

        Args:
            specific_chrom: Input chromosomes specification.

        Returns:
            list[int | str] | None: Cleaned list of chromosome identifiers or None.
        """
        if specific_chrom is None:
            return None

        if isinstance(specific_chrom, (int, str)):
            raw_items = [str(specific_chrom)]
        elif isinstance(specific_chrom, (list, tuple, set)):
            raw_items = [str(item) for item in specific_chrom]
        else:
            return None

        cleaned_items = []
        for item in raw_items:
            parts = [p.strip() for p in item.replace(",", " ").split() if p.strip()]
            for p in parts:
                p_clean = p[3:] if p.lower().startswith("chr") and p[3:].isdigit() else p
                if p_clean.isdigit():
                    val = int(p_clean)
                    if val not in cleaned_items:
                        cleaned_items.append(val)
                else:
                    if p not in cleaned_items:
                        cleaned_items.append(p)

        return cleaned_items if len(cleaned_items) > 0 else None

    @classmethod
    def _prepare_specific_chrom(
        cls, specific_chrom: list | tuple | str | int | set | None
    ) -> tuple[set[int], set[str]] | None:
        norm = cls._normalize_specific_chrom(specific_chrom)
        if norm is None:
            return None

        allowed_nums: set[int] = set()
        allowed_strs: set[str] = set()

        for item in norm:
            if isinstance(item, int):
                allowed_nums.add(item)
                allowed_strs.add(str(item))
                allowed_strs.add(f"chr{item}")
            else:
                s_lower = str(item).strip().lower()
                allowed_strs.add(s_lower)
                if s_lower.startswith("chr"):
                    bare = s_lower[3:]
                    allowed_strs.add(bare)
                    if bare.isdigit():
                        allowed_nums.add(int(bare))
                else:
                    allowed_strs.add(f"chr{s_lower}")
                    if s_lower.isdigit():
                        allowed_nums.add(int(s_lower))

        return allowed_nums, allowed_strs

    def _keep_chromosome(
        self,
        chrom: str,
        chrom_mode: str,
        autosomes: int,
        specific_chrom: list | tuple | str | int | set | None = None,
    ) -> bool:
        """
        Description:
        Decides whether a variant should be kept under the configured chromosome filter.

        Args:
            chrom (str): Chromosome label from the variant metadata.
            chrom_mode (str): Chromosome filter mode ("all" or "autosomes").
            autosomes (int): Number of autosomes kept when chrom_mode is "autosomes".
            specific_chrom: Specific chromosomes to keep when chrom_mode is "autosomes".

        Returns:
            bool: True if the variant should be kept, otherwise False.
        """
        if chrom_mode == "all":
            return True
        if chrom_mode != "autosomes":
            raise ValueError("chrom_mode must be 'all' or 'autosomes'")

        prepared = self._prepare_specific_chrom(specific_chrom)
        if prepared is not None:
            allowed_nums, allowed_strs = prepared
            chrom_num = self._parse_chromosome_number(chrom)
            if chrom_num is not None and chrom_num in allowed_nums:
                return True
            chrom_lower = chrom.strip().lower()
            if chrom_lower in allowed_strs:
                return True
            return False

        if autosomes < 1:
            raise ValueError("autosomes must be at least 1")

        chrom_num = self._parse_chromosome_number(chrom)
        return chrom_num is not None and 1 <= chrom_num <= autosomes

    def _log_chromosome_filter(
        self,
        skipped: int,
        chrom_mode: str,
        autosomes: int,
        specific_chrom: list | tuple | str | int | set | None = None,
    ) -> None:
        """
        Description:
        Logs a warning when variants are skipped by the chromosome filter.

        Args:
            skipped (int): Number of skipped variants.
            chrom_mode (str): Chromosome filter mode ("all" or "autosomes").
            autosomes (int): Number of autosomes kept when chrom_mode is "autosomes".
            specific_chrom: Specific chromosomes filter list.

        Returns:
            None
        """
        if skipped <= 0:
            return
        if chrom_mode == "autosomes":
            norm = self._normalize_specific_chrom(specific_chrom)
            if norm is not None:
                log.warning(
                    f"        Warning: Skipped {skipped} SNPs outside specific chromosomes {norm}."
                )
            else:
                log.warning(
                    f"        Warning: Skipped {skipped} SNPs outside autosomes 1..{autosomes}."
                )
        else:
            log.warning(f"        Warning: Skipped {skipped} SNPs excluded by chromosome filter.")

    def _get_base_path(self, file: str) -> str:
        """
        Description:
        Determines the base path by stripping known genotype extensions.

        Args:
            file (str): Input genotype file path.

        Returns:
            str: Base path without a known genotype extension.
        """
        file_str = str(file)
        for ext in self._known_extensions(self._GENOTYPE_EXTENSIONS):
            if file_str.endswith(ext):
                return file_str[:-len(ext)]
        return str(Path(file).with_suffix(''))

    def _read_bed(
        self,
        file: str,
        packed: bool,
        chunk_size: int,
        chrom_mode: str,
        autosomes: int,
        specific_chrom: list | tuple | str | int | set | None = None,
    ) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Internal reader for PLINK BED files. Handles both regular (uint8) and
        packed (2-bit) formats for GPU acceleration.

        Args:
            file (str): Path to the BED file (without extension or with .bed).
            packed (bool): If True, returns a 2-bit packed torch.Tensor. Defaults to False.

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        log.info("    Input format is BED.")

        base_path = self._get_base_path(file)
        fam_file = self._resolve_existing(base_path, [".fam"])
        bed_file = self._resolve_existing(base_path, [".bed"], requested_file=file)
        bim_file = self._resolve_existing(base_path, [".bim"])
        if fam_file is None or bed_file is None or bim_file is None:
            log.error(f"    Error: Required BED files missing for {base_path}")
            sys.exit(1)

        with self._open_text(fam_file) as fam:
            N = sum(1 for _ in fam)
        N_bytes = ceil(N / 4)

        keep_mask = []
        with self._open_text(bim_file) as bim:
            for line in bim:
                parts = line.strip().split()
                if not parts:
                    continue
                keep_mask.append(
                    self._keep_chromosome(parts[0], chrom_mode, autosomes, specific_chrom)
                )
        keep_mask = np.array(keep_mask, dtype=bool)

        with self._materialize_binary(bed_file) as readable_bed_file:
            file_size = readable_bed_file.stat().st_size
            assert ((file_size - 3) % N_bytes) == 0, "bim file doesn't match!"
            M_total = (file_size - 3) // N_bytes
            assert len(keep_mask) == M_total, "bim file doesn't match!"

            skipped = len(keep_mask) - keep_mask.sum()
            self._log_chromosome_filter(skipped, chrom_mode, autosomes, specific_chrom)

            if not packed:
                with open(readable_bed_file, "rb") as bed:
                    B_raw = np.fromfile(bed, dtype=np.uint8, offset=3)

                assert (B_raw.shape[0] % N_bytes) == 0, "bim file doesn't match!"
                M = B_raw.shape[0] // N_bytes
                B_raw = B_raw.reshape(M, N_bytes)

                B = np.ascontiguousarray(B_raw[keep_mask])
                M = B.shape[0]

                G = np.zeros((M, N), dtype=np.uint8)
                read_bed(B, G)
                del B, B_raw
                return G, N, M
            else:
                log.info("        Reading BED in packed 2-bit format for GPU use.")
                with open(readable_bed_file, "rb") as f:
                    f.seek(3)
                    buffer = bytearray(f.read())

                B_raw = np.frombuffer(buffer, dtype=np.uint8)
                assert (B_raw.shape[0] % N_bytes) == 0, "bim file doesn't match!"
                M = B_raw.shape[0] // N_bytes

                B_raw = B_raw.reshape(M, N_bytes)
                B_raw = np.ascontiguousarray(B_raw[keep_mask])
                M = B_raw.shape[0]

                B = torch.from_numpy(B_raw)
                M_bytes = (M + 3) // 4
                G = torch.zeros((M_bytes, N), dtype=torch.uint8)

                read_bed_packed(B.data_ptr(), G.data_ptr(), M, N_bytes, N, M_bytes)
                del B, B_raw
                return G, N, M

    def _read_vcf(
        self,
        file: str,
        packed: bool,
        chunk_size: int,
        chrom_mode: str,
        autosomes: int,
        specific_chrom: list | tuple | str | int | set | None = None,
    ) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Internal reader for VCF files using Cython-based parser.
        Handles both regular (uint8) and packed (2-bit) formats for GPU acceleration.

        Args:
            file (str): Path to the VCF file.
            packed (bool): If True, returns a 2-bit packed torch.Tensor. Defaults to False.
            chunk_size (int): Size of chunks to read for VCF files. Defaults to 4096.

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        log.info("    Input format is VCF.")

        base_path = self._get_base_path(file)
        vcf_file = self._resolve_existing(base_path, [".vcf"], requested_file=file)
        if vcf_file is None:
            log.error(f"    Error: VCF file not found for {base_path}")
            sys.exit(1)

        norm_spec = self._normalize_specific_chrom(specific_chrom)

        try:
            if not packed:
                G, N, M = read_vcf_file(
                    str(vcf_file),
                    chunk_size=chunk_size,
                    chrom_mode=chrom_mode,
                    autosomes=autosomes,
                    specific_chrom=norm_spec,
                )
                return np.ascontiguousarray(G), N, M
            else:
                log.info("        Reading VCF in packed 2-bit format for GPU use.")
                G_packed_np, N, M = read_vcf_file_packed(
                    str(vcf_file),
                    chunk_size=chunk_size,
                    chrom_mode=chrom_mode,
                    autosomes=autosomes,
                    specific_chrom=norm_spec,
                )
                G_packed = torch.from_numpy(G_packed_np)
                return G_packed, N, M
        except ValueError as err:
            log.error(
                f"    Error: {err}\n"
                "    Please check if your chromosome filter (--chrom_mode, --specific_chrom, --autosomes) "
                "excluded all variants, or if the input dataset has no variants."
            )
            sys.exit(1)

    def _read_pgen(
        self,
        file: str,
        packed: bool,
        chunk_size: int,
        chrom_mode: str,
        autosomes: int,
        specific_chrom: list | tuple | str | int | set | None = None,
    ) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Internal reader for PLINK PGEN files.

        Args:
            file (str): Path to the PGEN file (without extension or with .pgen).
            packed (bool): If True, returns a 2-bit packed torch.Tensor. Defaults to False.

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        log.info("    Input format is PGEN.")

        base_path = self._get_base_path(file)
        pgen_file = self._resolve_existing(base_path, [".pgen"], requested_file=file)
        var_file = self._resolve_existing(base_path, [".pvar", ".bim"])
        if pgen_file is None:
            log.error(f"    Error: PGEN file not found for {base_path}")
            sys.exit(1)
        if var_file is None:
            log.error(f"    Error: Variant file (.pvar or .bim) not found for {base_path}")
            sys.exit(1)

        keep_mask = []
        with self._open_text(var_file) as vf:
            for line in vf:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                keep_mask.append(
                    self._keep_chromosome(parts[0], chrom_mode, autosomes, specific_chrom)
                )
        keep_mask = np.array(keep_mask, dtype=bool)

        with self._materialize_binary(pgen_file) as readable_pgen_file:
            if packed:
                log.info("        Reading PGEN in packed 2-bit format for GPU use.")
                G_packed, N, M = read_pgen_file_packed(
                    str(readable_pgen_file),
                    chunk_size,
                    chrom_mode,
                    autosomes,
                    orig_filepath=str(pgen_file),
                )
                if not np.all(keep_mask):
                    G_packed = G_packed[keep_mask]
                    M = G_packed.shape[0]
                if M == 0:
                    log.error(
                        "    Error: No variants found in dataset matching chromosome filters.\n"
                        "    Please check if your chromosome filter (--chrom_mode, --specific_chrom, --autosomes) "
                        "excluded all variants, or if the input dataset has no variants."
                    )
                    sys.exit(1)
                return torch.from_numpy(G_packed), N, M

            G_np, N, M = read_pgen_file(
                str(readable_pgen_file),
                chunk_size,
                chrom_mode,
                autosomes,
                orig_filepath=str(pgen_file),
            )
            if not np.all(keep_mask):
                G_np = G_np[keep_mask]
                M = G_np.shape[0]
            if M == 0:
                log.error(
                    "    Error: No variants found in dataset matching chromosome filters.\n"
                    "    Please check if your chromosome filter (--chrom_mode, --specific_chrom, --autosomes) "
                    "excluded all variants, or if the input dataset has no variants."
                )
                sys.exit(1)
            return G_np, N, M

    def _check_files_exist(self, file: str, extensions: list[str], match_any: bool = False):
        """
        Description:
        Check if required files exist.

        Args:
            file (str): Path to the genotype file.
            extensions (list[str]): List of extensions to check for.
            match_any (bool): If True, check if any of the extensions exist. Defaults to False.

        Returns:
            None
        """
        base_path = self._get_base_path(file)

        if match_any:
            if self._resolve_existing(base_path, extensions, requested_file=file) is None:
                log.error(f"    Error: Could not find any of these files: {extensions} for {base_path}")
                sys.exit(1)
        else:
            missing = [
                base_path + ext
                for ext in extensions
                if self._resolve_existing(base_path, [ext], requested_file=file) is None
            ]
            if missing:
                log.error(f"    Error: Required files missing: {', '.join(missing)}")
                sys.exit(1)

    def read_data(
        self,
        file: str,
        packed: bool,
        chunk_size: int,
        chrom_mode: str,
        autosomes: int,
        specific_chrom: list | tuple | str | int | set | None = None,
    ) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Public wrapper to read genotype data from various formats (BED, VCF).
        Automatically detects format based on file extension.

        Args:
            file (str): Path to the genotype file.
            packed (bool): If True, returns a 2-bit packed torch.Tensor (BED, PGEN, VCF). Defaults to False.
            chunk_size (int): Size of chunks to read for VCF files. Defaults to 4096.
            chrom_mode (str): "all" to keep all chromosomes or "autosomes" to keep 1..autosomes.
            autosomes (int): Number of autosomes when chrom_mode is "autosomes".
            specific_chrom: List of specific chromosomes to analyze when chrom_mode is "autosomes".

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        file_path = Path(file)
        file_extensions = file_path.suffixes
        start = time.time()

        if chrom_mode not in {"all", "autosomes"}:
            raise ValueError("chrom_mode must be 'all' or 'autosomes'")

        norm_spec = self._normalize_specific_chrom(specific_chrom)
        if chrom_mode == "autosomes" and norm_spec is None:
            if autosomes < 1:
                raise ValueError("autosomes must be at least 1")

        if '.bed' in file_extensions:
            self._check_files_exist(file, ['.bed', '.fam', '.bim'])
            G, N, M = self._read_bed(
                file, packed, chunk_size, chrom_mode, autosomes, specific_chrom=specific_chrom
            )
        elif '.vcf' in file_extensions:
            self._check_files_exist(file, ['.vcf'], match_any=True)
            G, N, M = self._read_vcf(
                file, packed, chunk_size, chrom_mode, autosomes, specific_chrom=specific_chrom
            )
        elif '.pgen' in file_extensions:
            self._check_files_exist(file, ['.pgen', '.psam'])
            self._check_files_exist(file, ['.pvar', '.bim'], match_any=True)
            G, N, M = self._read_pgen(
                file, packed, chunk_size, chrom_mode, autosomes, specific_chrom=specific_chrom
            )
        else:
            log.error("    Invalid format. Unrecognized file format. Make sure file ends with .bed, .pgen or .vcf .")
            sys.exit(1)

        if not packed:
            mean_val = get_mean_unpacked(G) if M > 0 else 0.0
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (unpacked).")
                flip_unpacked(G)
        else:
            M_bytes = G.shape[0]
            mean_val = get_mean_packed(G.data_ptr(), M, N, M_bytes) if M > 0 else 0.0
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (packed).")
                flip_packed(G.data_ptr(), M, N, M_bytes)

        end = time.time()
        log.info(f"        Total time for reading={end - start:.3f}s")

        if M == 0:
            log.error(
                "    Error: Genotype data contains 0 SNPs after filtering.\n"
                "    Please check if your chromosome filter (--chrom_mode, --specific_chrom, --autosomes) "
                "excluded all variants, or if the input dataset has no variants."
            )
            sys.exit(1)
        if N == 0:
            log.error(
                "    Error: Genotype data contains 0 samples.\n"
                "    Please check if the input dataset is empty or corrupted."
            )
            sys.exit(1)

        return G, N, M
