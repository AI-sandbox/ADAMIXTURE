import logging
import sys
import time
from math import ceil
from pathlib import Path

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
    read_vcf_file,
    read_vcf_file_packed,
)

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class SNPReader:
    """
    Wrapper to read genotype data from several formats.
    """

    def _get_base_path(self, file: str) -> str:
        """
        Determines the base path by stripping known genotype extensions.
        """
        file_str = str(file)
        for ext in ['.bed', '.vcf.gz', '.vcf', '.pgen', '.psam', '.pvar', '.fam', '.bim']:
            if file_str.endswith(ext):
                return file_str[:-len(ext)]
        return str(Path(file).with_suffix(''))

    def _read_bed(self, file: str, packed: bool) -> tuple[torch.Tensor | np.ndarray, int, int]:
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
        fam_file = base_path + ".fam"
        bed_file = base_path + ".bed"

        with open(fam_file) as fam:
            N = sum(1 for _ in fam)
        N_bytes = ceil(N / 4)

        if not packed:
            with open(bed_file, "rb") as bed:
                B = np.fromfile(bed, dtype=np.uint8, offset=3)

            assert (B.shape[0] % N_bytes) == 0, "bim file doesn't match!"
            M = B.shape[0] // N_bytes
            B.shape = (M, N_bytes)

            bim_file = base_path + ".bim"
            keep_mask = []
            with open(bim_file) as bim:
                for line in bim:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    chrom = parts[0]
                    try:
                        int(chrom)
                        keep_mask.append(True)
                    except ValueError:
                        keep_mask.append(False)
            keep_mask = np.array(keep_mask, dtype=bool)
            assert len(keep_mask) == M, "bim file doesn't match!"

            skipped = len(keep_mask) - keep_mask.sum()
            if skipped > 0:
                log.warning(f"        Warning: Skipped {skipped} SNPs on non-numeric chromosomes (e.g. sex chromosomes).")

            B = np.ascontiguousarray(B[keep_mask])
            M = B.shape[0]

            G = np.zeros((M, N), dtype=np.uint8)
            read_bed(B, G)
            del B
            return G, N, M
        else:
            log.info("        Reading BED in packed 2-bit format for GPU use.")
            file_size = Path(bed_file).stat().st_size
            buffer = bytearray(file_size - 3)
            with open(bed_file, "rb") as f:
                f.seek(3)
                f.readinto(buffer)

            B_raw = np.frombuffer(buffer, dtype=np.uint8)
            assert (B_raw.shape[0] % N_bytes) == 0, "bim file doesn't match!"
            M = B_raw.shape[0] // N_bytes

            bim_file = base_path + ".bim"
            keep_mask = []
            with open(bim_file) as bim:
                for line in bim:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    chrom = parts[0]
                    try:
                        int(chrom)
                        keep_mask.append(True)
                    except ValueError:
                        keep_mask.append(False)
            keep_mask = np.array(keep_mask, dtype=bool)
            assert len(keep_mask) == M, "bim file doesn't match!"

            skipped = len(keep_mask) - keep_mask.sum()
            if skipped > 0:
                log.warning(f"        Warning: Skipped {skipped} SNPs on non-numeric chromosomes (e.g. sex chromosomes).")

            B_raw.shape = (M, N_bytes)
            B_raw = np.ascontiguousarray(B_raw[keep_mask])
            M = B_raw.shape[0]

            B = torch.from_numpy(B_raw)
            M_bytes = (M + 3) // 4
            G = torch.zeros((M_bytes, N), dtype=torch.uint8)

            read_bed_packed(B.data_ptr(), G.data_ptr(), M, N_bytes, N, M_bytes)
            del B
            return G, N, M

    def _read_vcf(self, file: str, packed: bool, chunk_size: int) -> tuple[torch.Tensor | np.ndarray, int, int]:
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

        if not packed:
            G, N, M = read_vcf_file(file, chunk_size=chunk_size)
            return np.ascontiguousarray(G), N, M
        else:
            log.info("        Reading VCF in packed 2-bit format for GPU use.")
            G_packed_np, N, M = read_vcf_file_packed(file, chunk_size=chunk_size)
            G_packed = torch.from_numpy(G_packed_np)
            return G_packed, N, M

    def _read_pgen(self, file: str, packed: bool) -> tuple[torch.Tensor | np.ndarray, int, int]:
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
        import pgenlib as pg

        with pg.PgenReader(str(file).encode()) as pgen_reader:
            num_vars = pgen_reader.get_variant_ct()
            num_samples = pgen_reader.get_raw_sample_ct()

            G_raw = np.empty((num_vars, num_samples), dtype=np.int8)
            pgen_reader.read_range(0, num_vars, G_raw)
            G_raw[G_raw == -9] = 3
            G = G_raw.view(np.uint8)

        base_path = self._get_base_path(file)
        pvar_file = Path(base_path + ".pvar")
        bim_file = Path(base_path + ".bim")
        if pvar_file.exists():
            var_file = pvar_file
        elif bim_file.exists():
            var_file = bim_file
        else:
            log.error(f"    Error: Variant file (.pvar or .bim) not found for {base_path}")
            sys.exit(1)

        keep_mask = []
        with open(var_file) as vf:
            for line in vf:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if not parts:
                    continue
                chrom = parts[0]
                try:
                    int(chrom)
                    keep_mask.append(True)
                except ValueError:
                    keep_mask.append(False)
        keep_mask = np.array(keep_mask, dtype=bool)

        assert len(keep_mask) == num_vars, f"Variant file line count {len(keep_mask)} does not match PGEN variant count {num_vars}!"

        skipped = len(keep_mask) - keep_mask.sum()
        if skipped > 0:
            log.warning(f"        Warning: Skipped {skipped} SNPs on non-numeric chromosomes (e.g. sex chromosomes).")

        G = np.ascontiguousarray(G[keep_mask])
        M, N = G.shape

        if packed:
            log.info("        Reading PGEN in packed 2-bit format for GPU use.")
            M_bytes = (M + 3) // 4
            G_packed = torch.zeros((M_bytes, N), dtype=torch.uint8)
            pack_genotypes(G.ctypes.data, G_packed.data_ptr(), M, N, M_bytes)
            return G_packed, N, M

        return G, N, M

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
            if not any(Path(base_path + ext).exists() for ext in extensions):
                log.error(f"    Error: Could not find any of these files: {extensions} for {base_path}")
                sys.exit(1)
        else:
            missing = [base_path + ext for ext in extensions if not Path(base_path + ext).exists()]
            if missing:
                log.error(f"    Error: Required files missing: {', '.join(missing)}")
                sys.exit(1)

    def read_data(self, file: str, packed: bool, chunk_size: int) -> tuple[torch.Tensor | np.ndarray, int, int]:
        """
        Description:
        Public wrapper to read genotype data from various formats (BED, VCF).
        Automatically detects format based on file extension.

        Args:
            file (str): Path to the genotype file.
            packed (bool): If True, returns a 2-bit packed torch.Tensor (BED, PGEN, VCF). Defaults to False.
            chunk_size (int): Size of chunks to read for VCF files. Defaults to 4096.

        Returns:
            tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N individuals, M SNPs)
        """
        file_path = Path(file)
        file_extensions = file_path.suffixes
        start = time.time()

        if '.bed' in file_extensions:
            self._check_files_exist(file, ['.bed', '.fam', '.bim'])
            G, N, M = self._read_bed(file, packed)
        elif '.vcf' in file_extensions:
            self._check_files_exist(file, ['.vcf', '.vcf.gz'], match_any=True)
            G, N, M = self._read_vcf(file, packed, chunk_size)
        elif '.pgen' in file_extensions:
            self._check_files_exist(file, ['.pgen', '.psam', '.pvar'])
            G, N, M = self._read_pgen(file, packed)
        else:
            log.error("    Invalid format. Unrecognized file format. Make sure file ends with .bed, .pgen or .vcf .")
            sys.exit(1)

        if not packed:
            mean_val = get_mean_unpacked(G)
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (unpacked).")
                flip_unpacked(G)
        else:
            M_bytes = G.shape[0]
            mean_val = get_mean_packed(G.data_ptr(), M, N, M_bytes)
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (packed).")
                flip_packed(G.data_ptr(), M, N, M_bytes)

        end = time.time()
        log.info(f"        Total time for reading={end - start:.3f}s")

        return G, N, M
