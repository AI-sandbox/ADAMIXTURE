import logging
import numpy as np
import torch
import sys
import time

from .utils_c import tools
from math import ceil
from pathlib import Path

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

class SNPReader:
    """
    Wrapper to read genotype data from several formats.
    """

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

        file_path = Path(file)
        fam_file = file_path.with_suffix(".fam")
        bed_file = file_path.with_suffix(".bed")
        
        with open(fam_file, "r") as fam:
            N = sum(1 for _ in fam)
        N_bytes = ceil(N / 4)
        
        if not packed:
            with open(bed_file, "rb") as bed:
                B = np.fromfile(bed, dtype=np.uint8, offset=3)
            
            assert (B.shape[0] % N_bytes) == 0, "bim file doesn't match!"
            M = B.shape[0] // N_bytes
            B.shape = (M, N_bytes)
            
            G = np.zeros((M, N), dtype=np.uint8)
            tools.read_bed(B, G)
            del B
            return G, N, M
        else:
            log.info("        Reading BED in packed 2-bit format for GPU use.")
            file_size = Path(bed_file).stat().st_size
            buffer = bytearray(file_size - 3)
            with open(bed_file, "rb") as f:
                f.seek(3)
                f.readinto(buffer)
            
            B = torch.frombuffer(buffer, dtype=torch.uint8)
            assert (B.shape[0] % N_bytes) == 0, "bim file doesn't match!"
            M = B.shape[0] // N_bytes
            
            M_bytes = (M + 3) // 4
            G = torch.zeros((M_bytes, N), dtype=torch.uint8)
            
            tools.read_bed_packed(B.data_ptr(), G.data_ptr(), M, N_bytes, N, M_bytes)
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
            G, N, M = tools.read_vcf_file(file, chunk_size=chunk_size)
            return np.ascontiguousarray(G), N, M
        else:
            log.info("        Reading VCF in packed 2-bit format for GPU use.")
            G_packed_np, N, M = tools.read_vcf_file_packed(file, chunk_size=chunk_size)
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
            
        M, N = G.shape

        if packed:
            log.info("        Reading PGEN in packed 2-bit format for GPU use.")
            M_bytes = (M + 3) // 4
            G_packed = torch.zeros((M_bytes, N), dtype=torch.uint8)
            tools.pack_genotypes(G.ctypes.data, G_packed.data_ptr(), M, N, M_bytes)
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
        file_path = Path(file)
        base_path = file_path
        for _ in range(len(file_path.suffixes)):
            base_path = base_path.with_suffix('')
        
        if match_any:
            if not any(base_path.with_suffix(ext).exists() for ext in extensions):
                log.error(f"    Error: Could not find any of these files: {extensions} for {base_path}")
                sys.exit(1)
        else:
            missing = [str(base_path.with_suffix(ext)) for ext in extensions if not base_path.with_suffix(ext).exists()]
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
            mean_val = tools.get_mean_unpacked(G)
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (unpacked).")
                tools.flip_unpacked(G)
        else:
            M_bytes = G.shape[0]
            mean_val = tools.get_mean_packed(G.data_ptr(), M, N, M_bytes)
            if mean_val >= 0.5:
                log.info("    Flipping genotype encoding (packed).")
                tools.flip_packed(G.data_ptr(), M, N, M_bytes)
        
        end = time.time()
        log.info(f"        Total time for reading={end - start:.3f}s")

        return G, N, M