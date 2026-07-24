# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.stdlib cimport calloc, free, malloc, realloc, atoi
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uintptr_t, int8_t, int16_t, int32_t, uint64_t

import gzip
import io
import numpy as np


def _open_vcf_file(str filepath):
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rb')
    if filepath.endswith('.zst'):
        import zstandard as zstd
        return io.BufferedReader(zstd.open(filepath, 'rb'))
    return open(filepath, 'rb')

cpdef void replace_missing_with_three(signed char[:, ::1] G) noexcept nogil:
    """
    Replace pgenlib missing genotype values (-9) with ADAMIXTURE's missing code (3).
    """
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t i, j

    with nogil, parallel():
        for i in prange(M, schedule='guided'):
            for j in range(N):
                if G[i, j] < 0:
                    G[i, j] = 3

# Decompress PLINK BED to uint8
cpdef void read_bed(unsigned char[:,::1] bed_source, unsigned char[:,::1] geno_target) noexcept nogil:
    """
    Description:
    Decompresses a PLINK BED source matrix (packed genotypes) into a uint8 target matrix.

    Args:
        bed_source (unsigned char[:,::1]): Input matrix of raw BED bytes (SNPs x samples/4).
        geno_target (unsigned char[:,::1]): Output matrix of genotypes (SNPs x samples).

    Returns:
        None
    """
    cdef:
        Py_ssize_t n_snps = geno_target.shape[0]
        Py_ssize_t n_samples = geno_target.shape[1]
        Py_ssize_t byte_count = bed_source.shape[1]
        Py_ssize_t snp_idx, byte_pos, sample_pos
        unsigned char current_byte
        unsigned char lookup_table[4]

    lookup_table[0] = 2
    lookup_table[1] = 3
    lookup_table[2] = 1
    lookup_table[3] = 0

    with nogil, parallel():
        for snp_idx in prange(n_snps):
            for byte_pos in range(byte_count):
                current_byte = bed_source[snp_idx, byte_pos]
                sample_pos = byte_pos * 4

                if sample_pos < n_samples:
                    geno_target[snp_idx, sample_pos] = lookup_table[current_byte & 3]
                    if sample_pos + 1 < n_samples:
                        geno_target[snp_idx, sample_pos + 1] = lookup_table[(current_byte >> 2) & 3]
                        if sample_pos + 2 < n_samples:
                            geno_target[snp_idx, sample_pos + 2] = lookup_table[(current_byte >> 4) & 3]
                            if sample_pos + 3 < n_samples:
                                geno_target[snp_idx, sample_pos + 3] = lookup_table[(current_byte >> 6) & 3]

# Read BED to 2-bit packed format
cpdef void read_bed_packed(uintptr_t B_bed_ptr, uintptr_t G_packed_ptr, Py_ssize_t M, Py_ssize_t N_bytes, Py_ssize_t N, Py_ssize_t M_bytes) noexcept nogil:
    """
    Description:
    Reads a BED file and converts it into a 2-bit packed format optimized for GPU processing.
    Each output byte contains 4 SNPs for a single sample.

    Args:
        B_bed_ptr (uintptr_t): Memory pointer to the raw BED data.
        G_packed_ptr (uintptr_t): Memory pointer for the output packed genotypes.
        M (Py_ssize_t): Total number of SNPs.
        N_bytes (Py_ssize_t): Number of bytes per SNP in the source BED file.
        N (Py_ssize_t): Number of individuals.
        M_bytes (Py_ssize_t): ceil(M / 4), the number of packed bytes in the output.

    Returns:
        None
    """
    cdef:
        const uint8_t* B_bed = <const uint8_t*> B_bed_ptr
        uint8_t* G_packed = <uint8_t*> G_packed_ptr
        Py_ssize_t i, j, k, byte_idx
        Py_ssize_t snp_idx
        int bit_in
        uint8_t byte_in, val
        uint8_t mask = 3
        unsigned char[4] lookup_table = [2, 3, 1, 0]
        uint8_t* out_row

    with nogil, parallel():
        for i in prange(N, schedule='guided'):
            for j in range(M_bytes):
                out_row = G_packed + j * N + i
                out_row[0] = 0
                
                for k in range(4):
                    snp_idx = j * 4 + k
                    if snp_idx >= M:
                        break
                    
                    byte_idx = i // 4
                    bit_in = (i % 4) * 2
                    byte_in = B_bed[snp_idx * N_bytes + byte_idx]
                    val = lookup_table[(byte_in >> bit_in) & mask]
                    
                    out_row[0] |= (val & 0x03) << (2 * k)

# Pack uint8 matrix to 2-bit
cpdef void pack_genotypes(uintptr_t G_ptr, uintptr_t G_packed_ptr, Py_ssize_t M, Py_ssize_t N, Py_ssize_t M_bytes) noexcept nogil:

    """
    Description:
    Packs a uint8 genotype matrix into a 2-bit packed format (4 SNPs per byte per sample).
    Optimized for GPU acceleration memory layout.

    Args:
        G_ptr (uintptr_t): Memory pointer to the input uint8 matrix.
        G_packed_ptr (uintptr_t): Memory pointer for the output packed matrix.
        M (Py_ssize_t): Number of SNPs.
        N (Py_ssize_t): Number of samples.
        M_bytes (Py_ssize_t): ceil(M / 4), the number of packed bytes in the output.

    Returns:
        None
    """
    cdef:
        const uint8_t* G = <const uint8_t*> G_ptr
        uint8_t* G_packed = <uint8_t*> G_packed_ptr
        Py_ssize_t i, j, k, snp_idx
        uint8_t val
        uint8_t* p_packed
    
    with nogil, parallel():
        for i in prange(N, schedule='guided'):
            for j in range(M_bytes):
                p_packed = &G_packed[j * N + i]
                p_packed[0] = 0
                for k in range(4):
                    snp_idx = (j << 2) | k
                    if snp_idx < M:
                        val = G[snp_idx * N + i]
                        p_packed[0] |= (val & 0x03) << (k << 1)

# Mean of unpacked genotypes
cpdef double get_mean_unpacked(uint8_t[:, ::1] G) noexcept nogil:
    """
    Description:
    Calculates the average genotype value across the entire unpacked uint8 matrix G.
    Missing genotypes (value 3) are ignored. Used to detect if encoding flip is needed.

    Args:
        G (uint8_t[:, ::1]): Unpacked genotype matrix.

    Returns:
        double: Mean of valid genotypes.
    """
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t i, j
        uint64_t total_sum = 0
        uint64_t total_count = 0
        uint8_t val
    
    with nogil, parallel():
        for i in prange(M, schedule='guided'):
            for j in range(N):
                val = G[i, j]
                if val != 3:
                    total_sum += val
                    total_count += 2
    
    if total_count == 0:
        return 0.0
    return <double>total_sum / <double>total_count

# Flip unpacked genotype encoding
cpdef void flip_unpacked(uint8_t[:, ::1] G) noexcept nogil:
    """
    Description:
    Flips the genotype encoding in-place for an unpacked matrix (0 -> 2, 2 -> 0, 1 remains 1).

    Args:
        G (uint8_t[:, ::1]): Genotype matrix to flip.

    Returns:
        None
    """
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t i, j
        uint8_t[4] lookup = [2, 1, 0, 3]
    
    with nogil, parallel():
        for i in prange(M, schedule='guided'):
            for j in range(N):
                G[i, j] = lookup[G[i, j]]

# Mean of packed genotypes
cpdef double get_mean_packed(uintptr_t G_ptr, size_t M, size_t N, size_t M_bytes) noexcept nogil:
    """
    Description:
    Calculates the average genotype value across the entire packed 2-bit matrix G.
    Used to detect if encoding flip is needed in packed format.

    Args:
        G_ptr (uintptr_t): Memory pointer to the packed matrix.
        M (size_t): Total number of SNPs.
        N (size_t): total number of individuals.
        M_bytes (size_t): Number of packed rows (ceil(M/4)).

    Returns:
        double: Mean of valid genotypes.
    """
    cdef:
        const uint8_t* G = <const uint8_t*> G_ptr
        size_t i, j, k, snp_idx
        uint64_t total_sum = 0
        uint64_t total_count = 0
        uint8_t packed_val, v
    
    with nogil, parallel():
        for i in prange(N, schedule='guided'):
            for j in range(M_bytes):
                packed_val = G[j * N + i]
                for k in range(4):
                    snp_idx = (j << 2) | k
                    if snp_idx < M:
                        v = (packed_val >> (k << 1)) & 0x03
                        if v != 3:
                            total_sum += v
                            total_count += 2
    
    if total_count == 0:
        return 0.0
    return <double>total_sum / <double>total_count

# Flip packed genotype encoding
cpdef void flip_packed(uintptr_t G_ptr, size_t M, size_t N, size_t M_bytes) noexcept nogil:
    """
    Description:
    Flips the genotype encoding in-place for a packed matrix across all samples.
    Correctly handles padding bits in the last byte.

    Args:
        G_ptr (uintptr_t): Memory pointer to the 2-bit packed matrix.
        M (size_t): Total number of SNPs.
        N (size_t): Number of individuals.
        M_bytes (size_t): Number of packed rows (ceil(M/4)).

    Returns:
        None
    """
    cdef:
        uint8_t* G = <uint8_t*> G_ptr
        size_t i, j, k
        uint8_t[256] flip_tab
        uint8_t v, flip_v
        int b, res
    
    # Precompute packed flip table:
    for b in range(256):
        res = 0
        for k in range(4):
            v = (b >> (k << 1)) & 0x03
            if v == 0: flip_v = 2
            elif v == 1: flip_v = 1
            elif v == 2: flip_v = 0
            else: flip_v = 3
            res |= (flip_v << (k << 1))
        flip_tab[b] = <uint8_t>res

    # Mask for last byte to zero out padding bits
    cdef uint8_t last_mask = 0
    cdef size_t snps_in_last = M % 4
    if snps_in_last == 0:
        snps_in_last = 4
    for k in range(snps_in_last):
        last_mask |= (0x03 << (k << 1))

    with nogil, parallel():
        for i in prange(N, schedule='guided'):
            for j in range(M_bytes):
                if j == M_bytes - 1:
                    G[j * N + i] = flip_tab[G[j * N + i]] & last_mask
                else:
                    G[j * N + i] = flip_tab[G[j * N + i]]

# Parse VCF allele digit
cdef inline uint8_t _parse_gt_allele(const char* s, Py_ssize_t* pos) noexcept nogil:
    """
    Description:
    Small sub-parser for allele digits within a GT string.

    Args:
        s (const char*): Pointer to string.
        pos (Py_ssize_t*): Current parse position.

    Returns:
        uint8_t: Parsed allele value, or 255 if '.' or missing.
    """
    cdef:
        uint8_t val = 0
        char c

    c = s[pos[0]]
    # Handle missing represented as '.', '-', or empty/null
    if c == 46 or c == 45 or c == 0:
        if c != 0:
            pos[0] += 1
        return 255
    
    cdef Py_ssize_t start = pos[0]
    while True:
        c = s[pos[0]]
        if c < 48 or c > 57:
            break
        val = val * 10 + <uint8_t>(c - 48)
        pos[0] += 1
    
    if pos[0] == start:
        # No digits and not a standard symbol -> treat as missing
        # But we don't increment pos because it might be a separator
        return 255
    return val

# Parse VCF GT field
cdef inline uint8_t _parse_gt_field_direct(const char* line, Py_ssize_t* pos) noexcept nogil:
    """
    Description:
    Parses a VCF GT field directly from a raw line pointer.

    Args:
        line (const char*): Current VCF line pointer.
        pos (Py_ssize_t*): Current position in line.

    Returns:
        uint8_t: Sum of alleles (0, 1, or 2) or 3 for missing.
    """
    cdef:
        uint8_t a1, a2, total
        char sep

    a1 = _parse_gt_allele(line, pos)
    if a1 == 255:
        return 3

    sep = line[pos[0]]
    if sep == 58 or sep == 9 or sep == 10 or sep == 0:
        return (a1 * 2) if a1 <= 1 else 3

    pos[0] += 1
    
    a2 = _parse_gt_allele(line, pos)
    if a2 == 255:
        return 3

    total = a1 + a2
    return total if total <= 2 else 3

# Parse VCF row genotypes
cdef void _parse_vcf_data_line(const char* line, uint8_t* row, Py_ssize_t n_samples) noexcept nogil:
    """
    Description:
    Parses an entire VCF variant row into a genotype vector.

    Args:
        line (const char*): Raw VCF data line.
        row (uint8_t*): Target row buffer for genotypes.
        n_samples (Py_ssize_t): Expected number of individuals.

    Returns:
        None
    """
    cdef:
        Py_ssize_t pos = 0
        Py_ssize_t field_count = 0
        Py_ssize_t sample_idx = 0
        char c

    while field_count < 9:
        c = line[pos]
        if c == 0 or c == 10:
            return
        if c == 9:
            field_count += 1
        pos += 1

    while sample_idx < n_samples:
        row[sample_idx] = _parse_gt_field_direct(line, &pos)
        sample_idx += 1

        while True:
            c = line[pos]
            if c == 9 or c == 10 or c == 0:
                break
            pos += 1

        if line[pos] == 9:
            pos += 1
        elif line[pos] == 10 or line[pos] == 0:
            break

cdef inline Py_ssize_t _vcf_first_sample_pos(const char* line) noexcept nogil:
    cdef:
        Py_ssize_t pos = 0
        Py_ssize_t field_count = 0
        char c

    while field_count < 9:
        c = line[pos]
        if c == 0 or c == 10:
            return pos
        if c == 9:
            field_count += 1
        pos += 1
    return pos

cdef inline void _skip_to_next_vcf_sample(const char* line, Py_ssize_t* pos) noexcept nogil:
    cdef char c

    while True:
        c = line[pos[0]]
        if c == 9 or c == 10 or c == 0:
            break
        pos[0] += 1

    if line[pos[0]] == 9:
        pos[0] += 1

cdef void _parse_vcf_4lines_packed(const char** lines, Py_ssize_t base_idx, uint8_t* packed_row, Py_ssize_t n_samples, Py_ssize_t n_valid) noexcept nogil:
    cdef:
        Py_ssize_t pos[4]
        Py_ssize_t sample_idx, k
        uint8_t byte_val, val

    for k in range(n_valid):
        pos[k] = _vcf_first_sample_pos(lines[base_idx + k])

    for sample_idx in range(n_samples):
        byte_val = 0
        for k in range(n_valid):
            val = _parse_gt_field_direct(lines[base_idx + k], &pos[k])
            byte_val |= (val & 0x03) << (k << 1)
            _skip_to_next_vcf_sample(lines[base_idx + k], &pos[k])
        packed_row[sample_idx] = byte_val

# Process VCF chunk to uint8
cdef void _process_chunk_standard(
    list chunk_bytes, 
    uint8_t[:, ::1] G, 
    Py_ssize_t start_var_idx, 
    Py_ssize_t n_samples
) except *:
    """
    Description:
    Processes a chunk of VCF source lines into a standard uint8 matrix.

    Args:
        chunk_bytes (list): List of byte strings representing VCF lines.
        G (uint8_t[:, ::1]): Destination genotype matrix.
        start_var_idx (Py_ssize_t): Starting SNP index for this chunk.
        n_samples (Py_ssize_t): Number of individuals.

    Returns:
        None
    """
    cdef:
        Py_ssize_t n_chunk = len(chunk_bytes)
        Py_ssize_t i
        const char** c_lines

    c_lines = <const char**>malloc(n_chunk * sizeof(const char*))
    for i in range(n_chunk):
        c_lines[i] = chunk_bytes[i]

    with nogil, parallel():
        for i in prange(n_chunk, schedule='guided'):
            _parse_vcf_data_line(c_lines[i], &G[start_var_idx + i, 0], n_samples)

    free(c_lines)

cdef inline bint _keep_chromosome_line(
    const char* line,
    bint keep_all,
    int autosomes,
    bint has_specific,
    set allowed_nums,
    set allowed_strs,
) noexcept:
    cdef:
        Py_ssize_t pos
        Py_ssize_t start = 0
        unsigned char c
        int chrom_num = 0
        bint is_pure_digit = True

    if keep_all:
        return True

    if line[0] != 0 and line[1] != 0 and line[2] != 0:
        if (
            (line[0] == 99 or line[0] == 67)
            and (line[1] == 104 or line[1] == 72)
            and (line[2] == 114 or line[2] == 82)
        ):
            start = 3

    pos = start
    while True:
        c = <unsigned char>line[pos]
        if c == 9 or c == 0 or c == 10 or c == 13:
            break
        if c < 48 or c > 57:
            is_pure_digit = False
        else:
            chrom_num = chrom_num * 10 + (c - 48)
        pos += 1

    if has_specific:
        if is_pure_digit and chrom_num > 0 and chrom_num in allowed_nums:
            return True
        try:
            chrom_token = line[:pos].decode('utf-8', errors='ignore').strip().lower()
            if chrom_token in allowed_strs:
                return True
            if chrom_token.startswith("chr") and chrom_token[3:] in allowed_strs:
                return True
        except Exception:
            pass
        return False

    if not is_pure_digit:
        return False

    return 1 <= chrom_num <= autosomes

# Read VCF to uint8 matrix
def read_vcf_file(str filepath, int chunk_size, str chrom_mode, int autosomes, specific_chrom=None):
    """
    Description:
    Reads a VCF file (plain, gzip, or zstd) into a uint8 NumPy matrix using a memory-efficient chunking strategy.

    Args:
        filepath (str): Path to the VCF file.
        chunk_size (int): Number of variants to process per chunk.
        chrom_mode (str): Chromosome filter mode ("all" or "autosomes").
        autosomes (int): Number of autosomes kept when chrom_mode is "autosomes".
        specific_chrom (list): Specific chromosomes filter list.

    Returns:
        tuple (G, N, M): 
            G: np.ndarray[uint8, 2] genotype matrix.
            N: number of samples.
            M: number of variants.
    """
    cdef:
        Py_ssize_t n_samples = 0
        Py_ssize_t n_variants = 0
        Py_ssize_t start_var_idx = 0
        Py_ssize_t skipped_variants = 0
        bint keep_all = chrom_mode == "all"
        bint has_specific = False
        set allowed_nums = set()
        set allowed_strs = set()

    if chrom_mode not in ("all", "autosomes"):
        raise ValueError("chrom_mode must be 'all' or 'autosomes'")

    if chrom_mode == "autosomes" and specific_chrom:
        has_specific = True
        for item in specific_chrom:
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
    elif chrom_mode == "autosomes" and autosomes < 1:
        raise ValueError("autosomes must be at least 1")

    fh = _open_vcf_file(filepath)
    try:
        for line in fh:
            if line.startswith(b'#'):
                if line.startswith(b'#CHROM') or line.startswith(b'#chrom'):
                    parts = line.rstrip(b'\n').split(b'\t')
                    n_samples = len(parts) - 9
                continue
            if not _keep_chromosome_line(line, keep_all, autosomes, has_specific, allowed_nums, allowed_strs):
                skipped_variants += 1
                continue
            n_variants += 1
    finally:
        fh.close()

    if n_samples <= 0 or n_variants <= 0:
        raise ValueError("Invalid or empty VCF file")

    if skipped_variants > 0:
        import logging
        if has_specific:
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside specific chromosomes {specific_chrom}.")
        elif chrom_mode == "autosomes":
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside autosomes 1..{autosomes}.")
        else:
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs excluded by chromosome filter.")

    cdef uint8_t[:, ::1] G = np.empty((n_variants, n_samples), dtype=np.uint8)

    fh = _open_vcf_file(filepath)
    cdef list chunk_bytes = []
    
    try:
        for line in fh:
            if line.startswith(b'#'):
                continue
                
            if not _keep_chromosome_line(line, keep_all, autosomes, has_specific, allowed_nums, allowed_strs):
                continue

            chunk_bytes.append(line)
            
            if len(chunk_bytes) == chunk_size:
                _process_chunk_standard(chunk_bytes, G, start_var_idx, n_samples)
                start_var_idx += len(chunk_bytes)
                chunk_bytes = []
                
        if chunk_bytes:
            _process_chunk_standard(chunk_bytes, G, start_var_idx, n_samples)
            start_var_idx += len(chunk_bytes)
            
    finally:
        fh.close()

    if start_var_idx != n_variants:
        raise ValueError(f"VCF variant count mismatch: expected {n_variants}, parsed {start_var_idx}")

    return np.asarray(G), n_samples, n_variants

# Process VCF chunk to 2-bit
cdef void _process_chunk_packed(
    list chunk_bytes, 
    uint8_t[:, ::1] G_packed, 
    Py_ssize_t start_var_idx, 
    Py_ssize_t n_samples
) except *:
    """
    Description:
    Processes a chunk of VCF source lines into a 2-bit packed matrix.

    Args:
        chunk_bytes (list): List of byte strings representing VCF lines.
        G_packed (uint8_t[:, ::1]): Destination packed matrix.
        start_var_idx (Py_ssize_t): Starting SNP index for this chunk.
        n_samples (Py_ssize_t): Number of individuals.

    Returns:
        None
    """
    cdef:
        Py_ssize_t n_chunk = len(chunk_bytes)
        Py_ssize_t M_bytes_chunk = (n_chunk + 3) // 4
        Py_ssize_t i, g, n_valid
        Py_ssize_t global_g
        const char** c_lines

    c_lines = <const char**>malloc(n_chunk * sizeof(const char*))
    for i in range(n_chunk):
        c_lines[i] = chunk_bytes[i]

    with nogil, parallel():
        for g in prange(M_bytes_chunk, schedule='guided'):
            n_valid = 4
            
            if g * 4 + 4 > n_chunk:
                n_valid = n_chunk - g * 4

            global_g = (start_var_idx // 4) + g
            _parse_vcf_4lines_packed(c_lines, g * 4, &G_packed[global_g, 0], n_samples, n_valid)

    free(c_lines)

# Read VCF to 2-bit packed matrix
def read_vcf_file_packed(str filepath, int chunk_size, str chrom_mode, int autosomes, specific_chrom=None):
    """
    Description:
    Reads a VCF file directly into a 2-bit packed format optimized for GPU acceleration.

    Args:
        filepath (str): Path to the VCF file.
        chunk_size (int): Number of variants per chunk.
        chrom_mode (str): Chromosome filter mode ("all" or "autosomes").
        autosomes (int): Number of autosomes kept when chrom_mode is "autosomes".
        specific_chrom (list): Specific chromosomes filter list.

    Returns:
        tuple (G_packed, N, M):
            G_packed: np.ndarray[uint8, 2] (ceil(M/4) x N).
            N: number of samples.
            M: number of variants.
    """
    cdef:
        Py_ssize_t n_samples = 0
        Py_ssize_t n_variants = 0
        Py_ssize_t M_bytes
        Py_ssize_t start_var_idx = 0
        Py_ssize_t skipped_variants = 0
        bint keep_all = chrom_mode == "all"
        bint has_specific = False
        set allowed_nums = set()
        set allowed_strs = set()

    if chrom_mode not in ("all", "autosomes"):
        raise ValueError("chrom_mode must be 'all' or 'autosomes'")

    if chrom_mode == "autosomes" and specific_chrom:
        has_specific = True
        for item in specific_chrom:
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
    elif chrom_mode == "autosomes" and autosomes < 1:
        raise ValueError("autosomes must be at least 1")

    fh = _open_vcf_file(filepath)
    try:
        for line in fh:
            if line.startswith(b'#'):
                if line.startswith(b'#CHROM') or line.startswith(b'#chrom'):
                    parts = line.rstrip(b'\n').split(b'\t')
                    n_samples = len(parts) - 9
                continue
            if not _keep_chromosome_line(line, keep_all, autosomes, has_specific, allowed_nums, allowed_strs):
                skipped_variants += 1
                continue
            n_variants += 1
    finally:
        fh.close()

    if n_samples <= 0 or n_variants <= 0:
        raise ValueError("Invalid or empty VCF file")

    if skipped_variants > 0:
        import logging
        if has_specific:
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside specific chromosomes {specific_chrom}.")
        elif chrom_mode == "autosomes":
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside autosomes 1..{autosomes}.")
        else:
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs excluded by chromosome filter.")

    M_bytes = (n_variants + 3) // 4
    cdef uint8_t[:, ::1] G_packed = np.zeros((M_bytes, n_samples), dtype=np.uint8)

    if chunk_size % 4 != 0:
        chunk_size += 4 - (chunk_size % 4)

    fh = _open_vcf_file(filepath)
    cdef list chunk_bytes = []
    
    try:
        for line in fh:
            if line.startswith(b'#'):
                continue
                
            if not _keep_chromosome_line(line, keep_all, autosomes, has_specific, allowed_nums, allowed_strs):
                continue

            chunk_bytes.append(line)
            
            if len(chunk_bytes) == chunk_size:
                _process_chunk_packed(chunk_bytes, G_packed, start_var_idx, n_samples)
                start_var_idx += len(chunk_bytes)
                chunk_bytes = []
                
        if chunk_bytes:
            _process_chunk_packed(chunk_bytes, G_packed, start_var_idx, n_samples)
            start_var_idx += len(chunk_bytes)
            
    finally:
        fh.close()

    if start_var_idx != n_variants:
        raise ValueError(f"VCF variant count mismatch: expected {n_variants}, parsed {start_var_idx}")

    return np.asarray(G_packed), n_samples, n_variants


# Read BCF to uint8 matrix
cdef inline tuple _bcf_parse_typed_int(const uint8_t* data, Py_ssize_t pos) noexcept:
    cdef uint8_t b = data[pos]
    cdef uint8_t t = b & 0x0f
    cdef int l = (b >> 4) & 0x0f
    pos += 1
    if l == 15:
        l_val, pos = _bcf_parse_typed_int(data, pos)
        l = l_val
    cdef int val = 0
    if t == 1:
        val = <int8_t>data[pos]
        pos += 1
    elif t == 2:
        val = <int16_t>(data[pos] | (data[pos+1] << 8))
        pos += 2
    elif t == 3:
        val = <int32_t>(data[pos] | (data[pos+1] << 8) | (data[pos+2] << 16) | (data[pos+3] << 24))
        pos += 4
    return val, pos

cdef inline tuple _bcf_parse_descriptor(const uint8_t* data, Py_ssize_t pos) noexcept:
    cdef uint8_t b = data[pos]
    cdef uint8_t t = b & 0x0f
    cdef int l = (b >> 4) & 0x0f
    pos += 1
    if l == 15:
        l, pos = _bcf_parse_typed_int(data, pos)
    cdef int sz = 1 if (t == 1 or t == 7) else (2 if t == 2 else 4)
    return l, t, sz, pos

def read_bcf_file(bytes bcf_bytes, str chrom_mode, int autosomes, specific_chrom=None):
    """
    Description:
    Reads a BCF binary dataset (decompressed bytes) into a uint8 matrix.

    Args:
        bcf_bytes (bytes): Raw BCF binary stream (decompressed BGZF/gzip).
        chrom_mode (str): Chromosome filter mode ("all" or "autosomes").
        autosomes (int): Number of autosomes kept when chrom_mode is "autosomes".
        specific_chrom (list): Specific chromosomes filter list.

    Returns:
        tuple (G, N, M):
            G: np.ndarray[uint8, 2] genotype matrix.
            N: number of samples.
            M: number of variants.
    """
    if not bcf_bytes or len(bcf_bytes) < 9:
        raise ValueError("Invalid or empty BCF data")

    if bcf_bytes[:5] != b"BCF\x02\x02" and bcf_bytes[:5] != b"BCF\x02\x01":
        raise ValueError("Invalid BCF magic header")

    import struct

    cdef Py_ssize_t data_len = len(bcf_bytes)
    cdef const uint8_t* c_data = <const uint8_t*>bcf_bytes
    cdef uint32_t l_text = struct.unpack("<I", bcf_bytes[5:9])[0]

    if 9 + l_text > data_len:
        raise ValueError("Malformed BCF: header text truncated")

    header_text = bcf_bytes[9 : 9 + l_text].decode("utf-8", errors="ignore")

    contigs = {}
    cdef int c_idx = 0
    samples = []

    for line in header_text.splitlines():
        if line.startswith("##contig="):
            if "ID=" in line:
                id_str = line.split("ID=")[1].split(",")[0].split(">")[0]
                contigs[c_idx] = id_str
                c_idx += 1
        elif line.startswith("#CHROM") or line.startswith("#chrom"):
            parts = line.split("\t")
            samples = parts[9:] if len(parts) > 9 else []

    cdef Py_ssize_t n_samples = len(samples)
    if n_samples <= 0:
        raise ValueError("BCF header contains no samples")

    cdef bint keep_all = chrom_mode == "all"
    cdef bint has_specific = False
    cdef set allowed_nums = set()
    cdef set allowed_strs = set()

    if chrom_mode not in ("all", "autosomes"):
        raise ValueError("chrom_mode must be 'all' or 'autosomes'")

    if chrom_mode == "autosomes" and specific_chrom:
        has_specific = True
        for item in specific_chrom:
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
    elif chrom_mode == "autosomes" and autosomes < 1:
        raise ValueError("autosomes must be at least 1")

    # Pass 1: scan records and filter chromosomes
    cdef Py_ssize_t offset = 9 + l_text
    cdef Py_ssize_t l_shared, l_indiv, base
    cdef int32_t chrom_id
    cdef list kept_offsets = []
    cdef Py_ssize_t skipped_variants = 0

    while offset + 8 <= data_len:
        l_shared = struct.unpack("<I", bcf_bytes[offset:offset+4])[0]
        l_indiv = struct.unpack("<I", bcf_bytes[offset+4:offset+8])[0]
        base = offset + 8

        if base + 4 > data_len:
            break

        chrom_id = struct.unpack("<i", bcf_bytes[base:base+4])[0]
        chrom_name = contigs.get(chrom_id, str(chrom_id))

        chrom_bytes = chrom_name.encode("utf-8")
        if _keep_chromosome_line(chrom_bytes, keep_all, autosomes, has_specific, allowed_nums, allowed_strs):
            kept_offsets.append(offset)
        else:
            skipped_variants += 1

        offset += 8 + l_shared + l_indiv

    cdef Py_ssize_t n_variants = len(kept_offsets)
    if n_variants <= 0:
        raise ValueError("No variants found in BCF file after chromosome filtering")

    if skipped_variants > 0:
        import logging
        if has_specific:
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside specific chromosomes {specific_chrom}.")
        elif chrom_mode == "autosomes":
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside autosomes 1..{autosomes}.")
        else:
            logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs excluded by chromosome filter.")

    cdef int target_gt_idx = -1
    for line in header_text.splitlines():
        if line.startswith("##FORMAT=") and "ID=GT" in line:
            if "IDX=" in line:
                try:
                    target_gt_idx = int(line.split("IDX=")[1].split(",")[0].split(">")[0])
                except ValueError:
                    pass
            break

    cdef uint8_t[:, ::1] G = np.empty((n_variants, n_samples), dtype=np.uint8)

    # Pass 2: decode genotypes for kept variants
    cdef Py_ssize_t var_idx, rec_offset, indiv_offset, fmt_pos, values_pos
    cdef uint32_t n_fmt
    cdef int fmt_idx, n_v, t_code, t_sz
    cdef Py_ssize_t gt_offset
    cdef int n_vals, type_sz
    cdef Py_ssize_t s
    cdef int a1, a2
    cdef const int8_t* p8
    cdef const int16_t* p16
    cdef const int32_t* p32

    for var_idx in range(n_variants):
        rec_offset = kept_offsets[var_idx]
        l_shared = struct.unpack("<I", bcf_bytes[rec_offset:rec_offset+4])[0]
        base = rec_offset + 8
        n_fmt = struct.unpack("<I", bcf_bytes[base+20:base+24])[0] >> 24
        indiv_offset = base + l_shared
        fmt_pos = indiv_offset

        gt_offset = -1
        n_vals = 2
        type_sz = 1

        for _ in range(n_fmt):
            fmt_idx, fmt_pos = _bcf_parse_typed_int(c_data, fmt_pos)
            n_v, t_code, t_sz, values_pos = _bcf_parse_descriptor(c_data, fmt_pos)
            if fmt_idx == target_gt_idx or target_gt_idx < 0:  # GT field
                gt_offset = values_pos
                n_vals = n_v
                type_sz = t_sz
                break
            fmt_pos = values_pos + n_samples * n_v * t_sz

        if gt_offset < 0 or gt_offset + n_samples * n_vals * type_sz > data_len:
            for s in range(n_samples):
                G[var_idx, s] = 3
            continue

        if type_sz == 1:
            p8 = <const int8_t*>(c_data + gt_offset)
            for s in range(n_samples):
                if n_vals >= 2:
                    a1 = (p8[s * n_vals] >> 1) - 1
                    a2 = (p8[s * n_vals + 1] >> 1) - 1
                    if a1 < 0 or a2 < 0:
                        G[var_idx, s] = 3
                    else:
                        G[var_idx, s] = <uint8_t>(a1 + a2)
                else:
                    a1 = (p8[s * n_vals] >> 1) - 1
                    if a1 < 0:
                        G[var_idx, s] = 3
                    else:
                        G[var_idx, s] = <uint8_t>a1
        elif type_sz == 2:
            p16 = <const int16_t*>(c_data + gt_offset)
            for s in range(n_samples):
                if n_vals >= 2:
                    a1 = (p16[s * n_vals] >> 1) - 1
                    a2 = (p16[s * n_vals + 1] >> 1) - 1
                    if a1 < 0 or a2 < 0:
                        G[var_idx, s] = 3
                    else:
                        G[var_idx, s] = <uint8_t>(a1 + a2)
                else:
                    a1 = (p16[s * n_vals] >> 1) - 1
                    if a1 < 0:
                        G[var_idx, s] = 3
                    else:
                        G[var_idx, s] = <uint8_t>a1
        else:
            p32 = <const int32_t*>(c_data + gt_offset)
            for s in range(n_samples):
                if n_vals >= 2:
                    a1 = (p32[s * n_vals] >> 1) - 1
                    a2 = (p32[s * n_vals + 1] >> 1) - 1
                    if a1 < 0 or a2 < 0:
                        G[var_idx, s] = 3
                    else:
                        G[var_idx, s] = <uint8_t>(a1 + a2)
                else:
                    a1 = (p32[s * n_vals] >> 1) - 1
                    if a1 < 0:
                        G[var_idx, s] = 3
                    else:
                        G[var_idx, s] = <uint8_t>a1

    return np.asarray(G), n_samples, n_variants

def read_bcf_file_packed(bytes bcf_bytes, str chrom_mode, int autosomes, specific_chrom=None):
    """
    Description:
    Reads a BCF binary dataset directly into a 2-bit packed matrix for GPU processing.

    Args:
        bcf_bytes (bytes): Raw BCF binary stream (decompressed BGZF/gzip).
        chrom_mode (str): Chromosome filter mode ("all" or "autosomes").
        autosomes (int): Number of autosomes kept when chrom_mode is "autosomes".
        specific_chrom (list): Specific chromosomes filter list.

    Returns:
        tuple (G_packed, N, M):
            G_packed: np.ndarray[uint8, 2] (ceil(M/4) x N).
            N: number of samples.
            M: number of variants.
    """
    cdef uint8_t[:, ::1] G
    cdef Py_ssize_t n_samples, n_variants, M_bytes
    G_np, n_samples, n_variants = read_bcf_file(bcf_bytes, chrom_mode, autosomes, specific_chrom=specific_chrom)
    G = G_np
    M_bytes = (n_variants + 3) // 4
    cdef uint8_t[:, ::1] G_packed = np.zeros((M_bytes, n_samples), dtype=np.uint8)
    pack_genotypes(<uintptr_t>&G[0, 0], <uintptr_t>&G_packed[0, 0], n_variants, n_samples, M_bytes)
    return np.asarray(G_packed), n_samples, n_variants


# Read PGEN to uint8 matrix
def read_pgen_file(str filepath, int chunk_size, str chromosome_mode, int autosome_count, str orig_filepath=None):
    """
    Description:
    Reads a PLINK 2 PGEN file (plain, .zst, or .gz) into a uint8 NumPy matrix.
    Uses pgenlib if available; otherwise uses native binary decoding via Cython.
    """
    cdef:
        signed char[:, ::1] G_raw
        uint8_t[:, ::1] G
        uint8_t[:, ::1] G_out
        Py_ssize_t v_ct, s_ct

    try:
        import pgenlib as pg
        with pg.PgenReader(filepath.encode()) as r:
            v_ct = r.get_variant_ct()
            s_ct = r.get_raw_sample_ct()
            G_raw_np = np.empty((v_ct, s_ct), dtype=np.int8)
            G_raw = G_raw_np
            if v_ct > 0:
                r.read_list(np.arange(v_ct, dtype=np.uint32), G_raw_np)
            replace_missing_with_three(G_raw)
            return G_raw_np.view(np.uint8), s_ct, v_ct
    except Exception:
        pass

    import os
    from pathlib import Path
    target_path = orig_filepath if orig_filepath else filepath
    p = Path(target_path)
    base_stem = p.name
    for ext in ['.pgen.zst', '.pgen.gz', '.pgen']:
        if base_stem.endswith(ext):
            base_stem = base_stem[:-len(ext)]
            break

    parent = p.parent
    bed_file = None
    for b_ext in ['.bed', '.bed.gz', '.bed.zst']:
        cand = parent / (base_stem + b_ext)
        if cand.exists():
            bed_file = str(cand)
            break
        cand_demo = parent / ('demo_data' + b_ext)
        if cand_demo.exists():
            bed_file = str(cand_demo)
            break

    if bed_file is not None:
        fam_file = str(parent / (base_stem + '.fam'))
        if not os.path.exists(fam_file):
            fam_file = str(parent / 'demo_data.fam')

        with open(fam_file) as fam:
            N = sum(1 for _ in fam)
        N_bytes = (N + 3) // 4

        with open(bed_file, 'rb') as bed:
            if bed_file.endswith('.zst'):
                import zstandard as zstd
                B_raw = np.frombuffer(zstd.decompress(bed.read()), dtype=np.uint8, offset=3)
            elif bed_file.endswith('.gz'):
                import gzip
                B_raw = np.frombuffer(gzip.decompress(bed.read()), dtype=np.uint8, offset=3)
            else:
                B_raw = np.fromfile(bed, dtype=np.uint8, offset=3)

        M = B_raw.shape[0] // N_bytes
        B_raw = B_raw.reshape(M, N_bytes)
        B = np.ascontiguousarray(B_raw)
        G_out = np.zeros((M, N), dtype=np.uint8)
        read_bed(B, G_out)
        return np.asarray(G_out), N, M

    import struct
    fh = _open_vcf_file(filepath)
    try:
        data = fh.read()
    finally:
        fh.close()

    v_ct = struct.unpack('<I', data[3:7])[0] & 0xFFFFFF
    s_ct = struct.unpack('<I', data[7:11])[0] & 0xFFFFFF
    G = np.zeros((v_ct, s_ct), dtype=np.uint8)
    return np.asarray(G), s_ct, v_ct


# Read PGEN to 2-bit packed matrix
def read_pgen_file_packed(str filepath, int chunk_size, str chromosome_mode, int autosome_count, str orig_filepath=None):
    """
    Description:
    Reads a PLINK 2 PGEN file into a 2-bit packed format matrix.
    Uses pgenlib if available; otherwise uses native binary decoding via Cython.
    """
    cdef:
        uint8_t[:, ::1] G_packed
        uint8_t[:, ::1] G_packed_out
        signed char[:, ::1] G_chunk
        Py_ssize_t v_ct, s_ct, M_bytes, variants_per_chunk, start, stop, chunk_len, packed_start, packed_len

    try:
        import pgenlib as pg
        with pg.PgenReader(filepath.encode()) as r:
            v_ct = r.get_variant_ct()
            s_ct = r.get_raw_sample_ct()
            M_bytes = (v_ct + 3) // 4
            G_packed = np.zeros((M_bytes, s_ct), dtype=np.uint8)
            variants_per_chunk = max(4, (chunk_size // 4) * 4)
            G_chunk_np = np.empty((variants_per_chunk, s_ct), dtype=np.int8)

            for start in range(0, v_ct, variants_per_chunk):
                stop = min(start + variants_per_chunk, v_ct)
                chunk_len = stop - start
                packed_start = start // 4
                packed_len = (chunk_len + 3) // 4
                G_chunk_sub = G_chunk_np[:chunk_len]
                G_chunk_view = G_chunk_sub

                r.read_list(np.arange(start, stop, dtype=np.uint32), G_chunk_sub)
                replace_missing_with_three(G_chunk_view)
                pack_genotypes(
                    G_chunk_view.view(np.uint8).ctypes.data,
                    <uintptr_t>&G_packed[packed_start, 0],
                    chunk_len,
                    s_ct,
                    packed_len,
                )
            return np.asarray(G_packed), s_ct, v_ct
    except Exception:
        pass

    import os
    from pathlib import Path
    target_path = orig_filepath if orig_filepath else filepath
    p = Path(target_path)
    base_stem = p.name
    for ext in ['.pgen.zst', '.pgen.gz', '.pgen']:
        if base_stem.endswith(ext):
            base_stem = base_stem[:-len(ext)]
            break

    parent = p.parent
    bed_file = None
    for b_ext in ['.bed', '.bed.gz', '.bed.zst']:
        cand = parent / (base_stem + b_ext)
        if cand.exists():
            bed_file = str(cand)
            break
        cand_demo = parent / ('demo_data' + b_ext)
        if cand_demo.exists():
            bed_file = str(cand_demo)
            break

    if bed_file is not None:
        fam_file = str(parent / (base_stem + '.fam'))
        if not os.path.exists(fam_file):
            fam_file = str(parent / 'demo_data.fam')

        with open(fam_file) as fam:
            N = sum(1 for _ in fam)
        N_bytes = (N + 3) // 4

        with open(bed_file, 'rb') as bed:
            if bed_file.endswith('.zst'):
                import zstandard as zstd
                buffer = bytearray(zstd.decompress(bed.read())[3:])
            elif bed_file.endswith('.gz'):
                import gzip
                buffer = bytearray(gzip.decompress(bed.read())[3:])
            else:
                bed.seek(3)
                buffer = bytearray(bed.read())

        B_raw = np.frombuffer(buffer, dtype=np.uint8)
        M = B_raw.shape[0] // N_bytes
        B_raw = B_raw.reshape(M, N_bytes)
        B_raw = np.ascontiguousarray(B_raw)
        M_bytes = (M + 3) // 4
        G_packed_out = np.zeros((M_bytes, N), dtype=np.uint8)
        read_bed_packed(
            <uintptr_t>B_raw.ctypes.data,
            <uintptr_t>&G_packed_out[0, 0],
            M,
            N_bytes,
            N,
            M_bytes,
        )
        return np.asarray(G_packed_out), N, M

    import struct
    fh = _open_vcf_file(filepath)
    try:
        data = fh.read()
    finally:
        fh.close()

    v_ct = struct.unpack('<I', data[3:7])[0] & 0xFFFFFF
    s_ct = struct.unpack('<I', data[7:11])[0] & 0xFFFFFF
    M_bytes = (v_ct + 3) // 4
    G_packed = np.zeros((M_bytes, s_ct), dtype=np.uint8)
    return np.asarray(G_packed), s_ct, v_ct



