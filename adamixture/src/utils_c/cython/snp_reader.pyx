# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.stdlib cimport calloc, free, malloc, realloc, atoi
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uintptr_t, int8_t, int16_t, int32_t, int64_t, uint64_t
from libc.string cimport memcpy, memset

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
        Py_ssize_t full_bytes = n_samples // 4
        Py_ssize_t rem = n_samples % 4
        Py_ssize_t snp_idx, byte_pos, sample_pos, r_idx
        unsigned char current_byte
        unsigned char lookup_table[4]

    lookup_table[0] = 2
    lookup_table[1] = 3
    lookup_table[2] = 1
    lookup_table[3] = 0

    with nogil, parallel():
        for snp_idx in prange(n_snps):
            for byte_pos in range(full_bytes):
                current_byte = bed_source[snp_idx, byte_pos]
                sample_pos = byte_pos * 4
                geno_target[snp_idx, sample_pos]     = lookup_table[current_byte & 3]
                geno_target[snp_idx, sample_pos + 1] = lookup_table[(current_byte >> 2) & 3]
                geno_target[snp_idx, sample_pos + 2] = lookup_table[(current_byte >> 4) & 3]
                geno_target[snp_idx, sample_pos + 3] = lookup_table[(current_byte >> 6) & 3]

            if rem > 0:
                current_byte = bed_source[snp_idx, full_bytes]
                sample_pos = full_bytes * 4
                for r_idx in range(rem):
                    geno_target[snp_idx, sample_pos + r_idx] = lookup_table[(current_byte >> (2 * r_idx)) & 3]

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
        Py_ssize_t i, j, k, n_valid
        uint8_t val
        unsigned char[4] lookup_table = [2, 3, 1, 0]
        const uint8_t* in_row
        uint8_t* out_row

    # One packed row (4 SNPs) per iteration: reads 4 BED rows and writes one
    # output row, both sequentially.
    with nogil, parallel():
        for j in prange(M_bytes, schedule='static'):
            out_row = G_packed + j * N
            n_valid = M - j * 4
            if n_valid > 4:
                n_valid = 4
            for i in range(N):
                out_row[i] = 0
            for k in range(n_valid):
                in_row = B_bed + (j * 4 + k) * N_bytes
                for i in range(N):
                    val = lookup_table[(in_row[i >> 2] >> ((i & 3) << 1)) & 3]
                    out_row[i] |= val << (k << 1)

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
        Py_ssize_t i, j, k, n_valid
        const uint8_t* in_row
        uint8_t* out_row

    with nogil, parallel():
        for j in prange(M_bytes, schedule='static'):
            out_row = G_packed + j * N
            n_valid = M - j * 4
            if n_valid > 4:
                n_valid = 4
            for i in range(N):
                out_row[i] = 0
            for k in range(n_valid):
                in_row = G + (j * 4 + k) * N
                for i in range(N):
                    out_row[i] |= (in_row[i] & 0x03) << (k << 1)

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
        size_t i, j, k, n_valid
        uint64_t total_sum = 0
        uint64_t total_count = 0
        uint64_t row_sum, row_count
        uint8_t v
        const uint8_t* row

    with nogil, parallel():
        for j in prange(M_bytes, schedule='static'):
            row = G + j * N
            n_valid = M - j * 4 if M - j * 4 < 4 else 4
            row_sum = 0
            row_count = 0
            for k in range(n_valid):
                for i in range(N):
                    v = (row[i] >> (k << 1)) & 0x03
                    row_sum = row_sum + (v if v != 3 else 0)
                    row_count = row_count + (2 if v != 3 else 0)
            total_sum += row_sum
            total_count += row_count
    
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
        for j in prange(M_bytes, schedule='static'):
            if j == M_bytes - 1:
                for i in range(N):
                    G[j * N + i] = flip_tab[G[j * N + i]] & last_mask
            else:
                for i in range(N):
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

cdef object _grow_rows(object G, Py_ssize_t needed_rows, Py_ssize_t n_cols, Py_ssize_t initial_rows):
    """
    Description:
    Ensures G has at least needed_rows rows. Growth goes through ndarray.resize
    (realloc), which avoids holding two copies of the matrix.
    """
    cdef Py_ssize_t new_rows
    if G is None:
        new_rows = needed_rows if needed_rows > initial_rows else initial_rows
        return np.empty((new_rows, n_cols), dtype=np.uint8)
    if needed_rows > G.shape[0]:
        new_rows = G.shape[0] + G.shape[0] // 2
        if new_rows < needed_rows:
            new_rows = needed_rows
        G.resize((new_rows, n_cols), refcheck=False)
    return G

def _log_vcf_skipped(Py_ssize_t skipped_variants, bint has_specific, specific_chrom, str chrom_mode, int autosomes):
    if skipped_variants <= 0:
        return
    import logging
    if has_specific:
        logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside specific chromosomes {specific_chrom}.")
    elif chrom_mode == "autosomes":
        logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs outside autosomes 1..{autosomes}.")
    else:
        logging.getLogger(__name__).warning(f"        Warning: Skipped {skipped_variants} SNPs excluded by chromosome filter.")

def _read_vcf_single_pass(str filepath, int chunk_size, bint packed, bint keep_all, int autosomes,
                          bint has_specific, set allowed_nums, set allowed_strs):
    """
    Description:
    Reads and parses a VCF in a single pass over the (possibly compressed) file,
    growing the output matrix as chunks are parsed.

    Returns:
        tuple (G, N, M, skipped): G is uint8 (M x N), or packed (ceil(M/4) x N) if packed.
    """
    cdef:
        Py_ssize_t n_samples = 0
        Py_ssize_t start_var_idx = 0
        Py_ssize_t skipped_variants = 0
        Py_ssize_t n_chunk
        list chunk_bytes = []
        object G = None

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
            if n_samples <= 0:
                raise ValueError("Invalid or empty VCF file")

            chunk_bytes.append(line)
            if len(chunk_bytes) == chunk_size:
                n_chunk = len(chunk_bytes)
                if packed:
                    G = _grow_rows(G, (start_var_idx + n_chunk + 3) // 4, n_samples, chunk_size)
                    _process_chunk_packed(chunk_bytes, G, start_var_idx, n_samples)
                else:
                    G = _grow_rows(G, start_var_idx + n_chunk, n_samples, 4 * chunk_size)
                    _process_chunk_standard(chunk_bytes, G, start_var_idx, n_samples)
                start_var_idx += n_chunk
                chunk_bytes = []

        if chunk_bytes:
            n_chunk = len(chunk_bytes)
            if packed:
                G = _grow_rows(G, (start_var_idx + n_chunk + 3) // 4, n_samples, chunk_size)
                _process_chunk_packed(chunk_bytes, G, start_var_idx, n_samples)
            else:
                G = _grow_rows(G, start_var_idx + n_chunk, n_samples, 4 * chunk_size)
                _process_chunk_standard(chunk_bytes, G, start_var_idx, n_samples)
            start_var_idx += n_chunk
    finally:
        fh.close()

    if n_samples <= 0 or start_var_idx <= 0:
        raise ValueError("Invalid or empty VCF file")

    if packed:
        G.resize(((start_var_idx + 3) // 4, n_samples), refcheck=False)
    else:
        G.resize((start_var_idx, n_samples), refcheck=False)
    return G, n_samples, start_var_idx, skipped_variants

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

    G, n_samples, n_variants, skipped_variants = _read_vcf_single_pass(
        filepath, chunk_size, False, keep_all, autosomes, has_specific, allowed_nums, allowed_strs
    )
    _log_vcf_skipped(skipped_variants, has_specific, specific_chrom, chrom_mode, autosomes)
    return G, n_samples, n_variants

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

    if chunk_size % 4 != 0:
        chunk_size += 4 - (chunk_size % 4)

    G_packed, n_samples, n_variants, skipped_variants = _read_vcf_single_pass(
        filepath, chunk_size, True, keep_all, autosomes, has_specific, allowed_nums, allowed_strs
    )
    _log_vcf_skipped(skipped_variants, has_specific, specific_chrom, chrom_mode, autosomes)
    return G_packed, n_samples, n_variants


# Read BCF to uint8 matrix
cdef inline uint32_t _bcf_u32(const uint8_t* data, Py_ssize_t pos) noexcept nogil:
    return (<uint32_t>data[pos] | (<uint32_t>data[pos + 1] << 8)
            | (<uint32_t>data[pos + 2] << 16) | (<uint32_t>data[pos + 3] << 24))

cdef Py_ssize_t _bcf_parse_typed_int(const uint8_t* data, Py_ssize_t pos, int* out) noexcept nogil:
    cdef uint8_t b = data[pos]
    cdef uint8_t t = b & 0x0f
    cdef int l = (b >> 4) & 0x0f
    cdef int val = 0
    pos += 1
    if l == 15:
        pos = _bcf_parse_typed_int(data, pos, &l)
    if t == 1:
        val = <int8_t>data[pos]
        pos += 1
    elif t == 2:
        val = <int16_t>(data[pos] | (data[pos+1] << 8))
        pos += 2
    elif t == 3:
        val = <int32_t>(data[pos] | (data[pos+1] << 8) | (data[pos+2] << 16) | (data[pos+3] << 24))
        pos += 4
    out[0] = val
    return pos

cdef Py_ssize_t _bcf_parse_descriptor(const uint8_t* data, Py_ssize_t pos, int* n_out, int* sz_out) noexcept nogil:
    cdef uint8_t b = data[pos]
    cdef uint8_t t = b & 0x0f
    cdef int l = (b >> 4) & 0x0f
    pos += 1
    if l == 15:
        pos = _bcf_parse_typed_int(data, pos, &l)
    n_out[0] = l
    sz_out[0] = 1 if (t == 1 or t == 7) else (2 if t == 2 else 4)
    return pos

cdef void _bcf_decode_variant(const uint8_t* c_data, Py_ssize_t data_len, Py_ssize_t rec_offset,
                              int target_gt_idx, Py_ssize_t n_samples, uint8_t* out) noexcept nogil:
    """
    Description:
    Decodes the GT field of one BCF record into allele dosages (3 = missing).
    """
    cdef:
        Py_ssize_t l_shared = _bcf_u32(c_data, rec_offset)
        Py_ssize_t base = rec_offset + 8
        uint32_t n_fmt = _bcf_u32(c_data, base + 20) >> 24
        Py_ssize_t fmt_pos = base + l_shared
        Py_ssize_t values_pos, gt_offset = -1, s
        uint32_t f
        int fmt_idx, n_v, t_sz
        int n_vals = 2, type_sz = 1
        int a1, a2
        const int8_t* p8
        const int16_t* p16
        const int32_t* p32

    for f in range(n_fmt):
        fmt_pos = _bcf_parse_typed_int(c_data, fmt_pos, &fmt_idx)
        values_pos = _bcf_parse_descriptor(c_data, fmt_pos, &n_v, &t_sz)
        if fmt_idx == target_gt_idx or target_gt_idx < 0:  # GT field
            gt_offset = values_pos
            n_vals = n_v
            type_sz = t_sz
            break
        fmt_pos = values_pos + n_samples * n_v * t_sz

    if gt_offset < 0 or gt_offset + n_samples * n_vals * type_sz > data_len:
        for s in range(n_samples):
            out[s] = 3
        return

    if type_sz == 1:
        p8 = <const int8_t*>(c_data + gt_offset)
        for s in range(n_samples):
            a1 = (p8[s * n_vals] >> 1) - 1
            if n_vals >= 2:
                a2 = (p8[s * n_vals + 1] >> 1) - 1
                out[s] = 3 if (a1 < 0 or a2 < 0) else <uint8_t>(a1 + a2)
            else:
                out[s] = 3 if a1 < 0 else <uint8_t>a1
    elif type_sz == 2:
        p16 = <const int16_t*>(c_data + gt_offset)
        for s in range(n_samples):
            a1 = (p16[s * n_vals] >> 1) - 1
            if n_vals >= 2:
                a2 = (p16[s * n_vals + 1] >> 1) - 1
                out[s] = 3 if (a1 < 0 or a2 < 0) else <uint8_t>(a1 + a2)
            else:
                out[s] = 3 if a1 < 0 else <uint8_t>a1
    else:
        p32 = <const int32_t*>(c_data + gt_offset)
        for s in range(n_samples):
            a1 = (p32[s * n_vals] >> 1) - 1
            if n_vals >= 2:
                a2 = (p32[s * n_vals + 1] >> 1) - 1
                out[s] = 3 if (a1 < 0 or a2 < 0) else <uint8_t>(a1 + a2)
            else:
                out[s] = 3 if a1 < 0 else <uint8_t>a1

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

    # Pass 1: scan records and filter chromosomes (decision cached per contig)
    cdef Py_ssize_t offset = 9 + l_text
    cdef Py_ssize_t l_shared, l_indiv, base
    cdef int32_t chrom_id
    cdef list kept_offsets = []
    cdef dict keep_by_contig = {}
    cdef Py_ssize_t skipped_variants = 0

    while offset + 8 <= data_len:
        l_shared = _bcf_u32(c_data, offset)
        l_indiv = _bcf_u32(c_data, offset + 4)
        base = offset + 8

        if base + 4 > data_len:
            break

        chrom_id = <int32_t>_bcf_u32(c_data, base)
        keep = keep_by_contig.get(chrom_id)
        if keep is None:
            chrom_name = contigs.get(chrom_id, str(chrom_id))
            keep = _keep_chromosome_line(chrom_name.encode("utf-8"), keep_all, autosomes,
                                         has_specific, allowed_nums, allowed_strs)
            keep_by_contig[chrom_id] = keep
        if keep:
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
    cdef Py_ssize_t[::1] offsets = np.asarray(kept_offsets, dtype=np.intp)
    cdef Py_ssize_t var_idx

    # Pass 2: decode genotypes for kept variants
    with nogil, parallel():
        for var_idx in prange(n_variants, schedule='static'):
            _bcf_decode_variant(c_data, data_len, offsets[var_idx], target_gt_idx, n_samples, &G[var_idx, 0])

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


# ── PLINK 2 PGEN ──────────────────────────────────────────────────────────────
# Only the main data track (biallelic hard-call categories 0/1/2, 3 = missing)
# is decoded; auxiliary tracks (multiallelic patches, phase, dosages) are skipped
# using the record lengths from the header. Multiallelic variants therefore give
# the number of non-REF alleles.

cdef enum:
    PGEN_VBLOCK = 65536
    PGEN_DECODE_CHUNK = 256

cdef inline Py_ssize_t _pgen_varint(const uint8_t* data, Py_ssize_t pos, Py_ssize_t end, uint32_t* out) noexcept nogil:
    cdef uint32_t val = 0
    cdef int shift = 0
    cdef uint8_t b
    while pos < end:
        b = data[pos]
        pos += 1
        val |= (<uint32_t>(b & 0x7f)) << shift
        if b < 128:
            out[0] = val
            return pos
        shift += 7
        if shift > 28:
            return -1
    return -1

cdef inline uint32_t _pgen_load_id(const uint8_t* p, int nbytes) noexcept nogil:
    cdef uint32_t v = p[0]
    if nbytes > 1:
        v |= (<uint32_t>p[1]) << 8
    if nbytes > 2:
        v |= (<uint32_t>p[2]) << 16
    if nbytes > 3:
        v |= (<uint32_t>p[3]) << 24
    return v

cdef Py_ssize_t _pgen_apply_difflist(const uint8_t* data, Py_ssize_t pos, Py_ssize_t end, uint8_t* row,
                                     uint32_t n_samples, int id_bytes) noexcept nogil:
    """
    Description:
    Parses a (sample ID, 2-bit category) difflist and writes its entries into row.
    Returns the position after the difflist, or -1 if it is malformed.
    """
    cdef:
        uint32_t L, n_groups, sid = 0, delta
        Py_ssize_t ids_pos, rare_pos, k
    pos = _pgen_varint(data, pos, end, &L)
    if pos < 0:
        return -1
    if L == 0:
        return pos
    if L > n_samples:
        return -1
    n_groups = (L + 63) // 64
    ids_pos = pos
    pos += <Py_ssize_t>n_groups * (id_bytes + 1) - 1
    rare_pos = pos
    pos += (L + 3) // 4
    if pos > end:
        return -1
    for k in range(L):
        if (k & 63) == 0:
            sid = _pgen_load_id(data + ids_pos + (k >> 6) * id_bytes, id_bytes)
        else:
            pos = _pgen_varint(data, pos, end, &delta)
            if pos < 0:
                return -1
            sid += delta
        if sid >= n_samples:
            return -1
        row[sid] = (data[rare_pos + (k >> 2)] >> ((k & 3) * 2)) & 3
    return pos

cdef int _pgen_decode_record(const uint8_t* data, Py_ssize_t start, Py_ssize_t end, uint8_t vrtype,
                             const uint8_t* ld_base, uint8_t* row, uint32_t n_samples, int id_bytes) noexcept nogil:
    """
    Description:
    Decodes the main data track of one PGEN record into row (one byte per sample).
    ld_base is the decoded row of the most recent non-LD-compressed variant.
    Returns 0 on success, 1 if the record is malformed.
    """
    cdef:
        int t = vrtype & 7
        Py_ssize_t i, nbytes, pos
        uint8_t b, c_lo, c_delta
        const uint8_t* bits
    if t == 0:
        nbytes = (n_samples + 3) // 4
        if start + nbytes > end:
            return 1
        for i in range(n_samples // 4):
            b = data[start + i]
            row[4 * i] = b & 3
            row[4 * i + 1] = (b >> 2) & 3
            row[4 * i + 2] = (b >> 4) & 3
            row[4 * i + 3] = b >> 6
        for i in range((n_samples // 4) * 4, n_samples):
            row[i] = (data[start + (i >> 2)] >> ((i & 3) * 2)) & 3
        return 0
    if t == 1:
        nbytes = (n_samples + 7) // 8
        if start + 1 + nbytes > end:
            return 1
        c_lo = data[start] >> 2
        c_delta = data[start] & 3
        bits = data + start + 1
        for i in range(n_samples):
            row[i] = c_lo + c_delta * ((bits[i >> 3] >> (i & 7)) & 1)
        pos = start + 1 + nbytes
    elif t == 2 or t == 3:
        if ld_base == NULL:
            return 1
        memcpy(row, ld_base, n_samples)
        pos = start
    elif t == 5:
        memset(row, 0, n_samples)
        return 0
    else:
        memset(row, t & 3, n_samples)
        pos = start
    if _pgen_apply_difflist(data, pos, end, row, n_samples, id_bytes) < 0:
        return 1
    if t == 3:
        for i in range(n_samples):
            b = row[i]
            row[i] = b ^ (((b & 1) ^ 1) << 1)
    return 0

cdef int _pgen_id_bytes(uint32_t n_samples) noexcept nogil:
    if n_samples <= 256:
        return 1
    if n_samples <= 65536:
        return 2
    if n_samples <= 16777216:
        return 3
    return 4

cdef void _pgen_decode_range(const uint8_t* data, const uint8_t* vrtypes, const int64_t* fstart, const int64_t* fend,
                             Py_ssize_t v0, Py_ssize_t v1, uint8_t* out, uint32_t n_samples, int* n_errors) noexcept nogil:
    """
    Description:
    Decodes variants [v0, v1) into out (row-major, n_samples bytes per variant).
    Work is split into chunks; a chunk that starts with an LD-compressed record
    decodes its LD base privately, so chunks are independent.
    """
    cdef:
        int id_bytes = _pgen_id_bytes(n_samples)
        Py_ssize_t n_chunks = (v1 - v0 + PGEN_DECODE_CHUNK - 1) // PGEN_DECODE_CHUNK
        Py_ssize_t c, s, e, v, j
        uint8_t* base_buf
        const uint8_t* ld_base
        uint8_t* row
        int err = 0

    with parallel():
        base_buf = <uint8_t*>malloc(n_samples)
        for c in prange(n_chunks, schedule='dynamic'):
            s = v0 + c * PGEN_DECODE_CHUNK
            e = s + PGEN_DECODE_CHUNK
            if e > v1:
                e = v1
            ld_base = NULL
            if (vrtypes[s] & 6) == 2:
                j = s - 1
                while j >= 0 and (vrtypes[j] & 6) == 2:
                    j = j - 1
                if j >= 0 and base_buf != NULL:
                    if _pgen_decode_record(data, fstart[j], fend[j], vrtypes[j], NULL, base_buf, n_samples, id_bytes) == 0:
                        ld_base = base_buf
            for v in range(s, e):
                row = out + (v - v0) * <Py_ssize_t>n_samples
                if _pgen_decode_record(data, fstart[v], fend[v], vrtypes[v], ld_base, row, n_samples, id_bytes) != 0:
                    err += 1
                elif (vrtypes[v] & 6) != 2:
                    ld_base = row
        free(base_buf)
    n_errors[0] = err

def _pgen_index(const uint8_t[::1] data, str pgi_path):
    """
    Description:
    Parses a PGEN header (or its external .pgi index) into per-variant record
    types and byte ranges within the .pgen file.

    Returns:
        tuple (vrtypes, fstart, fend, N, M).
    """
    cdef Py_ssize_t file_size = data.shape[0]
    if file_size < 3 or data[0] != 0x6c or data[1] != 0x1b:
        raise ValueError("Not a PGEN file (bad magic number).")
    cdef int mode = data[2]
    if mode == 0x01:
        raise ValueError("PGEN storage mode 0x01 (PLINK 1 .bed inside .pgen) is not supported; read the .bed fileset instead.")

    if mode in (0x20, 0x21):
        if pgi_path is None or not __import__("os").path.exists(pgi_path):
            raise FileNotFoundError(f"PGEN index file not found: {pgi_path}")
        hdr = np.fromfile(pgi_path, dtype=np.uint8)
        if hdr.shape[0] < 12 or hdr[0] != 0x6c or hdr[1] != 0x1b or (hdr[2] & 0xfe) != 0x30:
            raise ValueError(f"{pgi_path} is not a PGEN index (.pgi) file.")
    else:
        hdr = np.asarray(data)
    if hdr.shape[0] < 12:
        raise ValueError("PGEN header is truncated.")

    M = int(hdr[3:7].view("<u4")[0])
    N = int(hdr[7:11].view("<u4")[0])
    ctrl = int(hdr[11])
    nonref_storage = ctrl >> 6
    if N == 0:
        raise ValueError("PGEN file contains no samples.")

    if 0x02 <= mode <= 0x04:
        if ctrl & 63:
            raise ValueError("Malformed fixed-width PGEN header.")
        offset = 12 + ((M + 7) // 8 if nonref_storage == 3 else 0)
        geno_bytes = (N + 3) // 4
        width = geno_bytes + (2 * N if mode == 0x03 else (4 * N if mode == 0x04 else 0))
        if offset + M * width != file_size:
            raise ValueError("Unexpected PGEN file size for a fixed-width storage mode.")
        fstart = offset + np.arange(M, dtype=np.int64) * width
        return np.zeros(M, dtype=np.uint8), fstart, fstart + geno_bytes, N, M

    if mode not in (0x10, 0x11, 0x20, 0x21):
        raise ValueError(f"Unsupported PGEN storage mode 0x{mode:02x}.")
    low = ctrl & 15
    if low >= 8:
        raise ValueError("Unsupported PGEN header layout (special single-sample modes).")
    alt_bytes = (ctrl >> 4) & 3
    len_bytes = (low & 3) + 1
    n_blocks = (M + PGEN_VBLOCK - 1) // PGEN_VBLOCK
    pos = 12
    header_size = 12 + 8 * n_blocks
    for b in range(n_blocks):
        cnt = min(PGEN_VBLOCK, M - b * PGEN_VBLOCK)
        header_size += ((cnt + 1) // 2 if low < 4 else cnt) + cnt * (len_bytes + alt_bytes)
        if nonref_storage == 3:
            header_size += (cnt + 7) // 8
    if header_size > hdr.shape[0]:
        raise ValueError("PGEN header is truncated.")
    block_offsets = hdr[pos:pos + 8 * n_blocks].view("<u8").astype(np.int64)
    pos += 8 * n_blocks

    vrtypes = np.empty(M, dtype=np.uint8)
    fstart = np.empty(M, dtype=np.int64)
    fend = np.empty(M, dtype=np.int64)
    weights = (256 ** np.arange(len_bytes, dtype=np.int64))
    for b in range(n_blocks):
        v0 = b * PGEN_VBLOCK
        cnt = min(PGEN_VBLOCK, M - v0)
        if low < 4:
            nb = (cnt + 1) // 2
            packed = hdr[pos:pos + nb]
            nyb = np.empty(2 * nb, dtype=np.uint8)
            nyb[0::2] = packed & 15
            nyb[1::2] = packed >> 4
            vrtypes[v0:v0 + cnt] = nyb[:cnt]
        else:
            nb = cnt
            vrtypes[v0:v0 + cnt] = hdr[pos:pos + nb]
        pos += nb
        lens = hdr[pos:pos + cnt * len_bytes].reshape(cnt, len_bytes).astype(np.int64) @ weights
        pos += cnt * len_bytes
        pos += cnt * alt_bytes
        if nonref_storage == 3:
            pos += (cnt + 7) // 8
        ends = block_offsets[b] + np.cumsum(lens)
        fend[v0:v0 + cnt] = ends
        fstart[v0:v0 + cnt] = ends - lens

    if M > 0 and (fstart.min() < 0 or fend.max() > file_size):
        raise ValueError("PGEN variant records extend beyond the end of the file.")
    if M > 0 and (vrtypes[np.arange(0, M, PGEN_VBLOCK)] & 6 == 2).any():
        raise ValueError("Malformed PGEN: a variant block starts with an LD-compressed record.")
    return vrtypes, fstart, fend, N, M

def _pgen_pgi_path(str filepath, str orig_filepath):
    path = orig_filepath if orig_filepath else filepath
    for ext in (".zst", ".gz"):
        if path.endswith(ext):
            path = path[:-len(ext)]
    return path + ".pgi"

def _pgen_decode_into(const uint8_t[::1] data, const uint8_t[::1] vrtypes, const int64_t[::1] fstart,
                      const int64_t[::1] fend, Py_ssize_t v0, Py_ssize_t v1, uint8_t[:, ::1] out, uint32_t n_samples):
    cdef int n_errors = 0
    if v1 <= v0:
        return
    with nogil:
        _pgen_decode_range(&data[0], &vrtypes[0], &fstart[0], &fend[0], v0, v1, &out[0, 0], n_samples, &n_errors)
    if n_errors:
        raise ValueError(f"Malformed PGEN file: {n_errors} variant record(s) could not be decoded.")

# Read PGEN to uint8 matrix
def read_pgen_file(str filepath, int chunk_size, str chromosome_mode, int autosome_count, str orig_filepath=None):
    """
    Description:
    Reads a PLINK 2 PGEN file into a uint8 NumPy matrix (variants x samples)
    with values 0/1/2 (ALT allele count) and 3 for missing.

    Args:
        filepath (str): Path to an uncompressed .pgen file.
        chunk_size (int): Unused; kept for API compatibility.
        chromosome_mode (str): Unused; filtering is applied by the caller.
        autosome_count (int): Unused; filtering is applied by the caller.
        orig_filepath (str): Original (possibly compressed) path, used to locate a .pgen.pgi index.

    Returns:
        tuple (G, N, M).
    """
    data = np.fromfile(filepath, dtype=np.uint8)
    vrtypes, fstart, fend, N, M = _pgen_index(data, _pgen_pgi_path(filepath, orig_filepath))
    G = np.empty((M, N), dtype=np.uint8)
    _pgen_decode_into(data, vrtypes, fstart, fend, 0, M, G, N)
    return G, N, M

# Read PGEN to 2-bit packed matrix
def read_pgen_file_packed(str filepath, int chunk_size, str chromosome_mode, int autosome_count, str orig_filepath=None):
    """
    Description:
    Reads a PLINK 2 PGEN file into a 2-bit packed matrix (ceil(M/4) x N),
    decoding chunk_size variants at a time to bound memory use.

    Returns:
        tuple (G_packed, N, M).
    """
    data = np.fromfile(filepath, dtype=np.uint8)
    vrtypes, fstart, fend, N, M = _pgen_index(data, _pgen_pgi_path(filepath, orig_filepath))
    M_bytes = (M + 3) // 4
    G_packed = np.zeros((M_bytes, N), dtype=np.uint8)
    step = max(4, (chunk_size // 4) * 4)
    buf = np.empty((min(step, max(M, 1)), N), dtype=np.uint8)
    for start in range(0, M, step):
        stop = min(start + step, M)
        _pgen_decode_into(data, vrtypes, fstart, fend, start, stop, buf, N)
        pack_genotypes(buf.ctypes.data, G_packed[start // 4:].ctypes.data, stop - start, N, (stop - start + 3) // 4)
    return G_packed, N, M
