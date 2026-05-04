# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.stdlib cimport calloc, free, malloc, realloc, atoi
from libc.stdint cimport uint8_t, uint32_t, uintptr_t, int32_t, uint64_t

import gzip
import numpy as np

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
        size_t n_snps = geno_target.shape[0]
        size_t n_samples = geno_target.shape[1]
        size_t byte_count = bed_source.shape[1]
        size_t snp_idx, byte_pos, byte_offset, sample_pos
        unsigned char current_byte, geno_value
        unsigned char[4] lookup_table = [2, 3, 1, 0]
    
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
            # Each sample i
            for j in range(M_bytes):
                # Each packed byte j in the output (contains SNPs 4j...4j+3)
                out_row = G_packed + j * N + i
                out_row[0] = 0
                
                for k in range(4):
                    snp_idx = j * 4 + k
                    if snp_idx >= M:
                        break
                    
                    # Byte in BED file for snp_idx that contains sample i
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

# Pack 4 rows to 1 byte
cdef void _pack_4rows_into_byte(
    const uint8_t* buf,
    uint8_t* packed_row,
    Py_ssize_t N,
    Py_ssize_t n_valid
) noexcept nogil:
    """
    Description:
    Packs up to 4 raw rows (uint8) into a single byte-per-individual row (2-bit).

    Args:
        buf (const uint8_t*): Buffer with raw row data.
        packed_row (uint8_t*): Target row for packed genotypes.
        N (Py_ssize_t): Number of individuals.
        n_valid (Py_ssize_t): Number of valid rows in buf to pack.

    Returns:
        None
    """
    cdef:
        Py_ssize_t i, k
        uint8_t byte_val, val
    for i in range(N):
        byte_val = 0
        for k in range(n_valid):
            val = buf[k * N + i] & 0x03
            byte_val |= val << (k << 1)
        packed_row[i] = byte_val

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

# Read VCF to uint8 matrix
def read_vcf_file(str filepath, int chunk_size):
    """
    Description:
    Reads a VCF file (plain or gzip) into a uint8 NumPy matrix using a memory-efficient chunking strategy.

    Args:
        filepath (str): Path to the VCF file.
        chunk_size (int): Number of variants to process per chunk.

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

    is_gz = filepath.endswith('.gz')
    
    fh = gzip.open(filepath, 'rb') if is_gz else open(filepath, 'rb')
    try:
        for line in fh:
            if line.startswith(b'#'):
                if line.startswith(b'#CHROM') or line.startswith(b'#chrom'):
                    parts = line.rstrip(b'\n').split(b'\t')
                    n_samples = len(parts) - 9
                continue
            n_variants += 1
    finally:
        fh.close()

    if n_samples <= 0 or n_variants <= 0:
        raise ValueError("Invalid or empty VCF file")

    cdef uint8_t[:, ::1] G = np.empty((n_variants, n_samples), dtype=np.uint8)

    fh = gzip.open(filepath, 'rb') if is_gz else open(filepath, 'rb')
    chunk_bytes = []
    
    try:
        for line in fh:
            if line.startswith(b'#'):
                continue
                
            chunk_bytes.append(line)
            
            if len(chunk_bytes) == chunk_size:
                _process_chunk_standard(chunk_bytes, G, start_var_idx, n_samples)
                start_var_idx += chunk_size
                chunk_bytes = []
                
        if chunk_bytes:
            _process_chunk_standard(chunk_bytes, G, start_var_idx, n_samples)
            
    finally:
        fh.close()

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
        Py_ssize_t i, g, k, var_idx, n_valid
        Py_ssize_t global_g
        const char** c_lines
        uint8_t* local_buf

    c_lines = <const char**>malloc(n_chunk * sizeof(const char*))
    for i in range(n_chunk):
        c_lines[i] = chunk_bytes[i]

    with nogil, parallel():
        for g in prange(M_bytes_chunk, schedule='guided'):
            local_buf = <uint8_t*>malloc(4 * n_samples * sizeof(uint8_t))
            n_valid = 4
            
            if g * 4 + 4 > n_chunk:
                n_valid = n_chunk - g * 4
                
            for k in range(n_valid):
                var_idx = g * 4 + k
                _parse_vcf_data_line(c_lines[var_idx], local_buf + k * n_samples, n_samples)
                
            global_g = (start_var_idx // 4) + g
            _pack_4rows_into_byte(local_buf, &G_packed[global_g, 0], n_samples, n_valid)
            
            free(local_buf)

    free(c_lines)

# Read VCF to 2-bit packed matrix
def read_vcf_file_packed(str filepath, int chunk_size):
    """
    Description:
    Reads a VCF file directly into a 2-bit packed format optimized for GPU acceleration.

    Args:
        filepath (str): Path to the VCF file.
        chunk_size (int): Number of variants per chunk.

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

    is_gz = filepath.endswith('.gz')
    
    fh = gzip.open(filepath, 'rb') if is_gz else open(filepath, 'rb')
    try:
        for line in fh:
            if line.startswith(b'#'):
                if line.startswith(b'#CHROM') or line.startswith(b'#chrom'):
                    parts = line.rstrip(b'\n').split(b'\t')
                    n_samples = len(parts) - 9
                continue
            n_variants += 1
    finally:
        fh.close()

    if n_samples <= 0 or n_variants <= 0:
        raise ValueError("Invalid or empty VCF file")

    M_bytes = (n_variants + 3) // 4
    cdef uint8_t[:, ::1] G_packed = np.zeros((M_bytes, n_samples), dtype=np.uint8)

    if chunk_size % 4 != 0:
        chunk_size += 4 - (chunk_size % 4)

    fh = gzip.open(filepath, 'rb') if is_gz else open(filepath, 'rb')
    chunk_bytes = []
    
    try:
        for line in fh:
            if line.startswith(b'#'):
                continue
                
            chunk_bytes.append(line)
            
            if len(chunk_bytes) == chunk_size:
                _process_chunk_packed(chunk_bytes, G_packed, start_var_idx, n_samples)
                start_var_idx += chunk_size
                chunk_bytes = []
                
        if chunk_bytes:
            _process_chunk_packed(chunk_bytes, G_packed, start_var_idx, n_samples)
            
    finally:
        fh.close()

    return np.asarray(G_packed), n_samples, n_variants
