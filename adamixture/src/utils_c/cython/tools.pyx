# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.math cimport log, sqrtf, sqrt
from libc.stdlib cimport free, malloc
from libc.stdint cimport uint8_t, uint32_t, uintptr_t, int32_t, uint64_t

import numpy as np


# Row-wise genotype reconstruction and log-likelihood (fused)
cdef inline double _row_loglike(const uint8_t* g_row, const double* p, const double* Q, Py_ssize_t N, Py_ssize_t K) noexcept nogil:
    cdef:
        Py_ssize_t i, k
        uint8_t g_val
        double row_ll = 0.0, r
        const double *q

    for i in range(N):
        g_val = g_row[i]
        if g_val == 3:
            continue
        q = &Q[i * K]
        r = 0.0
        for k in range(K):
            r += p[k] * q[k]
        if r < 1e-15:
            r = 1e-15
        elif r > 1.0 - 1e-15:
            r = 1.0 - 1e-15

        if g_val == 0:
            row_ll += 2.0 * log(1.0 - r)
        elif g_val == 1:
            row_ll += log(r * (1.0 - r))
        else:
            row_ll += 2.0 * log(r)
    return row_ll

cdef inline double _row_loglike_packed(const uint8_t* packed_row, uint8_t shift, const double* p, const double* Q, Py_ssize_t N, Py_ssize_t K) noexcept nogil:
    cdef:
        Py_ssize_t i, k
        uint8_t g_val
        double row_ll = 0.0, r
        const double *q

    for i in range(N):
        g_val = (packed_row[i] >> shift) & 0x03
        if g_val == 3:
            continue
        q = &Q[i * K]
        r = 0.0
        for k in range(K):
            r += p[k] * q[k]
        if r < 1e-15:
            r = 1e-15
        elif r > 1.0 - 1e-15:
            r = 1.0 - 1e-15

        if g_val == 0:
            row_ll += 2.0 * log(1.0 - r)
        elif g_val == 1:
            row_ll += log(r * (1.0 - r))
        else:
            row_ll += 2.0 * log(r)
    return row_ll

# Calculate total log-likelihood
cpdef double loglikelihood(uint8_t[:,::1] G, double[:,::1] P, double[:,::1] Q) noexcept nogil:
    """
    Description:
    Calculates the total log-likelihood for the current model parameters (P, Q) given genotypes G.

    Args:
        G (uint8_t[:,::1]): Genotype matrix (SNPs x samples).
        P (double[:,::1]): Allele frequency matrix (SNPs x clusters).
        Q (double[:,::1]): Individual ancestry matrix (samples x clusters).

    Returns:
        double: The total log-likelihood.
    """
    cdef:
        Py_ssize_t M = G.shape[0]
        Py_ssize_t N = G.shape[1]
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t j
        double ll = 0.0
        const double* Q_ptr = &Q[0, 0]

    with nogil, parallel():
        for j in prange(M, schedule='guided'):
            ll += _row_loglike(&G[j, 0], &P[j, 0], Q_ptr, N, K)
    return ll


cpdef double loglikelihood_packed(uintptr_t G_ptr, size_t M, size_t N, double[:,::1] P, double[:,::1] Q) noexcept nogil:
    """
    Description:
    Calculates log-likelihood from a 2-bit packed genotype matrix without
    materialising the unpacked M x N matrix. Four SNP rows are stored per byte
    for each sample.
    """
    cdef:
        uint8_t* G = <uint8_t*>G_ptr
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t j
        size_t byte_idx
        uint8_t shift
        uint8_t* packed_row
        double ll = 0.0
        const double* Q_ptr = &Q[0, 0]

    with nogil, parallel():
        for j in prange(M, schedule='guided'):
            byte_idx = j >> 2
            shift = <uint8_t>((j & 3) << 1)
            packed_row = &G[byte_idx * N]
            ll += _row_loglike_packed(packed_row, shift, &P[j, 0], Q_ptr, N, K)
    return ll

cdef inline float _snp_af(const uint8_t* r, size_t N) noexcept nogil:
    cdef:
        size_t y
        uint32_t sum_int = 0, valid_cnt = 0
        uint8_t g, is_valid
    for y in range(N):
        g = r[y]
        is_valid = (g != 3)
        sum_int += g if is_valid else 0
        valid_cnt += is_valid
    return <float>sum_int / <float>(2 * valid_cnt) if valid_cnt > 0 else 0.0

# Calculate SNP allele frequencies
cpdef void alleleFrequency(uint8_t[:, ::1] G, float[::1] f, int M, int N) noexcept nogil:
    """
    Description:
    Calculates the allele frequencies for each SNP in the genotype matrix G.

    Args:
        G (uint8_t[:, ::1]): Unpacked genotype matrix (SNPs x samples).
        f (float[::1]): Output vector for frequencies (SNPs).
        M (int): Number of SNPs.
        N (int): Number of individuals.

    Returns:
        None
    """
    cdef size_t x
    for x in prange(M, schedule='guided'):
        f[x] = _snp_af(&G[x, 0], N)

# Calculate RMSE between matrices
cpdef double rmse_d(double[:,::1] Q1, double[:,::1] Q2, int N, int K) noexcept nogil:
    """
    Description:
    Calculates the Root Mean Square Error (RMSE) between two double matrices (Q1 and Q2).

    Args:
        Q1 (double[:,::1]): First matrix (samples x clusters).
        Q2 (double[:,::1]): Second matrix (samples x clusters).
        N (int): Number of samples.
        K (int): Number of clusters.

    Returns:
        double: The computed RMSE value.
    """
    cdef:
        size_t T = N * K
        size_t i
        double inv_T = 1.0 / <double>T
        double acc = 0.0
        double diff
        double* q1 = &Q1[0, 0]
        double* q2 = &Q2[0, 0]
    for i in range(T):
        diff = q1[i] - q2[i]
        acc += diff * diff
    return sqrt(acc * inv_T)

# Normalize vector to sum to 1
cdef inline void _norm_d(double* q, size_t K) noexcept nogil:
    """
    Description:
    Normalizes a row vector to sum to 1, with values clipped to ensure stability.

    Args:
        q (double*): The vector to normalize.
        K (size_t): Length of the vector (number of clusters).

    Returns:
        None
    """
    cdef:
        size_t k
        double t = 0.0, v = 0.0
        double raw, clipped
    for k in range(K):
        raw = q[k]
        if raw < 1e-5:
            clipped = 1e-5
        elif raw > 1.0-1e-5:
            clipped = 1.0-1e-5
        else:
            clipped = raw
        t += clipped
        q[k] = clipped
    v = 1.0 / t
    for k in range(K):
        q[k] *= v

# Normalize all rows of Q
cpdef void mapQ_d(double[:,::1] Q, int N, int K) noexcept nogil:
    """
    Description:
    Normalizes every row in the ancestry matrix Q so they each sum to 1.

    Args:
        Q (double[:,::1]): Ancestry matrix (samples x clusters).
        N (int): Number of samples.
        K (int): Number of populations.

    Returns:
        None
    """
    cdef size_t j
    with nogil, parallel():
        for j in prange(N, schedule='guided'):
            _norm_d(&Q[j,0], K)

# Clip all entries of P to domain
cpdef void mapP_d(double[:,::1] P, int M, int K) noexcept nogil:
    """
    Description:
    Clips all elements of the allele frequency matrix P to the valid domain [1e-5, 1-1e-5].

    Args:
        P (double[:,::1]): Allele frequency matrix (SNPs x clusters).
        M (int): Number of SNPs.
        K (int): Number of populations.

    Returns:
        None
    """
    cdef:
        size_t i, k
        double raw
        double* p
    with nogil, parallel():
        for i in prange(M, schedule='guided'):
            p = &P[i,0]
            for k in range(K):
                raw = p[k]
                if raw < 1e-5:
                    p[k] = 1e-5
                elif raw > 1.0 - 1e-5:
                    p[k] = 1.0 - 1e-5


# Jensen-Shannon based divergence
cpdef double KL(double[:, ::1] Q1, double[:, ::1] Q2, int N, int K) noexcept nogil:
    """
    Description:
    Calculates the generalized Kullback-Leibler (KL) divergence between two ancestry matrices.
    Specifically, it calculates a symmetric divergence measure based on Jensen-Shannon.

    Args:
        Q1 (double[:, ::1]): First ancestry matrix (samples x clusters).
        Q2 (double[:, ::1]): Second ancestry matrix (samples x clusters).
        N (int): Number of samples.
        K (int): Number of clusters.

    Returns:
        double: The average divergence per sample.
    """
    cdef:
        size_t i, k
        double eps = 1e-10
        double acc = 0.0
        double ai, bi, m
        double* pa
        double* pb
    for i in prange(N, schedule='guided'):
        pa = &Q1[i, 0]
        pb = &Q2[i, 0]
        for k in range(K):
            ai = pa[k]
            bi = pb[k]
            m = 0.5 * (ai + bi)
            acc += ai * log((ai) / m + eps)
    return acc / <double>N

# Decompress and center chunk for SVD
cpdef void decompress_block(const uint8_t[:, ::1] G, float[:, ::1] data_block, const float[::1] freq, const Py_ssize_t offset) noexcept nogil:
    """
    Description:
    Decompresses a block of genotypes from uint8 to float32 and centers them by subtracting 2*freq.
    Used for randomized SVD computations.

    Args:
        G (const uint8_t[:, ::1]): Source genotype matrix (SNPs x samples).
        data_block (float[:, ::1]): Output buffer for decompressed/centered block (chunk_size x samples).
        freq (const float[::1]): Allele frequencies for the SNPs in the chunk.
        offset (const Py_ssize_t): Starting SNP index for this block in the full matrix G.

    Returns:
        None
    """
    cdef:
        Py_ssize_t nr = data_block.shape[0]
        Py_ssize_t nc = data_block.shape[1]
        Py_ssize_t i, j, l
        const uint8_t *gr
        float val, u
        float *ptr
    for j in prange(nr, schedule='static'):
        l = offset + j
        u = 2.0 * freq[l]
        gr = &G[l, 0]
        ptr = &data_block[j, 0]
        for i in range(nc):
            if gr[i] != 3: # missing
                val = <float>gr[i] - u
                ptr[i] = val
            else:
                ptr[i] = 0.0
