# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.math cimport log, log1p, sqrtf, sqrt
from libc.stdlib cimport calloc, free, malloc
from libc.stdint cimport uint8_t, uint32_t, uintptr_t, int32_t, uint64_t

import numpy as np

# Row-wise genotype reconstruction
cdef inline void _reconstruct(double* Q, double* p, double* rec, Py_ssize_t N, Py_ssize_t K) noexcept nogil:
    """
    Description:
    Computes the reconstruction vector for all samples for a single SNP (p dot Q_i).

    Args:
        Q (double*): Full Q matrix (samples x clusters).
        p (double*): Single SNP row from P matrix (clusters).
        rec (double*): Output buffer for reconstruction values (samples).
        N (Py_ssize_t): Number of samples.
        K (Py_ssize_t): Number of ancestral populations.

    Returns:
        None
    """
    cdef:
        size_t i, k
        double* q
    for i in range(N):
        q = &Q[i*K]
        for k in range(K):
            rec[i] += p[k]*q[k]

# Calculate SNP log-likelihood
cdef inline double _ind_loglike(uint8_t* g, double* rec, Py_ssize_t N) noexcept nogil:
    """
    Description:
    Calculates the log-likelihood for a SNP across all individuals.

    Args:
        g (uint8_t*): Genotype vector for the SNP.
        rec (double*): Reconstructed values for the SNP.
        N (Py_ssize_t): Number of samples.

    Returns:
        double: Accumulated log-likelihood for this SNP.
    """
    cdef:
        size_t i
        double ll = 0.0
        double g_d
    for i in range(N):
        g_d = <double>g[i]
        ll += g_d*log(rec[i]) + (2.0-g_d)*log1p(-rec[i]) if g[i] != 3 else 0.0
        rec[i] = 0.0
    return ll


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
        size_t i, j
        double ll = 0.0
        double* rec
    with nogil, parallel():
        rec = <double*>calloc(N, sizeof(double))
        for j in prange(M, schedule='guided'):
            _reconstruct(&Q[0,0], &P[j,0], rec, N, K)
            ll += _ind_loglike(&G[j,0], rec, N)
        free(rec)
    return ll

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
    cdef:
        size_t x, y
        uint8_t* r
        float sum_val, denom
    for x in prange(M, schedule='guided'):
        sum_val = 0.0
        denom = 0.0
        r = &G[x,0]
        for y in range(N):
            if r[y] != 3:
                sum_val = sum_val + <float>r[y]
                denom = denom + 2.0
        f[x] = sum_val / denom

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
