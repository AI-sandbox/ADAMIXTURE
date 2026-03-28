# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.math cimport log, log1p, sqrtf, sqrt
from libc.stdlib cimport calloc, free, malloc, realloc, atoi
from libc.stdint cimport uint8_t, uint32_t, uintptr_t, int32_t, uint64_t
from libc.stdio cimport FILE, fopen, fclose, fgets, feof
from libc.string cimport memchr, strlen, strncmp

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

# Solve linear system for variable subset
cdef bint solve_subset_ptr(const double* A, const double* b, double* x, int* F, int K, double* aug, int* map_idx) noexcept nogil:
    """
    Description:
    Solves a linear system Ax = b for a subset of variables using Gaussian elimination with pivoting.

    Args:
        A (const double*): Full covariance matrix (K x K).
        b (const double*): Full target vector (K).
        x (double*): Output vector for solution (K).
        F (int*): Mask indicator of Free (1) vs Fixed (0) variables.
        K (int): Total number of variables.
        aug (double*): Buffer for augmented matrix (K * (K+1)).
        map_idx (int*): Buffer for mapping indices of free variables.

    Returns:
        bint: True if system was successfully solved, False if matrix was singular.
    """
    cdef int n = 0, i, j, k, max_row
    cdef double max_val, temp, pivot

    for i in range(K):
        if F[i] == 1:
            map_idx[n] = i
            n += 1

    if n == 0:
        return True

    for i in range(n):
        for j in range(n):
            aug[i * (n + 1) + j] = A[map_idx[i] * K + map_idx[j]]
        aug[i * (n + 1) + n] = b[map_idx[i]]

    for i in range(n):
        max_row = i
        max_val = aug[i * (n + 1) + i]
        if max_val < 0: max_val = -max_val
        for k in range(i+1, n):
            temp = aug[k * (n + 1) + i]
            if temp < 0: temp = -temp
            if temp > max_val:
                max_val = temp
                max_row = k
        
        if max_val < 1e-12:
            return False 

        if max_row != i:
            for j in range(i, n+1):
                temp = aug[i * (n + 1) + j]
                aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j]
                aug[max_row * (n + 1) + j] = temp

        pivot = aug[i * (n + 1) + i]
        for j in range(i, n+1):
            aug[i * (n + 1) + j] /= pivot

        for k in range(n):
            if k != i:
                temp = aug[k * (n + 1) + i]
                for j in range(i, n+1):
                    aug[k * (n + 1) + j] -= temp * aug[i * (n + 1) + j]

    for i in range(n):
        x[map_idx[i]] = aug[i * (n + 1) + n]
    return True

# Exact NNLS using Block Principal Pivoting
cdef void _exact_nnls_bpp_ptr(const double* A, const double* b, double* x, int K, double* aug, int* map_idx, int* F, double* y) noexcept nogil:
    """
    Description:
    Computes Exact Non-Negative Least Squares (NNLS) solution using Block Principal Pivoting.

    Args:
        A (const double*): Positive-definite matrix A (K x K).
        b (const double*): Target vector b (K).
        x (double*): Current and output solution vector x (K).
        K (int): Number of variables.
        aug (double*): Temporary buffer for augmented matrix.
        map_idx (int*): Temporary buffer for index mapping.
        F (int*): Temporary buffer for active set (Free/Fixed).
        y (double*): Temporary buffer for gradient/dual variable vector.

    Returns:
        None
    """
    cdef int i, j, iter_count
    cdef bint changed
    
    for i in range(K):
        F[i] = 1
        x[i] = 0.0
        
    for iter_count in range(50):
        for i in range(K):
            if F[i] == 0:
                x[i] = 0.0
                
        if not solve_subset_ptr(A, b, x, F, K, aug, map_idx):
            break

        for i in range(K):
            if F[i] == 0:
                y[i] = -b[i]
                for j in range(K):
                    if F[j] == 1:
                        y[i] += A[i * K + j] * x[j]
            else:
                y[i] = 0.0

        changed = False
        for i in range(K):
            if F[i] == 1 and x[i] < -1e-8:
                F[i] = 0
                changed = True
            elif F[i] == 0 and y[i] < -1e-8:
                F[i] = 1
                changed = True

        if not changed:
            break

# Exact BVLS using Block Principal Pivoting
cdef void _exact_bvls_bpp_ptr(const double* A, const double* b, double* x, int K, double* aug, int* map_idx, int* F, int* U, double* y, double* b_subset, double lower, double upper) noexcept nogil:
    """
    Description:
    Computes Exact Bounded Variable Least Squares (BVLS) solution using Block Principal Pivoting.

    Args:
        A (const double*): Positive-definite matrix A (K x K).
        b (const double*): Target vector b (K).
        x (double*): Current and output solution vector x (K).
        K (int): Number of variables.
        aug (double*): Temporary buffer for augmented matrix.
        map_idx (int*): Temporary buffer for index mapping.
        F (int*): Temporary buffer for active set indicator (Free/Fixed).
        U (int*): Temporary buffer for indicator if fixed at Upper bound (1) or Lower bound (0).
        y (double*): Temporary buffer for gradient vector.
        b_subset (double*): Temporary buffer for adjusted target vector.
        lower (double): Lower bound for variables.
        upper (double): Upper bound for variables.

    Returns:
        None
    """
    cdef int i, j, iter_count
    cdef bint changed
    
    # Initialize: everything Free (F=1), nothing at Upper (U=0)
    for i in range(K):
        F[i] = 1
        U[i] = 0
        x[i] = 0.0
        
    for iter_count in range(50):
        # Adjust b for subset solve: b_F_prime = b_F - A_FU * upper - A_FL * lower
        for i in range(K):
            if F[i] == 1:
                b_subset[i] = b[i]
                for j in range(K):
                    if F[j] == 0:
                        if U[j] == 1:
                            b_subset[i] -= A[i * K + j] * upper
                        else:
                            b_subset[i] -= A[i * K + j] * lower
        
        # solve A_FF * x_F = b_subset_F
        if not solve_subset_ptr(A, b_subset, x, F, K, aug, map_idx):
            break
            
        # Set x for non-free variables
        for i in range(K):
            if F[i] == 0:
                if U[i] == 1:
                    x[i] = upper
                else:
                    x[i] = lower
                    
        # Calculate gradient: y = Ax - b
        for i in range(K):
            y[i] = -b[i]
            for j in range(K):
                y[i] += A[i * K + j] * x[j]

        changed = False
        for i in range(K):
            if F[i] == 1:
                if x[i] < lower - 1e-8:
                    F[i] = 0
                    U[i] = 0
                    changed = True
                elif x[i] > upper + 1e-8:
                    F[i] = 0
                    U[i] = 1
                    changed = True
            else: # Fixed
                if U[i] == 0: # At Lower bound
                    if y[i] < -1e-8:
                        F[i] = 1
                        U[i] = 0
                        changed = True
                else: # At Upper bound
                    if y[i] > 1e-8:
                        F[i] = 1
                        U[i] = 0
                        changed = True

        if not changed:
            break

# Batch NNLS solver
cpdef void batch_nnls_bpp(double[:, ::1] A_cov, double[:, ::1] B_target, double[:, ::1] Out_mat) noexcept nogil:
    """
    Description:
    Solves NNLS for a batch of target vectors against a single covariance matrix.
    Parallelized across targets.

    Args:
        A_cov (double[:, ::1]): Single covariance matrix (K x K).
        B_target (double[:, ::1]): Batch of target vectors (M x K).
        Out_mat (double[:, ::1]): Matrix to store solutions (M x K).

    Returns:
        None
    """
    cdef int M = B_target.shape[0]
    cdef int K = A_cov.shape[0]
    cdef int i
    cdef const double* A_ptr = &A_cov[0, 0]
    
    cdef double* aug
    cdef int* map_idx
    cdef int* F
    cdef double* y
    
    with nogil, parallel():
        for i in prange(M, schedule='static'):
            aug = <double*>malloc(K * (K + 1) * sizeof(double))
            map_idx = <int*>malloc(K * sizeof(int))
            F = <int*>malloc(K * sizeof(int))
            y = <double*>malloc(K * sizeof(double))
            
            _exact_nnls_bpp_ptr(A_ptr, &B_target[i, 0], &Out_mat[i, 0], K, aug, map_idx, F, y)
            
            free(aug)
            free(map_idx)
            free(F)
            free(y)

# Batch BVLS solver
cpdef void batch_bvls_bpp(double[:, ::1] A_cov, double[:, ::1] B_target, double[:, ::1] Out_mat, double lower, double upper) noexcept nogil:
    """
    Description:
    Solves BVLS for a batch of target vectors against a single covariance matrix.
    Parallelized across targets.

    Args:
        A_cov (double[:, ::1]): Single covariance matrix (K x K).
        B_target (double[:, ::1]): Batch of target vectors (M x K).
        Out_mat (double[:, ::1]): Matrix to store solutions (M x K).
        lower (double): Lower bound for all variables.
        upper (double): Upper bound for all variables.

    Returns:
        None
    """
    cdef int M = B_target.shape[0]
    cdef int K = A_cov.shape[0]
    cdef int i
    cdef const double* A_ptr = &A_cov[0, 0]
    
    cdef double* aug
    cdef int* map_idx
    cdef int* F
    cdef int* U
    cdef double* y
    cdef double* b_subset
    
    with nogil, parallel():
        for i in prange(M, schedule='static'):
            aug = <double*>malloc(K * (K + 1) * sizeof(double))
            map_idx = <int*>malloc(K * sizeof(int))
            F = <int*>malloc(K * sizeof(int))
            U = <int*>malloc(K * sizeof(int))
            y = <double*>malloc(K * sizeof(double))
            b_subset = <double*>malloc(K * sizeof(double))
            
            _exact_bvls_bpp_ptr(A_ptr, &B_target[i, 0], &Out_mat[i, 0], K, aug, map_idx, F, U, y, b_subset, lower, upper)
            
            free(aug)
            free(map_idx)
            free(F)
            free(U)
            free(y)
            free(b_subset)

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
cpdef void flip_packed(uintptr_t G_ptr, size_t M_bytes, size_t N) noexcept nogil:
    """
    Description:
    Flips the genotype encoding in-place for a packed matrix across all samples.

    Args:
        G_ptr (uintptr_t): Memory pointer to the 2-bit packed matrix.
        M_bytes (size_t): Number of packed rows (ceil(M/4)).
        N (size_t): Number of individuals.

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

    with nogil, parallel():
        for i in prange(N, schedule='guided'):
            for j in range(M_bytes):
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
    if c == 46 or c == 0:
        pos[0] += 1
        return 255
    while True:
        c = s[pos[0]]
        if c < 48 or c > 57:
            break
        val = val * 10 + <uint8_t>(c - 48)
        pos[0] += 1
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
        return a1 if a1 <= 2 else 3

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
            if line.startswith(b'##'):
                continue
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
            if line.startswith(b'##'):
                continue
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