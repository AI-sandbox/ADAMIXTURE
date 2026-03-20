# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.math cimport log, log1p, sqrtf, sqrt
from libc.stdlib cimport calloc, free, malloc
from libc.stdint cimport uint8_t, uint32_t, uintptr_t, int32_t, uint64_t

# Read BED file:
cpdef void read_bed(unsigned char[:,::1] bed_source, unsigned char[:,::1] geno_target) noexcept nogil:
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

# Reconstruct G from P and Q:
cdef inline void _reconstruct(double* Q, double* p, double* rec, Py_ssize_t N, Py_ssize_t K) noexcept nogil:
    cdef:
        size_t i, k
        double* q
    for i in range(N):
        q = &Q[i*K]
        for k in range(K):
            rec[i] += p[k]*q[k]

# Calculate individual log-likelihood:
cdef inline double _ind_loglike(uint8_t* g, double* rec, Py_ssize_t N) noexcept nogil:
    cdef:
        size_t i
        double ll = 0.0
        double g_d
    for i in range(N):
        g_d = <double>g[i]
        ll += g_d*log(rec[i]) + (2.0-g_d)*log1p(-rec[i]) if g[i] != 3 else 0.0
        rec[i] = 0.0
    return ll

# Read Bed data file (packed):
# G_packed will be (M_bytes, N) where M_bytes = ceil(M/4)
cpdef void read_bed_packed(uintptr_t B_bed_ptr, uintptr_t G_packed_ptr, Py_ssize_t M, Py_ssize_t N_bytes, Py_ssize_t N, Py_ssize_t M_bytes) noexcept nogil:
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

# Calculate log-likelihood:
cpdef double loglikelihood(uint8_t[:,::1] G, double[:,::1] P, double[:,::1] Q) noexcept nogil:
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

# Calculate allele frequency:
cpdef void alleleFrequency(uint8_t[:, ::1] G, float[::1] f, int M, int N) noexcept nogil:
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

# Calculate RMSE:
cpdef double rmse_d(double[:,::1] Q1, double[:,::1] Q2, int N, int K) noexcept nogil:
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

# Normalize Q:
cdef inline void _norm_d(double* q, size_t K) noexcept nogil:
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

# Map Q:
cpdef void mapQ_d(double[:,::1] Q, int N, int K) noexcept nogil:
    cdef size_t j
    with nogil, parallel():
        for j in prange(N, schedule='guided'):
            _norm_d(&Q[j,0], K)

# Map P:
cpdef void mapP_d(double[:,::1] P, int M, int K) noexcept nogil:
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

# Solve subset of equations:
cdef bint solve_subset_ptr(const double* A, const double* b, double* x, int* F, int K, double* aug, int* map_idx) noexcept nogil:
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

# Exact NNLS (Block Principal Pivoting):
cdef void _exact_nnls_bpp_ptr(const double* A, const double* b, double* x, int K, double* aug, int* map_idx, int* F, double* y) noexcept nogil:
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

# Batch NNLS solver for P and Q updates
cpdef void batch_nnls_bpp(double[:, ::1] A_cov, double[:, ::1] B_target, double[:, ::1] Out_mat) noexcept nogil:
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

# Eval KL divergence:
cpdef double KL(double[:, ::1] Q1, double[:, ::1] Q2, int N, int K) noexcept nogil:
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

# Decompress a block of genotypes (centered) for SVD
cpdef void decompress_block(const uint8_t[:, ::1] G, float[:, ::1] data_block, const float[::1] freq, const Py_ssize_t offset) noexcept nogil:
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

# Pack a genotype matrix into 2-bit format (M_bytes, N)
cpdef void pack_genotypes(uintptr_t G_ptr, uintptr_t G_packed_ptr, Py_ssize_t M, Py_ssize_t N, Py_ssize_t M_bytes) noexcept nogil:
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

# Calculate mean of unpacked genotypes (ignoring 3):
cpdef double get_mean_unpacked(uint8_t[:, ::1] G) noexcept nogil:
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

# In-place flip of unpacked genotypes:
cpdef void flip_unpacked(uint8_t[:, ::1] G) noexcept nogil:
    cdef:
        size_t M = G.shape[0]
        size_t N = G.shape[1]
        size_t i, j
        uint8_t[4] lookup = [2, 1, 0, 3]
    
    with nogil, parallel():
        for i in prange(M, schedule='guided'):
            for j in range(N):
                G[i, j] = lookup[G[i, j]]

# Calculate mean of packed genotypes (ignoring 3):
cpdef double get_mean_packed(uintptr_t G_ptr, size_t M, size_t N, size_t M_bytes) noexcept nogil:
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

# In-place flip of packed genotypes:
cpdef void flip_packed(uintptr_t G_ptr, size_t M_bytes, size_t N) noexcept nogil:
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
