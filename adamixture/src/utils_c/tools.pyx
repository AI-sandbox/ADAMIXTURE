# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.math cimport log, log1p, sqrtf, sqrt
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint8_t

# Read Bed data file:
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

# Reconstruct from P and Q:
cdef inline void _reconstruct(double* Q, double* p, double* rec, Py_ssize_t N, Py_ssize_t K) noexcept nogil:
    cdef:
        size_t i, k
        double* q
    for i in range(N):
        q = &Q[i*K]
        for k in range(K):
            rec[i] += p[k]*q[k]

# Individual contribution:
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

# Log-likelihood:
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

# Allele frequencies:
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

cpdef float rmse(float[:, ::1] Q1, float[:, ::1] Q2, int N, int K) noexcept nogil:
    cdef:
        size_t T = N * K
        size_t i
        float inv_T = 1.0 / <float>T
        float diff
        double acc = 0.0
        float* q1 = &Q1[0, 0]
        float* q2 = &Q2[0, 0]
    for i in range(T):
        diff = q1[i] - q2[i]
        acc += diff * diff
    return sqrtf(<float>(acc * inv_T))

# Normalize Q:
cdef inline void _norm(float* q, size_t K) noexcept nogil:
    cdef:
        size_t k
        double t = 0.0, v = 0.0
        float raw, clipped
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
    v = 1.0 / <float>t
    for k in range(K):
        q[k] *= v

# Map Q:
cpdef void mapQ(float[:,::1] Q, int N, int K) noexcept nogil:
    cdef:
        size_t j
    for j in prange(N, schedule='guided'):
        _norm(&Q[j,0], K)

# Map P:
cpdef void mapP(float[:,::1] P, int M, int K) noexcept nogil:
    cdef:
        size_t i, k
        float raw
        float* p
    for i in prange(M, schedule='guided'):
        p = &P[i,0]
        for k in range(K):
            raw = p[k]
            if raw < 1e-5:
                p[k] = 1e-5
            elif raw > 1.0 - 1e-5:
                p[k] = 1.0 - 1e-5

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
    
# Eval RMSE:
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