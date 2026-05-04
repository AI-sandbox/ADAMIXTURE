# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, sqrt, pow
from libc.stdlib cimport calloc, free
from libc.stdint cimport uint32_t

# Update temporary accumulators for P and Q
cdef inline void _update_temp_factors(double* A, double* B, double* t, const double* p, 
                                const double* q, const unsigned char g, const double rec, 
                                const size_t K) noexcept nogil:
    """
    Description:
    Updates temporary accumulators for P and Q based on genotype and current reconstruction.

    Args:
        A (double*): Accumulator for P numerator.
        B (double*): Accumulator for P denominator.
        t (double*): Accumulated intermediate values for Q.
        p (double*): Current row of P matrix.
        q (double*): Current row of Q matrix.
        g (unsigned char): Genotype value (0, 1, or 2).
        rec (double): Reconstructed value (p dot q).
        K (size_t): Number of ancestral populations.

    Returns:
        None
    """
    cdef:
        size_t k
        double g_f = <double>g
        double a = g_f/rec
        double b = (2.0-g_f)/(1.0-rec)
        double a_minus_b = a - b
    for k in range(K):
        A[k] += q[k]*a
        B[k] += q[k]*b
        t[k] += p[k]*a_minus_b + b

# Perform EM update for P row
cdef inline void _updateEM_P(double* A, double* B, const double* p, double* P_EM, const size_t K) noexcept nogil:
    """
    Description:
    Performs EM update for a row of the P matrix using pre-calculated accumulators.

    Args:
        A (double*): Numerator accumulator.
        B (double*): Denominator accumulator.
        p (double*): Current P parameters.
        P_EM (double*): Output buffer for updated P parameters.
        K (size_t): Number of ancestral populations.

    Returns:
        None
    """
    cdef:
        size_t k
    for k in range(K):
        P_EM[k] = _clip_to_domain((A[k]*p[k])/(p[k]*(A[k] - B[k]) + B[k]))
        A[k] = 0.0
        B[k] = 0.0

# Perform EM update for Q row
cdef inline void _updateEM_Q(double* T, const double* q, double* Q_EM, const double a, const size_t K) noexcept nogil:
    """
    Description:
    Performs EM update for a row of the Q matrix using accumulated terms.

    Args:
        T (double*): Accumulated terms for Q.
        q (double*): Current Q parameters.
        Q_EM (double*): Output buffer for updated Q parameters.
        a (double): Scaling factor (1 / genotype count).
        K (size_t): Number of ancestral populations.

    Returns:
        None
    """
    cdef:
        size_t k
        double totalQ = 0.0
        double inv_totalQ
    for k in range(K):
        Q_EM[k] = _clip_to_domain(q[k] * T[k] * a)
        totalQ += Q_EM[k]
    inv_totalQ = 1.0 / totalQ
    for k in range(K):
        Q_EM[k] *= inv_totalQ
        T[k] = 0.0

# Update P using Adam optimizer
cpdef void adamUpdateP(double[:,::1] P0, const double[:,::1] P1, 
                      double[:,::1] m_P, double[:,::1] v_P, 
                      const double alpha, const double beta1, const double beta2, 
                      const double epsilon, const int t, const int M, const int K) noexcept nogil:
    """
    Description:
    Updates the P matrix using the Adam optimization algorithm.

    Args:
        P0 (double[:,::1]): Target P matrix to update.
        P1 (const double[:,::1]): EM-updated P matrix (gradient proxy).
        m_P (double[:,::1]): First moment vector for Adam.
        v_P (double[:,::1]): Second moment vector for Adam.
        alpha (double): Learning rate.
        beta1 (double): Exponential decay rate for the first moment.
        beta2 (double): Exponential decay rate for the second moment.
        epsilon (double): Small constant for numerical stability.
        t (int): Current iteration number.
        M (int): Number of SNPs.
        K (int): Number of ancestral populations.

    Returns:
        None
    """
    cdef:
        size_t i, j
        double delta, m_hat, v_hat, step
        double beta1_t = pow(beta1, t)
        double beta2_t = pow(beta2, t)
        double m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
        double v_scale = 1.0 / (1.0 - beta2_t) if beta2_t != 1.0 else 1.0
        double one_m_beta1 = 1.0 - beta1
        double one_m_beta2 = 1.0 - beta2
        double* p0_ptr
        const double* p1_ptr
        double* m_p_ptr
        double* v_p_ptr
    
    for i in prange(M, schedule='guided'):
        p0_ptr = &P0[i, 0]
        p1_ptr = &P1[i, 0]
        m_p_ptr = &m_P[i, 0]
        v_p_ptr = &v_P[i, 0]
        for j in range(K):
            # Update moments
            delta = p1_ptr[j] - p0_ptr[j]
            m_p_ptr[j] = beta1 * m_p_ptr[j] + one_m_beta1 * delta
            v_p_ptr[j] = beta2 * v_p_ptr[j] + one_m_beta2 * delta * delta
            
            # Apply updates
            m_hat = m_p_ptr[j] * m_scale
            v_hat = v_p_ptr[j] * v_scale
            step = alpha * m_hat / (sqrt(v_hat) + epsilon)
            p0_ptr[j] = _clip_to_domain(p0_ptr[j] + step)

# Update Q using Adam optimizer
cpdef void adamUpdateQ(double[:,::1] Q0, const double[:,::1] Q1, 
                      double[:,::1] m_Q, double[:,::1] v_Q, 
                      const double alpha, const double beta1, const double beta2, 
                      const double epsilon, const int t, const int N, const int K) noexcept nogil:
    """
    Description:
    Updates the Q matrix using the Adam optimization algorithm.

    Args:
        Q0 (double[:,::1]): Target Q matrix to update.
        Q1 (const double[:,::1]): EM-updated Q matrix (gradient proxy).
        m_Q (double[:,::1]): First moment vector for Adam.
        v_Q (double[:,::1]): Second moment vector for Adam.
        alpha (double): Learning rate.
        beta1 (double): Exponential decay rate for the first moment.
        beta2 (double): Exponential decay rate for the second moment.
        epsilon (double): Small constant for numerical stability.
        t (int): Current iteration number.
        N (int): Number of samples.
        K (int): Number of ancestral populations.

    Returns:
        None
    """
    cdef:
        size_t i, j
        double delta, m_hat, v_hat, step, sumQ, inv_sumQ
        double beta1_t = pow(beta1, t)
        double beta2_t = pow(beta2, t)
        double m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
        double v_scale = 1.0 / (1.0 - beta2_t) if beta2_t != 1.0 else 1.0
        double one_m_beta1 = 1.0 - beta1
        double one_m_beta2 = 1.0 - beta2
        double* q0
        const double* q1
        double* m_q
        double* v_q
    
    for i in prange(N, schedule='guided'):
        q0 = &Q0[i, 0]
        q1 = &Q1[i, 0]
        m_q = &m_Q[i, 0]
        v_q = &v_Q[i, 0]
        sumQ = 0.0
        for j in range(K):
            # Update moments
            delta = q1[j] - q0[j]
            m_q[j] = beta1 * m_q[j] + one_m_beta1 * delta
            v_q[j] = beta2 * v_q[j] + one_m_beta2 * delta * delta
            
            # Apply updates
            m_hat = m_q[j] * m_scale
            v_hat = v_q[j] * v_scale
            step = alpha * m_hat / (sqrt(v_hat) + epsilon)
            q0[j] = _clip_to_domain(q0[j] + step)

            # Accumulate for normalization
            sumQ = sumQ + q0[j]
        
        inv_sumQ = 1.0 / sumQ
        for j in range(K):
            q0[j] *= inv_sumQ

# Coordinated EM P-step
cpdef void P_step(const unsigned char[:,::1] G, double[:,::1] P, double[:,::1] P_EM, 
                const double[:,::1] Q, double[:,::1] Q_T, double[::1] q_bat, 
                const int K, const int M, const int N) noexcept nogil:
    """
    Description:
    Coordinated EM step for P matrix update. Calculates EM P and accumulates factors for Q update.

    Args:
        G (const unsigned char[:,::1]): Genotype matrix.
        P (double[:,::1]): Current P matrix.
        P_EM (double[:,::1]): Output buffer for EM-updated P.
        Q (const double[:,::1]): Current Q matrix.
        Q_T (double[:,::1]): Accumulator for Q update terms.
        q_bat (double[::1]): Accumulator for genotype weights per sample.
        K (int): Number of ancestral populations.
        M (int): Number of SNPs.
        N (int): Number of samples.

    Returns:
        None
    """
    cdef:
        size_t col, row, idx_n, idx_k
        double rec
        double* p_ptr
        double* dest_row 
        const unsigned char* g_ptr
        double* A
        double* B
        double* t
        double* q_len
        omp.omp_lock_t sync_lock

    omp.omp_init_lock(&sync_lock)
    
    with nogil, parallel():
        A = <double*>calloc(K, sizeof(double))
        B = <double*>calloc(K, sizeof(double))
        t = <double*>calloc(N * K, sizeof(double))
        q_len = <double*>calloc(N, sizeof(double))
        
        for row in prange(M, schedule='guided'):
            p_ptr = &P[row, 0]
            g_ptr = &G[row, 0]
            for col in range(N):
                if g_ptr[col] == 3:
                    continue
                q_len[col] += 2.0
                rec = _reconstruct(p_ptr, &Q[col, 0], K)
                _update_temp_factors(A, B, &t[col * K], p_ptr, &Q[col, 0], g_ptr[col], rec, K)
            _updateEM_P(A, B, p_ptr, &P_EM[row, 0], K)
            
        omp.omp_set_lock(&sync_lock)
        for idx_n in range(N):
            q_bat[idx_n] = q_bat[idx_n] + q_len[idx_n]
            dest_row = &Q_T[idx_n, 0] 
            for idx_k in range(K):
                dest_row[idx_k] += t[idx_n * K + idx_k]   
        omp.omp_unset_lock(&sync_lock)

        free(A)
        free(B)
        free(t)
        free(q_len)
        
    omp.omp_destroy_lock(&sync_lock)

# Coordinated EM Q-step
cpdef void Q_step(const double[:,::1] Q, double[:,::1] Q_EM, double[:,::1] T, 
                double[::1] q_bat, const int K, const int N) noexcept nogil:
    """
    Description:
    Coordinated EM step for Q matrix update.

    Args:
        Q (const double[:,::1]): Current Q matrix.
        Q_EM (double[:,::1]): Output buffer for EM-updated Q.
        T (double[:,::1]): Accumulated update terms for Q.
        q_bat (double[::1]): Weights (genotype counts) for normalization.
        K (int): Number of ancestral populations.
        N (int): Number of samples.

    Returns:
        None
    """
    cdef:
        size_t i, k
        double a
    for i in range(N):
        a = 1.0 / q_bat[i]
        _updateEM_Q(&T[i, 0], &Q[i, 0], &Q_EM[i, 0], a, K)
        q_bat[i] = 0.0

# Clip value to valid domain
cdef inline double _clip_to_domain(const double value) noexcept nogil:
    """
    Description:
    Clips a value to the valid range [1e-5, 1-1e-5] to ensure numerical stability.

    Args:
        value (const double): Value to clip.

    Returns:
        double: Clipped value.
    """
    return fmin(fmax(value, 1e-5), 1.0 - 1e-5)

# Single genotype reconstruction (p dot q)
cdef inline double _reconstruct(const double* p, const double* q, const size_t K) noexcept nogil:
    """
    Description:
    Computes the dot product of p and q vectors (single reconstruction point).

    Args:
        p (const double*): Row of P.
        q (const double*): Row of Q.
        K (size_t): Number of ancestral populations.

    Returns:
        double: Dot product result.
    """
    cdef:
        size_t k
        double rec = 0.0
    for k in range(K):
        rec += p[k]*q[k]
    return rec
