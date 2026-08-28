# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import prange
from libc.math cimport fabs, fmax, fmin
from libc.stdint cimport uint8_t

cdef void sweep(double* matrix_a, int sz, int k, double* tmp, bint inverse) noexcept nogil:
    cdef:
        double piv = matrix_a[k * sz + k]
        double p, pv
        int i, j
        
    if piv == 0.0:
        return
        
    p = 1.0 / piv
    for i in range(sz):
        tmp[i] = matrix_a[i * sz + k]
        matrix_a[i * sz + k] = 0.0
        matrix_a[k * sz + i] = 0.0
        
    if inverse:
        tmp[k] = 1.0
    else:
        tmp[k] = -1.0
        
    for i in range(sz):
        pv = p * tmp[i]
        for j in range(sz):
            matrix_a[j * sz + i] -= pv * tmp[j]

cdef int quadratic_program_local(double* delta, double* tableau, const double* par,
                                 int p, int c,
                                 double* d, double* tmp, uint8_t* swept) noexcept nogil:
    cdef:
        int sz = p + c + 1
        int i, iteration
        double small = 1e-5
        double tol = 1e-8
        double a, ai, ui, temp
        bint cycle_main_loop, critical, violation
        
    for i in range(p):
        delta[i] = 0.0
        
    for i in range(sz):
        d[i] = tableau[i * sz + i]
        
    for i in range(p):
        if d[i] <= 0.0 or tableau[i * sz + i] < d[i] * tol:
            return 0
        else:
            sweep(tableau, sz, i, tmp, False)
            
    for i in range(p):
        swept[i] = 1
        
    for i in range(p, p + c):
        if tableau[i * sz + i] >= 0.0:
            return 0
        else:
            sweep(tableau, sz, i, tmp, False)
            
    for iteration in range(1, 1001):
        a = 1.0
        for i in range(p):
            if swept[i]:
                ui = tableau[i * sz + (sz - 1)]
                if ui > 0.0:
                    ai = 1.0 - par[i] - delta[i]
                else:
                    ai = 0.0 - par[i] - delta[i]
                if fabs(ui) > 1e-10:
                    temp = ai / ui
                    if temp < a:
                        a = temp
                        
        for i in range(p):
            if swept[i]:
                ui = tableau[i * sz + (sz - 1)]
                delta[i] = delta[i] + a * ui
                tableau[i * sz + (sz - 1)] = (1.0 - a) * ui
                tableau[(sz - 1) * sz + i] = tableau[i * sz + (sz - 1)]
                
        cycle_main_loop = False
        for i in range(p):
            critical = (0.0 >= par[i] + delta[i] - small) or (1.0 <= par[i] + delta[i] + small)
            if swept[i] and (fabs(tableau[i * sz + i]) > 1e-10) and critical:
                sweep(tableau, sz, i, tmp, True)
                swept[i] = 0
                cycle_main_loop = True
                break
                
        if cycle_main_loop:
            continue
            
        for i in range(p):
            ui = tableau[i * sz + (sz - 1)]
            violation = (ui > 0.0 and 0.0 >= par[i] + delta[i] - small) or (ui < 0.0 and 1.0 <= par[i] + delta[i] + small)
            if (not swept[i]) and violation:
                sweep(tableau, sz, i, tmp, False)
                swept[i] = 1
                cycle_main_loop = True
                break
                
        if cycle_main_loop:
            continue
            
        return iteration
        
    return 0

cdef void project_q_simplex_row(double* b, int n, double pseudocount) noexcept nogil:
    cdef:
        double tau = 1.0 - n * pseudocount
        double tsum = 0.0
        double tmax = 0.0
        bint bget = False
        int i, j, key_idx
        double key_val
        int idx[256]
        
    for i in range(n):
        b[i] -= pseudocount
        idx[i] = i
        
    for i in range(1, n):
        key_idx = idx[i]
        key_val = b[key_idx]
        j = i - 1
        while j >= 0 and b[idx[j]] < key_val:
            idx[j + 1] = idx[j]
            j = j - 1
        idx[j + 1] = key_idx
        
    for i in range(n - 1):
        tsum += b[idx[i]]
        tmax = (tsum - tau) / (i + 1)
        if tmax >= b[idx[i + 1]]:
            bget = True
            break
            
    if not bget:
        tmax = (tsum + b[idx[n - 1]] - tau) / n
        
    for i in range(n):
        b[i] = fmax(b[i] - tmax, 0.0) + pseudocount

cdef void project_p_box_row(double* b, int n, double pseudocount) noexcept nogil:
    cdef int i
    for i in range(n):
        b[i] = fmin(fmax(b[i], pseudocount), 1.0 - pseudocount)

cdef void create_tableau_simplex_local(double* tableau, const double* matrix_q, const double* r,
                                       const double* x, const double[:,::1] v_kk, int K) noexcept nogil:
    cdef:
        int sz = K + 2
        int i, j
        double mu, norm1, sum_x
        double tmp_k[128]
        double tmp_k2[128]
        
    for i in range(sz * sz):
        tableau[i] = 0.0
        
    for i in range(K):
        tmp_k[i] = 0.0
        for j in range(K):
            tmp_k[i] += matrix_q[i * K + j] * v_kk[j, 0]
            
    for i in range(K):
        tmp_k2[i] = 0.0
        for j in range(K):
            tmp_k2[i] += v_kk[i, j] * tmp_k[j]
            
    norm1 = 0.0
    for i in range(K):
        norm1 += fabs(tmp_k2[i])
        
    mu = (norm1 - 2.0 * fabs(tmp_k2[0])) / K
    mu = 2.0 * mu
    if mu < 0.0:
        mu = 0.0
    
    for i in range(K):
        for j in range(K):
            tableau[i * sz + j] = matrix_q[i * K + j] + mu
            
    for i in range(K):
        tableau[i * sz + K] = 1.0
        tableau[K * sz + i] = 1.0
        tableau[i * sz + (K + 1)] = r[i]
        tableau[(K + 1) * sz + i] = r[i]
        
    tableau[K * sz + K] = 0.0
    
    sum_x = 0.0
    for i in range(K):
        sum_x += x[i]
        
    tableau[K * sz + (K + 1)] = 1.0 - sum_x
    tableau[(K + 1) * sz + K] = 1.0 - sum_x
    tableau[(K + 1) * sz + (K + 1)] = 0.0

cdef void create_tableau_box_local(double* tableau, const double* matrix_q, const double* r,
                                   int K) noexcept nogil:
    cdef:
        int sz = K + 1
        int i, j
        
    for i in range(sz * sz):
        tableau[i] = 0.0
        
    for i in range(K):
        for j in range(K):
            tableau[i * sz + j] = matrix_q[i * K + j]
            
    for i in range(K):
        tableau[i * sz + K] = r[i]
        tableau[K * sz + i] = r[i]
        
    tableau[K * sz + K] = 0.0

cdef void update_single_q(int i, const double[:,::1] Q, double[:,::1] Q_next, 
                          double[:,:,::1] XtX_q, double[:,::1] Xtz_q, 
                          const double[:,::1] v_kk, int K) noexcept nogil:
    cdef:
        double tableau[17000]
        double d_buf[130]
        double tmp_buf[130]
        uint8_t swept_buf[128]
        double delta[128]
        int k
        
    create_tableau_simplex_local(tableau, &XtX_q[i, 0, 0], &Xtz_q[i, 0], &Q[i, 0], v_kk, K)
    quadratic_program_local(delta, tableau, &Q[i, 0], K, 1, d_buf, tmp_buf, swept_buf)
    
    for k in range(K):
        Q_next[i, k] = Q[i, k] + delta[k]
    project_q_simplex_row(&Q_next[i, 0], K, 1e-5)

cdef void update_single_p(int j, const double[:,::1] P, double[:,::1] P_next, 
                          double[:,:,::1] XtX_p, double[:,::1] Xtz_p, int K) noexcept nogil:
    cdef:
        double tableau[17000]
        double d_buf[130]
        double tmp_buf[130]
        uint8_t swept_buf[128]
        double delta[128]
        int k
        
    create_tableau_box_local(tableau, &XtX_p[j, 0, 0], &Xtz_p[j, 0], K)
    quadratic_program_local(delta, tableau, &P[j, 0], K, 0, d_buf, tmp_buf, swept_buf)
    
    for k in range(K):
        P_next[j, k] = P[j, k] + delta[k]
    project_p_box_row(&P_next[j, 0], K, 1e-5)

cdef void compute_grad_hess_Q_unblocked(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                                        double[:,:,::1] XtX_q, double[:,::1] Xtz_q,
                                        int M, int N, int K) noexcept nogil:
    cdef:
        int i, j, k, k2
        double qp, g
        double oneT = 1.0
        double twoT = 2.0
        double term1_z, term2_z, term1, term2
        double pk, t1pk, t2pk

    for i in prange(N, schedule='static'):
        for k in range(K):
            Xtz_q[i, k] = 0.0
            for k2 in range(k, K):
                XtX_q[i, k, k2] = 0.0

        for j in range(M):
            g = <double>G[j, i]
            if g == 3.0:
                continue
            qp = 0.0
            for k in range(K):
                qp += Q[i, k] * P[j, k]
            qp = fmax(fmin(qp, 1.0 - 1e-10), 1e-10)
            
            term1_z = g / qp
            term2_z = (twoT - g) / (oneT - qp)
            term1 = term1_z / qp
            term2 = term2_z / (oneT - qp)
            
            for k in range(K):
                pk = P[j, k]
                Xtz_q[i, k] += term1_z * pk + term2_z * (oneT - pk)
                t1pk = term1 * pk
                t2pk = term2 * (oneT - pk)
                for k2 in range(k, K):
                    XtX_q[i, k, k2] += t1pk * P[j, k2] + t2pk * (oneT - P[j, k2])

        for k in range(K):
            for k2 in range(k):
                XtX_q[i, k, k2] = XtX_q[i, k2, k]

cpdef void compute_grad_hess_Q(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                               double[:,:,::1] XtX_q, double[:,::1] Xtz_q,
                               int M, int N, int K) noexcept nogil:
    cdef:
        int ib, i, i_end, j, k, k2
        double qp, g
        double term1_z, term2_z, term1, term2
        double pk, t1pk, t2pk

    if K <= 8:
        compute_grad_hess_Q_unblocked(G, Q, P, XtX_q, Xtz_q, M, N, K)
        return

    # Process nearby individuals together. G is row-major, so this changes the
    # strided G[:, i] reads into short contiguous runs and reuses each P row.
    for ib in prange(0, N, 8, schedule='static'):
        i_end = ib + 8
        if i_end > N:
            i_end = N

        for i in range(ib, i_end):
            for k in range(K):
                Xtz_q[i, k] = 0.0
                for k2 in range(k, K):
                    XtX_q[i, k, k2] = 0.0

        for j in range(M):
            for i in range(ib, i_end):
                g = <double>G[j, i]
                if g == 3.0:
                    continue
                qp = 0.0
                for k in range(K):
                    qp += Q[i, k] * P[j, k]
                qp = fmax(fmin(qp, 1.0 - 1e-10), 1e-10)

                term1_z = g / qp
                term2_z = (2.0 - g) / (1.0 - qp)
                term1 = term1_z / qp
                term2 = term2_z / (1.0 - qp)

                for k in range(K):
                    pk = P[j, k]
                    Xtz_q[i, k] += term1_z * pk + term2_z * (1.0 - pk)
                    t1pk = term1 * pk
                    t2pk = term2 * (1.0 - pk)
                    for k2 in range(k, K):
                        XtX_q[i, k, k2] += (
                            t1pk * P[j, k2] + t2pk * (1.0 - P[j, k2])
                        )

        for i in range(ib, i_end):
            for k in range(K):
                for k2 in range(k):
                    XtX_q[i, k, k2] = XtX_q[i, k2, k]

cpdef void compute_grad_hess_P(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                               double[:,:,::1] XtX_p, double[:,::1] Xtz_p, int M, int N, int K) noexcept nogil:
    cdef:
        int i, j, k, k2
        double qp, g
        double oneT = 1.0
        double twoT = 2.0
        double term1_z, term2_z, term1, term2, term_z_diff, term_sum
        double tqk

    for j in prange(M, schedule='static'):
        for k in range(K):
            Xtz_p[j, k] = 0.0
            for k2 in range(k, K):
                XtX_p[j, k, k2] = 0.0

        for i in range(N):
            g = <double>G[j, i]
            if g == 3.0:
                continue
            qp = 0.0
            for k in range(K):
                qp += Q[i, k] * P[j, k]
            qp = fmax(fmin(qp, 1.0 - 1e-10), 1e-10)
            
            term1_z = g / qp
            term2_z = (twoT - g) / (oneT - qp)
            term1 = term1_z / qp
            term2 = term2_z / (oneT - qp)
            term_z_diff = term1_z - term2_z
            term_sum = term1 + term2
            
            for k in range(K):
                Xtz_p[j, k] += term_z_diff * Q[i, k]
                tqk = term_sum * Q[i, k]
                for k2 in range(k, K):
                    XtX_p[j, k, k2] += tqk * Q[i, k2]

        for k in range(K):
            for k2 in range(k):
                XtX_p[j, k, k2] = XtX_p[j, k2, k]

cpdef void project_q_simplex(double[:,::1] Q, int N, int K) noexcept nogil:
    cdef int i
    for i in prange(N, schedule='static'):
        project_q_simplex_row(&Q[i, 0], K, 1e-5)

cpdef void project_p_box(double[:,::1] P, int M, int K) noexcept nogil:
    cdef int j
    for j in prange(M, schedule='static'):
        project_p_box_row(&P[j, 0], K, 1e-5)

cpdef void update_q_sqp(const uint8_t[:,::1] G, const double[:,::1] Q, double[:,::1] Q_next, 
                        const double[:,::1] P, double[:,:,::1] XtX_q, double[:,::1] Xtz_q, 
                        const double[:,::1] v_kk, int M, int N, int K) noexcept nogil:
    cdef int i
    
    compute_grad_hess_Q(G, Q, P, XtX_q, Xtz_q, M, N, K)
    

    for i in prange(N, schedule='static'):
        update_single_q(i, Q, Q_next, XtX_q, Xtz_q, v_kk, K)

cpdef void update_p_sqp(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P, 
                        double[:,::1] P_next, double[:,:,::1] XtX_p, double[:,::1] Xtz_p, 
                        int M, int N, int K) noexcept nogil:
    cdef int j
    
    compute_grad_hess_P(G, Q, P, XtX_p, Xtz_p, M, N, K)
    

    for j in prange(M, schedule='static'):
        update_single_p(j, P, P_next, XtX_p, Xtz_p, K)
