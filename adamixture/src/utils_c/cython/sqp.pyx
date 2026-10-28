# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
cimport openmp
from libc.math cimport fabs, fmax, fmin
from libc.stdint cimport uint8_t
from libc.stdlib cimport free, malloc

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

cdef void _grad_hess_q_block(const uint8_t* G, const double* Q, const double* P,
                             double* XtX, double* Xtz, double* ws,
                             int ib, int i_end, int M, int N, int K) noexcept nogil:
    cdef:
        int nb = i_end - ib
        int npair = K * (K + 1) // 2
        int i, j, k, k2, li, pr
        double g, m, q, inv_q, inv_1mq, t1z_v, t2z_v, t2v, pk, c, v
        double pj[128]
        const uint8_t* g_row
        double* qk
        double* h
        double* zk
        double* ak
        double* xtx
        double* xtz
        # Workspace layout (each row has nb entries, one per individual in the block)
        double* Qt = ws                    # K rows
        double* H = Qt + K * nb            # npair rows: sum (term1 + term2) p_k p_k2
        double* Z = H + npair * nb         # K rows: sum (term1_z - term2_z) p_k
        double* A = Z + K * nb             # K rows: sum term2 p_k
        double* C = A + K * nb             # sum term2
        double* Z0 = C + nb                # sum term2_z
        double* qp = Z0 + nb
        double* ts = qp + nb
        double* tdz = ts + nb
        double* t2 = tdz + nb

    for k in range(K):
        qk = Qt + k * nb
        for li in range(nb):
            qk[li] = Q[<size_t>(ib + li) * K + k]
    for li in range((npair + 2 * K + 2) * nb):
        H[li] = 0.0

    for j in range(M):
        g_row = G + <size_t>j * N + ib
        for k in range(K):
            pj[k] = P[<size_t>j * K + k]

        for li in range(nb):
            qp[li] = 0.0
        for k in range(K):
            pk = pj[k]
            qk = Qt + k * nb
            for li in range(nb):
                qp[li] += pk * qk[li]

        # Missing genotypes get zero weight instead of being skipped.
        for li in range(nb):
            g = <double>g_row[li]
            m = 1.0 if g_row[li] != 3 else 0.0
            q = fmax(fmin(qp[li], 1.0 - 1e-10), 1e-10)
            inv_q = 1.0 / q
            inv_1mq = 1.0 / (1.0 - q)
            t1z_v = m * g * inv_q
            t2z_v = m * (2.0 - g) * inv_1mq
            t2v = t2z_v * inv_1mq
            ts[li] = t1z_v * inv_q + t2v
            tdz[li] = t1z_v - t2z_v
            t2[li] = t2v
            C[li] += t2v
            Z0[li] += t2z_v

        for k in range(K):
            pk = pj[k]
            zk = Z + k * nb
            ak = A + k * nb
            for li in range(nb):
                zk[li] += pk * tdz[li]
                ak[li] += pk * t2[li]

        pr = 0
        for k in range(K):
            for k2 in range(k, K):
                c = pj[k] * pj[k2]
                h = H + pr * nb
                for li in range(nb):
                    h[li] += c * ts[li]
                pr = pr + 1

    for li in range(nb):
        i = ib + li
        xtx = XtX + <size_t>i * K * K
        xtz = Xtz + <size_t>i * K
        pr = 0
        for k in range(K):
            xtz[k] = Z[k * nb + li] + Z0[li]
            for k2 in range(k, K):
                v = H[pr * nb + li] - A[k * nb + li] - A[k2 * nb + li] + C[li]
                xtx[k * K + k2] = v
                xtx[k2 * K + k] = v
                pr = pr + 1

cpdef void compute_grad_hess_Q(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                               double[:,:,::1] XtX_q, double[:,::1] Xtz_q,
                               int M, int N, int K, int B) noexcept nogil:
    cdef:
        int ib, i_end
        int n_threads = openmp.omp_get_max_threads()
        int B_par = (N + 2 * n_threads - 1) // (2 * n_threads)
        double* ws

    # Keep at least ~2 blocks per thread; the result does not depend on B.
    B_par = ((B_par + 7) // 8) * 8
    if B_par < 16:
        B_par = 16
    if B > B_par:
        B = B_par

    # Uses (1-p)(1-p)^T = 11^T - 1p^T - p1^T + pp^T, so per genotype only
    # (term1 + term2) p p^T is accumulated, plus the rank-one corrections.
    # Loops run over the B individuals of a block (contiguous in G and in the
    # workspace) so they vectorize for any K.
    with nogil, parallel():
        ws = <double*>malloc(<size_t>B * (K * (K + 1) // 2 + 3 * K + 6) * sizeof(double))
        for ib in prange(0, N, B, schedule='static'):
            i_end = ib + B
            if i_end > N:
                i_end = N
            _grad_hess_q_block(&G[0, 0], &Q[0, 0], &P[0, 0], &XtX_q[0, 0, 0], &Xtz_q[0, 0],
                               ws, ib, i_end, M, N, K)
        free(ws)

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
                        const double[:,::1] v_kk, int M, int N, int K, int B) noexcept nogil:
    cdef int i
    
    compute_grad_hess_Q(G, Q, P, XtX_q, Xtz_q, M, N, K, B)
    

    for i in prange(N, schedule='static'):
        update_single_q(i, Q, Q_next, XtX_q, Xtz_q, v_kk, K)

cdef enum:
    P_CHUNK = 256

cdef void update_single_p_fused(int j, const uint8_t* g_row, const double* Qt,
                                const double* P, double* P_next,
                                int N, int K, double* ws) noexcept nogil:
    cdef:
        int sz = K + 1
        int chunk = P_CHUNK
        int i0, n, li, k, k2
        double g, m, q, inv_q, inv_1mq, t1z, t2z, pk, s
        double tableau[17000]
        double d_buf[130]
        double tmp_buf[130]
        uint8_t swept_buf[128]
        double delta[128]
        double pj[128]
        const uint8_t* g_c
        const double* qk
        const double* qk2
        double* wk
        # Workspace layout, P_CHUNK entries per row
        double* qp = ws
        double* ts = qp + P_CHUNK
        double* tdz = ts + P_CHUNK
        double* w = tdz + P_CHUNK          # K rows: ts * Q[:, k]

    for k in range(sz * sz):
        tableau[k] = 0.0

    for k in range(K):
        pj[k] = P[<size_t>j * K + k]

    # Qt is Q transposed (K x N), so every loop below runs over contiguous
    # individuals and vectorizes for any K.
    i0 = 0
    while i0 < N:
        n = N - i0
        if n > chunk:
            n = chunk
        g_c = g_row + i0

        for li in range(n):
            qp[li] = 0.0
        for k in range(K):
            pk = pj[k]
            qk = Qt + <size_t>k * N + i0
            for li in range(n):
                qp[li] += pk * qk[li]

        # Missing genotypes get zero weight instead of being skipped.
        for li in range(n):
            g = <double>g_c[li]
            m = 1.0 if g_c[li] != 3 else 0.0
            q = fmax(fmin(qp[li], 1.0 - 1e-10), 1e-10)
            inv_q = 1.0 / q
            inv_1mq = 1.0 / (1.0 - q)
            t1z = m * g * inv_q
            t2z = m * (2.0 - g) * inv_1mq
            ts[li] = t1z * inv_q + t2z * inv_1mq
            tdz[li] = t1z - t2z

        for k in range(K):
            qk = Qt + <size_t>k * N + i0
            wk = w + k * P_CHUNK
            s = 0.0
            for li in range(n):
                wk[li] = ts[li] * qk[li]
                s += tdz[li] * qk[li]
            tableau[k * sz + K] += s

        for k in range(K):
            wk = w + k * P_CHUNK
            for k2 in range(k, K):
                qk2 = Qt + <size_t>k2 * N + i0
                s = 0.0
                for li in range(n):
                    s += wk[li] * qk2[li]
                tableau[k * sz + k2] += s
        i0 = i0 + chunk

    for k in range(K):
        tableau[K * sz + k] = tableau[k * sz + K]
        for k2 in range(k):
            tableau[k * sz + k2] = tableau[k2 * sz + k]

    quadratic_program_local(delta, tableau, pj, K, 0, d_buf, tmp_buf, swept_buf)

    for k in range(K):
        P_next[<size_t>j * K + k] = pj[k] + delta[k]
    project_p_box_row(&P_next[<size_t>j * K], K, 1e-5)

cpdef void update_p_sqp_fused(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P, 
                              double[:,::1] P_next, int M, int N, int K) noexcept nogil:
    cdef:
        int i, j, k
        double* Qt = <double*>malloc(<size_t>K * N * sizeof(double))
        double* ws

    for k in range(K):
        for i in range(N):
            Qt[<size_t>k * N + i] = Q[i, k]

    with nogil, parallel():
        ws = <double*>malloc(<size_t>(K + 3) * P_CHUNK * sizeof(double))
        for j in prange(M, schedule='static'):
            update_single_p_fused(j, &G[j, 0], Qt, &P[0, 0], &P_next[0, 0], N, K, ws)
        free(ws)
    free(Qt)

cpdef void update_p_sqp_unfused(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P, 
                                double[:,::1] P_next, double[:,:,::1] XtX_p, double[:,::1] Xtz_p, 
                                int M, int N, int K) noexcept nogil:
    cdef int j
    compute_grad_hess_P(G, Q, P, XtX_p, Xtz_p, M, N, K)
    for j in prange(M, schedule='static'):
        update_single_p(j, P, P_next, XtX_p, Xtz_p, K)

def update_p_sqp(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P, 
                 double[:,::1] P_next, *args):
    cdef:
        int M, N, K
    if len(args) == 3:
        M = int(args[0])
        N = int(args[1])
        K = int(args[2])
        update_p_sqp_fused(G, Q, P, P_next, M, N, K)
    elif len(args) == 5:
        M = int(args[2])
        N = int(args[3])
        K = int(args[4])
        update_p_sqp_unfused(G, Q, P, P_next, args[0], args[1], M, N, K)
    else:
        raise TypeError(f"update_p_sqp expects either 7 or 9 arguments, got {4 + len(args)}")

