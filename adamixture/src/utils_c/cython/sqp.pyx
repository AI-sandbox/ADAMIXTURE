# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
cimport openmp
from libc.math cimport fabs, fmax, fmin
from libc.stdint cimport uint8_t
from libc.stdlib cimport free, malloc
from libc.string cimport memset

cdef extern from "blas_helper.h" nogil:
    int adm_blas_has_dgemm()
    int adm_blas_begin_single_thread()
    void adm_blas_end_single_thread(int prev)
    void adm_dgemm_nn(int M, int N, int K, double alpha,
                      const double* A, int lda,
                      const double* B, int ldb,
                      double beta, double* C, int ldc)
    void adm_dgemm_tn(int M, int N, int K, double alpha,
                      const double* A, int lda,
                      const double* B, int ldb,
                      double beta, double* C, int ldc)
    void adm_sqp_fill_4snps_f64(const uint8_t* g_row,
                                const double* qp0, const double* qp1,
                                const double* qp2, const double* qp3,
                                double* ts0, double* ts1,
                                double* ts2, double* ts3,
                                double* tdz0, double* tdz1,
                                double* tdz2, double* tdz3,
                                int n)
    void adm_sqp_fill_1snp_f64(const uint8_t* g_row, uint8_t shift,
                               const double* qp,
                               double* ts, double* tdz,
                               int n)


cpdef bint blas_has_dgemm():
    return adm_blas_has_dgemm() != 0


cdef void sweep(double* matrix_a, int sz, int k, double* tmp, bint inverse) noexcept nogil:
    cdef:
        double piv = matrix_a[k * sz + k]
        double p, pv
        int i, j

    if piv == 0.0:
        return

    p = 1.0 / piv
    for i in range(sz):
        tmp[i] = matrix_a[k * sz + i]
        matrix_a[i * sz + k] = 0.0
        matrix_a[k * sz + i] = 0.0

    if inverse:
        tmp[k] = 1.0
    else:
        tmp[k] = -1.0

    for i in range(sz):
        pv = p * tmp[i]
        for j in range(sz):
            matrix_a[i * sz + j] -= pv * tmp[j]


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

    for iteration in range(1, 1002):
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


cdef void create_tableau_simplex_local(double* tableau, const double* matrix_q, const double* r,
                                       const double* x, const double* v_kk, int K,
                                       double* tmp_k, double* tmp_k2) noexcept nogil:
    cdef:
        int sz = K + 2
        int i, j
        double mu, norm1, sum_x

    memset(tableau, 0, <size_t>sz * sz * sizeof(double))

    for i in range(K):
        tmp_k[i] = 0.0
        for j in range(K):
            tmp_k[i] += matrix_q[i * K + j] * v_kk[j * K]

    for i in range(K):
        tmp_k2[i] = 0.0
        for j in range(K):
            tmp_k2[i] += v_kk[i * K + j] * tmp_k[j]

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


cdef void project_q_simplex_row_idx(double* b, int n, double pseudocount, int* idx) noexcept nogil:
    cdef:
        double tau = 1.0 - n * pseudocount
        double tsum = 0.0
        double tmax = 0.0
        bint bget = False
        int i, j, key_idx
        double key_val

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


cdef void project_q_simplex_row(double* b, int n, double pseudocount) noexcept nogil:
    cdef:
        int idx_stack[256]
        int* idx_ptr = &idx_stack[0]
    if n > 256:
        idx_ptr = <int*>malloc(<size_t>n * sizeof(int))
        if idx_ptr == NULL:
            return
    project_q_simplex_row_idx(b, n, pseudocount, idx_ptr)
    if n > 256:
        free(idx_ptr)


cdef void project_p_box_row(double* p, int n, double pseudocount) noexcept nogil:
    cdef int i
    for i in range(n):
        p[i] = fmin(fmax(p[i], pseudocount), 1.0 - pseudocount)


cdef enum:
    Q_SNP_TILE = 256
    P_SNP_TILE = 128
    P_IND_TILE = 1040


cdef inline void _q_fill_4snps_f64(const uint8_t* g_row, double* qp0, double* qp1, double* qp2, double* qp3,
                                   double* ts0, double* ts1, double* ts2, double* ts3,
                                   double* tdz0, double* tdz1, double* tdz2, double* tdz3,
                                   double* t2_0, double* t2_1, double* t2_2, double* t2_3,
                                   double* C_acc, double* Z0_acc, int nb) noexcept nogil:
    cdef:
        int li
        uint8_t b, u0, u1, u2, u3
        double g0, g1, g2, g3, m0, m1, m2, m3
        double q0, q1, q2, q3, inv0, inv1, inv2, inv3
        double t2z, t2v

    for li in range(nb):
        b = g_row[li]

        u0 = b & 0x03
        m0 = <double>(u0 != 3)
        g0 = <double>u0
        q0 = fmax(fmin(qp0[li], 1.0 - 1e-10), 1e-10)
        inv0 = m0 / (q0 * (1.0 - q0))
        tdz0[li] = (g0 - 2.0 * q0) * inv0
        ts0[li] = (g0 * (1.0 - 2.0 * q0) + 2.0 * q0 * q0) * (inv0 * inv0)
        t2z = m0 * (2.0 - g0) * q0 * inv0
        t2v = t2z * q0 * inv0
        t2_0[li] = t2v
        C_acc[li] += t2v
        Z0_acc[li] += t2z

        u1 = (b >> 2) & 0x03
        m1 = <double>(u1 != 3)
        g1 = <double>u1
        q1 = fmax(fmin(qp1[li], 1.0 - 1e-10), 1e-10)
        inv1 = m1 / (q1 * (1.0 - q1))
        tdz1[li] = (g1 - 2.0 * q1) * inv1
        ts1[li] = (g1 * (1.0 - 2.0 * q1) + 2.0 * q1 * q1) * (inv1 * inv1)
        t2z = m1 * (2.0 - g1) * q1 * inv1
        t2v = t2z * q1 * inv1
        t2_1[li] = t2v
        C_acc[li] += t2v
        Z0_acc[li] += t2z

        u2 = (b >> 4) & 0x03
        m2 = <double>(u2 != 3)
        g2 = <double>u2
        q2 = fmax(fmin(qp2[li], 1.0 - 1e-10), 1e-10)
        inv2 = m2 / (q2 * (1.0 - q2))
        tdz2[li] = (g2 - 2.0 * q2) * inv2
        ts2[li] = (g2 * (1.0 - 2.0 * q2) + 2.0 * q2 * q2) * (inv2 * inv2)
        t2z = m2 * (2.0 - g2) * q2 * inv2
        t2v = t2z * q2 * inv2
        t2_2[li] = t2v
        C_acc[li] += t2v
        Z0_acc[li] += t2z

        u3 = (b >> 6) & 0x03
        m3 = <double>(u3 != 3)
        g3 = <double>u3
        q3 = fmax(fmin(qp3[li], 1.0 - 1e-10), 1e-10)
        inv3 = m3 / (q3 * (1.0 - q3))
        tdz3[li] = (g3 - 2.0 * q3) * inv3
        ts3[li] = (g3 * (1.0 - 2.0 * q3) + 2.0 * q3 * q3) * (inv3 * inv3)
        t2z = m3 * (2.0 - g3) * q3 * inv3
        t2v = t2z * q3 * inv3
        t2_3[li] = t2v
        C_acc[li] += t2v
        Z0_acc[li] += t2z


cdef inline void _q_fill_1snp_f64(const uint8_t* g_row, uint8_t shift,
                                  double* qp, double* ts, double* tdz, double* t2,
                                  double* C_acc, double* Z0_acc, int nb) noexcept nogil:
    cdef:
        int li
        uint8_t u
        double g, m, q, inv, t2z, t2v

    for li in range(nb):
        u = (g_row[li] >> shift) & 0x03
        m = <double>(u != 3)
        g = <double>u
        q = fmax(fmin(qp[li], 1.0 - 1e-10), 1e-10)
        inv = m / (q * (1.0 - q))
        tdz[li] = (g - 2.0 * q) * inv
        ts[li] = (g * (1.0 - 2.0 * q) + 2.0 * q * q) * (inv * inv)
        t2z = m * (2.0 - g) * q * inv
        t2v = t2z * q * inv
        t2[li] = t2v
        C_acc[li] += t2v
        Z0_acc[li] += t2z


cdef void _grad_hess_q_tile(const uint8_t* G, const double* Q, const double* P,
                            double* XtX, double* Xtz, double* Q_next,
                            const double* v_kk,
                            double* ws, uint8_t* swept_ws, int* idx_ws,
                            int ib, int i_end, int M, int N, int K,
                            int q_c_max) noexcept nogil:
    cdef:
        int nb = i_end - ib
        int npair = K * (K + 1) // 2
        int sz = K + 2
        int j0, c, n4, r4, r, k, k2, li, pr, off
        double pk, v
        const double* pj
        const double* qi
        double* pp
        double* xtx
        double* xtz
        double* q_out
        double* Qt_block = ws
        double* H_acc = Qt_block + <size_t>K * nb
        double* Z_acc = H_acc + <size_t>npair * nb
        double* A_acc = Z_acc + <size_t>K * nb
        double* C_acc = A_acc + <size_t>K * nb
        double* Z0_acc = C_acc + nb
        double* qp_tile = Z0_acc + nb
        double* tdz_tile = qp_tile + <size_t>q_c_max * nb
        double* ts_tile = tdz_tile + <size_t>q_c_max * nb
        double* t2_tile = ts_tile + <size_t>q_c_max * nb
        double* P_pairs_tile = t2_tile + <size_t>q_c_max * nb
        double* qp_d_ws = P_pairs_tile + <size_t>q_c_max * npair
        double* tableau = qp_d_ws
        double* d_buf = tableau + <size_t>sz * sz
        double* tmp_buf = d_buf + sz
        double* delta = tmp_buf + sz
        double* tmp_k = delta + K
        double* tmp_k2 = tmp_k + K
        double* xtx_local = (tmp_k2 + K) if (Q_next != NULL and (XtX == NULL or Xtz == NULL)) else NULL
        double* xtz_local = (xtx_local + <size_t>K * K) if xtx_local != NULL else NULL

    for k in range(K):
        for li in range(nb):
            Qt_block[k * nb + li] = Q[<size_t>(ib + li) * K + k]

    memset(H_acc, 0, <size_t>(npair + 2 * K + 2) * nb * sizeof(double))

    j0 = 0
    while j0 < M:
        c = M - j0
        if c > Q_SNP_TILE:
            c = Q_SNP_TILE

        adm_dgemm_nn(c, nb, K, 1.0, P + <size_t>j0 * K, K, Qt_block, nb, 0.0, qp_tile, nb)

        n4 = c >> 2
        for r4 in range(n4):
            off = (r4 << 2) * nb
            _q_fill_4snps_f64(
                G + <size_t>((j0 >> 2) + r4) * N + ib,
                qp_tile + off,
                qp_tile + off + nb,
                qp_tile + off + 2 * nb,
                qp_tile + off + 3 * nb,
                ts_tile + off,
                ts_tile + off + nb,
                ts_tile + off + 2 * nb,
                ts_tile + off + 3 * nb,
                tdz_tile + off,
                tdz_tile + off + nb,
                tdz_tile + off + 2 * nb,
                tdz_tile + off + 3 * nb,
                t2_tile + off,
                t2_tile + off + nb,
                t2_tile + off + 2 * nb,
                t2_tile + off + 3 * nb,
                C_acc, Z0_acc, nb,
            )

        for r in range(n4 << 2, c):
            off = r * nb
            _q_fill_1snp_f64(
                G + <size_t>((j0 + r) >> 2) * N + ib,
                <uint8_t>(((j0 + r) & 3) << 1),
                qp_tile + off, ts_tile + off, tdz_tile + off, t2_tile + off,
                C_acc, Z0_acc, nb,
            )

        for r in range(c):
            pj = P + <size_t>(j0 + r) * K
            pp = P_pairs_tile + <size_t>r * npair
            pr = 0
            for k in range(K):
                pk = pj[k]
                for k2 in range(k, K):
                    pp[pr] = pk * pj[k2]
                    pr += 1

        adm_dgemm_tn(npair, nb, c, 1.0, P_pairs_tile, npair, ts_tile, nb, 1.0, H_acc, nb)
        adm_dgemm_tn(K, nb, c, 1.0, P + <size_t>j0 * K, K, tdz_tile, nb, 1.0, Z_acc, nb)
        adm_dgemm_tn(K, nb, c, 1.0, P + <size_t>j0 * K, K, t2_tile, nb, 1.0, A_acc, nb)

        j0 += Q_SNP_TILE

    for li in range(nb):
        if XtX != NULL and Xtz != NULL:
            xtx = XtX + <size_t>(ib + li) * K * K
            xtz = Xtz + <size_t>(ib + li) * K
        elif Q_next != NULL:
            xtx = xtx_local
            xtz = xtz_local
        else:
            continue
        pr = 0
        for k in range(K):
            xtz[k] = Z_acc[k * nb + li] + Z0_acc[li]
            for k2 in range(k, K):
                v = H_acc[pr * nb + li] - A_acc[k * nb + li] - A_acc[k2 * nb + li] + C_acc[li]
                xtx[k * K + k2] = v
                xtx[k2 * K + k] = v
                pr += 1
        if Q_next != NULL:
            qi = Q + <size_t>(ib + li) * K
            q_out = Q_next + <size_t>(ib + li) * K
            create_tableau_simplex_local(tableau, xtx, xtz, qi, v_kk, K, tmp_k, tmp_k2)
            quadratic_program_local(delta, tableau, qi, K, 1, d_buf, tmp_buf, swept_ws)
            for k in range(K):
                q_out[k] = qi[k] + delta[k]
            project_q_simplex_row_idx(q_out, K, 1e-5, idx_ws)


cdef int _run_q_sqp(const uint8_t* G, const double* Q, double* Q_next, const double* P,
                    double* XtX_q, double* Xtz_q, const double* v_kk,
                    int M, int N, int K, int B) noexcept nogil:
    cdef:
        int ib, i_end, blk, n_blocks, B_eff, B_par, q_c_max, prev_blas
        int n_threads = openmp.omp_get_max_threads()
        int npair = K * (K + 1) // 2
        int sz = K + 2
        int alloc_err = 0
        int* err_ptr = &alloc_err
        bint ok
        size_t ws_doubles, qp_doubles
        double* ws
        uint8_t* swept_ws
        int* idx_ws

    if M <= 0 or N <= 0 or K <= 0:
        return 0

    if n_threads < 1:
        n_threads = 1

    B_eff = B if B < 128 else 128
    B_par = N // (4 * n_threads)
    B_par = (B_par // 8) * 8
    if B_par < 8:
        B_par = 8
    if B_eff > B_par:
        B_eff = B_par
    if B_eff > N:
        B_eff = N
    if B_eff < 1:
        B_eff = 1

    n_blocks = (N + B_eff - 1) // B_eff
    q_c_max = M if M < Q_SNP_TILE else Q_SNP_TILE
    qp_doubles = (<size_t>sz * sz + <size_t>2 * sz + <size_t>3 * K) if Q_next != NULL else 0
    if Q_next != NULL and (XtX_q == NULL or Xtz_q == NULL):
        qp_doubles += <size_t>K * K + <size_t>K
    ws_doubles = (
        <size_t>(npair + 3 * K + 2) * B_eff
        + <size_t>4 * q_c_max * B_eff
        + <size_t>q_c_max * npair
        + qp_doubles
    )

    prev_blas = adm_blas_begin_single_thread()
    with nogil, parallel():
        ws = <double*>malloc(ws_doubles * sizeof(double))
        swept_ws = <uint8_t*>malloc(<size_t>K * sizeof(uint8_t)) if Q_next != NULL else NULL
        idx_ws = <int*>malloc(<size_t>K * sizeof(int)) if Q_next != NULL else NULL
        ok = (ws != NULL) and (Q_next == NULL or (swept_ws != NULL and idx_ws != NULL))
        if not ok:
            err_ptr[0] = 1
        for blk in prange(n_blocks, schedule='dynamic', chunksize=1):
            if ok:
                ib = blk * B_eff
                i_end = ib + B_eff
                if i_end > N:
                    i_end = N
                _grad_hess_q_tile(G, Q, P, XtX_q, Xtz_q, Q_next, v_kk,
                                  ws, swept_ws, idx_ws, ib, i_end, M, N, K, q_c_max)
        free(idx_ws)
        free(swept_ws)
        free(ws)
    adm_blas_end_single_thread(prev_blas)
    return -1 if alloc_err else 0


cpdef void compute_grad_hess_Q(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                               double[:,:,::1] XtX_q, double[:,::1] Xtz_q,
                               int M, int N, int K, int B):
    if XtX_q is None or Xtz_q is None:
        raise ValueError("XtX_q and Xtz_q are required for compute_grad_hess_Q")
    cdef int rc = _run_q_sqp(&G[0, 0], &Q[0, 0], NULL, &P[0, 0], &XtX_q[0, 0, 0], &Xtz_q[0, 0], NULL, M, N, K, B)
    if rc != 0:
        raise MemoryError("Out of memory allocating SQP Q gradient/Hessian workspace")


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
                        const double[:,::1] v_kk, int M, int N, int K, int B):
    cdef:
        double* xtx_ptr = &XtX_q[0, 0, 0] if XtX_q is not None else NULL
        double* xtz_ptr = &Xtz_q[0, 0] if Xtz_q is not None else NULL
        int rc = _run_q_sqp(&G[0, 0], &Q[0, 0], &Q_next[0, 0], &P[0, 0], xtx_ptr, xtz_ptr, &v_kk[0, 0], M, N, K, B)
    if rc != 0:
        raise MemoryError("Out of memory allocating SQP Q workspace")


cdef void _update_p_tile(const uint8_t* G, const double* Q, const double* Qt, const double* Q_pairs,
                         const double* P, double* P_next, double* ws, uint8_t* swept_ws,
                         int j0, int c, int N, int K, int c_max, int n_max) noexcept nogil:
    cdef:
        int npair = K * (K + 1) // 2
        int sz = K + 1
        int i0, n, n4, r4, r, k, k2, pr, off
        double beta, v
        const double* hp
        const double* gp
        const double* pj
        double* p_out
        double* qp_tile = ws
        double* ts_tile = qp_tile + <size_t>c_max * n_max
        double* tdz_tile = ts_tile + <size_t>c_max * n_max
        double* H_p_tile = tdz_tile + <size_t>c_max * n_max
        double* g_p_tile = H_p_tile + <size_t>c_max * npair
        double* tableau = g_p_tile + <size_t>c_max * K
        double* d_buf = tableau + <size_t>sz * sz
        double* tmp_buf = d_buf + sz
        double* delta = tmp_buf + sz

    i0 = 0
    while i0 < N:
        n = N - i0
        if n > P_IND_TILE:
            n = P_IND_TILE
        beta = 0.0 if i0 == 0 else 1.0

        adm_dgemm_nn(c, n, K, 1.0, P + <size_t>j0 * K, K, Qt + i0, N, 0.0, qp_tile, n)

        n4 = c >> 2
        for r4 in range(n4):
            off = (r4 << 2) * n
            adm_sqp_fill_4snps_f64(
                G + <size_t>((j0 >> 2) + r4) * N + i0,
                qp_tile + off,
                qp_tile + off + n,
                qp_tile + off + 2 * n,
                qp_tile + off + 3 * n,
                ts_tile + off,
                ts_tile + off + n,
                ts_tile + off + 2 * n,
                ts_tile + off + 3 * n,
                tdz_tile + off,
                tdz_tile + off + n,
                tdz_tile + off + 2 * n,
                tdz_tile + off + 3 * n,
                n,
            )

        for r in range(n4 << 2, c):
            off = r * n
            adm_sqp_fill_1snp_f64(
                G + <size_t>((j0 + r) >> 2) * N + i0,
                <uint8_t>(((j0 + r) & 3) << 1),
                qp_tile + off, ts_tile + off, tdz_tile + off, n,
            )

        adm_dgemm_nn(c, npair, n, 1.0, ts_tile, n,
                     Q_pairs + <size_t>i0 * npair, npair, beta, H_p_tile, npair)
        adm_dgemm_nn(c, K, n, 1.0, tdz_tile, n,
                     Q + <size_t>i0 * K, K, beta, g_p_tile, K)

        i0 += P_IND_TILE

    for r in range(c):
        hp = H_p_tile + <size_t>r * npair
        gp = g_p_tile + <size_t>r * K
        pj = P + <size_t>(j0 + r) * K
        p_out = P_next + <size_t>(j0 + r) * K
        pr = 0
        for k in range(K):
            tableau[k * sz + K] = gp[k]
            tableau[K * sz + k] = gp[k]
            for k2 in range(k, K):
                v = hp[pr]
                tableau[k * sz + k2] = v
                tableau[k2 * sz + k] = v
                pr += 1
        tableau[K * sz + K] = 0.0
        quadratic_program_local(delta, tableau, pj, K, 0, d_buf, tmp_buf, swept_ws)
        for k in range(K):
            p_out[k] = pj[k] + delta[k]
        project_p_box_row(p_out, K, 1e-5)


cpdef void update_p_sqp(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                        double[:,::1] P_next, int M, int N, int K):
    cdef:
        int i, k, k2, pr, tile, n_p_tiles, j0, c, c_max, n_max, prev_blas
        int alloc_err = 0
        int* err_ptr = &alloc_err
        bint ok
        int npair = K * (K + 1) // 2
        int sz = K + 1
        double qik
        const double* qi
        double* qp_row
        double* Qt
        double* Q_pairs
        double* ws
        uint8_t* swept_ws
        size_t ws_doubles

    if M <= 0 or N <= 0 or K <= 0:
        return

    Qt = <double*>malloc(<size_t>K * N * sizeof(double))
    Q_pairs = <double*>malloc(<size_t>N * npair * sizeof(double))
    if Qt == NULL or Q_pairs == NULL:
        free(Q_pairs)
        free(Qt)
        raise MemoryError("Out of memory allocating SQP P shared buffers")

    for i in range(N):
        qi = &Q[i, 0]
        qp_row = Q_pairs + <size_t>i * npair
        pr = 0
        for k in range(K):
            qik = qi[k]
            Qt[<size_t>k * N + i] = qik
            for k2 in range(k, K):
                qp_row[pr] = qik * qi[k2]
                pr += 1

    n_p_tiles = (M + P_SNP_TILE - 1) // P_SNP_TILE
    c_max = M if M < P_SNP_TILE else P_SNP_TILE
    n_max = N if N < P_IND_TILE else P_IND_TILE
    ws_doubles = (
        <size_t>3 * c_max * n_max
        + <size_t>c_max * (npair + K)
        + <size_t>sz * sz + <size_t>2 * sz + <size_t>K
    )

    prev_blas = adm_blas_begin_single_thread()
    with nogil, parallel():
        ws = <double*>malloc(ws_doubles * sizeof(double))
        swept_ws = <uint8_t*>malloc(<size_t>K * sizeof(uint8_t))
        ok = (ws != NULL and swept_ws != NULL)
        if not ok:
            err_ptr[0] = 1
        for tile in prange(n_p_tiles, schedule='dynamic', chunksize=1):
            if ok:
                j0 = tile * P_SNP_TILE
                c = M - j0
                if c > P_SNP_TILE:
                    c = P_SNP_TILE
                _update_p_tile(&G[0, 0], &Q[0, 0], Qt, Q_pairs, &P[0, 0], &P_next[0, 0],
                               ws, swept_ws, j0, c, N, K, c_max, n_max)
        free(swept_ws)
        free(ws)
    adm_blas_end_single_thread(prev_blas)

    free(Q_pairs)
    free(Qt)
    if alloc_err:
        raise MemoryError("Out of memory allocating SQP P thread workspace")
