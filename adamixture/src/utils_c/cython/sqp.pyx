# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
cimport openmp
from libc.math cimport fabs, fmax, fmin, sqrt
from libc.stdint cimport uint8_t
from libc.stdlib cimport free, malloc
from libc.string cimport memset

cdef extern from "blas_helper.h" nogil:
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
    void adm_sgemm_nn(int M, int N, int K, float alpha,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc)
    void adm_sgemm_tn(int M, int N, int K, float alpha,
                      const float* A, int lda,
                      const float* B, int ldb,
                      float beta, float* C, int ldc)
    void adm_sqp_fill_4snps_mixed(const uint8_t* g_row,
                                  const double* qp0, const double* qp1,
                                  const double* qp2, const double* qp3,
                                  float* ts0, float* ts1,
                                  float* ts2, float* ts3,
                                  double* tdz0, double* tdz1,
                                  double* tdz2, double* tdz3,
                                  int n)
    void adm_sqp_fill_1snp_mixed(const uint8_t* g_row, uint8_t shift,
                                 const double* qp,
                                 float* ts, double* tdz,
                                 int n)


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
    cdef int idx_stack[256]
    cdef int* idx_heap
    if n <= 256:
        project_q_simplex_row_idx(b, n, pseudocount, idx_stack)
    else:
        idx_heap = <int*>malloc(<size_t>n * sizeof(int))
        project_q_simplex_row_idx(b, n, pseudocount, idx_heap)
        free(idx_heap)


cdef void project_p_box_row(double* b, int n, double pseudocount) noexcept nogil:
    cdef int i
    for i in range(n):
        b[i] = fmin(fmax(b[i], pseudocount), 1.0 - pseudocount)


cdef int _cholesky_solve_dense(double* L, const double* A, int lda,
                               const double* rhs, double* sol, int n,
                               double rel_tol) noexcept nogil:
    """Factors SPD matrix A (n x n, stride lda) into L L^T and solves L L^T sol = rhs."""
    cdef int i, j, k
    cdef double s, a_jj, d, inv_d

    for j in range(n):
        s = 0.0
        for k in range(j):
            s += L[j * n + k] * L[j * n + k]
        a_jj = A[j * lda + j]
        d = a_jj - s
        if d <= 0.0 or (rel_tol > 0.0 and d <= rel_tol * a_jj):
            return -1
        L[j * n + j] = sqrt(d)
        inv_d = 1.0 / L[j * n + j]
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += L[i * n + k] * L[j * n + k]
            L[i * n + j] = (A[i * lda + j] - s) * inv_d

    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i * n + k] * sol[k]
        sol[i] = (rhs[i] - s) / L[i * n + i]

    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k * n + i] * sol[k]
        sol[i] = (sol[i] - s) / L[i * n + i]

    return 0


cdef void _solve_simplex_qp_cholesky(const double* H, const double* r,
                                     const double* q0, double* out, int K,
                                     double* d_ws, int* i_ws) noexcept nogil:
    """Solves min_{x >= 0, sum x = 1} 0.5 (x - q0)^T H (x - q0) - r^T (x - q0)
    via primal active set with reduced null-space Cholesky solves."""
    cdef:
        double* x = d_ws
        double* x_star = x + K
        double* g_cur = x_star + K
        double* rhs = g_cur + K
        double* d_sol = rhs + K
        double* H_red = d_sol + K
        double* L = H_red + K * K
        int* free_mask = i_ws
        int* F = free_mask + K
        int* idx_buf = F + K
        int k, l, a, b, it, m, rdim, p_idx, ia, ib_, k0, drop_k, best_k, n_free, rc
        double tr_H, sum_q0, inv_s, c0, h_pp, h_iap, g_p, val, min_val, sum_d
        double g0, gk, lam, best_s, s_k, alpha, denom, step, tr_red, reg

    tr_H = 0.0
    sum_q0 = 0.0
    for k in range(K):
        tr_H += H[k * K + k]
        sum_q0 += q0[k]

    if sum_q0 > 0.0:
        inv_s = 1.0 / sum_q0
        for k in range(K):
            x[k] = q0[k] * inv_s
    else:
        inv_s = 1.0 / K
        for k in range(K):
            x[k] = inv_s

    n_free = 0
    sum_q0 = 0.0
    for k in range(K):
        if x[k] > 0.0:
            free_mask[k] = 1
            sum_q0 += x[k]
            n_free += 1
        else:
            x[k] = 0.0
            free_mask[k] = 0

    if n_free == 0:
        x[0] = 1.0
        free_mask[0] = 1
    elif fabs(sum_q0 - 1.0) > 1e-12:
        inv_s = 1.0 / sum_q0
        for k in range(K):
            x[k] *= inv_s

    if tr_H > 0.0:
        for it in range(4 * K):
            m = 0
            for k in range(K):
                if free_mask[k]:
                    F[m] = k
                    m += 1
            if m == 0:
                x[0] = 1.0
                free_mask[0] = 1
                F[0] = 0
                m = 1

            if m == 1:
                for k in range(K):
                    x_star[k] = 0.0
                x_star[F[0]] = 1.0
            else:
                p_idx = F[m - 1]
                rdim = m - 1
                for l in range(K):
                    x_star[l] = x[l] - q0[l]
                c0 = 1.0
                for a in range(m):
                    ia = F[a]
                    c0 -= x[ia]
                    gk = -r[ia]
                    for l in range(K):
                        gk += H[ia * K + l] * x_star[l]
                    g_cur[ia] = gk

                h_pp = H[p_idx * K + p_idx]
                g_p = g_cur[p_idx]
                for a in range(rdim):
                    ia = F[a]
                    h_iap = H[ia * K + p_idx]
                    rhs[a] = -(g_cur[ia] - g_p) - (h_iap - h_pp) * c0
                    for b in range(a + 1):
                        ib_ = F[b]
                        val = H[ia * K + ib_] - h_iap - H[p_idx * K + ib_] + h_pp
                        H_red[a * rdim + b] = val
                        H_red[b * rdim + a] = val

                rc = _cholesky_solve_dense(L, H_red, rdim, rhs, d_sol, rdim, 1e-6)
                if rc < 0:
                    tr_red = 0.0
                    for a in range(rdim):
                        tr_red += H_red[a * rdim + a]
                    reg = 1e-6 * (tr_red / rdim + 1.0)
                    for a in range(rdim):
                        H_red[a * rdim + a] += reg
                    rc = _cholesky_solve_dense(L, H_red, rdim, rhs, d_sol, rdim, 1e-12)
                    if rc < 0:
                        break

                for k in range(K):
                    x_star[k] = 0.0
                sum_d = 0.0
                for a in range(rdim):
                    ia = F[a]
                    x_star[ia] = x[ia] + d_sol[a]
                    sum_d += d_sol[a]
                x_star[p_idx] = x[p_idx] + c0 - sum_d

            min_val = 1.0
            for a in range(m):
                val = x_star[F[a]]
                if val < min_val:
                    min_val = val

            if min_val >= -1e-12:
                for a in range(m):
                    k = F[a]
                    x[k] = x_star[k] if x_star[k] > 0.0 else 0.0
                if m == K:
                    break
                for l in range(K):
                    x_star[l] = x[l] - q0[l]
                k0 = F[0]
                g0 = -r[k0]
                for l in range(K):
                    g0 += H[k0 * K + l] * x_star[l]
                lam = -g0
                best_k = -1
                best_s = -1e-9
                for k in range(K):
                    if free_mask[k] == 0:
                        gk = -r[k]
                        for l in range(K):
                            gk += H[k * K + l] * x_star[l]
                        s_k = gk + lam
                        if s_k < best_s:
                            best_s = s_k
                            best_k = k
                if best_k == -1:
                    break
                free_mask[best_k] = 1
            else:
                alpha = 1.0
                drop_k = -1
                for a in range(m):
                    k = F[a]
                    if x_star[k] < 0.0:
                        denom = x[k] - x_star[k]
                        if denom > 0.0:
                            step = x[k] / denom
                            if step < alpha:
                                alpha = step
                                drop_k = k
                if drop_k == -1:
                    break
                if alpha < 0.0:
                    alpha = 0.0
                for a in range(m):
                    k = F[a]
                    val = x[k] + alpha * (x_star[k] - x[k])
                    x[k] = val if val > 0.0 else 0.0
                x[drop_k] = 0.0
                free_mask[drop_k] = 0

    for k in range(K):
        out[k] = x[k]
    project_q_simplex_row_idx(out, K, 1e-5, idx_buf)


cdef void _solve_box_qp_cholesky(const double* H, const double* r,
                                 const double* p0, double* out, int K,
                                 double* d_ws, int* i_ws) noexcept nogil:
    """Solves min_{-p0 <= d <= 1 - p0} 0.5 d^T H d - r^T d
    via warm-started primal active set with Cholesky subproblem solves."""
    cdef:
        double* d = d_ws
        double* d_star = d + K
        double* rhs = d_star + K
        double* sol = rhs + K
        double* L = sol + K
        double* H_sub = L + K * K
        int* state = i_ws
        int* F = state + K
        int k, l, a, b, it, m, ia, ib_, rc, block_k, block_st, best_k
        double val, h_ab, alpha, lb, ub, denom, step, gk, best_viol

    for k in range(K):
        if H[k * K + k] <= 0.0:
            for l in range(K):
                out[l] = p0[l]
            project_p_box_row(out, K, 1e-5)
            return

    for k in range(K):
        if p0[k] <= 1e-5 and r[k] <= 0.0:
            state[k] = -1
            d[k] = -p0[k]
        elif p0[k] >= 1.0 - 1e-5 and r[k] >= 0.0:
            state[k] = 1
            d[k] = 1.0 - p0[k]
        else:
            state[k] = 0
            d[k] = 0.0

    for it in range(4 * K):
        m = 0
        for k in range(K):
            if state[k] == 0:
                F[m] = k
                m += 1
        for k in range(K):
            d_star[k] = d[k]

        if m > 0:
            for a in range(m):
                ia = F[a]
                val = r[ia]
                if m < K:
                    for l in range(K):
                        if state[l] != 0:
                            val -= H[ia * K + l] * d[l]
                rhs[a] = val
                for b in range(a + 1):
                    ib_ = F[b]
                    h_ab = H[ia * K + ib_]
                    H_sub[a * m + b] = h_ab
                    H_sub[b * m + a] = h_ab

            rc = _cholesky_solve_dense(L, H_sub, m, rhs, sol, m, 1e-7)
            if rc < 0:
                break
            for a in range(m):
                d_star[F[a]] = sol[a]

        alpha = 1.0
        block_k = -1
        block_st = 0
        for a in range(m):
            k = F[a]
            lb = -p0[k]
            ub = 1.0 - p0[k]
            if d_star[k] < lb - 1e-12:
                denom = d_star[k] - d[k]
                if denom < 0.0:
                    step = (lb - d[k]) / denom
                    if step < 0.0:
                        step = 0.0
                    if step < alpha:
                        alpha = step
                        block_k = k
                        block_st = -1
            elif d_star[k] > ub + 1e-12:
                denom = d_star[k] - d[k]
                if denom > 0.0:
                    step = (ub - d[k]) / denom
                    if step < 0.0:
                        step = 0.0
                    if step < alpha:
                        alpha = step
                        block_k = k
                        block_st = 1

        if block_k != -1:
            for a in range(m):
                k = F[a]
                d[k] = d[k] + alpha * (d_star[k] - d[k])
            d[block_k] = -p0[block_k] if block_st == -1 else (1.0 - p0[block_k])
            state[block_k] = block_st
        else:
            for a in range(m):
                k = F[a]
                d[k] = d_star[k]
            if m == K:
                break
            best_k = -1
            best_viol = 1e-9
            for k in range(K):
                if state[k] != 0:
                    gk = -r[k]
                    for l in range(K):
                        gk += H[k * K + l] * d[l]
                    if state[k] == -1 and -gk > best_viol:
                        best_viol = -gk
                        best_k = k
                    elif state[k] == 1 and gk > best_viol:
                        best_viol = gk
                        best_k = k
            if best_k == -1:
                break
            state[best_k] = 0

    for k in range(K):
        out[k] = p0[k] + d[k]
    project_p_box_row(out, K, 1e-5)


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
                            double* ws, float* ws_f32, int* qp_i_ws,
                            int ib, int i_end, int M, int N, int K,
                            int q_c_max, bint unit_simplex) noexcept nogil:
    cdef:
        int nb = i_end - ib
        int npair = K * (K + 1) // 2
        int j0, c, n4, r4, r, k, k2, li, pr, off, idx, n_acc
        double pk, v
        float pk_f
        const double* pj
        double* pp
        float* pp_f
        double* xtx
        double* xtz
        double* Qt_block = ws
        double* H_acc = Qt_block + <size_t>K * nb
        double* Z_acc = H_acc + <size_t>npair * nb
        double* qp_tile = Z_acc + <size_t>K * nb
        double* tdz_tile = qp_tile + <size_t>q_c_max * nb
        double* extra = tdz_tile + <size_t>q_c_max * nb
        double* ts_tile = NULL if unit_simplex else extra
        double* P_pairs_tile = NULL if unit_simplex else (ts_tile + <size_t>q_c_max * nb)
        double* A_acc = NULL if unit_simplex else (P_pairs_tile + <size_t>q_c_max * npair)
        double* C_acc = NULL if unit_simplex else (A_acc + <size_t>K * nb)
        double* Z0_acc = NULL if unit_simplex else (C_acc + nb)
        double* t2_tile = NULL if unit_simplex else (Z0_acc + nb)
        double* qp_d_ws = extra if unit_simplex else (t2_tile + <size_t>q_c_max * nb)
        float* ts_f32 = ws_f32 if unit_simplex else NULL
        float* P_pairs_f32 = (ts_f32 + <size_t>q_c_max * nb) if unit_simplex else NULL
        float* H_tile_f32 = (P_pairs_f32 + <size_t>q_c_max * npair) if unit_simplex else NULL

    for k in range(K):
        for li in range(nb):
            Qt_block[k * nb + li] = Q[<size_t>(ib + li) * K + k]

    memset(H_acc, 0, <size_t>(npair + K) * nb * sizeof(double))
    if not unit_simplex:
        memset(A_acc, 0, <size_t>(K + 2) * nb * sizeof(double))

    n_acc = npair * nb
    j0 = 0
    while j0 < M:
        c = M - j0
        if c > Q_SNP_TILE:
            c = Q_SNP_TILE

        adm_dgemm_nn(c, nb, K, 1.0, P + <size_t>j0 * K, K, Qt_block, nb, 0.0, qp_tile, nb)

        n4 = c >> 2
        if unit_simplex:
            for r4 in range(n4):
                off = (r4 << 2) * nb
                adm_sqp_fill_4snps_mixed(
                    G + <size_t>((j0 >> 2) + r4) * N + ib,
                    qp_tile + off,
                    qp_tile + off + nb,
                    qp_tile + off + 2 * nb,
                    qp_tile + off + 3 * nb,
                    ts_f32 + off,
                    ts_f32 + off + nb,
                    ts_f32 + off + 2 * nb,
                    ts_f32 + off + 3 * nb,
                    tdz_tile + off,
                    tdz_tile + off + nb,
                    tdz_tile + off + 2 * nb,
                    tdz_tile + off + 3 * nb,
                    nb,
                )

            for r in range(n4 << 2, c):
                off = r * nb
                adm_sqp_fill_1snp_mixed(
                    G + <size_t>((j0 + r) >> 2) * N + ib,
                    <uint8_t>(((j0 + r) & 3) << 1),
                    qp_tile + off, ts_f32 + off, tdz_tile + off, nb,
                )

            for r in range(c):
                pj = P + <size_t>(j0 + r) * K
                pp_f = P_pairs_f32 + <size_t>r * npair
                pr = 0
                for k in range(K):
                    pk_f = <float>pj[k]
                    for k2 in range(k, K):
                        pp_f[pr] = pk_f * <float>pj[k2]
                        pr += 1

            adm_sgemm_tn(npair, nb, c, 1.0, P_pairs_f32, npair, ts_f32, nb, 0.0, H_tile_f32, nb)
            for idx in range(n_acc):
                H_acc[idx] += <double>H_tile_f32[idx]
            adm_dgemm_tn(K, nb, c, 1.0, P + <size_t>j0 * K, K, tdz_tile, nb, 1.0, Z_acc, nb)
        else:
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
        xtx = XtX + <size_t>(ib + li) * K * K
        xtz = Xtz + <size_t>(ib + li) * K
        pr = 0
        if unit_simplex:
            for k in range(K):
                xtz[k] = Z_acc[k * nb + li]
                for k2 in range(k, K):
                    v = H_acc[pr * nb + li]
                    xtx[k * K + k2] = v
                    xtx[k2 * K + k] = v
                    pr += 1
        else:
            for k in range(K):
                xtz[k] = Z_acc[k * nb + li] + Z0_acc[li]
                for k2 in range(k, K):
                    v = H_acc[pr * nb + li] - A_acc[k * nb + li] - A_acc[k2 * nb + li] + C_acc[li]
                    xtx[k * K + k2] = v
                    xtx[k2 * K + k] = v
                    pr += 1
        if Q_next != NULL:
            _solve_simplex_qp_cholesky(xtx, xtz, Q + <size_t>(ib + li) * K,
                                       Q_next + <size_t>(ib + li) * K, K, qp_d_ws, qp_i_ws)


cdef void _run_q_sqp(const uint8_t* G, const double* Q, double* Q_next, const double* P,
                     double* XtX_q, double* Xtz_q, int M, int N, int K, int B) noexcept nogil:
    cdef:
        int i, k, ib, i_end, blk, n_blocks, B_eff, B_par, q_c_max, prev_blas
        int n_threads = openmp.omp_get_max_threads()
        int npair = K * (K + 1) // 2
        bint unit_simplex = True
        double row_sum
        size_t ws_doubles, qp_doubles, f32_floats
        double* ws
        float* ws_f32
        int* qp_i_ws

    if M <= 0 or N <= 0 or K <= 0:
        return

    for i in range(N):
        row_sum = 0.0
        for k in range(K):
            row_sum += Q[<size_t>i * K + k]
        if fabs(row_sum - 1.0) > 1e-11:
            unit_simplex = False
            break

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
    qp_doubles = <size_t>(2 * K * K + 5 * K) if Q_next != NULL else 0
    ws_doubles = (
        <size_t>(npair + 2 * K) * B_eff
        + <size_t>2 * q_c_max * B_eff
        + qp_doubles
    )
    if unit_simplex:
        f32_floats = (
            <size_t>q_c_max * B_eff
            + <size_t>q_c_max * npair
            + <size_t>npair * B_eff
        )
    else:
        f32_floats = 0
        ws_doubles += <size_t>(2 * q_c_max + K + 2) * B_eff + <size_t>q_c_max * npair

    prev_blas = adm_blas_begin_single_thread()
    with nogil, parallel():
        ws = <double*>malloc(ws_doubles * sizeof(double))
        ws_f32 = <float*>malloc(f32_floats * sizeof(float)) if unit_simplex else NULL
        qp_i_ws = <int*>malloc(<size_t>(3 * K) * sizeof(int)) if Q_next != NULL else NULL
        for blk in prange(n_blocks, schedule='dynamic', chunksize=1):
            ib = blk * B_eff
            i_end = ib + B_eff
            if i_end > N:
                i_end = N
            _grad_hess_q_tile(G, Q, P, XtX_q, Xtz_q, Q_next, ws, ws_f32, qp_i_ws,
                              ib, i_end, M, N, K, q_c_max, unit_simplex)
        free(qp_i_ws)
        free(ws_f32)
        free(ws)
    adm_blas_end_single_thread(prev_blas)


cpdef void compute_grad_hess_Q(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                               double[:,:,::1] XtX_q, double[:,::1] Xtz_q,
                               int M, int N, int K, int B) noexcept nogil:
    _run_q_sqp(&G[0, 0], &Q[0, 0], NULL, &P[0, 0], &XtX_q[0, 0, 0], &Xtz_q[0, 0], M, N, K, B)


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
    _run_q_sqp(&G[0, 0], &Q[0, 0], &Q_next[0, 0], &P[0, 0], &XtX_q[0, 0, 0], &Xtz_q[0, 0], M, N, K, B)


cdef void _update_p_tile(const uint8_t* G, const double* Q, const double* Qt, const float* Q_pairs_f32,
                         const double* P, double* P_next, double* ws, float* ws_f32, int* qp_i_ws,
                         int j0, int c, int N, int K, int c_max, int n_max) noexcept nogil:
    cdef:
        int npair = K * (K + 1) // 2
        int i0, n, n4, r4, r, k, k2, pr, off, idx, n_hp
        double beta, v
        const double* hp
        double* qp_tile = ws
        double* tdz_tile = qp_tile + <size_t>c_max * n_max
        double* H_p_tile = tdz_tile + <size_t>c_max * n_max
        double* g_p_tile = H_p_tile + <size_t>c_max * npair
        double* H_mat = g_p_tile + <size_t>c_max * K
        double* qp_d_ws = H_mat + <size_t>K * K
        float* ts_f32 = ws_f32
        float* H_tile_f32 = ts_f32 + <size_t>c_max * n_max

    n_hp = c * npair
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
            adm_sqp_fill_4snps_mixed(
                G + <size_t>((j0 >> 2) + r4) * N + i0,
                qp_tile + off,
                qp_tile + off + n,
                qp_tile + off + 2 * n,
                qp_tile + off + 3 * n,
                ts_f32 + off,
                ts_f32 + off + n,
                ts_f32 + off + 2 * n,
                ts_f32 + off + 3 * n,
                tdz_tile + off,
                tdz_tile + off + n,
                tdz_tile + off + 2 * n,
                tdz_tile + off + 3 * n,
                n,
            )

        for r in range(n4 << 2, c):
            off = r * n
            adm_sqp_fill_1snp_mixed(
                G + <size_t>((j0 + r) >> 2) * N + i0,
                <uint8_t>(((j0 + r) & 3) << 1),
                qp_tile + off, ts_f32 + off, tdz_tile + off, n,
            )

        adm_sgemm_nn(c, npair, n, 1.0, ts_f32, n,
                     Q_pairs_f32 + <size_t>i0 * npair, npair, 0.0, H_tile_f32, npair)
        if i0 == 0:
            for idx in range(n_hp):
                H_p_tile[idx] = <double>H_tile_f32[idx]
        else:
            for idx in range(n_hp):
                H_p_tile[idx] += <double>H_tile_f32[idx]

        adm_dgemm_nn(c, K, n, 1.0, tdz_tile, n,
                     Q + <size_t>i0 * K, K, beta, g_p_tile, K)

        i0 += P_IND_TILE

    for r in range(c):
        hp = H_p_tile + <size_t>r * npair
        pr = 0
        for k in range(K):
            for k2 in range(k, K):
                v = hp[pr]
                H_mat[k * K + k2] = v
                H_mat[k2 * K + k] = v
                pr += 1
        _solve_box_qp_cholesky(H_mat, g_p_tile + <size_t>r * K,
                               P + <size_t>(j0 + r) * K, P_next + <size_t>(j0 + r) * K,
                               K, qp_d_ws, qp_i_ws)


cpdef void update_p_sqp(const uint8_t[:,::1] G, const double[:,::1] Q, const double[:,::1] P,
                        double[:,::1] P_next, int M, int N, int K) noexcept nogil:
    cdef:
        int i, k, k2, pr, tile, n_p_tiles, j0, c, c_max, n_max, prev_blas
        int npair = K * (K + 1) // 2
        double qik
        float qik_f
        const double* qi
        float* qp_row
        double* Qt
        float* Q_pairs_f32
        double* ws
        float* ws_f32
        int* qp_i_ws
        size_t ws_doubles, f32_floats

    if M <= 0 or N <= 0 or K <= 0:
        return

    Qt = <double*>malloc(<size_t>K * N * sizeof(double))
    Q_pairs_f32 = <float*>malloc(<size_t>N * npair * sizeof(float))

    for i in range(N):
        qi = &Q[i, 0]
        qp_row = Q_pairs_f32 + <size_t>i * npair
        pr = 0
        for k in range(K):
            qik = qi[k]
            qik_f = <float>qik
            Qt[<size_t>k * N + i] = qik
            for k2 in range(k, K):
                qp_row[pr] = qik_f * <float>qi[k2]
                pr += 1

    n_p_tiles = (M + P_SNP_TILE - 1) // P_SNP_TILE
    c_max = M if M < P_SNP_TILE else P_SNP_TILE
    n_max = N if N < P_IND_TILE else P_IND_TILE
    ws_doubles = (
        <size_t>2 * c_max * n_max
        + <size_t>c_max * (npair + K)
        + <size_t>3 * K * K + <size_t>4 * K
    )
    f32_floats = <size_t>c_max * n_max + <size_t>c_max * npair

    prev_blas = adm_blas_begin_single_thread()
    with nogil, parallel():
        ws = <double*>malloc(ws_doubles * sizeof(double))
        ws_f32 = <float*>malloc(f32_floats * sizeof(float))
        qp_i_ws = <int*>malloc(<size_t>(2 * K) * sizeof(int))
        for tile in prange(n_p_tiles, schedule='dynamic', chunksize=1):
            j0 = tile * P_SNP_TILE
            c = M - j0
            if c > P_SNP_TILE:
                c = P_SNP_TILE
            _update_p_tile(&G[0, 0], &Q[0, 0], Qt, Q_pairs_f32, &P[0, 0], &P_next[0, 0],
                           ws, ws_f32, qp_i_ws, j0, c, N, K, c_max, n_max)
        free(qp_i_ws)
        free(ws_f32)
        free(ws)
    adm_blas_end_single_thread(prev_blas)

    free(Q_pairs_f32)
    free(Qt)
