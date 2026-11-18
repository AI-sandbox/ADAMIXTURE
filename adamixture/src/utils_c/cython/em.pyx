# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
cimport openmp as omp
from cython.parallel import parallel, prange
from libc.math cimport fmax, fmin, log, pow, sqrt
from libc.stdint cimport uint8_t, uint64_t, uintptr_t
from libc.stdlib cimport calloc, free, malloc
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
    void adm_em_fill_4snps(const uint8_t* g_row,
                           const double* r0, const double* r1,
                           const double* r2, const double* r3,
                           double* a0, double* a1, double* a2, double* a3,
                           double* d0, double* d1, double* d2, double* d3,
                           double* b_acc, int n)
    void adm_em_fill_1snp(const uint8_t* g_row, uint8_t shift,
                          const double* r_in,
                          double* a_out, double* d_out,
                          double* b_acc, int n)
    void adm_em_fill_4snps_ll(const uint8_t* g_row,
                              const double* r0, const double* r1,
                              const double* r2, const double* r3,
                              double* a0, double* a1, double* a2, double* a3,
                              double* d0, double* d1, double* d2, double* d3,
                              double* b_acc, double* p4_out, int n)
    void adm_em_fill_1snp_ll(const uint8_t* g_row, uint8_t shift,
                             const double* r_in,
                             double* a_out, double* d_out,
                             double* b_acc, double* p1_out, int n)


cdef enum:
    EM_SNP_TILE = 128
    EM_IND_TILE = 1040
    EM_NUM_LOCKS = 64
    EM_RED_LIMIT_BYTES = 67108864  # 64 MiB


cdef inline double _clip_to_domain(const double value) noexcept nogil:
    """Clip a probability estimate to [1e-5, 1 - 1e-5] for numerical stability."""
    return fmin(fmax(value, 1e-5), 1.0 - 1e-5)


cdef inline size_t _round_up_8(size_t x) noexcept nogil:
    return (x + 7) & ~(<size_t>7)


cdef inline size_t _em_plan_layout(
    int M, int N, int K, int n_threads, bint count_q_bat,
    size_t* out_qt_doubles,
    size_t* out_ll_doubles,
    size_t* out_ws_thr_doubles,
    size_t* out_q_red_doubles,
    bint* out_use_lockfree,
) noexcept nogil:
    cdef:
        int c_max = M if M < EM_SNP_TILE else EM_SNP_TILE
        int n_max = N if N < EM_IND_TILE else EM_IND_TILE
        size_t per_ind_q = <size_t>K + 1 + (<size_t>1 if count_q_bat else 0)
        size_t red_doubles = <size_t>n_threads * per_ind_q * <size_t>N
        bint use_lockfree = (red_doubles * sizeof(double) <= <size_t>EM_RED_LIMIT_BYTES)
        size_t ws_raw = <size_t>3 * c_max * n_max + <size_t>2 * c_max * K
        size_t qt_d = _round_up_8(<size_t>K * N)
        size_t ll_d = _round_up_8(<size_t>n_threads)
        size_t ws_thr_d, q_red_d
    if not use_lockfree:
        ws_raw += per_ind_q * <size_t>n_max
    ws_thr_d = _round_up_8(ws_raw)
    q_red_d = _round_up_8(red_doubles) if use_lockfree else 0
    out_qt_doubles[0] = qt_d
    out_ll_doubles[0] = ll_d
    out_ws_thr_doubles[0] = ws_thr_d
    out_q_red_doubles[0] = q_red_d
    out_use_lockfree[0] = use_lockfree
    return (qt_d + ll_d + <size_t>n_threads * ws_thr_d + q_red_d) * sizeof(double) + 64


cpdef size_t em_workspace_bytes(int M, int N, int K, bint count_q_bat=False) noexcept nogil:
    """Return required reusable workspace size in bytes for CPU EM / Adam-EM kernels."""
    cdef:
        int n_threads
        size_t qt_d, ll_d, ws_thr_d, q_red_d
        bint use_lockfree
    if M <= 0 or N <= 0 or K <= 0:
        return 0
    n_threads = omp.omp_get_max_threads()
    if n_threads < 1:
        n_threads = 1
    return _em_plan_layout(
        M, N, K, n_threads, count_q_bat,
        &qt_d, &ll_d, &ws_thr_d, &q_red_d, &use_lockfree,
    )


cdef inline void _count_packed_4snps(const uint8_t* g_row, double* cnt_acc, int n) noexcept nogil:
    cdef int li
    cdef uint8_t b
    for li in range(n):
        b = g_row[li]
        cnt_acc[li] += 2.0 * <double>(
            ((b & 0x03) != 3)
            + (((b >> 2) & 0x03) != 3)
            + (((b >> 4) & 0x03) != 3)
            + (((b >> 6) & 0x03) != 3)
        )


cdef inline void _count_packed_1snp(const uint8_t* g_row, uint8_t shift, double* cnt_acc, int n) noexcept nogil:
    cdef int li
    for li in range(n):
        cnt_acc[li] += 2.0 * <double>(((g_row[li] >> shift) & 0x03) != 3)


cdef inline void _cnt_packed_bytes(
    const uint8_t* g_row,
    uint64_t* cnt_thr,
    Py_ssize_t valid,
    int N,
) noexcept nogil:
    cdef:
        Py_ssize_t i
        uint8_t b
    if valid >= 4:
        for i in range(N):
            b = g_row[i]
            cnt_thr[i] += <uint64_t>(
                ((b & 0x03) != 3)
                + (((b >> 2) & 0x03) != 3)
                + (((b >> 4) & 0x03) != 3)
                + (((b >> 6) & 0x03) != 3)
            )
    else:
        for i in range(N):
            b = g_row[i]
            if valid >= 1:
                cnt_thr[i] += <uint64_t>((b & 0x03) != 3)
            if valid >= 2:
                cnt_thr[i] += <uint64_t>(((b >> 2) & 0x03) != 3)
            if valid >= 3:
                cnt_thr[i] += <uint64_t>(((b >> 4) & 0x03) != 3)


cdef inline double _reduce_inv_q_bat_col(
    const uint64_t* cnt_all,
    Py_ssize_t col,
    int n_threads,
    int N,
) noexcept nogil:
    cdef:
        int t
        uint64_t total_i = 0
    for t in range(n_threads):
        total_i += cnt_all[<size_t>t * <size_t>N + col]
    return 1.0 / (2.0 * <double>total_i) if total_i > 0 else 0.0


cpdef void precompute_inv_q_bat_packed(
    const unsigned char[:,::1] G,
    double[::1] inv_q_bat,
    const int M,
    const int N,
):
    """Precompute ``1 / (2 * valid_snps[i])`` per individual from 2-bit packed genotypes."""
    cdef:
        Py_ssize_t M_bytes, j4, valid, i
        int n_threads
        uint64_t* cnt_all
        const uint8_t* G_ptr
        double* inv_ptr

    if M <= 0 or N <= 0:
        return

    n_threads = omp.omp_get_max_threads()
    if n_threads < 1:
        n_threads = 1

    cnt_all = <uint64_t*>calloc(<size_t>n_threads * <size_t>N, sizeof(uint64_t))
    if cnt_all == NULL:
        raise MemoryError("Out of memory allocating genotype count buffer")

    M_bytes = (<Py_ssize_t>M + 3) // 4
    G_ptr = &G[0, 0]
    inv_ptr = &inv_q_bat[0]

    with nogil, parallel(num_threads=n_threads):
        for j4 in prange(M_bytes, schedule='static'):
            valid = <Py_ssize_t>M - (j4 << 2)
            _cnt_packed_bytes(
                G_ptr + j4 * <Py_ssize_t>N,
                cnt_all + <size_t>omp.omp_get_thread_num() * <size_t>N,
                valid,
                N,
            )

        for i in prange(N, schedule='static'):
            inv_ptr[i] = _reduce_inv_q_bat_col(cnt_all, i, n_threads, N)

    free(cnt_all)


cdef inline double _log_pbuf(const double* p_buf, int n) noexcept nogil:
    cdef:
        int li = 0
        double acc = 0.0
    while li + 8 <= n:
        acc += log(
            p_buf[li] * p_buf[li + 1] * p_buf[li + 2] * p_buf[li + 3]
            * p_buf[li + 4] * p_buf[li + 5] * p_buf[li + 6] * p_buf[li + 7]
        )
        li += 8
    while li < n:
        acc += log(p_buf[li])
        li += 1
    return acc


cdef double _em_snp_tile(
    const uint8_t* G,
    double* P,
    double* P_out,
    const double* Q,
    const double* Qt,
    double* ws_thr,
    double* qt_red,
    double* b_red,
    double* cnt_red,
    double* Q_T_global,
    double* q_bat_global,
    omp.omp_lock_t* locks,
    bint use_lockfree,
    bint count_q_bat,
    bint do_adam_p,
    bint compute_ll,
    double* m_P,
    double* v_P,
    double beta1,
    double one_m_beta1,
    double beta2,
    double one_m_beta2,
    double step_scale_p,
    double eps_scaled_p,
    int j0,
    int c,
    int N,
    int K,
    int c_max,
    int n_max,
) noexcept nogil:
    cdef:
        int i0, n, n4, r4, r, k, li, blk_idx, lock_id
        double beta_p, ak, dk, bk, pk, denom, p_em, delta, m_new, v_new, step_val
        double tile_ll = 0.0
        uint8_t shift
        const uint8_t* g_row
        double* r_tile = ws_thr
        double* ad_tile = r_tile + <size_t>c_max * n_max
        double* AD_p_tile = ad_tile + <size_t>2 * c_max * n_max
        double* extra = AD_p_tile + <size_t>2 * c_max * K
        double* Qt_tile = NULL if use_lockfree else extra
        double* b_tile = NULL if use_lockfree else (Qt_tile + <size_t>K * n_max)
        double* cnt_tile = NULL if (use_lockfree or not count_q_bat) else (b_tile + n_max)
        double* a_tile
        double* d_tile
        double* b_acc
        double* cnt_acc
        double* qt_dst
        int qt_ld
        double* A_p_tile
        double* D_p_tile
        double* p_row
        double* po_row
        double* mp_row
        double* vp_row
        double* a_row
        double* d_row
        double* qt_row
        double* qb_row
        double p_buf[1040]

    a_tile = ad_tile
    d_tile = ad_tile + <size_t>c * n_max  # updated per i0 since actual ldc is n

    i0 = 0
    while i0 < N:
        n = N - i0
        if n > EM_IND_TILE:
            n = EM_IND_TILE
        beta_p = 0.0 if i0 == 0 else 1.0

        adm_dgemm_nn(
            c, n, K, 1.0,
            P + <size_t>j0 * K, K,
            Qt + i0, N,
            0.0, r_tile, n,
        )

        a_tile = ad_tile
        d_tile = ad_tile + <size_t>c * n

        if use_lockfree:
            b_acc = b_red + i0
            cnt_acc = (cnt_red + i0) if count_q_bat else NULL
            qt_dst = qt_red + i0
            qt_ld = N
        else:
            b_acc = b_tile
            memset(b_acc, 0, <size_t>n * sizeof(double))
            if count_q_bat:
                cnt_acc = cnt_tile
                memset(cnt_acc, 0, <size_t>n * sizeof(double))
            else:
                cnt_acc = NULL
            qt_dst = Qt_tile
            qt_ld = n

        n4 = c >> 2
        if compute_ll:
            for r4 in range(n4):
                g_row = G + <size_t>((j0 >> 2) + r4) * N + i0
                adm_em_fill_4snps_ll(
                    g_row,
                    r_tile + (r4 << 2) * n,
                    r_tile + ((r4 << 2) + 1) * n,
                    r_tile + ((r4 << 2) + 2) * n,
                    r_tile + ((r4 << 2) + 3) * n,
                    a_tile + (r4 << 2) * n,
                    a_tile + ((r4 << 2) + 1) * n,
                    a_tile + ((r4 << 2) + 2) * n,
                    a_tile + ((r4 << 2) + 3) * n,
                    d_tile + (r4 << 2) * n,
                    d_tile + ((r4 << 2) + 1) * n,
                    d_tile + ((r4 << 2) + 2) * n,
                    d_tile + ((r4 << 2) + 3) * n,
                    b_acc,
                    p_buf,
                    n,
                )
                tile_ll += _log_pbuf(p_buf, n)
                if count_q_bat:
                    _count_packed_4snps(g_row, cnt_acc, n)

            for r in range(n4 << 2, c):
                g_row = G + <size_t>((j0 + r) >> 2) * N + i0
                shift = <uint8_t>(((j0 + r) & 3) << 1)
                adm_em_fill_1snp_ll(
                    g_row, shift,
                    r_tile + r * n,
                    a_tile + r * n,
                    d_tile + r * n,
                    b_acc,
                    p_buf,
                    n,
                )
                tile_ll += _log_pbuf(p_buf, n)
                if count_q_bat:
                    _count_packed_1snp(g_row, shift, cnt_acc, n)
        else:
            for r4 in range(n4):
                g_row = G + <size_t>((j0 >> 2) + r4) * N + i0
                adm_em_fill_4snps(
                    g_row,
                    r_tile + (r4 << 2) * n,
                    r_tile + ((r4 << 2) + 1) * n,
                    r_tile + ((r4 << 2) + 2) * n,
                    r_tile + ((r4 << 2) + 3) * n,
                    a_tile + (r4 << 2) * n,
                    a_tile + ((r4 << 2) + 1) * n,
                    a_tile + ((r4 << 2) + 2) * n,
                    a_tile + ((r4 << 2) + 3) * n,
                    d_tile + (r4 << 2) * n,
                    d_tile + ((r4 << 2) + 1) * n,
                    d_tile + ((r4 << 2) + 2) * n,
                    d_tile + ((r4 << 2) + 3) * n,
                    b_acc,
                    n,
                )
                if count_q_bat:
                    _count_packed_4snps(g_row, cnt_acc, n)

            for r in range(n4 << 2, c):
                g_row = G + <size_t>((j0 + r) >> 2) * N + i0
                shift = <uint8_t>(((j0 + r) & 3) << 1)
                adm_em_fill_1snp(
                    g_row, shift,
                    r_tile + r * n,
                    a_tile + r * n,
                    d_tile + r * n,
                    b_acc,
                    n,
                )
                if count_q_bat:
                    _count_packed_1snp(g_row, shift, cnt_acc, n)

        adm_dgemm_nn(
            2 * c, K, n, 1.0,
            ad_tile, n,
            Q + <size_t>i0 * K, K,
            beta_p, AD_p_tile, K,
        )

        if use_lockfree:
            adm_dgemm_tn(
                K, n, c, 1.0,
                P + <size_t>j0 * K, K,
                d_tile, n,
                1.0, qt_dst, qt_ld,
            )
        else:
            adm_dgemm_tn(
                K, n, c, 1.0,
                P + <size_t>j0 * K, K,
                d_tile, n,
                0.0, qt_dst, qt_ld,
            )
            blk_idx = i0 // EM_IND_TILE
            lock_id = blk_idx & (EM_NUM_LOCKS - 1)
            omp.omp_set_lock(&locks[lock_id])
            for li in range(n):
                qt_row = Q_T_global + <size_t>(i0 + li) * K
                bk = b_acc[li]
                for k in range(K):
                    qt_row[k] += Qt_tile[k * n + li] + bk
            if count_q_bat:
                qb_row = q_bat_global + i0
                for li in range(n):
                    qb_row[li] += cnt_acc[li]
            omp.omp_unset_lock(&locks[lock_id])

        i0 += EM_IND_TILE

    # All i0 blocks finished for SNP rows [j0, j0 + c): finalize P rows safely
    A_p_tile = AD_p_tile
    D_p_tile = AD_p_tile + <size_t>c * K

    for r in range(c):
        p_row = P + <size_t>(j0 + r) * K
        a_row = A_p_tile + <size_t>r * K
        d_row = D_p_tile + <size_t>r * K
        if do_adam_p:
            mp_row = m_P + <size_t>(j0 + r) * K
            vp_row = v_P + <size_t>(j0 + r) * K
            for k in range(K):
                pk = p_row[k]
                ak = a_row[k]
                dk = d_row[k]
                bk = ak - dk
                denom = pk * dk + bk
                p_em = _clip_to_domain((ak * pk) / denom if denom > 0.0 else 0.0)
                delta = p_em - pk
                m_new = beta1 * mp_row[k] + one_m_beta1 * delta
                v_new = beta2 * vp_row[k] + one_m_beta2 * delta * delta
                mp_row[k] = m_new
                vp_row[k] = v_new
                step_val = step_scale_p * m_new / (sqrt(v_new) + eps_scaled_p)
                p_row[k] = _clip_to_domain(pk + step_val)
        elif P_out != NULL:
            po_row = P_out + <size_t>(j0 + r) * K
            for k in range(K):
                pk = p_row[k]
                ak = a_row[k]
                dk = d_row[k]
                bk = ak - dk
                denom = pk * dk + bk
                po_row[k] = _clip_to_domain((ak * pk) / denom if denom > 0.0 else 0.0)

    return tile_ll


cdef void _apply_adam_q_row(
    double* q0,
    double* m_q,
    double* v_q,
    const double* t_row,
    double inv_qb,
    int K,
    double beta1,
    double one_m_beta1,
    double beta2,
    double one_m_beta2,
    double step_scale_q,
    double eps_scaled_q,
) noexcept nogil:
    cdef:
        int k
        double total_em = 0.0, inv_tot_em
        double sum_q = 0.0, inv_sum_q
        double q_em, delta, m_new, v_new, step_val
        double q_em_buf[256]
        double* q_em_ptr

    if K <= 256:
        q_em_ptr = &q_em_buf[0]
    else:
        q_em_ptr = <double*>malloc(<size_t>K * sizeof(double))
        if q_em_ptr == NULL:
            return

    for k in range(K):
        q_em = _clip_to_domain(q0[k] * t_row[k] * inv_qb)
        q_em_ptr[k] = q_em
        total_em += q_em

    inv_tot_em = 1.0 / total_em if total_em > 0.0 else 0.0
    for k in range(K):
        q_em = q_em_ptr[k] * inv_tot_em
        delta = q_em - q0[k]
        m_new = beta1 * m_q[k] + one_m_beta1 * delta
        v_new = beta2 * v_q[k] + one_m_beta2 * delta * delta
        m_q[k] = m_new
        v_q[k] = v_new
        step_val = step_scale_q * m_new / (sqrt(v_new) + eps_scaled_q)
        q0[k] = _clip_to_domain(q0[k] + step_val)
        sum_q += q0[k]

    if K > 256:
        free(q_em_ptr)

    inv_sum_q = 1.0 / sum_q if sum_q > 0.0 else 0.0
    for k in range(K):
        q0[k] *= inv_sum_q


cdef void _apply_em_q_row(
    const double* q_in,
    double* q_out,
    const double* t_row,
    double inv_qb,
    int K,
) noexcept nogil:
    cdef:
        int k
        double totalQ = 0.0, inv_totalQ, q_val
    for k in range(K):
        q_val = _clip_to_domain(q_in[k] * t_row[k] * inv_qb)
        q_out[k] = q_val
        totalQ += q_val
    if totalQ > 0.0:
        inv_totalQ = 1.0 / totalQ
        for k in range(K):
            q_out[k] *= inv_totalQ
    else:
        inv_totalQ = 1.0 / <double>K
        for k in range(K):
            q_out[k] = inv_totalQ


cdef void _em_dispatch_tile(
    int tile,
    const uint8_t* G,
    double* P,
    double* P_out,
    const double* Q,
    const double* Qt,
    double* ws_all,
    size_t ws_thr_d,
    double* q_red_all,
    size_t red_thr_stride,
    double* ll_partial,
    double* Q_T,
    double* q_bat,
    omp.omp_lock_t* locks,
    bint use_lockfree,
    bint count_q_bat,
    bint do_adam_p,
    bint compute_ll,
    double* m_P,
    double* v_P,
    double beta1,
    double one_m_beta1,
    double beta2,
    double one_m_beta2,
    double step_scale,
    double eps_scaled,
    int M,
    int N,
    int K,
    int c_max,
    int n_max,
) noexcept nogil:
    cdef:
        int tid = omp.omp_get_thread_num()
        int j0 = tile * EM_SNP_TILE
        int c = M - j0
        double* thr_ws = ws_all + <size_t>tid * ws_thr_d
        double* thr_red
        double* qt_red
        double* b_red
        double* cnt_red
        double tile_ll

    if c > EM_SNP_TILE:
        c = EM_SNP_TILE
    if use_lockfree:
        thr_red = q_red_all + <size_t>tid * red_thr_stride
        qt_red = thr_red
        b_red = thr_red + <size_t>K * N
        cnt_red = (b_red + N) if count_q_bat else NULL
    else:
        qt_red = NULL
        b_red = NULL
        cnt_red = NULL

    tile_ll = _em_snp_tile(
        G, P, P_out, Q, Qt, thr_ws,
        qt_red, b_red, cnt_red,
        Q_T, q_bat, locks,
        use_lockfree, count_q_bat, do_adam_p, compute_ll,
        m_P, v_P,
        beta1, one_m_beta1, beta2, one_m_beta2,
        step_scale, eps_scaled,
        j0, c, N, K, c_max, n_max,
    )
    if compute_ll:
        ll_partial[tid] += tile_ll


cdef void _em_finalize_q_ind(
    int i,
    double* Q,
    double* Q_out,
    double* Q_T,
    double* q_bat,
    const double* inv_q_bat,
    const double* q_red_all,
    size_t red_thr_stride,
    double* m_Q,
    double* v_Q,
    int n_threads,
    int N,
    int K,
    bint use_lockfree,
    bint accumulate_q_t,
    bint count_q_bat,
    bint do_q_em,
    bint do_adam_q,
    double beta1,
    double one_m_beta1,
    double beta2,
    double one_m_beta2,
    double step_scale,
    double eps_scaled,
) noexcept nogil:
    cdef:
        int t, k
        const double* thr_red
        double* qt_in_row
        double* q_t_ptr
        double b_sum, cnt_sum, inv_qb
        double q_t_buf[256]

    if use_lockfree:
        if K <= 256:
            q_t_ptr = &q_t_buf[0]
        else:
            q_t_ptr = &Q_T[<size_t>i * K]
        if not accumulate_q_t:
            for k in range(K):
                q_t_ptr[k] = 0.0
        else:
            qt_in_row = &Q_T[<size_t>i * K]
            for k in range(K):
                q_t_ptr[k] = qt_in_row[k]

        b_sum = 0.0
        cnt_sum = 0.0
        for t in range(n_threads):
            thr_red = q_red_all + <size_t>t * red_thr_stride
            b_sum += thr_red[<size_t>K * N + i]
            for k in range(K):
                q_t_ptr[k] += thr_red[<size_t>k * N + i]
            if count_q_bat:
                cnt_sum += thr_red[(<size_t>K + 1) * N + i]

        for k in range(K):
            q_t_ptr[k] += b_sum

        if count_q_bat and q_bat != NULL:
            q_bat[i] += cnt_sum
    else:
        q_t_ptr = &Q_T[<size_t>i * K]

    if do_adam_q:
        inv_qb = inv_q_bat[i] if inv_q_bat != NULL else (1.0 / q_bat[i] if q_bat[i] > 0.0 else 0.0)
        _apply_adam_q_row(
            &Q[<size_t>i * K],
            &m_Q[<size_t>i * K],
            &v_Q[<size_t>i * K],
            q_t_ptr,
            inv_qb,
            K,
            beta1, one_m_beta1, beta2, one_m_beta2,
            step_scale, eps_scaled,
        )
    elif do_q_em:
        inv_qb = inv_q_bat[i] if inv_q_bat != NULL else (1.0 / q_bat[i] if q_bat[i] > 0.0 else 0.0)
        _apply_em_q_row(
            &Q[<size_t>i * K],
            &Q_out[<size_t>i * K],
            q_t_ptr,
            inv_qb,
            K,
        )
    elif use_lockfree and q_t_ptr != &Q_T[<size_t>i * K]:
        qt_in_row = &Q_T[<size_t>i * K]
        for k in range(K):
            qt_in_row[k] = q_t_ptr[k]


cdef int _run_em_core(
    const uint8_t* G,
    double* P,
    double* P_out,
    double* Q,
    double* Q_out,
    double* Q_T,
    double* q_bat,
    const double* inv_q_bat,
    double* m_P,
    double* v_P,
    double* m_Q,
    double* v_Q,
    double alpha,
    double beta1,
    double beta2,
    double epsilon,
    int adam_t,
    bint accumulate_q_t,
    bint count_q_bat,
    bint do_q_em,
    bint do_adam_p,
    bint do_adam_q,
    bint compute_ll,
    double* ll_out,
    uint8_t* scratch_ptr,
    size_t scratch_len,
    int M,
    int N,
    int K,
) noexcept nogil:
    cdef:
        int n_threads, n_tiles, tile, i, k, lk, c_max, n_max, prev_blas
        size_t qt_d, ll_d, ws_thr_d, q_red_d, needed_bytes, per_ind_q, red_thr_stride
        uintptr_t aligned_addr
        bint use_lockfree
        bint own_mem = False
        uint8_t* raw_buf = NULL
        double* base
        double* Qt
        double* ll_partial
        double* ws_all
        double* q_red_all = NULL
        double* qr_row
        double total_ll
        double beta1_t, beta2_t, m_scale, v_sqrt, step_scale, eps_scaled
        double one_m_beta1 = 1.0 - beta1
        double one_m_beta2 = 1.0 - beta2
        omp.omp_lock_t locks[64]

    if M <= 0 or N <= 0 or K <= 0:
        if compute_ll and ll_out != NULL:
            ll_out[0] = 0.0
        return 0

    n_threads = omp.omp_get_max_threads()
    if n_threads < 1:
        n_threads = 1

    needed_bytes = _em_plan_layout(
        M, N, K, n_threads, count_q_bat,
        &qt_d, &ll_d, &ws_thr_d, &q_red_d, &use_lockfree,
    )

    if scratch_ptr != NULL and scratch_len >= needed_bytes:
        aligned_addr = (<uintptr_t>scratch_ptr + 63) & ~(<uintptr_t>63)
        base = <double*>aligned_addr
    else:
        own_mem = True
        raw_buf = <uint8_t*>malloc(needed_bytes)
        if raw_buf == NULL:
            return -1
        aligned_addr = (<uintptr_t>raw_buf + 63) & ~(<uintptr_t>63)
        base = <double*>aligned_addr

    Qt = base
    ll_partial = Qt + qt_d
    if compute_ll:
        memset(ll_partial, 0, <size_t>n_threads * sizeof(double))
    ws_all = ll_partial + ll_d
    if use_lockfree:
        q_red_all = ws_all + <size_t>n_threads * ws_thr_d
        memset(q_red_all, 0, q_red_d * sizeof(double))
    elif not accumulate_q_t:
        memset(Q_T, 0, <size_t>N * <size_t>K * sizeof(double))

    c_max = M if M < EM_SNP_TILE else EM_SNP_TILE
    n_max = N if N < EM_IND_TILE else EM_IND_TILE
    n_tiles = (M + EM_SNP_TILE - 1) // EM_SNP_TILE
    per_ind_q = <size_t>K + 1 + (<size_t>1 if count_q_bat else 0)
    red_thr_stride = per_ind_q * <size_t>N

    if do_adam_p or do_adam_q:
        beta1_t = pow(beta1, adam_t)
        beta2_t = pow(beta2, adam_t)
        m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
        v_sqrt = sqrt(1.0 - beta2_t) if beta2_t != 1.0 else 1.0
        step_scale = alpha * m_scale * v_sqrt
        eps_scaled = epsilon * v_sqrt
    else:
        step_scale = 0.0
        eps_scaled = 0.0

    # Transpose Q -> Qt (K x N) before parallel workers touch P or Q
    for k in range(K):
        qr_row = Qt + <size_t>k * N
        for i in range(N):
            qr_row[i] = Q[<size_t>i * K + k]

    if not use_lockfree:
        for lk in range(EM_NUM_LOCKS):
            omp.omp_init_lock(&locks[lk])

    prev_blas = adm_blas_begin_single_thread()
    with nogil, parallel(num_threads=n_threads):
        for tile in prange(n_tiles, schedule='dynamic', chunksize=1):
            _em_dispatch_tile(
                tile, G, P, P_out, Q, Qt,
                ws_all, ws_thr_d, q_red_all, red_thr_stride, ll_partial,
                Q_T, q_bat, locks,
                use_lockfree, count_q_bat, do_adam_p, compute_ll,
                m_P, v_P,
                beta1, one_m_beta1, beta2, one_m_beta2,
                step_scale, eps_scaled,
                M, N, K, c_max, n_max,
            )

        for i in prange(N, schedule='static'):
            _em_finalize_q_ind(
                i, Q, Q_out, Q_T, q_bat, inv_q_bat,
                q_red_all, red_thr_stride,
                m_Q, v_Q, n_threads, N, K,
                use_lockfree, accumulate_q_t, count_q_bat,
                do_q_em, do_adam_q,
                beta1, one_m_beta1, beta2, one_m_beta2,
                step_scale, eps_scaled,
            )

    adm_blas_end_single_thread(prev_blas)

    if not use_lockfree:
        for lk in range(EM_NUM_LOCKS):
            omp.omp_destroy_lock(&locks[lk])

    if compute_ll and ll_out != NULL:
        total_ll = 0.0
        for lk in range(n_threads):
            total_ll += ll_partial[lk]
        ll_out[0] = total_ll

    if own_mem:
        free(raw_buf)
    return 0


# Update P using Adam optimizer
cpdef void adamUpdateP(double[:,::1] P0, const double[:,::1] P1, 
                      double[:,::1] m_P, double[:,::1] v_P, 
                      const double alpha, const double beta1, const double beta2, 
                      const double epsilon, const int t, const int M, const int K) noexcept nogil:
    """
    Description:
    Updates the P matrix using the Adam optimization algorithm.
    """
    cdef:
        Py_ssize_t idx, total_mk
        double delta, step, m_new, v_new, p_val
        double beta1_t, beta2_t, m_scale, v_sqrt, step_scale, eps_scaled
        double one_m_beta1 = 1.0 - beta1
        double one_m_beta2 = 1.0 - beta2
        double* p0_ptr
        const double* p1_ptr
        double* m_p_ptr
        double* v_p_ptr

    if M <= 0 or K <= 0:
        return

    beta1_t = pow(beta1, t)
    beta2_t = pow(beta2, t)
    m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
    v_sqrt = sqrt(1.0 - beta2_t) if beta2_t != 1.0 else 1.0
    step_scale = alpha * m_scale * v_sqrt
    eps_scaled = epsilon * v_sqrt
    total_mk = <Py_ssize_t>M * <Py_ssize_t>K
    p0_ptr = &P0[0, 0]
    p1_ptr = &P1[0, 0]
    m_p_ptr = &m_P[0, 0]
    v_p_ptr = &v_P[0, 0]

    if total_mk < 2048:
        for idx in range(total_mk):
            p_val = p0_ptr[idx]
            delta = p1_ptr[idx] - p_val
            m_new = beta1 * m_p_ptr[idx] + one_m_beta1 * delta
            v_new = beta2 * v_p_ptr[idx] + one_m_beta2 * delta * delta
            m_p_ptr[idx] = m_new
            v_p_ptr[idx] = v_new
            step = step_scale * m_new / (sqrt(v_new) + eps_scaled)
            p0_ptr[idx] = _clip_to_domain(p_val + step)
    else:
        for idx in prange(total_mk, schedule='static'):
            p_val = p0_ptr[idx]
            delta = p1_ptr[idx] - p_val
            m_new = beta1 * m_p_ptr[idx] + one_m_beta1 * delta
            v_new = beta2 * v_p_ptr[idx] + one_m_beta2 * delta * delta
            m_p_ptr[idx] = m_new
            v_p_ptr[idx] = v_new
            step = step_scale * m_new / (sqrt(v_new) + eps_scaled)
            p0_ptr[idx] = _clip_to_domain(p_val + step)


# Update Q using Adam optimizer
cpdef void adamUpdateQ(double[:,::1] Q0, const double[:,::1] Q1, 
                      double[:,::1] m_Q, double[:,::1] v_Q, 
                      const double alpha, const double beta1, const double beta2, 
                      const double epsilon, const int t, const int N, const int K) noexcept nogil:
    """
    Description:
    Updates the Q matrix using the Adam optimization algorithm.
    """
    cdef:
        Py_ssize_t i, j
        double delta, step, sumQ, inv_sumQ, m_new, v_new, q_val
        double beta1_t, beta2_t, m_scale, v_sqrt, step_scale, eps_scaled
        double one_m_beta1 = 1.0 - beta1
        double one_m_beta2 = 1.0 - beta2
        double* q0
        const double* q1
        double* m_q
        double* v_q

    if N <= 0 or K <= 0:
        return

    beta1_t = pow(beta1, t)
    beta2_t = pow(beta2, t)
    m_scale = 1.0 / (1.0 - beta1_t) if beta1_t != 1.0 else 1.0
    v_sqrt = sqrt(1.0 - beta2_t) if beta2_t != 1.0 else 1.0
    step_scale = alpha * m_scale * v_sqrt
    eps_scaled = epsilon * v_sqrt

    if N < 64:
        for i in range(N):
            q0 = &Q0[i, 0]
            q1 = &Q1[i, 0]
            m_q = &m_Q[i, 0]
            v_q = &v_Q[i, 0]
            sumQ = 0.0
            for j in range(K):
                q_val = q0[j]
                delta = q1[j] - q_val
                m_new = beta1 * m_q[j] + one_m_beta1 * delta
                v_new = beta2 * v_q[j] + one_m_beta2 * delta * delta
                m_q[j] = m_new
                v_q[j] = v_new
                step = step_scale * m_new / (sqrt(v_new) + eps_scaled)
                q_val = _clip_to_domain(q_val + step)
                q0[j] = q_val
                sumQ += q_val
            inv_sumQ = 1.0 / sumQ
            for j in range(K):
                q0[j] *= inv_sumQ
    else:
        for i in prange(N, schedule='static'):
            q0 = &Q0[i, 0]
            q1 = &Q1[i, 0]
            m_q = &m_Q[i, 0]
            v_q = &v_Q[i, 0]
            sumQ = 0.0
            for j in range(K):
                q_val = q0[j]
                delta = q1[j] - q_val
                m_new = beta1 * m_q[j] + one_m_beta1 * delta
                v_new = beta2 * v_q[j] + one_m_beta2 * delta * delta
                m_q[j] = m_new
                v_q[j] = v_new
                step = step_scale * m_new / (sqrt(v_new) + eps_scaled)
                q_val = _clip_to_domain(q_val + step)
                q0[j] = q_val
                sumQ = sumQ + q_val
            inv_sumQ = 1.0 / sumQ
            for j in range(K):
                q0[j] *= inv_sumQ


# Coordinated EM P-step
cpdef void P_step(const unsigned char[:,::1] G, double[:,::1] P, double[:,::1] P_EM, 
                const double[:,::1] Q, double[:,::1] Q_T, double[::1] q_bat, 
                const int K, const int M, const int N):
    """
    Description:
    Coordinated EM step for P matrix update. Calculates EM P and accumulates factors for Q update.
    """
    cdef int rc
    if M <= 0 or N <= 0 or K <= 0:
        return
    with nogil:
        rc = _run_em_core(
            &G[0, 0], &P[0, 0], &P_EM[0, 0], <double*>&Q[0, 0], NULL,
            &Q_T[0, 0], &q_bat[0], NULL,
            NULL, NULL, NULL, NULL,
            0.0, 0.0, 0.0, 0.0, 0,
            True, True, False, False, False, False, NULL,
            NULL, 0, M, N, K,
        )
    if rc != 0:
        raise MemoryError("Out of memory allocating EM thread workspace")


# Coordinated EM Q-step
cpdef void Q_step(const double[:,::1] Q, double[:,::1] Q_EM, double[:,::1] T, 
                double[::1] q_bat, const int K, const int N) noexcept nogil:
    """
    Description:
    Coordinated EM step for Q matrix update.
    """
    cdef:
        Py_ssize_t i, k
        double a, qb
        double* t_row
    if N <= 0 or K <= 0:
        return
    if N < 64:
        for i in range(N):
            qb = q_bat[i]
            a = 1.0 / qb if qb > 0.0 else 0.0
            t_row = &T[i, 0]
            _apply_em_q_row(&Q[i, 0], &Q_EM[i, 0], t_row, a, K)
            for k in range(K):
                t_row[k] = 0.0
            q_bat[i] = 0.0
    else:
        for i in prange(N, schedule='static'):
            qb = q_bat[i]
            a = 1.0 / qb if qb > 0.0 else 0.0
            t_row = &T[i, 0]
            _apply_em_q_row(&Q[i, 0], &Q_EM[i, 0], t_row, a, K)
            for k in range(K):
                t_row[k] = 0.0
            q_bat[i] = 0.0


cpdef void em_step_inplace(
    const unsigned char[:,::1] G,
    double[:,::1] P,
    double[:,::1] Q,
    double[:,::1] T,
    const double[::1] inv_q_bat,
    const int K,
    const int M,
    const int N,
    unsigned char[::1] scratch=None,
):
    """Fused full EM step updating P and Q in place using precomputed inv_q_bat."""
    cdef:
        uint8_t* sc_ptr = &scratch[0] if (scratch is not None and scratch.shape[0] > 0) else NULL
        size_t sc_len = <size_t>scratch.shape[0] if scratch is not None else 0
        int rc
    if M <= 0 or N <= 0 or K <= 0:
        return
    with nogil:
        rc = _run_em_core(
            &G[0, 0], &P[0, 0], &P[0, 0], &Q[0, 0], &Q[0, 0],
            &T[0, 0], NULL, &inv_q_bat[0],
            NULL, NULL, NULL, NULL,
            0.0, 0.0, 0.0, 0.0, 0,
            False, False, True, False, False, False, NULL,
            sc_ptr, sc_len, M, N, K,
        )
    if rc != 0:
        raise MemoryError("Out of memory allocating EM thread workspace")


cpdef void adam_em_step_inplace(
    const unsigned char[:,::1] G,
    double[:,::1] P,
    double[:,::1] Q,
    double[:,::1] m_P,
    double[:,::1] v_P,
    double[:,::1] m_Q,
    double[:,::1] v_Q,
    double[:,::1] T,
    const double[::1] inv_q_bat,
    const double alpha,
    const double beta1,
    const double beta2,
    const double epsilon,
    const int t,
    const int K,
    const int M,
    const int N,
    unsigned char[::1] scratch=None,
):
    """Fused EM + Adam update updating P, Q, and moments in place in a single pass."""
    cdef:
        uint8_t* sc_ptr = &scratch[0] if (scratch is not None and scratch.shape[0] > 0) else NULL
        size_t sc_len = <size_t>scratch.shape[0] if scratch is not None else 0
        int rc
    if M <= 0 or N <= 0 or K <= 0:
        return
    with nogil:
        rc = _run_em_core(
            &G[0, 0], &P[0, 0], NULL, &Q[0, 0], NULL,
            &T[0, 0], NULL, &inv_q_bat[0],
            &m_P[0, 0], &v_P[0, 0], &m_Q[0, 0], &v_Q[0, 0],
            alpha, beta1, beta2, epsilon, t,
            False, False, False, True, True, False, NULL,
            sc_ptr, sc_len, M, N, K,
        )
    if rc != 0:
        raise MemoryError("Out of memory allocating EM thread workspace")


cpdef double em_targets_with_ll(
    const unsigned char[:,::1] G,
    double[:,::1] P,
    double[:,::1] P_EM,
    const double[:,::1] Q,
    double[:,::1] Q_EM,
    double[:,::1] T,
    const double[::1] inv_q_bat,
    const int K,
    const int M,
    const int N,
    unsigned char[::1] scratch=None,
):
    """Compute exact (P_EM, Q_EM) targets and exact log-likelihood at (P, Q) in one sweep."""
    cdef:
        uint8_t* sc_ptr = &scratch[0] if (scratch is not None and scratch.shape[0] > 0) else NULL
        size_t sc_len = <size_t>scratch.shape[0] if scratch is not None else 0
        double ll_val = 0.0
        int rc
    if M <= 0 or N <= 0 or K <= 0:
        return 0.0
    with nogil:
        rc = _run_em_core(
            &G[0, 0], &P[0, 0], &P_EM[0, 0], <double*>&Q[0, 0], &Q_EM[0, 0],
            &T[0, 0], NULL, &inv_q_bat[0],
            NULL, NULL, NULL, NULL,
            0.0, 0.0, 0.0, 0.0, 0,
            False, False, True, False, False, True, &ll_val,
            sc_ptr, sc_len, M, N, K,
        )
    if rc != 0:
        raise MemoryError("Out of memory allocating EM thread workspace")
    return ll_val
