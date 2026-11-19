#ifndef ADAMIXTURE_BLAS_HELPER_H
#define ADAMIXTURE_BLAS_HELPER_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static inline void _adm_scale_c(int M, int N, double beta, double* C, int ldc) {
    if (beta == 1.0) return;
    if (beta == 0.0) {
        for (int i = 0; i < M; ++i) {
            double* c_row = C + (size_t)i * ldc;
            for (int j = 0; j < N; ++j) {
                c_row[j] = 0.0;
            }
        }
    } else {
        for (int i = 0; i < M; ++i) {
            double* c_row = C + (size_t)i * ldc;
            for (int j = 0; j < N; ++j) {
                c_row[j] *= beta;
            }
        }
    }
}

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#include <dlfcn.h>

static int (*_adm_blas_get_threading)(void) = NULL;
static int (*_adm_blas_set_threading)(int) = NULL;
static int _adm_apple_blas_init = 0;

static inline void _adm_init_blas_syms(void) {
    if (!_adm_apple_blas_init) {
        _adm_blas_get_threading = (int (*)(void))dlsym(RTLD_DEFAULT, "BLASGetThreading");
        _adm_blas_set_threading = (int (*)(int))dlsym(RTLD_DEFAULT, "BLASSetThreading");
        _adm_apple_blas_init = 1;
    }
}

static inline int adm_blas_has_dgemm(void) {
    return 1;
}

static inline int adm_blas_begin_single_thread(void) {
    _adm_init_blas_syms();
    int prev = _adm_blas_get_threading ? _adm_blas_get_threading() : -1;
    if (_adm_blas_set_threading) {
        _adm_blas_set_threading(1); /* BLAS_THREADING_SINGLE_THREADED */
    }
    return prev;
}

static inline void adm_blas_end_single_thread(int prev) {
    if (prev >= 0 && _adm_blas_set_threading) {
        _adm_blas_set_threading(prev);
    }
}

static inline void adm_dgemm_nn(int M, int N, int K, double alpha,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double beta, double* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c(M, N, beta, C, ldc);
        return;
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline void adm_dgemm_tn(int M, int N, int K, double alpha,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double beta, double* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c(M, N, beta, C, ldc);
        return;
    }
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#else

#if defined(__linux__) || defined(_WIN32)
typedef void (*_adm_dgemm64_fn)(int, int, int, int64_t, int64_t, int64_t,
                                double, const double*, int64_t,
                                const double*, int64_t, double,
                                double*, int64_t);
typedef void (*_adm_dgemm32_fn)(int, int, int, int, int, int,
                                double, const double*, int,
                                const double*, int, double,
                                double*, int);
typedef int (*_adm_ob_get_threads_fn)(void);
typedef void (*_adm_ob_set_threads_fn)(int);

static _adm_dgemm64_fn _adm_dgemm64 = NULL;
static _adm_dgemm32_fn _adm_dgemm32 = NULL;
static _adm_ob_get_threads_fn _adm_ob_get = NULL;
static _adm_ob_set_threads_fn _adm_ob_set = NULL;
static int _adm_blas_init = 0;
#endif

#if defined(__linux__)
#include <dlfcn.h>
#include <link.h>
#include <string.h>

static inline void _adm_try_handle(void* h) {
    if (!h) return;
    if (!_adm_dgemm64) {
        _adm_dgemm64 = (_adm_dgemm64_fn)dlsym(h, "scipy_cblas_dgemm64_");
    }
    if (!_adm_dgemm64) {
        _adm_dgemm64 = (_adm_dgemm64_fn)dlsym(h, "cblas_dgemm64_");
    }
    if (!_adm_dgemm64 && !_adm_dgemm32) {
        _adm_dgemm32 = (_adm_dgemm32_fn)dlsym(h, "cblas_dgemm");
    }
    if (!_adm_dgemm64 && !_adm_dgemm32) {
        _adm_dgemm32 = (_adm_dgemm32_fn)dlsym(h, "scipy_cblas_dgemm");
    }
    if (!_adm_ob_get) {
        _adm_ob_get = (_adm_ob_get_threads_fn)dlsym(h, "scipy_openblas_get_num_threads64_");
    }
    if (!_adm_ob_get) {
        _adm_ob_get = (_adm_ob_get_threads_fn)dlsym(h, "openblas_get_num_threads");
    }
    if (!_adm_ob_set) {
        _adm_ob_set = (_adm_ob_set_threads_fn)dlsym(h, "scipy_openblas_set_num_threads64_");
    }
    if (!_adm_ob_set) {
        _adm_ob_set = (_adm_ob_set_threads_fn)dlsym(h, "openblas_set_num_threads");
    }
}

static int _adm_find_blas_cb(struct dl_phdr_info* info, size_t size, void* data) {
    (void)size;
    (void)data;
    const char* name = info->dlpi_name;
    if (!name || !name[0]) return 0;
    if (!strstr(name, "openblas") && !strstr(name, "blas") && !strstr(name, "mkl")) {
        return 0;
    }
    void* h = dlopen(name, RTLD_LAZY | RTLD_NOLOAD);
    if (!h) return 0;
    _adm_try_handle(h);
    return (_adm_dgemm64 || _adm_dgemm32) ? 1 : 0;
}

static inline void _adm_init_blas_syms(void) {
    if (!_adm_blas_init) {
        _adm_try_handle(RTLD_DEFAULT);
        if (!_adm_dgemm64 && !_adm_dgemm32) {
            dl_iterate_phdr(_adm_find_blas_cb, NULL);
        }
        _adm_blas_init = 1;
    }
}

#elif defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#ifdef small
#undef small
#endif

static inline void _adm_try_win_module(HMODULE h) {
    if (!h) return;
    if (!_adm_dgemm64) {
        _adm_dgemm64 = (_adm_dgemm64_fn)(void*)GetProcAddress(h, "scipy_cblas_dgemm64_");
    }
    if (!_adm_dgemm64) {
        _adm_dgemm64 = (_adm_dgemm64_fn)(void*)GetProcAddress(h, "cblas_dgemm64_");
    }
    if (!_adm_dgemm64 && !_adm_dgemm32) {
        _adm_dgemm32 = (_adm_dgemm32_fn)(void*)GetProcAddress(h, "cblas_dgemm");
    }
    if (!_adm_dgemm64 && !_adm_dgemm32) {
        _adm_dgemm32 = (_adm_dgemm32_fn)(void*)GetProcAddress(h, "scipy_cblas_dgemm");
    }
    if (!_adm_ob_get) {
        _adm_ob_get = (_adm_ob_get_threads_fn)(void*)GetProcAddress(h, "scipy_openblas_get_num_threads64_");
    }
    if (!_adm_ob_get) {
        _adm_ob_get = (_adm_ob_get_threads_fn)(void*)GetProcAddress(h, "openblas_get_num_threads");
    }
    if (!_adm_ob_set) {
        _adm_ob_set = (_adm_ob_set_threads_fn)(void*)GetProcAddress(h, "scipy_openblas_set_num_threads64_");
    }
    if (!_adm_ob_set) {
        _adm_ob_set = (_adm_ob_set_threads_fn)(void*)GetProcAddress(h, "openblas_set_num_threads");
    }
}

static inline void _adm_init_blas_syms(void) {
    if (!_adm_blas_init) {
        HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, 0);
        if (snap != INVALID_HANDLE_VALUE) {
            MODULEENTRY32 me;
            me.dwSize = sizeof(MODULEENTRY32);
            if (Module32First(snap, &me)) {
                do {
                    _adm_try_win_module(me.hModule);
                    if (_adm_dgemm64 || _adm_dgemm32) break;
                } while (Module32Next(snap, &me));
            }
            CloseHandle(snap);
        }
        _adm_blas_init = 1;
    }
}
#endif

#if defined(__linux__) || defined(_WIN32)
static inline int adm_blas_has_dgemm(void) {
    _adm_init_blas_syms();
    return (_adm_dgemm64 != NULL || _adm_dgemm32 != NULL);
}

static inline int adm_blas_begin_single_thread(void) {
    _adm_init_blas_syms();
    int prev = _adm_ob_get ? _adm_ob_get() : -1;
    if (_adm_ob_set) {
        _adm_ob_set(1);
    }
    return prev;
}

static inline void adm_blas_end_single_thread(int prev) {
    if (prev > 0 && _adm_ob_set) {
        _adm_ob_set(prev);
    }
}
#else
static inline void _adm_init_blas_syms(void) {}
static inline int adm_blas_has_dgemm(void) { return 0; }
static inline int adm_blas_begin_single_thread(void) { return -1; }
static inline void adm_blas_end_single_thread(int prev) { (void)prev; }
#endif

static inline void _adm_dgemm_nn_c(int M, int N, int K, double alpha,
                                   const double* A, int lda,
                                   const double* B, int ldb,
                                   double beta, double* C, int ldc) {
    _adm_scale_c(M, N, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        const double* a_row = A + (size_t)i * lda;
        double* c_row = C + (size_t)i * ldc;
        for (int k = 0; k < K; ++k) {
            double a_ik = alpha * a_row[k];
            const double* b_row = B + (size_t)k * ldb;
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

static inline void _adm_dgemm_tn_c(int M, int N, int K, double alpha,
                                   const double* A, int lda,
                                   const double* B, int ldb,
                                   double beta, double* C, int ldc) {
    _adm_scale_c(M, N, beta, C, ldc);
    for (int k = 0; k < K; ++k) {
        const double* a_row = A + (size_t)k * lda;
        const double* b_row = B + (size_t)k * ldb;
        for (int i = 0; i < M; ++i) {
            double a_ki = alpha * a_row[i];
            double* c_row = C + (size_t)i * ldc;
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ki * b_row[j];
            }
        }
    }
}

static inline void adm_dgemm_nn(int M, int N, int K, double alpha,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double beta, double* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c(M, N, beta, C, ldc);
        return;
    }
#if defined(__linux__) || defined(_WIN32)
    _adm_init_blas_syms();
    if (_adm_dgemm64) {
        _adm_dgemm64(101, 111, 111, (int64_t)M, (int64_t)N, (int64_t)K,
                     alpha, A, (int64_t)lda, B, (int64_t)ldb, beta, C, (int64_t)ldc);
        return;
    }
    if (_adm_dgemm32) {
        _adm_dgemm32(101, 111, 111, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    _adm_dgemm_nn_c(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline void adm_dgemm_tn(int M, int N, int K, double alpha,
                                const double* A, int lda,
                                const double* B, int ldb,
                                double beta, double* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c(M, N, beta, C, ldc);
        return;
    }
#if defined(__linux__) || defined(_WIN32)
    _adm_init_blas_syms();
    if (_adm_dgemm64) {
        _adm_dgemm64(101, 112, 111, (int64_t)M, (int64_t)N, (int64_t)K,
                     alpha, A, (int64_t)lda, B, (int64_t)ldb, beta, C, (int64_t)ldc);
        return;
    }
    if (_adm_dgemm32) {
        _adm_dgemm32(101, 112, 111, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    _adm_dgemm_tn_c(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#endif

static inline double _adm_ll_factor_branchless(double r_raw, uint8_t u) {
    double r = fmax(fmin(r_raw, 1.0 - 1e-15), 1e-15);
    double omr = 1.0 - r;
    double f1 = (u == 0) ? omr : r;
    double f2 = (u == 2) ? r : omr;
    return (u == 3) ? 1.0 : (f1 * f2);
}

static inline void adm_ll_fill_p4(
    const uint8_t* __restrict g_row,
    const double* __restrict r0,
    const double* __restrict r1,
    const double* __restrict r2,
    const double* __restrict r3,
    double* __restrict p4,
    size_t n)
{
    for (size_t li = 0; li < n; ++li) {
        uint8_t b = g_row[li];
        p4[li] = _adm_ll_factor_branchless(r0[li], b & 0x03)
               * _adm_ll_factor_branchless(r1[li], (b >> 2) & 0x03)
               * _adm_ll_factor_branchless(r2[li], (b >> 4) & 0x03)
               * _adm_ll_factor_branchless(r3[li], (b >> 6) & 0x03);
    }
}

static inline void adm_ll_fill_p1(
    const uint8_t* __restrict g_row,
    uint8_t shift,
    const double* __restrict r_in,
    double* __restrict p1,
    size_t n)
{
    for (size_t li = 0; li < n; ++li) {
        uint8_t u = (g_row[li] >> shift) & 0x03;
        p1[li] = _adm_ll_factor_branchless(r_in[li], u);
    }
}

static inline void adm_sqp_fill_4snps_f64(
    const uint8_t* __restrict g_row,
    const double* __restrict qp0,
    const double* __restrict qp1,
    const double* __restrict qp2,
    const double* __restrict qp3,
    double* __restrict ts0,
    double* __restrict ts1,
    double* __restrict ts2,
    double* __restrict ts3,
    double* __restrict tdz0,
    double* __restrict tdz1,
    double* __restrict tdz2,
    double* __restrict tdz3,
    int n)
{
    for (int li = 0; li < n; ++li) {
        uint8_t b = g_row[li];
        uint8_t u0 = b & 0x03;
        uint8_t u1 = (b >> 2) & 0x03;
        uint8_t u2 = (b >> 4) & 0x03;
        uint8_t u3 = (b >> 6) & 0x03;

        double m0 = (double)(u0 != 3);
        double m1 = (double)(u1 != 3);
        double m2 = (double)(u2 != 3);
        double m3 = (double)(u3 != 3);

        double g0 = (double)u0;
        double g1 = (double)u1;
        double g2 = (double)u2;
        double g3 = (double)u3;

        double q0 = fmax(fmin(qp0[li], 1.0 - 1e-10), 1e-10);
        double q1 = fmax(fmin(qp1[li], 1.0 - 1e-10), 1e-10);
        double q2 = fmax(fmin(qp2[li], 1.0 - 1e-10), 1e-10);
        double q3 = fmax(fmin(qp3[li], 1.0 - 1e-10), 1e-10);

        double inv0 = m0 / (q0 * (1.0 - q0));
        double inv1 = m1 / (q1 * (1.0 - q1));
        double inv2 = m2 / (q2 * (1.0 - q2));
        double inv3 = m3 / (q3 * (1.0 - q3));

        tdz0[li] = (g0 - 2.0 * q0) * inv0;
        tdz1[li] = (g1 - 2.0 * q1) * inv1;
        tdz2[li] = (g2 - 2.0 * q2) * inv2;
        tdz3[li] = (g3 - 2.0 * q3) * inv3;

        ts0[li] = (g0 * (1.0 - 2.0 * q0) + 2.0 * q0 * q0) * (inv0 * inv0);
        ts1[li] = (g1 * (1.0 - 2.0 * q1) + 2.0 * q1 * q1) * (inv1 * inv1);
        ts2[li] = (g2 * (1.0 - 2.0 * q2) + 2.0 * q2 * q2) * (inv2 * inv2);
        ts3[li] = (g3 * (1.0 - 2.0 * q3) + 2.0 * q3 * q3) * (inv3 * inv3);
    }
}

static inline void adm_sqp_fill_1snp_f64(
    const uint8_t* __restrict g_row,
    uint8_t shift,
    const double* __restrict qp,
    double* __restrict ts,
    double* __restrict tdz,
    int n)
{
    for (int li = 0; li < n; ++li) {
        uint8_t u = (g_row[li] >> shift) & 0x03;
        double m = (double)(u != 3);
        double g = (double)u;
        double q = fmax(fmin(qp[li], 1.0 - 1e-10), 1e-10);
        double inv = m / (q * (1.0 - q));
        tdz[li] = (g - 2.0 * q) * inv;
        ts[li] = (g * (1.0 - 2.0 * q) + 2.0 * q * q) * (inv * inv);
    }
}

static inline void adm_em_fill_4snps(
    const uint8_t* __restrict g_row,
    const double* __restrict r0,
    const double* __restrict r1,
    const double* __restrict r2,
    const double* __restrict r3,
    double* __restrict a0,
    double* __restrict a1,
    double* __restrict a2,
    double* __restrict a3,
    double* __restrict d0,
    double* __restrict d1,
    double* __restrict d2,
    double* __restrict d3,
    double* __restrict b_acc,
    int n)
{
    for (int li = 0; li < n; ++li) {
        uint8_t b = g_row[li];
        uint8_t u0 = b & 0x03;
        uint8_t u1 = (b >> 2) & 0x03;
        uint8_t u2 = (b >> 4) & 0x03;
        uint8_t u3 = (b >> 6) & 0x03;

        double m0 = (double)(u0 != 3);
        double m1 = (double)(u1 != 3);
        double m2 = (double)(u2 != 3);
        double m3 = (double)(u3 != 3);

        double g0 = (double)u0;
        double g1 = (double)u1;
        double g2 = (double)u2;
        double g3 = (double)u3;

        double q0 = fmax(fmin(r0[li], 1.0 - 1e-15), 1e-15);
        double q1 = fmax(fmin(r1[li], 1.0 - 1e-15), 1e-15);
        double q2 = fmax(fmin(r2[li], 1.0 - 1e-15), 1e-15);
        double q3 = fmax(fmin(r3[li], 1.0 - 1e-15), 1e-15);

        double w0 = q0 * (1.0 - q0);
        double w1 = q1 * (1.0 - q1);
        double w2 = q2 * (1.0 - q2);
        double w3 = q3 * (1.0 - q3);

        double w01 = w0 * w1;
        double w23 = w2 * w3;
        double inv_all = 1.0 / (w01 * w23);
        double inv01 = inv_all * w23;
        double inv23 = inv_all * w01;
        double inv0 = m0 * (inv01 * w1);
        double inv1 = m1 * (inv01 * w0);
        double inv2 = m2 * (inv23 * w3);
        double inv3 = m3 * (inv23 * w2);

        double av0 = g0 * (1.0 - q0) * inv0;
        double bv0 = (2.0 - g0) * q0 * inv0;
        a0[li] = av0;
        d0[li] = av0 - bv0;

        double av1 = g1 * (1.0 - q1) * inv1;
        double bv1 = (2.0 - g1) * q1 * inv1;
        a1[li] = av1;
        d1[li] = av1 - bv1;

        double av2 = g2 * (1.0 - q2) * inv2;
        double bv2 = (2.0 - g2) * q2 * inv2;
        a2[li] = av2;
        d2[li] = av2 - bv2;

        double av3 = g3 * (1.0 - q3) * inv3;
        double bv3 = (2.0 - g3) * q3 * inv3;
        a3[li] = av3;
        d3[li] = av3 - bv3;

        b_acc[li] += bv0 + bv1 + bv2 + bv3;
    }
}

static inline void adm_em_fill_1snp(
    const uint8_t* __restrict g_row,
    uint8_t shift,
    const double* __restrict r_in,
    double* __restrict a_out,
    double* __restrict d_out,
    double* __restrict b_acc,
    int n)
{
    for (int li = 0; li < n; ++li) {
        uint8_t u = (g_row[li] >> shift) & 0x03;
        double m = (double)(u != 3);
        double g = (double)u;
        double q = fmax(fmin(r_in[li], 1.0 - 1e-15), 1e-15);
        double inv = m / (q * (1.0 - q));
        double av = g * (1.0 - q) * inv;
        double bv = (2.0 - g) * q * inv;
        a_out[li] = av;
        d_out[li] = av - bv;
        b_acc[li] += bv;
    }
}

static inline void adm_em_fill_4snps_ll(
    const uint8_t* __restrict g_row,
    const double* __restrict r0,
    const double* __restrict r1,
    const double* __restrict r2,
    const double* __restrict r3,
    double* __restrict a0,
    double* __restrict a1,
    double* __restrict a2,
    double* __restrict a3,
    double* __restrict d0,
    double* __restrict d1,
    double* __restrict d2,
    double* __restrict d3,
    double* __restrict b_acc,
    double* __restrict p4_out,
    int n)
{
    for (int li = 0; li < n; ++li) {
        uint8_t b = g_row[li];
        uint8_t u0 = b & 0x03;
        uint8_t u1 = (b >> 2) & 0x03;
        uint8_t u2 = (b >> 4) & 0x03;
        uint8_t u3 = (b >> 6) & 0x03;

        double m0 = (double)(u0 != 3);
        double m1 = (double)(u1 != 3);
        double m2 = (double)(u2 != 3);
        double m3 = (double)(u3 != 3);

        double g0 = (double)u0;
        double g1 = (double)u1;
        double g2 = (double)u2;
        double g3 = (double)u3;

        double q0 = fmax(fmin(r0[li], 1.0 - 1e-15), 1e-15);
        double q1 = fmax(fmin(r1[li], 1.0 - 1e-15), 1e-15);
        double q2 = fmax(fmin(r2[li], 1.0 - 1e-15), 1e-15);
        double q3 = fmax(fmin(r3[li], 1.0 - 1e-15), 1e-15);

        double w0 = q0 * (1.0 - q0);
        double w1 = q1 * (1.0 - q1);
        double w2 = q2 * (1.0 - q2);
        double w3 = q3 * (1.0 - q3);

        double w01 = w0 * w1;
        double w23 = w2 * w3;
        double inv_all = 1.0 / (w01 * w23);
        double inv01 = inv_all * w23;
        double inv23 = inv_all * w01;
        double inv0 = m0 * (inv01 * w1);
        double inv1 = m1 * (inv01 * w0);
        double inv2 = m2 * (inv23 * w3);
        double inv3 = m3 * (inv23 * w2);

        double av0 = g0 * (1.0 - q0) * inv0;
        double bv0 = (2.0 - g0) * q0 * inv0;
        a0[li] = av0;
        d0[li] = av0 - bv0;

        double av1 = g1 * (1.0 - q1) * inv1;
        double bv1 = (2.0 - g1) * q1 * inv1;
        a1[li] = av1;
        d1[li] = av1 - bv1;

        double av2 = g2 * (1.0 - q2) * inv2;
        double bv2 = (2.0 - g2) * q2 * inv2;
        a2[li] = av2;
        d2[li] = av2 - bv2;

        double av3 = g3 * (1.0 - q3) * inv3;
        double bv3 = (2.0 - g3) * q3 * inv3;
        a3[li] = av3;
        d3[li] = av3 - bv3;

        b_acc[li] += bv0 + bv1 + bv2 + bv3;

        p4_out[li] = _adm_ll_factor_branchless(q0, u0)
                   * _adm_ll_factor_branchless(q1, u1)
                   * _adm_ll_factor_branchless(q2, u2)
                   * _adm_ll_factor_branchless(q3, u3);
    }
}

static inline void adm_em_fill_1snp_ll(
    const uint8_t* __restrict g_row,
    uint8_t shift,
    const double* __restrict r_in,
    double* __restrict a_out,
    double* __restrict d_out,
    double* __restrict b_acc,
    double* __restrict p1_out,
    int n)
{
    for (int li = 0; li < n; ++li) {
        uint8_t u = (g_row[li] >> shift) & 0x03;
        double m = (double)(u != 3);
        double g = (double)u;
        double q = fmax(fmin(r_in[li], 1.0 - 1e-15), 1e-15);
        double inv = m / (q * (1.0 - q));
        double av = g * (1.0 - q) * inv;
        double bv = (2.0 - g) * q * inv;
        a_out[li] = av;
        d_out[li] = av - bv;
        b_acc[li] += bv;
        p1_out[li] = _adm_ll_factor_branchless(q, u);
    }
}

#endif /* ADAMIXTURE_BLAS_HELPER_H */
