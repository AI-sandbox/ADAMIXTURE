#ifndef ADAMIXTURE_BLAS_HELPER_H
#define ADAMIXTURE_BLAS_HELPER_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>

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

#if defined(__linux__)
#include <dlfcn.h>
#include <stdint.h>

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
static int _adm_linux_blas_init = 0;

static inline void _adm_init_blas_syms(void) {
    if (!_adm_linux_blas_init) {
        _adm_dgemm64 = (_adm_dgemm64_fn)dlsym(RTLD_DEFAULT, "cblas_dgemm64_");
        if (!_adm_dgemm64) {
            _adm_dgemm32 = (_adm_dgemm32_fn)dlsym(RTLD_DEFAULT, "cblas_dgemm");
        }
        _adm_ob_get = (_adm_ob_get_threads_fn)dlsym(RTLD_DEFAULT, "openblas_get_num_threads");
        _adm_ob_set = (_adm_ob_set_threads_fn)dlsym(RTLD_DEFAULT, "openblas_set_num_threads");
        _adm_linux_blas_init = 1;
    }
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
static inline int adm_blas_begin_single_thread(void) { return -1; }
static inline void adm_blas_end_single_thread(int prev) { (void)prev; }
#endif

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
#if defined(__linux__)
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
#if defined(__linux__)
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

#endif /* ADAMIXTURE_BLAS_HELPER_H */

