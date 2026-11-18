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

static inline void _adm_scale_c_f32(int M, int N, float beta, float* C, int ldc) {
    if (beta == 1.0f) return;
    if (beta == 0.0f) {
        for (int i = 0; i < M; ++i) {
            float* c_row = C + (size_t)i * ldc;
            for (int j = 0; j < N; ++j) {
                c_row[j] = 0.0f;
            }
        }
    } else {
        for (int i = 0; i < M; ++i) {
            float* c_row = C + (size_t)i * ldc;
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

static inline void adm_sgemm_nn(int M, int N, int K, float alpha,
                                const float* A, int lda,
                                const float* B, int ldb,
                                float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c_f32(M, N, beta, C, ldc);
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline void adm_sgemm_tn(int M, int N, int K, float alpha,
                                const float* A, int lda,
                                const float* B, int ldb,
                                float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c_f32(M, N, beta, C, ldc);
        return;
    }
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

#else

#if defined(__linux__)
#include <dlfcn.h>

typedef void (*_adm_dgemm64_fn)(int, int, int, int64_t, int64_t, int64_t,
                                double, const double*, int64_t,
                                const double*, int64_t, double,
                                double*, int64_t);
typedef void (*_adm_dgemm32_fn)(int, int, int, int, int, int,
                                double, const double*, int,
                                const double*, int, double,
                                double*, int);
typedef void (*_adm_sgemm64_fn)(int, int, int, int64_t, int64_t, int64_t,
                                float, const float*, int64_t,
                                const float*, int64_t, float,
                                float*, int64_t);
typedef void (*_adm_sgemm32_fn)(int, int, int, int, int, int,
                                float, const float*, int,
                                const float*, int, float,
                                float*, int);
typedef int (*_adm_ob_get_threads_fn)(void);
typedef void (*_adm_ob_set_threads_fn)(int);

static _adm_dgemm64_fn _adm_dgemm64 = NULL;
static _adm_dgemm32_fn _adm_dgemm32 = NULL;
static _adm_sgemm64_fn _adm_sgemm64 = NULL;
static _adm_sgemm32_fn _adm_sgemm32 = NULL;
static _adm_ob_get_threads_fn _adm_ob_get = NULL;
static _adm_ob_set_threads_fn _adm_ob_set = NULL;
static int _adm_linux_blas_init = 0;

static inline void _adm_init_blas_syms(void) {
    if (!_adm_linux_blas_init) {
        _adm_dgemm64 = (_adm_dgemm64_fn)dlsym(RTLD_DEFAULT, "cblas_dgemm64_");
        if (!_adm_dgemm64) {
            _adm_dgemm32 = (_adm_dgemm32_fn)dlsym(RTLD_DEFAULT, "cblas_dgemm");
        }
        _adm_sgemm64 = (_adm_sgemm64_fn)dlsym(RTLD_DEFAULT, "cblas_sgemm64_");
        if (!_adm_sgemm64) {
            _adm_sgemm32 = (_adm_sgemm32_fn)dlsym(RTLD_DEFAULT, "cblas_sgemm");
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

static inline void _adm_sgemm_nn_c(int M, int N, int K, float alpha,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float beta, float* C, int ldc) {
    _adm_scale_c_f32(M, N, beta, C, ldc);
    for (int i = 0; i < M; ++i) {
        const float* a_row = A + (size_t)i * lda;
        float* c_row = C + (size_t)i * ldc;
        for (int k = 0; k < K; ++k) {
            float a_ik = alpha * a_row[k];
            const float* b_row = B + (size_t)k * ldb;
            for (int j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

static inline void _adm_sgemm_tn_c(int M, int N, int K, float alpha,
                                   const float* A, int lda,
                                   const float* B, int ldb,
                                   float beta, float* C, int ldc) {
    _adm_scale_c_f32(M, N, beta, C, ldc);
    for (int k = 0; k < K; ++k) {
        const float* a_row = A + (size_t)k * lda;
        const float* b_row = B + (size_t)k * ldb;
        for (int i = 0; i < M; ++i) {
            float a_ki = alpha * a_row[i];
            float* c_row = C + (size_t)i * ldc;
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

static inline void adm_sgemm_nn(int M, int N, int K, float alpha,
                                const float* A, int lda,
                                const float* B, int ldb,
                                float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c_f32(M, N, beta, C, ldc);
        return;
    }
#if defined(__linux__)
    _adm_init_blas_syms();
    if (_adm_sgemm64) {
        _adm_sgemm64(101, 111, 111, (int64_t)M, (int64_t)N, (int64_t)K,
                     alpha, A, (int64_t)lda, B, (int64_t)ldb, beta, C, (int64_t)ldc);
        return;
    }
    if (_adm_sgemm32) {
        _adm_sgemm32(101, 111, 111, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    _adm_sgemm_nn_c(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline void adm_sgemm_tn(int M, int N, int K, float alpha,
                                const float* A, int lda,
                                const float* B, int ldb,
                                float beta, float* C, int ldc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) {
        _adm_scale_c_f32(M, N, beta, C, ldc);
        return;
    }
#if defined(__linux__)
    _adm_init_blas_syms();
    if (_adm_sgemm64) {
        _adm_sgemm64(101, 112, 111, (int64_t)M, (int64_t)N, (int64_t)K,
                     alpha, A, (int64_t)lda, B, (int64_t)ldb, beta, C, (int64_t)ldc);
        return;
    }
    if (_adm_sgemm32) {
        _adm_sgemm32(101, 112, 111, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif
    _adm_sgemm_tn_c(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
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

static inline void adm_sqp_fill_4snps_mixed(
    const uint8_t* __restrict g_row,
    const double* __restrict qp0,
    const double* __restrict qp1,
    const double* __restrict qp2,
    const double* __restrict qp3,
    float* __restrict ts0,
    float* __restrict ts1,
    float* __restrict ts2,
    float* __restrict ts3,
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

        double w0 = (g0 * (1.0 - 2.0 * q0) + 2.0 * q0 * q0) * (inv0 * inv0);
        double w1 = (g1 * (1.0 - 2.0 * q1) + 2.0 * q1 * q1) * (inv1 * inv1);
        double w2 = (g2 * (1.0 - 2.0 * q2) + 2.0 * q2 * q2) * (inv2 * inv2);
        double w3 = (g3 * (1.0 - 2.0 * q3) + 2.0 * q3 * q3) * (inv3 * inv3);

        ts0[li] = (float)fmin(w0, 1e30);
        ts1[li] = (float)fmin(w1, 1e30);
        ts2[li] = (float)fmin(w2, 1e30);
        ts3[li] = (float)fmin(w3, 1e30);
    }
}

static inline void adm_sqp_fill_1snp_mixed(
    const uint8_t* __restrict g_row,
    uint8_t shift,
    const double* __restrict qp,
    float* __restrict ts,
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
        double w = (g * (1.0 - 2.0 * q) + 2.0 * q * q) * (inv * inv);
        ts[li] = (float)fmin(w, 1e30);
    }
}

#endif /* ADAMIXTURE_BLAS_HELPER_H */
