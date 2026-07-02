#include <metal_stdlib>
using namespace metal;

#define BPP_MAX_ITER 50
#define SINGULAR_TOL 1e-12f
#define PIVOT_TOL 1e-8f

static inline bool solve_subset(
    device const float* A, thread float* b, thread float* x,
    thread int* F, int K, thread float* aug, thread int* map_idx)
{
    int n = 0;
    for (int i = 0; i < K; i++) {
        if (F[i] == 1) {
            map_idx[n] = i;
            n++;
        }
    }
    if (n == 0) return true;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (n + 1) + j] = A[map_idx[i] * K + map_idx[j]];
        }
        aug[i * (n + 1) + n] = b[map_idx[i]];
    }

    for (int i = 0; i < n; i++) {
        int max_row = i;
        float max_val = fabs(aug[i * (n + 1) + i]);

        for (int k = i + 1; k < n; k++) {
            float tmp = fabs(aug[k * (n + 1) + i]);
            if (tmp > max_val) {
                max_val = tmp;
                max_row = k;
            }
        }

        if (max_val < SINGULAR_TOL) return false;

        if (max_row != i) {
            for (int j = i; j <= n; j++) {
                float tmp = aug[i * (n + 1) + j];
                aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        float pivot = aug[i * (n + 1) + i];
        for (int j = i; j <= n; j++) {
            aug[i * (n + 1) + j] /= pivot;
        }

        for (int k = 0; k < n; k++) {
            if (k != i) {
                float tmp = aug[k * (n + 1) + i];
                for (int j = i; j <= n; j++) {
                    aug[k * (n + 1) + j] -= tmp * aug[i * (n + 1) + j];
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        x[map_idx[i]] = aug[i * (n + 1) + n];
    }
    return true;
}

kernel void bvls_bpp_kernel(
    device float* X [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device const float* B [[buffer(2)]],
    device const float* X0 [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& K [[buffer(5)]],
    constant float& lower [[buffer(6)]],
    constant float& upper [[buffer(7)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= (uint)M) return;

    device const float* b = B + row * K;
    device const float* x0 = X0 + row * K;
    device float* x_out = X + row * K;

    float aug[64 * 65];
    int map_idx[64];
    int F[64];
    int U[64];
    float x[64];
    float b_subset[64];

    bool changed = false;
    for (int i = 0; i < K; i++) {
        F[i] = 1;
        U[i] = 0;
        x[i] = x0[i];
        if (x[i] < lower - PIVOT_TOL) {
            F[i] = 0;
            U[i] = 0;
            changed = true;
        } else if (x[i] > upper + PIVOT_TOL) {
            F[i] = 0;
            U[i] = 1;
            changed = true;
        }
    }

    if (!changed) {
        for (int i = 0; i < K; i++) {
            x_out[i] = x[i];
        }
        return;
    }

    for (int iter = 1; iter < BPP_MAX_ITER; iter++) {
        for (int i = 0; i < K; i++) {
            if (F[i] == 1) {
                b_subset[i] = b[i];
                for (int j = 0; j < K; j++) {
                    if (F[j] == 0) {
                        b_subset[i] -= A[i * K + j] * ((U[j] == 1) ? upper : lower);
                    }
                }
            }
        }

        if (!solve_subset(A, b_subset, x, F, K, aug, map_idx)) {
            break;
        }

        for (int i = 0; i < K; i++) {
            if (F[i] == 0) {
                x[i] = (U[i] == 1) ? upper : lower;
            }
        }

        changed = false;
        for (int i = 0; i < K; i++) {
            float y_i = -b[i];
            for (int j = 0; j < K; j++) {
                y_i = fma(A[i * K + j], x[j], y_i);
            }

            if (F[i] == 1) {
                if (x[i] < lower - PIVOT_TOL) {
                    F[i] = 0; U[i] = 0; changed = true;
                } else if (x[i] > upper + PIVOT_TOL) {
                    F[i] = 0; U[i] = 1; changed = true;
                }
            } else {
                if (U[i] == 0 && y_i < -PIVOT_TOL) {
                    F[i] = 1; U[i] = 0; changed = true;
                } else if (U[i] == 1 && y_i > PIVOT_TOL) {
                    F[i] = 1; U[i] = 0; changed = true;
                }
            }
        }

        if (!changed) break;
    }

    for (int i = 0; i < K; i++) {
        x_out[i] = x[i];
    }
}
