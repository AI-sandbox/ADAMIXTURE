#include <metal_stdlib>
using namespace metal;

#define SQP_MAX_K 32
#define SQP_MAX_SZ 34

static inline void sweep(thread float* matrix_a, int sz, int k, thread float* tmp, bool inverse) {
    float piv = matrix_a[k * sz + k];
    if (piv == 0.0f) return;
    float p = 1.0f / piv;
    for (int i = 0; i < sz; ++i) {
        tmp[i] = matrix_a[i * sz + k];
        matrix_a[i * sz + k] = 0.0f;
        matrix_a[k * sz + i] = 0.0f;
    }
    tmp[k] = inverse ? 1.0f : -1.0f;
    for (int j = 0; j < sz; ++j) {
        float tj = tmp[j];
        int row = j * sz;
        for (int i = 0; i < sz; ++i) {
            matrix_a[row + i] = fma(-(p * tmp[i]), tj, matrix_a[row + i]);
        }
    }
}

static inline int quadratic_program_device(
    thread float* delta, thread float* tableau, device const float* par,
    int p, int c, thread float* d, thread float* tmp, thread int* swept)
{
    int sz = p + c + 1;
    float small = 1e-5f;
    float tol = 1e-8f;

    for (int i = 0; i < p; i++) {
        delta[i] = 0.0f;
    }

    for (int i = 0; i < sz; i++) {
        d[i] = tableau[i * sz + i];
    }

    for (int i = 0; i < p; i++) {
        if (d[i] <= 0.0f || tableau[i * sz + i] < d[i] * tol) {
            return 0;
        } else {
            sweep(tableau, sz, i, tmp, false);
        }
    }

    for (int i = 0; i < p; i++) {
        swept[i] = 1;
    }

    for (int i = p; i < p + c; i++) {
        if (tableau[i * sz + i] >= 0.0f) {
            return 0;
        } else {
            sweep(tableau, sz, i, tmp, false);
        }
    }

    for (int iteration = 1; iteration <= 1000; iteration++) {
        float a = 1.0f;
        for (int i = 0; i < p; i++) {
            if (swept[i]) {
                float ui = tableau[i * sz + (sz - 1)];
                float ai;
                if (ui > 0.0f) {
                    ai = 1.0f - par[i] - delta[i];
                } else {
                    ai = 0.0f - par[i] - delta[i];
                }
                if (fabs(ui) > 1e-10f) {
                    float temp = ai / ui;
                    if (temp < a) {
                        a = temp;
                    }
                }
            }
        }

        for (int i = 0; i < p; i++) {
            if (swept[i]) {
                float ui = tableau[i * sz + (sz - 1)];
                delta[i] += a * ui;
                tableau[i * sz + (sz - 1)] = (1.0f - a) * ui;
                tableau[(sz - 1) * sz + i] = tableau[i * sz + (sz - 1)];
            }
        }

        bool cycle_main_loop = false;
        for (int i = 0; i < p; i++) {
            bool critical = (0.0f >= par[i] + delta[i] - small) || (1.0f <= par[i] + delta[i] + small);
            if (swept[i] && (fabs(tableau[i * sz + i]) > 1e-10f) && critical) {
                sweep(tableau, sz, i, tmp, true);
                swept[i] = 0;
                cycle_main_loop = true;
                break;
            }
        }

        if (cycle_main_loop) continue;

        for (int i = 0; i < p; i++) {
            float ui = tableau[i * sz + (sz - 1)];
            bool violation = (ui > 0.0f && 0.0f >= par[i] + delta[i] - small) ||
                             (ui < 0.0f && 1.0f <= par[i] + delta[i] + small);
            if (!swept[i] && violation) {
                sweep(tableau, sz, i, tmp, false);
                swept[i] = 1;
                cycle_main_loop = true;
                break;
            }
        }

        if (cycle_main_loop) continue;

        return iteration;
    }
    return 0;
}

static inline void project_q_simplex_row(device float* b, int n, float pseudocount) {
    float tau = 1.0f - n * pseudocount;
    float tsum = 0.0f;
    float tmax = 0.0f;
    bool bget = false;
    int idx[SQP_MAX_K];
    float vals[SQP_MAX_K];

    for (int i = 0; i < n; i++) {
        vals[i] = b[i] - pseudocount;
        idx[i] = i;
    }

    for (int i = 1; i < n; i++) {
        int key_idx = idx[i];
        float key_val = vals[key_idx];
        int j = i - 1;
        while (j >= 0 && vals[idx[j]] < key_val) {
            idx[j + 1] = idx[j];
            j = j - 1;
        }
        idx[j + 1] = key_idx;
    }

    for (int i = 0; i < n - 1; i++) {
        tsum += vals[idx[i]];
        tmax = (tsum - tau) / (i + 1);
        if (tmax >= vals[idx[i + 1]]) {
            bget = true;
            break;
        }
    }

    if (!bget) {
        tmax = (tsum + vals[idx[n - 1]] - tau) / n;
    }

    for (int i = 0; i < n; i++) {
        float val = vals[i] - tmax;
        if (val < 0.0f) val = 0.0f;
        b[i] = val + pseudocount;
    }
}

static inline void project_p_box_row(device float* b, int n, float pseudocount) {
    for (int i = 0; i < n; i++) {
        float val = b[i];
        if (val < pseudocount) val = pseudocount;
        if (val > 1.0f - pseudocount) val = 1.0f - pseudocount;
        b[i] = val;
    }
}

static inline void create_tableau_simplex_device(
    thread float* tableau, device const float* matrix_q, device const float* r,
    device const float* x, device const float* v_kk, int K)
{
    int sz = K + 2;
    float tmp_k[SQP_MAX_K];
    float tmp_k2[SQP_MAX_K];

    for (int i = 0; i < sz * sz; i++) {
        tableau[i] = 0.0f;
    }

    for (int i = 0; i < K; i++) {
        tmp_k[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            tmp_k[i] += matrix_q[i * K + j] * v_kk[j * K + 0];
        }
    }

    for (int i = 0; i < K; i++) {
        tmp_k2[i] = 0.0f;
        for (int j = 0; j < K; j++) {
            tmp_k2[i] += v_kk[i * K + j] * tmp_k[j];
        }
    }

    float norm1 = 0.0f;
    for (int i = 0; i < K; i++) {
        norm1 += fabs(tmp_k2[i]);
    }

    float mu = (norm1 - 2.0f * fabs(tmp_k2[0])) / K;
    mu = 2.0f * mu;
    if (mu < 0.0f) {
        mu = 0.0f;
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            tableau[i * sz + j] = matrix_q[i * K + j] + mu;
        }
    }

    for (int i = 0; i < K; i++) {
        tableau[i * sz + K] = 1.0f;
        tableau[K * sz + i] = 1.0f;
        tableau[i * sz + (K + 1)] = -r[i];
        tableau[(K + 1) * sz + i] = -r[i];
    }

    tableau[K * sz + K] = 0.0f;

    float sum_x = 0.0f;
    for (int i = 0; i < K; i++) {
        sum_x += x[i];
    }

    tableau[K * sz + (K + 1)] = 1.0f - sum_x;
    tableau[(K + 1) * sz + K] = 1.0f - sum_x;
    tableau[(K + 1) * sz + (K + 1)] = 0.0f;
}

static inline void create_tableau_box_device(
    thread float* tableau, device const float* matrix_q, device const float* r,
    device const float* x, int K)
{
    int sz = K + 1;
    for (int i = 0; i < sz * sz; i++) {
        tableau[i] = 0.0f;
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            tableau[i * sz + j] = matrix_q[i * K + j];
        }
    }

    for (int i = 0; i < K; i++) {
        tableau[i * sz + K] = -r[i];
        tableau[K * sz + i] = -r[i];
    }

    tableau[K * sz + K] = 0.0f;
}

kernel void sqp_solve_q_kernel(
    device float* Q_next [[buffer(0)]],
    device const float* XtX_q [[buffer(1)]],
    device const float* Xtz_q [[buffer(2)]],
    device const float* Q [[buffer(3)]],
    device const float* v_kk [[buffer(4)]],
    constant int& N [[buffer(5)]],
    constant int& K [[buffer(6)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= (uint)N) return;

    float tableau[SQP_MAX_SZ * SQP_MAX_SZ];
    float d_buf[SQP_MAX_SZ];
    float tmp_buf[SQP_MAX_SZ];
    int swept[SQP_MAX_K];
    float delta[SQP_MAX_K];

    create_tableau_simplex_device(tableau, XtX_q + i * K * K, Xtz_q + i * K, Q + i * K, v_kk, K);
    quadratic_program_device(delta, tableau, Q + i * K, K, 1, d_buf, tmp_buf, swept);

    for (int k = 0; k < K; k++) {
        Q_next[i * K + k] = Q[i * K + k] + delta[k];
    }
    project_q_simplex_row(Q_next + i * K, K, 1e-5f);
}

kernel void sqp_solve_p_kernel(
    device float* P_next [[buffer(0)]],
    device const float* XtX_p [[buffer(1)]],
    device const float* Xtz_p [[buffer(2)]],
    device const float* P [[buffer(3)]],
    constant int& M [[buffer(4)]],
    constant int& K [[buffer(5)]],
    uint j [[thread_position_in_grid]])
{
    if (j >= (uint)M) return;

    float tableau[SQP_MAX_SZ * SQP_MAX_SZ];
    float d_buf[SQP_MAX_SZ];
    float tmp_buf[SQP_MAX_SZ];
    int swept[SQP_MAX_K];
    float delta[SQP_MAX_K];

    create_tableau_box_device(tableau, XtX_p + j * K * K, Xtz_p + j * K, P + j * K, K);
    quadratic_program_device(delta, tableau, P + j * K, K, 0, d_buf, tmp_buf, swept);

    for (int k = 0; k < K; k++) {
        P_next[j * K + k] = P[j * K + k] + delta[k];
    }
    project_p_box_row(P_next + j * K, K, 1e-5f);
}
