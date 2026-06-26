#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

__device__ void sweep(double* matrix_a, int sz, int k, double* tmp, bool inverse) {
    double piv = matrix_a[k * sz + k];
    if (piv == 0.0) return;
    double p = 1.0 / piv;
    for (int i = 0; i < sz; ++i) {
        tmp[i] = matrix_a[i * sz + k];
        matrix_a[i * sz + k] = 0.0;
        matrix_a[k * sz + i] = 0.0;
    }
    if (inverse) {
        tmp[k] = 1.0;
    } else {
        tmp[k] = -1.0;
    }
    for (int i = 0; i < sz; ++i) {
        double pv = p * tmp[i];
        for (int j = 0; j < sz; ++j) {
            matrix_a[j * sz + i] -= pv * tmp[j];
        }
    }
}

__device__ int quadratic_program_device(
    double* delta, double* tableau, const double* par,
    int p, int c,
    double* d, double* tmp, int* swept)
{
    int sz = p + c + 1;
    double small = 1e-5;
    double tol = 1e-8;
    
    for (int i = 0; i < p; i++) {
        delta[i] = 0.0;
    }
    
    for (int i = 0; i < sz; i++) {
        d[i] = tableau[i * sz + i];
    }
    
    for (int i = 0; i < p; i++) {
        if (d[i] <= 0.0 || tableau[i * sz + i] < d[i] * tol) {
            return 0;
        } else {
            sweep(tableau, sz, i, tmp, false);
        }
    }
    
    for (int i = 0; i < p; i++) {
        swept[i] = 1;
    }
    
    for (int i = p; i < p + c; i++) {
        if (tableau[i * sz + i] >= 0.0) {
            return 0;
        } else {
            sweep(tableau, sz, i, tmp, false);
        }
    }
    
    for (int iteration = 1; iteration <= 1000; iteration++) {
        double a = 1.0;
        for (int i = 0; i < p; i++) {
            if (swept[i]) {
                double ui = tableau[i * sz + (sz - 1)];
                double ai;
                if (ui > 0.0) {
                    ai = 1.0 - par[i] - delta[i];
                } else {
                    ai = 0.0 - par[i] - delta[i];
                }
                if (fabs(ui) > 1e-10) {
                    double temp = ai / ui;
                    if (temp < a) {
                        a = temp;
                    }
                }
            }
        }
        
        for (int i = 0; i < p; i++) {
            if (swept[i]) {
                double ui = tableau[i * sz + (sz - 1)];
                delta[i] += a * ui;
                tableau[i * sz + (sz - 1)] = (1.0 - a) * ui;
                tableau[(sz - 1) * sz + i] = tableau[i * sz + (sz - 1)];
            }
        }
        
        bool cycle_main_loop = false;
        for (int i = 0; i < p; i++) {
            bool critical = (0.0 >= par[i] + delta[i] - small) || (1.0 <= par[i] + delta[i] + small);
            if (swept[i] && (fabs(tableau[i * sz + i]) > 1e-10) && critical) {
                sweep(tableau, sz, i, tmp, true);
                swept[i] = 0;
                cycle_main_loop = true;
                break;
            }
        }
        
        if (cycle_main_loop) continue;
        
        for (int i = 0; i < p; i++) {
            double ui = tableau[i * sz + (sz - 1)];
            bool violation = (ui > 0.0 && 0.0 >= par[i] + delta[i] - small) || (ui < 0.0 && 1.0 <= par[i] + delta[i] + small);
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

__device__ void project_q_simplex_row(double* b, int n, double pseudocount) {
    double tau = 1.0 - n * pseudocount;
    double tsum = 0.0;
    double tmax = 0.0;
    bool bget = false;
    int idx[100];
    
    for (int i = 0; i < n; i++) {
        b[i] -= pseudocount;
        idx[i] = i;
    }
    
    for (int i = 1; i < n; i++) {
        int key_idx = idx[i];
        double key_val = b[key_idx];
        int j = i - 1;
        while (j >= 0 && b[idx[j]] < key_val) {
            idx[j + 1] = idx[j];
            j = j - 1;
        }
        idx[j + 1] = key_idx;
    }
    
    for (int i = 0; i < n - 1; i++) {
        tsum += b[idx[i]];
        tmax = (tsum - tau) / (i + 1);
        if (tmax >= b[idx[i + 1]]) {
            bget = true;
            break;
        }
    }
    
    if (!bget) {
        tmax = (tsum + b[idx[n - 1]] - tau) / n;
    }
    
    for (int i = 0; i < n; i++) {
        double val = b[i] - tmax;
        if (val < 0.0) val = 0.0;
        b[i] = val + pseudocount;
    }
}

__device__ void project_p_box_row(double* b, int n, double pseudocount) {
    for (int i = 0; i < n; i++) {
        double val = b[i];
        if (val < pseudocount) val = pseudocount;
        if (val > 1.0 - pseudocount) val = 1.0 - pseudocount;
        b[i] = val;
    }
}

__device__ void create_tableau_simplex_device(
    double* tableau, const double* matrix_q, const double* r,
    const double* x, const double* v_kk, int K)
{
    int sz = K + 2;
    double tmp_k[100];
    double tmp_k2[100];
    
    for (int i = 0; i < sz * sz; i++) {
        tableau[i] = 0.0;
    }
    
    for (int i = 0; i < K; i++) {
        tmp_k[i] = 0.0;
        for (int j = 0; j < K; j++) {
            tmp_k[i] += matrix_q[i * K + j] * v_kk[j * K + 0];
        }
    }
    
    for (int i = 0; i < K; i++) {
        tmp_k2[i] = 0.0;
        for (int j = 0; j < K; j++) {
            tmp_k2[i] += v_kk[i * K + j] * tmp_k[j];
        }
    }
    
    double norm1 = 0.0;
    for (int i = 0; i < K; i++) {
        norm1 += fabs(tmp_k2[i]);
    }
    
    double mu = (norm1 - 2.0 * fabs(tmp_k2[0])) / K;
    mu = 2.0 * mu;
    if (mu < 0.0) {
        mu = 0.0;
    }
    
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < K; j++) {
            tableau[i * sz + j] = matrix_q[i * K + j] + mu;
        }
    }
    
    for (int i = 0; i < K; i++) {
        tableau[i * sz + K] = 1.0;
        tableau[K * sz + i] = 1.0;
        tableau[i * sz + (K + 1)] = -r[i];
        tableau[(K + 1) * sz + i] = -r[i];
    }
    
    tableau[K * sz + K] = 0.0;
    
    double sum_x = 0.0;
    for (int i = 0; i < K; i++) {
        sum_x += x[i];
    }
    
    tableau[K * sz + (K + 1)] = 1.0 - sum_x;
    tableau[(K + 1) * sz + K] = 1.0 - sum_x;
    tableau[(K + 1) * sz + (K + 1)] = 0.0;
}

__device__ void create_tableau_box_device(
    double* tableau, const double* matrix_q, const double* r,
    const double* x, int K)
{
    int sz = K + 1;
    for (int i = 0; i < sz * sz; i++) {
        tableau[i] = 0.0;
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
    
    tableau[K * sz + K] = 0.0;
}

__global__ void sqp_solve_q_kernel(
    const double* __restrict__ XtX_q,
    const double* __restrict__ Xtz_q,
    const double* __restrict__ Q,
    double* __restrict__ Q_next,
    const double* __restrict__ v_kk,
    int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    double tableau[102 * 102];
    double d_buf[102];
    double tmp_buf[102];
    int swept[100];
    double delta[100];
    
    create_tableau_simplex_device(tableau, XtX_q + i * K * K, Xtz_q + i * K, Q + i * K, v_kk, K);
    quadratic_program_device(delta, tableau, Q + i * K, K, 1, d_buf, tmp_buf, swept);
    
    for (int k = 0; k < K; k++) {
        Q_next[i * K + k] = Q[i * K + k] + delta[k];
    }
    project_q_simplex_row(Q_next + i * K, K, 1e-5);
}

__global__ void sqp_solve_p_kernel(
    const double* __restrict__ XtX_p,
    const double* __restrict__ Xtz_p,
    const double* __restrict__ P,
    double* __restrict__ P_next,
    int M, int K)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;
    
    double tableau[102 * 102];
    double d_buf[102];
    double tmp_buf[102];
    int swept[100];
    double delta[100];
    
    create_tableau_box_device(tableau, XtX_p + j * K * K, Xtz_p + j * K, P + j * K, K);
    quadratic_program_device(delta, tableau, P + j * K, K, 0, d_buf, tmp_buf, swept);
    
    for (int k = 0; k < K; k++) {
        P_next[j * K + k] = P[j * K + k] + delta[k];
    }
    project_p_box_row(P_next + j * K, K, 1e-5);
}

static inline int sqp_threads_for_k(int64_t K) {
    if (K <= 8) return 256;
    if (K <= 16) return 128;
    if (K <= 32) return 64;
    return 32;
}

torch::Tensor sqp_solve_q_cuda(
    const torch::Tensor& XtX_q,
    const torch::Tensor& Xtz_q,
    const torch::Tensor& Q,
    const torch::Tensor& v_kk,
    int64_t N, int64_t K)
{
    TORCH_CHECK(K >= 1 && K <= 100, "K must be between 1 and 100");
    
    auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(Q.device());
    torch::Tensor Q_next = torch::empty({N, K}, opts);
    
    int threads = sqp_threads_for_k(K);
    int blocks = (N + threads - 1) / threads;
    
    sqp_solve_q_kernel<<<blocks, threads>>>(
        XtX_q.data_ptr<double>(),
        Xtz_q.data_ptr<double>(),
        Q.data_ptr<double>(),
        Q_next.data_ptr<double>(),
        v_kk.data_ptr<double>(),
        (int)N, (int)K
    );
    
    return Q_next;
}

torch::Tensor sqp_solve_p_cuda(
    const torch::Tensor& XtX_p,
    const torch::Tensor& Xtz_p,
    const torch::Tensor& P,
    int64_t M, int64_t K)
{
    TORCH_CHECK(K >= 1 && K <= 100, "K must be between 1 and 100");
    
    auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(P.device());
    torch::Tensor P_next = torch::empty({M, K}, opts);
    
    int threads = sqp_threads_for_k(K);
    int blocks = (M + threads - 1) / threads;
    
    sqp_solve_p_kernel<<<blocks, threads>>>(
        XtX_p.data_ptr<double>(),
        Xtz_p.data_ptr<double>(),
        P.data_ptr<double>(),
        P_next.data_ptr<double>(),
        (int)M, (int)K
    );
    
    return P_next;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sqp_solve_q_cuda", &sqp_solve_q_cuda, "SQP solve Q (CUDA)");
    m.def("sqp_solve_p_cuda", &sqp_solve_p_cuda, "SQP solve P (CUDA)");
}

TORCH_LIBRARY(sqp_kernel, m) {
    m.def("sqp_solve_q_cuda(Tensor XtX_q, Tensor Xtz_q, Tensor Q, Tensor v_kk, int N, int K) -> Tensor");
    m.def("sqp_solve_p_cuda(Tensor XtX_p, Tensor Xtz_p, Tensor P, int M, int K) -> Tensor");
}

TORCH_LIBRARY_IMPL(sqp_kernel, CUDA, m) {
    m.impl("sqp_solve_q_cuda", TORCH_FN(sqp_solve_q_cuda));
    m.impl("sqp_solve_p_cuda", TORCH_FN(sqp_solve_p_cuda));
}
