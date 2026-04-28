#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_K 64
#define BPP_MAX_ITER 50
#define SINGULAR_TOL 1e-12
#define PIVOT_TOL 1e-8

// Gaussian elimination with partial pivoting on the Free subset.
// Mirrors solve_subset_ptr from tools.pyx exactly.
__device__ bool solve_subset(
    const double* __restrict__ A, const double* b, double* x,
    const int* F, int K, double* aug, int* map_idx)
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
        for (int j = 0; j < n; j++)
            aug[i * (n + 1) + j] = A[map_idx[i] * K + map_idx[j]];
        aug[i * (n + 1) + n] = b[map_idx[i]];
    }

    for (int i = 0; i < n; i++) {
        int max_row = i;
        double max_val = aug[i * (n + 1) + i];
        if (max_val < 0.0) max_val = -max_val;

        for (int k = i + 1; k < n; k++) {
            double tmp = aug[k * (n + 1) + i];
            if (tmp < 0.0) tmp = -tmp;
            if (tmp > max_val) {
                max_val = tmp;
                max_row = k;
            }
        }

        if (max_val < SINGULAR_TOL) return false;

        if (max_row != i) {
            for (int j = i; j <= n; j++) {
                double tmp = aug[i * (n + 1) + j];
                aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j];
                aug[max_row * (n + 1) + j] = tmp;
            }
        }

        double pivot = aug[i * (n + 1) + i];
        for (int j = i; j <= n; j++)
            aug[i * (n + 1) + j] /= pivot;

        for (int k = 0; k < n; k++) {
            if (k != i) {
                double tmp = aug[k * (n + 1) + i];
                for (int j = i; j <= n; j++)
                    aug[k * (n + 1) + j] -= tmp * aug[i * (n + 1) + j];
            }
        }
    }

    for (int i = 0; i < n; i++)
        x[map_idx[i]] = aug[i * (n + 1) + n];
    return true;
}

// One thread per row: exact BVLS via Block Principal Pivoting.
// Mirrors _exact_bvls_bpp_ptr from tools.pyx exactly.
__global__ void bvls_bpp_kernel(
    const double* __restrict__ A,   // K x K (shared across all rows)
    const double* __restrict__ B,   // M x K
    double* __restrict__ X,         // M x K (output)
    int M, int K,
    double lower, double upper)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    const double* b = B + row * K;
    double* x = X + row * K;

    double aug[MAX_K * (MAX_K + 1)];
    int    map_idx[MAX_K];
    int    F[MAX_K];
    int    U[MAX_K];
    double y[MAX_K];
    double b_subset[MAX_K];

    for (int i = 0; i < K; i++) {
        F[i] = 1;
        U[i] = 0;
        x[i] = 0.0;
    }

    for (int iter = 0; iter < BPP_MAX_ITER; iter++) {
        // Adjust b for fixed variables
        for (int i = 0; i < K; i++) {
            if (F[i] == 1) {
                b_subset[i] = b[i];
                for (int j = 0; j < K; j++) {
                    if (F[j] == 0) {
                        if (U[j] == 1)
                            b_subset[i] -= A[i * K + j] * upper;
                        else
                            b_subset[i] -= A[i * K + j] * lower;
                    }
                }
            }
        }

        if (!solve_subset(A, b_subset, x, F, K, aug, map_idx))
            break;

        // Set x for non-free variables
        for (int i = 0; i < K; i++) {
            if (F[i] == 0) {
                x[i] = (U[i] == 1) ? upper : lower;
            }
        }

        // Gradient: y = Ax - b
        for (int i = 0; i < K; i++) {
            y[i] = -b[i];
            for (int j = 0; j < K; j++)
                y[i] += A[i * K + j] * x[j];
        }

        // Pivoting
        bool changed = false;
        for (int i = 0; i < K; i++) {
            if (F[i] == 1) {
                if (x[i] < lower - PIVOT_TOL) {
                    F[i] = 0; U[i] = 0; changed = true;
                } else if (x[i] > upper + PIVOT_TOL) {
                    F[i] = 0; U[i] = 1; changed = true;
                }
            } else {
                if (U[i] == 0 && y[i] < -PIVOT_TOL) {
                    F[i] = 1; U[i] = 0; changed = true;
                } else if (U[i] == 1 && y[i] > PIVOT_TOL) {
                    F[i] = 1; U[i] = 0; changed = true;
                }
            }
        }

        if (!changed) break;
    }
}

// Host wrapper
torch::Tensor batch_bvls_bpp_cuda(
    const torch::Tensor& A,     // K x K
    const torch::Tensor& B,     // M x K
    double lower, double upper)
{
    int M = B.size(0);
    int K = B.size(1);

    TORCH_CHECK(K <= MAX_K, "K=", K, " exceeds MAX_K=", MAX_K);
    TORCH_CHECK(A.size(0) == K && A.size(1) == K, "A must be K x K");

    auto opts = torch::TensorOptions().dtype(torch::kFloat64).device(B.device());
    torch::Tensor X = torch::zeros({M, K}, opts);

    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    bvls_bpp_kernel<<<blocks, threads>>>(
        A.data_ptr<double>(),
        B.data_ptr<double>(),
        X.data_ptr<double>(),
        M, K, lower, upper
    );

    return X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_bvls_bpp_cuda", &batch_bvls_bpp_cuda, "Batch BVLS via Block Principal Pivoting (CUDA)");
}

TORCH_LIBRARY(bvls_kernel, m) {
    m.def("batch_bvls_bpp_cuda(Tensor A, Tensor B, float lower, float upper) -> Tensor");
}

TORCH_LIBRARY_IMPL(bvls_kernel, CUDA, m) {
    m.impl("batch_bvls_bpp_cuda", TORCH_FN(batch_bvls_bpp_cuda));
}
