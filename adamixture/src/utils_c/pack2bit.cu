#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdint>
#include <string>

// ================================
// KERNELS (Packed G is M_bytes x N)
// ================================

// input: (M_bytes, N), output: (chunk_size, N)
// m_start is the index of the first SNP in the chunk
__global__ void unpack2bit_chunk_kernel_uint8_SNPsPacked(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint64_t M_total, uint64_t N_total,
    int64_t m_start, int64_t chunk_size,
    int64_t byte_offset, int64_t threads_per_block)
{
    uint64_t tid = blockIdx.x * threads_per_block + threadIdx.x;
    uint64_t total_threads = gridDim.x * threads_per_block;
    uint64_t total_elems = (uint64_t)chunk_size * N_total;

    for (uint64_t idx = tid; idx < total_elems; idx += total_threads) {
        uint64_t local_m = idx / N_total;
        uint64_t n = idx % N_total;
        
        uint64_t global_m = (uint64_t)m_start + local_m;
        if (global_m >= M_total) continue;

        uint64_t row_idx_full = global_m >> 2ULL; // m // 4
        if (row_idx_full < (uint64_t)byte_offset) continue;
        uint64_t row_idx = row_idx_full - (uint64_t)byte_offset;

        uint8_t bit_shift = (global_m & 3ULL) << 1ULL; // (m % 4) * 2
        
        uint8_t packed = input[row_idx * N_total + n];
        uint8_t val = (packed >> bit_shift) & 0x03u;
        
        output[idx] = val;
    }
}

__global__ void unpack2bit_chunk_kernel_center_SNPsPacked(
    const uint8_t* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ f,
    uint64_t M_total, uint64_t N_total,
    int64_t m_start, int64_t chunk_size,
    int64_t byte_offset, int64_t threads_per_block)
{
    uint64_t tid = blockIdx.x * threads_per_block + threadIdx.x;
    uint64_t total_threads = gridDim.x * threads_per_block;
    uint64_t total_elems = (uint64_t)chunk_size * N_total;

    for (uint64_t idx = tid; idx < total_elems; idx += total_threads) {
        uint64_t local_m = idx / N_total;
        uint64_t n = idx % N_total;
        
        uint64_t global_m = (uint64_t)m_start + local_m;
        if (global_m >= M_total) continue;

        uint64_t row_idx_full = global_m >> 2ULL;
        if (row_idx_full < (uint64_t)byte_offset) continue;
        uint64_t row_idx = row_idx_full - (uint64_t)byte_offset;

        uint8_t bit_shift = (global_m & 3ULL) << 1ULL;
        
        uint8_t packed = input[row_idx * N_total + n];
        uint8_t val = (packed >> bit_shift) & 0x03u;

        float out_val = (val == 3u) ? 0.0f : ((float)val - 2.0f * f[global_m]);
        output[idx] = out_val;
    }
}

// ================================
// CUDA WRAPPERS (host-side)
// ================================

torch::Tensor unpack2bit_gpu_chunk_uint8(
    const torch::Tensor& input_gpu, int64_t m_start, int64_t chunk_size,
    int64_t M_total, int64_t byte_offset, int64_t threads_per_block)
{
    int64_t N_total = input_gpu.size(1);
    int64_t actual_chunk = std::min(chunk_size, M_total - m_start);
    
    if (actual_chunk <= 0) {
        return torch::empty({0, N_total}, torch::dtype(torch::kUInt8).device(input_gpu.device()));
    }

    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(input_gpu.device());
    torch::Tensor output = torch::empty({actual_chunk, N_total}, opts);

    uint64_t total_elems = (uint64_t)actual_chunk * (uint64_t)N_total;
    uint32_t blocks = (uint32_t)((total_elems + threads_per_block - 1) / threads_per_block);
    blocks = std::min(blocks, 65535u);

    unpack2bit_chunk_kernel_uint8_SNPsPacked<<<blocks, (uint32_t)threads_per_block>>>(
        input_gpu.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        (uint64_t)M_total, (uint64_t)N_total,
        m_start, actual_chunk,
        byte_offset, threads_per_block
    );

    return output;
}

torch::Tensor unpack2bit_gpu_chunk_center(
    const torch::Tensor& input_gpu, const torch::Tensor& f_gpu,
    int64_t m_start, int64_t chunk_size,
    int64_t M_total, int64_t byte_offset, int64_t threads_per_block)
{
    int64_t N_total = input_gpu.size(1);
    int64_t actual_chunk = std::min(chunk_size, M_total - m_start);

    if (actual_chunk <= 0) {
        return torch::empty({0, N_total}, torch::dtype(torch::kFloat32).device(input_gpu.device()));
    }

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(input_gpu.device());
    torch::Tensor output = torch::empty({actual_chunk, N_total}, opts);

    uint64_t total_elems = (uint64_t)actual_chunk * (uint64_t)N_total;
    uint32_t blocks = (uint32_t)((total_elems + threads_per_block - 1) / threads_per_block);
    blocks = std::min(blocks, 65535u);

    unpack2bit_chunk_kernel_center_SNPsPacked<<<blocks, (uint32_t)threads_per_block>>>(
        input_gpu.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        f_gpu.data_ptr<float>(),
        (uint64_t)M_total, (uint64_t)N_total,
        m_start, actual_chunk,
        byte_offset, threads_per_block
    );

    return output;
}

// ================================
// PYBIND11 + TORCH DISPATCH
// ================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("unpack2bit_gpu_chunk_center", &unpack2bit_gpu_chunk_center, "Unpack + center");
    m.def("unpack2bit_gpu_chunk_uint8", &unpack2bit_gpu_chunk_uint8, "Unpack to uint8");
}

TORCH_LIBRARY(pack2bit, m) {
    m.def("unpack2bit_gpu_chunk_center(Tensor input, Tensor f, int m_start, int chunk_size, int M_total, int byte_offset, int threads_per_block) -> Tensor");
    m.def("unpack2bit_gpu_chunk_uint8(Tensor input, int m_start, int chunk_size, int M_total, int byte_offset, int threads_per_block) -> Tensor");
}

TORCH_LIBRARY_IMPL(pack2bit, CUDA, m) {
    m.impl("unpack2bit_gpu_chunk_center", TORCH_FN(unpack2bit_gpu_chunk_center));
    m.impl("unpack2bit_gpu_chunk_uint8", TORCH_FN(unpack2bit_gpu_chunk_uint8));
}
