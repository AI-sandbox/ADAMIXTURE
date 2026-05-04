#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// Mask held-out entries in packed G: save original 2-bit values, set to 3.
// Uses atomicOr for thread-safe byte modification when multiple entries
// share the same packed byte (up to 4 SNPs per byte per sample).
__global__ void mask_entries_packed_kernel(
    uint8_t* G_packed,
    const int64_t* __restrict__ flat_indices,
    uint8_t* __restrict__ saved_values,
    int64_t num_entries,
    int64_t N)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    int64_t flat_idx = flat_indices[tid];
    int64_t snp    = flat_idx / N;
    int64_t sample = flat_idx % N;

    int64_t byte_row = snp >> 2;
    int geno_bit_off = (int)(snp & 3) << 1;   // 0, 2, 4, or 6
    int64_t packed_linear = byte_row * N + sample;

    // Save original 2-bit genotype (thread-unique position, safe non-atomic read)
    uint8_t orig = G_packed[packed_linear];
    saved_values[tid] = (orig >> geno_bit_off) & 0x03;

    // Set to missing (0b11) via atomic OR on the containing 32-bit word
    unsigned int* word_addr =
        (unsigned int*)((size_t)(G_packed + packed_linear) & ~(size_t)3);
    unsigned int byte_in_word = (unsigned int)((size_t)(G_packed + packed_linear) & 3);
    unsigned int or_mask = (unsigned int)(0x03 << geno_bit_off) << (byte_in_word * 8);
    atomicOr(word_addr, or_mask);
}

// Restore held-out entries in packed G from saved 2-bit values.
// Uses atomicCAS loop for thread-safe clear-and-set on shared bytes.
__global__ void restore_entries_packed_kernel(
    uint8_t* G_packed,
    const int64_t* __restrict__ flat_indices,
    const uint8_t* __restrict__ saved_values,
    int64_t num_entries,
    int64_t N)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    int64_t flat_idx = flat_indices[tid];
    int64_t snp    = flat_idx / N;
    int64_t sample = flat_idx % N;

    int64_t byte_row = snp >> 2;
    int geno_bit_off = (int)(snp & 3) << 1;
    int64_t packed_linear = byte_row * N + sample;

    uint8_t val = saved_values[tid] & 0x03;

    unsigned int* word_addr =
        (unsigned int*)((size_t)(G_packed + packed_linear) & ~(size_t)3);
    unsigned int byte_in_word = (unsigned int)((size_t)(G_packed + packed_linear) & 3);
    unsigned int bit_shift  = byte_in_word * 8 + geno_bit_off;
    unsigned int clear_mask = ~((unsigned int)0x03 << bit_shift);
    unsigned int set_bits   = (unsigned int)val << bit_shift;

    unsigned int assumed, old_word = *word_addr;
    do {
        assumed = old_word;
        old_word = atomicCAS(word_addr, assumed, (assumed & clear_mask) | set_bits);
    } while (assumed != old_word);
}

// ================================
// HOST WRAPPERS
// ================================

torch::Tensor mask_entries_packed_cuda(
    torch::Tensor G_packed,
    const torch::Tensor& flat_indices,
    int64_t N)
{
    int64_t num = flat_indices.size(0);
    auto saved = torch::empty({num},
        torch::dtype(torch::kUInt8).device(flat_indices.device()));
    if (num == 0) return saved;

    int threads = 256;
    int blocks  = (int)((num + threads - 1) / threads);

    mask_entries_packed_kernel<<<blocks, threads>>>(
        G_packed.data_ptr<uint8_t>(),
        flat_indices.data_ptr<int64_t>(),
        saved.data_ptr<uint8_t>(),
        num, N);
    return saved;
}

void restore_entries_packed_cuda(
    torch::Tensor G_packed,
    const torch::Tensor& flat_indices,
    const torch::Tensor& saved_values,
    int64_t N)
{
    int64_t num = flat_indices.size(0);
    if (num == 0) return;

    int threads = 256;
    int blocks  = (int)((num + threads - 1) / threads);

    restore_entries_packed_kernel<<<blocks, threads>>>(
        G_packed.data_ptr<uint8_t>(),
        flat_indices.data_ptr<int64_t>(),
        saved_values.data_ptr<uint8_t>(),
        num, N);
}

// ================================
// PYBIND11 + TORCH DISPATCH
// ================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mask_entries_packed_cuda",    &mask_entries_packed_cuda,
          "Mask held-out entries in packed G (save + set missing)");
    m.def("restore_entries_packed_cuda", &restore_entries_packed_cuda,
          "Restore held-out entries in packed G from saved values");
}

TORCH_LIBRARY(cv_mask_kernel, m) {
    m.def("mask_entries_packed_cuda(Tensor G_packed, Tensor flat_indices, int N) -> Tensor");
    m.def("restore_entries_packed_cuda(Tensor G_packed, Tensor flat_indices, Tensor saved_values, int N) -> ()");
}

TORCH_LIBRARY_IMPL(cv_mask_kernel, CUDA, m) {
    m.impl("mask_entries_packed_cuda",    TORCH_FN(mask_entries_packed_cuda));
    m.impl("restore_entries_packed_cuda", TORCH_FN(restore_entries_packed_cuda));
}
