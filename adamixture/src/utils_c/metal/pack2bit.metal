#include <metal_stdlib>
using namespace metal;

kernel void unpack2bit_chunk_kernel_uint8(
    device uchar* output [[buffer(0)]],
    device const uchar* input [[buffer(1)]],
    constant int& m_start [[buffer(2)]],
    constant int& chunk_size [[buffer(3)]],
    constant int& M_total [[buffer(4)]],
    constant int& byte_offset [[buffer(5)]],
    constant int& N_total [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    uint n_total = (uint)N_total;
    uint total_elems = (uint)chunk_size * n_total;
    if (idx >= total_elems) return;

    uint local_m = idx / n_total;
    uint n = idx - local_m * n_total;
    uint global_m = (uint)m_start + local_m;
    if (global_m >= (uint)M_total) return;

    uint row_idx_full = global_m >> 2u;
    if (row_idx_full < (uint)byte_offset) return;
    uint row_idx = row_idx_full - (uint)byte_offset;

    uchar bit_shift = (uchar)((global_m & 3u) << 1u);
    uchar packed = input[row_idx * n_total + n];
    output[idx] = (packed >> bit_shift) & 0x03u;
}

kernel void unpack2bit_chunk_kernel_center(
    device float* output [[buffer(0)]],
    device const uchar* input [[buffer(1)]],
    device const float* f [[buffer(2)]],
    constant int& m_start [[buffer(3)]],
    constant int& chunk_size [[buffer(4)]],
    constant int& M_total [[buffer(5)]],
    constant int& byte_offset [[buffer(6)]],
    constant int& N_total [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    uint n_total = (uint)N_total;
    uint total_elems = (uint)chunk_size * n_total;
    if (idx >= total_elems) return;

    uint local_m = idx / n_total;
    uint n = idx - local_m * n_total;
    uint global_m = (uint)m_start + local_m;
    if (global_m >= (uint)M_total) return;

    uint row_idx_full = global_m >> 2u;
    if (row_idx_full < (uint)byte_offset) return;
    uint row_idx = row_idx_full - (uint)byte_offset;

    uchar bit_shift = (uchar)((global_m & 3u) << 1u);
    uchar packed = input[row_idx * n_total + n];
    uchar val = (packed >> bit_shift) & 0x03u;

    output[idx] = (val == 3u) ? 0.0f : ((float)val - 2.0f * f[global_m]);
}
