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

kernel void em_ab_packed_f32(
    device float* a_out [[buffer(0)]],
    device float* b_out [[buffer(1)]],
    device const uchar* packed_g [[buffer(2)]],
    device const float* rec_in [[buffer(3)]],
    constant int& m_start [[buffer(4)]],
    constant int& chunk_rows [[buffer(5)]],
    constant int& byte_offset [[buffer(6)]],
    constant int& N_total [[buffer(7)]],
    uint idx [[thread_position_in_grid]])
{
    uint n_tot = (uint)N_total;
    uint total = (uint)chunk_rows * n_tot;
    if (idx >= total) return;

    uint lm = idx / n_tot;
    uint n = idx - lm * n_tot;
    uint gm = (uint)m_start + lm;
    uint row_idx = (gm >> 2u) - (uint)byte_offset;
    uchar shift = (uchar)((gm & 3u) << 1u);
    uchar u = (packed_g[row_idx * n_tot + n] >> shift) & 0x03u;
    if (u == 3u) {
        a_out[idx] = 0.0f;
        b_out[idx] = 0.0f;
        return;
    }

    float r = clamp(rec_in[idx], 1e-5f, 1.0f - 1e-5f);
    float omr = 1.0f - r;
    float inv = 1.0f / (r * omr);
    float gf = (float)u;
    a_out[idx] = gf * omr * inv;
    b_out[idx] = (2.0f - gf) * r * inv;
}
