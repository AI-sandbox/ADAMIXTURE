from __future__ import annotations

from functools import lru_cache
from importlib import resources

import torch


def _read_shader(name: str) -> str:
    return resources.files(__package__).joinpath(name).read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def _shader(name: str):
    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("torch.mps.compile_shader is required for Metal kernels")
    return torch.mps.compile_shader(_read_shader(name))


def _check_mps_float32(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.device.type != "mps":
            raise ValueError(f"Expected an MPS tensor, got {tensor.device}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"Expected float32 tensor for MPS Metal kernel, got {tensor.dtype}")


def sqp_solve_q_mps(
    XtX_q: torch.Tensor,
    Xtz_q: torch.Tensor,
    Q: torch.Tensor,
    v_kk: torch.Tensor,
    N: int,
    K: int,
) -> torch.Tensor:
    _check_mps_float32(XtX_q, Xtz_q, Q, v_kk)
    if not (1 <= K <= 32):
        raise ValueError("K must be between 1 and 32 for the Metal SQP kernel")
    Q_next = torch.empty((N, K), dtype=torch.float32, device=Q.device)
    _shader("sqp_kernel.metal").sqp_solve_q_kernel(Q_next, XtX_q, Xtz_q, Q, v_kk, int(N), int(K))
    return Q_next


def sqp_solve_p_mps(
    XtX_p: torch.Tensor,
    Xtz_p: torch.Tensor,
    P: torch.Tensor,
    M: int,
    K: int,
) -> torch.Tensor:
    _check_mps_float32(XtX_p, Xtz_p, P)
    if not (1 <= K <= 32):
        raise ValueError("K must be between 1 and 32 for the Metal SQP kernel")
    P_next = torch.empty((M, K), dtype=torch.float32, device=P.device)
    _shader("sqp_kernel.metal").sqp_solve_p_kernel(P_next, XtX_p, Xtz_p, P, int(M), int(K))
    return P_next


def unpack2bit_gpu_chunk_uint8_mps(
    input_mps: torch.Tensor,
    m_start: int,
    chunk_size: int,
    M_total: int,
    byte_offset: int,
) -> torch.Tensor:
    if input_mps.device.type != "mps" or input_mps.dtype != torch.uint8:
        raise ValueError("Expected an MPS uint8 tensor")
    N_total = input_mps.size(1)
    actual_chunk = min(chunk_size, M_total - m_start)
    if actual_chunk <= 0:
        return torch.empty((0, N_total), dtype=torch.uint8, device=input_mps.device)
    output = torch.empty((actual_chunk, N_total), dtype=torch.uint8, device=input_mps.device)
    _shader("pack2bit.metal").unpack2bit_chunk_kernel_uint8(
        output, input_mps, int(m_start), int(actual_chunk), int(M_total), int(byte_offset), int(N_total)
    )
    return output


def unpack2bit_gpu_chunk_center_mps(
    input_mps: torch.Tensor,
    f_mps: torch.Tensor,
    m_start: int,
    chunk_size: int,
    M_total: int,
    byte_offset: int,
) -> torch.Tensor:
    if input_mps.device.type != "mps" or input_mps.dtype != torch.uint8:
        raise ValueError("Expected an MPS uint8 genotype tensor")
    if f_mps.device.type != "mps" or f_mps.dtype != torch.float32:
        raise ValueError("Expected an MPS float32 frequency tensor")
    N_total = input_mps.size(1)
    actual_chunk = min(chunk_size, M_total - m_start)
    if actual_chunk <= 0:
        return torch.empty((0, N_total), dtype=torch.float32, device=input_mps.device)
    output = torch.empty((actual_chunk, N_total), dtype=torch.float32, device=input_mps.device)
    _shader("pack2bit.metal").unpack2bit_chunk_kernel_center(
        output, input_mps, f_mps, int(m_start), int(actual_chunk), int(M_total), int(byte_offset), int(N_total)
    )
    return output


def batch_bvls_bpp_mps(A: torch.Tensor, B: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    _check_mps_float32(A, B)
    M = B.size(0)
    K = B.size(1)
    if K > 64:
        raise ValueError("K exceeds the current Metal BVLS limit (64)")
    if A.shape != (K, K):
        raise ValueError("A must be K x K")
    A_inv = torch.linalg.inv(A)
    X0 = torch.matmul(B, A_inv.transpose(0, 1)).contiguous()
    X = torch.empty((M, K), dtype=torch.float32, device=B.device)
    _shader("bvls_kernel.metal").bvls_bpp_kernel(X, A.contiguous(), B.contiguous(), X0, int(M), int(K), float(lower), float(upper))
    return X
