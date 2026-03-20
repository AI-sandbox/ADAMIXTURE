import logging
import random
import sys
import torch
import numpy as np

from pathlib import Path

from .snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def read_data(tr_file: str, packed: bool = False) -> tuple[torch.Tensor | np.ndarray, int, int]:
    """
    Description:
    Reads SNP data from a file (BED, VCF, etc.) and returns the genotype matrix and dimensions.

    Args:
        tr_file (str): Path to the SNP data file.
        packed (bool): If True, return a 2-bit packed torch.Tensor. Defaults to False.

    Returns:
        tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N samples, M SNPs)
    """
    snp_reader = SNPReader()
    G, N, M = snp_reader.read_data(tr_file, packed=packed)
    log.info(f"    Data contains {N} samples and {M} SNPs.")
   
    return G, N, M

def get_tuning_params(device: torch.device, verbose: bool = True) -> int:
    """
    Description:
    Returns optimal CUDA kernel parameters (threads_per_block) based on the device properties.

    Args:
        device (torch.device): The target computation device.
        verbose (bool, optional): Whether to log the detected device and parameters. Defaults to True.

    Returns:
        int: Number of threads per block for CUDA operations.
    """
    if device.type == "cpu":
        threads_per_block = 1
    elif device.type == "cuda":
        major = torch.cuda.get_device_properties(device.index).major
        if major >= 8:  # Ampere or newer
            threads_per_block = 512
        elif major >= 7:  # Volta / Turing
            threads_per_block = 256
        else:
            threads_per_block = 128
    elif device.type == "mps":
        threads_per_block = 64
    else:
        threads_per_block = 128

    return threads_per_block

def write_outputs(Q: np.ndarray, run_name: str, K: int, out_path: str, P: np.ndarray = None) -> None:
    """
    Description:
    Saves the inferred ancestry proportions (Q) and optionally the allele frequencies (P).

    Args:
        Q (np.ndarray): Q matrix to be saved.
        run_name (str): Identifier for the run, used in file naming.
        K (int): Number of populations, included in the file name.
        out_path (str | Path): Directory where the output files should be saved.
        P (np.ndarray, optional): P matrix to be saved. Defaults to None.

    Returns:
        None
    """
    out_path = Path(out_path)
    np.savetxt(out_path/f"{run_name}.{K}.Q", Q, delimiter=' ')
    if P is not None:
        np.savetxt(out_path/f"{run_name}.{K}.P", P, delimiter=' ')
        log.info("    Q and P matrices saved.")
    else:
        log.info("    Q matrix saved.")
    return

def set_seed(seed: int) -> None:
    """
    Description:
    Sets the random seed for NumPy, Python's random, and PyTorch (CPU and CUDA).

    Args:
        seed (int): The seed value to use.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_free_gpu_memory(device: torch.device) -> float:
    """
    Description:
    Calculates the currently available free GPU memory for tensor allocation.

    Args:
        device (torch.device): The target GPU device.

    Returns:
        float: Free memory in megabytes (MB).
    """
    device = torch.device(device)
    torch.cuda.synchronize(device)
    free_cuda, _ = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free_for_tensors = free_cuda + (reserved - allocated)
    return round(free_for_tensors/(1024 ** 2), 2)

def manage_gpu_memory(G: torch.Tensor | np.ndarray, device: torch.device, M: int, N: int, K: int, chunk_size: int) -> torch.Tensor:
    """
    Description:
    Determines if the genotype matrix fits in GPU memory and moves it if possible.
    Otherwise, leaves it on the CPU for streaming.

    Args:
        G (torch.Tensor | np.ndarray): Packed or unpacked genotype matrix.
        device (torch.device): Target computation device.
        M (int): Number of individuals.
        N (int): Number of SNPs.
        K (int): Number of ancestral populations.
        chunk_size (int): Expected batch size for computations.

    Returns:
        torch.Tensor: Genotype tensor on the selected device.
    """
    if device.type == 'mps':
        # Always keep on CPU if MPS, but convert to torch tensor if it's numpy
        if isinstance(G, np.ndarray):
            return torch.from_numpy(G)
        return G

    if device.type != 'cuda':
        return G
    
    memory_GPU = get_free_gpu_memory(device)
    
    bytes_per_float32 = 4
    bytes_per_float64 = 8
    
    L = max(K + 10, 20)
    
    # RSVD Matrices (approx peak)
    U_bytes = M * L * bytes_per_float32
    V_bytes = N * L * bytes_per_float64
    S_bytes = L * bytes_per_float64
    Qmat_bytes = N * L * bytes_per_float64
    Usmall_bytes = M * L * bytes_per_float32
    rsvd_total = U_bytes + S_bytes + V_bytes + Qmat_bytes + Usmall_bytes
    
    # EM-Adam matrices
    P_bytes = M * K * bytes_per_float32
    Q_bytes = N * K * bytes_per_float32
    adam_bytes = 4 * M * K * bytes_per_float32 # approx A, B, targets
    
    peak_memory_MB = max(rsvd_total, P_bytes + Q_bytes + adam_bytes) / (1024 ** 2)
    
    # 2-bit Data tensor size:
    memory_data_MB = (torch.numel(G)) / (1024 ** 2)
    
    # Chunk memory (chunk_size, N float32 during unpacking)
    memory_chunk_MB = chunk_size * N * bytes_per_float32 / (1024 ** 2)
    
    # Total required with some buffer
    total_required_MB = memory_data_MB + peak_memory_MB + memory_chunk_MB
    
    if memory_GPU * 0.95 - total_required_MB > 0:
        log.info(f"    Moving genotype matrix to GPU...")
        G = G.to(device)
    else:
        log.info(f"    Genotype matrix too large for GPU, keeping on CPU...")
    
    return G

def load_extensions(device: torch.device) -> None:
    """
    Description:
    Dynamically compiles and loads the `pack2bit` CUDA extension using Ninja.

    Args:
        device (torch.device): The computation device. Triggered only if 'cuda'.

    Returns:
        None
    """
    if device.type == 'cuda':
        import os
        from torch.utils.cpp_extension import load
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_path = os.path.abspath(os.path.join(current_dir, "utils_c", "pack2bit.cu"))
        
        if not os.path.exists(source_path):
            log.error(f"CUDA source file not found: {source_path}")
            return

        log.info("    Loading CUDA extensions...")
        load(name="pack2bit", 
             sources=[source_path], 
             verbose=False, keep_intermediates=False, 
             extra_cuda_cflags=['-O3', '--use_fast_math'], extra_cflags=['-O3'])

def get_unpacker(device: torch.device, threads_per_block: int):
    """
    Description:
    Returns a specialized function for unpacking genotype chunks based on the device.
    This eliminates repeated 'if device.type == ...' checks in tight loops.

    Args:
        device (torch.device): Target computation device.
        threads_per_block (int): Threads per block for CUDA operations.

    Returns:
        Callable: A function with signature (G, start_idx, actual_chunk_size, M) -> torch.Tensor.
    """
    if device.type == 'mps':
        def unpack_mps(G, start_idx, actual_chunk_size, M):
            return G[start_idx:start_idx + actual_chunk_size, :].to(device, non_blocking=True)
        return unpack_mps
    
    def unpack_cuda(G, start_idx, actual_chunk_size, M):
        if G.device.type == 'cpu':
            byte_start = start_idx // 4
            byte_end = (start_idx + actual_chunk_size + 3) // 4
            G_sub = G[byte_start:byte_end, :].to(device, non_blocking=True)
            return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G_sub, start_idx, actual_chunk_size, M, byte_start, threads_per_block)
        return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G, start_idx, actual_chunk_size, M, 0, threads_per_block)
    
    return unpack_cuda

def get_logl_calculator(device: torch.device):
    """
    Description:
    Returns the optimal log-likelihood calculation function for the given device.
    Handles the high-precision requirements of log-likelihood by falling back to CPU for MPS.

    Args:
        device (torch.device): Target computation device.

    Returns:
        Callable: A function with signature (G, P, Q, M, N, batch_size, threads_per_block) -> float.
    """
    if device.type == 'mps':
        from .utils_c import tools
        def logl_mps(G, P, Q, M, N, batch_size, threads_per_block):
            G_cpu = G.numpy() if isinstance(G, torch.Tensor) else G
            return tools.loglikelihood(G_cpu, P.cpu().numpy().astype(np.float64), Q.cpu().numpy().astype(np.float64))
        return logl_mps
    
    from ..model.em_adam_gpu import loglikelihood_gpu
    return loglikelihood_gpu

def _unpack_chunk_uint8(G: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int, device: torch.device, threads_per_block: int) -> torch.Tensor:
    """
    Description:
    Legacy wrapper for get_unpacker for backward compatibility.
    """
    unpacker = get_unpacker(device, threads_per_block)
    return unpacker(G, start_idx, actual_chunk_size, M)

def get_centering_unpacker(device: torch.device, threads_per_block: int):
    """
    Description:
    Returns a specialized function for unpacking and centering genotype chunks.
    Essential for SVD performance on both CUDA and MPS.

    Args:
        device (torch.device): Target computation device.
        threads_per_block (int): Threads per block for CUDA operations.

    Returns:
        Callable: A function (G, f, start_idx, actual_chunk_size, M) -> centered_float32_chunk.
    """
    if device.type == 'mps':
        def unpack_center_mps(G, f, start_idx, actual_chunk_size, M):
            G_chunk = G[start_idx:start_idx + actual_chunk_size, :].to(device, non_blocking=True)
            f_chunk = f[start_idx:start_idx + actual_chunk_size].unsqueeze(1)
            return G_chunk.float() - 2.0 * f_chunk
        return unpack_center_mps
    
    def unpack_center_cuda(G, f, start_idx, actual_chunk_size, M):
        if G.device.type == 'cpu':
            byte_start = start_idx // 4
            byte_end = (start_idx + actual_chunk_size + 3) // 4
            G_sub = G[byte_start:byte_end, :].to(device, non_blocking=True)
            return torch.ops.pack2bit.unpack2bit_gpu_chunk_center(G_sub, f, start_idx, actual_chunk_size, M, byte_start, threads_per_block)
        return torch.ops.pack2bit.unpack2bit_gpu_chunk_center(G, f, start_idx, actual_chunk_size, M, 0, threads_per_block)
    
    return unpack_center_cuda

def freq_batch_math(G_chunk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Description:
    Core math for calculating allele frequencies on a genotype chunk (uint8).

    Args:
        G_chunk (torch.Tensor): Unpacked uint8 genotype chunk.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: (sum of alleles, count of non-missing genotypes)
    """
    mask = (G_chunk != 3)
    f_batch = (G_chunk.float() * mask.float()).sum(dim=1)
    denom_batch = mask.sum(dim=1, dtype=torch.float32)
    return f_batch, denom_batch

freq_batch_compiled = torch.compile(freq_batch_math, disable=not hasattr(torch, "compile"))

def calculate_frequencies_gpu(G_torch: torch.Tensor, M: int, chunk_size: int, device_obj: torch.device, threads_per_block: int) -> torch.Tensor:
    """
    Description:
    Calculates allele frequencies iteratively using GPU-accelerated chunks.

    Args:
        G_torch (torch.Tensor): Genotype matrix.
        M (int): Number of individuals.
        chunk_size (int): Batch size to process genotypes.
        device_obj (torch.device): GPU computation device.
        threads_per_block (int): CUDA thread scaling factor.

    Returns:
        torch.Tensor: Computed 1D allele frequencies (float32).
    """
    f_torch = torch.zeros(M, dtype=torch.float32, device=device_obj)
    denom_torch = torch.zeros(M, dtype=torch.float32, device=device_obj)
    
    unpacker = get_unpacker(device_obj, threads_per_block)
    
    for m in range(0, M, chunk_size):
        actual_chunk_size = min(chunk_size, M - m)
        G_chunk = unpacker(G_torch, m, actual_chunk_size, M)
        
        f_b, d_b = freq_batch_compiled(G_chunk)
        f_torch[m:m+actual_chunk_size] = f_b
        denom_torch[m:m+actual_chunk_size] = d_b
    
    valid = denom_torch > 0
    f_torch[valid] = f_torch[valid] / (2.0 * denom_torch[valid])
    return f_torch