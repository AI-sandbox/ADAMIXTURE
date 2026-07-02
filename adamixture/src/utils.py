import logging
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from .snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

MIN_GPU_CHUNK_SIZE = 2
MIN_CPU_CHUNK_SIZE = 2


def is_gpu_oom(exc: BaseException) -> bool:
    """
    Return True for CUDA/MPS out-of-memory exceptions raised by PyTorch or device kernels.
    """
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or any(
        phrase in message
        for phrase in (
            "cuda out of memory",
            "mps backend out of memory",
            "mps out of memory",
            "metal out of memory",
            "out of memory on mps",
            "out of memory on gpu",
        )
    )


def is_cuda_oom(exc: BaseException) -> bool:
    """
    Backward-compatible alias for GPU out-of-memory detection.
    """
    return is_gpu_oom(exc)


def reduce_gpu_chunk_or_raise(chunk_size: int, exc: BaseException, context: str) -> int:
    """
    Halve a GPU chunk size after an OOM and clear CUDA/MPS allocator caches.
    """
    if chunk_size <= MIN_GPU_CHUNK_SIZE:
        raise exc

    new_chunk_size = max(chunk_size // 2, MIN_GPU_CHUNK_SIZE)
    exc.__traceback__ = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    log.warning(f"    GPU out of memory while {context}: retrying with chunk_size={new_chunk_size}.")
    return new_chunk_size


def is_cpu_memory_error(exc: BaseException) -> bool:
    """
    Return True for CPU allocation failures raised by NumPy, PyTorch, or Cython.
    """
    message = str(exc).lower()
    return isinstance(exc, MemoryError) or any(
        phrase in message
        for phrase in (
            "unable to allocate",
            "cannot allocate memory",
            "out of memory",
            "bad allocation",
            "std::bad_alloc",
            "defaultcpuallocator",
        )
    )


def reduce_cpu_chunk_or_raise(chunk_size: int, exc: BaseException, context: str) -> int:
    """
    Halve a CPU chunk size after an allocation failure.
    """
    if chunk_size <= MIN_CPU_CHUNK_SIZE:
        raise exc

    new_chunk_size = max(chunk_size // 2, MIN_CPU_CHUNK_SIZE)
    exc.__traceback__ = None
    log.warning(f"    CPU out of memory: retrying with chunk_size={new_chunk_size}.")
    return new_chunk_size


def set_cuda_arch_list_if_needed(device: torch.device) -> None:
    """
    Set TORCH_CUDA_ARCH_LIST from the active CUDA device to avoid compiling
    extensions for every visible GPU architecture.
    """
    import os

    if "TORCH_CUDA_ARCH_LIST" in os.environ or device.type != "cuda":
        return

    device_index = device.index if device.index is not None else torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_index)
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"


def read_data(tr_file: str, packed: bool = False, chunk_size: int = 4096,
              chromosome_mode: str = "autosomes", autosome_count: int = 22,
              verbose: bool = True) -> tuple[torch.Tensor | np.ndarray, int, int]:
    """
    Description:
    Reads SNP data from a file (BED, VCF, etc.) and returns the genotype matrix and dimensions.

    Args:
        tr_file (str): Path to the SNP data file.
        packed (bool): If True, return a 2-bit packed torch.Tensor. Defaults to False.
        chunk_size (int): Size of chunks to read for VCF files. Defaults to 4096.
        verbose (bool): If True, log the number of samples and SNPs. Defaults to True.
        chromosome_mode (str): "all" to keep all chromosomes or "autosomes" to keep 1..autosome_count.
        autosome_count (int): Number of autosomes when chromosome_mode is "autosomes".

    Returns:
        tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N samples, M SNPs)
    """
    current_chunk_size = chunk_size
    while True:
        try:
            if current_chunk_size != chunk_size:
                log.warning(f"    Retrying data read with chunk_size={current_chunk_size}.")
            snp_reader = SNPReader()
            G, N, M = snp_reader.read_data(
                tr_file,
                packed=packed,
                chunk_size=current_chunk_size,
                chromosome_mode=chromosome_mode,
                autosome_count=autosome_count,
            )
            break
        except (MemoryError, RuntimeError) as exc:
            if not is_cpu_memory_error(exc) or current_chunk_size <= MIN_CPU_CHUNK_SIZE:
                raise
            current_chunk_size = reduce_cpu_chunk_or_raise(
                current_chunk_size, exc, "reading genotype data"
            )
    if verbose:
        log.info(f"    Data contains {N} samples and {M} SNPs.")

    return G, N, M

def get_tuning_params(device: torch.device) -> int:
    """
    Description:
    Returns optimal CUDA kernel parameters (threads_per_block) based on the device properties.

    Args:
        device (torch.device): The target computation device.

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

def get_dtype(device: torch.device) -> torch.dtype:
    """
    Description:
    Returns the recommended floating point precision for the given device.
    MPS does not support float64, so float32 is returned. For other devices,
    float64 is preferred for precision.

    Args:
        device (torch.device): Target computation device.

    Returns:
        torch.dtype: Recommended dtype (float32 or float64).
    """
    if device.type == "mps":
        return torch.float32
    return torch.float64

def write_outputs(Q: np.ndarray, run_name: str, K: int, out_path: str | Path, P: np.ndarray = None) -> None:
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

def manage_gpu_memory(
    G: torch.Tensor | np.ndarray,
    device: torch.device,
    M: int,
    N: int,
    K: int,
    chunk_size: int,
    algorithm: str,
    q_hist: int = 3,
    include_initialization: bool = True,
) -> torch.Tensor:
    """
    Description:
    Determines if the genotype matrix fits in GPU memory and moves it if possible.
    Otherwise, leaves it on the CPU for streaming. The estimate accounts for the
    selected optimizer because Adam-EM and SQP + ZAL-QN have different peak buffers.

    Args:
        G (torch.Tensor | np.ndarray): Packed or unpacked genotype matrix.
        device (torch.device): Target computation device.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        chunk_size (int): Expected batch size for computations.
        algorithm (str): Optimizer name ("adamem" or "brqn").
        q_hist (int): ZAL-QN history depth used by "brqn". Defaults to 3.
        include_initialization (bool): Whether to include SVD/ALS initialization buffers.
            Defaults to True.

    Returns:
        torch.Tensor | np.ndarray: Genotype tensor on the selected device.
    """
    if device.type == 'mps':
        # Always keep on CPU if MPS, but convert to torch tensor if it's numpy
        if isinstance(G, np.ndarray):
            return torch.from_numpy(G)
        return G

    if device.type != 'cuda':
        return G

    if isinstance(G, np.ndarray):
        G = torch.from_numpy(G)

    memory_GPU = get_free_gpu_memory(device)

    bytes_per_float32 = 4
    bytes_per_float64 = 8

    L = max(K + 10, 20)

    # SVD Matrices (approx peak)
    # proj_basis (float32), accum_mat (float32), orth_matrix (float32)
    svd_total = (M * L * bytes_per_float32) + (2 * N * L * bytes_per_float32)

    # ALS peak memory (mostly float64)
    # P, Z, P_free, B_target_P, x_batch => ~5 matrices of (M x K) in float64
    # Q, Q0, V, I_q, B_target_Q => ~5 matrices of (N x K) in float64
    als_total = (5 * M * K * bytes_per_float64) + (5 * N * K * bytes_per_float64)

    dim = (M + N) * K
    chunk_rows = min(chunk_size, M)
    chunk_uint8 = chunk_rows * N

    if algorithm == 'adamem':
        # Adam-EM peak memory (float64 on CUDA):
        # current P/Q, optimizer moments, EM accumulation buffers, best P/Q,
        # plus the large per-chunk EM/log-likelihood temporaries.
        optimizer_total = (
            (8 * M * K * bytes_per_float64)
            + (7 * N * K * bytes_per_float64)
            + (1 * N * bytes_per_float64)
            + (6 * chunk_rows * N * bytes_per_float64)
            + chunk_uint8
        )
    elif algorithm == 'brqn':
        # SQP + ZAL-QN peak memory (float64 on CUDA):
        # Hessian/gradient buffers, QN history, current/next/best parameter
        # vectors, and chunk temporaries from the P/Q SQP updates.
        hessian_total = ((M + N) * K * K + (M + N) * K) * bytes_per_float64
        qn_history_total = 3 * dim * q_hist * bytes_per_float64
        parameter_total = 9 * dim * bytes_per_float64
        q_update_chunk_total = (
            (7 * chunk_rows * N + 2 * N * K * K + 2 * N * K + chunk_rows * K)
            * bytes_per_float64
        ) + chunk_uint8
        p_update_chunk_total = (
            (5 * chunk_rows * N + chunk_rows * K * K + chunk_rows * K)
            * bytes_per_float64
        ) + chunk_uint8
        optimizer_total = (
            hessian_total
            + qn_history_total
            + parameter_total
            + max(q_update_chunk_total, p_update_chunk_total)
        )
    else:
        raise ValueError(f"Unknown algorithm for GPU memory estimate: {algorithm}")

    if include_initialization:
        peak_memory = max(svd_total, als_total, optimizer_total)
    else:
        peak_memory = optimizer_total
    peak_memory_MB = peak_memory / (1024 ** 2)

    memory_data_MB = (G.numel() * G.element_size()) / (1024 ** 2)

    # Base chunk unpacking memory is already accounted for in svd/adam formulas but adding base unpack buffer:
    memory_chunk_MB = chunk_size * N * bytes_per_float32 / (1024 ** 2)

    # Total required with some buffer
    total_required_MB = memory_data_MB + peak_memory_MB + memory_chunk_MB

    if memory_GPU * 0.95 - total_required_MB > 0:
        log.info("    Moving genotype matrix to GPU...")
        G = G.to(device)
    else:
        log.info("    Genotype matrix too large for GPU, keeping on CPU...")

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
        set_cuda_arch_list_if_needed(device)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_path = os.path.abspath(os.path.join(current_dir, "utils_c", "cuda", "pack2bit.cu"))

        if not os.path.exists(source_path):
            log.error(f"CUDA source files not found in {os.path.join(current_dir, 'utils_c', 'cuda')}")
            return

        log.info("    Loading CUDA extensions...")
        cuda_flags = ['-O3', '--use_fast_math']
        cpp_flags = ['-O3']

        load(name="pack2bit", sources=[source_path],
             verbose=False, extra_cuda_cflags=cuda_flags, extra_cflags=cpp_flags)

        load(name="bvls_kernel",
             sources=[os.path.join(current_dir, "utils_c", "cuda", "bvls_kernel.cu")],
             verbose=False, extra_cuda_cflags=cuda_flags, extra_cflags=cpp_flags)

        load(name="sqp_kernel",
             sources=[os.path.join(current_dir, "utils_c", "cuda", "sqp_kernel.cu")],
             verbose=False, extra_cuda_cflags=cuda_flags, extra_cflags=cpp_flags)


def is_packed_genotype_tensor(G: torch.Tensor, M: int) -> bool:
    """
    Return True when ``G`` is a 2-bit packed genotype matrix with ceil(M / 4) rows.
    """
    return isinstance(G, torch.Tensor) and G.ndim == 2 and G.size(0) != M


def get_unpacker(device: torch.device, threads_per_block: int) -> Callable[[torch.Tensor, int, int, int], torch.Tensor]:
    """
    Description:
    Returns a specialized function for unpacking genotype chunks based on the device.
    CUDA inputs are expected to be packed 2-bit matrices.

    Args:
        device (torch.device): Target computation device.
        threads_per_block (int): Threads per block for CUDA operations.

    Returns:
        Callable: A function with signature (G, start_idx, actual_chunk_size, M) -> torch.Tensor.
    """
    if device.type == 'mps':
        from .utils_c import metal

        def unpack_mps(G: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
            """
            Description:
            Slices an unpacked genotype chunk or unpacks a 2-bit packed chunk
            with a Metal kernel, returning a uint8 tensor on MPS.

            Args:
                G (torch.Tensor): Unpacked genotype tensor.
                start_idx (int): First SNP row in the chunk.
                actual_chunk_size (int): Number of SNP rows to slice.
                M (int): Total number of SNPs.

            Returns:
                torch.Tensor: Genotype chunk on the MPS device.
            """
            if is_packed_genotype_tensor(G, M):
                byte_start = start_idx // 4
                byte_end = (start_idx + actual_chunk_size + 3) // 4
                G_sub = G[byte_start:byte_end, :].to(device, non_blocking=True)
                return metal.unpack2bit_gpu_chunk_uint8_mps(
                    G_sub, start_idx, actual_chunk_size, M, byte_start
                )
            return G[start_idx:start_idx + actual_chunk_size, :].to(device, non_blocking=True)
        return unpack_mps

    def unpack_cuda(G: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
        """
        Description:
        Unpacks a packed 2-bit genotype chunk for CUDA computation.

        Args:
            G (torch.Tensor): Packed genotype tensor on CPU or CUDA.
            start_idx (int): First SNP row in the unpacked coordinate system.
            actual_chunk_size (int): Number of SNP rows to unpack.
            M (int): Total number of SNPs.

        Returns:
            torch.Tensor: Unpacked uint8 genotype chunk on the CUDA device.
        """
        if G.device.type == 'cpu':
            byte_start = start_idx // 4
            byte_end = (start_idx + actual_chunk_size + 3) // 4
            G_sub = G[byte_start:byte_end, :].to(device, non_blocking=True)
            return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G_sub, start_idx, actual_chunk_size, M, byte_start, threads_per_block)
        return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G, start_idx, actual_chunk_size, M, 0, threads_per_block)

    return unpack_cuda


def loglikelihood_cpu_chunked(
    G: torch.Tensor | np.ndarray,
    P: torch.Tensor,
    Q: torch.Tensor,
    M: int,
    batch_size: int,
) -> float:
    """
    Compute log-likelihood on CPU in float64 for unpacked or 2-bit packed genotypes.
    """
    from .utils_c import tools

    del batch_size

    if isinstance(G, torch.Tensor):
        G_cpu_tensor = G.detach().cpu()
        G_cpu = G_cpu_tensor.numpy()
        packed = is_packed_genotype_tensor(G, M)
        G_ptr = G_cpu_tensor.data_ptr() if packed else None
    else:
        G_cpu = G
        packed = G_cpu.shape[0] != M
        G_ptr = G_cpu.ctypes.data if packed else None

    P_cpu = np.ascontiguousarray(P.detach().cpu().numpy(), dtype=np.float64)
    Q_cpu = np.ascontiguousarray(Q.detach().cpu().numpy(), dtype=np.float64)

    if packed:
        return tools.loglikelihood_packed(G_ptr, M, Q_cpu.shape[0], P_cpu, Q_cpu)
    return tools.loglikelihood(np.ascontiguousarray(G_cpu), P_cpu, Q_cpu)


def get_logl_calculator(device: torch.device) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int], float]:
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
        def logl_mps(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, N: int, batch_size: int, threads_per_block: int) -> float:
            """
            Description:
            Computes log-likelihood on CPU for MPS tensors.

            Args:
                G (torch.Tensor): Genotype tensor.
                P (torch.Tensor): Allele-frequency tensor.
                Q (torch.Tensor): Ancestry-proportion tensor.
                M (int): Number of SNPs.
                N (int): Number of individuals.
                batch_size (int): Batch size argument kept for signature compatibility.
                threads_per_block (int): CUDA tuning argument kept for signature compatibility.

            Returns:
                float: Log-likelihood value.
            """
            del N, threads_per_block
            return loglikelihood_cpu_chunked(G, P, Q, M, batch_size)
        return logl_mps

    from ..model.em_adam_gpu import loglikelihood_gpu
    def logl_gpu_wrapped(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, N: int, batch_size: int, threads_per_block: int) -> float:
        """
        Description:
        Computes log-likelihood on CUDA using the packed-genotype GPU implementation.

        Args:
            G (torch.Tensor): Packed genotype tensor.
            P (torch.Tensor): Allele-frequency tensor.
            Q (torch.Tensor): Ancestry-proportion tensor.
            M (int): Number of SNPs.
            N (int): Number of individuals.
            batch_size (int): SNP batch size.
            threads_per_block (int): CUDA tuning parameter.

        Returns:
            float: Log-likelihood value.
        """
        return loglikelihood_gpu(G, P, Q, M, N, batch_size, device, threads_per_block)
    return logl_gpu_wrapped


def _calculate_frequencies_cpu_once(G: np.ndarray, M: int, N: int, chunk_size: int) -> np.ndarray:
    """
    Calculate allele frequencies on CPU in SNP chunks.
    """
    from .utils_c import tools

    f = np.zeros(M, dtype=np.float32)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        tools.alleleFrequency(G[start:end], f[start:end], end - start, N)
    return f


def calculate_frequencies_cpu(G: np.ndarray, M: int, N: int, chunk_size: int) -> np.ndarray:
    """
    Calculate allele frequencies on CPU, retrying with smaller chunks on memory errors.
    """
    current_chunk_size = chunk_size

    while True:
        try:
            if current_chunk_size != chunk_size:
                log.warning(f"    Retrying frequency calculation with chunk_size={current_chunk_size}.")
            return _calculate_frequencies_cpu_once(G, M, N, current_chunk_size)
        except (MemoryError, RuntimeError) as exc:
            if not is_cpu_memory_error(exc) or current_chunk_size <= MIN_CPU_CHUNK_SIZE:
                raise
            current_chunk_size = reduce_cpu_chunk_or_raise(
                current_chunk_size, exc, "calculating allele frequencies"
            )


def get_centering_unpacker(device: torch.device, threads_per_block: int) -> Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]:
    """
    Description:
    Returns a specialized function for unpacking and centering genotype chunks.
    CUDA inputs are expected to be packed 2-bit matrices.

    Args:
        device (torch.device): Target computation device.
        threads_per_block (int): Threads per block for CUDA operations.

    Returns:
        Callable: A function (G, f, start_idx, actual_chunk_size, M) -> centered_float32_chunk.
    """
    if device.type == 'mps':
        from .utils_c import metal

        def unpack_center_mps(G: torch.Tensor, f: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
            """
            Description:
            Slices and centers an unpacked genotype chunk, or unpacks and
            centers a 2-bit packed chunk using Metal.

            Args:
                G (torch.Tensor): Unpacked genotype tensor.
                f (torch.Tensor): Allele-frequency vector.
                start_idx (int): First SNP row in the chunk.
                actual_chunk_size (int): Number of SNP rows to slice.
                M (int): Total number of SNPs.

            Returns:
                torch.Tensor: Centered float32 genotype chunk on MPS.
            """
            if is_packed_genotype_tensor(G, M):
                byte_start = start_idx // 4
                byte_end = (start_idx + actual_chunk_size + 3) // 4
                G_sub = G[byte_start:byte_end, :].to(device, non_blocking=True)
                return metal.unpack2bit_gpu_chunk_center_mps(
                    G_sub, f, start_idx, actual_chunk_size, M, byte_start
                )
            G_chunk = G[start_idx:start_idx + actual_chunk_size, :].to(device, non_blocking=True)
            f_chunk = f[start_idx:start_idx + actual_chunk_size].unsqueeze(1)
            return G_chunk.float() - 2.0 * f_chunk
        return unpack_center_mps

    def unpack_center_cuda(G: torch.Tensor, f: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
        """
        Description:
        Unpacks and centers a packed 2-bit genotype chunk for CUDA SVD.

        Args:
            G (torch.Tensor): Packed genotype tensor on CPU or CUDA.
            f (torch.Tensor): Allele-frequency vector on CUDA.
            start_idx (int): First SNP row in the unpacked coordinate system.
            actual_chunk_size (int): Number of SNP rows to unpack.
            M (int): Total number of SNPs.

        Returns:
            torch.Tensor: Centered float32 genotype chunk on CUDA.
        """
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

def _calculate_frequencies_gpu_once(G_torch: torch.Tensor, M: int, chunk_size: int, device_obj: torch.device, threads_per_block: int) -> torch.Tensor:
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


def calculate_frequencies_gpu(G_torch: torch.Tensor, M: int, chunk_size: int, device_obj: torch.device, threads_per_block: int) -> torch.Tensor:
    """
    Calculate allele frequencies on GPU, retrying with smaller chunks on CUDA OOM.
    """
    current_chunk_size = chunk_size

    while True:
        try:
            if current_chunk_size != chunk_size:
                log.warning(f"    Retrying frequency calculation with chunk_size={current_chunk_size}.")
            return _calculate_frequencies_gpu_once(
                G_torch, M, current_chunk_size, device_obj, threads_per_block
            )
        except RuntimeError as exc:
            if not is_gpu_oom(exc) or current_chunk_size <= MIN_GPU_CHUNK_SIZE:
                raise
            current_chunk_size = reduce_gpu_chunk_or_raise(
                current_chunk_size, exc, "calculating allele frequencies"
            )
