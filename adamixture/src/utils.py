import logging
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import argparse

from .snp_reader import SNPReader

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

def read_data(tr_file: str, packed: bool = False, chunk_size: int = 4096, verbose: bool = True) -> tuple[torch.Tensor | np.ndarray, int, int]:
    """
    Description:
    Reads SNP data from a file (BED, VCF, etc.) and returns the genotype matrix and dimensions.

    Args:
        tr_file (str): Path to the SNP data file.
        packed (bool): If True, return a 2-bit packed torch.Tensor. Defaults to False.
        chunk_size (int): Size of chunks to read for VCF files. Defaults to 4096.
        verbose (bool): If True, log the number of samples and SNPs. Defaults to True.

    Returns:
        tuple[torch.Tensor | np.ndarray, int, int]: (genotype matrix, N samples, M SNPs)
    """
    snp_reader = SNPReader()
    G, N, M = snp_reader.read_data(tr_file, packed=packed, chunk_size=chunk_size)
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

def manage_gpu_memory(G: torch.Tensor | np.ndarray, device: torch.device, M: int, N: int, K: int, chunk_size: int) -> torch.Tensor:
    """
    Description:
    Determines if the genotype matrix fits in GPU memory and moves it if possible.
    Otherwise, leaves it on the CPU for streaming.

    Args:
        G (torch.Tensor | np.ndarray): Packed or unpacked genotype matrix.
        device (torch.device): Target computation device.
        M (int): Number of SNPs.
        N (int): Number of individuals.
        K (int): Number of ancestral populations.
        chunk_size (int): Expected batch size for computations.

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

    memory_GPU = get_free_gpu_memory(device)

    bytes_per_float32 = 4
    bytes_per_float64 = 8

    L = max(K + 10, 20)

    # SVD Matrices (approx peak)
    # proj_basis (float32), accum_mat (float32), orth_matrix (float32)
    svd_total = (M * L * bytes_per_float32) + (2 * N * L * bytes_per_float32)

    # ALS Peak memory (mostly float64)
    # P, Z, P_free, B_target_P, x_batch => ~5 matrices of (M x K) in float64
    # Q, Q0, V, I_q, B_target_Q => ~5 matrices of (N x K) in float64
    als_total = (5 * M * K * bytes_per_float64) + (5 * N * K * bytes_per_float64)

    # EM-Adam Peak memory (float64 on CUDA)
    # P, m_P, v_P, A_accum, B_accum, P_EM => 6 matrices of (M x K) in float64
    # Q, m_Q, v_Q, T_accum, Q_EM => 5 matrices of (N x K) in float64
    # Batch memory during em_batch_math => ~6 tensors of (chunk_size x N) in float64
    adam_total = (6 * M * K * bytes_per_float64) + \
                 (5 * N * K * bytes_per_float64) + \
                 (6 * chunk_size * N * bytes_per_float64)

    peak_memory_MB = max(svd_total, als_total, adam_total) / (1024 ** 2)

    # 2-bit Data tensor size:
    memory_data_MB = (torch.numel(G)) / (1024 ** 2)

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

        load(name="cv_mask_kernel",
             sources=[os.path.join(current_dir, "utils_c", "cuda", "cv_mask_kernel.cu")],
             verbose=False, extra_cuda_cflags=cuda_flags, extra_cflags=cpp_flags)

        load(name="sqp_kernel",
             sources=[os.path.join(current_dir, "utils_c", "cuda", "sqp_kernel.cu")],
             verbose=False, extra_cuda_cflags=cuda_flags, extra_cflags=cpp_flags)

def get_unpacker(device: torch.device, threads_per_block: int) -> Callable[[torch.Tensor, int, int, int], torch.Tensor]:
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
        def unpack_mps(G: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
            return G[start_idx:start_idx + actual_chunk_size, :].to(device, non_blocking=True)
        return unpack_mps

    def unpack_cuda(G: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
        if G.device.type == 'cpu':
            byte_start = start_idx // 4
            byte_end = (start_idx + actual_chunk_size + 3) // 4
            G_sub = G[byte_start:byte_end, :].to(device, non_blocking=True)
            return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G_sub, start_idx, actual_chunk_size, M, byte_start, threads_per_block)
        return torch.ops.pack2bit.unpack2bit_gpu_chunk_uint8(G, start_idx, actual_chunk_size, M, 0, threads_per_block)

    return unpack_cuda

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
        from .utils_c import tools
        def logl_mps(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, N: int, batch_size: int, threads_per_block: int) -> float:
            G_cpu = G.numpy() if isinstance(G, torch.Tensor) else G
            return tools.loglikelihood(G_cpu, P.cpu().numpy().astype(np.float64), Q.cpu().numpy().astype(np.float64))
        return logl_mps

    from ..model.em_adam_gpu import loglikelihood_gpu
    def logl_gpu_wrapped(G: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, N: int, batch_size: int, threads_per_block: int) -> float:
        return loglikelihood_gpu(G, P, Q, M, N, batch_size, device, threads_per_block)
    return logl_gpu_wrapped

def get_centering_unpacker(device: torch.device, threads_per_block: int) -> Callable[[torch.Tensor, torch.Tensor, int, int, int], torch.Tensor]:
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
        def unpack_center_mps(G: torch.Tensor, f: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
            G_chunk = G[start_idx:start_idx + actual_chunk_size, :].to(device, non_blocking=True)
            f_chunk = f[start_idx:start_idx + actual_chunk_size].unsqueeze(1)
            return G_chunk.float() - 2.0 * f_chunk
        return unpack_center_mps

    def unpack_center_cuda(G: torch.Tensor, f: torch.Tensor, start_idx: int, actual_chunk_size: int, M: int) -> torch.Tensor:
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

def plot_single_k(args: argparse.Namespace, K: int, Q_np: np.ndarray) -> None:
    """
    Description:
    Generates a Q-matrix plot for a single value of K.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        K (int): The number of populations.
        Q_np (np.ndarray): The inferred ancestry proportions matrix.

    Returns:
        None
    """
    from .plot import plot_q_matrix

    labels = None
    if args.labels and Path(args.labels).exists():
        with open(args.labels) as fh:
            labels = [line.strip() for line in fh if line.strip()]

    labels2 = None
    if hasattr(args, 'labels2') and args.labels2 and Path(args.labels2).exists():
        with open(args.labels2) as fh:
            labels2 = [line.strip() for line in fh if line.strip()]

    labels3 = None
    if hasattr(args, 'labels3') and args.labels3 and Path(args.labels3).exists():
        with open(args.labels3) as fh:
            labels3 = [line.strip() for line in fh if line.strip()]

    colors = None
    if args.colors and Path(args.colors).exists():
        with open(args.colors) as fh:
            colors = [line.strip() for line in fh if line.strip()]
        if len(colors) != K:
            log.warning(f"    Number of colors in {args.colors} ({len(colors)}) does not match K={K}. Using default colors.")
            colors = None

    plot_path = Path(args.save_dir) / f"{args.name}.{K}.{args.plot_format}"
    log.info(f"    Generating plot: {plot_path}")
    plot_q_matrix(Q_np, plot_path, dpi=args.plot_dpi, format=args.plot_format,
                  labels=labels, labels2=labels2, labels3=labels3, custom_colors=colors)

def plot_combined(args: argparse.Namespace, k_values: list[int], trained_plot: dict[int, tuple]) -> None:
    """
    Description:
    Generates a combined single plot for all K sweep values.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        k_values (list[int]): List of K values to include in the plot.
        trained_plot (dict[int, tuple]): Dictionary mapping each K to its (P_np, Q_np) matrices.

    Returns:
        None
    """
    log.info("\n    Generating combined single plot for all K sweep values...")
    import matplotlib.pyplot as plt

    from .plot import _MAX_LABEL_LEN, _draw_brackets

    # Load labels if available
    labels = None
    if args.labels and Path(args.labels).exists():
        with open(args.labels) as fh:
            labels = [line.strip() for line in fh if line.strip()]

    labels2 = None
    if hasattr(args, 'labels2') and args.labels2 and Path(args.labels2).exists():
        with open(args.labels2) as fh:
            labels2 = [line.strip() for line in fh if line.strip()]

    labels3 = None
    if hasattr(args, 'labels3') and args.labels3 and Path(args.labels3).exists():
        with open(args.labels3) as fh:
            labels3 = [line.strip() for line in fh if line.strip()]

    colors_list = None
    if args.colors and Path(args.colors).exists():
        with open(args.colors) as fh:
            colors_list = [line.strip() for line in fh if line.strip()]

    # Validate hierarchical consistency: each lower-level label must belong to
    # exactly one higher-level group (e.g. "Barcelona" → only "Spain", not also "France").
    def _check_hierarchy(child_lbls, parent_lbls, child_name, parent_name):
        mapping: dict = {}
        conflicts: list[str] = []
        for child, parent in zip(child_lbls, parent_lbls, strict=False):
            if child in mapping:
                if mapping[child] != parent:
                    conflicts.append(child)
            else:
                mapping[child] = parent
        if conflicts:
            log.warning(
                f"    Warning: Some {child_name} labels appear in more than one "
                f"{parent_name} group. Ignoring {parent_name}."
            )
            return False
        return True

    if labels is not None and labels2 is not None:
        if not _check_hierarchy(labels, labels2, '--labels', '--labels2'):
            labels2 = None
    if labels2 is not None and labels3 is not None:
        if not _check_hierarchy(labels2, labels3, '--labels2', '--labels3'):
            labels3 = None

    num_runs = len(k_values)

    first_K = k_values[0]
    first_Q = trained_plot[first_K][1]
    n_samples_global = first_Q.shape[0]

    if labels is not None and len(labels) == n_samples_global:
        if labels3 is not None and labels2 is not None:
            sort_idx = np.lexsort((labels, labels2, labels3))
        elif labels2 is not None:
            sort_idx = np.lexsort((labels, labels2))
        else:
            dominant_cluster = np.argmax(first_Q, axis=1)
            sort_idx = np.lexsort((np.max(first_Q, axis=1), dominant_cluster, labels))
    else:
        dominant_cluster = np.argmax(first_Q, axis=1)
        sort_idx = np.lexsort((np.max(first_Q, axis=1), dominant_cluster))

    labels_sorted = [labels[i] for i in sort_idx] if labels is not None and len(labels) == n_samples_global else None
    labels2_sorted = [labels2[i] for i in sort_idx] if labels2 is not None else None
    labels3_sorted = [labels3[i] for i in sort_idx] if labels3 is not None else None

    pop_boundaries: list[int] = []
    pop_tick_positions: list[float] = []
    pop_tick_labels: list[str] = []
    if labels_sorted is not None:
        current_label = labels_sorted[0]
        start_idx = 0
        for idx, lbl in enumerate(labels_sorted):
            if lbl != current_label:
                pop_boundaries.append(idx)
                pop_tick_positions.append((start_idx + idx) / 2)
                tick_text = str(current_label).title()
                if len(tick_text) > _MAX_LABEL_LEN:
                    tick_text = tick_text[:_MAX_LABEL_LEN - 1] + '…'
                pop_tick_labels.append(tick_text)
                start_idx = idx
                current_label = lbl
        tick_text = str(current_label).title()
        if len(tick_text) > _MAX_LABEL_LEN:
            tick_text = tick_text[:_MAX_LABEL_LEN - 1] + '…'
        pop_tick_positions.append((start_idx + n_samples_global) / 2)
        pop_tick_labels.append(tick_text)

    def _build_brackets_list(sorted_lbls):
        if sorted_lbls is None:
            return []
        items = []
        current_name = sorted_lbls[0]
        seg_start = 0
        for idx, name in enumerate(sorted_lbls):
            if name != current_name:
                items.append({'name': current_name, 'start': seg_start, 'end': idx})
                seg_start = idx
                current_name = name
        items.append({'name': current_name, 'start': seg_start, 'end': n_samples_global})
        return items

    i2_items = _build_brackets_list(labels2_sorted)
    i3_items = _build_brackets_list(labels3_sorted)

    max_l1_len = min(max((len(str(lbl)) for lbl in pop_tick_labels), default=0), _MAX_LABEL_LEN)
    max_l2_len = min(max((len(item['name']) for item in i2_items), default=0), _MAX_LABEL_LEN)
    max_l3_len = min(max((len(item['name']) for item in i3_items), default=0), _MAX_LABEL_LEN)

    plot_height_in = 2.5 * num_runs
    l1_height_in = 0.5 + max_l1_len * 0.08 if labels_sorted else 0.0
    l2_height_in = 0.8 + max_l2_len * 0.08 if i2_items else 0.0
    l3_height_in = 0.8 + max_l3_len * 0.08 if i3_items else 0.0

    total_labels_height_in = l1_height_in + l2_height_in + l3_height_in
    if total_labels_height_in == 0:
        total_labels_height_in = 0.6

    fig_height = plot_height_in + total_labels_height_in
    bottom_margin = total_labels_height_in / fig_height

    fig, axes = plt.subplots(nrows=num_runs, ncols=1, figsize=(15, fig_height), squeeze=False)
    axes = axes.flatten()

    for i, K_val in enumerate(k_values):
        ax = axes[i]
        P_np, Q_np = trained_plot[K_val]
        Q_sorted = Q_np[sort_idx]
        n_samples, K_curr = Q_sorted.shape

        if colors_list is not None and len(colors_list) >= K_curr:
            colors = colors_list[:K_curr]
        else:
            cmap = plt.colormaps.get_cmap('tab20')
            colors = cmap(np.arange(K_curr) % 20)

        Q_cum = np.cumsum(Q_sorted, axis=1)
        x = np.arange(n_samples)
        zeros = np.zeros(n_samples)

        for j in range(K_curr):
            lower = Q_cum[:, j - 1] if j > 0 else zeros
            upper = Q_cum[:, j]
            ax.fill_between(x, lower, upper, facecolor=colors[j], edgecolor='none', linewidth=0, rasterized=True)

        for boundary in pop_boundaries:
            ax.axvline(x=boundary, color='black', linestyle='--', linewidth=0.5)

        ax.set_xlim(0, n_samples)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"K={K_curr}", rotation=0, ha='right', va='center', labelpad=10, fontweight='bold')
        ax.set_yticks([0.0, 0.5, 1.0])

        is_bottom = (i == num_runs - 1)
        if is_bottom and labels_sorted is not None:
            ax.set_xticks(pop_tick_positions)
            ax.set_xticklabels(pop_tick_labels, rotation=90, ha='center', fontsize=6)
            ax.tick_params(axis='x', which='both', length=0, pad=5)

            _CHAR_INCH = 0.08
            _GAP_INCH = 0.35
            _TICK_PAD_INCH = 0.15
            _SUBPLOT_HEIGHT = 2.5

            y_l1_bottom_in = -(_TICK_PAD_INCH + max_l1_len * _CHAR_INCH)
            y_i2_in = y_l1_bottom_in - _GAP_INCH
            y_i2 = y_i2_in / _SUBPLOT_HEIGHT

            y_l2_bottom_in = y_i2_in - 0.15 - max_l2_len * _CHAR_INCH
            y_i3_in = y_l2_bottom_in - _GAP_INCH
            y_i3 = y_i3_in / _SUBPLOT_HEIGHT

            if i2_items:
                _draw_brackets(ax, i2_items, y_bracket=y_i2, fontsize=6)
            if i3_items:
                _draw_brackets(ax, i3_items, y_bracket=y_i3, fontsize=6)
        else:
            ax.set_xticks([])

        if is_bottom and labels_sorted is None:
            ax.set_xlabel("Samples")

    plt.subplots_adjust(bottom=bottom_margin, hspace=0.25)

    single_plot_path = Path(args.save_dir) / f"{args.name}.{k_values[0]}_{k_values[-1]}.{args.plot_format}"
    fig.savefig(single_plot_path, dpi=args.plot_dpi, format=args.plot_format, bbox_inches='tight')
    log.info(f"    Combined single plot saved to: {single_plot_path}")
    plt.close(fig)
