import argparse
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from .compat import is_torch_loaded, require_torch
from .entry import _fix_macos_libomp
from .model.br_qn import q_hessian_block_size
from .src import utils
from .src.adamixture import setup, train_k
from .src.cv import run_cross_validation


def run(
    data: Any = None,
    bfile: str | Path | np.ndarray | None = None,
    k: int | None = None,
    K: int | None = None,
    threads: int = 1,
    seed: int = 42,
    tol: float = 0.1,
    cv: int | None = None,
    no_freqs: bool = False,
    plink: bool = True,
    n_indiv: int | None = None,
    n_loci: int | None = None,
    chrom_mode: str = "all",
    verbose: bool = True,
    device: str = "cpu",
    **extra_kwargs: Any,
) -> Any:
    """
    Programmatic entry point for ADAMIXTURE clustering on CPU (using BR-QN + ALS).
    Designed for seamless integration with R wrappers (reticulate / tidygenclust)
    and Python scripts.

    Args:
        data: Genotype data (in-memory numpy array, or file path/prefix).
        bfile: Alias for data / PLINK file prefix.
        k: Number of ancestral clusters (single K).
        K: Alias for k.
        threads: Number of CPU threads to use.
        seed: Random seed.
        tol: Convergence tolerance for BR-QN.
        cv: Number of cross-validation folds (None or 0 disables CV).
        no_freqs: If True, do not return P (allele frequencies) matrix.
        plink: If True, reads PLINK format. If False, reads raw binary genotypes dumped by tidypopgen.
        n_indiv: Number of individuals (required when plink=False).
        n_loci: Number of SNPs/loci (required when plink=False).
        chrom_mode: Chromosome filter mode ('all' or 'autosomes'). Default is 'all'.
        verbose: If True, log progress.
        **extra_kwargs: Optional advanced overrides (max_iter, max_als, tol_als, power, tol_svd, chunk_size, etc.).

    Returns:
        - If no_freqs=True and no CV: Q (np.ndarray of shape [N, K])
        - If no_freqs=False and no CV: (Q, P) (tuple of np.ndarrays)
        - If no_freqs=True and CV: (Q, {"avg": cv_score})
        - If no_freqs=False and CV: (Q, P, {"avg": cv_score})
    """
    _fix_macos_libomp()

    # Resolve aliases
    target_data = data if data is not None else (bfile if bfile is not None else extra_kwargs.get("data_path"))
    if target_data is None:
        raise ValueError("Must provide genotype data via 'data' or 'bfile'.")

    target_k = K if K is not None else k
    if target_k is None:
        min_k = extra_kwargs.get("min_k")
        max_k = extra_kwargs.get("max_k")
        if min_k is not None and max_k is not None:
            k_values = list(range(int(min_k), int(max_k) + 1))
        else:
            raise ValueError("Must specify number of clusters via 'k' (or 'K').")
    else:
        k_values = [int(target_k)]

    # Internal parameters with fixed defaults (can be overridden via extra_kwargs if needed)
    max_iter = int(extra_kwargs.get("max_iter", extra_kwargs.get("iter", 10000)))
    tol = float(extra_kwargs.get("tol", extra_kwargs.get("tole", tol)))
    max_als = int(extra_kwargs.get("max_als", extra_kwargs.get("als_iter", 1000)))
    tol_als = float(extra_kwargs.get("tol_als", extra_kwargs.get("als_tole", 1e-4)))
    beta_als = float(extra_kwargs.get("beta_als", 0.8))
    power = int(extra_kwargs.get("power", 5))
    tol_svd = float(extra_kwargs.get("tol_svd", 1e-1))
    chunk_size = int(extra_kwargs.get("chunk_size", extra_kwargs.get("chunk", 8192)))

    save_dir = extra_kwargs.get("save_dir")
    name = extra_kwargs.get("name", "adamixture")

    # Configure threads for numerical libraries & PyTorch
    th = str(int(threads))
    os.environ["MKL_NUM_THREADS"] = th
    os.environ["MKL_MAX_THREADS"] = th
    os.environ["OMP_NUM_THREADS"] = th
    os.environ["OMP_MAX_THREADS"] = th
    os.environ["NUMEXPR_NUM_THREADS"] = th
    os.environ["NUMEXPR_MAX_THREADS"] = th
    os.environ["OPENBLAS_NUM_THREADS"] = th
    os.environ["OPENBLAS_MAX_THREADS"] = th
    if platform.system() == "Darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if is_torch_loaded():
        require_torch().set_num_threads(int(threads))

    # Set random seed
    utils.set_seed(int(seed))

    K_max = max(k_values)

    # Read data (always CPU unpacked uint8)
    G, N, M = utils.read_data(
        target_data,
        packed=False,
        chunk_size=chunk_size,
        chrom_mode=chrom_mode,
        verbose=verbose,
        plink=plink,
        n_indiv=n_indiv,
        n_loci=n_loci,
    )

    # Setup: frequencies and SVD (default: brqn + als)
    device_obj, threads_per_block, f, U, S, V, G = setup(
        G,
        N,
        M,
        K_max,
        int(seed),
        power,
        tol_svd,
        chunk_size,
        device,
        original=True,
        init_original="als",
        q_hist=3,
        compute_svd=True,
    )

    results: dict[int, tuple] = {}
    cv_enabled = cv is not None and int(cv) > 1

    for this_K in k_values:
        q_block = q_hessian_block_size(this_K)

        P, Q = train_k(
            G,
            N,
            M,
            this_K,
            U,
            S,
            V,
            f,
            int(seed),
            0.005,      # lr (Adam-EM fallback, unused in brqn)
            0.80,       # beta1
            0.88,       # beta2
            1e-8,       # reg_adam
            max_iter,
            5,          # check
            max_als,
            tol_als,
            0.5,        # lr_decay
            1e-4,       # min_lr
            chunk_size,
            3,          # patience
            tol,
            device_obj,
            threads_per_block,
            original=True,
            rtol=tol,
            Q_hist=3,
            init_original="als",
            em_init_steps=5,
            q_block=q_block,
            beta_als=beta_als,
        )

        P_np = P.cpu().numpy() if hasattr(P, "cpu") else np.asarray(P)
        Q_np = Q.cpu().numpy() if hasattr(Q, "cpu") else np.asarray(Q)
        P_np = np.ascontiguousarray(P_np, dtype=np.float64)
        Q_np = np.ascontiguousarray(Q_np, dtype=np.float64)

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            utils.write_outputs(Q_np, name, this_K, save_dir, P=None if no_freqs else P_np)

        # Cross-validation
        cv_score = None
        if cv_enabled:
            cv_args = argparse.Namespace(cv=int(cv), seed=int(seed))
            cv_index = run_cross_validation(
                cv_args, G, N, M, this_K, P_np, Q_np, q_block=q_block
            )
            cv_score = float(cv_index)

        # Build return tuple for this K
        cv_dict = {"avg": cv_score} if cv_score is not None else None
        if not no_freqs:
            if cv_dict is not None:
                res_k = (Q_np, P_np, cv_dict)
            else:
                res_k = (Q_np, P_np)
        else:
            if cv_dict is not None:
                res_k = (Q_np, cv_dict)
            else:
                res_k = Q_np

        results[this_K] = res_k

    if len(k_values) == 1:
        return results[k_values[0]]

    return results


def adamixture_run(args: Any = None, **kwargs: Any) -> Any:
    """
    Wrapper compatible with fastmixture_run signature.
    Accepts an argparse.Namespace object or dictionary/keyword arguments,
    extracts the parameters, and invokes run().
    """
    if args is not None:
        if isinstance(args, dict):
            merged = dict(args)
            merged.update(kwargs)
            return run(**merged)
        if hasattr(args, "__dict__"):
            ns_dict = vars(args)
        else:
            ns_dict = {
                k: getattr(args, k)
                for k in dir(args)
                if not k.startswith("_") and not callable(getattr(args, k))
            }
        merged = dict(ns_dict)
        merged.update(kwargs)
        return run(**merged)

    return run(**kwargs)
