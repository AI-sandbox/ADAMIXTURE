import argparse
import os
import sys
from pathlib import Path

import numpy as np


for env_var in (
    "MKL_NUM_THREADS",
    "MKL_MAX_THREADS",
    "OMP_NUM_THREADS",
    "OMP_MAX_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "OPENBLAS_NUM_THREADS",
    "OPENBLAS_MAX_THREADS",
):
    os.environ.setdefault(env_var, "1")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adamixture.model.als_gpu import ALS_gpu
from adamixture.model.br_qn_gpu import optimize_original_gpu
from adamixture.model.em_adam_gpu import optimize_parameters_gpu
from adamixture.src import utils
from adamixture.src.adamixture import setup


K = 7
SEED = 42
CHUNK_SIZE = 8192
POWER = 5
TOL_SVD = 1e-1
MAX_ALS = 1000
TOL_ALS = 1e-4

LR = 0.005
BETA1 = 0.80
BETA2 = 0.88
REG_ADAM = 1e-8
MAX_ITER = 10000
CHECK = 5
LR_DECAY = 0.5
MIN_LR = 1e-4
PATIENCE = 3
TOL = 0.1
Q_HIST = 3


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _save(path: Path, array) -> None:
    arr = _to_numpy(array)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, arr, fmt="%.18e")
    print(f"wrote {path.relative_to(ROOT)}")


def _canonicalize_svd(U, V):
    U_np = np.array(_to_numpy(U), copy=True)
    V_np = np.array(_to_numpy(V), copy=True)
    for col in range(U_np.shape[1]):
        pivot = int(np.argmax(np.abs(U_np[:, col])))
        if U_np[pivot, col] < 0:
            U_np[:, col] *= -1
            V_np[:, col] *= -1
    return U_np, V_np


def _device_output_name(device: str) -> str:
    return "gpu" if device == "cuda" else device


def generate_algorithm(device: str, algorithm: str) -> None:
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False.")

    data_path = ROOT / "adamixture" / "demo" / "data" / "demo_data.bed"
    out_dir = ROOT / "adamixture" / "demo" / "outputs" / _device_output_name(device) / algorithm

    print(f"Generating {device}/{algorithm} expected files")
    G, N, M = utils.read_data(
        str(data_path),
        packed=True,
        chunk_size=CHUNK_SIZE,
        chrom_mode="autosomes",
        autosomes=22,
        verbose=True,
    )

    is_brqn = algorithm == "brqn"
    device_obj, threads_per_block, f, U, S, V, G = setup(
        G,
        N,
        M,
        K,
        SEED,
        POWER,
        TOL_SVD,
        CHUNK_SIZE,
        device,
        original=is_brqn,
        init_original="als",
        q_hist=Q_HIST,
    )

    U_save, V_save = _canonicalize_svd(U, V)
    _save(out_dir / "demo_run.svd.U.expected", U_save)
    _save(out_dir / "demo_run.svd.S.expected", S)
    _save(out_dir / "demo_run.svd.V.expected", V_save)

    P_als, Q_als = ALS_gpu(
        U.contiguous(),
        S.contiguous(),
        V.contiguous(),
        f,
        SEED,
        M,
        K,
        MAX_ALS,
        TOL_ALS,
        device_obj,
    )
    _save(out_dir / "demo_run.als.P.expected", P_als)
    _save(out_dir / "demo_run.als.Q.expected", Q_als)

    if is_brqn:
        P_opt, Q_opt = optimize_original_gpu(
            G,
            P_als.clone(),
            Q_als.clone(),
            MAX_ITER,
            K,
            M,
            N,
            TOL,
            Q_HIST,
            PATIENCE,
            device_obj,
            CHUNK_SIZE,
            threads_per_block,
        )
        prefix = "brqn"
    else:
        P_opt, Q_opt = optimize_parameters_gpu(
            G,
            P_als.clone(),
            Q_als.clone(),
            LR,
            BETA1,
            BETA2,
            REG_ADAM,
            MAX_ITER,
            CHECK,
            K,
            M,
            N,
            LR_DECAY,
            MIN_LR,
            PATIENCE,
            TOL,
            device_obj,
            CHUNK_SIZE,
            threads_per_block,
        )
        prefix = "adamem"

    _save(out_dir / f"demo_run.{prefix}.P.expected", P_opt)
    _save(out_dir / f"demo_run.{prefix}.Q.expected", Q_opt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ADAMIXTURE device-specific demo expected outputs.")
    parser.add_argument("--device", choices=["cuda", "mps"], required=True)
    parser.add_argument("--algorithms", nargs="+", choices=["brqn", "adamem"], default=["brqn", "adamem"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for algorithm in args.algorithms:
        generate_algorithm(args.device, algorithm)


if __name__ == "__main__":
    main()
