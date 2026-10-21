from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "adamixture" / "demo" / "data"
OUTPUTS_DIR = ROOT / "adamixture" / "demo" / "outputs"

READER_EXPECTED_DIR = OUTPUTS_DIR / "reader"
CPU_BRQN_EXPECTED_DIR = OUTPUTS_DIR / "cpu" / "brqn"
CPU_ADAMEM_EXPECTED_DIR = OUTPUTS_DIR / "cpu" / "adamem"
GPU_BRQN_EXPECTED_DIR = OUTPUTS_DIR / "gpu" / "brqn"
GPU_ADAMEM_EXPECTED_DIR = OUTPUTS_DIR / "gpu" / "adamem"
MPS_BRQN_EXPECTED_DIR = OUTPUTS_DIR / "mps" / "brqn"
MPS_ADAMEM_EXPECTED_DIR = OUTPUTS_DIR / "mps" / "adamem"

K = 7
SEED = 42
CHUNK_SIZE = 8192
POWER = 5
TOL_SVD = 1e-1
MAX_ALS = 1000
TOL_ALS = 1e-4
BETA_ALS = 0.0

EXPECTED_RTOL = 2e-2
EXPECTED_ATOL = 2e-2
EXPECTED_LOGL_ATOL = 50.0
