import os

# Idle OpenMP threads must yield their cores: the kernels alternate with
# multithreaded BLAS calls (SVD, ALS), and spinning workers of either runtime
# starve the other one (up to 50x slower ALS on Linux with OpenBLAS).
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

from ._version import __version__ as __version__  # noqa: E402
from ._version import __version_tuple__ as __version_tuple__  # noqa: E402
from .api import run, adamixture_run  # noqa: E402

__all__ = ["run", "adamixture_run", "__version__", "__version_tuple__"]
