"""Compiled kernels.

On x86-64 CPUs with AVX2 and FMA, the builds of the hot kernels compiled with
those instructions (``cython_avx2``) are registered under the names of this
package, so every ``from ...utils_c.cython import sqp`` gets them. Set
``ADAMIXTURE_DISABLE_AVX2=1`` to force the baseline build.
"""
import importlib
import os
import sys

_AVX2_MODULES = ("tools", "em", "br_qn", "bvls", "sqp")

variant = "baseline"


def _cpu_has_avx2_fma() -> bool:
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:
        try:
            from numpy.core._multiarray_umath import __cpu_features__ as features
        except ImportError:
            return False
    return bool(features.get("AVX2")) and bool(features.get("FMA3"))


if os.environ.get("ADAMIXTURE_DISABLE_AVX2", "0") != "1" and _cpu_has_avx2_fma():
    try:
        _modules = {name: importlib.import_module(f"{__package__}_avx2.{name}") for name in _AVX2_MODULES}
    except ImportError:
        _modules = {}
    for _name, _module in _modules.items():
        sys.modules[f"{__name__}.{_name}"] = _module
        globals()[_name] = _module
    if _modules:
        variant = "avx2"
