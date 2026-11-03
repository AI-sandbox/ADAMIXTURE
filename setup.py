import logging
import os
import platform
import re
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

system = platform.system()
common_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

compile_args = []
link_args = []
mac_include = []

if system == "Linux":
    compile_args = ['-O3', '-ffast-math', '-fno-wrapv']
    link_args = ['-lm']
    if "CC" not in os.environ:
        os.environ["CC"] = "gcc"
    if "CXX" not in os.environ:
        os.environ["CXX"] = "g++"
elif system == "Darwin":  # macOS
    omp_path = ""
    for p in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp", os.path.expanduser("~/.local/libomp")]:
        if os.path.exists(p):
            omp_path = p
            break

    if omp_path:
        compile_args = ['-O3', '-ffast-math', '-fno-wrapv']
        link_args = ['-lm', f'-L{omp_path}/lib', f'-Wl,-rpath,{omp_path}/lib']
        common_macros.append(('HAVE_OPENMP', '1'))
        mac_include = [f'{omp_path}/include']
    else:
        log.warning("OpenMP (libomp) not found via Homebrew; will attempt fallback build.")

    if "CC" not in os.environ:
        os.environ["CC"] = "clang"
    if "CXX" not in os.environ:
        os.environ["CXX"] = "clang++"
elif system == "Windows":
    if os.environ.get("CC", "").endswith("gcc"):
        compile_args = ['-O3']
        link_args = []
    else:
        compile_args = ['/O2']
        link_args = []
else:
    log.info(f"System not recognized: {system}")

# Opt-in: tune for the build machine (binaries may not run on older CPUs).
if os.environ.get("ADAMIXTURE_NATIVE", "0") == "1" and system in ("Linux", "Darwin"):
    if platform.machine().lower() in ("arm64", "aarch64"):
        compile_args.append('-mcpu=native')
    else:
        compile_args.append('-march=native')


class build_ext_openmp(build_ext):
    openmp_compile_args = {
        "msvc": [["/openmp"]],
        "intel": [["-qopenmp"]],
        "*": [["-fopenmp"], ["-Xpreprocessor", "-fopenmp"]],
    }
    openmp_link_args = {
        "msvc": [[]],
        "intel": [["-qopenmp"]],
        "*": [["-fopenmp"], ["-lomp"]],
    }

    def build_extension(self, ext):
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith("intel"):
            compiler = "intel"
        if compiler not in self.openmp_compile_args:
            compiler = "*"

        compile_original = getattr(self.compiler, "_compile", None)

        if compile_original:
            def compile_patched(obj, src, ext, cc_args, extra_postargs, pp_opts):
                if src.lower().endswith(".c"):
                    extra_postargs = [
                        arg for arg in extra_postargs if not arg.lower().startswith("-std")
                    ]
                return compile_original(obj, src, ext, cc_args, extra_postargs, pp_opts)

            self.compiler._compile = compile_patched

        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args = list(ext.extra_link_args)

        for c_args, l_args in zip(
            self.openmp_compile_args[compiler], self.openmp_link_args[compiler]
        ):
            try:
                ext.extra_compile_args = _extra_compile_args + c_args
                ext.extra_link_args = _extra_link_args + l_args
                print(">>> Attempting build with OpenMP support:", c_args, l_args)
                return super().build_extension(ext)
            except Exception as e:
                print(f">>> Compiling with OpenMP flags '{' '.join(c_args)}' failed: {e}")

        print(">>> Compiling with OpenMP support failed; re-trying without OpenMP.")
        ext.extra_compile_args = _extra_compile_args
        ext.extra_link_args = _extra_link_args
        return super().build_extension(ext)


CYTHON_DIR = "adamixture/src/utils_c/cython"
AVX2_DIR = "adamixture/src/utils_c/cython_avx2"
# Kernels that are also built with AVX2 + FMA; cython/__init__.py picks that
# build at import time when the CPU supports it.
AVX2_MODULES = ("tools", "em", "br_qn", "bvls", "sqp")


def avx2_compile_args() -> list[str] | None:
    """Flags for the AVX2 + FMA kernel variants, or None when the wheel does
    not target x86-64 (or a native build was requested)."""
    if os.environ.get("ADAMIXTURE_NATIVE", "0") == "1" or os.environ.get("ADAMIXTURE_NO_AVX2", "0") == "1":
        return None
    machine = platform.machine().lower()
    # cibuildwheel cross-compiles macOS x86_64 wheels on arm64 runners.
    archs = re.findall(r"-arch\s+(\S+)", os.environ.get("ARCHFLAGS", ""))
    if system == "Darwin" and archs:
        machine = archs[0] if len(archs) == 1 else "universal"
    if machine not in ("x86_64", "amd64"):
        return None
    if system == "Windows" and not os.environ.get("CC", "").endswith("gcc"):
        return ["/arch:AVX2"]
    return ["-mavx2", "-mfma"]


def avx2_stub(name: str) -> str:
    """Writes cython_avx2/<name>.pyx, which includes the kernel source with
    its compiler directives, so both builds share one implementation."""
    with open(f"{CYTHON_DIR}/{name}.pyx") as fh:
        header = fh.readline()
    content = (header if header.startswith("# cython:") else "") + f'include "../cython/{name}.pyx"\n'
    os.makedirs(AVX2_DIR, exist_ok=True)
    init = f"{AVX2_DIR}/__init__.py"
    if not os.path.exists(init):
        open(init, "w").close()
    stub = f"{AVX2_DIR}/{name}.pyx"
    if not os.path.exists(stub) or open(stub).read() != content:
        with open(stub, "w") as fh:
            fh.write(content)
    return stub


def kernel_extension(package: str, name: str, source: str, extra_args: list[str] = ()) -> Extension:
    return Extension(
        name=f"adamixture.src.utils_c.{package}.{name}",
        sources=[source],
        extra_compile_args=compile_args + list(extra_args),
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()] + mac_include,
        define_macros=common_macros,
    )


extensions = [
    kernel_extension("cython", name, f"{CYTHON_DIR}/{name}.pyx")
    for name in ("tools", "em", "br_qn", "clumppling", "snp_reader", "bvls", "sqp")
]
avx2_args = avx2_compile_args()
if avx2_args is not None:
    extensions += [kernel_extension("cython_avx2", name, avx2_stub(name), avx2_args) for name in AVX2_MODULES]


setup(
    cmdclass={"build_ext": build_ext_openmp},
    ext_modules=cythonize(extensions),
    include_package_data=True,
    package_data={
        "adamixture": [
            "src/utils_c/cuda/*.cu",
            "src/utils_c/cython/*.pyx",
            "src/utils_c/metal/*.metal",
            "demo/README.md",
            "demo/run_demo.sh",
            "demo/run_diagnostics.py",
            "demo/generate_device_expected.py",
            "demo/data/*",
            "demo/outputs/reader/*.expected",
            "demo/outputs/cpu/brqn/*.expected",
            "demo/outputs/cpu/adamem/*.expected",
            "demo/outputs/gpu/brqn/*.expected",
            "demo/outputs/gpu/adamem/*.expected",
            "demo/outputs/mps/brqn/*.expected",
            "demo/outputs/mps/adamem/*.expected",
        ],
    },
)

