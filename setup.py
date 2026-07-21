import logging
import os
import platform
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
    for p in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]:
        if os.path.exists(p):
            omp_path = p
            break

    if omp_path:
        compile_args = ['-O3', '-ffast-math', '-fno-wrapv']
        link_args = ['-lm', f'-L{omp_path}/lib']
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


extensions = [
    Extension(
        name="adamixture.src.utils_c.cython.tools",
        sources=["adamixture/src/utils_c/cython/tools.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()] + mac_include,
        define_macros=common_macros
    ),
    Extension(
        name="adamixture.src.utils_c.cython.em",
        sources=["adamixture/src/utils_c/cython/em.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()] + mac_include,
        define_macros=common_macros
    ),
    Extension(
        name="adamixture.src.utils_c.cython.br_qn",
        sources=["adamixture/src/utils_c/cython/br_qn.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()] + mac_include,
        define_macros=common_macros
    ),
    Extension(
        name="adamixture.src.utils_c.cython.snp_reader",
        sources=["adamixture/src/utils_c/cython/snp_reader.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()] + mac_include,
        define_macros=common_macros
    ),
    Extension(
        name="adamixture.src.utils_c.cython.bvls",
        sources=["adamixture/src/utils_c/cython/bvls.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()] + mac_include,
        define_macros=common_macros
    ),
    Extension(
        name="adamixture.src.utils_c.cython.sqp",
        sources=["adamixture/src/utils_c/cython/sqp.pyx"],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        include_dirs=[numpy.get_include()] + mac_include,
        define_macros=common_macros
    ),
]


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

