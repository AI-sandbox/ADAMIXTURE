import logging
import os
import platform
import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

system = platform.system()
common_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

if system == "Linux":
    compile_args = ['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv']
    link_args = ['-fopenmp', '-lm']
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    mac_include = []
elif system == "Darwin":  # macOS
    # Try to find Homebrew libomp
    omp_path = ""
    for p in ["/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"]:
        if os.path.exists(p):
            omp_path = p
            break

    if omp_path:
        compile_args = ['-Xpreprocessor', '-fopenmp', '-O3', '-ffast-math', '-fno-wrapv']
        link_args = ['-lomp', '-lm', f'-L{omp_path}/lib']
        common_macros.append(('HAVE_OPENMP', '1'))
        mac_include = [f'{omp_path}/include']
    else:
        log.error("\n" + "="*80)
        log.error("ERROR: OpenMP (libomp) not found! Installation WILL FAIL on macOS.")
        log.error("Please install it via Homebrew: brew install libomp")
        log.error("="*80 + "\n")
        sys.exit(1)

    os.environ["CC"] = "clang"
    os.environ["CXX"] = "clang++"
elif system == "Windows":
    if os.environ.get("CC", "").endswith("gcc"):
        compile_args = ['-O3', '-fopenmp']
    else:
        compile_args = ['/O2', '/openmp']
    mac_include = []
else:
    log.info(f"System not recognized: {system}")
    sys.exit(1)
    mac_include = []


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
]


setup(
    ext_modules=cythonize(extensions),
    include_package_data=True,
    package_data={
        "adamixture": ["src/utils_c/cuda/*.cu", "src/utils_c/cython/*.pyx"],
    },
)
