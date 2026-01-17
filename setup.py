from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os 

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
    
extensions = [
    Extension(
        name="adamixture.src.utils_c.tools",
        sources=["adamixture/src/utils_c/tools.pyx"],
        extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv'],
        extra_link_args=['-fopenmp', '-lm'],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
    Extension(
    name="adamixture.src.utils_c.em",
    sources=["adamixture/src/utils_c/em.pyx"],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv'],
    extra_link_args=['-fopenmp', '-lm'],
    include_dirs=[numpy.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
    Extension(
    name="adamixture.src.utils_c.rsvd",
    sources=["adamixture/src/utils_c/rsvd.pyx"],
    extra_compile_args=['-fopenmp', '-O3', '-ffast-math', '-march=native', '-fno-wrapv'],
    extra_link_args=['-fopenmp', '-lm'],
    include_dirs=[numpy.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
    ),
]

setup(
    ext_modules=cythonize(extensions),
    include_package_data=True,
)
