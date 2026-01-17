# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, infer_types=True

import numpy as np
cimport numpy as np
cimport openmp as omp
cimport cython
from cython.parallel import prange, parallel
from libc.stdlib cimport calloc, free

np.import_array()

# -----------------------------------------------------------------------------
# Kernels Cython (ahora A es M×N)
# -----------------------------------------------------------------------------

cdef void _multiply_AT_omega(
    const np.uint8_t[:, ::1] A_view,           # (M, N)
    const np.float32_t[:, ::1] Omega_view,     # (N, k′)
    const np.float32_t[::1] f_view,            # (M,)
    np.float32_t[:, ::1] Y_view,               # (N, k′)
    Py_ssize_t M, Py_ssize_t N, Py_ssize_t k_prime) noexcept nogil:

    cdef:
        Py_ssize_t Y_size = N * k_prime
        float centered_val
        int nthreads
        np.float32_t **buffers
        int tid, t
        Py_ssize_t l, i, j, x
        np.float32_t *Y_thr
        const np.uint8_t *a_ptr
        float f2

    nthreads = omp.omp_get_max_threads()
    buffers = <np.float32_t **> calloc(nthreads, sizeof(np.float32_t *))

    with nogil, parallel():
        tid = omp.omp_get_thread_num()
        buffers[tid] = <np.float32_t *> calloc(Y_size, sizeof(np.float32_t))
        Y_thr = buffers[tid]

        for i in prange(M, schedule="static"):
            a_ptr = &A_view[i, 0]
            f2 = 2.0 * f_view[i]
            for l in range(N):
                if a_ptr[l] != 3:
                    centered_val = <float>a_ptr[l] - f2
                    for j in range(k_prime):
                        Y_thr[l * k_prime + j] += centered_val * Omega_view[i, j]

    with gil:
        for t in range(nthreads):
            for x in range(Y_size):
                Y_view[0, x] += buffers[t][x]
            free(buffers[t])
        free(buffers)


cdef inline void _multiply_A_omega(
    const np.uint8_t[:, ::1] A_view,           # (M, N)
    const np.float32_t[:, ::1] Omega_view,     # (N, k′)
    const np.float32_t[::1] f_view,            # (M,)
    np.float32_t[:, ::1] Y_view,               # (M, k′)
    Py_ssize_t M, Py_ssize_t N, Py_ssize_t k_prime) noexcept nogil:

    cdef:
        size_t i, j, l
        float temp_sum, centered_val
        const np.uint8_t *a_ptr
        float f2

    for i in prange(M, schedule="guided"):
        a_ptr = &A_view[i, 0]
        f2 = 2.0 * f_view[i]
        for j in range(k_prime):
            temp_sum = 0.0
            for l in range(N):
                if a_ptr[l] != 3:
                    centered_val = <float>a_ptr[l] - f2
                    temp_sum =  temp_sum + centered_val * Omega_view[l, j]
            
            Y_view[i, j] = temp_sum

# -----------------------------------------------------------------------------
# Python-callable wrappers
# -----------------------------------------------------------------------------

def multiply_AT_omega(np.ndarray[np.uint8_t, ndim=2, mode="c"] A_np,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] Omega_np,
                      np.ndarray[np.float32_t, ndim=1, mode="c"] f_np,
                      np.ndarray[np.float32_t, ndim=2, mode="c"] Y_np):
    cdef int M = A_np.shape[0]
    cdef int N = A_np.shape[1]
    cdef int k_prime = Omega_np.shape[1]

    _multiply_AT_omega(A_np, Omega_np, f_np, Y_np, M, N, k_prime)


def multiply_A_omega(np.ndarray[np.uint8_t, ndim=2, mode="c"] A_np,
                     np.ndarray[np.float32_t, ndim=2, mode="c"] Omega_np,
                     np.ndarray[np.float32_t, ndim=1, mode="c"] f_np,
                     np.ndarray[np.float32_t, ndim=2, mode="c"] Y_np):
    cdef int M = A_np.shape[0]
    cdef int N = A_np.shape[1]
    cdef int k_prime = Omega_np.shape[1]

    _multiply_A_omega(A_np, Omega_np, f_np, Y_np, M, N, k_prime)
