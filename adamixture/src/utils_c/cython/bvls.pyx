# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.math cimport fabs, sqrt
from libc.stdlib cimport malloc, free

import numpy as np

# Solve linear system for variable subset
cdef int solve_subset_ptr(const double* A, const double* b, double* x, int* F, int K, double* aug, int* map_idx) noexcept nogil:
    """
    Description:
    Solves a linear system Ax = b for a subset of variables using Cholesky (LL^T) decomposition.
    Exploits symmetry and positive definiteness of covariance submatrices without pivot searching.

    Args:
        A (const double*): Full covariance matrix (K x K).
        b (const double*): Full target vector (K).
        x (double*): Output vector for solution (K).
        F (int*): Mask indicator of Free (1) vs Fixed (0) variables.
        K (int): Total number of variables.
        aug (double*): Buffer for Cholesky L matrix and forward solve y (K * (K+1)).
        map_idx (int*): Buffer for mapping indices of free variables.

    Returns:
        int: Number of free variables n if successful, or -1 if matrix was singular/not positive definite.
    """
    cdef int n = 0, i, j, k
    cdef double s, d, inv_d
    cdef double* L = aug
    cdef double* y

    for i in range(K):
        if F[i] == 1:
            map_idx[n] = i
            n += 1

    if n == 0:
        return 0

    y = &aug[n * n]

    # Cholesky factorization: A_sub = L L^T (lower triangular)
    for j in range(n):
        s = 0.0
        for k in range(j):
            s += L[j * n + k] * L[j * n + k]
        d = A[map_idx[j] * K + map_idx[j]] - s
        if d <= 1e-14:
            return -1
        L[j * n + j] = sqrt(d)
        inv_d = 1.0 / L[j * n + j]
        for i in range(j + 1, n):
            s = 0.0
            for k in range(j):
                s += L[i * n + k] * L[j * n + k]
            L[i * n + j] = (A[map_idx[i] * K + map_idx[j]] - s) * inv_d

    # Forward substitution: L y = b_sub
    for i in range(n):
        s = 0.0
        for k in range(i):
            s += L[i * n + k] * y[k]
        y[i] = (b[map_idx[i]] - s) / L[i * n + i]

    # Backward substitution: L^T x_sub = y
    for i in range(n - 1, -1, -1):
        s = 0.0
        for k in range(i + 1, n):
            s += L[k * n + i] * x[map_idx[k]]
        x[map_idx[i]] = (y[i] - s) / L[i * n + i]

    return n

# Exact NNLS using Block Principal Pivoting
cdef void _exact_nnls_bpp_ptr(const double* A, const double* A_inv, const double* b, double* x, int K, double* aug, int* map_idx, int* F, double* y) noexcept nogil:
    """
    Description:
    Computes Exact Non-Negative Least Squares (NNLS) solution using Block Principal Pivoting.

    Args:
        A (const double*): Positive-definite matrix A (K x K).
        A_inv (const double*): Precomputed inverse of matrix A (K x K).
        b (const double*): Target vector b (K).
        x (double*): Current and output solution vector x (K).
        K (int): Number of variables.
        aug (double*): Temporary buffer for augmented matrix.
        map_idx (int*): Temporary buffer for index mapping.
        F (int*): Temporary buffer for active set (Free/Fixed).
        y (double*): Temporary buffer for gradient/dual variable vector.

    Returns:
        None
    """
    cdef int i, j, idx, iter_count, n_free
    cdef bint all_nonneg = True, changed
    cdef double acc
    
    # Iteration 0: Analytical unconstrained solution x = A_inv @ b
    for i in range(K):
        acc = 0.0
        for j in range(K):
            acc += A_inv[i * K + j] * b[j]
        if acc < -1e-8:
            F[i] = 0
            x[i] = 0.0
            all_nonneg = False
        else:
            F[i] = 1
            x[i] = acc

    # Fast-path: if all components >= 0, KKT conditions are satisfied (global optimum)
    if all_nonneg:
        return

    # Iterations 1..49: Reduced sub-systems for variables violating bounds
    for iter_count in range(1, 50):
        for i in range(K):
            if F[i] == 0:
                x[i] = 0.0
                
        n_free = solve_subset_ptr(A, b, x, F, K, aug, map_idx)
        if n_free < 0:
            break

        for i in range(K):
            if F[i] == 0:
                acc = -b[i]
                for idx in range(n_free):
                    j = map_idx[idx]
                    acc += A[i * K + j] * x[j]
                y[i] = acc
            else:
                y[i] = 0.0

        changed = False
        for i in range(K):
            if F[i] == 1 and x[i] < -1e-8:
                F[i] = 0
                changed = True
            elif F[i] == 0 and y[i] < -1e-8:
                F[i] = 1
                changed = True

        if not changed:
            break

# Exact BVLS using Block Principal Pivoting
cdef void _exact_bvls_bpp_ptr(const double* A, const double* A_inv, const double* b, double* x, int K, double* aug, int* map_idx, int* F, int* U, double* y, double* b_subset, double lower, double upper) noexcept nogil:
    """
    Description:
    Computes Exact Bounded Variable Least Squares (BVLS) solution using Block Principal Pivoting.

    Args:
        A (const double*): Positive-definite matrix A (K x K).
        A_inv (const double*): Precomputed inverse of matrix A (K x K).
        b (const double*): Target vector b (K).
        x (double*): Current and output solution vector x (K).
        K (int): Number of variables.
        aug (double*): Temporary buffer for augmented matrix.
        map_idx (int*): Temporary buffer for index mapping.
        F (int*): Temporary buffer for active set indicator (Free/Fixed).
        U (int*): Temporary buffer for indicator if fixed at Upper bound (1) or Lower bound (0).
        y (double*): Temporary buffer for gradient vector.
        b_subset (double*): Temporary buffer for adjusted target vector.
        lower (double): Lower bound for variables.
        upper (double): Upper bound for variables.

    Returns:
        None
    """
    cdef int i, j, iter_count
    cdef bint changed = False
    cdef double acc
    
    # Iteration 0: Analytical unconstrained solution x = A_inv @ b
    for i in range(K):
        acc = 0.0
        for j in range(K):
            acc += A_inv[i * K + j] * b[j]
        if acc < lower - 1e-8:
            F[i] = 0
            U[i] = 0
            x[i] = lower
            changed = True
        elif acc > upper + 1e-8:
            F[i] = 0
            U[i] = 1
            x[i] = upper
            changed = True
        else:
            F[i] = 1
            U[i] = 0
            x[i] = acc

    # Fast-path: If all variables are within bounds, x is the unconstrained minimizer (optimal).
    if not changed:
        return
        
    for iter_count in range(1, 50):
        for i in range(K):
            if F[i] == 1:
                b_subset[i] = b[i]
                for j in range(K):
                    if F[j] == 0:
                        if U[j] == 1:
                            b_subset[i] -= A[i * K + j] * upper
                        else:
                            b_subset[i] -= A[i * K + j] * lower
        
        if solve_subset_ptr(A, b_subset, x, F, K, aug, map_idx) < 0:
            break
            
        for i in range(K):
            if F[i] == 0:
                if U[i] == 1:
                    x[i] = upper
                else:
                    x[i] = lower
                    
        for i in range(K):
            y[i] = -b[i]
            for j in range(K):
                y[i] += A[i * K + j] * x[j]

        changed = False
        for i in range(K):
            if F[i] == 1:
                if x[i] < lower - 1e-8:
                    F[i] = 0
                    U[i] = 0
                    changed = True
                elif x[i] > upper + 1e-8:
                    F[i] = 0
                    U[i] = 1
                    changed = True
            else: # Fixed
                if U[i] == 0: # At Lower bound
                    if y[i] < -1e-8:
                        F[i] = 1
                        U[i] = 0
                        changed = True
                else: # At Upper bound
                    if y[i] > 1e-8:
                        F[i] = 1
                        U[i] = 0
                        changed = True

        if not changed:
            break

# Batch NNLS solver
cpdef void batch_nnls_bpp(double[:, ::1] A_cov, double[:, ::1] B_target, double[:, ::1] Out_mat):
    """
    Description:
    Solves NNLS for a batch of target vectors against a single covariance matrix.
    Parallelized across targets.

    Args:
        A_cov (double[:, ::1]): Single covariance matrix (K x K).
        B_target (double[:, ::1]): Batch of target vectors (M x K).
        Out_mat (double[:, ::1]): Matrix to store solutions (M x K).

    Returns:
        None
    """
    cdef int M = B_target.shape[0]
    cdef int K = A_cov.shape[0]
    cdef int i
    cdef const double* A_ptr = &A_cov[0, 0]
    cdef double[:, ::1] A_inv = np.ascontiguousarray(np.linalg.pinv(A_cov), dtype=np.float64)
    cdef const double* A_inv_ptr = &A_inv[0, 0]
    
    cdef double* aug
    cdef int* map_idx
    cdef int* F
    cdef double* y
    
    with nogil, parallel():
        aug = <double*>malloc(K * (K + 1) * sizeof(double))
        map_idx = <int*>malloc(K * sizeof(int))
        F = <int*>malloc(K * sizeof(int))
        y = <double*>malloc(K * sizeof(double))

        for i in prange(M, schedule='static'):
            _exact_nnls_bpp_ptr(A_ptr, A_inv_ptr, &B_target[i, 0], &Out_mat[i, 0], K, aug, map_idx, F, y)

        free(aug)
        free(map_idx)
        free(F)
        free(y)

# Batch BVLS solver
cpdef void batch_bvls_bpp(double[:, ::1] A_cov, double[:, ::1] B_target, double[:, ::1] Out_mat, double lower, double upper):
    """
    Description:
    Solves BVLS for a batch of target vectors against a single covariance matrix.
    Parallelized across targets.

    Args:
        A_cov (double[:, ::1]): Single covariance matrix (K x K).
        B_target (double[:, ::1]): Batch of target vectors (M x K).
        Out_mat (double[:, ::1]): Matrix to store solutions (M x K).
        lower (double): Lower bound for all variables.
        upper (double): Upper bound for all variables.

    Returns:
        None
    """
    cdef int M = B_target.shape[0]
    cdef int K = A_cov.shape[0]
    cdef int i
    cdef const double* A_ptr = &A_cov[0, 0]
    cdef double[:, ::1] A_inv = np.ascontiguousarray(np.linalg.pinv(A_cov), dtype=np.float64)
    cdef const double* A_inv_ptr = &A_inv[0, 0]
    
    cdef double* aug
    cdef int* map_idx
    cdef int* F
    cdef int* U
    cdef double* y
    cdef double* b_subset
    
    with nogil, parallel():
        aug = <double*>malloc(K * (K + 1) * sizeof(double))
        map_idx = <int*>malloc(K * sizeof(int))
        F = <int*>malloc(K * sizeof(int))
        U = <int*>malloc(K * sizeof(int))
        y = <double*>malloc(K * sizeof(double))
        b_subset = <double*>malloc(K * sizeof(double))

        for i in prange(M, schedule='static'):
            _exact_bvls_bpp_ptr(A_ptr, A_inv_ptr, &B_target[i, 0], &Out_mat[i, 0], K, aug, map_idx, F, U, y, b_subset, lower, upper)

        free(aug)
        free(map_idx)
        free(F)
        free(U)
        free(y)
        free(b_subset)
