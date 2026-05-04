# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
from libc.math cimport fabs
from libc.stdlib cimport malloc, free

# Solve linear system for variable subset
cdef bint solve_subset_ptr(const double* A, const double* b, double* x, int* F, int K, double* aug, int* map_idx) noexcept nogil:
    """
    Description:
    Solves a linear system Ax = b for a subset of variables using Gaussian elimination with pivoting.

    Args:
        A (const double*): Full covariance matrix (K x K).
        b (const double*): Full target vector (K).
        x (double*): Output vector for solution (K).
        F (int*): Mask indicator of Free (1) vs Fixed (0) variables.
        K (int): Total number of variables.
        aug (double*): Buffer for augmented matrix (K * (K+1)).
        map_idx (int*): Buffer for mapping indices of free variables.

    Returns:
        bint: True if system was successfully solved, False if matrix was singular.
    """
    cdef int n = 0, i, j, k, max_row
    cdef double max_val, temp, pivot

    for i in range(K):
        if F[i] == 1:
            map_idx[n] = i
            n += 1

    if n == 0:
        return True

    for i in range(n):
        for j in range(n):
            aug[i * (n + 1) + j] = A[map_idx[i] * K + map_idx[j]]
        aug[i * (n + 1) + n] = b[map_idx[i]]

    for i in range(n):
        max_row = i
        max_val = aug[i * (n + 1) + i]
        if max_val < 0: max_val = -max_val
        for k in range(i+1, n):
            temp = aug[k * (n + 1) + i]
            if temp < 0: temp = -temp
            if temp > max_val:
                max_val = temp
                max_row = k
        
        if max_val < 1e-12:
            return False 

        if max_row != i:
            for j in range(i, n+1):
                temp = aug[i * (n + 1) + j]
                aug[i * (n + 1) + j] = aug[max_row * (n + 1) + j]
                aug[max_row * (n + 1) + j] = temp

        pivot = aug[i * (n + 1) + i]
        for j in range(i, n+1):
            aug[i * (n + 1) + j] /= pivot

        for k in range(n):
            if k != i:
                temp = aug[k * (n + 1) + i]
                for j in range(i, n+1):
                    aug[k * (n + 1) + j] -= temp * aug[i * (n + 1) + j]

    for i in range(n):
        x[map_idx[i]] = aug[i * (n + 1) + n]
    return True

# Exact NNLS using Block Principal Pivoting
cdef void _exact_nnls_bpp_ptr(const double* A, const double* b, double* x, int K, double* aug, int* map_idx, int* F, double* y) noexcept nogil:
    """
    Description:
    Computes Exact Non-Negative Least Squares (NNLS) solution using Block Principal Pivoting.

    Args:
        A (const double*): Positive-definite matrix A (K x K).
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
    cdef int i, j, iter_count
    cdef bint changed
    
    for i in range(K):
        F[i] = 1
        x[i] = 0.0
        
    for iter_count in range(50):
        for i in range(K):
            if F[i] == 0:
                x[i] = 0.0
                
        if not solve_subset_ptr(A, b, x, F, K, aug, map_idx):
            break

        for i in range(K):
            if F[i] == 0:
                y[i] = -b[i]
                for j in range(K):
                    if F[j] == 1:
                        y[i] += A[i * K + j] * x[j]
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
cdef void _exact_bvls_bpp_ptr(const double* A, const double* b, double* x, int K, double* aug, int* map_idx, int* F, int* U, double* y, double* b_subset, double lower, double upper) noexcept nogil:
    """
    Description:
    Computes Exact Bounded Variable Least Squares (BVLS) solution using Block Principal Pivoting.

    Args:
        A (const double*): Positive-definite matrix A (K x K).
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
    cdef bint changed
    
    # Initialize: everything Free (F=1), nothing at Upper (U=0)
    for i in range(K):
        F[i] = 1
        U[i] = 0
        x[i] = 0.0
        
    for iter_count in range(50):
        # Adjust b for subset solve: b_F_prime = b_F - A_FU * upper - A_FL * lower
        for i in range(K):
            if F[i] == 1:
                b_subset[i] = b[i]
                for j in range(K):
                    if F[j] == 0:
                        if U[j] == 1:
                            b_subset[i] -= A[i * K + j] * upper
                        else:
                            b_subset[i] -= A[i * K + j] * lower
        
        # solve A_FF * x_F = b_subset_F
        if not solve_subset_ptr(A, b_subset, x, F, K, aug, map_idx):
            break
            
        # Set x for non-free variables
        for i in range(K):
            if F[i] == 0:
                if U[i] == 1:
                    x[i] = upper
                else:
                    x[i] = lower
                    
        # Calculate gradient: y = Ax - b
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
cpdef void batch_nnls_bpp(double[:, ::1] A_cov, double[:, ::1] B_target, double[:, ::1] Out_mat) noexcept nogil:
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
    
    cdef double* aug
    cdef int* map_idx
    cdef int* F
    cdef double* y
    
    with nogil, parallel():
        for i in prange(M, schedule='static'):
            aug = <double*>malloc(K * (K + 1) * sizeof(double))
            map_idx = <int*>malloc(K * sizeof(int))
            F = <int*>malloc(K * sizeof(int))
            y = <double*>malloc(K * sizeof(double))
            
            _exact_nnls_bpp_ptr(A_ptr, &B_target[i, 0], &Out_mat[i, 0], K, aug, map_idx, F, y)
            
            free(aug)
            free(map_idx)
            free(F)
            free(y)

# Batch BVLS solver
cpdef void batch_bvls_bpp(double[:, ::1] A_cov, double[:, ::1] B_target, double[:, ::1] Out_mat, double lower, double upper) noexcept nogil:
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
    
    cdef double* aug
    cdef int* map_idx
    cdef int* F
    cdef int* U
    cdef double* y
    cdef double* b_subset
    
    with nogil, parallel():
        for i in prange(M, schedule='static'):
            aug = <double*>malloc(K * (K + 1) * sizeof(double))
            map_idx = <int*>malloc(K * sizeof(int))
            F = <int*>malloc(K * sizeof(int))
            U = <int*>malloc(K * sizeof(int))
            y = <double*>malloc(K * sizeof(double))
            b_subset = <double*>malloc(K * sizeof(double))
            
            _exact_bvls_bpp_ptr(A_ptr, &B_target[i, 0], &Out_mat[i, 0], K, aug, map_idx, F, U, y, b_subset, lower, upper)
            
            free(aug)
            free(map_idx)
            free(F)
            free(U)
            free(y)
            free(b_subset)
