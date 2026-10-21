# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
from cython.parallel import parallel, prange
cimport openmp
from libc.math cimport fabs, log
from libc.stdint cimport int32_t, uint8_t
from libc.stdlib cimport calloc, free, malloc

# ZAL Quasi-Newton: update U and V history matrices
cpdef void update_UV_ZAL(double[::1] U_flat, double[::1] V_flat, const double[::1] x, const double[::1] x_next,
                         const double[::1] x_next2, const int iteration, const int Q_hist, const Py_ssize_t dim) noexcept nogil:
    """
    Updates U and V history matrices in parallel.
    """
    cdef:
        Py_ssize_t col = (iteration - 1) % Q_hist
        Py_ssize_t offset = col * dim
        Py_ssize_t i

    for i in prange(dim, schedule='static'):
        U_flat[offset + i] = x_next[i] - x[i]
        V_flat[offset + i] = x_next2[i] - x_next[i]

# ZAL Quasi-Newton: compute the QN extrapolated point
cpdef void qn_extrapolate_ZAL(double[::1] x_qn, const double[::1] x_next, const double[::1] x, 
                              const double[::1] U_flat, const double[::1] V_flat, const int n_cols,
                              const Py_ssize_t dim, double[::1] UtUmV_workspace, double[::1] coeff_workspace) noexcept nogil:
    """
    Computes the ZAL quasi-Newton extrapolation using pre-allocated workspaces
    to avoid malloc/free and parallelizing vector operations.
    """
    cdef:
        int q = n_cols
        size_t i, j, k, inner_i, inner_j, idx
        double val_final
        int max_row
        double max_val, temp, pivot
        openmp.omp_lock_t zal_lock
        double* local_mat
        double* diff_UV
        double diff_x
        double u_ik

    # 1. Build the linear system U^T @ (U - V) and U^T @ (x - x_next)
    for idx in range(q * (q + 1)):
        UtUmV_workspace[idx] = 0.0

    openmp.omp_init_lock(&zal_lock)

    with nogil, parallel():
        local_mat = <double*>calloc(q * (q + 1), sizeof(double))
        diff_UV = <double*>malloc(q * sizeof(double))

        for k in prange(dim, schedule='static'):
            diff_x = x[k] - x_next[k]
            for inner_j in range(q):
                diff_UV[inner_j] = U_flat[inner_j * dim + k] - V_flat[inner_j * dim + k]

            for inner_i in range(q):
                u_ik = U_flat[inner_i * dim + k]
                for inner_j in range(q):
                    local_mat[inner_i * (q + 1) + inner_j] += u_ik * diff_UV[inner_j]
                local_mat[inner_i * (q + 1) + q] += u_ik * diff_x

        openmp.omp_set_lock(&zal_lock)
        for idx in range(q * (q + 1)):
            UtUmV_workspace[idx] += local_mat[idx]
        openmp.omp_unset_lock(&zal_lock)

        free(diff_UV)
        free(local_mat)

    openmp.omp_destroy_lock(&zal_lock)

    # 2. Solve UtUmV_workspace @ coeff_workspace = rhs via Gaussian Elimination
    # (This happens in the q x q space which is tiny, so it is sequential)
    for i in range(q):
        # Partial pivoting
        max_row = i
        max_val = fabs(UtUmV_workspace[i * (q + 1) + i])
        for k in range(i + 1, q):
            temp = fabs(UtUmV_workspace[k * (q + 1) + i])
            if temp > max_val:
                max_val = temp
                max_row = k

        if max_row != i:
            for j in range(q + 1):
                temp = UtUmV_workspace[i * (q + 1) + j]
                UtUmV_workspace[i * (q + 1) + j] = UtUmV_workspace[max_row * (q + 1) + j]
                UtUmV_workspace[max_row * (q + 1) + j] = temp

        pivot = UtUmV_workspace[i * (q + 1) + i]
        for j in range(i, q + 1):
            UtUmV_workspace[i * (q + 1) + j] /= pivot

        for k in range(q):
            if k != i:
                temp = UtUmV_workspace[k * (q + 1) + i]
                for j in range(i, q + 1):
                    UtUmV_workspace[k * (q + 1) + j] -= temp * UtUmV_workspace[i * (q + 1) + j]

    # Extract coefficients
    for i in range(q):
        coeff_workspace[i] = UtUmV_workspace[i * (q + 1) + q]

    # 3. Compute x_qn = x_next - V @ coeff (Fully parallelized)
    for k in prange(dim, schedule='static'):
        val_final = x_next[k]
        for i in range(q):
            val_final = val_final - V_flat[i * dim + k] * coeff_workspace[i]
        x_qn[k] = val_final

# Computing the sum of squared binomial deviance residuals
cpdef double deviance_squared_sum(uint8_t[::1] geno_values, long[::1] flat_entries, double[:,::1] P, 
                                  double[:,::1] Q, Py_ssize_t N) noexcept nogil:
    """
    Computes the sum of squared binomial deviance residuals using pre-extracted genotype values.
    """
    cdef:
        Py_ssize_t n_entries = flat_entries.shape[0]
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t i, k
        long idx
        Py_ssize_t row, col
        uint8_t g_u8
        double mu, dev
        double total = 0.0
        double eps = 1e-10

    for i in prange(n_entries, schedule='static'):
        idx = flat_entries[i]
        row = idx // N
        col = idx % N
        g_u8 = geno_values[i]

        mu = 0.0
        for k in range(K):
            mu = mu + Q[col, k] * P[row, k]
        mu = 2.0 * mu

        if mu < eps:
            mu = eps
        elif mu > 2.0 - eps:
            mu = 2.0 - eps

        if g_u8 == 0:
            dev = 2.0 * log(2.0 / (2.0 - mu))
        elif g_u8 == 1:
            dev = -log(mu * (2.0 - mu))
        else:
            dev = 2.0 * log(2.0 / mu)

        total += dev

    return total

cpdef void mask_entries_i32(uint8_t[:, ::1] G, int32_t[:] flat_entries, uint8_t[::1] saved_values, Py_ssize_t N) noexcept nogil:
    cdef:
        Py_ssize_t n_entries = flat_entries.shape[0]
        Py_ssize_t i
        int32_t idx
        uint8_t* g_flat = &G[0, 0]

    for i in prange(n_entries, schedule='static'):
        idx = flat_entries[i]
        saved_values[i] = g_flat[idx]
        g_flat[idx] = 3

cpdef void restore_entries_i32(uint8_t[:, ::1] G, int32_t[:] flat_entries, uint8_t[::1] saved_values, Py_ssize_t N) noexcept nogil:
    cdef:
        Py_ssize_t n_entries = flat_entries.shape[0]
        Py_ssize_t i
        int32_t idx
        uint8_t* g_flat = &G[0, 0]

    for i in prange(n_entries, schedule='static'):
        idx = flat_entries[i]
        g_flat[idx] = saved_values[i]

cpdef void mask_entries_i64(uint8_t[:, ::1] G, long[:] flat_entries, uint8_t[::1] saved_values, Py_ssize_t N) noexcept nogil:
    cdef:
        Py_ssize_t n_entries = flat_entries.shape[0]
        Py_ssize_t i
        long idx
        uint8_t* g_flat = &G[0, 0]

    for i in prange(n_entries, schedule='static'):
        idx = flat_entries[i]
        saved_values[i] = g_flat[idx]
        g_flat[idx] = 3

cpdef void restore_entries_i64(uint8_t[:, ::1] G, long[:] flat_entries, uint8_t[::1] saved_values, Py_ssize_t N) noexcept nogil:
    cdef:
        Py_ssize_t n_entries = flat_entries.shape[0]
        Py_ssize_t i
        long idx
        uint8_t* g_flat = &G[0, 0]

    for i in prange(n_entries, schedule='static'):
        idx = flat_entries[i]
        g_flat[idx] = saved_values[i]

cpdef double deviance_squared_sum_i32(uint8_t[::1] geno_values, int32_t[:] flat_entries, double[:,::1] P,
                                      double[:,::1] Q, Py_ssize_t N) noexcept nogil:
    cdef:
        Py_ssize_t n_entries = flat_entries.shape[0]
        Py_ssize_t K = Q.shape[1]
        Py_ssize_t i, k
        int32_t idx
        Py_ssize_t row, col
        uint8_t g_u8
        double mu, dev
        double total = 0.0
        double eps = 1e-10

    for i in prange(n_entries, schedule='static'):
        idx = flat_entries[i]
        row = idx // N
        col = idx % N
        g_u8 = geno_values[i]

        mu = 0.0
        for k in range(K):
            mu = mu + Q[col, k] * P[row, k]
        mu = 2.0 * mu

        if mu < eps:
            mu = eps
        elif mu > 2.0 - eps:
            mu = 2.0 - eps

        if g_u8 == 0:
            dev = 2.0 * log(2.0 / (2.0 - mu))
        elif g_u8 == 1:
            dev = -log(mu * (2.0 - mu))
        else:
            dev = 2.0 * log(2.0 / mu)

        total += dev

    return total
