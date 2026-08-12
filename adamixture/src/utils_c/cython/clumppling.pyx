# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""
Combinatorial kernels backing the in-tree Clumppling port in adamixture/src/clumppling.py.

Adapted from Clumppling (https://github.com/PopGenClustering/Clumppling), MIT licensed,
by Xiran Liu et al. The original package delegates these steps to third-party solvers
(cvxpy/GLPK for the cluster-alignment ILP, cdlib/python-louvain for mode detection).
The routines below reimplement them so that only NumPy is needed at the Python level.
"""
from libc.float cimport DBL_MAX
from libc.math cimport exp, fabs, lgamma, log

import numpy as np


def solve_lap(double[:, ::1] cost):
    """
    Description:
    Solves the rectangular linear sum assignment problem, matching every row of the
    cost matrix to a distinct column at minimum total cost, via the Jonker-Volgenant
    shortest augmenting path method.

    Args:
        cost (double[:, ::1]): Cost matrix with no more rows than columns.

    Returns:
        np.ndarray: Column index assigned to each row.
    """
    cdef Py_ssize_t n = cost.shape[0]
    cdef Py_ssize_t m = cost.shape[1]

    if n > m:
        raise ValueError("solve_lap requires a cost matrix with no more rows than columns.")

    assignment = np.empty(n, dtype=np.intp)
    cdef Py_ssize_t[::1] out = assignment

    # Dual potentials (u, v), current match of each column (p) and augmenting path parents (way).
    cdef double[::1] u = np.zeros(n + 1, dtype=np.float64)
    cdef double[::1] v = np.zeros(m + 1, dtype=np.float64)
    cdef double[::1] minv = np.empty(m + 1, dtype=np.float64)
    cdef Py_ssize_t[::1] p = np.zeros(m + 1, dtype=np.intp)
    cdef Py_ssize_t[::1] way = np.zeros(m + 1, dtype=np.intp)
    cdef unsigned char[::1] used = np.empty(m + 1, dtype=np.uint8)

    cdef Py_ssize_t i, j, j0, j1, i0
    cdef double delta, cur

    with nogil:
        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            for j in range(m + 1):
                minv[j] = DBL_MAX
                used[j] = 0

            while True:
                used[j0] = 1
                i0 = p[j0]
                delta = DBL_MAX
                j1 = 0
                for j in range(1, m + 1):
                    if used[j] == 0:
                        cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for j in range(m + 1):
                    if used[j] != 0:
                        u[p[j]] += delta
                        v[j] -= delta
                    else:
                        minv[j] -= delta
                j0 = j1
                if p[j0] == 0:
                    break

            while j0 != 0:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1

        for j in range(1, m + 1):
            if p[j] != 0:
                out[p[j] - 1] = j - 1

    return assignment


cdef inline Py_ssize_t _next_index(unsigned long long* state, Py_ssize_t bound) noexcept nogil:
    """
    Description:
    Draws an index in [0, bound) from a 64-bit linear congruential generator, so that the
    node visiting order is shuffled reproducibly without depending on NumPy's RNG.

    Args:
        state (unsigned long long*): Mutable generator state.
        bound (Py_ssize_t): Exclusive upper bound of the drawn index.

    Returns:
        Py_ssize_t: Drawn index.
    """
    state[0] = state[0] * <unsigned long long>6364136223846793005ULL + <unsigned long long>1442695040888963407ULL
    return <Py_ssize_t>((state[0] >> 33) % <unsigned long long>bound)


cdef double _modularity(double[::1] internals, double[::1] com_degrees,
                        Py_ssize_t n, double links, double resolution) noexcept nogil:
    """
    Description:
    Computes the resolution-weighted modularity of the current partition. Empty
    communities contribute zero, so every slot can be summed unconditionally.

    Args:
        internals (double[::1]): Weight of the edges internal to each community.
        com_degrees (double[::1]): Total weighted degree of each community.
        n (Py_ssize_t): Number of community slots.
        links (double): Total edge weight of the graph.
        resolution (double): Resolution parameter.

    Returns:
        double: Modularity of the partition.
    """
    cdef Py_ssize_t c
    cdef double result = 0.0
    cdef double share
    for c in range(n):
        share = com_degrees[c] / (2.0 * links)
        result += internals[c] * resolution / links - share * share
    return result


def one_level(double[:, ::1] W, double resolution, unsigned long long seed):
    """
    Description:
    Runs the local-moving phase of Louvain on a weighted graph, repeatedly moving each
    node to the neighbouring community with the largest modularity gain until a full
    sweep no longer improves the partition.

    Args:
        W (double[:, ::1]): Symmetric weight matrix whose diagonal holds self-loop weights.
        resolution (double): Resolution parameter.
        seed (unsigned long long): Seed for the node visiting order.

    Returns:
        tuple: Community label of every node, and the modularity of the resulting partition.
    """
    cdef Py_ssize_t n = W.shape[0]

    labels = np.arange(n, dtype=np.intp)
    cdef Py_ssize_t[::1] com = labels
    cdef double[::1] degrees = np.empty(n, dtype=np.float64)
    cdef double[::1] com_degrees = np.empty(n, dtype=np.float64)
    cdef double[::1] internals = np.empty(n, dtype=np.float64)
    cdef double[::1] shared = np.zeros(n, dtype=np.float64)
    cdef Py_ssize_t[::1] touched = np.empty(n, dtype=np.intp)
    cdef unsigned char[::1] seen = np.zeros(n, dtype=np.uint8)
    cdef Py_ssize_t[::1] order = np.empty(n, dtype=np.intp)

    cdef Py_ssize_t i, j, t, n_touched, pos, swap
    cdef Py_ssize_t node, com_node, best_com, c, tmp
    cdef double links = 0.0, trace = 0.0, row_sum
    cdef double degc_totw, remove_cost, incr, best_incr, own_weight
    cdef double cur_mod, new_mod = 0.0
    cdef bint modified = True
    cdef unsigned long long state = seed

    with nogil:
        for i in range(n):
            row_sum = 0.0
            for j in range(n):
                row_sum += W[i, j]
            degrees[i] = row_sum + W[i, i]
            com_degrees[i] = degrees[i]
            internals[i] = W[i, i]
            links += row_sum
            trace += W[i, i]
        links = (links + trace) / 2.0

        if links > 0.0:
            new_mod = _modularity(internals, com_degrees, n, links, resolution)

            while modified:
                cur_mod = new_mod
                modified = False

                for i in range(n):
                    order[i] = i
                for i in range(n - 1, 0, -1):
                    swap = _next_index(&state, i + 1)
                    tmp = order[i]
                    order[i] = order[swap]
                    order[swap] = tmp

                for pos in range(n):
                    node = order[pos]
                    com_node = com[node]
                    degc_totw = degrees[node] / (2.0 * links)

                    # Weight linking this node to each neighbouring community.
                    n_touched = 0
                    for j in range(n):
                        if j != node and W[node, j] != 0.0:
                            c = com[j]
                            if seen[c] == 0:
                                seen[c] = 1
                                touched[n_touched] = c
                                n_touched += 1
                            shared[c] += W[node, j]

                    own_weight = shared[com_node]
                    remove_cost = -own_weight + resolution * (com_degrees[com_node] - degrees[node]) * degc_totw
                    com_degrees[com_node] -= degrees[node]
                    internals[com_node] -= own_weight + W[node, node]

                    best_com = com_node
                    best_incr = 0.0
                    for t in range(n_touched):
                        c = touched[t]
                        incr = remove_cost + shared[c] - resolution * com_degrees[c] * degc_totw
                        if incr > best_incr:
                            best_incr = incr
                            best_com = c

                    com[node] = best_com
                    com_degrees[best_com] += degrees[node]
                    internals[best_com] += shared[best_com] + W[node, node]
                    if best_com != com_node:
                        modified = True

                    for t in range(n_touched):
                        c = touched[t]
                        seen[c] = 0
                        shared[c] = 0.0

                new_mod = _modularity(internals, com_degrees, n, links, resolution)
                if new_mod - cur_mod < 1e-7:
                    break

    return labels, new_mod


cdef double _regularized_gamma_p(double a, double x) noexcept nogil:
    """
    Description:
    Evaluates the regularized lower incomplete gamma function P(a, x), using the series
    expansion below the transition point and the continued fraction above it.

    Args:
        a (double): Shape parameter, strictly positive.
        x (double): Evaluation point.

    Returns:
        double: Value of P(a, x).
    """
    cdef double ap, total, delta, an, b, c, d, h
    cdef int i

    if x <= 0.0:
        return 0.0

    if x < a + 1.0:
        ap = a
        total = 1.0 / a
        delta = total
        for i in range(1000):
            ap += 1.0
            delta *= x / ap
            total += delta
            if fabs(delta) < fabs(total) * 1e-15:
                break
        return total * exp(-x + a * log(x) - lgamma(a))

    b = x + 1.0 - a
    c = 1e300
    d = 1.0 / b
    h = d
    for i in range(1, 1001):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if fabs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if fabs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if fabs(delta - 1.0) < 1e-15:
            break
    return 1.0 - exp(-x + a * log(x) - lgamma(a)) * h


def tracy_widom_cdf(double[::1] x, double shape, double scale, double shift):
    """
    Description:
    Evaluates a Tracy-Widom cumulative distribution function through the shifted gamma
    approximation of Chiani (2014), which reproduces the first three moments of the exact
    law and stays within a few thousandths of it across the support.

    Args:
        x (double[::1]): Points at which the distribution is evaluated.
        shape (double): Shape parameter of the fitted gamma distribution.
        scale (double): Scale parameter of the fitted gamma distribution.
        shift (double): Location offset of the fitted gamma distribution.

    Returns:
        np.ndarray: Cumulative probabilities at the requested points.
    """
    cdef Py_ssize_t n = x.shape[0]
    values = np.empty(n, dtype=np.float64)
    cdef double[::1] out = values
    cdef Py_ssize_t i

    with nogil:
        for i in range(n):
            out[i] = _regularized_gamma_p(shape, (x[i] + shift) / scale)

    return values
