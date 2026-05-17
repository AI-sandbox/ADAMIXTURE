import torch

from ..src import utils
from .em_adam_gpu import EMAdamOptimizer


def _flatten_PQ_gpu(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Flattens P and Q into a single 1-D parameter vector [P_flat, Q_flat].
    """
    return torch.cat([P.ravel(), Q.ravel()])

def _unflatten_PQ_gpu(x: torch.Tensor, P: torch.Tensor, Q: torch.Tensor, M: int, K: int) -> None:
    """
    Unflattens a 1-D parameter vector back into P and Q matrices.
    """
    P.copy_(x[:M * K].view(M, K))
    Q.copy_(x[M * K:].view(-1, K))

def _mapPQ_gpu(P: torch.Tensor, Q: torch.Tensor) -> None:
    """
    Projects P and Q back onto the feasible set (non-negative, Q rows sum to 1).
    """
    torch.clamp_(P, 1e-5, 1.0 - 1e-5)
    torch.clamp_(Q, 1e-5, 1.0 - 1e-5)
    Q.div_(Q.sum(dim=1, keepdim=True))

def _update_history_math(x: torch.Tensor, x_next: torch.Tensor, x_next2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes U and V history increments.
    """
    return x_next - x, x_next2 - x_next

def _extrapolate_math(x_next: torch.Tensor, V_sub: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """
    Computes the QN extrapolated vector.
    """
    return x_next - V_sub @ alpha

# Compilation
_mapPQ_compiled = torch.compile(_mapPQ_gpu, disable=not hasattr(torch, "compile"))
_update_history_compiled = torch.compile(_update_history_math, disable=not hasattr(torch, "compile"))
_extrapolate_compiled = torch.compile(_extrapolate_math, disable=not hasattr(torch, "compile"))

def polish_br_qn_gpu(G: torch.Tensor, P_init: torch.Tensor, Q_init: torch.Tensor,
                    M: int, N: int, K: int, args,
                    device: torch.device, threads_per_block: int,
                    n_iters: int = 3, Q_hist: int = 3) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Polishes P and Q tensors on GPU using block-relaxation with ZAL quasi-Newton
    acceleration. Designed for cross-validation fold polishing.
    """
    P = P_init.clone()
    Q = Q_init.clone()

    # We use EMAdamOptimizer to reuse its accumulation buffers and EM logic,
    # but we skip the Adam step and use pure EM updates.
    optimizer = EMAdamOptimizer(P.shape, Q.shape, 0.0, 0.0, 0.0, 0.0, device)
    unpacker = utils.get_unpacker(device, threads_per_block)
    chunk_size = int(args.chunk_size)

    dim = M * K + N * K
    U = torch.zeros((dim, Q_hist), dtype=P.dtype, device=device)
    V = torch.zeros((dim, Q_hist), dtype=P.dtype, device=device)

    for it in range(1, n_iters + 1):
        # --- Block-relaxation step 1: (P, Q) -> (P1, Q1) ---
        P1_EM, Q1_EM = optimizer.run_em_step(G, P, Q, M, chunk_size, unpacker)
        P1 = P1_EM.clone()
        Q1 = Q1_EM.clone()

        # --- Block-relaxation step 2: (P1, Q1) -> (P2, Q2) ---
        P2_EM, Q2_EM = optimizer.run_em_step(G, P1, Q1, M, chunk_size, unpacker)
        P2 = P2_EM.clone()
        Q2 = Q2_EM.clone()

        # --- Flatten for QN ---
        x = _flatten_PQ_gpu(P, Q)
        x_next = _flatten_PQ_gpu(P1, Q1)
        x_next2 = _flatten_PQ_gpu(P2, Q2)

        # --- Update UV history ---
        col = (it - 1) % Q_hist
        u_new, v_new = _update_history_compiled(x, x_next, x_next2)
        U[:, col] = u_new
        V[:, col] = v_new

        # --- QN extrapolation ---
        n_cols = min(it, Q_hist)
        U_sub = U[:, :n_cols]
        V_sub = V[:, :n_cols]

        # Linear system: (U^T @ (U - V)) alpha = U^T @ (x - x_next)
        LHS = U_sub.T @ (U_sub - V_sub)
        RHS = U_sub.T @ (x - x_next)

        try:
            # Solve the tiny (Q_hist x Q_hist) system
            alpha = torch.linalg.solve(LHS, RHS)
        except RuntimeError:
            # Fallback if the system is singular/ill-conditioned
            alpha = torch.linalg.lstsq(LHS, RHS).solution

        x_qn = _extrapolate_compiled(x_next, V_sub, alpha)

        # --- Unflatten and project ---
        _unflatten_PQ_gpu(x_qn, P, Q, M, K)
        _mapPQ_compiled(P, Q)

    del optimizer, U, V
    return P, Q
