import numpy as np
import pytest

torch = pytest.importorskip('torch')

from adamixture.model import br_qn_gpu as brqn
from adamixture.src import utils
from adamixture.src.compat import Device
from adamixture.src.utils_c import pack_genotypes


@pytest.fixture(params=['mps', 'cuda'])
def gpu_device(request):
    name = request.param
    if name == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('MPS unavailable')
    if name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    device = torch.device(name)
    utils.load_extensions(Device(name))
    return device


def tensors(device):
    rng = np.random.default_rng(9)
    M, N, K = 31, 8, 2
    P = rng.uniform(.1, .9, (M, K))
    Q = rng.dirichlet([2., 2.], N)
    G = rng.binomial(2, P @ Q.T).astype(np.uint8)
    G[2, 1] = 3
    packed = np.zeros(((M + 3) // 4, N), dtype=np.uint8)
    pack_genotypes(G.ctypes.data, packed.ctypes.data, M, N, packed.shape[0])
    dtype = utils.get_dtype(device)
    return (torch.from_numpy(packed), torch.tensor(P, dtype=dtype, device=device),
            torch.tensor(Q, dtype=dtype, device=device), M, N, K)


def test_gpu_qn_cannot_discard_improving_sqp_steps(gpu_device, monkeypatch):
    G, P, Q, M, N, K = tensors(gpu_device)
    initial_ll = utils.loglikelihood_cpu_chunked(G, P, Q, M, 16)
    p0, q0 = P.clone(), Q.clone()

    def poor_candidate(p, q, **kwargs):
        p.copy_(p0)
        q.copy_(q0)

    monkeypatch.setattr(brqn, '_mapPQ_gpu', poor_candidate)
    p, q = brqn.optimize_original_gpu(G, P, Q, 1, K, M, N, 0., 3, 2, gpu_device, 16, 128)
    assert utils.loglikelihood_cpu_chunked(G, p, q, M, 16) > initial_ll + 1.


@pytest.mark.parametrize('accept_at', [2, None])
def test_gpu_qn_backtracks_up_to_four_candidates(gpu_device, monkeypatch, accept_at):
    G, P, Q, M, N, K = tensors(gpu_device)
    evaluated = []
    candidates = []
    project = brqn._mapPQ_gpu

    def record_projection(p, q, **kwargs):
        before = (p.clone(), q.clone())
        project(p, q, **kwargs)
        candidates.append((before, (p.clone(), q.clone())))

    def score(g, p, q, *args):
        evaluated.append((p.clone(), q.clone()))
        if len(evaluated) == 1:
            return 10.
        return 11. if len(evaluated) - 1 == accept_at else 9.

    monkeypatch.setattr(brqn, '_mapPQ_gpu', record_projection)
    monkeypatch.setattr(utils, 'get_logl_calculator', lambda device: score)
    monkeypatch.setattr(torch.linalg, 'solve', lambda lhs, rhs: torch.zeros_like(rhs))
    p, q = brqn.optimize_original_gpu(G, P, Q, 1, K, M, N, 0., 3, 2, gpu_device, 16, 128)
    assert len(candidates) == (accept_at or 4)
    for previous, current in zip(candidates, candidates[1:]):
        for after, before, base in zip(previous[1], current[0], evaluated[0]):
            torch.testing.assert_close(before, (after + base) * .5)
    expected = evaluated[-1] if accept_at else evaluated[0]
    torch.testing.assert_close(p, expected[0])
    torch.testing.assert_close(q, expected[1])
