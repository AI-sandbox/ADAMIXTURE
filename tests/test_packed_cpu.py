"""Equivalence of CPU 2-bit storage, including odd rows and shared-byte CV masks."""
from argparse import Namespace
import numpy as np
import pytest
from adamixture.src.utils_c import mask_entries_i32, restore_entries_i32, mask_entries_i64, restore_entries_i64
from adamixture.src import utils
from adamixture.src.utils_c.cython import sqp, tools
from adamixture.model.br_qn import optimize_original
from adamixture.model.em_adam import optimize_parameters
from adamixture.model.svd import RSVD
from adamixture.src.adamixture import initialize_em_cpu
from adamixture.src.projection import optimize_projection_cpu
from adamixture.src.supervised import init_p_supervised
from adamixture.src.cv import run_cross_validation


def matrices(M=37, N=31, K=3):
    rng = np.random.default_rng(2718)
    G = rng.integers(0, 4, (M, N), dtype=np.uint8)
    P = rng.uniform(.1, .9, (M, K))
    Q = rng.uniform(.1, .9, (N, K)); Q /= Q.sum(1, keepdims=True)
    return G, utils.ensure_packed(G, M), P, Q


@pytest.mark.parametrize('M,N', [(1, 1), (2, 13), (5, 17), (37, 31), (8, 257)])
def test_conversion_and_likelihood(M, N):
    G, packed, P, Q = matrices(M, N)
    np.testing.assert_array_equal(G, tools.unpack_block(packed, 0, M))
    assert packed.nbytes == ((M + 3) // 4) * N
    np.testing.assert_allclose(tools.loglikelihood(G, P, Q), tools.loglikelihood_packed(packed.ctypes.data, M, N, P, Q), rtol=1e-12)
    f = np.full(M, .3, dtype=np.float32)
    for start in range(M):
        a = np.empty((M-start,N),np.float32); b = np.empty_like(a)
        a[:] = np.where(G[start:] == 3, 0., G[start:].astype(np.float32) - 2*f[start:,None])
        tools.decompress_block(packed,b,f,start)
        np.testing.assert_array_equal(a,b)
        a *= f[start:,None]
        tools.decompress_block_scaled(packed,b,f,f,start)
        np.testing.assert_array_equal(a,b)


def test_both_sqp_updates_and_brqn():
    G, packed, P, Q = matrices(N=257)
    M,K = P.shape; N=Q.shape[0]
    basis = np.linalg.svd(np.ones((1,K)), full_matrices=True)[2].T.copy()
    h=np.empty((N,K,K));g=np.empty((N,K));q=np.empty_like(Q);p=np.empty_like(P)
    sqp.update_q_sqp(packed,Q,q,P,h,g,basis,M,N,K,16)
    sqp.update_p_sqp(packed,Q,P,p,M,N,K)
    assert np.isfinite(h).all() and np.isfinite(g).all()
    assert np.isfinite(p).all() and np.isfinite(q).all()
    np.testing.assert_allclose(q.sum(axis=1),1.,atol=1e-12)
    assert (p>0).all() and (p<1).all()
    a=optimize_original(G,P.copy(),Q.copy(),4,K,M,N,0.,3,5,verbose=False)
    b=optimize_original(packed,P.copy(),Q.copy(),4,K,M,N,0.,3,5,verbose=False)
    for aa,bb in zip(a,b): np.testing.assert_allclose(aa,bb,rtol=1e-10,atol=1e-10)


def test_em_svd_projection_and_supervised_initialization():
    G, packed, P, Q = matrices()
    M,K=P.shape;N=Q.shape[0]
    fa=utils.calculate_frequencies_cpu(G,M,N,7);fb=utils.calculate_frequencies_cpu(packed,M,N,7)
    np.testing.assert_array_equal(fa,fb)
    for a,b in zip(RSVD(G,N,M,fa,2,42,2,.1,7),RSVD(packed,N,M,fb,2,42,2,.1,7)):
        np.testing.assert_allclose(a,b,rtol=1e-6,atol=1e-6)
    for a,b in zip(initialize_em_cpu(G,42,M,N,K,2),initialize_em_cpu(packed,42,M,N,K,2)):
        np.testing.assert_allclose(a,b,rtol=1e-11,atol=1e-11)
    kwargs=dict(lr=.005,beta1=.8,beta2=.88,reg_adam=1e-8,max_iter=3,check=1,K=K,M=M,N=N,lr_decay=.5,min_lr=1e-4,patience=3,tol_adam=.1)
    for a,b in zip(optimize_parameters(G,P.copy(),Q.copy(),**kwargs),optimize_parameters(packed,P.copy(),Q.copy(),**kwargs)):
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(optimize_projection_cpu(G,P,7,M=M,N=N),optimize_projection_cpu(packed,P,7,M=M,N=N),rtol=1e-12,atol=1e-12)
    y=np.arange(N)%K+1
    np.testing.assert_allclose(init_p_supervised(G,y,K,M),init_p_supervised(packed,y,K,M))


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_cv_masks_shared_bytes_and_restores_on_exit(index_dtype):
    G,packed,P,Q=matrices()
    original=packed.copy();M,N=G.shape
    entries=np.array([0,N,2*N,3*N,4*N,N+1],dtype=index_dtype)
    mask,restore=(mask_entries_i32,restore_entries_i32) if index_dtype==np.int32 else (mask_entries_i64,restore_entries_i64)
    saved=np.empty(len(entries),np.uint8)
    mask(packed,entries,saved,N)
    expected=G.copy();expected.ravel()[entries]=3
    np.testing.assert_array_equal(tools.unpack_block(packed,0,M),expected)
    restore(packed,entries,saved,N)
    np.testing.assert_array_equal(packed,original)
    args=Namespace(cv=2,seed=42)
    a=run_cross_validation(args,G,N,M,P.shape[1],P,Q)
    b=run_cross_validation(args,packed,N,M,P.shape[1],P,Q)
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(packed,original)


@pytest.mark.parametrize('suffix', ['bed', 'bed.gz', 'bed.zst', 'vcf', 'vcf.gz', 'vcf.zst', 'pgen', 'bcf', 'bcf.gz', 'bcf.zst'])
def test_file_readers_match_uint8(suffix):
    from tests.config import DATA_DIR
    a,N,M=utils.read_data(DATA_DIR/f'demo_data.{suffix}',packed=False,chrom_mode='all',verbose=False)
    b,n,m=utils.read_data(DATA_DIR/f'demo_data.{suffix}',chrom_mode='all',verbose=False)
    assert (N,M)==(n,m)
    np.testing.assert_array_equal(a,tools.unpack_block(b,0,M))


@pytest.mark.parametrize('algorithm', ['brqn','adamem'])
@pytest.mark.parametrize('init', ['svd','random'])
def test_cli_complete_fit_with_packed_cpu(tmp_path, algorithm, init):
    import subprocess
    import sys
    from tests.config import DATA_DIR
    cmd=[sys.executable,'-c','from adamixture.entry import main; main()',
         '--data_path',str(DATA_DIR/'demo_data.bed'),'--save_dir',str(tmp_path),
         '--name','packed','-k','3','-t','1','--device','cpu',
         '--algorithm',algorithm,'--init',init,
         '--max_iter','3','--max_als','10','--als_starts','1','--cv','2']
    r=subprocess.run(cmd,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    assert 'CV index' in r.stdout,r.stdout+r.stderr
    q=np.loadtxt(tmp_path/'packed.3.Q');p=np.loadtxt(tmp_path/'packed.3.P')
    assert q.shape==(105,3) and p.shape==(8451,3)
    assert np.isfinite(q).all() and np.isfinite(p).all()
    np.testing.assert_allclose(q.sum(axis=1),1.,atol=2e-6)


@pytest.mark.parametrize('mode',['project','supervised-brqn','supervised-adamem'])
def test_projection_and_supervised_cli(tmp_path,mode):
    import subprocess
    import sys
    from tests.config import DATA_DIR
    common=['--data_path',str(DATA_DIR/'demo_data.bed'),'--save_dir',str(tmp_path),'--name','fit','--device','cpu']
    if mode=='project':
        module='adamixture.project_entry'
        p_path=tmp_path/'source.P';np.savetxt(p_path,np.full((8451,3),.5))
        extra=['--p_path',str(p_path)]
    else:
        module='adamixture.supervised_entry'
        labels=tmp_path/'labels.txt';labels.write_text('\n'.join(str(i%3) for i in range(105))+'\n')
        extra=['--labels',str(labels),'--algorithm',mode.split('-')[1],'--max_iter','2']
    r=subprocess.run([sys.executable,'-c',f'from {module} import main; main()',*common,*extra],capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    q=np.loadtxt(tmp_path/'fit.3.Q')
    assert q.shape==(105,3) and np.isfinite(q).all()


def test_public_api_reads_packed_by_default(monkeypatch):
    import adamixture
    seen=[]
    read=utils.read_data
    def observed(*args,**kwargs):
        result=read(*args,**kwargs)
        seen.append((kwargs.get("packed"),result[0].shape,result[2]))
        return result
    monkeypatch.setattr(utils,"read_data",observed)
    G,_,_,_=matrices()
    adamixture.run(data=G,k=3,max_iter=2,max_als=3,als_starts=1,verbose=False)
    assert seen[0][0]=="cpu"
    assert seen[0][1][0]==(seen[0][2]+3)//4
