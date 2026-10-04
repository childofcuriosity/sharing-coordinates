import itertools
import numpy as np
from weighted import response,moments,task_costs,clustering_costs
from src.routing_geometry import balanced_partitions

def fixture():
    rng=np.random.default_rng(938)
    a=rng.dirichlet(np.ones(3),6);b=rng.normal(size=(3,7))
    idx=np.asarray(list(balanced_partitions(6,3)))
    return rng,a,b,idx

def hard(g,groups):
    out=np.zeros_like(g)
    for group in groups:out[group]=g[group].sum(0)
    return out

def test_full_operator_empirical_identity():
    rng,a,b,idx=fixture();gs=rng.normal(size=(8,6,7));ys=np.array([response(a,b,g,.7,1.3) for g in gs])
    costs,_=task_costs(*moments(gs,ys),indices=idx,eta_b=.7)
    brute=[np.mean([np.sum((y-.7*hard(g,groups))**2) for g,y in zip(gs,ys)]) for groups in idx]
    np.testing.assert_allclose(costs,brute,atol=1e-11,rtol=1e-12)

def test_isotropic_recovers_router_clustering():
    _,a,b,idx=fixture();gs=np.eye(42).reshape(42,6,7);ys=np.array([response(a,b,g) for g in gs])
    costs,_=task_costs(*moments(gs,ys),indices=idx)
    ja=clustering_costs(a,np.eye(3),idx)
    np.testing.assert_allclose(costs-costs[0],2*7*2*(ja-ja[0])/42,atol=1e-12)

def test_batch_formula_all_subsets_and_mean_not_discarded():
    rng,a,b,idx=fixture();gs=rng.normal(size=(6,6,7))+2;ys=np.array([response(a,b,g) for g in gs])
    costs,_=task_costs(*moments(gs,ys,batch=3),indices=idx)
    batches=[gs[list(ids)].mean(0) for ids in itertools.combinations(range(6),3)]
    brute=[np.mean([np.sum((response(a,b,g)-hard(g,groups))**2) for g in batches]) for groups in idx]
    np.testing.assert_allclose(costs,brute,atol=1e-10,rtol=1e-12)
    assert np.max(np.abs(costs-task_costs(*moments(gs,ys),indices=idx)[0]))>1e-3

def test_activation_distortion_matches_direct_outputs():
    rng,a,b,idx=fixture();x=rng.normal(size=(6,11,7))
    v=np.einsum('kip,kjp->kij',np.einsum('rd,ltd->lrt',b,x),np.einsum('rd,ltd->lrt',b,x))/11
    costs=clustering_costs(a,v,idx);theta=a@b;brute=[]
    for groups in idx:
        total=0.
        for group in groups:
            center=theta[group].mean(0)
            for i in group:total+=np.mean((x[i]@(theta[i]-center))**2)
        brute.append(total)
    np.testing.assert_allclose(costs,brute,atol=1e-12)

def test_task_covariance_can_change_partition():
    # Deliberate low-contrast router and two opposite task-gradient directions:
    # mixing the signs cancels the hard response, unlike isotropic clustering.
    a=np.array([[.6,.4],[.55,.45],[.45,.55],[.4,.6]])
    b=np.eye(2);idx=np.asarray(list(balanced_partitions(4,2)))
    g=np.array([[1.,0.],[1.,0.],[-1.,0.],[-1.,0.]])
    gs=np.array([g,-g]);ys=np.array([response(a,b,x) for x in gs])
    costs,_=task_costs(*moments(gs,ys),indices=idx)
    assert np.argmin(costs)!=np.argmin(clustering_costs(a,np.eye(2),idx))
    assert costs[0]>min(costs)+1
