import itertools
import numpy as np
from scipy.integrate import solve_ivp
from src.routing_geometry import (balanced_partitions,partition,partition_cost,
    labels_from_partition,squared_distances,router_assignment,full_response,
    hard_response_score,boundary_example,flow_rhs)


def test_two_basis_global_objectives_agree():
    rng=np.random.default_rng(8)
    for _ in range(12):
        t=rng.uniform(.05,.95,8);a=np.stack([t,1-t],1);b=rng.normal(size=(2,11))
        groups=list(balanced_partitions(8,2));d=squared_distances(a@b)
        optimum=min(groups,key=lambda p:partition_cost(d,p))
        assert partition(router_assignment(a)[0])==optimum


def test_isotropic_bases_do_not_equate_assignment_and_free_clustering():
    n=np.array([[8,11,4],[8,5,10],[10,9,4],[5,11,7],[6,10,7],[11,5,7]])
    a=n/23.;labels,score,gap=router_assignment(a)
    p=partition(labels);groups=list(balanced_partitions(6,3));d=squared_distances(n)
    costs=sorted((partition_cost(d,g),g) for g in groups)
    assert p==((0,3),(1,4),(2,5))
    assert costs[0]==(14.,((0,2),(1,5),(3,4)))
    assert costs[1][0]==31 and partition_cost(d,p)==41
    assert abs(score-60/23)<1e-14 and abs(gap-1/23)<1e-14


def test_full_response_distortion_reduces_to_router_kmeans():
    rng=np.random.default_rng(16);a=rng.uniform(.1,1,(6,3));a/=a.sum(1,keepdims=True)
    b=rng.normal(size=(3,9));eta_b=.7;eta_z=1.3;temperature=.9
    r=full_response(a,b,eta_b,eta_z,temperature);constants=[]
    for groups in balanced_partitions(6,3):
        labels=labels_from_partition(groups);h=np.eye(3)[labels]
        rh=eta_b*np.kron(h@h.T,np.eye(9))
        full=float(np.square(r-rh).sum())
        constants.append(full-eta_b**2*9*hard_response_score(a,labels))
    assert np.ptp(constants)<1e-11
    g=list(balanced_partitions(6,3));d=squared_distances(a)
    p1=min(g,key=lambda p:partition_cost(d,p))
    p2=min(g,key=lambda p:hard_response_score(a,labels_from_partition(p)))
    assert p1==p2


def test_quadratic_gradient_flow_crosses_router_boundary_only():
    a,b,p,q,target=boundary_example();z=np.log(a);y=np.concatenate([z.ravel(),b.ravel()])
    def rhs(t,y):
        dz,db=flow_rhs(y[:18].reshape(6,3),y[18:].reshape(3,9),target)
        return np.concatenate([dz.ravel(),db.ravel()])
    partitions=list(balanced_partitions(6,3))
    for end,expected in [(-.01,p),(.01,q)]:
        sol=solve_ivp(rhs,[0,end],y,rtol=1e-11,atol=1e-13)
        zz=sol.y[:18,-1].reshape(6,3);aa=np.exp(zz-zz.max(1,keepdims=True));aa/=aa.sum(1,keepdims=True)
        bb=sol.y[18:,-1].reshape(3,9)
        assert partition(router_assignment(aa)[0])==partition(expected)
        d=squared_distances(aa@bb)
        assert min(partitions,key=lambda x:partition_cost(d,x))==partition(p)


def test_same_weights_have_different_unique_response_optima():
    a=np.array([[8,11,4],[8,5,10],[10,9,4],[5,11,7],[6,10,7],[11,5,7]])/23.
    b=np.concatenate([np.eye(3),np.zeros((3,6))],axis=1)
    m=np.array([[2,1,7],[4,4,2],[5,2,3]])/10.
    transformed=a@m;other_b=np.linalg.solve(m,b)
    assert np.allclose(transformed@other_b,a@b,atol=1e-15)
    assert transformed.min()>0 and np.linalg.matrix_rank(transformed)==3
    groups=list(balanced_partitions(6,3))
    original=sorted((partition_cost(squared_distances(a),g),g) for g in groups)
    other=sorted((partition_cost(squared_distances(transformed),g),g) for g in groups)
    assert original[0][1]!=other[0][1]
    assert np.isclose(other[0][0],200/52900)
    assert np.isclose(other[1][0]-other[0][0],12/52900)
    for aa,bb,expected in [(a,b,original[0][1]),(transformed,other_b,other[0][1])]:
        r=full_response(aa,bb)
        def error(g):
            h=np.eye(3)[labels_from_partition(g)]
            return np.square(r-np.kron(h@h.T,np.eye(9))).sum()
        assert min(groups,key=error)==expected


def test_weight_response_tradeoff_exact_costs():
    a=np.array([[8,11,4],[8,5,10],[10,9,4],[5,11,7],[6,10,7],[11,5,7]])/23.
    b=np.concatenate([np.diag([4.,1.,1.]),np.zeros((3,6))],axis=1)
    groups=list(balanced_partitions(6,3));da=squared_distances(a);dt=squared_distances(a@b)
    dynamic=min(groups,key=lambda g:partition_cost(da,g))
    weight=min(groups,key=lambda g:partition_cost(dt,g))
    assert dynamic!=weight
    assert np.isclose(partition_cost(dt,dynamic),119/529)
    assert np.isclose(partition_cost(dt,weight),65/529)
    r=full_response(a,b)
    def error(g):
        h=np.eye(3)[labels_from_partition(g)]
        return np.square(r-np.kron(h@h.T,np.eye(9))).sum()
    assert np.isclose(error(weight)-error(dynamic),1296/529)


def test_boundary_derivative_is_strict_and_matches_chain_rule():
    a,b,p,q,target=boundary_example();e=np.eye(3)[p]-np.eye(3)[q]
    dz,db=flow_rhs(np.log(a),b,target)
    derivative=0.;negative_squared_norm=0.
    for row,delta,dzi in zip(a,e,dz):
        c=np.diag(row)-np.outer(row,row)
        derivative+=delta@c@dzi
        negative_squared_norm-=np.linalg.norm(b.T@c@c@delta)**2
    assert derivative<-.1
    assert np.isclose(derivative,negative_squared_norm,atol=1e-14)
    # Chain-rule factor derivative must equal the full effective response.
    adot=np.stack([(np.diag(row)-np.outer(row,row))@dzi for row,dzi in zip(a,dz)])
    theta_dot=adot@b+a@db
    expected=-full_response(a,b)@(a@b-target).ravel()
    assert np.allclose(theta_dot.ravel(),expected,atol=1e-14)
