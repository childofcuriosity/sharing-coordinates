import numpy as np
import torch
from experiments.train_routing_crossing import run
from experiments.train_llm_routing_trajectory import cluster,events,PARTS
from src.routing_geometry import boundary_example,flow_rhs,partition_cost,squared_distances


def test_autograd_sgd_enters_strict_disagreement_and_control_stays():
    a,b,_,_,target=boundary_example();z=np.log(a);dz,db=flow_rhs(z,b,target)
    z0=z-.1*dz;b0=b-.1*db
    trained=run(z0,b0,target,.02,torch.float64,'test')
    assert trained['initially_eligible'] and trained['sustained_crossing_start_step'] is not None
    assert trained['loss_nonincreasing']
    assert np.linalg.norm(np.asarray(trained['final']['b'])-b0)>0
    assert np.linalg.norm(np.asarray(trained['final']['z'])-z0)>0
    av=np.exp(z0-z0.max(1,keepdims=True));av/=av.sum(1,keepdims=True)
    stationary=run(z0,b0,av@b0,.02,torch.float64,'control')
    assert stationary['first_crossing'] is None
    assert np.allclose(stationary['final']['z'],z0,atol=1e-14)


def test_gram_enumeration_matches_direct_effective_pair_costs():
    rng=np.random.default_rng(72);a=rng.uniform(.1,1,(12,4));a/=a.sum(1,keepdims=True)
    b=rng.normal(size=(4,13));result=cluster(a,b@b.T)
    assert len(PARTS)==15400 and len(set(PARTS))==15400
    distance=squared_distances(a@b)
    direct=sorted((partition_cost(distance,p),p) for p in PARTS)
    assert result['partition']==direct[0][1]
    assert np.isclose(result['cost'],direct[0][0])
    assert np.isclose(result['gap'],direct[1][0]-direct[0][0])


def test_event_classifier_distinguishes_router_and_weight_boundary():
    p=((0,1,2),(3,4,5),(6,7,8),(9,10,11))
    q=((0,1,3),(2,4,5),(6,7,8),(9,10,11))
    def row(c,w,r):return dict(coefficient=dict(partition=c,score=4.,gap=.1),weight=dict(partition=w,cost=1.,gap=.1),router=dict(partition=r,cost=1.,gap=.1))
    assert events(row(p,p,p),row(q,p,p))==['coefficient_entry','router_boundary']
    assert events(row(p,p,p),row(p,q,p))==['coefficient_entry','router_entry']
    assert events(row(p,q,p),row(p,q,p))==[]
