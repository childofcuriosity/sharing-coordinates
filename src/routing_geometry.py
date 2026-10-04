"""Balanced decisions: coefficient assignment, geometry, and Euclidean response.

New exploratory analysis; original frozen LLM estimators are unchanged.
"""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def partition(labels):
    labels=np.asarray(labels)
    return tuple(sorted(tuple(map(int,np.flatnonzero(labels==k))) for k in np.unique(labels)))


def balanced_partitions(n,k):
    if n%k:raise ValueError('Equal capacities required')
    m=n//k
    def generate(remaining):
        if not remaining:
            yield ();return
        first=remaining[0]
        for rest in itertools.combinations(remaining[1:],m-1):
            group=(first,)+rest;chosen=set(group)
            for tail in generate(tuple(i for i in remaining if i not in chosen)):
                yield (group,)+tail
    yield from generate(tuple(range(n)))


def labels_from_partition(groups):
    labels=np.empty(sum(map(len,groups)),dtype=int)
    for k,group in enumerate(groups):labels[list(group)]=k
    return labels


def router_assignment(a):
    n,k=a.shape;m=n//k
    cost=-np.repeat(a,m,axis=1)
    rows,cols=linear_sum_assignment(cost)
    labels=np.empty(n,dtype=int);labels[rows]=cols//m
    score=float(a[np.arange(n),labels].sum())
    alternative=-np.inf
    # Every distinct labeled assignment omits at least one chosen row/class.
    for i,c in enumerate(labels):
        other=cost.copy();other[i,c*m:(c+1)*m]=np.inf
        rr,cc=linear_sum_assignment(other)
        alternative=max(alternative,float(a[rr,cc//m].sum()))
    return labels,score,score-alternative


def squared_distances(x):
    return ((x[:,None]-x[None,:])**2).sum(-1)


def partition_cost(distance,groups):
    return sum(sum(distance[i,j] for i,j in itertools.combinations(g,2))/len(g) for g in groups)


def hard_response_score(a,labels):
    """Partition-dependent term of full operator Hilbert--Schmidt error.

Multiply by eta_B**2 * D and add a partition-independent diagonal term.
"""
    same=np.asarray(labels)[:,None]==np.asarray(labels)[None,:]
    return float(np.square(a@a.T-same).sum())


def full_response(a,b,eta_b=1.,eta_z=1.,temperature=1.):
    n,k=a.shape;d=b.shape[1]
    result=eta_b*np.kron(a@a.T,np.eye(d))
    for i,row in enumerate(a):
        c=np.diag(row)-np.outer(row,row)
        result[i*d:(i+1)*d,i*d:(i+1)*d]+=eta_z/temperature**2*b.T@c@c@b
    return result


def flow_rhs(z,b,target,eta_b=1.,eta_z=1.,temperature=1.):
    logits=z/temperature;exp=np.exp(logits-logits.max(1,keepdims=True));a=exp/exp.sum(1,keepdims=True)
    g=a@b-target
    dz=np.empty_like(z)
    for i,row in enumerate(a):
        c=np.diag(row)-np.outer(row,row)
        dz[i]=-eta_z/temperature*c@b@g[i]
    return dz,-eta_b*a.T@g


def boundary_example():
    numerator=np.array([[40,24,28],[24,44,24],[32,28,32],[43,37,12],[24,20,48],[37,31,24]])
    a=numerator/92.;b=np.concatenate([np.eye(3),np.zeros((3,6))],axis=1)
    p=np.array([0,1,2,1,2,0]);q=np.array([0,1,2,0,2,1])
    difference=np.eye(3)[p]-np.eye(3)[q];g=np.zeros((6,9))
    for i,row in enumerate(a):
        c=np.diag(row)-np.outer(row,row);g[i]=b.T@c@c@difference[i]
    return a,b,p,q,a@b-g
