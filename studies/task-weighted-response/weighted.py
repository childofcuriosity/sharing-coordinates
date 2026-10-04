"""Balanced task-response objectives, without constructing a full response matrix."""
import numpy as np
from common import PARENT
from src.routing_geometry import balanced_partitions
PARTS=list(balanced_partitions(12,4))
INDICES=np.asarray(PARTS,dtype=np.int64)

def response(a,b,g,eta_b=1.,gamma=1.):
    c=np.array([np.diag(row)-np.outer(row,row) for row in a])
    local=np.einsum('lij,lj->li',c@c,g@b.T)@b
    return eta_b*(a@a.T)@g+gamma*local

def moments(gs,ys,batch=1):
    n=len(gs)
    if not 1<=batch<=n or n<2:raise ValueError('Need n>=2 and 1<=batch<=n')
    s=sum(g@g.T for g in gs)/n
    t=sum(y@g.T for y,g in zip(ys,gs))/n
    c=sum(float((y*y).sum()) for y in ys)/n
    gm=np.mean(gs,axis=0);ym=np.mean(ys,axis=0)
    return batch_moments(s,t,c,gm@gm.T,ym@gm.T,float((ym*ym).sum()),n,batch)

def batch_moments(s,t,c,sm,tm,cm,n,batch):
    alpha=(n-batch)/(batch*(n-1))
    sb=alpha*s+(1-alpha)*sm
    tb=alpha*t+(1-alpha)*tm
    return sb,(tb+tb.T)/2,alpha*c+(1-alpha)*cm

def task_costs(s,t,c,indices=INDICES,eta_b=1.):
    m=indices.shape[-1]
    w=2*eta_b*t-eta_b**2*m*s
    scores=w[indices[:,:,:,None],indices[:,:,None,:]].sum((1,2,3))
    return c-scores,w

def clustering_costs(a,metric,indices=INDICES):
    x=a[indices];d=x-x.mean(axis=2,keepdims=True)
    if metric.ndim==2:return np.einsum('pgik,kl,pgil->p',d,metric,d)
    return np.einsum('pgik,pgikl,pgil->p',d,metric[indices],d)

def labels(index):
    out=np.empty(12,dtype=np.int64)
    for k,g in enumerate(PARTS[index]):out[list(g)]=k
    return out
