"""Budget-matched partitions using router or effective-parameter geometry."""
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def balanced_assignment(scores):
    if scores.ndim!=2 or scores.shape[0]%scores.shape[1]:
        raise ValueError('Equal group sizes require L divisible by K')
    l,k=scores.shape
    costs=-np.repeat(scores.detach().cpu().double().numpy(),l//k,axis=1)
    rows,cols=linear_sum_assignment(costs)
    labels=np.empty(l,dtype=np.int64);labels[rows]=cols//(l//k)
    return torch.tensor(labels,dtype=torch.long)


def balanced_effective_partition(coordinates,k,iterations=50):
    x=coordinates.detach().cpu().double()
    selected=[0]
    for _ in range(k-1):
        distance=torch.cdist(x,x[selected]).square().min(dim=1).values
        distance[selected]=-1
        selected.append(int(distance.argmax()))
    centers=x[selected].clone();previous=None
    for _ in range(iterations):
        labels=balanced_assignment(-torch.cdist(x,centers).square())
        if previous is not None and torch.equal(previous,labels):break
        centers=torch.stack([x[labels==j].mean(0) for j in range(k)])
        previous=labels
    return labels


def partition_difference(first,second):
    return float(((first[:,None]==first[None,:])!=(second[:,None]==second[None,:])).double().mean())


def refit_bases(theta,labels,k):
    if any(int((labels==j).sum())==0 for j in range(k)):
        raise ValueError('Empty group changes parameter budget')
    return torch.stack([theta[labels.to(theta.device)==j].mean(0) for j in range(k)])
