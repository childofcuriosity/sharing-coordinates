"""Finite query construction and exact compressed observation bookkeeping.

Estimators take observation records only. Truth projection is a separate
evaluation object and retains the error outside the observed row space.
"""
import itertools
import math
import torch

from src.response_tomography import _coordinate_complement_codes
from src.trusted_recovery import Observation


def observed_frame(theta, rank):
    _,singular,right=torch.linalg.svd(theta,full_matrices=False)
    return right[:rank],singular


def designed_queries(theta,rank,count):
    u,_=observed_frame(theta,rank)
    w=_coordinate_complement_codes(u,theta.shape[0],1e-10)
    return torch.stack([(w+u[p%rank][None])/math.sqrt(2*theta.shape[0])
                        for p in range(count)])


def normalized_queries(values):
    norm=values.flatten(1).norm(dim=1)
    if bool((norm<=0).any()) or not bool(torch.isfinite(norm).all()):
        raise ValueError('Zero/nonfinite task-gradient direction')
    return values/norm[:,None,None]


def perturb(value,scale,generator):
    if scale==0:
        return value.clone()
    noise=torch.randn(value.shape,dtype=value.dtype,device=value.device,generator=generator)
    return value+noise*(scale*value.norm()/noise.norm())


def compress_on_frame(theta,gradients,responses,u,relative_noise,singular=None):
    """Same exact least-squares compression as trusted_recovery, reused frame."""
    x=gradients@u.T
    gp=gradients-x@u
    y=responses@u.T
    rp=responses-y@u
    normal=torch.einsum('qld,qmd->lm',gp,gp)
    cross=torch.einsum('qld,qmd->lm',rp,gp)
    eig,vectors=torch.linalg.eigh(normal)
    if float(eig[0])<=1e-14*max(float(eig[-1]),1e-300):
        raise ValueError('Complement design numerically singular')
    root=(vectors*eig.sqrt())@vectors.T
    invroot=(vectors*eig.rsqrt())@vectors.T
    target=cross@invroot
    gram=target@invroot
    unexplained=rp-gram@gp
    theta_u=theta@u.T
    theta_norm=float(theta.norm());response_norm=float(responses.norm())
    theta_tail=float((theta-theta_u@u).square().sum())/theta_norm**2
    response_tail=float(unexplained.square().sum())/response_norm**2
    if singular is None:
        singular=torch.linalg.svdvals(theta)
    obs=Observation(theta_u.cpu(),root.cpu(),target.cpu(),x.cpu(),y.cpu(),
        theta_norm,response_norm,theta_tail+response_tail,
        float(gradients.square().sum()),relative_noise,singular.tolist(),eig.tolist())
    local_sv=torch.linalg.svdvals(x.transpose(0,1))
    ranks=(local_sv>1e-12*local_sv[:,:1]).sum(-1)
    condition=dict(complement_eigenvalues=eig.tolist(),local_singular_values=local_sv.tolist(),
        local_min_rank=int(ranks.min()),local_min_sigma=float(local_sv[:,-1].min()),
        sufficient_rank=int(ranks.min())==u.shape[0],
        theta_tail_squared_relative=theta_tail,response_tail_squared_relative=response_tail)
    return obs,condition


def truth_for_scoring(a,b,u):
    c=b@u.T
    tail=b-c@u
    return dict(a=a.cpu(),c=c.cpu(),basis_tail_squared=float(tail.square().sum()),
                router_norm=float(a.norm()),basis_norm=float(b.norm()))


def score_factors(ahat,chat,truth):
    best=None
    for perm in itertools.permutations(range(ahat.shape[1])):
        ids=list(perm)
        ae=float((ahat-truth['a'][:,ids]).norm())/truth['router_norm']
        be=math.sqrt(float((chat-truth['c'][ids]).square().sum())+
                     truth['basis_tail_squared'])/truth['basis_norm']
        row=dict(orbit_relative_error=math.hypot(ae,be),router_relative_error=ae,
                 basis_relative_error=be,permutation=ids)
        if best is None or row['orbit_relative_error']<best['orbit_relative_error']:
            best=row
    return best


def response_error(a,c,obs,condition):
    from src.trusted_recovery import response_coordinates
    shared=((a@a.T)@obs.complement_root-obs.complement_target).square().sum()
    local=(response_coordinates(a,c,obs.coordinates)-obs.response_coordinates).square().sum()
    return math.sqrt(float(shared+local)/obs.response_norm**2+
                     condition['response_tail_squared_relative'])


def cpu_state(obs):
    return obs.tensor_dict()
