"""Observation-only noisy recovery and explicitly local credibility diagnostics.

This module never accepts ground-truth factors. Estimated-subspace recovery and
local certificates are distinct from a global true-factor guarantee.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from scipy.optimize import least_squares

from src.response_factor_recovery import recover_factors


@dataclass
class Observation:
    theta: torch.Tensor
    complement_root: torch.Tensor
    complement_target: torch.Tensor
    coordinates: torch.Tensor
    response_coordinates: torch.Tensor
    theta_norm: float
    response_norm: float
    constant_squared_residual: float
    gradient_squared_norm: float
    relative_noise: float
    product_singular_values: list
    complement_eigenvalues: list

    @property
    def depth(self):
        return self.theta.shape[0]

    @property
    def rank(self):
        return self.theta.shape[1]

    def tensor_dict(self):
        return dict(self.__dict__)


def compress_observations(theta, gradients, responses, rank, relative_noise):
    """Exact least-squares compression for factors in the estimated row space.

    Full orthogonal residuals are retained. Reject numerically singular
    complement Gram matrices rather than quietly dropping measured directions.
    """
    _, singular, right = torch.linalg.svd(theta, full_matrices=False)
    u = right[:rank]
    theta_u = theta @ u.T
    x = gradients @ u.T
    gp = gradients - x @ u
    y = responses @ u.T
    rp = responses - y @ u
    normal = torch.einsum('qld,qmd->lm', gp, gp)
    cross = torch.einsum('qld,qmd->lm', rp, gp)
    eig, vectors = torch.linalg.eigh(normal)
    if float(eig[0]) <= 1e-14 * max(float(eig[-1]), 1e-300):
        raise ValueError('Complement design is numerically singular')
    root = (vectors * eig.sqrt()) @ vectors.T
    inverse_root = (vectors * eig.rsqrt()) @ vectors.T
    target = cross @ inverse_root
    q_ls = cross @ torch.linalg.inv(normal)
    unexplained = rp - q_ls @ gp
    theta_norm = float(theta.norm())
    response_norm = float(responses.norm())
    constant = float((theta-theta_u@u).square().sum())/theta_norm**2
    constant += float(unexplained.square().sum())/response_norm**2
    obs = Observation(theta_u, root, target, x, y, theta_norm, response_norm,
                      constant, float(gradients.square().sum()), relative_noise,
                      singular.tolist(), eig.tolist())
    return obs, u


def response_coordinates(a, c, coordinates):
    q = a @ a.T
    covariance = torch.diag_embed(a) - a.unsqueeze(-1)*a.unsqueeze(-2)
    local = c.T @ (covariance @ covariance) @ c
    return q @ coordinates + torch.einsum('lik,plk->pli', local, coordinates)


def model_vector(a, c, obs):
    return torch.cat(((a@c/obs.theta_norm).flatten(),
                      ((a@a.T)@obs.complement_root/obs.response_norm).flatten(),
                      (response_coordinates(a,c,obs.coordinates)/obs.response_norm).flatten()))


def target_vector(obs):
    return torch.cat(((obs.theta/obs.theta_norm).flatten(),
                      (obs.complement_target/obs.response_norm).flatten(),
                      (obs.response_coordinates/obs.response_norm).flatten()))


def residual_norm(a, c, obs):
    return math.sqrt(max(0.,float((model_vector(a,c,obs)-target_vector(obs)).square().sum())
                         + obs.constant_squared_residual))


def simplex_tangent(k, dtype=torch.float64, device='cpu'):
    # Helmert contrasts: orthonormal columns, exactly orthogonal to all-ones.
    h = torch.zeros(k,k-1,dtype=dtype,device=device)
    for j in range(k-1):
        scale = math.sqrt((j+1)*(j+2))
        h[:j+1,j] = 1/scale
        h[j+1,j] = -(j+1)/scale
    return h


def spectral_estimate(obs):
    gram = obs.complement_target @ torch.linalg.inv(obs.complement_root)
    remainder = obs.response_coordinates - gram @ obs.coordinates
    local = torch.stack([remainder[:,i,:].T @ torch.linalg.pinv(
        obs.coordinates[:,i,:].T,rtol=1e-12) for i in range(obs.depth)])
    eye = torch.eye(obs.rank,dtype=obs.theta.dtype,device=obs.theta.device)
    a,c,details = recover_factors(obs.theta,eye,gram,local)
    return a,c,details


def project_spectral(a,c,obs):
    a = a.clamp_min(1e-8)
    a = a/a.sum(-1,keepdim=True)
    c = torch.linalg.lstsq(a,obs.theta,rcond=1e-12).solution
    return a,c


def fit_joint(obs, initial, seed=3200000, starts=3, max_evaluations=80):
    """Simplex-constrained fit; fixed starts and observed-residual selection."""
    l,k = obs.theta.shape
    if obs.theta.device.type!='cpu' or obs.theta.dtype!=torch.float64:
        raise ValueError('Small compressed solver requires CPU float64')
    h = simplex_tangent(k)
    rng = torch.Generator().manual_seed(seed)
    candidates=[]
    for index in range(starts):
        if index==0 and initial is not None:
            a,c = initial
        elif index==1 and initial is not None:
            a=(initial[0].clamp_min(1e-8).log()+.1*torch.randn(l,k,generator=rng,dtype=torch.double)).softmax(-1)
            c=torch.linalg.lstsq(a,obs.theta,rcond=1e-12).solution
        else:
            a=torch.randn(l,k,generator=rng,dtype=torch.double).softmax(-1)
            c=torch.linalg.lstsq(a,obs.theta,rcond=1e-12).solution
        cscale=max(float(c.norm()),1e-12)
        start=torch.cat(((a.clamp_min(1e-12).log()@h).flatten(),(c/cscale).flatten()))
        def decode(v):
            logits=v[:l*(k-1)].reshape(l,k-1)@h.T
            return logits.softmax(-1),v[l*(k-1):].reshape(k,k)*cscale
        target=target_vector(obs)
        def fun_t(v):
            aa,cc=decode(v)
            return model_vector(aa,cc,obs)-target
        jac_t=torch.func.jacfwd(fun_t)
        def fun_np(v):
            return fun_t(torch.from_numpy(v)).detach().numpy()
        def jac_np(v):
            return jac_t(torch.from_numpy(v)).detach().numpy()
        result=least_squares(fun_np,start.numpy(),jac=jac_np,method='trf',
                             max_nfev=max_evaluations,ftol=1e-12,xtol=1e-12,gtol=1e-12,
                             x_scale='jac')
        aa,cc=decode(torch.from_numpy(result.x))
        residual=residual_norm(aa,cc,obs)
        candidates.append(dict(a=aa.detach(),c=cc.detach(),residual=residual,
            nfev=int(result.nfev),njev=int(result.njev or 0),status=int(result.status),
            optimality=float(result.optimality),start_index=index))
    best=min(candidates,key=lambda r:(r['residual'],r['start_index']))
    return best,candidates


def local_diagnostics(a,c,obs, roundoff_floor=1e-10):
    """Numerical evaluation of a conditional, estimated-subspace local bound.

    The real-arithmetic theorem requires verified singular-value/curvature
    bounds. Floating-point SVD here is not interval-certified. Global membership
    of true factors in the local feasible component is NOT established.
    """
    l,k=a.shape
    h=simplex_tangent(k,dtype=a.dtype,device=a.device)
    scale_a=max(float(a.norm()),1e-12); scale_c=max(float(c.norm()),1e-12)
    n=l*(k-1)+c.numel()
    def forward(delta):
        da=delta[:l*(k-1)].reshape(l,k-1)@h.T*scale_a
        dc=delta[l*(k-1):].reshape_as(c)*scale_c
        return model_vector(a+da,c+dc,obs)
    zero=torch.zeros(n,dtype=a.dtype,device=a.device)
    jac=torch.func.jacfwd(forward)(zero)
    singular=torch.linalg.svdvals(jac)
    mu=float(singular[-1])
    residual=residual_norm(a,c,obs)
    noise=math.sqrt(2)*obs.relative_noise/max(1-obs.relative_noise,1e-12)
    numerical_noise=noise+roundoff_floor
    min_entry=float(a.min())
    row_error=float((a.sum(-1)-1).abs().max())
    feasible=min_entry>0 and row_error<1e-8
    radius_feasible=max(0.,min_entry/scale_a*.5)
    covariance=torch.diag_embed(a)-a.unsqueeze(-1)*a.unsqueeze(-2)
    cmax=float(torch.linalg.matrix_norm(covariance,ord=2).max())
    def curvature(radius):
        mb=scale_c*(1+radius)
        c_bound=min(1.,cmax+3*scale_a*radius)
        mat=torch.tensor([[(18+4*c_bound)*mb*mb*scale_a*scale_a,
                           12*c_bound*mb*scale_a*scale_c],
                          [12*c_bound*mb*scale_a*scale_c,2*c_bound*c_bound*scale_c*scale_c]],
                         dtype=a.dtype,device=a.device)
        hlocal=float(torch.linalg.eigvalsh(mat)[-1])
        response=(2*scale_a*scale_a+hlocal)*math.sqrt(obs.gradient_squared_norm)/obs.response_norm
        product=2*scale_a*scale_c/obs.theta_norm
        return math.hypot(product,response)
    upper=curvature(radius_feasible)
    radius=min(radius_feasible,mu/max(upper,1e-300))
    upper=curvature(radius)
    denominator=mu-upper*radius/2
    distance_bound=(residual+noise)/max(denominator,1e-300)
    boundary_separation=mu*radius-upper*radius*radius/2
    local_component=(feasible and residual<=noise and radius>0 and
                     residual+noise<boundary_separation)
    theta_noise=obs.relative_noise/max(1-obs.relative_noise,1e-12)*obs.theta_norm
    observed_sigma=obs.product_singular_values[k-1]
    subspace_ratio=theta_noise/max(observed_sigma-theta_noise,1e-300)
    linear_score=(residual+numerical_noise)/max(mu,1e-300)
    return dict(residual=residual,noise_budget=noise,numerical_noise_floor=roundoff_floor,
        feasibility=feasible,router_min_entry=min_entry,router_row_sum_error=row_error,
        router_sigma_min=float(torch.linalg.svdvals(a)[-1]),basis_sigma_min=float(torch.linalg.svdvals(c)[-1]),
        tangent_sigma_min=mu,tangent_sigma_max=float(singular[0]),
        linearized_score=linear_score,subspace_ratio=subspace_ratio,
        empirical_score=max(linear_score,subspace_ratio),
        local_radius=radius,curvature_upper_bound=upper,local_distance_bound=distance_bound,
        boundary_separation=boundary_separation,local_feasible_component_pass=local_component,
        certificate_scope='conditional estimated-subspace local component; numerical non-interval evaluation',
        no_global_truth_guarantee=True)
