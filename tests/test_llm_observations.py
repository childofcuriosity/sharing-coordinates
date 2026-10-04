import math
import torch
from src.llm_observations import (designed_queries,compress_on_frame,observed_frame,
    score_factors,truth_for_scoring,response_error)
from src.trusted_recovery import compress_observations,residual_norm,spectral_estimate
from experiments.run_autograd_tomography import autodiff_response


def fixture():
    torch.manual_seed(6)
    a=torch.randn(8,4,dtype=torch.double).softmax(-1)
    b=torch.randn(4,51,dtype=torch.double)
    return a,b


def test_repeated_complement_queries_are_unit_and_recover_factors():
    a,b=fixture();theta=a@b
    g=designed_queries(theta,4,4)
    torch.testing.assert_close(g.flatten(1).norm(dim=1),torch.ones(4,dtype=torch.double))
    u,_=observed_frame(theta,4)
    gp=g-(g@u.T)@u
    torch.testing.assert_close(gp[0],gp[-1],rtol=1e-12,atol=1e-12)
    r=torch.stack([autodiff_response(a,b,x) for x in g])
    obs,condition=compress_on_frame(theta,g,r,u,0.)
    aa,c,_=spectral_estimate(obs)
    score=score_factors(aa,c,truth_for_scoring(a,b,u))
    assert score['orbit_relative_error']<1e-6
    # The PSD square root in spectral recovery acts on a covariance with an
    # exact null eigenvalue. Backend rounding there gives sqrt(epsilon)-scale
    # recovery error; use the same reconstruction tolerance as the factors.
    # Keep the exact response/compression identity independently much tighter.
    assert response_error(aa,c,obs,condition)<1e-6
    assert response_error(a,b@u.T,obs,condition)<1e-12


def test_compression_keeps_off_subspace_and_unexplained_response_errors():
    a,b=fixture();theta=a@b
    noisy=theta+.01*torch.randn_like(theta)
    u,_=observed_frame(noisy,4)
    g=designed_queries(noisy,4,4)
    r=torch.stack([autodiff_response(a,b,x) for x in g])+.001*torch.randn_like(g)
    obs,condition=compress_on_frame(noisy,g,r,u,.01)
    other,other_u=compress_observations(noisy,g,r,4,.01)
    torch.testing.assert_close(u,other_u)
    candidate_c=b@u.T
    torch.testing.assert_close(torch.tensor(residual_norm(a,candidate_c,obs)),
                               torch.tensor(residual_norm(a,candidate_c,other)))
    direct_r=torch.stack([autodiff_response(a,candidate_c@u,x) for x in g])
    expected=float((direct_r-r).norm()/r.norm())
    assert abs(response_error(a,candidate_c,obs,condition)-expected)<1e-10
    truth=truth_for_scoring(a,b,u)
    score=score_factors(a,candidate_c,truth)
    assert score['basis_relative_error']>0
    assert abs(score['basis_relative_error']-float((candidate_c@u-b).norm()/b.norm()))<1e-12


def test_insufficient_query_rank_is_not_hidden():
    a,b=fixture();theta=a@b;u,_=observed_frame(theta,4)
    g=designed_queries(theta,4,1)
    r=torch.stack([autodiff_response(a,b,x) for x in g])
    obs,condition=compress_on_frame(theta,g,r,u,0.)
    assert condition['local_min_rank']==1
    assert not condition['sufficient_rank']
