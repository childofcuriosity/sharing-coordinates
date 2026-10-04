import math
import torch
import pytest
from experiments.run_autograd_tomography import autodiff_response
from src.trusted_recovery import (compress_observations,model_vector,target_vector,
    residual_norm,simplex_tangent,fit_joint,local_diagnostics,spectral_estimate,project_spectral)


def test_split_and_budget_gate_fail_closed():
    from experiments.trusted_protocol import check_split
    with pytest.raises(ValueError,match='Unregistered'):
        check_split('synthetic','evaluation',100)
    with pytest.raises(ValueError,match='budget'):
        check_split('synthetic','evaluation',200,budget=81)


def test_calibration_ties_and_feasibility():
    from experiments.calibrate_trusted_recovery import select_threshold
    rows=[]
    for i in range(22):
        result=dict(status='ok',diagnostics=dict(feasibility=True,empirical_score=1 if i<20 else 2),
                    evaluation=dict(quality='good' if i<20 else 'bad'))
        rows.append(dict(key=dict(kind='synthetic'),methods=dict(joint_fit=result)))
    chosen=select_threshold(rows,'joint_fit','empirical_score')
    assert chosen['threshold']==1 and chosen['accepted']==20
    rows[0]['methods']['joint_fit']['diagnostics']['feasibility']=False
    assert select_threshold(rows,'joint_fit','empirical_score')['threshold'] is None


def instance(seed=82,noise=1e-5):
    rng=torch.Generator().manual_seed(seed)
    a=torch.randn(6,3,generator=rng,dtype=torch.double).softmax(-1)
    b=torch.randn(3,16,generator=rng,dtype=torch.double)
    g=torch.randn(8,6,16,generator=rng,dtype=torch.double)
    g=g/g.flatten(1).norm(dim=1)[:,None,None]
    r=torch.stack([autodiff_response(a,b,v) for v in g])
    theta=a@b
    nt=torch.randn(theta.shape,generator=rng,dtype=torch.double)
    nr=torch.randn(r.shape,generator=rng,dtype=torch.double)
    theta=theta+noise*theta.norm()*nt/nt.norm()
    r=r+noise*r.norm()*nr/nr.norm()
    obs,u=compress_observations(theta,g,r,3,noise)
    return a,b,g,r,theta,obs,u


def test_compression_preserves_full_objective_and_tangent_metric():
    a,b,g,r,theta,obs,u=instance()
    h=simplex_tangent(3)
    torch.testing.assert_close(h.T@h,torch.eye(2,dtype=h.dtype))
    torch.testing.assert_close(h.sum(0),torch.zeros(2,dtype=h.dtype),atol=1e-15,rtol=0)
    for c in (b@u.T,torch.eye(3,dtype=torch.double)):
        full=sum(float((autodiff_response(a,c@u,gg)-rr).square().sum()) for gg,rr in zip(g,r))/float(r.square().sum())
        full+=float((a@c@u-theta).square().sum())/float(theta.square().sum())
        assert abs(residual_norm(a,c,obs)**2-full)<1e-12


def test_joint_fit_decreases_observed_residual_without_truth_inputs():
    a,b,g,r,theta,obs,u=instance()
    ah,ch,_=spectral_estimate(obs)
    initial=project_spectral(ah,ch,obs)
    fit,candidates=fit_joint(obs,initial,starts=1,max_evaluations=30)
    assert fit['residual']<=residual_norm(*initial,obs)+1e-12
    assert fit['a'].min()>0
    torch.testing.assert_close(fit['a'].sum(-1),torch.ones(6,dtype=torch.double))
    assert len(candidates)==1


def test_local_bound_carries_scope_and_rejects_infeasible_candidate():
    a,b,g,r,theta,obs,u=instance()
    diagnostics=local_diagnostics(a,b@u.T,obs)
    assert diagnostics['no_global_truth_guarantee']
    assert diagnostics['curvature_upper_bound']>0
    bad=a.clone();bad[0,0]=-1
    failed=local_diagnostics(bad,b@u.T,obs)
    assert not failed['feasibility']
    assert not failed['local_feasible_component_pass']


def test_curvature_bound_dominates_measured_directional_hessian():
    a,b,g,r,theta,obs,u=instance(noise=0.)
    c=b@u.T
    h=simplex_tangent(3)
    da=local_diagnostics(a,c,obs)
    sa=float(a.norm());sc=float(c.norm())
    n=6*2+3*3
    def fun(v):
        return model_vector(a+sa*v[:12].reshape(6,2)@h.T,
                            c+sc*v[12:].reshape(3,3),obs)
    rng=torch.Generator().manual_seed(998)
    for _ in range(4):
        p=torch.randn(n,generator=rng,dtype=torch.double);p/=p.norm()
        q=torch.randn(n,generator=rng,dtype=torch.double);q/=q.norm()
        zero=torch.zeros(n,dtype=torch.double)
        inner=lambda x:torch.func.jvp(fun,(x,),(p,))[1]
        second=torch.func.jvp(inner,(zero,),(q,))[1]
        assert float(second.norm())<=da['curvature_upper_bound']
