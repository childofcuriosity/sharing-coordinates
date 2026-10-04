import torch
from torch.nn import functional as F
from src.language import LanguageConfig,build_language_model
from src.response_identifiability import pack_basis_matrix,basis_linear_modules
from src.enhancement import effective_forward,same_size_gauge,balanced_effective_partition


def test_actual_loss_forward_and_chain_rule_match():
    torch.manual_seed(14)
    model=build_language_model(LanguageConfig(depth=4,num_bases=2,dimension=8,
        hidden_dimension=16,num_heads=2,sequence_length=8,router_init='random')).double()
    p=pack_basis_matrix(model)
    a=model.router.probabilities()
    theta=(a.detach()@p.bases).requires_grad_()
    x=torch.randint(256,(2,8));y=torch.randint(256,(2,8))
    actual=model(x)
    effective=effective_forward(model,theta,p.layout,x)
    torch.testing.assert_close(effective,actual,atol=1e-12,rtol=1e-12)
    g,=torch.autograd.grad(F.cross_entropy(effective.flatten(0,1),y.flatten()),theta)
    parameters=[model.router.logits]+[t for _,m in basis_linear_modules(model) for t in (m.weight,m.bias) if t is not None]
    true=torch.autograd.grad(F.cross_entropy(actual.flatten(0,1),y.flatten()),parameters)
    b=p.bases.detach().requires_grad_()
    dz,db=torch.autograd.grad((a@b*g).sum(),(model.router.logits,b))
    torch.testing.assert_close(dz,true[0],atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(db,torch.cat([t.reshape(2,-1) for t in true[1:]],1),atol=1e-12,rtol=1e-10)


def test_fixed_size_baseline_is_gauge_independent_and_preserves_sizes():
    theta=torch.tensor([[0.,0.],[0.,.1],[3.,3.],[3.,3.1]],dtype=torch.double)
    labels,details=balanced_effective_partition(theta,[2,2],restarts=2,iterations=5)
    assert sorted(torch.bincount(labels).tolist())==[2,2]
    assert labels[0]==labels[1] and labels[2]==labels[3] and labels[0]!=labels[2]
    assert abs(details['sse']-.01)<1e-10


def test_gauge_search_retains_failure_and_validity():
    a=torch.full((4,2),.5,dtype=torch.double)
    result=same_size_gauge(a,5,'signed_local',trials=50)
    assert result['status']=='search_failed' and result['counts']['attempted']==50
    rng=torch.Generator().manual_seed(5)
    a=torch.randn(12,4,generator=rng,dtype=torch.double).softmax(-1)
    result=same_size_gauge(a,7,'signed_local',trials=500)
    if result['selected']:
        m=torch.tensor(result['selected']['matrix'],dtype=a.dtype)
        torch.testing.assert_close(m.sum(-1),torch.ones(4,dtype=a.dtype))
        assert (a@m).min()>0 and torch.linalg.cond(m)<=30
        assert sorted(torch.bincount((a@m).argmax(-1),minlength=4).tolist())==result['sorted_sizes']
