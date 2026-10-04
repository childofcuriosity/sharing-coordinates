import torch
from torch import nn
from src.llm_shared import SharedBank
from src.llm_shared import domain_documents
from experiments.run_autograd_tomography import autodiff_response
from src.response_identifiability import apply_euclidean_response


def test_injected_module_is_direct_product_and_preserves_unselected_channels():
    torch.manual_seed(3)
    bank=SharedBank(8,11,13,rank=4,channels=5,seed=41).double()
    hidden=torch.randn(2,7,11,dtype=torch.double)
    output=torch.randn(2,7,13,dtype=torch.double)
    # inject explicitly computes in FP32 for the backbone, so test direct
    # representational equality at its implemented dtype, not double precision.
    bank=bank.float();hidden=hidden.float();output=output.float()
    theta=bank.prepare()
    actual=bank.inject(2,hidden,output)
    expected=output.clone()
    expected[...,bank.output_ids]+=hidden@theta[2].reshape(5,11).T
    torch.testing.assert_close(actual,expected)
    untouched=[i for i in range(13) if i not in bank.output_ids.tolist()]
    torch.testing.assert_close(actual[...,untouched],output[...,untouched],rtol=0,atol=0)
    # PyTorch's recursive Module.apply must remain usable.
    bank.apply(lambda m:None)


def test_effective_gradient_chain_matches_parameter_backward():
    torch.manual_seed(8)
    bank=SharedBank(8,11,13,rank=4,channels=5,seed=41)
    inputs=torch.randn(8,3,11)
    baseline=torch.randn(8,3,13)
    theta=bank.prepare()
    loss=sum(bank.inject(i,inputs[i],baseline[i]).sin().square().mean() for i in range(8))
    dz,db,g=torch.autograd.grad(loss,(bank.logits,bank.bases,theta))
    z=bank.logits.detach().clone().requires_grad_()
    b=bank.bases.detach().clone().requires_grad_()
    ez,eb=torch.autograd.grad((z.softmax(-1)@b*g).sum(),(z,b))
    torch.testing.assert_close(dz,ez)
    torch.testing.assert_close(db,eb)


def test_autodiff_response_matches_known_law_and_permutation():
    torch.manual_seed(22)
    a=torch.randn(8,4,dtype=torch.double).softmax(-1)
    b=torch.randn(4,31,dtype=torch.double)
    g=torch.randn(8,31,dtype=torch.double)
    measured=autodiff_response(a,b,g)
    analytic=apply_euclidean_response(a,b,g,temperature=1.,eta_z=1.,eta_b=1.)
    torch.testing.assert_close(measured,analytic,atol=1e-12,rtol=1e-12)
    p=[2,0,3,1]
    torch.testing.assert_close(autodiff_response(a[:,p],b[p],g),measured,atol=1e-12,rtol=1e-12)


def test_wikitext_roles_use_disjoint_articles_not_one_giant_document():
    docs=domain_documents('general','test')
    assert len(docs)>20
    probe_ids={i for i,_ in docs[::2]}
    loss_ids={i for i,_ in docs[1::2]}
    assert probe_ids.isdisjoint(loss_ids)
    assert len(probe_ids)==len(docs[::2])
