import torch
from experiments.run_task_gradients import natural_reconstruct
from experiments.run_autograd_tomography import autodiff_response,analytic_components,predict
from src.response_tomography import build_tomography_design


def test_natural_full_rank_design_recovers_and_predicts():
    torch.manual_seed(542)
    a=torch.randn(6,3,dtype=torch.double).softmax(-1)
    b=torch.randn(3,17,dtype=torch.double)
    design=build_tomography_design(a@b,3)
    g=torch.randn(5,6,17,dtype=torch.double)
    r=torch.stack([autodiff_response(a,b,v) for v in g])
    q,t,conditions=natural_reconstruct(g,r,design.row_basis)
    tq,tt=analytic_components(a,b,design,1.)
    torch.testing.assert_close(q,tq,atol=1e-10,rtol=1e-10)
    torch.testing.assert_close(t,tt,atol=1e-10,rtol=1e-10)
    assert conditions['complement_numerical_rank']==6
    assert conditions['local_min_rank']==3
    heldout=torch.randn(6,17,dtype=torch.double)
    torch.testing.assert_close(predict(heldout,q,t,design),autodiff_response(a,b,heldout),atol=1e-10,rtol=1e-10)


def test_repeated_directions_do_not_pass_local_rank_condition():
    torch.manual_seed(543)
    a=torch.randn(6,3,dtype=torch.double).softmax(-1)
    b=torch.randn(3,17,dtype=torch.double)
    design=build_tomography_design(a@b,3)
    g=torch.randn(1,6,17,dtype=torch.double).expand(5,-1,-1)
    r=torch.stack([autodiff_response(a,b,v) for v in g])
    _,_,conditions=natural_reconstruct(g,r,design.row_basis)
    assert conditions['local_min_rank']==1
