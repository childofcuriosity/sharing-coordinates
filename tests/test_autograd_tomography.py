import torch

from experiments.run_autograd_tomography import autodiff_response, predict, reconstruct
from src.response_identifiability import apply_euclidean_response
from src.response_tomography import build_tomography_design, tomography_probe


def test_independent_autodiff_oracle_reconstruction_and_corruption():
    rng=torch.Generator().manual_seed(9071)
    a=torch.softmax(torch.randn(5,3,generator=rng,dtype=torch.float64),-1)
    b=torch.randn(3,12,generator=rng,dtype=torch.float64)
    design=build_tomography_design(a@b,3)
    eta_z,eta_b,tau=.7,1.3,.8
    observations=[]
    for i in range(3):
        g=tomography_probe(design,i)
        observed=autodiff_response(a,b,g,eta_z=eta_z,eta_b=eta_b,tau=tau)
        expected=apply_euclidean_response(a,b,g,eta_z=eta_z,eta_b=eta_b,temperature=tau)
        torch.testing.assert_close(observed,expected,atol=1e-12,rtol=1e-12)
        observations.append(observed)
    gram,local=reconstruct(observations,design,eta_b)
    g=torch.randn(5,12,generator=rng,dtype=torch.float64)
    measured=autodiff_response(a,b,g,eta_z=eta_z,eta_b=eta_b,tau=tau)
    torch.testing.assert_close(predict(g,gram,local,design,eta_b),measured,atol=1e-11,rtol=1e-11)
    damaged=[v.clone() for v in observations]
    damaged[0]+=0.1*torch.randn(5,12,generator=rng,dtype=torch.float64)
    wrong_gram,wrong_local=reconstruct(damaged,design,eta_b)
    assert float((predict(g,wrong_gram,wrong_local,design,eta_b)-measured).norm())>1e-3
