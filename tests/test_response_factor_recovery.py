import itertools

import pytest
import torch

from experiments.run_autograd_tomography import autodiff_response, reconstruct
from src.response_factor_recovery import recover_factors
from src.response_tomography import build_tomography_design, tomography_probe


@pytest.mark.parametrize('k', [2, 3, 4])
def test_recover_factors_from_observations_only(k):
    generator = torch.Generator().manual_seed(510000+k)
    a = torch.randn(k+2, k, generator=generator, dtype=torch.float64).softmax(-1)
    b = torch.randn(k, 2*k+3, generator=generator, dtype=torch.float64)
    theta = a @ b
    design = build_tomography_design(theta, k)
    responses = [autodiff_response(a, b, tomography_probe(design, i),
                                  eta_z=.7, eta_b=1.3, tau=.8) for i in range(k)]
    gram, local = reconstruct(responses, design, eta_b=1.3)
    ahat, bhat, diagnostics = recover_factors(theta, design.row_basis, gram, local,
                                             eta_z=.7, tau=.8)
    errors = []
    for permutation in itertools.permutations(range(k)):
        index = list(permutation)
        errors.append(max(float((ahat-a[:, index]).norm()/a.norm()),
                          float((bhat-b[index]).norm()/b.norm())))
    assert min(errors) < 1e-6
    assert diagnostics['product_relative_residual'] < 1e-6
    assert diagnostics['router_row_sum_max_error'] < 1e-6
    with pytest.raises(ValueError, match='spectrum'):
        recover_factors(theta, design.row_basis, gram, local, eta_z=.7, tau=.8,
                        weights=torch.zeros(k+2, dtype=torch.float64))
