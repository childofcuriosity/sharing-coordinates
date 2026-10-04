import math
import torch
from experiments.run_autograd_tomography import autodiff_response


def test_projection_bridge_for_different_product_spaces():
    rng=torch.Generator().manual_seed(811)
    for _ in range(20):
        a=torch.randn(5,3,generator=rng,dtype=torch.double).softmax(-1)
        b=torch.randn(3,9,generator=rng,dtype=torch.double)
        ap=(a.log()+1e-6*torch.randn(a.shape,generator=rng,dtype=torch.double)).softmax(-1)
        bp=b+1e-6*torch.randn(b.shape,generator=rng,dtype=torch.double)
        alpha=min(float(a.min()),float(ap.min()))
        sa=min(float(torch.linalg.svdvals(a)[-1]),float(torch.linalg.svdvals(ap)[-1]))
        sb=min(float(torch.linalg.svdvals(b)[-1]),float(torch.linalg.svdvals(bp)[-1]))
        eps=float((a@b-ap@bp).norm())
        assert eps<min(alpha*sb/2,sa*sb/4)
        p=a@torch.linalg.pinv(a)
        at=p@ap
        bt=torch.linalg.pinv(at)@(a@b)
        torch.testing.assert_close(at.sum(-1),torch.ones(5,dtype=a.dtype),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(at@bt,a@b,atol=1e-12,rtol=1e-12)
        assert float((at-ap).norm())<=eps/sb+1e-12
        assert float((bt-bp).norm())<=2*eps/sa+1e-12
        assert at.min()>=alpha/2
        m=2*max(float(b.norm()),float(bp.norm()))
        lr=math.hypot(2*math.sqrt(5)+6*m*m,2*m)
        displacement=math.hypot(float((at-ap).norm()),float((bt-bp).norm()))
        g=torch.randn(5,9,generator=rng,dtype=a.dtype)
        difference=autodiff_response(at,bt,g)-autodiff_response(ap,bp,g)
        assert float(difference.norm())<=lr*displacement*float(g.norm())+1e-12
