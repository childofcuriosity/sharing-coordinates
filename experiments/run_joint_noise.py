"""Prospective synthetic joint product/response noise and conditioning audit."""
import argparse
import json
from pathlib import Path
import torch
from experiments.train_enhancement import source_hashes
from experiments.run_task_gradients import orbit_error
from experiments.run_autograd_tomography import autodiff_response,reconstruct
from src.response_tomography import build_tomography_design,tomography_probe
from src.response_factor_recovery import recover_factors


def run():
    torch.set_num_threads(2)
    records=[]
    for seed in range(10):
        rng=torch.Generator().manual_seed(780000+seed)
        base=torch.randn(6,3,generator=rng,dtype=torch.double).softmax(-1)
        b=torch.randn(3,16,generator=rng,dtype=torch.double)
        onehot=torch.nn.functional.one_hot(torch.arange(6)%3,3).double()
        for regime in ('interior','rank','boundary'):
            for level in ((1.,) if regime=='interior' else (1.,.1,.01,.001)):
                a=base if regime=='interior' else level*base+(1-level)*(1/3 if regime=='rank' else onehot)
                theta=a@b
                # Common random directions across noise scales in each cell.
                tn=torch.randn(theta.shape,generator=rng,dtype=torch.double)
                tn=tn/tn.norm()*theta.norm()
                rn=[torch.randn(theta.shape,generator=rng,dtype=torch.double) for _ in range(3)]
                for noise in (0.,1e-8,1e-6,1e-4,1e-2):
                    measured=theta+noise*tn
                    left,singular,right=torch.linalg.svd(measured,full_matrices=False)
                    estimate=(left[:,:3]*singular[:3])@right[:3]
                    row=dict(seed=seed,regime=regime,level=level,relative_noise=noise,
                        router_min_entry=float(a.min()),router_sigma_min=float(torch.linalg.svdvals(a)[-1]),
                        product_noise_norm=float((measured-theta).norm()),rank_truncation_residual=float((measured-estimate).norm()))
                    try:
                        design=build_tomography_design(estimate,3)
                        responses=[];errors=[]
                        for j in range(3):
                            response=autodiff_response(a,b,tomography_probe(design,j))
                            perturb=noise*rn[j]/rn[j].norm()*response.norm()
                            responses.append(response+perturb);errors.append(float(perturb.norm()))
                        q,t=reconstruct(responses,design)
                        ah,bh,diagnostics=recover_factors(estimate,design.row_basis,q,t)
                        row.update(status='ok',**orbit_error(ah,bh,a,b),diagnostics=diagnostics,response_noise_norms=errors)
                    except (ValueError,RuntimeError) as error:
                        row.update(status='failed',error=str(error))
                    records.append(row)
    sources=source_hashes(['experiments/run_joint_noise.py','experiments/run_task_gradients.py',
        'experiments/train_enhancement.py','experiments/run_autograd_tomography.py',
        'src/response_tomography.py','src/response_identifiability.py','src/response_factor_recovery.py',
        'research/JOINT_NOISE_PROTOCOL.md'])
    return dict(format='joint-noise-v1',records=records,source_sha256=sources,torch_version=torch.__version__)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise SystemExit('Refusing overwrite')
    result=run();a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('records',len(result['records']))
