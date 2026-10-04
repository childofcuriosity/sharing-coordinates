"""Small full-coordinate Taylor checks; truth used only for theorem checking."""
import json
from pathlib import Path
import torch
from experiments.build_trusted_observations import synthetic
from experiments.run_autograd_tomography import autodiff_response
from experiments.train_enhancement import source_hashes
from src.trusted_recovery import Observation,local_diagnostics,simplex_tangent

def main():
    torch.set_num_threads(2)
    rows=[]
    for seed in range(4):
        for regime in ['interior','rank_01','rank_001','boundary_001']:
            a,b,g,_=synthetic(seed,regime);g=g[:4];theta=a@b
            response=torch.stack([autodiff_response(a,b,v) for v in g])
            obs=Observation(theta,torch.zeros(6,6,dtype=torch.double),
                torch.zeros(6,6,dtype=torch.double),g,response,float(theta.norm()),
                float(response.norm()),0.,float(g.square().sum()),1e-10,
                torch.linalg.svdvals(theta).tolist(),[])
            rng=torch.Generator().manual_seed(3500000+seed)
            da=torch.randn(6,2,generator=rng,dtype=torch.double)@simplex_tangent(3).T
            db=torch.randn(b.shape,generator=rng,dtype=torch.double)
            aa=a+1e-12*da;bb=b+1e-12*db
            d=local_diagnostics(aa,bb,obs)
            truth_distance=((float((aa-a).norm())/float(aa.norm()))**2+
                (float((bb-b).norm())/float(bb.norm()))**2)**.5
            inside=truth_distance<=d['local_radius']
            bound_valid=(not inside or truth_distance<=d['local_distance_bound'])
            if not bound_valid: raise RuntimeError('Full-coordinate local bound violated')
            rows.append(dict(seed=seed,regime=regime,truth_distance=truth_distance,
                truth_inside_radius=inside,bound_holds=bound_valid,diagnostics=d))
    path=Path('results/trusted_recovery/full_coordinate_checks.json')
    data=dict(format='trusted-full-coordinate-check-v1',purpose='Theorem sanity check, not recovery benchmark.',
        rows=rows,source_sha256=source_hashes(['experiments/check_trusted_full_coordinates.py',
            'src/trusted_recovery.py','experiments/build_trusted_observations.py',
            'experiments/run_autograd_tomography.py']))
    with path.open('x') as f: json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
    print(dict(cases=len(rows),inside=sum(r['truth_inside_radius'] for r in rows),
               passes=sum(r['diagnostics']['local_feasible_component_pass'] for r in rows)))
if __name__=='__main__': main()
