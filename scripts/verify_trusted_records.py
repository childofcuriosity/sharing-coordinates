"""Recompute residuals and truth scores from saved estimates/observations."""
import json
import math
from pathlib import Path
import torch
from experiments.trusted_protocol import ROOT,METHODS,expected_paths
from experiments.run_trusted_recovery import relative_pair_distance
from src.trusted_recovery import Observation,residual_norm

def main():
    torch.set_num_threads(2)
    checked=0;failures=0;largest=0.
    for split in ['calibration','evaluation']:
        for path in expected_paths(split):
            data=json.loads(path.read_text())
            obs=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
            for row,record in zip(data['records'],obs['records']):
                if row['status']!='ok':
                    failures+=len(METHODS);continue
                if set(row['methods'])!=set(METHODS): raise ValueError('Missing method')
                for method,r in row['methods'].items():
                    if r['status']!='ok': failures+=1;continue
                    a=torch.tensor(r['estimate']['a'],dtype=torch.double)
                    c=torch.tensor(r['estimate']['c'],dtype=torch.double)
                    truth=record['truth']
                    score=relative_pair_distance(a,c,truth['a'],truth['c'],
                        truth['a_norm'],truth['b_norm'],truth['b_outside_squared'])
                    for key in ['orbit_relative_error','router_relative_error','basis_relative_error']:
                        if not math.isclose(score[key],r['evaluation'][key],rel_tol=1e-11,abs_tol=1e-12):
                            raise ValueError('Stored truth error differs: '+str(row['key']))
                    for key,which in [('residual','fit'),('heldout_residual','holdout')]:
                        value=residual_norm(a,c,Observation(**record[which]))
                        diff=abs(value-r['diagnostics'][key]);largest=max(largest,diff)
                        if not math.isclose(value,r['diagnostics'][key],rel_tol=1e-10,abs_tol=1e-11):
                            raise ValueError('Stored residual differs: '+str(row['key']))
                    if method=='joint_fit':
                        candidates=r['solver']['starts']
                        if len(candidates)!=3 or any(v['nfev']>80 for v in candidates):
                            raise ValueError('Wrong solver budget')
                        if not math.isclose(min(v['residual'] for v in candidates),r['diagnostics']['residual'],rel_tol=1e-10,abs_tol=1e-11):
                            raise ValueError('Selected start is not minimum residual')
                    checked+=1
    print(json.dumps(dict(checked_estimates=checked,recorded_failures=failures,max_residual_recompute_difference=largest)))
if __name__=='__main__': main()
