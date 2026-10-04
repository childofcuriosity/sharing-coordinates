"""Run observation-only methods, then separately score truth and store diagnostics."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time
import torch
from experiments.trusted_protocol import check_split,sha,ROOT,FREEZE,POLICY

from experiments.train_enhancement import source_hashes
from src.trusted_recovery import (Observation,spectral_estimate,project_spectral,
    fit_joint,local_diagnostics,residual_norm)


def relative_pair_distance(a,c,other_a,other_c,a_norm,b_norm,outside_squared=0.):
    choices=[]
    for permutation in itertools.permutations(range(a.shape[1])):
        ids=list(permutation)
        ae=float((a-other_a[:,ids]).norm())/a_norm
        be=(float((c-other_c[ids]).square().sum())+outside_squared)**.5/b_norm
        choices.append((math_hypot(ae,be),ae,be,list(permutation)))
    error,ae,be,permutation=min(choices)
    return dict(orbit_relative_error=error,router_relative_error=ae,basis_relative_error=be,permutation=permutation)


def math_hypot(a,b):
    return (a*a+b*b)**.5


def estimate(obs,hold,seed,starts=3,budget=80):
    """No truth input. All method selection uses training observations only."""
    outcomes={}
    initial=None
    try:
        a,c,details=spectral_estimate(obs)
        outcomes['spectral']=dict(a=a,c=c,solver=details)
        initial=project_spectral(a,c,obs)
        outcomes['projected_spectral']=dict(a=initial[0],c=initial[1],solver={})
    except (RuntimeError,ValueError) as error:
        outcomes['spectral']=dict(status='failed',error=str(error))
        outcomes['projected_spectral']=dict(status='failed',error=str(error))
    try:
        best,candidates=fit_joint(obs,initial,seed=seed,starts=starts,max_evaluations=budget)
        details={k:v for k,v in best.items() if k not in ('a','c')}
        details['starts']=[{k:v for k,v in row.items() if k not in ('a','c')} for row in candidates]
        noise=2**.5*obs.relative_noise/max(1-obs.relative_noise,1e-12)+1e-10
        competitive=[row for row in candidates if row['residual']<=best['residual']+2*noise]
        disagreement=max(relative_pair_distance(row['a'],row['c'],best['a'],best['c'],
            float(best['a'].norm()),float(best['c'].norm()))['orbit_relative_error'] for row in competitive)
        details['competitive_start_disagreement']=disagreement
        outcomes['joint_fit']=dict(a=best['a'],c=best['c'],solver=details)
    except (RuntimeError,ValueError) as error:
        outcomes['joint_fit']=dict(status='failed',error=str(error))
    for name,result in outcomes.items():
        if result.get('status')=='failed': continue
        a,c=result['a'],result['c']
        try:
            diagnostics=local_diagnostics(a,c,obs)
            heldout=residual_norm(a,c,hold)
            noise=diagnostics['noise_budget']+diagnostics['numerical_noise_floor']
            combined=(diagnostics['residual']+heldout+2*noise)/max(diagnostics['tangent_sigma_min'],1e-300)
            disagreement=result['solver'].get('competitive_start_disagreement',0.)
            diagnostics.update(heldout_residual=heldout,
                residual_only_score=math_hypot(diagnostics['residual'],heldout),
                condition_only_score=1/max(diagnostics['tangent_sigma_min'],1e-300),
                empirical_score=max(combined,diagnostics['subspace_ratio'],disagreement))
            result.update(status='ok',diagnostics=diagnostics)
        except (RuntimeError,ValueError) as error:
            result.update(status='failed',error=str(error))
    return outcomes


def score_truth(outcomes,truth):
    """Only this isolated evaluation step reads hidden factors."""
    for result in outcomes.values():
        if result.get('status')!='ok':
            result.pop('a',None);result.pop('c',None)
            continue
        a,c=result.pop('a'),result.pop('c')
        result['estimate']={'a':a.tolist(),'c':c.tolist()}
        result['evaluation']=relative_pair_distance(a,c,truth['a'],truth['c'],
            truth['a_norm'],truth['b_norm'],truth['b_outside_squared'])
        error=result['evaluation']['orbit_relative_error']
        result['evaluation']['quality']='good' if error<=.05 else ('bad' if error>.10 else 'intermediate')
    return outcomes


def run(path,starts=3,budget=80,limit=None):
    torch.set_num_threads(2)
    data=torch.load(path,map_location='cpu',weights_only=False)
    check_split(data['kind'],data['split'],data['seed'],pilot=data['pilot'],starts=starts,budget=budget,limit=limit)
    if data['split']!='development':
        for source,digest in data['source_sha256'].items():
            if sha(ROOT/source)!=digest: raise ValueError('Observation source mismatch: '+source)
    records=[]
    started=time.monotonic()
    for index,record in enumerate(data['records']):
        if limit is not None and index>=limit: break
        if record['status']!='ok':
            records.append(dict(key=record['key'],status='observation_failed',error=record['error']))
            continue
        obs=Observation(**record['fit']);hold=Observation(**record['holdout'])
        result=estimate(obs,hold,seed=3200000+data['seed'],starts=starts,budget=budget)
        scored=score_truth(result,record['truth'])
        records.append(dict(key=record['key'],status='ok',methods=scored))
        print(json.dumps(dict(index=index,key=record['key'],errors={k:v.get('evaluation',{}).get('orbit_relative_error') for k,v in scored.items()},
                         seconds=time.monotonic()-started)),flush=True)
    sources=source_hashes(['experiments/run_trusted_recovery.py','src/trusted_recovery.py',
        'experiments/train_enhancement.py','src/response_factor_recovery.py','research/TRUSTED_RECOVERY_PROTOCOL.md'])
    return dict(format='trusted-recovery-v1',kind=data['kind'],split=data['split'],seed=data['seed'],
        pilot=data['pilot'],records=records,starts=starts,max_evaluations=budget,
        observation_file=str(path),observation_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        freeze_sha256=sha(FREEZE) if data['split']!='development' else None,
        policy_sha256=sha(POLICY) if data['split']=='evaluation' else None,
        source_sha256=sources,torch_version=torch.__version__,seconds=time.monotonic()-started)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--starts',type=int,default=3)
    p.add_argument('--budget',type=int,default=80)
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    if args.output.exists(): raise SystemExit('Refusing recovery overwrite')
    result=run(args.observations,args.starts,args.budget,args.limit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')
