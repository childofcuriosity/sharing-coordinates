"""Observation-only recovery, followed by separately labelled truth scoring."""
import argparse
import itertools
import json
import math
import time
import torch

from src.llm_shared import sha256
from src.llm_observations import score_factors,response_error
from src.trusted_recovery import (Observation,spectral_estimate,project_spectral,
    fit_joint,local_diagnostics,residual_norm)
from experiments.llm_protocol import CONFIG,MODELS,check_freeze,validate_unit,unit_path


def disagreement(a,c,aa,cc):
    return min(math.hypot(float((a-aa[:,list(p)]).norm()/a.norm()),
               float((c-cc[list(p)]).norm()/c.norm()))
               for p in itertools.permutations(range(a.shape[1])))


def estimate(obs,hold,condition,held_condition,seed):
    """No ground-truth factors, errors, labels or model-family tuning input."""
    outcomes={};initial=None
    try:
        a,c,details=spectral_estimate(obs)
        outcomes['spectral']=dict(a=a,c=c,solver=details)
        initial=project_spectral(a,c,obs)
        outcomes['projected_spectral']=dict(a=initial[0],c=initial[1],solver={})
    except (RuntimeError,ValueError) as exc:
        for method in ('spectral','projected_spectral'):
            outcomes[method]=dict(status='failed',error=str(exc))
    if not condition['sufficient_rank']:
        outcomes['joint_fit']=dict(status='not_run_insufficient_query_rank')
    else:
        try:
            best,candidates=fit_joint(obs,initial,seed,CONFIG['starts'],CONFIG['max_evaluations'])
            details={k:v for k,v in best.items() if k not in ('a','c')}
            details['starts']=[{k:v for k,v in item.items() if k not in ('a','c')} for item in candidates]
            allowance=2*math.sqrt(2)*obs.relative_noise/max(1-obs.relative_noise,1e-12)+1e-10
            competitive=[r for r in candidates if r['residual']<=best['residual']+allowance]
            details['competitive_start_disagreement']=max(disagreement(best['a'],best['c'],r['a'],r['c']) for r in competitive)
            outcomes['joint_fit']=dict(a=best['a'],c=best['c'],solver=details)
        except (RuntimeError,ValueError) as exc:
            outcomes['joint_fit']=dict(status='failed',error=str(exc))
    for method,row in outcomes.items():
        if 'a' not in row:continue
        try:
            a,c=row['a'],row['c']
            diagnostic=local_diagnostics(a,c,obs)
            held=response_error(a,c,hold,held_condition)
            noise=diagnostic['noise_budget']+diagnostic['numerical_noise_floor']
            score=(diagnostic['residual']+held+2*noise)/max(diagnostic['tangent_sigma_min'],1e-300)
            diagnostic.update(heldout_response_error=held,
                residual_only_score=math.hypot(diagnostic['residual'],held),
                condition_only_score=1/max(diagnostic['tangent_sigma_min'],1e-300),
                empirical_score=max(score,diagnostic['subspace_ratio'],
                                    row['solver'].get('competitive_start_disagreement',0.)),
                admissible=diagnostic['feasibility'] and condition['sufficient_rank'])
            if any(isinstance(v,float) and not math.isfinite(v) for v in diagnostic.values()):
                raise ValueError('Nonfinite diagnostics')
            row.update(status='ok',diagnostics=diagnostic)
        except (RuntimeError,ValueError,OverflowError) as exc:
            row.pop('a',None);row.pop('c',None)
            row.update(status='failed',error=str(exc))
    return outcomes


def score(outcomes,truth):
    for row in outcomes.values():
        if row.get('status')!='ok':continue
        a,c=row.pop('a'),row.pop('c')
        evaluation=score_factors(a,c,truth)
        error=evaluation['orbit_relative_error']
        evaluation['quality']='good' if error<=CONFIG['good_error'] else ('bad' if error>CONFIG['bad_error'] else 'intermediate')
        row.update(estimate=dict(a=a.tolist(),c=c.tolist()),evaluation=evaluation)
    return outcomes


def run(args):
    freeze=check_freeze();validate_unit(args.model,args.seed,args.split)
    out=unit_path(args.model,args.seed,args.split)
    target=out/'recovery.json'
    if target.exists():raise ValueError('Refusing recovery overwrite')
    metadata=json.loads((out/'observations.json').read_text())
    if metadata['freeze_sha256']!=freeze or not all(metadata['gates'].values()):
        raise ValueError('Observation provenance/gates invalid')
    if sha256(out/'observations.pt')!=metadata['observations_sha256']:
        raise ValueError('Observation bytes changed')
    torch.set_num_threads(1)
    data=torch.load(out/'observations.pt',weights_only=True)
    start=time.monotonic();records=[]
    progress=out/'recovery-progress.jsonl'
    if progress.exists():raise ValueError('Partial recovery exists; inspect before retry')
    with progress.open('x') as log:
        for i,case in enumerate(data['cases']):
            tick=time.monotonic()
            row={k:case[k] for k in ('noise','family','q','status')}
            if case['status']!='ok':
                row['error']=case['error']
            else:
                obs=Observation(**case['observation']);hold=Observation(**case['heldout'])
                outcomes=estimate(obs,hold,case['condition'],case['heldout_condition'],5100000+args.seed)
                row.update(methods=score(outcomes,case['truth']),condition=case['condition'])
            row['seconds']=time.monotonic()-tick
            records.append(row)
            log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
            print(args.model,args.seed,i+1,len(data['cases']),round(row['seconds'],2),flush=True)
    result=dict(format='llm-recovery-v1',model=args.model,seed=args.seed,split=args.split,
        freeze_sha256=freeze,observations_sha256=metadata['observations_sha256'],records=records,
        seconds=time.monotonic()-start)
    with target.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=list(MODELS))
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--split',required=True,choices=['calibration','evaluation'])
    run(p.parse_args())
