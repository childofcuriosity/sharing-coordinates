"""Strict evidence assembly; no selection or retuning of frozen estimators."""
import argparse
import collections
import csv
import itertools
import io
import json
from pathlib import Path
import numpy as np

from src.llm_shared import sha256
from experiments.llm_protocol import CONFIG, MODELS, ROOT, check_freeze, unit_path
from experiments.calibrate_llm_recovery import METHODS, SCORES, accepted


def statistics(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return dict(n=0,median=None,p90=None,max=None,min=None)
    if not np.isfinite(values).all():raise ValueError('Nonfinite summary input')
    return dict(n=len(values),median=float(np.median(values)),p90=float(np.quantile(values,.9)),
                max=float(values.max()),min=float(values.min()))


def aggregate(rows):
    result=dict(total=len(rows),status=dict(collections.Counter(r['status'] for r in rows)),
        errors=statistics([r['error'] for r in rows if r['error'] is not None]),
        heldout=statistics([r['heldout'] for r in rows if r['heldout'] is not None]),
        bad=sum(r['quality']=='bad' for r in rows),good=sum(r['quality']=='good' for r in rows),
        admissible=sum(r['admissible'] for r in rows),acceptance={})
    for score in SCORES:
        subset=[r for r in rows if r[score+'_accepted']]
        n=len(subset);bad=sum(r['quality']=='bad' for r in subset)
        result['acceptance'][score]=dict(accepted=n,bad=bad,coverage=n/len(rows) if rows else None,
            accepted_bad_fraction=bad/n if n else None,errors=statistics([r['error'] for r in subset]))
    return result


def groups(rows,keys):
    grouped=collections.defaultdict(list)
    for row in rows:grouped[tuple(row[k] for k in keys)].append(row)
    return [dict(zip(keys,key),summary=aggregate(value)) for key,value in grouped.items()]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--recovery-only',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();freeze=check_freeze()
    policy_path=ROOT/'research/LLM_ACCEPTANCE_POLICY.json'
    policy=json.loads(policy_path.read_text())
    assert policy['freeze_sha256']==freeze
    inputs={str(policy_path.relative_to(ROOT)):sha256(policy_path)}
    for path,digest in policy['calibration_inputs'].items():
        assert sha256(ROOT/path)==digest, path
    rows=[];units=[];decisions=[]
    def read(path):
        inputs[str(path.relative_to(ROOT))]=sha256(path)
        obj=json.loads(path.read_text())
        assert obj['freeze_sha256']==freeze,path
        return obj
    expected=set(itertools.product(CONFIG['noise'],['designed','gaussian','natural_general','natural_code','natural_math'],CONFIG['queries']))
    for model in MODELS:
        for seed in CONFIG['seeds']:
            out=unit_path(model,seed,'evaluation')
            training=read(out/'training.json');obs=read(out/'observations.json');recovery=read(out/'recovery.json')
            for obj in (training,obs,recovery):
                assert (obj['model'],obj['seed'],obj['split'])==(model,seed,'evaluation')
            assert all(training['gates'].values()) and all(obs['gates'].values())
            assert sha256(out/'checkpoint.pt')==training['checkpoint_sha256']
            assert sha256(out/'observations.pt')==obs['observations_sha256']==recovery['observations_sha256']
            assert len(training['training'])==CONFIG['steps'] and training['config']==CONFIG
            for domain,roles in training['data'].items():
                assert not set(roles['loss']['document_ids']) & set(roles['probe']['document_ids'])
                assert len(set(training['gradient_records'][domain]['block_indices']))==12
            signatures=[(r['noise'],r['family'],r['q']) for r in recovery['records']]
            assert len(signatures)==len(set(signatures))==len(expected) and set(signatures)==expected
            units.append(dict(model=model,seed=seed,dimensions=training['dimensions'],
                parameters=training['backbone_parameters'],trainable=training['trainable_parameters'],
                training_seconds=training['training_seconds'],training_total_seconds=training['seconds'],
                observation_seconds=obs['seconds'],recovery_seconds=recovery['seconds'],
                peak_gpu_bytes=max(training['peak_gpu_bytes'],obs['peak_gpu_bytes']),
                router_movement=training['router_mean_l1_movement'],basis_movement=training['basis_relative_movement'],
                router_min_entry=training['router_min_entry'],router_sigma_min=min(training['router_singular_values']),
                basis_sigma_min=min(training['basis_singular_values']),controls=obs['controls'],
                training_controls=training['controls'],finite_steps=obs['finite_steps'],
                baseline_nll={k:float(np.mean(v)) for k,v in training['baseline_nll'].items()},
                adapted_nll={k:float(np.mean(v)) for k,v in training['adapted_nll'].items()}))
            for case in recovery['records']:
                for method in METHODS:
                    result=case.get('methods',{}).get(method,dict(status=case['status']))
                    good=result['status']=='ok'
                    row=dict(model=model,seed=seed,method=method,family=case['family'],q=case['q'],noise=case['noise'],
                        status=result['status'],admissible=bool(good and result['diagnostics']['admissible']),
                        error=result['evaluation']['orbit_relative_error'] if good else None,
                        quality=result['evaluation']['quality'] if good else 'no_return',
                        heldout=result['diagnostics']['heldout_response_error'] if good else None,
                        local_rank=case.get('condition',{}).get('local_min_rank'),
                        local_sigma=case.get('condition',{}).get('local_min_sigma'),
                        case_seconds=case['seconds'])
                    for score in SCORES:
                        row[score]=result['diagnostics'][score] if good else None
                        row[score+'_accepted']=accepted(result,score,policy['policy'][method][score])
                    rows.append(row)
            if not args.recovery_only and model in CONFIG['decision_models']:
                decision=read(out/'decisions.json')
                assert decision['policy_sha256']==sha256(policy_path)
                assert decision['recovery_sha256']==sha256(out/'recovery.json')
                assert len(decision['recovery_batch_indices'])==CONFIG['decision_steps']
                assert decision['trainable_recovery_parameters']==CONFIG['rank']*training['dimensions'][1]
                native=decision['actions']['native_router']
                for name,action in decision['actions'].items():
                    if action['status']!='ok':
                        decisions.append(dict(model=model,seed=seed,action=name,domain='all',status=action['status']));continue
                    assert len(action['training_nll'])==CONFIG['decision_steps']
                    assert action['group_sizes']==[training['dimensions'][0]//CONFIG['rank']]*CONFIG['rank']
                    for domain in CONFIG['evaluation_domains']:
                        row=dict(model=model,seed=seed,action=name,domain=domain,status='ok',
                            partition_change=action['co_membership_difference_from_native'],
                            recovery_error=action.get('recovery_error'),
                            combined_acceptance=action.get('combined_acceptance'),
                            residual_acceptance=action.get('residual_acceptance'),
                            original_nll=float(np.mean(decision['original_nll'][domain])))
                        for timing in ('immediate_nll','after_recovery_nll'):
                            vals=np.asarray(action[timing][domain]);ref=np.asarray(native[timing][domain])
                            assert vals.shape==ref.shape==(16,)
                            row[timing]=float(vals.mean());row[timing+'_delta_native']=float((vals-ref).mean())
                        decisions.append(row)
    # Resample adaptation checkpoints within each fixed model. This is not a
    # population interval across arbitrary pretrained models or independent probes.
    bootstrap={};rng=np.random.default_rng(5300000)
    for method in METHODS:
        bootstrap[method]={}
        for score in SCORES:
            counts=np.zeros((len(MODELS),len(CONFIG['seeds']),3))
            for i,model in enumerate(MODELS):
                for j,seed in enumerate(CONFIG['seeds']):
                    subset=[r for r in rows if r['method']==method and r['model']==model and r['seed']==seed]
                    accepted_rows=[r for r in subset if r[score+'_accepted']]
                    counts[i,j]=[len(subset),len(accepted_rows),sum(r['quality']=='bad' for r in accepted_rows)]
            indices=rng.integers(0,3,size=(2000,len(MODELS),3))
            samples=counts[np.arange(len(MODELS))[None,:,None],indices].sum(axis=(1,2))
            ratios=samples[:,2]/np.maximum(samples[:,1],1)
            bootstrap[method][score]=dict(coverage_interval=np.quantile(samples[:,1]/samples[:,0],[.025,.975]).tolist(),
                accepted_bad_fraction_interval=np.quantile(ratios,[.025,.975]).tolist() if counts[:,:,1].sum() else None)
    summary=dict(format='llm-summary-v1',freeze_sha256=freeze,input_sha256=inputs,units=units,
        counts=dict(models=len(MODELS),checkpoints=len(units),recovery_cases=len(rows)//len(METHODS),method_outputs=len(rows)),
        by_method=groups(rows,['method']),by_model_method=groups(rows,['model','method']),
        by_query_noise_family=groups(rows,['method','q','noise','family']),
        by_model_query_noise=groups(rows,['model','method','q','noise']),
        by_model_family=groups(rows,['model','method','family']),bootstrap=bootstrap,
        bootstrap_scope='2000 stratified checkpoint resamples, three adapter seeds within each fixed pretrained model; descriptive only',
        worst_cases=sorted([r for r in rows if r['error'] is not None],key=lambda r:-r['error'])[:20],decisions=decisions)
    target=ROOT/'results/llm'/('recovery-summary.json' if args.recovery_only else 'summary.json')
    outputs={target:json.dumps(summary,indent=2,allow_nan=False)+'\n'}
    f=io.StringIO(newline='')
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    outputs[ROOT/'results/llm/recovery-cases.csv']=f.getvalue()
    if decisions:
        fields=sorted(set().union(*(r.keys() for r in decisions)))
        f=io.StringIO(newline='')
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(decisions)
        outputs[ROOT/'results/llm/decision-cases.csv']=f.getvalue()
    for path,payload in outputs.items():
        if args.check:
            if path.read_bytes()!=payload.encode():raise ValueError('Derived output differs: '+str(path))
        else:path.write_bytes(payload.encode())
    print(json.dumps(summary['counts']))
    for row in summary['by_method']:print(json.dumps(row))


if __name__=='__main__':main()
