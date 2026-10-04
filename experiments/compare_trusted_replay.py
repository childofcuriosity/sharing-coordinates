"""Compare independent CPU solver replay without assuming bitwise equality."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments.trusted_protocol import ROOT,METHODS,SCORES,POLICY,sha,accepted,expected_paths
from experiments.summarize_trusted_recovery import measures

def build():
    policy=json.loads(POLICY.read_text());primary=[];replay=[];inputs={}
    mismatches={m:dict(status=0,quality=0,acceptance=0,max_abs_error_difference=0.,local_pass=0) for m in METHODS}
    versions=set()
    for p in expected_paths('evaluation'):
        q=ROOT/'results/trusted_recovery/clean_cpu'/p.name
        a=json.loads(p.read_text());b=json.loads(q.read_text())
        if a['observation_sha256']!=b['observation_sha256'] or a['source_sha256']!=b['source_sha256']:
            raise ValueError('Replay did not use the same observations/sources')
        if a['policy_sha256']!=b['policy_sha256'] or b['policy_sha256']!=sha(POLICY):
            raise ValueError('Replay changed acceptance policy')
        if [r['key'] for r in a['records']]!=[r['key'] for r in b['records']]:
            raise ValueError('Replay keys differ')
        versions.add(b['torch_version'])
        inputs[str(q.relative_to(ROOT))]=sha(q)
        primary.extend(a['records']);replay.extend(b['records'])
        for ar,br in zip(a['records'],b['records']):
            for m in METHODS:
                x=ar['methods'][m];y=br['methods'][m];d=mismatches[m]
                d['status']+=x['status']!=y['status']
                t=policy['rules'][m]['empirical_score']['threshold']
                d['acceptance']+=accepted(x,'empirical_score',t)!=accepted(y,'empirical_score',t)
                if x['status']==y['status']=='ok':
                    d['quality']+=x['evaluation']['quality']!=y['evaluation']['quality']
                    d['local_pass']+=x['diagnostics']['local_feasible_component_pass']!=y['diagnostics']['local_feasible_component_pass']
                    d['max_abs_error_difference']=max(d['max_abs_error_difference'],
                        abs(x['evaluation']['orbit_relative_error']-y['evaluation']['orbit_relative_error']))
    groups={}
    for kind in ['synthetic','language']:
        groups[kind]={}
        for m in METHODS:
            t=policy['rules'][m]['empirical_score']['threshold']
            groups[kind][m]={name:measures([r for r in rows if r['key']['kind']==kind],m,'empirical_score',t)
                for name,rows in [('primary',primary),('clean_cpu',replay)]}
    return dict(format='trusted-replay-comparison-v1',cases=len(primary),torch_versions=sorted(versions),
        interpretation='Same frozen observations and source. Optimization is not promised bitwise reproducible; all disagreements retained.',
        policy_sha256=sha(POLICY),replay_sha256=inputs,mismatches=mismatches,groups=groups)

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    data=build();text=json.dumps(data,indent=2,allow_nan=False)+'\n'
    path=ROOT/'research/TRUSTED_REPLAY_COMPARISON.json'
    if args.check:
        if path.read_text()!=text: raise ValueError('Replay comparison mismatch')
    else: path.write_text(text)
    print(json.dumps(data['mismatches'],indent=2))
if __name__=='__main__': main()
