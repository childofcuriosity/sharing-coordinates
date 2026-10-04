"""Freeze acceptance thresholds from declared calibration units only."""
import json
from pathlib import Path
from src.llm_shared import sha256
from experiments.llm_protocol import CONFIG,check_freeze,unit_path,ROOT

SCORES=['residual_only_score','condition_only_score','empirical_score']
METHODS=['spectral','projected_spectral','joint_fit']


def choose_threshold(rows,score):
    eligible=[r for r in rows if r.get('status')=='ok' and r['diagnostics']['admissible']]
    eligible.sort(key=lambda r:r['diagnostics'][score])
    best=None;bad=0;count=0
    for i,row in enumerate(eligible):
        count+=1;bad+=row['evaluation']['quality']=='bad'
        value=row['diagnostics'][score]
        if i+1<len(eligible) and eligible[i+1]['diagnostics'][score]==value:continue
        if count>=CONFIG['minimum_calibration_accepted'] and bad/count<=CONFIG['acceptance_bad_target']:
            best=dict(threshold=value,accepted=count,bad=bad,eligible=len(eligible),total=len(rows))
    return best or dict(threshold=None,accepted=0,bad=0,eligible=len(eligible),total=len(rows),reason='no_feasible_threshold')


def accepted(row,score,policy):
    threshold=policy['threshold']
    return bool(threshold is not None and row.get('status')=='ok'
        and row['diagnostics']['admissible'] and row['diagnostics'][score]<=threshold)


def main():
    freeze=check_freeze();all_rows=[];inputs={}
    for model in CONFIG['calibration_models']:
        for seed in CONFIG['calibration_seeds']:
            path=unit_path(model,seed,'calibration')/'recovery.json'
            data=json.loads(path.read_text())
            if data['freeze_sha256']!=freeze or data['split']!='calibration':raise ValueError('Bad calibration provenance')
            inputs[str(path.relative_to(ROOT))]=sha256(path)
            for row in data['records']:
                if row['family'] not in ('designed','gaussian','natural_general'):
                    raise ValueError('Held-out domain leaked into calibration')
                all_rows.append(row)
    policies={}
    for method in METHODS:
        rows=[r.get('methods',{}).get(method,dict(status='observation_failed')) for r in all_rows]
        policies[method]={name:choose_threshold(rows,name) for name in SCORES}
    output=dict(format='llm-acceptance-policy-v1',freeze_sha256=freeze,calibration_inputs=inputs,
        policy=policies,scope='Empirical calibration; no distribution-free or global theorem guarantee')
    with (ROOT/'research/LLM_ACCEPTANCE_POLICY.json').open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    print(json.dumps(policies,indent=2))


if __name__=='__main__':main()
