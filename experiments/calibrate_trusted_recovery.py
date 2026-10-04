"""Freeze thresholds on calibration only; never reads evaluation outputs."""
import json
from datetime import datetime,timezone
from experiments.trusted_protocol import (ROOT,FREEZE,POLICY,METHODS,SCORES,
    expected_paths,load_records,eligible,sha)

def select_threshold(rows,method,score):
    candidates=[]
    for row in rows:
        r=row.get('methods',{}).get(method,{'status':'failed'})
        if eligible(r,score):
            candidates.append((r['diagnostics'][score],
                r['evaluation']['quality']=='bad',row['key']['kind']=='synthetic'))
    candidates.sort()
    accepted_count=bad=synthetic_count=0
    best=None
    for i,(value,is_bad,is_synthetic) in enumerate(candidates):
        accepted_count+=1;bad+=is_bad;synthetic_count+=is_synthetic
        if i+1<len(candidates) and candidates[i+1][0]==value: continue
        if bad/accepted_count<=.05 and synthetic_count>=20:
            best=dict(threshold=value,accepted=accepted_count,bad=bad,
                      synthetic_accepted=synthetic_count)
    return best or dict(threshold=None,accepted=0,bad=0,synthetic_accepted=0)

def main():
    if POLICY.exists(): raise SystemExit('Refusing policy overwrite')
    paths=expected_paths('calibration')
    rows=load_records(paths,'calibration')
    if len(rows)!=984: raise ValueError('Incomplete calibration cohort')
    policy=dict(format='trusted-calibration-v1',created_utc=datetime.now(timezone.utc).isoformat(),
        freeze_sha256=sha(FREEZE),calibration_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        scope='Pooled calibration, fixed separately per method/score; same thresholds transferred to both domains.',
        constraints=dict(max_bad_fraction=.05,min_accepted_synthetic=20),
        rules={m:{s:select_threshold(rows,m,s) for s in SCORES} for m in METHODS})
    POLICY.write_text(json.dumps(policy,indent=2,allow_nan=False)+'\n')
    print(json.dumps(policy['rules'],indent=2))

if __name__=='__main__': main()
