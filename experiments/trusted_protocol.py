"""Prospective split enforcement and local source/threshold freeze."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/'research/TRUSTED_RECOVERY_FREEZE.json'
POLICY = ROOT/'results/trusted_recovery/calibration_policy.json'
SOURCES = [
 'src/trusted_recovery.py','experiments/build_trusted_observations.py',
 'experiments/run_trusted_recovery.py','experiments/trusted_protocol.py',
 'experiments/calibrate_trusted_recovery.py','research/TRUSTED_RECOVERY_PROTOCOL.md',
 'experiments/train_enhancement.py','experiments/run_autograd_tomography.py',
 'src/response_factor_recovery.py','src/response_identifiability.py',
 'src/enhancement.py','src/language.py','src/models.py','src/patterns.py']
SPLITS = {
 'synthetic': {'development':range(4),'calibration':range(100,120),'evaluation':range(200,240)},
 'language': {'development':[10],'calibration':[20,21],'evaluation':[22,23,24,25]}}
METHODS = ['spectral','projected_spectral','joint_fit']
SCORES = ['empirical_score','residual_only_score','condition_only_score']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_inventory():
    return {p:sha(ROOT/p) for p in SOURCES}

def check_split(kind,split,seed,pilot=False,starts=3,budget=80,limit=None):
    if seed not in SPLITS[kind][split]: raise ValueError('Unregistered seed/split')
    if split=='development': return
    if pilot or limit is not None or starts!=3 or budget!=80:
        raise ValueError('Primary budget cannot differ from frozen protocol')
    frozen=json.loads(FREEZE.read_text())
    if source_inventory()!=frozen['source_sha256']:
        raise ValueError('Frozen source mismatch')
    if split=='evaluation':
        policy=json.loads(POLICY.read_text())
        if policy['freeze_sha256']!=sha(FREEZE): raise ValueError('Threshold freeze mismatch')
        for path,digest in policy['calibration_sha256'].items():
            if sha(ROOT/path)!=digest: raise ValueError('Calibration changed after freeze')

def eligible(result,score):
    import math
    return (result.get('status')=='ok' and result['diagnostics']['feasibility']
            and math.isfinite(result['diagnostics'][score]))

def accepted(result,score,threshold):
    return threshold is not None and eligible(result,score) and result['diagnostics'][score]<=threshold

def load_records(paths,split):
    rows=[]
    for path in paths:
        d=json.loads(Path(path).read_text())
        if d['split']!=split or d['pilot']: raise ValueError('Wrong split or pilot')
        check_split(d['kind'],split,d['seed'],starts=d['starts'],budget=d['max_evaluations'])
        for src,digest in d['source_sha256'].items():
            if sha(ROOT/src)!=digest: raise ValueError('Recovery source mismatch: '+src)
        obs=Path(d['observation_file'])
        if not obs.is_absolute(): obs=ROOT/obs
        if sha(obs)!=d['observation_sha256']: raise ValueError('Observation hash mismatch')
        rows.extend(d['records'])
    return rows

def expected_paths(split):
    base=ROOT/'results/trusted_recovery'/split
    return [base/f'{kind}{seed}.json' for kind in SPLITS for seed in SPLITS[kind][split]]
