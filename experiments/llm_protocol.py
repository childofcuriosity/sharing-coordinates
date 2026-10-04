"""Frozen cross-model design and source checks shared by execution stages."""
import json
from pathlib import Path
from src.llm_shared import sha256

ROOT=Path(__file__).resolve().parents[1]
MODELS={
    'pythia160m':'EleutherAI/pythia-160m',
    'pythia1b':'EleutherAI/pythia-1b',
    'pythia28b':'EleutherAI/pythia-2.8b',
    'qwen06b':'Qwen/Qwen3-0.6B',
    'qwen17b':'Qwen/Qwen3-1.7B',
    'qwen4b':'Qwen/Qwen3-4B',
    'qwen8b':'Qwen/Qwen3-8B',
    'smol360m':'HuggingFaceTB/SmolLM2-360M',
    'smol17b':'HuggingFaceTB/SmolLM2-1.7B',
}
CONFIG=dict(rank=4,channels=64,steps=256,batch=4,length=128,
    basis_lr=1e-3,router_lr=1e-2,warmup=16,minimum_lr_fraction=.1,
    weight_decay=0.,clip=1.,queries=[1,4,8],query_pool=12,
    noise=[0.,1e-6,1e-4,1e-2],starts=3,max_evaluations=80,
    seeds=[0,1,2],calibration_models=['pythia160m','pythia1b'],calibration_seeds=[101,102],
    calibration_domain='general',evaluation_domains=['general','code','math'],
    good_error=.05,bad_error=.10,acceptance_bad_target=.05,minimum_calibration_accepted=20,
    decision_models=['pythia1b','qwen4b','smol17b'],decision_steps=64,
    decision_noise=[1e-4,1e-2],decision_query=4,decision_lr=1e-4)


def check_freeze():
    target=ROOT/'research/LLM_STUDY_FREEZE.json'
    frozen=json.loads(target.read_text())
    if frozen['config']!=CONFIG or frozen['models']!=MODELS:
        raise ValueError('Frozen configuration mismatch')
    for name,digest in frozen['source_sha256'].items():
        if sha256(ROOT/name)!=digest:
            raise ValueError('Frozen source changed: '+name)
    for row in json.loads((ROOT/'research/LLM_CORPUS_LOCK.json').read_text())['documents']:
        if sha256(ROOT/row['path'])!=row['sha256']:
            raise ValueError('Corpus changed: '+row['path'])
    return sha256(target)


def validate_unit(model,seed,split):
    if model not in MODELS:
        raise ValueError(model)
    if split=='calibration':
        if model not in CONFIG['calibration_models'] or seed not in CONFIG['calibration_seeds']:
            raise ValueError('Not a declared calibration unit')
    elif split=='evaluation':
        if seed not in CONFIG['seeds']:
            raise ValueError('Not a declared evaluation seed')
    else:
        raise ValueError(split)


def unit_path(model,seed,split):
    return ROOT/'results/llm'/split/f'{model}-seed{seed}'
