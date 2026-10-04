"""Controlled response observations; compress without discarding residuals."""
import argparse
import gc
import json
import time
import torch

from src.llm_shared import sha256
from src.llm_observations import (observed_frame,designed_queries,normalized_queries,
    perturb,compress_on_frame,truth_for_scoring,cpu_state)
from src.response_identifiability import apply_euclidean_response
from experiments.run_autograd_tomography import autodiff_response,relative_error
from experiments.llm_protocol import CONFIG,MODELS,check_freeze,validate_unit,unit_path


def run(args):
    freeze=check_freeze();validate_unit(args.model,args.seed,args.split)
    out=unit_path(args.model,args.seed,args.split)
    target=out/'observations.pt'
    if target.exists():raise RuntimeError('Refusing observation overwrite')
    training=json.loads((out/'training.json').read_text())
    if training['freeze_sha256']!=freeze or not all(training['gates'].values()):
        raise ValueError('Invalid training provenance')
    if sha256(out/'checkpoint.pt')!=training['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    torch.set_num_threads(4);torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.reset_peak_memory_stats()
    start=time.monotonic()
    factors=torch.load(out/'checkpoint.pt',weights_only=True,map_location=args.device)
    a,b=factors['a'],factors['b'];theta=a@b
    natural={}
    for domain,meta in training['gradient_records'].items():
        if sha256(meta['path'])!=meta['sha256']:raise ValueError('Gradient record changed')
        natural[domain]=torch.load(meta['path'],weights_only=True)['gradients']
    cases=[];controls=[];timings=[];finite=[]
    for ni,noise in enumerate(CONFIG['noise']):
        rng=torch.Generator(device=args.device).manual_seed(5000000+args.seed)
        observed_theta=perturb(theta,noise,rng)
        u,singular=observed_frame(observed_theta,CONFIG['rank'])
        truth=truth_for_scoring(a,b,u)
        families=['designed','gaussian']+['natural_'+d for d in natural]
        for fi,family in enumerate(families):
            tick=time.monotonic()
            rng=torch.Generator(device=args.device).manual_seed(5010000+args.seed+fi*100)
            if family=='designed':
                fitting=designed_queries(observed_theta,CONFIG['rank'],8)
                holdout=normalized_queries(torch.randn(4,*theta.shape,device=args.device,dtype=torch.double,generator=rng))
                gradients=torch.cat((fitting,holdout))
                del fitting,holdout
            elif family=='gaussian':
                gradients=normalized_queries(torch.randn(12,*theta.shape,device=args.device,dtype=torch.double,generator=rng))
            else:
                gradients=normalized_queries(natural[family.removeprefix('natural_')].to(args.device,dtype=torch.double))
            responses=[]
            for index,g in enumerate(gradients):
                measured=autodiff_response(a,b,g)
                if ni==0 and index==0:
                    p=list(reversed(range(CONFIG['rank'])))
                    alternative=.65*torch.eye(CONFIG['rank'],device=args.device,dtype=torch.double)+.35/CONFIG['rank']
                    transformed_a=a@alternative;transformed_b=torch.linalg.solve(alternative,b)
                    controls.append(dict(family=family,
                        formula_relative=relative_error(measured,apply_euclidean_response(a,b,g)),
                        permutation_gap=relative_error(autodiff_response(a[:,p],b[p],g),measured),
                        nonpermutation_gap=relative_error(autodiff_response(transformed_a,transformed_b,g),measured),
                        nonpermutation_product_error=relative_error(transformed_a@transformed_b,theta)))
                    if family=='natural_general':
                        for dtype in (torch.float64,torch.float32,torch.bfloat16):
                            z=a.log().to(dtype).requires_grad_();bb=b.to(dtype).requires_grad_();gg=g.to(dtype)
                            base=z.softmax(-1)@bb
                            dz,db=torch.autograd.grad((base*gg).sum(),(z,bb))
                            for step in (1e-3,1e-4,1e-5):
                                with torch.no_grad():
                                    after=(z-step*dz).softmax(-1)@(bb-step*db)
                                value=(base.detach().double()-after.double())/step
                                finite.append(dict(dtype=str(dtype),step=step,error=relative_error(value,measured)))
                            del z,bb,gg,base,dz,db,value,after
                responses.append(perturb(measured,noise,rng))
            responses=torch.stack(responses)
            common=dict(noise=noise,family=family,truth=truth)
            try:
                held,held_condition=compress_on_frame(observed_theta,gradients[8:],responses[8:],u,noise,singular)
                for q in CONFIG['queries']:
                    try:
                        obs,condition=compress_on_frame(observed_theta,gradients[:q],responses[:q],u,noise,singular)
                        cases.append(dict(**common,q=q,status='ok',observation=cpu_state(obs),condition=condition,
                                          heldout=cpu_state(held),heldout_condition=held_condition))
                    except (RuntimeError,ValueError) as exc:
                        cases.append(dict(**common,q=q,status='compression_failed',error=str(exc)))
            except (RuntimeError,ValueError) as exc:
                for q in CONFIG['queries']:
                    cases.append(dict(**common,q=q,status='holdout_compression_failed',error=str(exc)))
            torch.cuda.synchronize()
            timings.append(dict(family=family,noise=noise,seconds=time.monotonic()-tick))
            print(args.model,args.seed,family,noise,'cases',len(cases),flush=True)
            del gradients,responses,measured
            gc.collect()
        del observed_theta,u,truth
    gates=dict(formula=max(c['formula_relative'] for c in controls)<1e-10,
               permutation=max(c['permutation_gap'] for c in controls)<1e-10,
               product=max(c['nonpermutation_product_error'] for c in controls)<1e-10)
    torch.save(dict(format='llm-observations-v1',freeze_sha256=freeze,
                    training_sha256=sha256(out/'training.json'),cases=cases),target)
    summary=dict(model=args.model,seed=args.seed,split=args.split,freeze_sha256=freeze,
        observations_sha256=sha256(target),cases=len(cases),
        compression_failures=sum(c['status']!='ok' for c in cases),controls=controls,gates=gates,
        finite_steps=finite,timings=timings,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        seconds=time.monotonic()-start)
    with (out/'observations.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
    if not all(gates.values()):raise RuntimeError('Observation gate failed; result retained')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=list(MODELS))
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--split',required=True,choices=['calibration','evaluation'])
    p.add_argument('--device',required=True)
    run(p.parse_args())
