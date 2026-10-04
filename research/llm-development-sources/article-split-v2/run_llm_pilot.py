"""Prospective frozen-backbone LLM adaptation and gradient-measurement pilot."""
import argparse
import datetime
import json
import time
from pathlib import Path

import torch
from src.llm_shared import load_model,attach_bank,make_blocks,token_loss,sha256
from experiments.run_autograd_tomography import autodiff_response,relative_error
from experiments.run_task_gradients import natural_reconstruct,orbit_error
from src.response_factor_recovery import recover_factors
from src.response_identifiability import apply_euclidean_response


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic()
    model,tokenizer,entry=load_model(args.model,args.device)
    bank,modules=attach_bank(model,args.seed)
    train,train_meta=make_blocks(tokenizer,'general','train','adapt',args.length)
    valid,valid_meta=make_blocks(tokenizer,'general','valid','probe',args.length,128)
    rng=torch.Generator().manual_seed(args.seed+4800000)
    initial_a=bank.logits.softmax(-1).detach().cpu()
    initial_b=bank.bases.detach().cpu().clone()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        bank.disabled=True
        baseline=[float(token_loss(model,b.to(args.device),bank)) for b in valid[:8].split(2)]
        bank.disabled=False
    optimizer=torch.optim.AdamW([{'params':[bank.bases],'lr':1e-3},
                                {'params':[bank.logits],'lr':1e-2}],weight_decay=0.)
    losses=[]; indices=[]
    train_start=time.monotonic()
    for step in range(args.steps):
        ids=torch.randperm(len(train),generator=rng)[:args.batch]
        batch=train[ids].to(args.device)
        optimizer.zero_grad(set_to_none=True)
        loss=token_loss(model,batch,bank)
        loss.backward()
        gradnorm=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.))
        optimizer.step()
        losses.append(dict(step=step+1,nll=float(loss.detach()),gradient_norm=gradnorm))
        indices.append(ids.tolist())
        if step%10==0:
            print(args.model,step+1,losses[-1]['nll'],flush=True)
    torch.cuda.synchronize()
    train_seconds=time.monotonic()-train_start
    with torch.no_grad():
        adapted=[float(token_loss(model,b.to(args.device),bank)) for b in valid[:8].split(2)]
    theta=(bank.logits.softmax(-1)@bank.bases).detach()
    gradients=[]
    controls={}
    for i,batch in enumerate(valid[8:24]):
        independent=theta.detach().clone().requires_grad_()
        loss=token_loss(model,batch[None].to(args.device),bank,independent)
        grad,=torch.autograd.grad(loss,independent)
        if i==0:
            actual=token_loss(model,batch[None].to(args.device),bank)
            dz,db,actual_g=torch.autograd.grad(actual,(bank.logits,bank.bases,bank.theta))
            za=bank.logits.detach().clone().requires_grad_()
            bb=bank.bases.detach().clone().requires_grad_()
            ez,eb=torch.autograd.grad((za.softmax(-1)@bb*actual_g).sum(),(za,bb))
            controls=dict(forward_loss_abs=float(abs(actual.detach()-loss.detach())),
                router_chain_relative=relative_error(ez,dz),basis_chain_relative=relative_error(eb,db),
                separate_backward_effective_gradient_relative=relative_error(grad,actual_g))
        gradients.append(grad.detach().cpu().double())
    a=bank.logits.detach().double().softmax(-1)
    b=bank.bases.detach().double()
    # The independently evaluated FP64 factor product defines audit truth;
    # BF16/FP32 task gradients are simply fixed effective-gradient directions.
    product=a@b
    _,singular,right=torch.linalg.svd(product,full_matrices=False)
    u=right[:bank.rank]
    grad=torch.stack(gradients).to(args.device)
    norms=grad.flatten(1).norm(dim=1)
    grad=grad/norms[:,None,None]
    responses=torch.stack([autodiff_response(a,b,g) for g in grad])
    analytic=apply_euclidean_response(a,b,grad[0],temperature=1.,eta_z=1.,eta_b=1.)
    controls['response_formula_relative']=relative_error(responses[0],analytic)
    recovery=[]
    for q in (1,4,8,12):
        gram,local,condition=natural_reconstruct(grad[:q],responses[:q],u)
        prediction=gram@grad[12:]+torch.einsum('lik,plk->pli',local,grad[12:]@u.T)@u
        record=dict(q=q,condition=condition,heldout_response_relative=relative_error(prediction,responses[12:]))
        try:
            ah,bh,details=recover_factors(product,u,gram,local)
            record['recovery']=dict(status='ok',**orbit_error(ah,bh,a,b),details=details)
        except (ValueError,RuntimeError) as exc:
            record['recovery']=dict(status='failed',error=str(exc))
        recovery.append(record)
    output=dict(format='llm-pilot-v1',purpose='Development only; not confirmatory evidence',
        model=args.model,revision=entry['revision'],seed=args.seed,steps=args.steps,
        batch=args.batch,length=args.length,modules=modules,rank=bank.rank,channels=bank.channels,
        dimensions=list(product.shape),trainable_parameters=sum(p.numel() for p in bank.parameters()),
        backbone_parameters=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in bank.parameters()),
        baseline_validation_nll=baseline,adapted_validation_nll=adapted,training=losses,
        train_indices=indices,train_data=train_meta,validation_data=valid_meta,
        router_mean_l1_movement=float((a.cpu()-initial_a).abs().sum(-1).mean()),
        basis_relative_movement=float((b.cpu()-initial_b).norm()/initial_b.norm()),
        router_min_entry=float(a.min()),router_singular_values=torch.linalg.svdvals(a).tolist(),
        basis_singular_values=torch.linalg.svdvals(b).tolist(),product_singular_values=singular.tolist(),
        controls=controls,recovery=recovery,gradient_norms=norms.tolist(),
        training_seconds=train_seconds,training_tokens_per_second=args.steps*args.batch*args.length/train_seconds,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,
        source_sha256={p:sha256(p) for p in ['experiments/run_llm_pilot.py','src/llm_shared.py',
            'src/response_factor_recovery.py','experiments/run_task_gradients.py',
            'experiments/run_autograd_tomography.py','research/LLM_STUDY_PROTOCOL.md']},
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        torch_version=torch.__version__)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(output,f,indent=2,allow_nan=False)
    torch.save(dict(a=a.cpu(),b=b.cpu(),modules=modules,model=args.model,revision=entry['revision']),
               args.output.with_suffix('.pt'))
    print(json.dumps({k:output[k] for k in ['controls','training_tokens_per_second','peak_gpu_bytes','seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--seed',type=int,default=90)
    p.add_argument('--steps',type=int,default=20)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--length',type=int,default=128)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Refusing overwrite')
    run(args)
