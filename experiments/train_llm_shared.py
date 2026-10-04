"""Train frozen-backbone shared modules and save true task gradients."""
import argparse
import datetime
import json
import math
import time
from pathlib import Path
import torch

from src.llm_shared import load_model,attach_bank,make_blocks,token_loss,sha256
from experiments.llm_protocol import CONFIG,MODELS,check_freeze,validate_unit,unit_path
from experiments.run_autograd_tomography import relative_error


def evaluate(model,bank,blocks,device):
    with torch.no_grad():
        return [float(token_loss(model,b.to(device),bank)) for b in blocks[:32].split(2)]


def run(args):
    frozen=check_freeze()
    validate_unit(args.model,args.seed,args.split)
    out=unit_path(args.model,args.seed,args.split)
    if (out/'training.json').exists():
        raise ValueError('Refusing to overwrite completed training')
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    start=time.monotonic()
    model,tokenizer,entry=load_model(MODELS[args.model],args.device)
    bank,modules=attach_bank(model,args.seed,CONFIG['rank'],CONFIG['channels'])
    train,train_meta=make_blocks(tokenizer,'general','train','adapt',CONFIG['length'])
    split='valid' if args.split=='calibration' else 'test'
    domains=['general'] if args.split=='calibration' else CONFIG['evaluation_domains']
    eval_blocks={};probe_blocks={};data={}
    for domain in domains:
        eval_blocks[domain],em=make_blocks(tokenizer,domain,split,'loss',CONFIG['length'],256)
        probe_blocks[domain],pm=make_blocks(tokenizer,domain,split,'probe',CONFIG['length'],256)
        data[domain]=dict(loss=em,probe=pm)
    torch.cuda.reset_peak_memory_stats()
    bank.disabled=True
    baseline={d:evaluate(model,bank,x,args.device) for d,x in eval_blocks.items()}
    bank.disabled=False
    initial_a=bank.logits.softmax(-1).detach().cpu()
    initial_b=bank.bases.detach().cpu().clone()
    optimizer=torch.optim.AdamW([{'params':[bank.bases],'lr':CONFIG['basis_lr']},
                                {'params':[bank.logits],'lr':CONFIG['router_lr']}],weight_decay=0.)
    rng=torch.Generator().manual_seed(4900000+args.seed)
    losses=[];batches=[]
    train_start=time.monotonic()
    for step in range(CONFIG['steps']):
        if step<CONFIG['warmup']:
            multiplier=(step+1)/CONFIG['warmup']
        else:
            phase=(step-CONFIG['warmup'])/(CONFIG['steps']-CONFIG['warmup']-1)
            multiplier=.1+.9*(1+math.cos(math.pi*phase))/2
        for group,lr in zip(optimizer.param_groups,[CONFIG['basis_lr'],CONFIG['router_lr']]):
            group['lr']=lr*multiplier
        ids=torch.randperm(len(train),generator=rng)[:CONFIG['batch']]
        optimizer.zero_grad(set_to_none=True)
        loss=token_loss(model,train[ids].to(args.device),bank)
        loss.backward()
        gradnorm=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),CONFIG['clip']))
        optimizer.step()
        losses.append(dict(step=step+1,nll=float(loss.detach()),gradient_norm=gradnorm,lr_multiplier=multiplier))
        batches.append(ids.tolist())
        if (step+1)%64==0:
            print(args.model,args.seed,step+1,losses[-1]['nll'],flush=True)
    torch.cuda.synchronize()
    train_seconds=time.monotonic()-train_start
    adapted={d:evaluate(model,bank,x,args.device) for d,x in eval_blocks.items()}
    a=bank.logits.detach().double().softmax(-1).cpu()
    b=bank.bases.detach().double().cpu()
    checkpoint=out/'checkpoint.pt'
    torch.save(dict(a=a,b=b,logits=bank.logits.detach().cpu(),bases=bank.bases.detach().cpu(),
        modules=modules,model=MODELS[args.model],revision=entry['revision'],freeze_sha256=frozen),checkpoint)
    gradient_dir=Path('.cache/llm/gradients')/args.split/f'{args.model}-seed{args.seed}'
    gradient_dir.mkdir(parents=True,exist_ok=True)
    measured={};controls={}
    for domain,blocks in probe_blocks.items():
        generator=torch.Generator().manual_seed(4910000+args.seed)
        indices=torch.randperm(len(blocks),generator=generator)[:CONFIG['query_pool']].tolist()
        gradients=[];nll=[]
        theta=(bank.logits.softmax(-1)@bank.bases).detach()
        for j,index in enumerate(indices):
            independent=theta.clone().requires_grad_()
            loss=token_loss(model,blocks[index:index+1].to(args.device),bank,independent)
            grad,=torch.autograd.grad(loss,independent)
            if j==0:
                actual=token_loss(model,blocks[index:index+1].to(args.device),bank)
                dz,db,gg=torch.autograd.grad(actual,(bank.logits,bank.bases,bank.theta))
                z=bank.logits.detach().clone().requires_grad_()
                bb=bank.bases.detach().clone().requires_grad_()
                ez,eb=torch.autograd.grad((z.softmax(-1)@bb*gg).sum(),(z,bb))
                controls[domain]=dict(forward_loss_abs=float(abs(actual.detach()-loss.detach())),
                    router_chain_relative=relative_error(ez,dz),basis_chain_relative=relative_error(eb,db),
                    separate_backward_gradient_relative=relative_error(grad,gg))
            gradients.append(grad.detach().cpu());nll.append(float(loss.detach()))
        path=gradient_dir/(domain+'.pt')
        tensor=torch.stack(gradients)
        torch.save(dict(gradients=tensor,indices=indices,nll=nll,freeze_sha256=frozen),path)
        measured[domain]=dict(path=str(path),sha256=sha256(path),block_indices=indices,nll=nll,
            norms=tensor.flatten(1).norm(dim=1).tolist(),shape=list(tensor.shape))
    gates=dict(finite=all(math.isfinite(x['nll']) for x in losses),
        forward=max(v['forward_loss_abs'] for v in controls.values())<1e-6,
        same_graph_chain=max(max(v['router_chain_relative'],v['basis_chain_relative']) for v in controls.values())<1e-5)
    result=dict(format='llm-training-v1',model=args.model,model_id=MODELS[args.model],revision=entry['revision'],
        split=args.split,seed=args.seed,freeze_sha256=frozen,config=CONFIG,modules=modules,
        dimensions=[bank.depth,bank.channels*bank.inputs],trainable_parameters=sum(p.numel() for p in bank.parameters()),
        backbone_parameters=sum(p.numel() for p in model.parameters())-sum(p.numel() for p in bank.parameters()),
        baseline_nll=baseline,adapted_nll=adapted,training=losses,batch_indices=batches,
        train_data=train_meta,data=data,gradient_records=measured,controls=controls,gates=gates,
        checkpoint_sha256=sha256(checkpoint),router_mean_l1_movement=float((a-initial_a).abs().sum(-1).mean()),
        basis_relative_movement=float((b-initial_b).norm()/initial_b.norm()),
        router_min_entry=float(a.min()),router_singular_values=torch.linalg.svdvals(a).tolist(),
        basis_singular_values=torch.linalg.svdvals(b).tolist(),
        training_seconds=train_seconds,training_tokens=CONFIG['steps']*CONFIG['batch']*CONFIG['length'],
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch_version=torch.__version__)
    with (out/'training.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(unit=out.name,gates=gates,seconds=result['seconds'])),flush=True)
    if not all(gates.values()):raise RuntimeError('Training/measurement gate failed; preserve result')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True,choices=list(MODELS))
    p.add_argument('--seed',required=True,type=int)
    p.add_argument('--split',required=True,choices=['calibration','evaluation'])
    p.add_argument('--device',required=True)
    run(p.parse_args())
