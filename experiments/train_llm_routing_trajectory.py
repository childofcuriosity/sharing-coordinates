"""Real next-token training with exact balanced decision traces."""
import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from src.llm_shared import SharedBank,load_model,make_blocks,token_loss,sha256
from src.routing_geometry import balanced_partitions,partition,router_assignment
from experiments.llm_protocol import ROOT,MODELS,CONFIG,check_freeze

PARTS=list(balanced_partitions(12,4))
INDICES=np.asarray(PARTS)


def attach_twelve(model,seed):
    for p in model.parameters():p.requires_grad_(False)
    if hasattr(model,'gpt_neox'):
        modules=[(f'gpt_neox.layers.{i}.attention.dense',l.attention.dense) for i,l in enumerate(model.gpt_neox.layers)]
    else:
        modules=[(f'model.layers.{i}.self_attn.o_proj',l.self_attn.o_proj) for i,l in enumerate(model.model.layers)]
    indices=torch.linspace(0,len(modules)-1,12).round().long().tolist()
    selected=[modules[i] for i in indices];shape=selected[0][1].weight.shape
    assert len(set(indices))==12 and all(m.weight.shape==shape for _,m in selected)
    bank=SharedBank(12,shape[1],shape[0],4,64,seed,selected[0][1].weight.device)
    model.add_module('sharing_bank',bank)
    for i,(_,module) in enumerate(selected):
        module.register_forward_hook(lambda module,args,out,i=i:bank.inject(i,args[0],out))
    model.config.use_cache=False
    return bank,[n for n,_ in selected]


def cluster(a,gram):
    diff=a[:,None]-a[None,:]
    dist=np.einsum('ijk,kl,ijl->ij',diff,gram,diff)
    costs=np.zeros(len(PARTS))
    for i,j in itertools.combinations(range(3),2):
        costs+=dist[INDICES[:,:,i],INDICES[:,:,j]].sum(1)/3
    order=np.argsort(costs);best=int(order[0])
    return dict(partition=PARTS[best],cost=float(costs[best]),gap=float(costs[order[1]]-costs[best]))


def snapshot(bank,step):
    a=bank.logits.detach().double().softmax(-1).cpu().numpy()
    b=bank.bases.detach().double();gram=(b@b.T).cpu().numpy()
    labels,score,gap=router_assignment(a)
    return dict(step=step,a=a.tolist(),gram=gram.tolist(),coefficient=dict(partition=partition(labels),
        labels=labels.tolist(),score=score,gap=gap),weight=cluster(a,gram),router=cluster(a,np.eye(4)),
        min_a=float(a.min()),sigma_a=float(np.linalg.svd(a,compute_uv=False)[-1]),
        sigma_b=float(np.sqrt(max(0,np.linalg.eigvalsh(gram)[0]))))


def strict(row,key):
    obj=row[key];value=obj.get('cost',obj.get('score'))
    return obj['gap']>1e-8*max(1,abs(value))


def agrees(row,key):return row[key]['partition']==row['weight']['partition']


def events(before,after):
    found=[]
    for key in ['coefficient','router']:
        if (strict(before,key) and strict(before,'weight') and strict(after,key) and strict(after,'weight')
            and agrees(before,key) and not agrees(after,key)):
            found.append(key+'_entry')
            if key=='coefficient' and before['weight']['partition']==after['weight']['partition']:
                found.append('router_boundary')
    return found


def cpu_copy(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    return copy.deepcopy(x)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True,choices=['pythia160m','qwen06b','smol360m'])
    parser.add_argument('--seed',required=True,type=int)
    parser.add_argument('--optimizer',required=True,choices=['adamw','sgd'])
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args();check_freeze();torch.set_num_threads(2)
    torch.manual_seed(args.seed);torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'results/llm_routing_trajectories'/f'{args.model}-{args.optimizer}-seed{args.seed}'
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();model,tok,entry=load_model(MODELS[args.model],args.device)
    bank,modules=attach_twelve(model,args.seed)
    train,meta=make_blocks(tok,'general','train','adapt',128)
    valid,vmeta=make_blocks(tok,'general','valid','loss',128,32)
    def evaluate():
        with torch.no_grad():return float(np.mean([float(token_loss(model,x.to(args.device),bank)) for x in valid.split(2)]))
    initial_validation=evaluate()
    groups=[dict(params=[bank.bases],lr=.001),dict(params=[bank.logits],lr=.01)]
    opt=torch.optim.AdamW(groups,weight_decay=0) if args.optimizer=='adamw' else torch.optim.SGD(groups,momentum=0,weight_decay=0)
    rng=torch.Generator().manual_seed(4900000+args.seed)
    torch.save(dict(logits=bank.logits.detach().cpu(),bases=bank.bases.detach().cpu()),out/'initial.pt')
    trace=[snapshot(bank,0)];found=[];saved=set()
    for step in range(256):
        mult=(step+1)/16 if step<16 else .1+.9*(1+math.cos(math.pi*(step-16)/(256-16-1)))/2
        for group,lr in zip(opt.param_groups,[.001,.01]):group['lr']=lr*mult
        ids=torch.randperm(len(train),generator=rng)[:4]
        # Store a replay buffer only until each event type has its first witness.
        preserve=len(saved)<3
        before_state=cpu_copy(dict(logits=bank.logits,bases=bank.bases,optimizer=opt.state_dict())) if preserve else None
        opt.zero_grad(set_to_none=True)
        loss=token_loss(model,train[ids].to(args.device),bank);loss.backward()
        gradnorm=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.))
        grads=cpu_copy(dict(logits=bank.logits.grad,bases=bank.bases.grad)) if preserve else None
        opt.step();current=snapshot(bank,step+1)
        current.update(nll=float(loss.detach()),batch_indices=ids.tolist(),gradient_norm=gradnorm,lr_multiplier=mult)
        for kind in events(trace[-1],current):
            found.append(dict(kind=kind,step=step+1))
            if kind not in saved:
                torch.save(dict(before=before_state,after=cpu_copy(dict(logits=bank.logits,bases=bank.bases)),
                    clipped_gradients=grads,batch_indices=ids.tolist(),step=step+1,kind=kind),out/(kind+'.pt'))
                saved.add(kind)
        trace.append(current)
        if (step+1)%64==0:print(out.name,step+1,'events',len(found),flush=True)
    for event in found:
        key='router' if event['kind']=='router_entry' else 'coefficient'
        window=trace[event['step']:event['step']+5]
        event['sustained_five']=len(window)==5 and all(strict(r,key) and strict(r,'weight') and not agrees(r,key) for r in window)
    final_validation=evaluate()
    torch.save(dict(logits=bank.logits.detach().cpu(),bases=bank.bases.detach().cpu()),out/'final.pt')
    result=dict(model=args.model,model_id=MODELS[args.model],revision=entry['revision'],seed=args.seed,
        optimizer=args.optimizer,modules=modules,scope='Frozen real pretrained backbone; 12 selected projections, 64 output channels, K4 adapter; natural next-token task.',
        train_data=meta,validation_data=vmeta,initial_validation_nll=initial_validation,final_validation_nll=final_validation,
        source_sha256={p:sha256(ROOT/p) for p in ['experiments/train_llm_routing_trajectory.py','src/llm_shared.py','src/routing_geometry.py']},
        protocol_sha256=sha256(ROOT/'research/LLM_ROUTING_TRAJECTORY_PROTOCOL.md'),
        torch_version=torch.__version__,numpy_version=np.__version__,events=found,trace=trace,
        seconds=time.monotonic()-start,checkpoints={p.name:sha256(p) for p in out.glob('*.pt')})
    with (out/'trajectory.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(out.name,'complete',len(found),'events','validation',initial_validation,final_validation,flush=True)

if __name__=='__main__':main()
