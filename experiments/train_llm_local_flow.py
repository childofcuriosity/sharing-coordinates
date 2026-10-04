"""Fixed natural language batch, full FP32 vanilla SGD near an observed boundary."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src.llm_shared import load_model,make_blocks,sha256
from experiments.train_llm_routing_trajectory import attach_twelve,snapshot,strict,agrees
from experiments.llm_protocol import ROOT,MODELS


def main():
    torch.set_num_threads(2);torch.cuda.set_device('cuda:3');torch.backends.cuda.matmul.allow_tf32=False
    unit=ROOT/'results/llm_routing_trajectories/smol360m-adamw-seed1'
    cp=torch.load(unit/'router_boundary.pt',map_location='cpu',weights_only=True)
    original=json.loads((unit/'trajectory.json').read_text())
    model,tok,_=load_model(MODELS['smol360m'],'cuda:3');bank,_=attach_twelve(model,1);model.float()
    train,meta=make_blocks(tok,'general','train','adapt',128)
    assert meta['token_sha256']==original['train_data']['token_sha256']
    batch=train[cp['batch_indices']].to('cuda:3')
    e=(F.one_hot(torch.tensor(original['trace'][70]['coefficient']['labels']),4)-
       F.one_hot(torch.tensor(original['trace'][71]['coefficient']['labels']),4)).numpy()
    out=ROOT/'results/llm_local_flow';out.mkdir(exist_ok=False)
    def loss_fn():
        bank.prepare()
        pred=model(input_ids=batch[:,:-1],use_cache=False).logits
        return F.cross_entropy(pred.flatten(0,1),batch[:,1:].reshape(-1))
    runs=[];initial_velocity=None
    for lr in [.02,.01,.005,.0025,0.]:
        with torch.no_grad():bank.logits.copy_(cp['before']['logits']);bank.bases.copy_(cp['before']['bases'])
        opt=torch.optim.SGD([bank.logits,bank.bases],lr=lr)
        count=round(.4/lr) if lr else 20;trace=[]
        for step in range(count+1):
            opt.zero_grad(set_to_none=True);loss=loss_fn()
            current=snapshot(bank,step)
            current.update(time=step*lr,loss=float(loss.detach()),p_minus_q=float((np.asarray(current['a'])*e).sum()))
            trace.append(current)
            if step==count:break
            loss.backward()
            if initial_velocity is None:
                a=bank.logits.detach().double().softmax(-1);ee=torch.tensor(e,device=a.device)
                normal=a*(ee-(a*ee).sum(-1,keepdim=True))
                initial_velocity=float((normal*(-bank.logits.grad.double())).sum())
            opt.step()
        crossings=[r['step'] for r in trace if strict(r,'coefficient') and strict(r,'weight') and not agrees(r,'coefficient') and r['weight']['partition']==trace[0]['weight']['partition']]
        checkpoint=out/f'lr-{lr}.pt';torch.save(dict(logits=bank.logits.detach().cpu(),bases=bank.bases.detach().cpu()),checkpoint)
        runs.append(dict(lr=lr,steps=count,initially_agree=agrees(trace[0],'coefficient'),crossing_steps=crossings,
            fixed_weight_partition=all(r['weight']['partition']==trace[0]['weight']['partition'] for r in trace),
            max_loss_increase=float(np.max(np.diff([r['loss'] for r in trace]))),
            checkpoint_sha256=sha256(checkpoint),trace=trace))
        print('lr',lr,'first crossing',crossings[:1],'loss',trace[0]['loss'],trace[-1]['loss'],flush=True)
    convergence=[]
    for coarse,fine in zip(runs[:3],runs[1:4]):
        c=torch.load(out/f"lr-{coarse['lr']}.pt",weights_only=True);f=torch.load(out/f"lr-{fine['lr']}.pt",weights_only=True)
        error=float(torch.sqrt(sum((c[k].double()-f[k].double()).square().sum() for k in c)))
        convergence.append(dict(coarse_lr=coarse['lr'],fine_lr=fine['lr'],endpoint_parameter_distance=error))
    result=dict(scope='Post hoc selected real trained state; unchanged natural-language task, fixed actual batch; full FP32 equal-rate Euclidean SGD; not natural-initialization prevalence.',
        source_sha256=sha256(Path(__file__)),protocol_sha256=sha256(ROOT/'research/LLM_LOCAL_FLOW_PROTOCOL.md'),
        input_checkpoint_sha256=sha256(unit/'router_boundary.pt'),token_sha256=meta['token_sha256'],batch_indices=cp['batch_indices'],
        initial_coefficient_gap_velocity=initial_velocity,
        linearized_crossing_time=-runs[0]['trace'][0]['p_minus_q']/initial_velocity,
        convergence=convergence,runs=runs)
    with (out/'local_flow.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('Velocity',initial_velocity,'linearized crossing',result['linearized_crossing_time'],flush=True)
    print(convergence,flush=True)

if __name__=='__main__':main()
