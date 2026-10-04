"""Replay retained optimizer updates and decompose exact clustering-gap changes."""
import json
from pathlib import Path
import numpy as np
import torch
from experiments.llm_protocol import ROOT
from src.llm_shared import sha256


def projection(groups,n=12):
    q=np.zeros((n,n))
    for group in groups:q[np.ix_(group,group)]=1/len(group)
    return q


def audit(path):
    record=json.loads((path/'trajectory.json').read_text());results=[]
    for event_path in sorted(path.glob('*_entry.pt'))+sorted(path.glob('router_boundary.pt')):
        cp=torch.load(event_path,map_location='cpu',weights_only=True)
        before=cp['before'];after=cp['after'];step=cp['step'];left=record['trace'][step-1];right=record['trace'][step]
        z=before['logits'].clone().requires_grad_();b=before['bases'].clone().requires_grad_()
        groups=[dict(params=[b],lr=.001),dict(params=[z],lr=.01)]
        opt=torch.optim.AdamW(groups,weight_decay=0) if record['optimizer']=='adamw' else torch.optim.SGD(groups)
        opt.load_state_dict(before['optimizer'])
        z.grad=cp['clipped_gradients']['logits'].clone();b.grad=cp['clipped_gradients']['bases'].clone();opt.step()
        # CPU versus CUDA update arithmetic need not be byte identical.
        replay_error=max(float((z.detach()-after['logits']).abs().max()),float((b.detach()-after['bases']).abs().max()))
        assert replay_error<2e-6,(event_path,replay_error)
        za=before['logits'].double().numpy();zb=after['logits'].double().numpy()
        a0=before['logits'].double().softmax(-1).numpy();a1=after['logits'].double().softmax(-1).numpy()
        b0=before['bases'].double().numpy();b1=after['bases'].double().numpy()
        t0=a0@b0;t1=a1@b1;dt=t1-t0
        result=dict(kind=cp['kind'],step=step,cpu_update_replay_max_abs=replay_error,checkpoint_sha256=sha256(event_path))
        if left['weight']['partition']!=right['weight']['partition']:
            # F=J_new-J_old; positive before the old->new optimum switch.
            w=projection(left['weight']['partition'])-projection(right['weight']['partition'])
            f0=float((t0*(w@t0)).sum());f1=float((t1*(w@t1)).sum())
            da=a1-a0;db=b1-b0
            linear_a=float(2*((w@t0)*(da@b0)).sum())
            linear_b=float(2*((w@t0)*(a0@db)).sum())
            cross=float(2*((w@t0)*(da@db)).sum())
            quadratic=float((dt*(w@dt)).sum())
            error=abs((f1-f0)-(linear_a+linear_b+cross+quadratic))
            assert error<1e-9*max(1,abs(f0),abs(f1))
            result['weight_boundary']=dict(gap_before=f0,gap_after=f1,router_component=linear_a,
                basis_component=linear_b,bilinear_component=cross,quadratic_component=quadratic,identity_error=error)
        if left['router']['partition']!=right['router']['partition']:
            w=projection(left['router']['partition'])-projection(right['router']['partition'])
            da=a1-a0
            f0=float((a0*(w@a0)).sum());f1=float((a1*(w@a1)).sum())
            linear=float(2*((w@a0)*da).sum());quadratic=float((da*(w@da)).sum())
            error=abs(f1-f0-linear-quadratic)
            assert error<1e-12
            result['response_boundary']=dict(gap_before=f0,gap_after=f1,
                linear_component=linear,quadratic_component=quadratic,identity_error=error)
        if left['coefficient']['partition']!=right['coefficient']['partition']:
            e=np.eye(4)[left['coefficient']['labels']]-np.eye(4)[right['coefficient']['labels']]
            dz=zb-za
            linear=sum(float(er@(np.diag(ar)-np.outer(ar,ar))@delta) for er,ar,delta in zip(e,a0,dz))
            f0=float((e*a0).sum());f1=float((e*a1).sum())
            result['coefficient_boundary']=dict(gap_before=f0,gap_after=f1,linear_prediction=linear,
                observed_change=f1-f0,remainder=f1-f0-linear)
        results.append(result)
    return dict(unit=path.name,events=results)


def main():
    torch.set_num_threads(1)
    rows=[audit(p.parent) for p in sorted((ROOT/'results/llm_routing_trajectories').glob('*/trajectory.json'))]
    if len(rows)!=18:raise ValueError(f'Expected all 18 declared runs, got {len(rows)}')
    result=dict(scope='Replays stored clipped gradients through optimizer, not a fresh language-model backward pass; exact gap decomposition uses retained factors.',
        source_sha256=sha256(Path(__file__)),runs=rows)
    path=ROOT/'results/llm_routing_trajectories/event_audit.json'
    with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print('Audited',len(rows),'runs;',sum(len(r['events']) for r in rows),'retained event updates')

if __name__=='__main__':main()
