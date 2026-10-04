"""Fresh LM backward and optimizer replay at a retained natural-task crossing."""
import argparse
import json
from pathlib import Path
import torch
from src.llm_shared import load_model,make_blocks,token_loss,sha256
from experiments.train_llm_routing_trajectory import attach_twelve
from experiments.llm_protocol import ROOT,MODELS


def main():
    p=argparse.ArgumentParser();p.add_argument('--unit',required=True);p.add_argument('--kind',required=True)
    p.add_argument('--device',default='cuda:3');args=p.parse_args()
    torch.set_num_threads(2);torch.cuda.set_device(args.device);torch.backends.cuda.matmul.allow_tf32=False
    unit=ROOT/'results/llm_routing_trajectories'/args.unit
    r=json.loads((unit/'trajectory.json').read_text());cp=torch.load(unit/(args.kind+'.pt'),map_location='cpu',weights_only=True)
    model,tok,_=load_model(MODELS[r['model']],args.device);bank,modules=attach_twelve(model,r['seed'])
    assert modules==r['modules']
    with torch.no_grad():bank.logits.copy_(cp['before']['logits']);bank.bases.copy_(cp['before']['bases'])
    blocks,meta=make_blocks(tok,'general','train','adapt',128)
    assert meta['token_sha256']==r['train_data']['token_sha256']
    groups=[dict(params=[bank.bases],lr=.001),dict(params=[bank.logits],lr=.01)]
    optimizer=torch.optim.AdamW(groups,weight_decay=0) if r['optimizer']=='adamw' else torch.optim.SGD(groups)
    optimizer.load_state_dict(cp['before']['optimizer'])
    loss=token_loss(model,blocks[cp['batch_indices']].to(args.device),bank)
    loss.backward();gradnorm=float(torch.nn.utils.clip_grad_norm_(bank.parameters(),1.))
    errors={}
    for name,parameter in [('logits',bank.logits),('bases',bank.bases)]:
        g=parameter.grad.detach().cpu();expected=cp['clipped_gradients'][name]
        errors[name]=dict(max_abs=float((g-expected).abs().max()),relative=float((g-expected).norm()/expected.norm().clamp_min(1e-30)))
    left=r['trace'][cp['step']-1];right=r['trace'][cp['step']]
    # Differential of old-versus-new coefficient score under Euclidean descent.
    a=bank.logits.detach().double().softmax(-1)
    old=torch.tensor(left['coefficient']['labels'],device=args.device)
    new=torch.tensor(right['coefficient']['labels'],device=args.device)
    e=torch.nn.functional.one_hot(old,4)-torch.nn.functional.one_hot(new,4)
    normal=a*(e-(a*e).sum(-1,keepdim=True))
    gradient_direction=float((normal*(-bank.logits.grad.double())).sum())
    optimizer.step()
    update_error=max(float((bank.logits.detach().cpu()-cp['after']['logits']).abs().max()),
                     float((bank.bases.detach().cpu()-cp['after']['bases']).abs().max()))
    loss_error=abs(float(loss.detach())-right['nll'])
    result=dict(unit=args.unit,kind=args.kind,step=cp['step'],nll=float(loss.detach()),nll_abs_error=loss_error,
        gradient_norm=gradnorm,gradient_replay=errors,actual_update_max_abs_error=update_error,
        coefficient_score_velocity_under_clipped_euclidean_descent=gradient_direction,
        scope='Fresh pretrained-backbone forward/backward on recorded natural language batch, then restored optimizer update. Euclidean direction is a local diagnostic, not an AdamW trajectory theorem.',
        source_sha256=sha256(Path(__file__)),event_checkpoint_sha256=sha256(unit/(args.kind+'.pt')))
    result['passed']=loss_error<1e-5 and max(v['relative'] for v in errors.values())<1e-4 and update_error<2e-6
    out=unit/('backward-replay-'+args.kind+'.json')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
    assert result['passed']

if __name__=='__main__':main()
