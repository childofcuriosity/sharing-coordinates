"""Registered secondary conditioning stress of learned factors, not new LLMs."""
import json
import time
import torch
from src.llm_shared import sha256
from src.llm_observations import (observed_frame,designed_queries,normalized_queries,
    perturb,compress_on_frame,truth_for_scoring)
from experiments.run_autograd_tomography import autodiff_response
from experiments.recover_llm_shared import estimate,score
from experiments.calibrate_llm_recovery import accepted,SCORES
from experiments.llm_protocol import ROOT,check_freeze,unit_path
from src.llm_decisions import balanced_assignment


def main():
    freeze=check_freeze();protocol_path=ROOT/'research/LLM_STRESS_PROTOCOL.json'
    protocol=json.loads(protocol_path.read_text())
    assert protocol['source_sha256']==sha256(__file__)
    policy_path=ROOT/'research/LLM_ACCEPTANCE_POLICY.json'
    policy=json.loads(policy_path.read_text())
    assert policy['freeze_sha256']==freeze
    out=ROOT/'results/llm/conditioning-stress.json'
    if out.exists():raise ValueError('Refusing stress overwrite')
    torch.set_num_threads(4);device='cuda:0'
    torch.backends.cuda.matmul.allow_tf32=False
    records=[];inputs={};start=time.monotonic()
    for model in protocol['models']:
        path=unit_path(model,0,'evaluation')/'checkpoint.pt'
        inputs[str(path.relative_to(ROOT))]=sha256(path)
        checkpoint=torch.load(path,weights_only=True,map_location=device)
        aa,bb=checkpoint['a'],checkpoint['b']
        left,sv,right=torch.linalg.svd(bb,full_matrices=False)
        labels=balanced_assignment(aa.cpu()).to(device)
        hard=torch.nn.functional.one_hot(labels,4).double()
        for axis in protocol['axes']:
            for level in protocol['levels']:
                a,b=aa,bb
                if axis=='router_rank':a=level*aa+(1-level)/4
                elif axis=='router_interior':a=level*aa+(1-level)*hard
                elif axis=='basis_rank':
                    shrunk=sv.clone();shrunk[-1]*=level;b=(left*shrunk)@right
                theta=a@b
                for noise in protocol['noise']:
                    rng=torch.Generator(device=device).manual_seed(5400000)
                    observed=perturb(theta,noise,rng);u,singular=observed_frame(observed,4)
                    gradients=designed_queries(observed,4,4)
                    hold=normalized_queries(torch.randn(4,*theta.shape,device=device,dtype=torch.double,generator=rng))
                    all_g=torch.cat((gradients,hold))
                    response=torch.stack([perturb(autodiff_response(a,b,g),noise,rng) for g in all_g])
                    row=dict(model=model,axis=axis,level=level,noise=noise,
                        alpha=float(a.min()),sigma_a=float(torch.linalg.svdvals(a)[-1]),
                        sigma_b=float(torch.linalg.svdvals(b)[-1]))
                    try:
                        obs,condition=compress_on_frame(observed,all_g[:4],response[:4],u,noise,singular)
                        held,held_condition=compress_on_frame(observed,all_g[4:],response[4:],u,noise,singular)
                        methods=score(estimate(obs,held,condition,held_condition,5400001),truth_for_scoring(a,b,u))
                        for method,result in methods.items():
                            result['acceptance']={name:accepted(result,name,policy['policy'][method][name]) for name in SCORES}
                        row.update(status='ok',condition=condition,methods=methods)
                    except (ValueError,RuntimeError) as exc:row.update(status='failed',error=str(exc))
                    records.append(row)
                    print(model,axis,level,noise,row['status'],flush=True)
    result=dict(format='llm-secondary-stress-v1',freeze_sha256=freeze,
        protocol_sha256=sha256(protocol_path),policy_sha256=sha256(policy_path),input_sha256=inputs,
        records=records,seconds=time.monotonic()-start,
        scope='Artificial transformations of learned factors; designed queries only. Not retrained models or task gradients at the transformed checkpoint.')
    with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)


if __name__=='__main__':main()
