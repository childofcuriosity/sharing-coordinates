"""Read-only provenance and independent real-observation checks."""
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT,PARENT,read,sha,check_freeze
from weighted import PARTS,response


def main():
    torch.set_num_threads(2);check_freeze();lock=read(ROOT/'SELECTION_LOCK.json')
    plan=read(ROOT/'ANALYSIS_PLAN_LOCK.json')
    assert plan['source_sha256']==sha(ROOT/'analyze.py')
    assert plan['evaluation_files_present']==0
    assert plan['created_unix']<min(r['start_unix'] for r in read(ROOT/'EVALUATE_EXECUTION.json')['records'])
    for p,d in lock['selections'].items():assert sha(ROOT/p)==d,p
    checks=[];worst=0.;jvp_worst=0.
    for folder in sorted((ROOT/'results').glob('*-seed*')):
        if not folder.is_dir():continue
        s=read(folder/'selection.json');e=read(folder/'evaluation.json')
        for r in [s,e]:
            for p,d in r['artifacts'].items():assert sha(folder/p)==d,(folder,p)
        assert e['selection_sha256']==sha(folder/'selection.json')
        sel={d['document_sha256'] for d in s['data']['documents']};ev={d['document_sha256'] for d in e['data']['documents']}
        assert len(sel)==16 and len(ev)==32 and not sel&ev
        checkpoint=torch.load(PARENT/f"results/llm_routing_trajectories/{s['model']}-adamw-seed{s['seed']}/final.pt",map_location='cpu',weights_only=True)
        z=checkpoint['logits'].double();b=checkpoint['bases'].double();a=z.softmax(-1).numpy();bb=b.numpy()
        selectcost=np.load(folder/'selection_objectives.npz')
        for method in ['weight','isotropic_response','task_response','task_single','activation']:
            assert s['selected'][method]['index']==int(np.argmin(selectcost[method]))
        assert s['selected']['direct_loss']['index']==s['direct_candidates'][int(np.argmin(np.mean(s['direct_document_nll'],axis=1)))]
        candidates=sorted(set(v['index'] for v in s['selected'].values()))
        for phase in ['selection','evaluation']:
            original=torch.load(folder/f'{phase}_gradients.pt',map_location='cpu',weights_only=True)
            gs=original['gradients'];n=len(gs);alpha=1. # Actual measured four-document batch gradients; no synthetic batch recombination.
            # Independently form actual update differences rather than S/T affinities.
            sums=np.zeros(len(candidates));mean_d=[np.zeros(gs.shape[1:],dtype=np.float64) for _ in candidates]
            for j,g in enumerate(gs):
                gg=g.double().numpy();y=response(a,bb,gg)
                if phase=='selection' and j==0:
                    zz=z.clone().requires_grad_();bbb=b.clone().requires_grad_()
                    product=zz.softmax(-1)@bbb;gz,gb=torch.autograd.grad((product*g.double()).sum(),(zz,bbb))
                    _,observed=torch.func.jvp(lambda x,t:x.softmax(-1)@t,(zz.detach(),bbb.detach()),(gz,gb))
                    err=np.linalg.norm(observed.numpy()-y)/max(np.linalg.norm(y),1e-30);jvp_worst=max(jvp_worst,float(err));assert err<1e-12
                for k,index in enumerate(candidates):
                    hard=np.zeros_like(gg)
                    for group in PARTS[index]:hard[list(group)]=gg[list(group)].sum(0)
                    d=y-hard;sums[k]+=np.sum(d*d)/n;mean_d[k]+=d/n
            predicted=alpha*sums+(1-alpha)*np.array([np.sum(d*d) for d in mean_d])
            recorded=np.load(folder/f'{phase}_objectives.npz')['task_response']
            for k,index in enumerate(candidates):
                error=abs(predicted[k]-recorded[index])/max(abs(predicted[k]),1e-30)
                worst=max(worst,float(error));assert error<1e-10,(folder,phase,index,error)
                checks.append(dict(unit=folder.name,phase=phase,index=index,relative_error=float(error)))
        for index,action in e['actions'].items():
            assert set(action['method_aliases'])=={m for m,v in s['selected'].items() if v['index']==int(index)}
            for opt,r in action['recovery'].items():
                assert r['steps']==len(r['training_nll'])==64 and len(r['document_nll'])==32
                assert np.isfinite(r['document_nll']).all() and np.isfinite(r['training_nll']).all()
                assert sha(folder/f'partition-{index}-{opt}.pt')==r['checkpoint_sha256']
    result=dict(units=9,independent_direct_batch_objectives=len(checks),maximum_relative_objective_error=worst,
        real_factor_autograd_jvp_checks=9,maximum_relative_jvp_error=jvp_worst,checks=checks,
        sources_and_artifacts_match=True,selection_before_evaluation_verified=True,
        parent_publication_pdf_unchanged=sha(PARENT/'paper/main.pdf')==read(ROOT/'FREEZE.json')['parent_inputs']['paper/main.pdf'])
    print(json.dumps(result,indent=2,allow_nan=False))

if __name__=='__main__':main()
