"""Summarize the complete declared grids and plot retained crossing trajectories."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.llm_shared import sha256
from experiments.llm_protocol import ROOT


def main():
    toy=json.loads((ROOT/'results/routing_training/training.json').read_text())
    paths=sorted((ROOT/'results/llm_routing_trajectories').glob('*/trajectory.json'))
    real=[json.loads(p.read_text()) for p in paths];assert len(real)==18
    fp32=json.loads((ROOT/'results/llm_local_flow/local_flow.json').read_text())
    fp64=json.loads((ROOT/'results/llm_local_flow_fp64/local_flow.json').read_text())
    local=json.loads((ROOT/'results/llm_local_flow_fp64_norm/local_flow.json').read_text())
    grouped={}
    for optimizer in ['adamw','sgd']:
        runs=[r for r in real if r['optimizer']==optimizer]
        grouped[optimizer]=dict(runs=len(runs),validation_improved=sum(r['final_validation_nll']<r['initial_validation_nll'] for r in runs))
        for kind in ['coefficient_entry','router_boundary','router_entry']:
            grouped[optimizer][kind]=dict(runs_with_event=sum(any(e['kind']==kind for e in r['events']) for r in runs),
                runs_with_sustained_event=sum(any(e['kind']==kind and e['sustained_five'] for e in r['events']) for r in runs))
    r=next(r for r in real if r['model']=='smol360m' and r['optimizer']=='adamw' and r['seed']==1)
    before,after=r['trace'][70:72]
    summary=dict(scope='Exploratory grids, one selected local real-model follow-up, and a controlled construction; no prevalence or downstream-recovery claim.',
        toy=toy['summary'],real=grouped,
        local_fp32=dict(crossing_runs=sum(bool(v['crossing_steps']) for v in fp32['runs']),runs=5,convergence=fp32['convergence']),
        outer_fp64_with_fp32_norm=dict(convergence=fp64['convergence'],crossing_runs=sum(bool(v['crossing_steps']) for v in fp64['runs'])),
        local_fp64=dict(crossing_runs=sum(bool(v['crossing_steps']) for v in local['runs']),runs=4,
            first_crossing_times=[v['crossing_steps'][0]*v['lr'] if v['crossing_steps'] else None for v in local['runs']],
            convergence=local['convergence'],initial_gap_velocity=local['initial_coefficient_gap_velocity'],
            linearized_crossing_time=local['linearized_crossing_time'],
            min_a=min(t['min_a'] for v in local['runs'] for t in v['trace']),
            min_sigma_a=min(t['sigma_a'] for v in local['runs'] for t in v['trace']),
            min_sigma_b=min(t['sigma_b'] for v in local['runs'] for t in v['trace'])),
        natural_witness=dict(unit='smol360m-adamw-seed1',step=71,before=before,after=after),
        input_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths+[ROOT/'results/routing_training/training.json',ROOT/'results/llm_local_flow/local_flow.json',ROOT/'results/llm_local_flow_fp64/local_flow.json',ROOT/'results/llm_local_flow_fp64_norm/local_flow.json']},
        source_sha256=sha256(Path(__file__)))
    out=ROOT/'results/routing_training'
    with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2,allow_nan=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(11,7.6),layout='constrained')
    primary=[v for v in toy['primary'] if v['dtype']=='torch.float64']
    for v in primary:
        ax[0,0].plot([t['time'] for t in v['trace']],[t['p_minus_q'] for t in v['trace']],label=f"SGD h={v['lr']}",alpha=.8)
    ref=toy['forward_ode_reference'];ax[0,0].plot([t['time'] for t in ref],[t['p_minus_q'] for t in ref],'k--',label='ODE reference')
    ax[0,0].axhline(0,color='gray',lw=.8);ax[0,0].set(title='A. Controlled fixed quadratic loss',xlabel='Training time = step x learning rate',ylabel='Old minus new assignment score')
    ax[0,0].legend(fontsize=8)
    e=np.eye(4)[before['coefficient']['labels']]-np.eye(4)[after['coefficient']['labels']]
    window=r['trace'][60:81]
    ax[0,1].plot([t['step'] for t in window],[float((np.asarray(t['a'])*e).sum()) for t in window],'-o',ms=3)
    ax[0,1].axhline(0,color='gray',lw=.8);ax[0,1].axvline(71,color='firebrick',ls='--',lw=1)
    ax[0,1].set(title='B. Natural SmolLM2 training (AdamW)',xlabel='Update',ylabel='Old minus new assignment score')
    ax[0,1].text(.03,.05,'Global weight partition unchanged at 70 -> 71',transform=ax[0,1].transAxes,fontsize=8)
    for v in local['runs']:
        ax[1,0].plot([t['time'] for t in v['trace']],[t['p_minus_q'] for t in v['trace']],label=f"h={v['lr']}")
    ax[1,0].axhline(0,color='gray',lw=.8)
    ax[1,0].set(title='C. Same real trained state + fixed real text (FP64 SGD)',xlabel='Training time = step x learning rate',ylabel='Old minus new assignment score')
    ax[1,0].legend(fontsize=8)
    conv=local['convergence'];h=np.array([c['coarse_lr'] for c in conv]);err=np.array([c['endpoint_parameter_distance'] for c in conv])
    ax[1,1].loglog(h,err,'o-',label='Real LM: adjacent step resolutions')
    ax[1,1].loglog(h,err[0]*h/h[0],'k--',label='First-order slope guide')
    ax[1,1].set(title='D. Real LM: numerical refinement',xlabel='Coarser learning rate',ylabel='Endpoint parameter distance')
    ax[1,1].legend(fontsize=8)
    fig.suptitle('Crossing evidence: controlled construction, natural trajectory, local Euclidean reproduction',fontsize=12)
    fig.savefig(out/'crossing_evidence.png',dpi=180);fig.savefig(out/'crossing_evidence.pdf')
    print(json.dumps({k:summary[k] for k in ['real','local_fp32','local_fp64']},indent=2))

if __name__=='__main__':main()
