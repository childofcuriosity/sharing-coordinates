"""Derive publication tables and vector figures from frozen dynamics records."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.routing_geometry import balanced_partitions,partition_cost,squared_distances

ROOT=Path(__file__).resolve().parents[1]

def load(p):return json.loads((ROOT/p).read_text())

def generate(out):
    (out/'generated').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    n=np.array([[8,11,4],[8,5,10],[10,9,4],[5,11,7],[6,10,7],[11,5,7]])
    n0=np.array([[40,24,28],[24,44,24],[32,28,32],[43,37,12],[24,20,48],[37,31,24]])
    m=np.array([[2,1,7],[4,4,2],[5,2,3]])
    cert=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',
          r'Partition & $N$ & $N\operatorname{Diag}(4,1,1)$ & $NM_{\rm int}$ & $N_0$ \\',r'\midrule']
    for groups in balanced_partitions(6,3):
        label='$'+r'\,|\,'.join(''.join(str(i+1) for i in g) for g in groups)+'$'
        values=[int(partition_cost(squared_distances(x),groups)) for x in [n,n@np.diag([4,1,1]),n@m,n0]]
        cert.append(label+' & '+' & '.join(map(str,values))+r' \\')
    cert += [r'\bottomrule\end{tabular}',r'\caption{Complete integer SSE certificates. A label such as $12|34|56$ denotes three pairs. Divide columns by $23^2$, $23^2$, $230^2$, and $92^2$, respectively, for the stated router/effective-weight costs. Here $M_{\rm int}=10M$. Each column is computed by Equation~\eqref{eq:pairwise-sse}; no numerical optimizer is used.}',r'\label{tab:decision-certificates}',r'\end{table}']
    (out/'generated/decision_certificates.tex').write_text('\n'.join(cert)+'\n')
    records=[json.loads(p.read_text()) for p in sorted((ROOT/'results/llm_routing_trajectories').glob('*/trajectory.json'))]
    assert len(records)==18
    names={'pythia160m':'Pythia-160M','qwen06b':'Qwen3-0.6B','smol360m':'SmolLM2-360M'}
    def count(runs,kind):return sum(any(e['kind']==kind for e in r['events']) for r in runs)
    table=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{llrrr}',r'\toprule',r'Model & Optimizer & Coefficient & Router only & Response \\',r'\midrule']
    for model in names:
        rs=[r for r in records if r['model']==model and r['optimizer']=='adamw']
        table.append(names[model]+' & AdamW & '+' & '.join(f'{count(rs,k)}/3' for k in ['coefficient_entry','router_boundary','router_entry'])+r' \\')
    rs=[r for r in records if r['optimizer']=='sgd']
    table.append('All three & SGD & '+' & '.join(f'{count(rs,k)}/9' for k in ['coefficient_entry','router_boundary','router_entry'])+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Runs with at least one strict agreement-to-disagreement transition against global weight clustering, over the entire 256-step trajectory. Coefficient means $S_A$ assignment; router only is its subset with unchanged weight partition; response means $J_A$ clustering under the Euclidean-response objective, not an AdamW-optimal decision. All runs use 12 selected projections, $K=4$, and 64 channels. Counts are descriptive, not prevalence estimates.}',r'\label{tab:training-crossings}',r'\end{table}']
    (out/'generated/dynamics_crossings.tex').write_text('\n'.join(table)+'\n')
    table=[r'\begin{table}[htbp]',r'\centering\small',r'\begin{tabular}{llrrrrr}',r'\toprule',r'Model & Optimizer & Seed & Coeff. & Router only & Response & Valid NLL before/after \\',r'\midrule']
    for r in records:
        vals=[]
        for kind in ['coefficient_entry','router_boundary','router_entry']:
            steps=[str(e['step']) for e in r['events'] if e['kind']==kind];vals.append(','.join(steps) if steps else '--')
        table.append(f"{names[r['model']]} & {r['optimizer']} & {r['seed']} & "+' & '.join(vals)+f" & {r['initial_validation_nll']:.4f}/{r['final_validation_nll']:.4f}"+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{All 18 exploratory runs. Event columns give every transition step, not independent samples. Validation uses 32 fixed WikiText validation blocks per model, not a full-corpus benchmark. Identical seeds reuse the same initial router across models; architecture, pretraining, and selected layers differ. SGD uses the original schedule without retuning.}',r'\label{tab:all-dynamics-runs}',r'\end{table}']
    (out/'generated/dynamics_all_runs.tex').write_text('\n'.join(table)+'\n')
    local=load('results/llm_local_flow_fp64_norm/local_flow.json');fp32=load('results/llm_local_flow/local_flow.json');toy=load('results/routing_training/training.json')
    table=[r'\begin{table}[t]',r'\centering\small',r'\begin{tabular}{rrrrr}',r'\toprule',r'Step size & Updates & First crossing & Training time & Adjacent endpoint distance \\',r'\midrule']
    for i,r in enumerate(local['runs']):
        first=r['crossing_steps'][0]
        if i<3:
            mantissa,exponent=f"{local['convergence'][i]['endpoint_parameter_distance']:.2e}".split('e')
            err=f"${mantissa}\\times10^{{{int(exponent)}}}$"
        else:err='--' 
        table.append(f"{r['lr']:.4f} & {r['steps']} & {first} & {first*r['lr']:.3f} & {err}"+r' \\')
    table += [r'\bottomrule\end{tabular}',r'\caption{Selected real-model local SGD with FP64 factors, backbone arithmetic and dtype-preserving RMSNorm. Training time is update count times step size. Endpoint distances compare each run to the next finer step size at time $0.4$, not to an exact flow solution. All four runs keep the weight partition fixed.}',r'\label{tab:local-flow-refinement}',r'\end{table}']
    (out/'generated/dynamics_local_flow.tex').write_text('\n'.join(table)+'\n')
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors=['#0072B2','#D55E00','#009E73','#CC79A7'];styles=['-','--','-.',':']
    r=next(r for r in records if r['model']=='smol360m' and r['optimizer']=='adamw' and r['seed']==1)
    old,new=r['trace'][70:72];e=np.eye(4)[old['coefficient']['labels']]-np.eye(4)[new['coefficient']['labels']]
    fig,axes=plt.subplots(1,3,figsize=(10,2.9),layout='constrained')
    window=r['trace'][65:76];axes[0].plot([t['step'] for t in window],[float((np.array(t['a'])*e).sum()) for t in window],'-o',ms=3,color=colors[0]);axes[0].axhline(0,color='gray',lw=.8)
    axes[0].axvline(71,color=colors[1],ls='--',lw=.8);axes[0].set(xlabel='AdamW update',ylabel='Old minus new score')
    for i,run in enumerate(local['runs']):
        axes[1].plot([t['time'] for t in run['trace']],[t['p_minus_q'] for t in run['trace']],color=colors[i],ls=styles[i],label=f"h={run['lr']}")
    axes[1].axhline(0,color='gray',lw=.8);axes[1].set(xlabel='Cumulative SGD step size',ylabel='Old minus new score');axes[1].legend(fontsize=7)
    h=np.array([x['coarse_lr'] for x in local['convergence']]);err=np.array([x['endpoint_parameter_distance'] for x in local['convergence']])
    axes[2].loglog(h,err,'o-',color=colors[0],label='Observed');axes[2].loglog(h,err[0]*h/h[0],'k--',label='Slope 1')
    axes[2].set(xlabel='Coarser step size',ylabel='Endpoint distance');axes[2].legend(fontsize=7)
    for i,ax in enumerate(axes):ax.text(.02,.98,'('+chr(97+i)+')',transform=ax.transAxes,va='top',fontweight='bold')
    fig.savefig(out/'figures/dynamics_crossing.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,2.9),layout='constrained')
    rs=[r for r in toy['primary'] if r['dtype']=='torch.float64']
    for i,r in enumerate(rs):axes[0].plot([t['time'] for t in r['trace']],[t['p_minus_q'] for t in r['trace']],ls=styles[i],color=colors[i],label=f"h={r['lr']}")
    ref=toy['forward_ode_reference'];axes[0].plot([t['time'] for t in ref],[t['p_minus_q'] for t in ref],'k--',lw=.8,label='ODE')
    axes[0].axhline(0,color='gray',lw=.8);axes[0].set(xlabel='Cumulative SGD step size',ylabel='Old minus new score');axes[0].legend(fontsize=7)
    axes[1].loglog([r['lr'] for r in rs],[r['endpoint_parameter_error_vs_flow'] for r in rs],'o-',color=colors[0]);axes[1].set(xlabel='Step size',ylabel='Error against ODE reference')
    for data,color,label in [(fp32,colors[1],'Real model FP32'),(local,colors[0],'Real model FP64')]:
        axes[2].loglog([c['coarse_lr'] for c in data['convergence']],[c['endpoint_parameter_distance'] for c in data['convergence']],'o-',color=color,label=label)
    axes[2].set(xlabel='Coarser step size',ylabel='Adjacent endpoint distance');axes[2].legend(fontsize=7)
    for i,ax in enumerate(axes):ax.text(.02,.98,'('+chr(97+i)+')',transform=ax.transAxes,va='top',fontweight='bold')
    fig.savefig(out/'figures/dynamics_controls.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);generate(out)
            for f in out.rglob('*'):
                if f.is_file():
                    target=ROOT/'paper'/f.relative_to(out)
                    assert f.read_bytes()==target.read_bytes(),target
        print('Dynamics tables and vector figures match frozen records')
    else:generate(ROOT/'paper')

if __name__=='__main__':main()
