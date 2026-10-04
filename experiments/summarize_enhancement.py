"""Validate additive evidence and derive enhancement tables and precision figure."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read_bound(path):
    value=json.loads(path.read_text())
    sources=value.get('source_sha256',value.get('metadata',{}).get('source_sha256',{}))
    if not sources:
        raise ValueError(f'Missing source attestation: {path}')
    for name,digest in sources.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Source hash mismatch: {path}: {name}')
    return value


def paired_interval(a,b,seed):
    difference=np.asarray(a)-np.asarray(b)
    rng=np.random.default_rng(seed)
    samples=difference[rng.integers(0,len(difference),size=(10000,len(difference)))].mean(1)
    return dict(delta_bpb=float(difference.mean()),conditional_bootstrap_95=np.quantile(samples,[.025,.975]).tolist())


def derive():
    folder=ROOT/'results/enhancement'
    decisions=[];tasks=[];raw={}
    for seed in (10,11,12):
        path=folder/f'seed{seed}/decisions.json'
        r=read_bound(path)
        training=read_bound(folder/f'seed{seed}/training.json')
        digest=hashlib.sha256((folder/f'seed{seed}/checkpoint.pt').read_bytes()).hexdigest()
        if r['seed']!=seed or r['checkpoint_sha256']!=digest or training['checkpoint_sha256']!=digest:
            raise ValueError('Training checkpoint binding mismatch')
        if training['config']['steps']!=6000 or training['metadata']['saved_at_step']!=6000:
            raise ValueError('Primary training budget mismatch')
        raw[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows={}
        for name,action in r['actions'].items():
            if name in ('positive_stochastic','signed_local'):
                search=r['searches'][name]
                if not search['equivalence_pass'] or search['status']!='found':
                    raise ValueError('Invalid gauge action')
                if sorted(action['group_sizes'])!=sorted(r['actions']['native']['group_sizes']):
                    raise ValueError('Partition size mismatch')
            row={}
            for step in ('0','100','500'):
                stage=action['stages'][step]
                if len(stage['batch_bpb'])!=64 or stage['tokens']!=64*8*128:
                    raise ValueError('Evaluation budget mismatch')
                if abs(np.mean(stage['batch_bpb'])-stage['instantaneous_bpb'])>1e-12:
                    raise ValueError('BPB aggregate mismatch')
                contrast=paired_interval(stage['batch_bpb'],r['actions']['native']['stages'][step]['batch_bpb'],790000+seed+int(step))
                row[step]=dict(bpb=stage['instantaneous_bpb'],**contrast)
            rows[name]=row
        decisions.append(dict(seed=seed,router_movement=r['router_movement'],
            soft_bpb=r['soft']['instantaneous_bpb'],weak_model=r['weak_model'],initialization_dominated=r['initialization_dominated'],
            searches={k:dict(status=v['status'],counts=v['counts']) for k,v in r['searches'].items()},actions=rows))
    for seed in (0,1,2,10,11,12):
        path=folder/f'task_seed{seed}.json';r=read_bound(path)
        checkpoint=ROOT/(f'results/checkpoints/language_seed{seed}.pt' if seed<3 else f'results/enhancement/seed{seed}/checkpoint.pt')
        if r['seed']!=seed or r['checkpoint_sha256']!=hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError('Task checkpoint mismatch')
        if not all(r['gates'].values()) or len(r['reconstructions'])!=4:
            raise ValueError('Task measurement gate failure')
        raw[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        tasks.append(dict(seed=seed,reconstructions=r['reconstructions'],finite_steps=r['finite_steps'],
            nonpermutation_gap_range=[min(r['charts']['nonpermutation']['response_relative_gaps']),max(r['charts']['nonpermutation']['response_relative_gaps'])]))
    jointpath=folder/'joint_noise.json';joint=read_bound(jointpath)
    raw[str(jointpath.relative_to(ROOT))]=hashlib.sha256(jointpath.read_bytes()).hexdigest()
    if len(joint['records'])!=450:
        raise ValueError('Joint-noise grid incomplete')
    return dict(format='enhancement-summary-v1',raw_sha256=raw,decisions=decisions,tasks=tasks,joint_noise=joint['records'])


def tables(summary):
    d=[r'\begin{table}[t]',r'\centering\small',
       r'\caption{Prospective equal-size folding audit. Signed BPB differences are gauge-selected minus native-selected folds after matched recovery updates. Each row is one independent training seed. Positive-stochastic search succeeds in 3/3 seeds; signed-local search fails in 3/3, each with 5,000 router-only candidates.}',
       r'\label{tab:enhanced-decisions}',r'\begin{tabular}{rrrrr}',r'\toprule',
       r'Seed & Router movement & $\Delta_0$ & $\Delta_{100}$ & $\Delta_{500}$ \\',r'\midrule']
    for row in summary['decisions']:
        action=row['actions'].get('positive_stochastic')
        values=[f"{action[s]['delta_bpb']:+.5f}" if action else '--' for s in ('0','100','500')]
        d.append(f"{row['seed']} & {row['router_movement']:.3f} & "+' & '.join(values)+r' \\')
    d += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    t=[r'\begin{table}[t]',r'\centering\small',
       r'\caption{Four natural task-gradient probes with known response structure. Original and new checkpoints use the separately registered 1,500- and 6,000-step configurations. Recovery error is the combined relative factor error after evaluation-only permutation alignment. These are FP64 fixed-checkpoint measurements, not observed training trajectories.}',
       r'\label{tab:natural-gradients}',r'\begin{tabular}{llrr}',r'\toprule',
       r'Configuration & Seed & Max. held-out response error & Factor error \\',r'\midrule']
    for row in summary['tasks']:
        r=next(x for x in row['reconstructions'] if x['q']==4)
        def sci(v):
            mantissa,exponent=f'{v:.2e}'.split('e')
            return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
        t.append(f"{'Original' if row['seed']<3 else 'New'} & {row['seed']} & {sci(max(r['structured_heldout_errors']))} & {sci(r['recovery']['orbit_relative_error'])}"+r' \\')
    t += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return {'enhanced_decisions_table.tex':'\n'.join(d)+'\n','natural_gradients_table.tex':'\n'.join(t)+'\n'}


def figure(summary,target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.65),layout='constrained')
    for dtype,color in [('torch.float64','#0072B2'),('torch.float32','#E69F00'),('torch.bfloat16','#009E73')]:
        rows=[[x['response_relative_error'] for x in r['finite_steps'] if x['dtype']==dtype] for r in summary['tasks'] if r['seed']>=10]
        x=[.1,.01,.001,.0001,.00001];rows=np.array(rows)
        axes[0].loglog(x,np.median(rows,axis=0),'-o',ms=3,label=dtype.replace('torch.',''),color=color)
        axes[0].fill_between(x,rows.min(0),rows.max(0),alpha=.15,color=color)
    axes[0].set(xlabel='Finite Euclidean step size',ylabel='Relative response error')
    axes[0].legend(fontsize=7)
    for level,color in [(1.,'#0072B2'),(.1,'#E69F00'),(.01,'#009E73'),(.001,'#CC79A7')]:
        xs=[1e-8,1e-6,1e-4,1e-2];ys=[]
        for noise in xs:
            values=[r['orbit_relative_error'] for r in summary['joint_noise'] if r['regime']=='rank' and r['level']==level and r['relative_noise']==noise and r['status']=='ok']
            ys.append(np.median(values))
        axes[1].loglog(xs,ys,'-o',ms=3,color=color,label=f'rank scale {level:g}')
    axes[1].set(xlabel='Relative product and response noise',ylabel='Median factor-recovery error')
    axes[1].legend(fontsize=7)
    for axis in axes:
        axis.tick_params(labelsize=8)
        axis.grid(alpha=.2,which='major')
    fig.savefig(target,metadata={'CreationDate':None,'ModDate':None})
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
    s=derive()
    outputs={ROOT/'results/enhancement/summary.json':json.dumps(s,indent=2,allow_nan=False)+'\n'}
    outputs.update({ROOT/'paper/generated'/k:v for k,v in tables(s).items()})
    for path,content in outputs.items():
        if a.check:
            if path.read_text()!=content: raise SystemExit(f'Stale enhancement derivative: {path}')
        else: path.write_text(content)
    if not a.check: figure(s,ROOT/'paper/figures/enhancement_precision.pdf')
    print('Enhancement source bindings and derivatives verified')
