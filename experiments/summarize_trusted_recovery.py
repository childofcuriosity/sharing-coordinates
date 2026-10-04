"""Rebuild tables and plots from complete, source-bound frozen evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from experiments.trusted_protocol import (ROOT,FREEZE,POLICY,METHODS,SCORES,
    SPLITS,source_inventory,sha,accepted,expected_paths)

def validate_load(split):
    policy=json.loads(POLICY.read_text());freeze=json.loads(FREEZE.read_text())
    if freeze['source_sha256']!=source_inventory(): raise ValueError('Frozen sources changed')
    if policy['freeze_sha256']!=sha(FREEZE): raise ValueError('Policy source freeze changed')
    rows=[]
    for path in expected_paths(split):
        data=json.loads(path.read_text())
        kind=data['kind'];seed=data['seed']
        if path.name!=f'{kind}{seed}.json' or seed not in SPLITS[kind][split]:
            raise ValueError('Wrong or duplicate independent unit')
        if data['split']!=split or data['pilot'] or data['starts']!=3 or data['max_evaluations']!=80:
            raise ValueError('Wrong protocol')
        if data['freeze_sha256']!=sha(FREEZE): raise ValueError('Recovery freeze changed')
        if split=='calibration':
            if sha(path)!=policy['calibration_sha256'][str(path.relative_to(ROOT))]:
                raise ValueError('Frozen calibration changed')
        elif data['policy_sha256']!=sha(POLICY): raise ValueError('Recovery used another policy')
        for src,digest in data['source_sha256'].items():
            if sha(ROOT/src)!=digest: raise ValueError('Recovery source mismatch')
        # Execution paths are provenance; relocate by the registered logical ID.
        obs_path=path.with_suffix('.pt')
        if sha(obs_path)!=data['observation_sha256']: raise ValueError('Observation mismatch')
        obs=torch.load(obs_path,map_location='cpu',weights_only=False)
        if (obs['kind'],obs['split'],obs['seed'])!=(kind,split,seed):
            raise ValueError('Observation identity mismatch')
        for src,digest in obs['source_sha256'].items():
            if sha(ROOT/src)!=digest: raise ValueError('Observation source mismatch')
        regimes=['language'] if kind=='language' else ['interior','rank_01','rank_001','boundary_001']
        expected={(regime,noise,q) for regime in regimes for noise in [0.,1e-8,1e-6,1e-4,1e-3,1e-2] for q in [4,12]}
        keys=[(r['key']['regime'],r['key']['noise'],r['key']['q']) for r in data['records']]
        if set(keys)!=expected or len(keys)!=len(expected): raise ValueError('Incomplete/duplicated grid')
        if [r['key'] for r in data['records']]!=[r['key'] for r in obs['records']]:
            raise ValueError('Observation/recovery key mismatch')
        if kind=='language':
            checkpoint=ROOT/obs['metadata']['checkpoint_path']
            if sha(checkpoint)!=obs['metadata']['checkpoint_sha256']: raise ValueError('Checkpoint changed')
            training=json.loads(checkpoint.with_name('training.json').read_text())
            if training['checkpoint_sha256']!=sha(checkpoint): raise ValueError('Training checkpoint mismatch')
            if training['config']['seed']!=seed or training['config']['steps']!=6000:
                raise ValueError('Training budget mismatch')
            for src,digest in training['metadata']['source_sha256'].items():
                if sha(ROOT/src)!=digest: raise ValueError('Training source mismatch')
        rows.extend(data['records'])
    return rows

def result(row,method):
    return row.get('methods',{}).get(method,{'status':'failed'})

def measures(rows,method,score,threshold):
    rr=[result(r,method) for r in rows]
    ok=[r.get('status')=='ok' for r in rr]
    good=[o and r['evaluation']['quality']=='good' for o,r in zip(ok,rr)]
    bad=[o and r['evaluation']['quality']=='bad' for o,r in zip(ok,rr)]
    acc=[accepted(r,score,threshold) for r in rr]
    errors=[r['evaluation']['orbit_relative_error'] for o,r in zip(ok,rr) if o]
    n=len(rr);na=sum(acc);nb=sum(bad);ng=sum(good)
    ratio=lambda x,y: x/y if y else None
    return dict(n=n,successes=sum(ok),failures=n-sum(ok),good=ng,bad=nb,
        intermediate=sum(ok)-ng-nb,accepted=na,accepted_bad=sum(a and b for a,b in zip(acc,bad)),
        coverage=ratio(na,n),bad_rejection_recall=ratio(sum(b and not a for a,b in zip(acc,bad)),nb),
        good_false_rejection=ratio(sum(g and not a for a,g in zip(acc,good)),ng),
        accepted_bad_fraction=ratio(sum(a and b for a,b in zip(acc,bad)),na),
        median_error=float(np.median(errors)) if errors else None,
        selected_solver_budget_exhausted=sum(o and r.get('solver',{}).get('status')==0 for o,r in zip(ok,rr)),
        p90_error=float(np.quantile(errors,.9)) if errors else None,
        max_error=max(errors) if errors else None,
        numerical_local_pass=sum(o and r['diagnostics']['local_feasible_component_pass'] for o,r in zip(ok,rr)))

def bootstrap(rows,method,score,threshold):
    seeds=sorted(set(r['key']['seed'] for r in rows))
    if len(seeds)<10: return None
    groups=[[r for r in rows if r['key']['seed']==seed] for seed in seeds]
    stats=[measures(group,method,score,threshold) for group in groups]
    # Cluster resampling on counts, preserving all correlated noise/probe variants.
    counts=np.array([[d['n'],d['accepted'],d['bad'],d['good'],d['accepted_bad'],
        d['bad']-(d['accepted_bad']),d['good']*(d['good_false_rejection'] or 0)] for d in stats])
    rng=np.random.default_rng(3600000)
    draws=counts[rng.integers(0,len(seeds),size=(2000,len(seeds)))].sum(axis=1)
    pairs={'coverage':(1,0),'accepted_bad_fraction':(4,1),
           'bad_rejection_recall':(5,2),'good_false_rejection':(6,3)}
    intervals={}
    for key,(num,den) in pairs.items():
        values=draws[draws[:,den]>0,num]/draws[draws[:,den]>0,den]
        intervals[key]=np.quantile(values,[.025,.975]).tolist() if len(values) else None
    return dict(unit='synthetic seed; all 48 variants retained together',resamples=2000,
                percentile_95=intervals)

def summarize():
    validate_load('calibration')
    rows=validate_load('evaluation')
    policy=json.loads(POLICY.read_text())
    output=dict(format='trusted-summary-v1',evaluation_rows=len(rows),policy_sha256=sha(POLICY),
        groups={},paired={},input_sha256={str(p.relative_to(ROOT)):sha(p) for p in expected_paths('evaluation')})
    groups={kind:[r for r in rows if r['key']['kind']==kind] for kind in SPLITS}
    for regime in ['interior','rank_01','rank_001','boundary_001']:
        groups['synthetic_'+regime]=[r for r in groups['synthetic'] if r['key']['regime']==regime]
    for seed in SPLITS['language']['evaluation']:
        groups[f'language_seed{seed}']=[r for r in groups['language'] if r['key']['seed']==seed]
    for kind in SPLITS:
        for q in [4,12]:
            groups[f'{kind}_q{q}']=[r for r in groups[kind] if r['key']['q']==q]
    for group,items in groups.items():
        output['groups'][group]={}
        for method in METHODS:
            output['groups'][group][method]={}
            for score in SCORES:
                threshold=policy['rules'][method][score]['threshold']
                d=measures(items,method,score,threshold)
                if group=='synthetic': d['cluster_bootstrap']=bootstrap(items,method,score,threshold)
                output['groups'][group][method][score]=d
        if group in SPLITS:
            paired=[]
            for row in items:
                a=result(row,'projected_spectral');b=result(row,'joint_fit')
                if a.get('status')==b.get('status')=='ok':
                    paired.append((row['key']['seed'],a['evaluation']['orbit_relative_error'],
                                   b['evaluation']['orbit_relative_error']))
            ratios=[b/max(a,1e-15) for _,a,b in paired]
            output['paired'][group]=dict(n=len(paired),joint_better=sum(b<a for _,a,b in paired),
                median_joint_over_projected_error=float(np.median(ratios)),
                geometric_joint_over_projected_error=float(np.exp(np.mean(np.log(np.maximum(ratios,1e-15))))))
    return output,rows

def table(summary):
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Frozen held-out noisy recovery. Each synthetic instance contributes 48 correlated variants; each language checkpoint contributes 12. Acceptance uses the calibrated empirical score. Bad means factor error $>0.10$; good means $\le0.05$. Local passes concern only a numerical restricted-subspace component.}',
        r'\label{tab:trusted-recovery}',r'\begin{tabular}{llrrrrr}',r'\toprule',
        r'Domain & Method & Median error & Bad/total & Accept & Bad/accept & Local \\',r'\midrule']
    labels={'spectral':'Spectral','projected_spectral':'Projected','joint_fit':'Joint fit'}
    for kind,name in [('synthetic','Synthetic'),('language','Byte-LM')]:
        for method in METHODS:
            d=summary['groups'][kind][method]['empirical_score']
            bad_fraction=f"{d['accepted_bad']}/{d['accepted']}" if d['accepted'] else '--'
            lines.append(f"{name} & {labels[method]} & {d['median_error']:.2g} & {d['bad']}/{d['n']} & {d['accepted']} & {bad_fraction} & {d['numerical_local_pass']} "+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(lines)+'\n'

def figure(rows,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(7.5,2.7),constrained_layout=True)
    colors={'spectral':'#9b5968','projected_spectral':'#c28b23','joint_fit':'#257496'}
    for method in METHODS:
        for kind,ax in [('synthetic',axes[0]),('language',axes[1])]:
            selected=[r for r in rows if r['key']['kind']==kind]
            xs=[1e-10,1e-8,1e-6,1e-4,1e-3,1e-2]
            ys=[];lo=[];hi=[]
            for noise in [0.,1e-8,1e-6,1e-4,1e-3,1e-2]:
                values=[result(r,method)['evaluation']['orbit_relative_error'] for r in selected
                    if r['key']['noise']==noise and result(r,method).get('status')=='ok']
                ys.append(np.median(values));lo.append(np.quantile(values,.1));hi.append(np.quantile(values,.9))
            ax.loglog(xs,ys,'o-',color=colors[method],label=method.replace('_',' '),markersize=3)
            ax.fill_between(xs,lo,hi,color=colors[method],alpha=.12)
    for ax,title in zip(axes,['Synthetic: 40 independent seeds','Byte-LM: 4 independent checkpoints']):
        ax.set_title(title,fontsize=10);ax.set_xlabel('Relative observation noise (0 at left)',fontsize=9)
        ax.grid(alpha=.2);ax.set_xticks([1e-10,1e-6,1e-2],['0',r'$10^{-6}$',r'$10^{-2}$'])
    axes[0].set_ylabel('Combined relative factor error',fontsize=9)
    axes[1].legend(fontsize=7,loc='best')
    fig.savefig(path,metadata={'CreationDate':None});plt.close(fig)

def diagnostic_table(summary):
    lines=[r'\begin{table}[t]',r'\centering',r'\small',
        r'\caption{Synthetic held-out diagnostic comparison (percent). Coverage is the accepted fraction; bad rejection is recall on errors $>0.10$; good rejection is false rejection on errors $\le0.05$; accepted bad is the bad fraction among accepted outputs. Each threshold was frozen using calibration only.}',
        r'\label{tab:trusted-diagnostics}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Method & Score & Coverage & Bad reject & Good reject & Accepted bad \\',r'\midrule']
    labels={'spectral':'Spectral','projected_spectral':'Projected','joint_fit':'Joint fit'}
    slabels={'empirical_score':'Combined','residual_only_score':'Residual','condition_only_score':'Condition'}
    for method in METHODS:
        for score in SCORES:
            d=summary['groups']['synthetic'][method][score]
            values=' & '.join(f'{100*d[k]:.2f}' for k in ['coverage','bad_rejection_recall','good_false_rejection','accepted_bad_fraction'])
            lines.append(f'{labels[method]} & {slabels[score]} & {values} '+r'\\')
    return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',r'\end{table}'])+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    summary,rows=summarize()
    outputs={ROOT/'results/trusted_recovery/summary.json':json.dumps(summary,indent=2,allow_nan=False)+'\n',
             ROOT/'paper/generated/trusted_recovery_table.tex':table(summary),
             ROOT/'paper/generated/trusted_diagnostics_table.tex':diagnostic_table(summary)}
    for path,text in outputs.items():
        if args.check:
            if path.read_text()!=text: raise ValueError('Derived artifact mismatch: '+str(path))
        else: path.write_text(text)
    if not args.check: figure(rows,ROOT/'paper/figures/trusted_recovery.pdf')
    print(json.dumps({k:summary['groups'][k]['joint_fit']['empirical_score'] for k in SPLITS},indent=2))
if __name__=='__main__': main()
