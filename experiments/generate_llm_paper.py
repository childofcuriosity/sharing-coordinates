"""Generate descriptive paper tables/figures from all validated LLM records."""
import csv
import argparse
import json
import tempfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.llm_protocol import ROOT,MODELS,CONFIG
OUT=ROOT/'paper'

NAMES=dict(pythia160m='Pythia 160M',pythia1b='Pythia 1B',pythia28b='Pythia 2.8B',
    qwen06b='Qwen3 0.6B',qwen17b='Qwen3 1.7B',qwen4b='Qwen3 4B',qwen8b='Qwen3 8B',
    smol360m='SmolLM2 360M',smol17b='SmolLM2 1.7B')


def sci(x):
    base,exponent=f'{x:.1e}'.split('e')
    return '$'+base+r'\!\times\!10^{'+str(int(exponent))+'}$'


def table(name,header,body,caption,label,alignment):
    text='\n'.join([r'\begin{table}[t]',r'\centering\small',
        r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+alignment+'}',
        r'\toprule',header+r'\\',r'\midrule',*body,r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (OUT/'generated'/name).write_text(text)


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    temporary=tempfile.TemporaryDirectory(prefix='llm-paper-') if args.check else None
    if temporary:
        OUT=Path(temporary.name);(OUT/'generated').mkdir();(OUT/'figures').mkdir()
    summary=json.loads((ROOT/'results/llm/summary.json').read_text())
    with (ROOT/'results/llm/recovery-cases.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        for key in ('noise','q','seed'):r[key]=float(r[key])
        r['error']=float(r['error']) if r['error'] else None
    def select(**kwargs):return [r for r in rows if all(r[k]==v for k,v in kwargs.items())]
    body=[];costbody=[];adaptbody=[]
    for model in MODELS:
        units=[u for u in summary['units'] if u['model']==model]
        natural=[r for r in select(model=model,method='joint_fit',q=4,noise=1e-4) if r['family'].startswith('natural')]
        design=select(model=model,method='joint_fit',q=4,noise=1e-4,family='designed')
        clean=[r for r in select(model=model,method='joint_fit',q=4,noise=0) if r['family'].startswith('natural')]
        body.append(' & '.join([NAMES[model],str(units[0]['dimensions'][0]),str(units[0]['dimensions'][1]),
            sci(max(r['error'] for r in clean)),sci(max(r['error'] for r in natural)),sci(max(r['error'] for r in design))])+r'\\')
        costbody.append(' & '.join([NAMES[model],f"{units[0]['trainable']/1e6:.3f}",
            f"{np.median([u['training_seconds'] for u in units]):.1f}",
            f"{np.median([u['observation_seconds'] for u in units]):.1f}",
            f"{np.median([u['recovery_seconds'] for u in units]):.1f}",
            f"{max(u['peak_gpu_bytes'] for u in units)/2**30:.2f}"])+r'\\')
        adaptbody.append(' & '.join([NAMES[model]]+[f"{np.mean([u['adapted_nll'][d]-u['baseline_nll'][d] for u in units]):+.4f}" for d in CONFIG['evaluation_domains']])+r'\\')
    table('llm_models_table.tex',r'Model & $L$ & $D$ & Natural, $0$ & Natural, $10^{-4}$ & Designed, $10^{-4}$',body,
        r'Joint-fit factor error with $q=4$: maxima over three adaptation seeds, and (for natural queries) three text domains. Noise affects both product and response. $D$ is the adapter basis dimension, not the backbone parameter count. Each reported maximum includes all declared cases.',
        'tab:llm-models','lrrrrr')
    table('llm_cost_table.tex','Model & Adapter (M) & Train (s) & Observe (s) & Recover (s) & Peak (GiB)',costbody,
        'Per-checkpoint costs: median over three seeds; maximum allocated GPU memory across training and observation. Training excludes loading, evaluation and gradient collection; observation covers the entire noise/query sweep; recovery is single-thread CPU time elapsed for all three estimators. Timings are local measurements under concurrent jobs.',
        'tab:llm-cost','lrrrrrr')
    table('llm_adaptation_table.tex','Model & General & Code & Math',adaptbody,
        'Adapter-induced change in held-out token NLL (adapted minus frozen backbone), averaged over three adapter seeds. Negative is lower. Domains use 32 fixed blocks per checkpoint. Tokenizers differ, so changes are within-model comparisons; these are not answer-accuracy or code-execution scores.',
        'tab:llm-adaptation','lrrr')
    body=[]
    for method in summary['by_method']:
        name={'spectral':'Spectral','projected_spectral':'Projected','joint_fit':'Joint'}[method['method']]
        for score,label in [('residual_only_score','Residual'),('condition_only_score','Condition'),('empirical_score','Combined')]:
            a=method['summary']['acceptance'][score]
            body.append(' & '.join([name,label,str(a['accepted']),f"{100*a['coverage']:.1f}\\%",str(a['bad']),
                '--' if a['accepted_bad_fraction'] is None else f"{100*a['accepted_bad_fraction']:.2f}\\%"])+r'\\')
    table('llm_acceptance_table.tex','Estimator & Score & Accepted & Coverage & Bad & Bad / accepted',body,
        r'Unchanged calibration policies transferred to 1,620 held-out cases per estimator. Coverage includes all 540 insufficient-rank cases at $q=1$; joint fitting is not run there. Bad means permutation-aligned factor error $>0.10$. An empty accepted set has undefined bad fraction. Raw spectral estimates fail the strict simplex gate in calibration, yielding an all-reject policy.',
        'tab:llm-acceptance','llrrrr')
    body=[]
    for model in CONFIG['decision_models']:
        for action,label in [('effective_clustering','Effective clustering'),('random_balanced','Random balanced'),('joint_noise_0.0001',r'Recovered, $10^{-4}$'),('joint_noise_0.01',r'Recovered, $10^{-2}$')]:
            vals=[r for r in summary['decisions'] if r['model']==model and r['action']==action]
            cells=[]
            for d in CONFIG['evaluation_domains']:
                x=[r['after_recovery_nll_delta_native'] for r in vals if r['domain']==d]
                cells.append(f'{np.mean(x):+.4f}')
            body.append(' & '.join([NAMES[model],label,*cells])+r'\\')
    table('llm_decision_table.tex','Model & Action & General & Code & Math',body,
        r'Post-recovery NLL minus native-router partition, averaged over three seeds. All actions use four balanced groups, a common effective-tensor refit and 64 matched updates. Both recovered and effective-clustering partitions equal the native partition at all nine checkpoints. Small Pythia discrepancies under equivalent partitions are numerical execution variation, not structural improvements. Per-seed values and immediate losses are retained in the released CSV.',
        'tab:llm-decisions','llrrr')
    plt.rcParams.update({'font.size':10.5,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(8.2,2.8),sharey=True)
    families=[('designed','Designed'),('gaussian','Gaussian'),('natural','Natural (three domains)')]
    colors=['#0072B2','#D55E00','#009E73']
    for ax,(family,title) in zip(axes,families):
        for color,(method,label) in zip(colors,[('spectral','Spectral'),('projected_spectral','Projected'),('joint_fit','Joint')]):
            med=[];lo=[];hi=[]
            for noise in CONFIG['noise']:
                vals=[r['error'] for r in select(method=method,q=4,noise=noise) if r['family'].startswith(family)]
                med.append(np.median(vals));lo.append(np.quantile(vals,.1));hi.append(np.quantile(vals,.9))
            ax.plot(range(4),med,'o-',color=color,label=label,lw=1.3,ms=3)
            ax.fill_between(range(4),lo,hi,color=color,alpha=.13)
        ax.set_yscale('log');ax.set_xticks(range(4),['0',r'$10^{-6}$',r'$10^{-4}$',r'$10^{-2}$'])
        ax.set_title(title);ax.set_xlabel('Relative observation noise');ax.grid(alpha=.2)
    axes[0].set_ylabel('Factor error (aligned)');axes[-1].legend(fontsize=9,loc='lower right')
    fig.tight_layout();fig.savefig(OUT/'figures/llm_recovery.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    stress=json.loads((ROOT/'results/llm/conditioning-stress.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(8.2,2.8),sharey=True)
    for ax,axis,title in zip(axes,['router_rank','router_interior','basis_rank'],['Router rank margin','Router interior margin','Basis rank margin']):
        for color,model in zip(colors,CONFIG['decision_models']):
            r=[r for r in stress['records'] if r['model']==model and r['axis']==axis and r['noise']==1e-4]
            ax.loglog([v['level'] for v in r],[v['methods']['joint_fit']['evaluation']['orbit_relative_error'] for v in r],'o-',color=color,label=NAMES[model],ms=3)
        ax.set_title(title);ax.set_xlabel('Retained margin scale');ax.invert_xaxis();ax.grid(alpha=.2)
    axes[0].set_ylabel('Joint-fit factor error');axes[-1].legend(fontsize=9)
    fig.tight_layout();fig.savefig(OUT/'figures/llm_conditioning.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
    if temporary:
        for path in OUT.rglob('*'):
            if path.is_file() and path.read_bytes()!=(ROOT/'paper'/path.relative_to(OUT)).read_bytes():
                raise ValueError('Generated paper artifact differs: '+str(path.relative_to(OUT)))
        temporary.cleanup()


if __name__=='__main__':main()
