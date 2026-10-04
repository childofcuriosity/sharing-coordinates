"""Report every fixed unit/method/outcome; no best-method or endpoint selection."""
import csv,json
import numpy as np
from common import ROOT,read,write,check_freeze,sha
METHODS=['weight','isotropic_response','task_response','task_single','activation','direct_loss','random']
MODELS=['pythia160m','qwen06b','smol360m']

def main():
    check_freeze();execution=read(ROOT/'EXECUTION.json');rows=[];units=[];bootstrap={}
    for model in MODELS:
        differences=[];doc_hashes=None
        for seed in range(3):
            folder=ROOT/'results'/f'{model}-seed{seed}';s=read(folder/'selection.json');e=read(folder/'evaluation.json')
            assert e['selection_sha256']==sha(folder/'selection.json')
            hashes=[d['document_sha256'] for d in e['data']['documents']]
            if doc_hashes is None:doc_hashes=hashes
            assert doc_hashes==hashes
            for method in METHODS:
                index=s['selected'][method]['index'];a=e['actions'][str(index)]
                for opt in ['sgd','adamw']:
                    z=a['recovery'][opt]
                    rows.append(dict(model=model,seed=seed,method=method,optimizer=opt,partition=index,
                        soft_nll=float(np.mean(e['soft_document_nll'])),immediate_nll=float(np.mean(a['immediate_document_nll'])),
                        final_nll=float(np.mean(z['document_nll'])),heldout_response_relative_mse=a['heldout_response_relative_mse'],
                        standalone_selection_seconds=s['cost']['standalone_selection_seconds'][method],
                        recovery_seconds=z['training_seconds'],recovery_tokens=z['prediction_tokens'],
                        shared_execution_aliases='|'.join(a['method_aliases'])))
            unit=dict(model=model,seed=seed,comparisons={})
            task=e['actions'][str(s['selected']['task_response']['index'])]
            for comparator in ['weight','activation','direct_loss','isotropic_response','task_single','random']:
                control=e['actions'][str(s['selected'][comparator]['index'])]
                entry={'same_partition':s['selected']['task_response']['index']==s['selected'][comparator]['index'],
                    'immediate_nll_difference':float(np.mean(task['immediate_document_nll'])-np.mean(control['immediate_document_nll'])),
                    'heldout_response_error_ratio':task['heldout_response_mse']/control['heldout_response_mse'],
                    'selection_seconds_difference':s['cost']['standalone_selection_seconds']['task_response']-s['cost']['standalone_selection_seconds'][comparator]}
                for opt in ['sgd','adamw']:
                    entry[opt+'_nll_difference']=float(np.mean(task['recovery'][opt]['document_nll'])-np.mean(control['recovery'][opt]['document_nll']))
                unit['comparisons'][comparator]=entry
            units.append(unit)
            weight=e['actions'][str(s['selected']['weight']['index'])]
            differences.append(np.array(task['recovery']['sgd']['document_nll'])-weight['recovery']['sgd']['document_nll'])
        values=np.mean(differences,axis=0);rng=np.random.default_rng(840000+MODELS.index(model))
        sampled=values[rng.integers(len(values),size=(5000,len(values)))].mean(1)
        bootstrap[model]=dict(mean_difference=float(values.mean()),conditional_document_bootstrap_95=list(map(float,np.quantile(sampled,[.025,.975]))),
            scope='32 articles resampled jointly across three fixed checkpoints; not a model/seed population interval')
    with (ROOT/'results/all_methods.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for comp in ['weight','activation','direct_loss','isotropic_response','task_single','random']:
        cs=[u['comparisons'][comp] for u in units];diff=np.array([c['sgd_nll_difference'] for c in cs])
        summary[comp]=dict(sgd_mean_nll_difference=float(diff.mean()),adamw_mean_nll_difference=float(np.mean([c['adamw_nll_difference'] for c in cs])),
            sgd_lower_nll=int((diff<0).sum()),sgd_exact_ties=int((diff==0).sum()),sgd_higher_nll=int((diff>0).sum()),
            sgd_improved_by_at_least_0005=int((diff<=-.005).sum()),sgd_worsened_by_at_least_0005=int((diff>=.005).sum()),
            same_partition=sum(c['same_partition'] for c in cs),heldout_response_better=sum(c['heldout_response_error_ratio']<1 for c in cs))
    actual_recovery=sum(a['recovery'][o]['training_seconds'] for m in MODELS for s in range(3) for a in read(ROOT/'results'/f'{m}-seed{s}'/'evaluation.json')['actions'].values() for o in ['sgd','adamw'])
    costs={method:dict(selection_mean_seconds=float(np.mean([r['standalone_selection_seconds'] for r in rows if r['method']==method and r['optimizer']=='sgd'])),
        sgd_recovery_mean_seconds=float(np.mean([r['recovery_seconds'] for r in rows if r['method']==method and r['optimizer']=='sgd']))) for method in METHODS}
    result=dict(scope='All fixed 9 checkpoints, 7 methods and 2 recovery optimizers; exploratory existing-model study',comparisons=summary,units=units,conditional_intervals=bootstrap,costs=costs,
        actual_unique_recovery_seconds=actual_recovery,wall_makespan_seconds=execution['makespan_seconds'],summed_job_elapsed_seconds=execution['per_job_elapsed_seconds'],
        caveat='Neither makespan nor sum of elapsed seconds is measured GPU active time; aliases share physical execution.',
        freeze_sha256=sha(ROOT/'FREEZE.json'),selection_lock_sha256=sha(ROOT/'SELECTION_LOCK.json'))
    write(ROOT/'results/summary.json',result)
    lines=['# 完整结果：任务加权响应对照','', '所有差值均为 task_response 减对照；NLL 越低越好。未按模型、种子、优化器或终点挑选正结果。','',
        '| 对照 | SGD 平均 NLL 差 | AdamW 平均 NLL 差 | SGD 改善/相同/变差 | 改善至少0.005 | 同分组 | 测试响应更好 |',
        '|---|---:|---:|---|---:|---:|---:|']
    for name,z in summary.items():lines.append(f"| {name} | {z['sgd_mean_nll_difference']:+.6f} | {z['adamw_mean_nll_difference']:+.6f} | {z['sgd_lower_nll']}/{z['sgd_exact_ties']}/{z['sgd_higher_nll']} | {z['sgd_improved_by_at_least_0005']}/9 | {z['same_partition']}/9 | {z['heldout_response_better']}/9 |")
    lines += ['', '## 每个检查点的主比较（对 weight）','', '| 模型 | 种子 | 即时NLL差 | SGD后NLL差 | AdamW后NLL差 | 测试响应误差比 |','|---|---:|---:|---:|---:|---:|']
    for u in units:
        z=u['comparisons']['weight'];lines.append(f"| {u['model']} | {u['seed']} | {z['immediate_nll_difference']:+.6f} | {z['sgd_nll_difference']:+.6f} | {z['adamw_nll_difference']:+.6f} | {z['heldout_response_error_ratio']:.4f} |")
    lines += ['', '## 条件评估区间','', '每模型三种子共同重采样32篇测试文章；不表示模型或训练总体的不确定性。']
    for m,z in bootstrap.items():lines.append(f"- {m}: {z['mean_difference']:+.6f}, 95%区间 {z['conditional_document_bootstrap_95']}")
    lines += ['', '## 成本','', '| 方法 | 独立运行的平均选择秒数 | 单次平均SGD恢复秒数 |','|---|---:|---:|']
    for m,z in costs.items():lines.append(f"| {m} | {z['selection_mean_seconds']:.3f} | {z['sgd_recovery_mean_seconds']:.3f} |")
    lines += ['',f"全流程并发 makespan {execution['makespan_seconds']:.2f}s；各任务elapsed之和 {execution['per_job_elapsed_seconds']:.2f}s；实际独特动作的恢复计时之和 {actual_recovery:.2f}s。均不冒充测得的GPU active time。",
        '', '完整126行结果、所有选择候选/得分、逐文档损失、训练记录及阶段计时保留在 results/。同分组复用同一动作结果，成本栏同时给独立运行所需的测量成本。',
        '', '这份表是事实汇总，不自动声称任务方法有实用收益。最终解释见 REPORT.md。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
