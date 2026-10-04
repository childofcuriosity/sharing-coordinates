from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1];out=root/'batch-correction';out.mkdir()
for name in ['tests','research','results','logs']:(out/name).mkdir()
for name in ['common.py','weighted.py','run_study.py','launch.py','analyze.py','verify.py','THEORY.md','PROTOCOL.md']:
 shutil.copyfile(root/name,out/name)
shutil.copyfile(root/'tests/test_weighted.py',out/'tests/test_weighted.py')
shutil.copyfile(root/'research/DESIGN_REVIEW.md',out/'research/DESIGN_REVIEW.md')
p=out/'common.py';p.write_text(p.read_text().replace("PARENT=ROOT.parents[1]","PARENT=ROOT.parents[2]"))
p=out/'launch.py';p.write_text(p.read_text().replace('ROOT,write,sha,check_freeze','ROOT,PARENT,write,sha,check_freeze').replace("PARENT/'.cache/llm/venv/bin/python'","PARENT/'.cache/llm/venv/bin/python'"))
p=out/'run_study.py';s=p.read_text()
s=s.replace('def collect_gradients(model,bank,blocks,theta,a,b,device,out):','def collect_gradients(model,bank,blocks,theta,a,b,device,out,batch_size=4):')
s=s.replace('start=now();n=len(blocks);gs=[];ls=[]','''start=now();gs=[];ls=[]
    if batch_size==4:
        rng=np.random.default_rng(850000+len(blocks))
        batch_ids=[ids.tolist() for _ in range(64//len(blocks)) for ids in np.split(rng.permutation(len(blocks)),len(blocks)//4)]
    else:batch_ids=[[i] for i in range(len(blocks))]
    n=len(batch_ids)
    document_nll=evaluate(model,bank,blocks,device,theta)''')
s=s.replace('for block in blocks:\n        independent=', 'for ids in batch_ids:\n        independent=')
s=s.replace('block[None].to(device),independent','blocks[ids].to(device),independent')
s=s.replace("{'gradients':torch.stack(gs),'losses':ls,'arithmetic'", "{'gradients':torch.stack(gs),'losses':ls,'batch_indices':batch_ids,'batch_size':batch_size,'arithmetic'")
s=s.replace('return raw,ls,duration','return raw,document_nll,duration')
s=s.replace("v,activation_seconds=activation_moments", "single_raw,_,single_gradient_seconds=collect_gradients(model,bank,blocks,theta,a,b,args.device,out/'selection_single_gradients.pt',batch_size=1)\n    v,activation_seconds=activation_moments")
s=s.replace("'task_response':lambda:moment_costs(raw,4)[0]", "'task_response':lambda:moment_costs(raw,1)[0]")
s=s.replace("'task_single':lambda:moment_costs(raw,1)[0]", "'task_single':lambda:moment_costs(single_raw,1)[0]")
s=s.replace("for k in ['task_response','task_single']:independent_cost[k]+=gradient_seconds", "independent_cost['task_response']+=gradient_seconds\n    independent_cost['task_single']+=single_gradient_seconds")
s=s.replace('gradient_response_seconds=gradient_seconds,activation_seconds=', 'gradient_response_seconds=gradient_seconds,single_gradient_seconds=single_gradient_seconds,activation_seconds=')
s=s.replace('gradient_forward_backward_calls=16,activation_forward_calls=4', 'gradient_forward_backward_calls=16,single_gradient_forward_backward_calls=16,soft_reference_forward_calls_per_collection=4,activation_forward_calls=4')
s=s.replace('prediction_tokens_per_selection_pass=2048', 'prediction_tokens_per_selection_pass=2048,prediction_tokens_in_batch_gradient_collection=8192')
s=s.replace('task,w=moment_costs(raw,4);single,_=moment_costs(raw,1)', 'task,w=moment_costs(raw,1)')
s=s.replace('task_response=task,task_single=single,**raw','task_response=task,**raw')
s=s.replace('task[index]/batch_moments(**raw,batch=4)[2]', 'task[index]/batch_moments(**raw,batch=1)[2]')
s=s.replace('evaluation_gradient_calls=32', 'evaluation_gradient_calls=16,evaluation_soft_reference_forward_calls=8')
s=s.replace('(select if args.phase==\'select\' else assess)(args,out)', "with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):\n            (select if args.phase=='select' else assess)(args,out)")
p.write_text(s)
p=out/'verify.py';s=p.read_text().replace('alpha=(n-4)/(4*(n-1))','alpha=1. # Actual measured four-document batch gradients; no synthetic batch recombination.')
p.write_text(s)
p=out/'PROTOCOL.md';s=p.read_text();s='''# 数值校正版：实际批量梯度与数学注意力后端

校正原因是首轮预先定义的数值检查，而不是 NLL 结果。默认 SDPA 下 Pythia 单文档梯度平均与直接四文档梯度差35.87%；数学后端0.144%，高精度近数值零。该差异表明不能假设当前 FP32 单文档观测精确重构批量执行。首轮所有选择、评估和成本保留于父目录。本版在查看首轮性能汇总之前冻结，仍属探索性修订，不是全新独立盲测。

本版在选择、评估、恢复的每次 forward/backward 中显式固定 SDPA 数学后端。任务主方法直接收集实际 batch=4 的平均损失梯度：选择16篇，每轮随机排列成4批，重复4轮，共16个批量梯度；测试32篇每轮8批，重复2轮，共16个批量梯度。排列 RNG 固定为850000+文档数，所有种子/模型使用相同索引规则。每篇选择文档参与4个梯度批，测试文档参与2个，均不当作独立新文档。经验目标直接平均这16个批量上的误差，不再调用单文档到四文档的统计重构公式。THEORY.md 的公式仍数学成立，但不作为数值执行完全一致的保证。

`task_single` 另测16个单文档梯度作敏感性，单独记录它自己的前向/反向成本；不复用为主方法的批量观测。原16/32文档切分、所有检查点、7个方法、64候选直接损失控制、refit、64步两种恢复优化器、主终点和0.005幅度参考均不变。所有9个校正版选择再次锁定后才打开本版测试评估。测试NLL与软参照按每次4篇评估。额外原软模型参照前向的次数/时间单列在测量成本中。

下面继承原设计的其余规定；涉及任务主方法的单文档重构描述以以上明确修订为准。

'''+s
s=s.replace('16 篇的梯度/软响应矩，经 b=4 无放回批量修正，枚举最小化 THEORY.md 的 F。','直接测量16个四文档批量的梯度/软响应矩，枚举最小化 THEORY.md 的 F。')
s=s.replace('同一测量，使用 b=1 目标；批量定义敏感性，不替代主方法。','另测16个单文档梯度，使用 b=1 目标；批量定义敏感性，不替代主方法。')
p.write_text(s)
p=out/'research/DESIGN_REVIEW.md';p.write_text(p.read_text()+'''\n\n校正补充：真实模型的批量梯度一致性检查暴露 Pythia 默认 SDPA 数值问题，触发统一数学注意力后端和直接 batch=4 梯度测量。此修订不按损失排序作出。保留首轮，不用只呈现校正版来隐藏失败；新增成本与首轮研究成本都记录。推导的有限池组合公式与浮点程序的精确等价必须区分。校正版仍只有16/32篇文章，重复组成批次不增加独立样本数。\n''')
print(out)
