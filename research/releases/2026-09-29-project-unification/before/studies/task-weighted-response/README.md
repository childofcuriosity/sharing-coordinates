# Task-weighted sharing response — 独立研究扩展

本目录与 `../sharing-coordinates` 并列。原论文、发布包、源代码和冻结结果保持只读；开始/结束均核对其原 manifest。这里的推导与实验不是原论文已发表的新结论，也没有自动合入稿件。

**已完成：** 首轮及数值校正版各9个检查点、7种方法、2种恢复优化器，完整结果均保留。优先阅读 [综合报告](REPORT.md) 和 [校正版结果](batch-correction/RESULTS.md)。校正版未显示稳定实用收益：相对权重聚类，SGD后平均NLL差为 +0.005224（更差），3好/2同/4差。不能将两版合并为18个独立复现。

数值修订原因见 [修订记录](research/CORRECTION_LOG.md)，校正版复核入口见 [校正版说明](batch-correction/README.md)。[VALIDATION.json](VALIDATION.json) 汇总公式、数据分离、执行及原发布保护检查；`MANIFEST.sha256` 单独封存整个新增研究。

- [推导](THEORY.md)：任务梯度分布下的精确平衡分组目标、批量修正和条件性泛化边界。
- [固定协议](PROTOCOL.md)：选择/评估分离、七个对照、两个恢复优化器、成本和无收益规则。
- [设计审查](research/DESIGN_REVIEW.md)：使用 scientific-critical-thinking，注明 Scientific Agent Skills 来源。
- [精确反例](research/EXACT_TASK_EXAMPLE.md)：更好地匹配当前响应也可能意味着抵消更新，不保证任务改善。
- [完整结果](RESULTS.md)及[研究解释](REPORT.md)：必须在全部固定单位结束后生成；运行中不以部分结果下结论。
- `results/all_methods.csv` 保留全部 9×7×2=126 行方法结果；方法同分组时明确共享同一次动作。
- `FREEZE.json` 固定研究代码、协议、原发布和输入；`SELECTION_LOCK.json` 在九个选择全部结束后锁定，之后才打开 test；`ANALYSIS_PLAN_LOCK.json` 记录分析代码在评估之前固定。

## 检查现有结果

从本目录运行（所需模型和 Python 环境位于原项目）：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=. ../sharing-coordinates/.cache/llm/venv/bin/python -m pytest -q tests -p no:cacheprovider
../sharing-coordinates/.cache/llm/venv/bin/python verify.py
../sharing-coordinates/.cache/llm/venv/bin/python ../sharing-coordinates/scripts/write_manifest.py --check
```

`verify.py` 核对源/原发布绑定、数据职责、候选最优性、所有动作与记录哈希，并从原始任务梯度直接构造软/硬更新差，独立核查任务目标。每个真实检查点额外用 reverse-mode gradient + forward JVP 检查响应公式。核验无需训练或 GPU。

## 重新运行而不覆盖

`launch.py` 使用 cuda:0、1、2，每个模型依次运行种子0、1、2；所有选择阶段结束并封锁后才执行评估。输出使用独占创建，拒绝覆盖。完整重跑应在一个新的并列研究目录内，复制顶层 `.py`、THEORY.md、PROTOCOL.md、FREEZE.json、ANALYSIS_PLAN_LOCK.json、tests/test_weighted.py 和 research/DESIGN_REVIEW.md，创建空的 results/ 与 logs/，保留相同的 `../sharing-coordinates` 输入位置。然后执行：

```bash
../sharing-coordinates/.cache/llm/venv/bin/python launch.py
../sharing-coordinates/.cache/llm/venv/bin/python analyze.py
../sharing-coordinates/.cache/llm/venv/bin/python verify.py
```

若原项目已经修改，冻结检查会拒绝执行，应使用对应版本的输入，不能简单刷新哈希绕过。上游预训练权重与数据已经按原库存 hash 核对，未复制入本目录。

## 解释成本

`standalone_selection_seconds` 给出每个方法单独选择需要的前置测量及搜索时间，排除共同模型加载/数据处理（另列）。即使同分组共享一次恢复执行，也保留单次独立恢复成本。随机抽一个分组的微小 CPU 时间未单独测量，记录的0不是“零计算成本”。完整同步阶段计时、候选数、前向/反向调用数、token数及峰值 allocated 显存均保留。

预训练骨干从缓存 BF16 数值统一转成 FP32；任务测量/恢复关闭 autocast 和 TF32。任务响应在 FP64 下由原生因子计算，不是黑箱因子恢复实验。长文章 tokenizer 的长度警告仅来自切块前的全文编码，实际模型输入固定128个位置，未把整篇文章传入模型。
