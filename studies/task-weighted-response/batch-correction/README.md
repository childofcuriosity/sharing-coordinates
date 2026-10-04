# 直接批量观测校正版

这是父目录首轮实验的明确数值修订，动机、触发检查及冻结时间见 [修订记录](../research/CORRECTION_LOG.md)。完整协议见 [PROTOCOL.md](PROTOCOL.md)；完整结果在结束后写入 [RESULTS.md](RESULTS.md)，科学解释统一见 [父报告](../REPORT.md)。

代码、协议和分析在本版选择/评估前再次冻结，所有模型、种子、数据职责、对照、恢复预算及主比较保留。统一数学 SDPA；实际测量四文档批量梯度。不是新的一组独立检查点。

从本目录核验：

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHONPATH=. ../../../.cache/llm/venv/bin/python -m pytest -q tests -p no:cacheprovider
../../../.cache/llm/venv/bin/python verify.py
```

重新运行必须使用同层级的新实验目录、空结果目录及匹配的冻结父项目，不能覆盖已有输出。`common.py` 定位统一项目根目录 `sharing-coordinates`。新实验须建立自己的冻结记录，不能复用本历史实验的冻结身份。`launch.py` 使用 cuda:3/4/5，顺序执行每个模型的三个种子；全部选择锁定后才执行评估。所有 GPU 同步计时记录在逐单位 JSON 中。随机选择时间未单独测量，表中的0不表示免费。
