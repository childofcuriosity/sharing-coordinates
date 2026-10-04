> Historical handoff record (2026-09-29). This working tree was integrated on 2026-10-04; see [INTEGRATION.md](INTEGRATION.md) for current scope. The original handoff manifest is archived at `research/releases/2026-10-04-local-integration/handoff-MANIFEST.sha256`. Exclusion lists and validation below describe the original handoff, not the integrated tree.

# 项目交接说明

本分支是 Sharing Coordinates 的轻量交接版。任务加权研究统一放在 `studies/task-weighted-response/`，不需要第二个项目。

## 阅读顺序

1. `paper/main.pdf` 与 `paper/main.tex`：当前论文。
2. `README.md`、`src/`、`experiments/`、`tests/`：主项目和代码。
3. `studies/task-weighted-response/REPORT.md`：后续研究和负结果。
4. `studies/task-weighted-response/batch-correction/RESULTS.md`：数值校正版。
5. `research/releases/2026-09-29-project-unification/`：目录迁移及校验记录。

## 体积与复现范围

保留论文、源码、测试、协议、冻结记录、结果摘要、小型逐次记录和历史说明。按统一文件类型和体积规则排除数据集、模型检查点、原始梯度/数组、JSONL 原始轨迹、压缩包和超过 5 MiB 的文件；完整路径、大小和 SHA-256 见 `HANDOFF_EXCLUDED_FILES.json`。不按实验结果好坏筛选。

这不是完整实验存档。历史 FREEZE、VALIDATION、子目录 MANIFEST 等保持原义，描述当时的完整实验。它们不能证明当前轻量交接版具备全部复核输入。完整文件清单保存在 `FULL_RESEARCH_MANIFEST.sha256`；根 `MANIFEST.sha256` 描述本交接版实际内容。

在项目根目录核验交接文件：

```bash
sha256sum -c MANIFEST.sha256
```

源码测试按 `pyproject.toml` 安装依赖后运行；首轮和校正版测试分别在各自目录执行 `PYTHONPATH=. python -m pytest -q tests -p no:cacheprovider`，避免同名模块混用。完整实验 verify.py 需要先恢复被排除的原始输入。

## 完整本地版本

交付机器的 `aslora/outputs/sharing-coordinates-local-20260929/` 保存完整研究文件，含两个实验版本的数据、检查点和原始梯度；`LOCAL_VERSION.json` 逐文件记录 SHA-256，并有同名 `.tar.gz` 存档及其 SHA-256。模型下载缓存和 Python 环境仍留在原工作目录，不包含在存档中。

需要完整复核时，使用本地完整快照作为独立工作副本，按 REPRODUCIBILITY.md 准备环境与预训练模型缓存；不要把交接版 README/清单覆盖到历史快照里。当前机器可使用原项目 `.cache/llm/venv/bin/python`。预训练模型版本与字节记录见 research/LLM_MODEL_BYTES.json；这些缓存不在 GitHub 分支上。

此前 2026-09-25 的交付件仅是历史版本。本次没有重跑训练，也没有修改论文结论。

## 本次交接检查

交接目录中实际执行：主项目因子分解、坐标变换、响应可辨识性、响应探测及分组几何共 29 项测试；任务加权首轮与校正版各 5 项测试。合计 39 项通过。没有运行依赖被排除大文件的全套实验复核。详见 HANDOFF_VALIDATION.json。

根目录清单使用 `sha256sum -c MANIFEST.sha256` 核验；原 scripts/write_manifest.py 和子模块清单是完整研究版工具，不用于重新生成本交接清单。
