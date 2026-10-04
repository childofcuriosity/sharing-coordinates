# 2026-09-29 项目统一

本研究由 `aslora/task-weighted-response/` 移入 `sharing-coordinates/studies/task-weighted-response/`，现在属于同一个项目。主 README 是统一入口。

只修改父项目路径、启动解释器路径、说明及冻结校验兼容逻辑；论文、数值算法、模型、结果、FREEZE.json、选择锁和分析锁未修改。历史报告中的“独立保存”“原项目只读”描述当时的实验操作，不再表示需要维护第二个项目。

迁移前源码和清单保存在 `../../research/releases/2026-09-29-project-unification/`。其中 MIGRATION.json 同时记录修改前后哈希。check_freeze() 对修改的源码核查原始归档与当前迁移版本，其余冻结源码及模型、数据仍按原哈希验证。父项目旧 MANIFEST 身份由归档保留；当前项目完整性由根目录 scripts/write_manifest.py --check 验证。历史 FREEZE 不被刷新成新的实验声明。

从本目录执行 README 中的检查；批量校正版在 batch-correction/ 中独立检查，避免同名 Python 模块互相覆盖。新实验应创建新的协议、冻结与输出目录。

aslora/outputs/ 中的 2026-09-25 交付件保留为历史快照，不是当前开发项目；其中的旧路径仅适用于当时快照。
