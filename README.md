# Sharing Coordinates

**Balanced Factorizations and the Interpretation of Shared Weights**

[Current paper](paper/main.pdf) | [LaTeX source](paper/main.tex) | [Evidence and scope](research/BALANCED_MANUSCRIPT_EVIDENCE.md)

准备在博士申请后恢复研究时，先读[中文接续记录](research/BALANCED_MANUSCRIPT_EVIDENCE.md)：其中保存了研究动机、已有二阶段结果、可视化文献核查与研究人员讨论计划，以及下一步概率问题。

This project asks whether initialization and optimization help make shared-weight interpretations reliable, even though the same effective weights admit different coefficient heatmaps. The current manuscript develops the balance relation and records the human reasoning that redirected the investigation toward actual training and the adequacy of existing practice.

## The mathematical result

Write layer weights as $W=DC$: columns of $D$ are shared bases and columns of $C$ describe individual layers. At balance, $D^\top D=CC^\top$ and $C^\top C=(W^\top W)^{1/2}$, where the latter is a positive semidefinite matrix square root.

The first identity equates dictionary-column geometry with coefficient-row geometry. The second determines coefficient-column geometry from the effective weights. Orthogonal changes of basis can change individual heatmap cells while preserving these relations. Plotted coefficients remain signed values.

The paper starts with a unique optimal effective product and studies its balanced factorizations. Equal-rate Euclidean gradient flow preserves balance when initialized at balance. This classical result does not imply that arbitrary initialization, SGD, or AdamW always converges to exact balance. For fixed matrix observations with isotropic residual covariance, the paper also derives genuine second-stage SGD attraction to weighted balance on the small-step time scale. From any entry point on the optimal fiber, this gives an asymptotic high-probability statement and the same coefficient geometry up to a known scale. Transfer to practical schedules and downstream allocation benefits remains future work.

This is a working theoretical manuscript using established balance theory and elementary matrix arguments. Technical novelty and a new application benefit have not been established. It examines the actual claims and parameterizations of MASA, ASLoRA, NPAS, and SuperWeights.

## Build the current paper

From `paper/`, run twice:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The active text is [balanced_stage.tex](paper/sections/balanced_stage.tex). Updates replace the existing manuscript; Git history records earlier versions.

## Earlier exploratory work

The repository also contains optimizer-response recovery, sharing-decision experiments, and ASLoRA investigations from earlier stages. These records are retained for traceability. They are not empirical evidence for the current manuscript's open natural-training probability question. Earlier section files are not included by the current main.tex.

- [Reproduction guide](REPRODUCIBILITY.md) and [integration notes](INTEGRATION.md) describe the earlier computational artifacts.
- [Task-weighted follow-up](studies/task-weighted-response/REPORT.md) reports the comparison with downstream grouping decisions.
- [ASLoRA protocol](research/ASLORA_EMPIRICAL_PROTOCOL.md) describes the local reimplementation study.
- `src/`, `experiments/`, `tests/`, and `results/` contain corresponding code and records; `research/` contains derivations and source audits.

Earlier validation and release records apply to the versions they name.
