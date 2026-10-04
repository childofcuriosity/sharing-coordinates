# arXiv 发布前贡献边界复核

核查日期：2026-09-25。范围：理论近邻及最小可陈述贡献；不评定实验充分性，不替代完整证明审计。使用 paper-review 技能。阅读 `paper/main.tex`、`paper/sections/theory.tex`、`paper/sections/related_work.tex`、`paper/references.bib`，并对照 exact proof、quantitative inverse 和 finite-noisy appendix 的关键步骤。既有 `frontier_response_identifiability.md` 只用于寻找入口；以下判定基于本轮重新打开的一手全文。未修改论文。

## 结论

**可以保留一个明确但较窄的模型专属贡献，不能把它包装成新的 gauge 原理、softmax 对称性原理、robust tensor uniqueness 或通用 probing 方法。** 已核查文献没有直接给出当前论文的完整命题：同一有效乘积、正且满列秩的有限 softmax router、满行秩共享 basis、固定且已知的正 Euclidean block rates/temperature，完整瞬时响应相等当且仅当同时置换。该判断是有限范围检索的结果，不是优先权证明。

理论实质是把这些熟知工具接成一个有具体观察模型、具体秩条件和边界反例的逆问题。不能从“组成步骤已知”直接推出“完整命题已发表”；同样，不能仅凭未发现完全相同标题就宣称重大理论突破。可辩护增量在于 **response → Gram/covariance → diagonal family → permutation** 的完整归约及其稳定误差传递，以及从 noisy observed product 构造有限 probes 后对整个可行集的直径控制。

最值得调整的贡献措辞是 robust inverse：**Bhaskara et al. Theorem 5 固定条件参数后已经对误差给出线性界。** 因此，`linear` 这个指数本身不能作为本工作的区别；区别只能是我们这里的观察量、稳定归约、约束消除 scaling、显式端到端常数和有限噪声测量桥梁。

## 五个核心核查对象

### 1. 静态歧义：Vu Thanh, Gillis, Lecron (2023)

来源：[作者 arXiv 全文，Section IV-A，PDF 第 6 页](https://arxiv.org/pdf/2209.12638)，期刊 DOI：[10.1109/TSP.2023.3289704](https://doi.org/10.1109/TSP.2023.3289704)。本轮逐项检查了该节的 W(α)、H(α) 及其逆矩阵公式，不只检查摘要。

该节直接给出 W(δ)=W((1+δ)I−δ11ᵀ/K)，H(δ)=((1+δ)⁻¹I+δ11ᵀ/((1+δ)K))H。令 X=Θᵀ、W=Bᵀ、H=Aᵀ、s=δ/(1+δ)，即得到稿件的 uniformizing path。静态非唯一性是直接包含关系；entropy 后果来自该 uniformizing gauge。该变换将每行坐标差乘以正数 1−s，因此保持坐标排序和全部 argmax ties；argmax partition 的改变需要另一种可行、非均匀的 gauge，不能归给此单条路径。该节后续的 sufficiently-scattered/minimum-volume 结论有额外识别条件，不能移植成严格正 router 的无条件唯一性。稿件当前归属基本准确。

### 2. softmax optimizer symmetry：Lau and Su (2026)

来源：[arXiv v4，Section 3.5，Definitions 3.4–3.5、Proposition 3.5](https://arxiv.org/html/2605.18106v4#S3.SS5)。本轮读取了该节全部主体及命题。

其模型是 p(x;W)=softmax(Wx)，使用 W→PW+1aᵀ 的 expert permutation/shared-logit-shift 对称性，并定义/构造 permutation-equivariant、shift-invariant、horizontal router updates。这直接覆盖“softmax 标签置换及共享平移与 optimizer compatibility 有关”的原理。

但它不是对任意两个有限矩阵分解 AB=A′B′ 的 response-observation converse。它从指定群作用设计兼容 update，而当前命题从一个固定点对的完整响应相等，排除该乘积纤维上所有剩余非置换 gauge。不能把“全域 equivariance 群”与“任意两个 charts 的观测可分性”混为一谈。未在核查段落发现当前完整结论。

### 3. softmax response 公式：Varre, Rofin, Flammarion (2026)

来源：[arXiv v1，Appendix D.1，Lemmas D.1–D.2、Property D.3](https://arxiv.org/html/2603.06248v1#A4.SS1)。本轮核查了梯度流与 Jacobian 公式。

对 β=Vs，其 Lemma D.2 推导已出现 dV/dt=rsᵀ、ds/dt=C(s)²Vᵀr，随后 product rule 给出 ||s||²r+VC(s)²Vᵀr。故本稿 response 的单 router 核心项已有明确先例，多层共享 basis 的 cross-row Gram 项是同一链式法则。

注意原文接下来某行和 M(s,V) 定义省掉平方，而前面的 ds/dt 及 Property D.3 Jacobian 保留支持 C² 的结构；不可无说明抄录省平方后的表达式。这个记号/代数不一致不使 response 原理成为本稿创新。来源讨论 polarization/convergence，所核查段落未给出跨多个满秩 router rows 的同乘积逆识别定理。

### 4. 联合对角化：Afsari (2008)

来源：[作者全文，Theorem 2.3；Section 4、Theorems 4.1–4.2](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf)，[DOI](https://doi.org/10.1137/060655997)。本轮检查了唯一性条件和一阶扰动分析段落。

Theorem 2.3 用 diagonal profiles 的非共线性给出 exact nonorthogonal joint diagonalizer 的唯一性（模 permutation/scaling）；Section 4 给出特定 cost/stationary diagonalizer 的一阶敏感性，依赖 conditioning 和 uniqueness modulus。

稿件一旦得到 Diag(aᵢ)=M Diag(aᵢ′)Mᵀ，A 满列秩就已强于 profiles 两两非共线。最后的 monomial/permutation 识别是这个成熟代数问题的特例。可保留的是此前从 response 恢复 Gram、用 PSD singularity 分离 identity term、消除 basis 并稳定取 covariance square root 的具体归约；不能把最后 diagonal-algebra step 独立列成新发现。

### 5. Robust inverse：Bhaskara, Charikar, Vijayaraghavan (2014)

来源：[COLT 官方 PDF，Theorem 5，PDF 第 7 页](https://proceedings.mlr.press/v35/bhaskara14a.pdf)，[官方记录](https://proceedings.mlr.press/v35/bhaskara14a.html)。浏览器公式抽取不完整，因此另行下载官方 PDF，以 `pdftotext -layout` 核对完整定理。

定理要求 bounded rank-R decompositions 及 robust Kruskal ranks 之和至少 2R+2；其 ε=ε′/(R⁶∏ϑ₅) 推出因子误差≤ε′，模统一 permutation 和 diagonal scalings。因此在条件参数固定时，误差依赖已是线性的，而且比较对象不预设在正确 local branch。

稿件 diagonal tensor 的参考 factors [I,I,A′] 均有 robust Kruskal rank K；K≥2 时 3K≥2K+2。故最后 robust permutation/scaling 环节已有直接适用的通用工具。当前工作的增量不是全局性或线性指数本身，而是从 response discrepancy 稳定得到该 tensor discrepancy，以及把约束/秩界和原始 B 一并纳入最终可检验陈述。显式常数可能有说明价值，但未证明更尖锐或数值可用，不能当作普遍优越性。

## 补充近邻：用于排除过宽表述

| 一手来源与已核查位置 | 本次对照结果 |
|---|---|
| [Singh, arXiv v1，Definition 3.1、Lemma 3.2、Section 4](https://arxiv.org/html/2608.05136v1#S3) | (U,V)→(UM,VM⁻ᵀ) 的全域 Euclidean isometry 当且仅当 M orthogonal；GD equivariance 与 Adam 的坐标依赖已有。它没有本稿 simplex/covariance 观测的 pointwise converse。尤其不能拿全域 isometry iff 直接替代本稿由两点响应相等推出 M orthogonal 的证明。 |
| [Nguyen et al., NeurIPS 2023，Proposition 1、Theorem 1，PDF pp. 5–6，Appendix A.1](https://proceedings.neurips.cc/paper_files/paper/2023/file/0ef6ffcb85a2d238fc4761860c31ded4-Paper-Conference.pdf) | exact-fitted softmax-gating Gaussian MoE 已有 conditional-density Hellinger 与参数距离 D₁ 的线性下界，模 softmax translation。其观察是条件密度函数且有模型特定可识别条件，不是有限静态参数 product 加 optimizer-response oracle。因此不能泛称“首次 softmax identifiability / linear inverse”。 |
| [Halikias and Townsend，Section 2、Definition 2.1、Lemma 2.2、Section 2.1.2](https://arxiv.org/pdf/2212.09841) | 在线性结构族中通过矩阵向量乘法恢复参数；k×k block-diagonal 矩阵用 k 个重复坐标向量恢复。稿件的 K 次 probes 是利用 shared Gram 加局部 block/support 的专门 multiplexing，方法论不是新领域。该文的参数计数下界限定线性参数族，不可自动用来声称本稿 K 是最优 query 数。 |

## 哪部分是拼接，哪部分仍值得陈述

1. **直接已有或很近的标准推导**：静态 SSMF path；Jacobian–Gram response；C² 的 softmax flow；GL factor gauge 与 Euclidean orthogonal subgroup；joint diagonalizer permutation/scaling 唯一性；robust tensor inverse；结构化 matrix-vector probing。
2. **当前可陈述的模型专属定理**：在同一有效乘积 fiber 上，固定 Euclidean response 的完整 stabilizer 恰为置换。必要的桥梁是真正从观察相等导出 diagonal family，而非预先假设群作用保持整个参数空间的 metric。
3. **较具体的后续增量**：用 α、s_A、s_B 等具名非退化条件，追踪 response 到原始 (A,B) 轨道距离的统一常数；在 product noise 下用 observed top-K subspace 构造 probes，以 ||B(I−P_hat)||≤2ε_Θ/s_A 对所有可行解统一控制，得到整个可行集的直径。此结果是前述工具的定量整合，不是新型 robust tensor uniqueness。外部 domain margins、D−K≥L、主动 probes、误差阈值必须与结论一起呈现。

本稿 exact proof 中的 PSD singularity→完整 Gram、C² 的 PSD square root→C、取消 rank-one 部分→diagonal family，给出了清楚的逻辑工作量。因此“只是把文献原定理换符号”不是证据支持的完整评价。另一方面，定理证明使用的机制大多成熟，工作应靠问题设定的价值、完整边界和测量可操作性站住，而非对基础工具做 novelty 包装。

## 建议采用的最小贡献表述

> 对有限宽度、严格正且满秩的 softmax-router/shared-basis 分解，我们刻画同一有效乘积上由已知 Euclidean block optimizer 诱导的完整瞬时响应：该响应恰在同时置换下保持不变。我们将这一模型专属归约定量化，在具名非退化域上控制原始因子到置换轨道的距离，并在 D−K≥L 及规定 product-noise 阈值下，利用从观测乘积构造的 K 个 probes 控制整个可行集的直径。证明利用既有的 factor symmetry、softmax flow、joint diagonalization/robust tensor uniqueness 和 structured probing 思想。

不要加 first、首次、首个全局线性逆、发现 softmax 仅有 permutation symmetry、普遍恢复 sharing structure、恢复真实历史或 AdamW 轨迹。不要把完整 response oracle 的结论直接改写成“一段训练日志即可识别”。

## 核查限制和发布前处理

本轮额外检索了 `softmax optimizer response identifiability permutation Jacobian Gram factorization`、`softmax symmetry optimizer permutation identifiability` 和 robust inverse 近邻；搜索结果没有构成新的、直接相同的主定理来源。该检索不穷尽公开文献。以上核心来源均获得了具体全文段落；Bhaskara 浏览器截图失败后用官方 PDF 文本复核，不把失败截图当证据。

没有重新审核全部 bibliography，也没有本轮逐篇核查 Dorrell、Tran、最新 Amsel probing 定理或 inaccessible Karuturi workshop 全文；不根据旧 frontier 笔记为其作新的排除结论。全文未读完或未可得的近邻仍可能改变贡献评价。未独立重证 manuscript 全部常数，未评定优化求解器成功率。

建议发布前将 Bhaskara 关系表述为“已有 fixed-condition linear robust tensor inverse；本文贡献为 tailored end-to-end response inverse”；其余当前主文的范围限制总体与本轮证据一致。本文档只确认可以谨慎陈述的理论边界，不提供优先权担保或录用预测。


## 最终改稿复核（2026-09-25）

复核基准 HEAD：`64258be6dbbe65107976c210d245b1dbbfe502d4`；这是含未提交修改的工作区审查，不能仅用 HEAD 重建下列文本，因此记录逐文件 SHA-256。

| 已核查文本 | SHA-256 |
|---|---|
| `paper/main.tex` | `d831fefcf5b0709eb29171cbf63c6d9ce8d8a715fc940d925bab1e3574db71b3` |
| `paper/sections/introduction.tex` | `0fa60149308bfdb31346d912ef9d0ba36419a4adc17440d9ea135499d9658d79` |
| `paper/sections/related_work.tex` | `096b44be74e941dcff33e23e1f12f6e793dac919a3db662c03af883a4c6466b7` |
| `paper/sections/theory.tex` | `6d73b4655d92e182ef8bb20f1f318ba1549f7563d78056d66c8322753f2b8a5e` |
| `paper/sections/appendix_literature.tex` | `26105b042523a5dc46448325b893ecd94e0830595e64dca6841861f77b10e6d0` |

结论：新版摘要、引言、related work 和理论主线采用的最小贡献边界与本轮一手核查一致。Bhaskara 固定 conditioning 参数后的线性 inverse 已明确承认；finite-noisy 主张限定于指定域、维度和 product-noise 条件，未将其表述成新的一般 probing 原理，也未声称 query 最优或求解器保证。response flow 定义已移至 `problem.tex`，必要的正 rates/temperature、任意 G 及排除 direct factor penalties 的范围仍保留。引言对 uniformizing 保持 argmax、其他 gauge 可改变 argmax 的区分是正确的。

复核时反馈了以下局部问题；它们不否定核心贡献，但应在最终生成 PDF 前处理：

- `appendix_theory.tex` 开头仍有 `\eqref{eq:vu_ssmf_path}`，而主文删除了唯一对应 label。可在 literature appendix 的既有 W(δ),H(δ) 公式恢复 label，或改成 appendix 引用。
- `appendix_literature.tex` 的 SSMF 段把 local gauge、entropy 和 argmax 一起接在 uniformizing 后面，虽然写的是 “known ambiguity”，仍宜显式区分 entropy 来自 uniformizing、argmax 来自不同非均匀 gauge。本报告已作同样澄清。
- theory 的 “Gram information alone is insufficient” 应说明是一般情形或 K≥3：K=2 时固定全一向量的 orthogonal transformations 只有 I 与 swap。
- 引言把固定 checkpoint 的完整 map 描述为比 “a training trajectory” 更强，若按信息偏序理解并不严格；可改为它与 passive trajectory 属于不同观察、且强于单个 realized gradient。跨时间轨迹可能包含固定点 map 没有的历史信息。

此次是对改稿表述和引用关系的复核；没有新增全世界检索，没有重新核算所有定量常数，也未独立审查新增 Tran/Wang 段落的全部全文条件。上表 SHA 之后的任何编辑应另记最终版本，不得默认为已逐字复核。
