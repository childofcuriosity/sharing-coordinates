# Current manuscript: evidence and chapter plan

## 两个月后回来，从这里接着做

记录日期：2026-10-09。用户预计接下来约两个月主要忙于博士申请，暂时可能无暇推进。本节保存当前判断与待办，方便恢复；不是要求在两个月后自动启动任务或联系别人。

### 我们到底想回答什么

共享少量参数基，组合出更多层的参数：$W=DC$。用户最初关心这种节省参数的实际用法，以及论文里用组合系数热力图解释层与基的做法。构造等价分解，确实能让热图大变而有效权重不变，但这本身没有回答自然训练中是否容易出问题。

用户选定的方向是 implicit bias：初始化和优化共同偏好哪些分解，以及支持某种解释的区域占多少实际概率质量。目标是解释已有做法为什么够用、什么时候不够用，不是先决定热图不可靠，再寻找反例。

当前直觉是：热图是在学到的坐标系里展示 W。某些读法反映 W 的稳定几何；某些读法只描述本次学到的基如何被使用；再赋予一个基某种功能含义，则需要对应证据。不能预先说功能解释都是牵强附会，也不能仅凭一行特别亮认定功能成立。究竟哪些解释值得保障，要从作者真正说了什么出发。

### 已有结果：不要恢复时又丢掉

本阶段先接受训练达到唯一最优有效权重 W_star 的设定，研究其分解流形上的选择。唯一性确定产品，成功达到最优是本阶段的前提；不把第一阶段全局收敛另立成当前必须解决的任务。

论文第 4.2 节已有实质的第二阶段结果：固定矩阵数据、各向同性残差协方差、最小因子宽度、普通同学习率 SGD 下，在 t=n eta^2 的小步长极限中，目标分解流形上的漂移降低

    Psi(D,C) = L ||D||_F^2 + m ||C||_F^2,

从每个流形入口趋向加权平衡集合

    L D^T D = m C C^T.

已有渐近概率量词：给定入口和 epsilon、theta，先选足够大的有限慢时间 T，再取足够小的步长 eta，可使距离该集合小于 epsilon 的概率超过 1-theta。它不是指定实际步长的数值保证，但也绝不能被缩写成“平衡只是个假设，SGD 选择尚未研究”。完整推导在当前论文第 4.2 节；历史详细记录在本机 `research/SECOND_STAGE_SELECTION.md` 第 8 节（该历史文件未随本次论文提交上传，不能把它当作 GitHub 必备依赖）。

普通平衡下 C^T C=(W^T W)^(1/2)，且 D^T D=C C^T；加权平衡只有已知整体比例差异。因此同一 W 的所选分解具有相同的层系数几何，基列与系数行也有对应关系。这里是矩阵平方根，热图仍读原始有符号系数，不是逐元素平方。

剩余表示可写为 C(Q)=a Q Sigma^(1/2) V^T。Q 仍可能改变热图外观；已证明的几何关系在整个平衡族上成立，不需要先证明 Q 均匀。更慢旋转的历史专项记录在本机 `research/SGD_SLOW_ROTATION_THEOREM.md`；如要使用其结论，回查原始推导和条件，不能把时间尺度直接当作混合时间或现实训练的均匀分布保证。

### 还要做什么，以及为什么

**先把可视化的研究需求弄清楚。** 继续读相关可视化与解释性研究，不能只围绕矩阵分解理论找文献。对每个相关案例，保存原文段落、图与图注、具体版本和位置，并紧挨着给中文解读：作者展示了什么量、声称说明了什么、用什么证据支持、是否据此采取了某个操作。分别辨认几何相似性、当前基的使用模式、功能含义；这只是帮助准确理解，不要求作者只能属于其中一类。

现有入口为论文第 5 节及其原文：MASA 的系数与基相似性图、NPAS 的系数或嵌入聚类、ASLoRA 的历史矩阵距离与共享分配图、SuperWeights 的梯度相似性。不要混淆行/列所代表的对象，也不要把“原文一笔带过”解释成作者看不懂或迫切需要我们的理论。下一轮还应寻找直接讨论参数/系数可视化有效性的工作，检查是否已有其他技术路线回答同类问题。

**与相关研究人员讨论方向和价值。** 在整理出具体案例后，准备简短讨论材料：原文实际解释、我们的相关数学对应、尚未解决的概率问题。交流要弄清：这种解释是否是他们真正使用或关心的；他们凭什么信任它；如果有等价热图，是否会改变他们的判断；怎样的理论或实证结果会改变其使用方式。保留“现有读法根本不受影响”或“这个保证增益很小”的可能。目的不是让对方认可既定选题，而是判断是否值得继续、问题该怎么定。尚未选定收件人，也未获得发送授权；只先准备邮件草稿，由用户审阅后自行发送或明确授权发送。

**再为实际有意义的解释计算概率。** 把具体解释写成成功条件 E(W,C)，研究

    P_{Q ~ mu_train}[E(W_star, C(Q))] >= 1-theta.

mu_train 来自指定初始化、抽样、优化器与训练终点。旋转反例只能标出失败点，不能估计失败集合的概率质量；不能用“不旋转不变”代替“不可靠”的论证。可以先在均匀旋转条件模型下计算模式占比，但应明确这是条件结果，并研究它和训练分布的联系。图案常见也不等于赋予它的功能解释正确。

研究顺序仍按用户确定的路线：先检查第二阶段是否决定剩余分布；若能决定，就不必先解第一阶段的入口分布；若仍依赖入口，再回头研究初始化如何把概率质量送到各处。不要因为熟悉初始化理论就重新把主线改成第一阶段。实际方法的约束、系数生成网络、优化器和残差统计需要保留，不能为了证明好看换成无约束模型后声称已解释原应用。

### 恢复工作的最短入口

先读本节，再读当前论文第 3 节、第 4.2 节和第 7 节，回查原始对话中用户的研究动机与纠偏；摘要只用于定位。然后完成一个具体应用案例的“原文—中文理解—实际判断—现有证据—我们的关系”记录，据此准备给相关研究者的讨论草稿。此时再决定下一个定理应该保障什么，不急着扩大模型、补实验或追求发表叙事。若原始对话不可访问，明确缺口，用可核验推导恢复，不假装已经回顾。

本记录中，研究动机、概率标准、第二阶段优先及联系研究人员评估价值来自用户明确要求；具体成功条件和适用训练分布仍待选定，未宣称已完成。

Updated 2026-10-09. Active manuscript: `paper/main.tex`, including `paper/sections/balanced_stage.tex`. This is a working ledger, not another manuscript version.

## Evidence map

| Claim | Evidence | Scope |
| --- | --- | --- |
| Research motivation and human interventions | Author's conversation, recorded in the research-history section | Attribution of direction and corrections; no general conclusion about AI capability |
| Balance conservation | Du, Hu, Lee (2018), Theorem 2.2; differentiation in text | Equal-rate Euclidean flow, free factors; convergence is conditional |
| Dictionary-column and coefficient-row geometry | Expansion of the balance identity in text | Exact balance; basis comparisons, not layer comparisons |
| Coefficient Gram identity | Complete algebraic proof in text | Any balanced product, including rank-deficient products |
| Minimum factor norm and orthogonal freedom | Proofs using compact SVD and product-preserving variations | Complete orthogonal parameterization stated for rank(W)=r |
| Second-stage SGD attraction | Existing project derivation SECOND_STAGE_SELECTION.md ?8; Li?Wang?Arora (2022), Theorem 4.6 and Corollary 5.2; proof reproduced in manuscript | Fixed isotropic matrix observations, minimal width, small-step limit, arbitrary entry on optimal fiber; weighted balance |
| MASA | arXiv:2508.04581v2, method, experimental setup, Figures 3–10 | Joint training and pretrained PCA distinguished; displayed axes checked |
| NPAS | arXiv:2006.10598v4, Sections 3.1–3.2, Appendix B.3 | Direct coefficients versus affine embeddings; template resizing |
| ASLoRA | arXiv:2412.10135v2, Sections 3.2–3.5, Figure 3 | Historical matrix averages; allocation plot |
| SuperWeights | WACV 2024, Section 2.2, Table 3 | Gradient compatibility differs from static factor geometry |
| Prior identifiability work | Carrington, Bharath, Preston (2019), full text | Similarity evaluation and orthogonal freedom already studied |
| Prior SVD grouping work | Shamrai, arXiv:2601.11626v1 | Existing error-controlled clustering |

Primary URLs and bibliographic entries are in the manuscript. Independent citation review accompanies this update.

## Chapter plan

1. Motivation: equivalent heatmaps versus actual-training relevance.
2. Setting: parameter-saving shared basis, unique optimal product, factorization fiber.
3. Results: balance, D–C and W–C relations, remaining coordinate freedom.
4. Optimization: classical sufficient case and the unproved practical extension.
5. Applications: match each relation to what authors actually plot or cluster.
6. Research history: human decisions that determined the scientific question.
7. Future work: actual-training probability and concrete downstream decisions.

The unresolved link is whether actual application optimizers and parameterizations produce sufficient balance, with useful error and probability bounds. Uniform rotation, mixing time, rarity of coordinate changes, and improved downstream performance are not established results in this version. Earlier exploratory experiments are not repurposed as proofs of these claims.

The SGD model result is included as established within its assumptions: attraction from every fiber entry point and a sequential-time/step-size asymptotic probability statement. The open practical question must not erase this result or reopen first-stage convergence, which is outside the selected scope.
