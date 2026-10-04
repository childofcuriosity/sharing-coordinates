# 理论主线预印本独立编辑审计

审计日期：2026-09-25。对象：55 页 `paper/main.pdf`（SHA256 `49f03f899d7529cfcaffed6584f60f0116ffb5863a775ff74fedd43448953cea`）及对应 sections/main.tex。通读范围包含摘要、主文、参考文献和附录 A–M；图表按 PDF 文本及图注审读，未做逐像素视觉审计。未读取历史 review 判词。本文是针对该快照的编辑与论据边界审计，不是实验重跑、完整参考文献真实性核验或形式化证明认证。使用 paper-review 技能；外部来源核验范围见 `frontier_release_editorial.md`。

## 判断

存在清晰、可独立成文的理论贡献：同一有效乘积上的 Euclidean response stabilizer、带显式常数的置换轨道逆稳定性，以及从有噪乘积设计 K 个查询后控制整个可行集直径的延伸。主证明读取后未发现明显破坏这条链的代数反例；这不等于逐项证明认证。当前首要障碍是一个可直接修正的错误表述、旧新协议错位，以及主文仍按三条经验审计并列贡献组织。没有理由为此继续堆大型实验。

建议完成下表高优先级修正后，把稿件作为**明确限定观测模型的理论预印本**建议公开。未获得外部人工审稿不构成否决预印本的充分理由；作者责任、科学有效性与是否已有正式同行评审是不同问题。保持 ICLR2027 模板和 `Author information pending` 占位，不创造模板之外的批准手续。本审计不执行上传。

## 具体问题表

| 编号/优先级 | 位置与可复核证据 | 问题及影响 | 最小修正 |
|---|---|---|---|
| E1 必修：数学表述错误 | Introduction p.2，`introduction.tex` 原 29–31 行：uniformizing gauge “changing router entropy and potentially its argmax assignments”；附录 A 的改变 argmax 示例用另一个 M | 对 M_s=(1-s)I+s11ᵀ/K，(aM_s)_j−(aM_s)_k=(1-s)(a_j−a_k)。0<s<1 时排序、ties 和 argmax 全保留。不能把一般 gauge 的后果归给该 uniformizing path。 | 改为该路径改变 entropy；**其他**可行非均匀 gauges 可改变 argmax，并引用现有例子。无需新实验。 |
| E2 必修：贡献重心错位 | p.2–3 三条 contributions：第一条理论，第二 ASLoRA，第三 folding；p.8 Sec.5.2 称 ASLoRA “central real-system experiment”；标题和摘要第二段占比明显 | 读者会把文章识别为 gauge 决策审计论文，而最新全局 noisy theorem 在贡献列表中反而缺席。ASLoRA 的 LoRA 参数化并非本定理的 softmax-router 模型，不能作为定理应用验证。 | 贡献改成：精确 stabilizer；量化逆与有限 noisy measurement consequence；简洁数值说明。ASLoRA 降为末尾辅助实例或附录，删除 central 等主次信号。 |
| E3 必修：查询协议不相同 | Cor.1/p.6 & App.B.4：仅第一查询有 v_j，后续纯 u_r；Cor.3/App.D Eq.116：每个查询都重复 W，且 ∥G_p∥F=1；Sec.6.1/表2用 Cor.1；Sec.6.4/表6用 natural task gradients；App.M synthetic 用 Gaussian，LM 用 task gradients | 旧实验支持公式、旧 tomography 或带实测 rank 的自然查询。它们没有直接执行 App.D 的新版共同查询设计，也未把误差代入 CΘ、CR 验证 whole-feasible-set 界。摘要把新定理与精确恢复数值相邻摆放易产生实证覆盖错觉。 | 加一条明确映射：“The finite-noisy diameter bound is theoretical; the numerical studies use the exact-product design or rank-checked natural/Gaussian probes.” 不声称既有 noisy recovery 验证 Cor.3；不需为理论成立增加训练实验。若做一个小型数值 sanity check，必须称 implementation check，不能称验证全局 quantifier。 |
| E4 必修：三种保证需固定术语 | Cor.2/Theorem4 全局 common-product；Cor.5 小 product error 的成对界；Cor.3/Theorem5 全可行集；Prop.12/App.E candidate local component；表7 Local=140/1920、2/48 | 正文大部分已有正确 caveats。但 Discussion “joint product–response extension remains local” 易被读成同 Prop.12 一样需要邻域/同分支；Cor.5 只要求 product 差小及域条件，并不假定因子先接近，且借全局 C* 控制大 response 差。 | Cor.5称“small-product-perturbation pairwise bound”；Prop.12保留“candidate-local feasible-component bound”；Cor.3称“whole-feasible-set diameter bound”。表7 Local 标明 numerical local passes，或移附录。不可把 local passes、empirical acceptance 当 Cor.3证书。 |
| E5 必修：协议/结果时代错位 | Sec.5.3 p.8 用 seed 0–2、初始化主导、same-size 与 signed-local 未执行；Sec.6.3 p.10 主结果是 seed10–12，same-size 已执行且 positive 3/3，signed-local0/3。Sec.5引言仍说“to a failed replication check” | 单看方法会误以为主结果缺少已宣称控制；这些句子单独描述旧实验可能正确，但没有明确历史标记。 | 主文协议描述当前主要证据；旧实验整段移到已存在 App.L.1 或标题加 Historical。不把历史 1/3 与新 3/3 当相同训练设置的直接复现率比较。 |
| E6 应修：篇幅与重复 | 主文到 p.13，参考文献至 p.16；Sec.3七种问题、Sec.4.1静态背景、Sec.4.2后大段逐个先行机制说明、Sec.5独立协议、Sec.6.1三个表、Discussion重复全部 caveats | 核心证明没有主文 proof sketch，却被 provenance 与多轮“additive/subsequent”过程叙事挤占。这是投稿/公开阅读质量问题，不是页数形式手续。 | 以最终结果组织，而非形成史；主文保留一个证明思路、一个保证对照表、最多两张数值表。旧 analytic extraction 与 full-dimensional autodiff 的历史区别放统一附录。 |
| E7 应修：贡献边界过度分散 | Related work 末句只保留 common-product chain + ASLoRA；App.F p.42 novelty summary 也未含 finite noisy extension；新 Cor.3 在 p.6 | 不是过度 novelty，反而让最新版贡献与文献定位不一致。详细列出每个已知组成并不能替代一句精确新命题。 | 统一为 model-specific complete identification/stability chain，加 noisy-product designed-query consequence。把 generic probing、joint diagonalization、robust tensor stability明确作为工具；勿称全新一般逆定理或 optimal K-query result。 |
| E8 应修：经验拒绝失败可以更短 | Sec.6.5 & 表7，combined accepted bad=92/1782=5.16%，residual-only=68/1757=3.87%；CI[4.50%,5.79%]；App.M最大误差10.54、误接受0.820 | 失败公开充分，值得保留。但把 empirical calibration failure 置于摘要篇幅中心会让文章看似主张可信恢复算法；“exceeds target”只是点估计大于目标，不应升级为显著违背5%的统计结论。 | 主文一句说明联合拟合改善恢复但没有可靠全局拒绝保证；保留 tail failures 和 residual-only comparator 附录。需要数值时把5.16%写成 observed fraction，不说 statistically significant violation。 |
| E9 应修：不要让幅度成为另一个中心 | 摘要 ASLoRA 11/30与loss区间；Sec.6.2/6.3两个决策实验；全文反复解释历史图恢复 | ASLoRA展示representation-sensitive action，未证明固定规范动作最优、全部gauge有害、最终test regret或真实部署普遍性。现稿多数已主动限制，不能在缩写时丢失。 | 摘要最多一句辅助例子，不必罗列全部经验数字。主文一个小段保留 canonical action、reimplementation、one first merge；更多折叠实验移附录。 |
| E10 应修：披露中的手续性阻断语 | p.16 AI disclosure：“requires human-author review ... before public release” | 该句是稿件自加流程要求，不能据其循环推断研究不能建议预印本；也不能虚称人工审核已经完成。技能引用属于过程说明而非结果证据。 | 用事实性 AI assistance disclosure + authors responsible，删除未经用户要求的 release approval 前置条件；保留作者占位。不要暗示 AI skill 提供科学背书。 |
| E11 可改：观测成本表述 | K queries 在主文强调，App.D才说每次返回 L×D full response；全文 Θ 和probe构造都依赖内部访问 | K不是K个标量，也不是K个普通训练样本；不应由query数推出低成本、被动可用或部署友好。 | 主定理附近一句写明每次query输出完整L×D响应、已知正block rates/temperature、固定checkpoint、无AdamW state。|
| E12 可改：argmax不具自动稳健性 | factor orbit inverse→小误差；决策实例读argmax | 即使因子恢复稳定，无top-two margin仍不能保证离散partition稳定。当前稿未明确给出该保证，所以不是已证实错误；把理论与决策联系加紧时很易越界。 | 保持“identifies coordinates”而非“robustly recovers structural decisions”。若想写partition corollary，只需附加router top-two margin并给简短推论，无需新增实验。 |

## 核心保证与现有证据的对应关系

| 对象 | 已有理论允许的结论 | 当前数值能说明什么 | 不允许推到哪里 |
|---|---|---|---|
| 完整 R 与精确 Θ | Theorem1，满秩，Euclidean共同已知rates/τ，至置换 | 表1固定alternative有区别；表2独立autodiff公式与unseen response吻合 | 少数固定alternative测试不能证明所有alternative；不识别历史图 |
| 精确Θ + Cor.1 K查询 | 给定response结构下恢复完整R；再由定理识别因子 | 表2operator重构、表3精确因子重构 | 不自动支持有噪Θ；不证明PSD-clipped算法噪声稳定 |
| 精确Θ + natural/Gaussian查询 | 满足Prop.11/Prop.7 rank条件时tomography | 表6六checkpoints通过rank并恢复精确测量 | 不覆盖任何自然训练trajectory或全部任务分布 |
| 有噪Θ + App.D重复W的unit查询 | 指定域上，εΘ阈值内全可行集diameter，任意符合norm预算噪声 | 当前稿没有该query设计的独立数值界评估 | 不可称表7实验验证全局证书；不保证求解器找feasible点 |
| noisy restricted-subspace solver | Prop.12候选局部分支；不保证truth进入该子空间 | 联合拟合经验改善；局部passes和拒绝失败 | 不替代全空间/global保证；所有start接近不能证明唯一 |
| ASLoRA canonical ties | 静态gauge的辅助决策反例 | 3 seeds中raw score→index action→validation consequence | 不应用softmax定理；不宣称full-training regret/普遍有害 |

App.D 的 proof architecture 清楚：最佳 rank-K 子空间残差→所有 feasible B 的 off-subspace 统一上界→共同支撑结构算子的稳定 decoding→回到原始 B→调用小 product 差的全局因子 inverse。投影是证明中间量，不是真值子空间假设。把这段思路放到主文，比继续增加恢复模型更能强化主贡献。

## 建议 9 页主线（参考文献与附录另计）

保持 ICLR2027 模板、当前作者占位及 preprint header。目标是内容压缩，不靠缩字号/边距。55页总长度本身不是不可公开理由。

| 预计页数 | 内容 | 保留/移动 |
|---|---|---|
| 1.0 | 摘要约180–230词 + 开头问题 | 直接提出“Θ不确定当前chart；额外Euclidean response何时足够”。精确结论、稳定性、finite noisy consequence、数值一两句。删除摘要多条决策结果统计。 |
| 0.8 | Introduction与三条贡献 | 三条分别为exact classification、quantitative inverse及finite noisy consequence、bounded numerical illustration。ASLoRA只有辅助动机句。 |
| 0.6 | Related work | SSMF、Jacobian-Gram/optimizer metric、joint diagonalization/robust inverse/probing三段；详细逐文变量对应留App.F。 |
| 0.8 | Setup与观测 | Θ、softmax、R、共同metric、permutation distance、域；用四行表区分exact product/full response/finite queries/noisy product。七问题taxonomy缩为一个段落。 |
| 1.6 | Exact response identification | 公式、Theorem1、四步proof sketch：共同产品给M；响应给Gram；PSD平方根给covariance；diagonal algebra给permutation。说明rank反例，静态SSMF只一段。 |
| 1.2 | Quantitative inverse | 清晰陈述uniform α/sA/sB/M，常数依赖与linear bound；小product perturbation bridge；一个degeneracy例说明为何边界重要。完整常数仍附录。 |
| 1.3 | Finite observations with noise | Cor.1短陈述后突出Cor.3 whole feasible set；明写新query、ε阈值、输出维度、domain external assumptions；上述四步proof sketch。把Prop.12放附录。 |
| 1.1 | Numerical illustrations | 合并operator/factor/natural-gradient主要数据，说明exact测量与有限精度；简述joint-noise recovery及拒绝失败，直接标注不验证Cor.3。建议至多两表。 |
| 0.3 | Auxiliary decision example | ASLoRA一段或移附录只在此指引；canonical action、更换评分坐标、适用边界。folding详细内容全移附录。 |
| 0.3 | Limitations/conclusion | 已知Euclidean metric、nondegenerate域、query cost、非historical target。避免再重复每个实验形成史。 |

附录建议先放核心证明（现在B/C/D），再放自然query扩展和局部恢复结果，再放当前数值协议；历史 sanity sweeps和辅助决策排最后。已有原始记录、失败与provenance不能删去；移动论文呈现不等于抹去历史证据。

## 外部证据缺口与预印本建议的边界

1. **核心理论不依赖新的外部部署数据。** 对指定模型的数学定理，严格证明本身是主要证据。当前无需证明AdamW历史图、部署收益或ASLoRA最终准确率，前提是明确不作这些主张。
2. **Novelty不应升级为排他优先权。** 稿件自己保留Karuturi等全文未获取情形；本审计只核验三个一手元数据/摘要来源，不宣称对2026全部相关论文完成新一轮技术搜索。保留具体模型的正面贡献陈述，不说first/only。缺一篇可访问全文不是自动否决预印本，但若要声称严格优先权则仍需更完整证据。
3. **ASLoRA真实实现缺口可以通过范围限定处理。** 源算法candidate set有歧义、λ操作定义缺失、未找到官方code。这限制原方法复现/部署断言，不影响“documented reimplementation中的bounded witness”。把它设为辅助实例即可，不需要为主理论重新跑完整ASLoRA。
4. **若要称实用可信恢复系统，证据仍不足。** 全局域常数未从数据验证，界很保守，数值拒绝未达到所报5%目标，噪声是受控Gaussian而非未知optimizer error。本文应主动不作这一系统性主张；这是一项科学范围限制，不是必须增加另一轮实验才可公开的手续。

完成E1–E5并统一摘要、贡献与结论之后，保留当前清楚的失败与范围披露，就有理由建议公开一个理论优先的arXiv预印本。本建议不等同于ICLR接收预测或完整独立数学认证。

## Final review：重构稿闭环与公开建议

本节复核对象是重构后的《Sharing Coordinates: Identifiability from Optimizer Response》，54页PDF，主文结束于第7页，附录A–O。与前文55页初始审计不同，本节给出修订后判断。复核重新读取了当前 main、全部修改主文与搬迁附录、核心证明和保证交叉引用；检查了最终PDF文本和编译日志。没有读取历史review判词，没有修改paper文件。

### 版本绑定

- 审阅时 `paper/main.pdf` SHA256：`59e26618226c2757dad3e5ac979e18ebef409bd946085321bf26190afc1ca6b1`。
- `pdftotext -layout paper/main.pdf` 的完整输出字节 SHA256：`a31de800d1ba92a9bc0c5aa6f5f97d75ce8b981c88454fc4db9ba1306ad5e43e`。
- 42个源码文件内容绑定 SHA256：`408ce292f00f63a63bab2dd46e1527d9abd2811863a618845edcecb637fa2824`。算法：依次取 `main.tex`、`references.bib`、按路径排序的 `sections/*.tex`、按路径排序的 `generated/*.tex`；对每个文件串联“相对paper路径UTF-8 + NUL + 原始文件字节 + NUL”，计算整体SHA256。
- 源码绑定不包含模板、图像等非该42文件资产，因此不是完整发布包校验和。打包重新编译可能只改变PDF时间戳等元数据；这时应优先对照源码与PDF文本hash，不把不同PDF二进制hash直接判成内容变更。

### 十二项闭环

| 项目 | 最终结果 | 复核依据 |
|---|---|---|
| E1 uniformizing argmax | 已解决 | Introduction明确其保持ordering/argmax，另行指出其他非均匀gauge可改变partition。 |
| E2理论主线 | 已解决 | 新标题和摘要只以response identification/stability为中心；三条贡献变为exact classification、量化及finite noisy保证、限定范围数值说明。ASLoRA仅辅助段与附录。 |
| E3查询协议不同 | 已解决 | Sec.4.3明确新query每次重复W；Sec.5开头明确现有数值没有实证评估Cor.3全可行集界；摘要也明确separate protocols。 |
| E4局部/全局 | 已解决 | Sec.4.2将Cor.5称small-product-perturbation bridge并写无candidate neighborhood/common branch假设；Sec.4.3与Sec.5区分Cor.3和Prop.12。App.C同步。 |
| E5新旧协议 | 已解决 | 原benchmark现为App.G，明确historical settings；G.3指向后续App.N，禁止pooling成同一replication rate。 |
| E6篇幅与证明 | 已解决 | 主文7页、两表，新增exact证明思路、linear inverse机制和finite-noisy桥梁。比9页建议更短但未损失核心条件，无需填充至8页。 |
| E7novelty边界 | 已解决 | Related work和App.F均纳入noisy-product consequence；并新增明确承认固定条件下linear robust tensor inverse先例。我另行读取Bhaskara2014主论文Theorem5，确认该承认有一手依据。 |
| E8拒绝诊断失败 | 已解决 | Sec.5保留5.16%及CI、3.87% residual-only comparator、10.54 tail与0.82 false accept；只称point estimate高于目标。App.H保留完整失败。 |
| E9决策幅度 | 已解决 | 摘要不再陈列ASLoRA/folding数字；辅助段明确非softmax定理的LoRA应用、非universal harm/最终regret。 |
| E10披露前置手续 | 已解决 | AI disclosure真实说明协助和责任，不再把未完成人工审核设为发布前置许可；作者占位保留。 |
| E11观测成本 | 已解决 | Cor.3紧邻文字明示full L×D response；Model固定checkpoint/law与已知rates/temperature，Discussion重复实质条件。 |
| E12argmax稳健性 | 已解决 | Model的Target and scope明确没有coordinate gap就不推出stable argmax decisions。 |

### 搬迁后新增问题的处理

复核中发现的唯一未定义引用是旧 `eq:vu_ssmf_path`；现在App.B.1改为指向文献附录，最终日志没有undefined/multiply-defined引用警告，也无Overfull警告。静态展开实际输入文件的label检查未见重复label；模板宏的参数占位不是缺失引用。最终PDF文本没有 `??`。

主文Corollary编号为1（stable inverse）、2（exact finite tomography）、3（finite noisy diameter）；附录的proof references与该顺序对应。Theorem1/4/5分别对应exact/full inverse/finite noisy完整证明。附录G–O移动后，正文的协议范围、App.N新训练与App.O noisy recovery指引均一致。两张主表已按Table1后Table2显示，不再发生Table2浮到Table1之前。最后两处“three main tables”及“intended empirical contribution”旧措辞也已改为original/auxiliary。

### 科学判断与真实发布建议

**建议把该版本作为理论优先的arXiv预印本公开；本轮未发现剩余科学阻断。** 这不是仅有版式合格的判断：文章有一个明确的model-specific识别命题、完整可跟踪的证明、带域条件的量化逆以及有限有噪观测延伸；主文现在正面解释从response到Gram/covariance/diagonal algebra的链条，并说明全局直径如何覆盖未投影的原始因子。科学增量虽建立在已知工具上，但其限定模型与观测下的完整归约能够作为可公开讨论、可反驳与可检验的研究结果。

数值证据现在恰当地承担实现与边界说明，不再替代全称命题的证明。无新query实验、没有AdamW trajectory恢复、没有完整ASLoRA官方复现，都不反驳论文实际宣称的条件性理论结果。对不可靠经验拒绝与极端误差的披露完整，不能据此将文章误认成普适可信求解器。已知先例和不可获取全文的范围保留，未使用first/optimal等需要额外外部证据的优先权断言。

建议的置信边界：本轮没有重跑全部训练或独立重算每个实验JSON，也没有形式化验证每项证明；这限制我能宣称的独立认证强度，**不等于发现结果错误，也不构成额外人工批准门槛**。后续普通预印本讨论可能发现更紧假设、常数或先例，当前稿已提供足够具体条件与证明供检查。本推荐不预测ICLR接收，不声称已获外部人类同行评审，不等于已经上传arXiv。保持用户要求的ICLR2027模板与作者占位。
