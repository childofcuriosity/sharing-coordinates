# 从非唯一最优 W 的选择偏置到系数热图：谱恢复路线的定理筛选

日期：2026-10-08。范围：本轮子任务只核验能够实际接到系数列关系的 implicit bias 定理，并完成代数传递；不把 low rank 当作热图可靠性的同义词。

## 1. 筛选问题与结论

问题一：有没有对原本存在许多零 loss 有效矩阵的二因子问题，证明普通优化器选择某个具有已知结构的 W？问题二：该定理能否经由可证明的不等式，保证实际训练系数 C 的某类热图读法？

目前最能直接使用的是 **低秩矩阵 sensing 的选择/恢复定理**。它提供 W 的 operator-norm error，而不只是“倾向低秩”的定性描述。下文把该 error 与实际因子的 imbalance 一起，传递成系数 Gram、夹角和聚类距离误差。由此确实得到一个选择偏置→热图列关系保证的闭合条件链。

它不自动证明 ASLoRA/MASA 的训练满足 sensing/RIP，也不凭空产生目标矩阵的分组结构。它说明：如果数据与目标的关系可被这种模型刻画，优化器在非唯一 minimizer 中的选择可以保住数据结构对应的系数列关系。

## 2. 主候选：非对称、过参数化矩阵 sensing

Johan S. Wind, *Implicit Regularization Makes Overparameterized Asymmetric Matrix Sensing Robust to Perturbations*, arXiv:2309.01796v2 (2025-08-22)。原始全文：https://arxiv.org/html/2309.01796v2 。注意 v1 标题不同；本记录使用 v2 的 theorem 编号。

核验位置：[Theorem 2.8](https://arxiv.org/html/2309.01796v2#S2.Thmtheorem8)、[Theorem 2.9](https://arxiv.org/html/2309.01796v2#S2.Thmtheorem9)、[Remark 2.6](https://arxiv.org/html/2309.01796v2#S2.Thmtheorem6)、[Theorem 6.1](https://arxiv.org/html/2309.01796v2#S6.Thmtheorem1)。若 Y=U_Y Sigma_Y V_Yᵀ，初始化 misalignment 的完整定义是 alpha=sqrt(2)||[D0;C0ᵀ]||/sigma_r(U_YᵀD0+V_YᵀC0ᵀ)。

Theorem 2.8 研究 f(D,C)=1/2 ||A(DC−Y)||²，Y 为 rank-r 矩阵，kappa=||Y||/sigma_r(Y)，允许因子宽度 h 大于 r。测量满足 rank-(r+1) RIP，常数 rho≤c1/(sqrt(r) kappa²)。初始化 stacked operator norm 为 a0，misalignment alpha 有常数上界，并要求

\[
a_0\le \frac{c_2\sqrt{\sigma_r(Y)}}{\min(n_1,n_2)^2/r^2+\kappa},\qquad
\eta\le\frac{c_3}{\kappa^3\|Y\|\log(\sigma_r(Y)/a_0^2)}.
\]

扰动范数上界 xi≤c4||Y||/[kappa³ log(sigma_r/a0²)]。在

\[
K=\frac{C_2}{\eta\sigma_r(Y)}\log\frac{\sigma_r(Y)}{a_0^2}
\]

步后得到

\[
e:=\|D_KC_K-Y\|\lesssim a_0^{4/3}\sigma_r(Y)^{1/3}+\kappa\xi.
\tag{1}
\]

取 xi=0 即为无额外扰动的 GD。Remark 2.6 用独立高斯小初始化、h≥max(n1,n2,2r) 保证初始 misalignment 以高概率有界。Theorem 2.9 将固定 RIP 测量换成每步独立的新高斯 measurement batch，给出同一 error，以至少 1−delta 的概率成立，batch size 下界为 C[log(n1+n2)+log K+log(1/delta)] r(n1+n2) kappa⁴。它不是对有限固定数据集反复抽样的普遍 SGD 定理。

本段主来源的文字性摘述控制在简短 theorem card；以下各节是本轮自行推导，不是声称原论文讨论热图。

为何真的是非唯一 W 的选择：若 A 有非零 kernel，Y+Z 对任何 Z∈ker A 都具有零 measurement loss；宽因子可表示许多这样的矩阵。RIP 排除低秩歧义，却不排除所有高秩零 loss 矩阵。定理证明小初始化 GD 在指定时间接近 Y，而不是随便选高秩插值解。其量化结论是有限时间恢复，不应写成所有固定初始化无限时间都精确收敛 Y。

一个同时满足这些条件的维数 regime 是 n1=n2=n、target rank r=1、kappa=1、h≥n，固定一个满足 theorem 的小 RIP 常数 rho。Gaussian measurements 所需 m=O(n/rho²)，选 n 足够大使 m<n²。于是 A 必有非零 kernel，且 h≥n 可表示任意 Y+Z；同时 rank-2 RIP 可以成立。不能把此例的 h 强行降到 2：若秩限制让两个可表示解之差落入 RIP 阶数，W 可能已经唯一。主定理对初始 alpha 的条件也不允许把 high-width Gaussian guarantee 无证明地套到 h=2。

## 3. 自推桥接定理：控制实际 C，不事后重新平衡

以下均为 operator norm。令

\[
W=DC,\quad H=D^\top D-CC^\top,\quad b=\|H\|,
\quad G=C^\top C,\quad G_*=(Y^\top Y)^{1/2}.
\]

精确恒等式为

\[
W^\top W=G^2+C^\top HC.
\tag{2}
\]

因此 exact balance 时，G=|W|；不必假设最小宽度。更重要的是，非零 imbalance 也可定量处理。

先证明 ||C||²≤||W||+b。取 CCᵀ 的最大特征值 t 和单位特征向量 u；若 t>0，v=Cᵀu/sqrt(t) 为单位向量，于是

\[
\|W\|^2\ge v^\top W^\top Wv
=t\,u^\top(D^\top D)u\ge t(t-b).
\]

解二次不等式得 t≤[b+sqrt(b²+4||W||²)]/2≤b+||W||。

假设 ||W−Y||≤e，则由 (2)

\[
\|G^2-Y^\top Y\|
\le (2\|Y\|+e)e+(\|Y\|+e+b)b=:d.
\]

对于 PSD 矩阵 A,B，若 ||A−B||≤d，则 A≤B+dI；平方根的 operator monotonicity 给出 sqrt(A)≤sqrt(B+dI)≤sqrt(B)+sqrt(d)I，交换 A,B 即得到 ||sqrt(A)−sqrt(B)||≤sqrt(d)。故

\[
\|C^\top C-G_*\|\le
\sqrt{(2\|Y\|+e)e+(\|Y\|+e+b)b}.
\tag{3-old}
\]

root 随后提出并经此子任务逐步审计的更强不等式，使 imbalance 以线性而非平方根进入。记 S=|W|。因 −bI≼H≼bI，精确恒等式给

\[
G^2-bG\preceq S^2\preceq G^2+bG.
\]

令 R=(S²+b²I/4)^{1/2}。右侧不等式加 b²I/4 并取 operator-monotone 平方根得 R≼G+bI/2，因此 G≽R−bI/2。左侧同理给

\[
|G-bI/2|=\{(G-bI/2)^2\}^{1/2}\preceq R,
\]

而任意 Hermitian A 均有 A≼|A|，故 G≼R+bI/2。这里只给平方后两侧使用平方根，没有把 noncommuting 矩阵的 Loewner 顺序任意平方；每一步合法。又因为 R 是 S 的函数，两者可同时对角化，逐特征值可得 0≼R−S≼bI/2。合并得到

\[
-\frac b2 I\preceq G-S\preceq bI,
\qquad\|G-S\|\le b.
\]

再把恢复误差传到 |W|，主用桥梁改为

\[
\boxed{\|C^\top C-G_*\|\le\zeta,
\quad\zeta=b+\sqrt{(2\|Y\|+e)e}.}
\tag{3}
\]

若 H≽0，则 G²≼S² 直接给 G≼S；结合 G≽R−bI/2≽S−bI/2，有 0≼S−G≼bI/2。因此 (3) 中 b 可进一步替换为 b/2。实际使用可取新旧两个合法上界的较小者。特别在 e=0 时，旧式只给 O(sqrt(b))，新式给 O(b)；单侧零初始化且 invariant H PSD 时常数再减半。

这在 Y rank deficient 时仍有效，不需要通过最小奇异值排除低秩；代价是平方根型的误差传递。

### 实际 GD 中 b 从哪里来

对 f(DC) 的普通两侧同学习率 GD，交叉的一阶项精确抵消：

\[
H_{k+1}-H_k=\eta^2\{(\nabla_Df)^\top\nabla_Df
-(\nabla_Cf)(\nabla_Cf)^\top\}.
\]

所以无需额外假设即有

\[
b_K\le b_0+\eta^2\sum_{k<K}(\|\nabla_D f_k\|_F^2+\|\nabla_C f_k\|_F^2).
\tag{4}
\]

若步长还满足常规的下降条件 f_{k+1}≤f_k−eta||grad f_k||²/2，则

\[
b_K\le b_0+2\eta f_0.
\tag{5}
\]

此下降条件需单独由 bounded-trajectory Hessian/步长验证，不能仅因 Wind 定理写了小学习率就不核对地宣称 (5) 已在原文保证。其原始初始化 stacked norm≤a0 直接给 b0≤a0²。于是 (1)+(3)+(5) 是一个明确、可检查的 GD→实际 C 保证；对 fresh-batch SGD 不可直接使用 f telescoping，仍应使用原文 imbalance 控制或 (4) 的相应 stochastic bound。

补上这个额外步长的一个保守而显式的验证：Wind Theorem 6.1(1) 在证明所用的整个 recovery interval 给出 stacked factor operator norm≤M=(3/2)sqrt(||Y||)。令 d0=min(n1,n2)，L_A 为 A 从 Frobenius norm 到 Euclidean norm 的 operator norm。每两个相邻迭代的直线段仍在此凸 norm ball 内。沿该段，对 combined Frobenius-unit perturbation (P,Q)，

\[
|\nabla^2f[(P,Q),(P,Q)]|
\le L_A^2\{2M^2+\sqrt{d_0}M^2+\|Y\|_F\}=:L_f.
\]

这是由 Hessian 的两项 ||A(PC+DQ)||² 和 2〈A(DC−Y),A(PQ)〉，及 ||PQ||F≤1/2 得到。故在原定理步长限制之外再取 eta≤1/L_f，下降条件成立，(5) 得到验证。它可能保守，但不修改优化器，也没有用 exact balance 假设冒充 actual GD。原文 Theorem 6.1(2) 本身还提供固定小常数级的 imbalance bound；这里的能量式能显示 eta→0 时该部分误差如何消失。

时间范围核验：Appendix B Eq.(17) 的 T2=5/sigma_r log(sigma_r/a0²)，正是 Theorem 4.3 的 T；主文 4.3 后明确 K eta=T。因此不是拿只覆盖 warm-up 的范数界控制更晚的输出。

## 4. 实际获得哪些热图保证

设有关列的目标 Gram 对角元 G*ii≥m>zeta。则列长度平方误差≤zeta。

任意两列的平方距离满足

\[
\big|\|c_i-c_j\|^2-(G_{*,ii}+G_{*,jj}-2G_{*,ij})\big|\le2\zeta,
\tag{6}
\]

因为 e_i−e_j 的平方长度为 2。若目标 within-cluster 最大平方距离为 a，between-cluster 最小平方距离为 b_sep，且 b_sep−a>4zeta，则某个中间距离阈值在训练后的 C 上仍完全区分这两类 pair。这里得到的是同一距离判据的保证；不是未经证明保证任意 clustering 算法输出标签。

余弦相似度的误差满足

\[
\left|\frac{G_{ij}}{\sqrt{G_{ii}G_{jj}}}
-\frac{G_{*,ij}}{\sqrt{G_{*,ii}G_{*,jj}}}\right|
\le\frac{2\zeta}{m-\zeta}.
\tag{7}
\]

证明：分子误差项≤zeta/(m−zeta)；每个对角相对误差≤zeta/m，因此两个归一化分母的相对差也至多 zeta/(m−zeta)，再用目标相关系数≤1。

这些平方量只是用于证明向量几何的工具。它们并未把原始系数热图换成元素平方占比，也不据此把原始热图的大系数比例夸大。

## 5. 接到原始系数，而非止步于 Gram

### 5.1 任意宽度、无需 Haar 的原始形状保证（可接主定理）

令 x=c_i/||c_i||、y=c_j/||c_j||，并选择 s∈{−1,1} 使内积非负。若目标 absolute correlation 为 rho*，则 (7) 保证

\[
\|x-sy\|_2\le \sqrt{2(1-\rho_*+\gamma)}=:a,
\quad\gamma=2\zeta/(m-\zeta).
\tag{8a}
\]

所以每一 row 都满足 ||x_l|−|y_l||≤a。此结论是原始系数绝对值的归一化形状，不是逐元素平方，并且对任意隐藏宽度、任意共同正交朝向都成立。若参考列 x 的某 row 与其余 row 的绝对值差至少 Delta>2a，则 y 同一 row 也严格最大。这里不假设一定存在该 margin；它给出“列近共线→已有显著模式在另一列保留”的明确保证。

### 5.2 二基均匀概率换算（只作独立几何模块，不能冒充主定理的非唯一 regime）

以下限隐藏宽度为 2、给定列非零。在 Haar 内部朝向这一另加分布假设下，记两列的无向夹角 phi=arccos(|cos phi_signed|)∈[0,pi/2]。用原始绝对值比，要求两列都满足“同一个 row 的系数绝对值至少是另一 row 的 k 倍”，k≥1。令 a=arctan(1/k)。四个坐标轴方向各贡献长度 (2a−phi)_+ 的允许公共角度，故精确概率是

\[
\boxed{P_{same}(\phi,k)=\frac{2}{\pi}(2\arctan(1/k)-\phi)_+.}
\tag{8}
\]

这是“同一行主导”，不是允许两列分别由不同行主导。它在 phi=0 时退化为单列 4 arctan(1/k)/pi；phi=pi/4、k=2 时约为 9.03%，允许不同行主导的总概率则约为 18.07%。phi=pi/4、k=3 时两者都为零。

由 (7) 令 gamma=2zeta/(m−zeta)，rho*=|G*ij|/sqrt(G*iiG*jj)。有 |cos phi|≥max(0,rho*−gamma)，所以

\[
P_{same}\ge\frac2\pi\left[2\arctan(1/k)
-\arccos\{\max(0,\rho_*-\gamma)\}\right]_+.
\tag{9}
\]

**只要另一个适用二基且 W 非唯一的选择定理给出同类 error，(9) 就能把它换成原始热图概率下界。** 本文主候选的明确非唯一 Gaussian regime 使用 h≥n，不能直接接二基公式。这一节只记录可核查的几何换算，不算已闭合的主路线。Haar 朝向也尚需实际初始化/优化对称性或既定均匀基准支持；恢复定理本身不证明 Haar。任意宽度下已闭合的是 (3)–(7)、(8a)。

rank-one 特例最清楚。若 Y=sigma uvᵀ，则 G*=sigma vvᵀ，所有非零列的目标 absolute correlation 为 1。近似恢复+近似平衡让它们近共线；精确恢复平衡时 c_j=sqrt(sigma)v_j q，所有列原始系数模式成比例。它保证模式共享，而不能独自保证 q 的某一个坐标很大。Haar 下共享 dominant row 的概率为 1（ties 概率零），但 dominant row 的实际大小比还取决于隐藏宽度。

## 6. 第二候选为何目前不作为主结论

Jiang, Chen, Ding, *Algorithmic Regularization in Model-free Overparametrized Asymmetric Matrix Factorization*, SIAM J. Math. Data Sci. 2023；公开完整稿 https://arxiv.org/pdf/2203.02839 ，正文 Theorems 4.1–4.2。这是 ||FGᵀ−X||F² 的 fully observed 问题，small initialization + early stopping 依次逼近 truncated SVD。它能通过 (3)–(9) 对早停的谱形状作分析，但此时有效最优 W=X 唯一；截断点甚至不是该 loss 的局部最小值。因此它不能冒充本轮用户要求的“在非唯一最小 loss W 集合中选择”的 theorem。保留为未来有限训练/早停扩展，不充当当前胜出的候选。

旧的对称 PSD small-random-init spectral recovery（Stöger–Soltanolkotabi 2021, https://arxiv.org/abs/2106.15013 ）概念相近，但 tied factor UUᵀ 的假设比实际 DC 强。非对称 sensing 已有可读的直接定理，没有必要从 PSD 结果无证明地跳到 DC。

## 7. 收束判断

真正筛出的能力是：**低秩 sensing 的选择偏置，能将非唯一插值集合缩到一个目标附近；结合 imbalance 控制，可以保证列间相似/分离，并经 (8a) 保证原始系数列的共同模式。** 本节 sensing 定理的高宽度 regime 不能直接接二基公式。本轮后续的 regression 推导已另外完成二基实际训练概率的连接，见[综合报告](IMPLICIT_BIAS_HEATMAP_RESULTS.md)及[独立审计 T11–T12](IB_HEATMAP_TRANSFER_AUDIT.md)；两条路线的宽度、数据条件保持区分。

当前仍缺的实践桥梁是：实际应用 loss 能否支持类似 recovery/subspace selection，及目标列关系的来源。这不是要求知道整个 W 数值，而是需要足够的结构条件，例如簇间 margin、rank-one 因子 loading 下界或右子空间几何。单独“低核范数/低秩”不蕴含任意有利的热图关系；在此选择定理里，数据所携带的 Y 结构与算法的选择偏置共同承担结论。

## 8. 补充：共同设计矩阵的 min-norm 选择，比 sensing 更直接，但不能忽略因子失衡

核验来源：Hancheng Min, Salma Tarmoun, René Vidal, Enrique Mallada, *On the Explicit Role of Initialization on the Convergence and Implicit Bias of Overparametrized Linear Networks*, ICML 2021, [正式页面](https://proceedings.mlr.press/v139/min21c.html)、[正式 PDF](https://proceedings.mlr.press/v139/min21c/min21c.pdf)、[作者完整稿含证明](https://hanchmin.github.io/assets/pdf/MTVM2021ICML.pdf)。注意 min21a 是另一篇论文，不可误引。

设 X∈R^{n×d}、rank(X)=r<d，Y∈R^{n×m}。论文研究 GF 优化 f(U,V)=1/2||XUVᵀ−Y||F²，宽度 h≥min(m,d)，min-norm 解 Theta_hat=X†Y。把输入空间分为 row(X) 与 ker(X)，令 U=Phi1 U1+Phi2 U2，Sigma_x 为 X 的正奇异值平方组成的对角矩阵。

原文 Proposition 1（正式 PDF 第 5 页）证明：若 U1,V 收敛至最优 loss，且初值满足

\[
V_0U_{2,0}^\top=0,\qquad U_{1,0}U_{2,0}^\top=0,
\tag{10}
\]

则 UVᵀ→Theta_hat。原文 Eq.(7) 给 U2(t)=U2(0)，Eq.(16) 给两个 cross block 的齐次线性 ODE，故 (10) 是精确 invariant。仅令初始 UVᵀ 没有数据 nullspace 分量并不够，第二个条件实质必要于这个证明。

原文 Theorem 1（第 4 页）定义 invariant Lambda=U1ᵀU1−VᵀV，c=[lambda_r(Lambda)]_+ + [lambda_m(−Lambda)]_+，得到

\[
f(t)-f_*\le e^{-2\lambda_r(\Sigma_x)c t}(f(0)-f_*).
\tag{11}
\]

c>0 时还保证因子收敛至全局最优。Proposition 1 本身有条件依赖收敛，不能单独把它当作 global convergence theorem。

Theorem 2（第 6 页，Eq.20；公式已看原 PDF 图确认）：iid N(0,h^{-2 alpha}) 两因子初始化，1/4<alpha≤1/2，宽度 h>h0^{1/(4alpha−1)}，h0 是 m,d,1/delta,lambda1(Sigma_x)/lambda_r(Sigma_x)^3 的某多项式。概率至少 1−delta 下，

\[
\|U_\infty V_\infty^\top-X^\dagger Y\|_2
\le 2 C_*^{1/h^{1-2\alpha}}\sqrt{m+r}
\frac{\sqrt{m+d}+\frac12\log(2/\delta)}{h^{2\alpha-1/2}},
\quad C_*=\exp\left(1+\frac{\sqrt{\lambda_1(\Sigma_x)}}{\lambda_r(\Sigma_x)}\|Y\|_F\right).
\tag{12}
\]

这是 fixed-data 下真正非唯一 regression solution 的高概率选择结论；不像 sensing，不需要把目标列几何先假设为 planted ground truth。然而 **(12) 不能直接接平衡恒等式**。其 Lemma 1 保证 lambda_r(Lambda)+lambda_m(−Lambda)>h^{1−2alpha}，通过足够 imbalance 保证收敛。alpha=1/2 时该尺度不随宽度消失；alpha<1/2 时反而增长。因此宽度增加让有效 W 接近 minnorm，并未让实际 CᵀC 接近 |W|。总因子 imbalance 为 UᵀU−VᵀV=Lambda+U2ᵀU2，也须另行控制。不能用 canonical rebalancing 后的 C 替代训练出来的热图。

### 可直接推出的原始系数关系：不经过平方热图或 Haar

把论文符号换成 D=U、C=Vᵀ，则 GF 为

\[
\dot C=D^\top X^\top(Y-XDC).
\]

任意任务对比向量 a 若满足 Ya=0，令 z=Ca，则

\[
\dot z=-D^\top X^\top XDz,\qquad
\frac{d}{dt}\|z\|_2^2=-2\|XDz\|_2^2\le0.
\tag{13}
\]

因而 ||C(t)a||≤||C(0)a|| 对全时间成立；若 C(0)a=0，则 C(t)a=0 精确成立。取 a=e_i−e_j 就是“相同 target 列→相同原始 coefficient 列”；取一般 a 则是保存 targets 的线性关系。若 C0=0，则 ker Y⊂ker C(t)，所以 row(C(t))⊂row(Y)、rank C(t)≤rank Y。rank-one targets 下所有系数列以对应 loading 精确成比例，不需要先解出某个具体 W，也不需要平衡假设。

必须准确归因：(13) 是 GF 的 invariant/contraction 直接结果，不是 Theorem 2 minnorm 结论的额外功劳。Theorem 2 的双高斯 init 给 E||C0a||²=h^{1−2alpha}||a||²，在 alpha≤1/2 时该 bound 不趋零。因此它支持 W 的选择，却不自动支持该原始系数相似性的 vanishing-error 保证；一侧零初始化是有意义的实质条件差别，不是可以省略的技术细节。

本次筛选判断：**Proposition 1 + Theorem 1 是本轮一侧零初始化 regression 路线的文献锚点；双高斯宽网络 Theorem 2 是 W-selection 的广义版本，但 actual-C bridge 未由它自动补齐。** 一侧零初始化分支已完成自包含收敛、选择、实际因子和概率推导，见[综合报告](IMPLICIT_BIAS_HEATMAP_RESULTS.md)。没有用“随机宽度更一般”掩盖失衡造成的断点。
