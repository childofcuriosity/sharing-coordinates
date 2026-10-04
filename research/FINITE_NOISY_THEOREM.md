# 有限带噪观测下的全局可行集直径定理

本轮交付是数学证明，不以数值实验作为依据。准确定位：**已有全响应
逆稳定性定理的显式有限观测推论**。新的证明步骤处理带噪观测子空间、
单位范数有限探针和整个可行集合之间的误差传递，不声称建立了独立的
一般逆问题理论。

完整排版证明见 paper/sections/appendix_finite_noisy_theory.tex，
主文陈述见 paper/sections/theory.tex。旧基础定理的全部证明仍在
appendix_theory.tex 和 appendix_enhancement_theory.tex 中。

## 精确结论

固定 L≥K≥2，D−K≥L，以及已知正学习率和温度，记 λ=η_Z/τ²。
给定正参数 α,s_A,s_B,M，考虑域

\[
\mathcal D=\{(A,B):A\mathbf1=\mathbf1,\ A_{ij}\ge\alpha,\
\sigma_{\min}(A)\ge s_A,\
\sigma_{\min}(B)\ge s_B,\ \|B\|_F\le M\}.
\]

这些是外部声明的域条件，并非由候选解自动验证。

从任意带噪观测产品 \(\widehat\Theta\) 取前 K 个右奇异向量为 U 的行，
令 \(\widehat P=U^\top U\)。在正交补中选 L 个正交单位行组成 W。
这样 \(UU^\top=I_K,\ WW^\top=I_L,\ WU^\top=0\)。
若 \(u_p^\top\) 是 U 的第 p 行，选

\[
G_p=\frac{W+\mathbf1_Lu_p^\top}{\sqrt{2L}},
\qquad p=1,\ldots,K.
\]

每个探针的 Frobenius 范数恰为 1，且只依赖观测产品。
每次查询返回完整的 L×D 响应矩阵，不是一个标量。

定义 \(\mathcal F\) 为域内**全部且仅有**满足以下约束的因子：

\[
\|AB-\widehat\Theta\|_F\le\varepsilon_\Theta,\qquad
\left(\sum_{p=1}^K
\|\mathcal R_{A,B}(G_p)-\widehat R_p\|_F^2\right)^{1/2}
\le\varepsilon_R.
\]

若
\[
0\le\varepsilon_\Theta\le
\min\{\alpha s_B/4,s_A s_B/8\},
\]
则
\[
\boxed{\operatorname{diam}_{d_{\rm perm}}(\mathcal F)
\le C_\Theta\varepsilon_\Theta+C_R\varepsilon_R.}
\]

这里
\[
d_{\rm perm}((A,B),(A',B'))=
\min_\Pi\sqrt{\|A'-A\Pi\|_F^2+\|B'-\Pi^\top B\|_F^2}.
\]
\(\varepsilon_R\) 无需额外小量条件。空集的直径约定为零；
恢复含义要求真实因子属于此域且满足噪声预算，从而属于可行集。

## 全部常数

记
\[
h=\sqrt{2L}(K^{-1/2}+1+\sqrt L),\qquad
\beta=\frac{8\lambda M}{s_A}(h\sqrt K+1),
\]
\[
\kappa=\sqrt{s_B^{-2}+4s_A^{-2}},\qquad
L_R=\sqrt{(2\eta_B\sqrt L+24\lambda M^2)^2+(4\lambda M)^2}.
\]

继承的逆常数不是未证明的黑箱：其定义完整展开如下，原证明及本轮审计
均已逐项检查。

\[
L_M=\max\{1,4\sqrt L/s_A\},\quad
c_G=\frac{4L}{\eta_Bs_A^2},\quad c_H=\frac8{\lambda s_B^2},
\]
\[
\mu_*=\frac\alpha2\left(1+\frac1{4L_M^2}\right),\quad
c_J=\frac{L_M^2(c_G+c_H)}{\mu_*},\quad
c_D=\frac{2\sqrt{LK}}{s_A}c_J,
\]
\[
c_{\rm row}=\sqrt{Kc_D^2+(c_G+Kc_D^2)^2},\quad
\delta_0=\min\{1,(2c_G)^{-1},(2\sqrt Kc_{\rm row})^{-1}\},
\]
\[
C_*=\max\left\{
2\sqrt{L+L_M^2M^2}\sqrt Kc_{\rm row},
\frac{4\sqrt{L+M^2}}{\delta_0}\right\}.
\]
最终取
\[
\boxed{C_R=2hC_*,\qquad
C_\Theta=\beta C_*+2\kappa(1+C_*L_R).}
\]

常数很保守。本定理没有证明最优噪声放大率，也没有保证这个界对当前
有限精度实验给出紧的误差预测。

## 证明链

**第一步：同时控制整个可行集合的子空间误差。**

对任意 \((A,B)\in\mathcal F\)，AB 是 \(\widehat\Theta\) 的一个秩 K
近似。最佳秩 K 近似性质给出
\[
\|\widehat\Theta(I-\widehat P)\|_F\le\varepsilon_\Theta,
\quad
\|AB(I-\widehat P)\|_F\le2\varepsilon_\Theta.
\]
左乘 \(A^\dagger\) 得
\[
\|B-B\widehat P\|_F\le2\varepsilon_\Theta/s_A.
\]

这没有把估计子空间当作真实子空间，而是量化了所有可行 B 在其外部
的部分。令 \(\overline B=B\widehat P\) 仅用于分析。
由于 \(\|C(a_i)\|_2\le1\)，响应中的共享 Gram 项不变，局部项满足
\[
\|\mathcal R_{A,B}-\mathcal R_{A,\overline B}\|_{F\to F}
\le t:=4\lambda M\varepsilon_\Theta/s_A.
\]
不需要 \(\overline B\) 满秩或属于原来的参数域。

**第二步：K 个探针对共同支撑的整个线性算子类给出稳定界。**

考虑
\(\mathcal S(G)=QG+\mathcal T(GU^\top)U\)，
其中每层 T_i 为任意 K×K 矩阵。
对于两个这样的算子，令
\(E_p=\Delta\mathcal S(G_p)\)，
\(E^2=\sum_p\|E_p\|_F^2\)，\(c=(2L)^{-1/2}\)。
正交投影给出
\[
E_pW^\top=c\Delta Q,\qquad
\|\Delta Q\|_F\le E/(c\sqrt K).
\]
另一侧投影并逐探针堆叠，得到
\[
\sqrt{\sum_i\|\Delta T_i\|_F^2}
\le E/c+\sqrt K\|\Delta Q\mathbf1_L\|_2
\le(1+\sqrt L)E/c.
\]
所以
\[
\|\Delta\mathcal S\|_{F\to F}
\le\|\Delta Q\|_2+\max_i\|\Delta T_i\|_2
\le hE.
\]
在因子响应中，Q 已包含系数 η_B，因此 h 不需要另除以学习率。

**第三步：从有限观测返回原始、未投影的完整响应。**

任取两个可行因子 x,x'，原始响应堆叠的差不超过 \(2\varepsilon_R\)。
每个因子投影造成的堆叠误差不超过 \(\sqrt Kt\)。
第二步再加上两端完整算子的投影误差，得到
\[
\|\mathcal R_x-\mathcal R_{x'}\|_{F\to F}
\le h(2\varepsilon_R+2\sqrt Kt)+2t
=2h\varepsilon_R+\beta\varepsilon_\Theta.
\]

**第四步：调用已经证明并重新审计的全局因子逆界。**

原始两对因子仍属于原域，其产品差至多 \(2\varepsilon_\Theta\)。
给定阈值恰好确保旧联合扰动推论的产品小量条件成立。它给出
\[
d_{\rm perm}(x,x')
\le C_*\|\mathcal R_x-\mathcal R_{x'}\|_{F\to F}
+\kappa(1+C_*L_R)\|AB-A'B'\|_F.
\]
代入第三步即得目标不等式。该旧逆定理包括远距离情形的域直径论证，
并非仅在一个局部分支上成立。因此，本步覆盖 \(\mathcal F\) 中任意
两点，取上确界便得到整个可行集合的直径界。证毕。

## 审查及边界

- 独立推导、基础定理审计、反例审查三条工作线；审查记录见同目录
  FINITE_NOISY_INDEPENDENT_ROUTE.md、FINITE_NOISY_FOUNDATION_AUDIT.md、
  FINITE_NOISY_ADVERSARIAL_AUDIT.md。
- 逐项核对 2 倍配对误差、√K 堆叠误差、单位范数探针、全部常数及
  是否偷偷要求投影后满秩或真解在附近。
- 主文曾用 “contain every” 描述可行集合，会允许任意超集；
  反例审查指出后改为 “consist of all”，与正式定义完全一致。
- 噪声无需独立性，可以在声明的范数预算内任意相关。
- 不依赖奇异向量基底的连续选择。
- K=1 时 A 固定为全一列，直接有直径≤2ε_Theta/√L，无需响应。
- 这不是求解器收敛定理、未知优化器状态定理或完全由数据验证域条件的
  证书。审查由独立模型工作线完成，不代表形式化证明助手验证或外部
  人类同行评审。
