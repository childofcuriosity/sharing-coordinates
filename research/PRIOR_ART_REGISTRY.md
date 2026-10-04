# Prior-art and claim registry

Last audited: 2026-08-23.  This registry records what is already known, the
exact map into the manuscript's notation, and the narrow residue that may be
claimed.  A search hit or title is not counted as a theorem audit: locations
below refer to the primary full text unless an entry is explicitly marked
unresolved.

## Classification rules

- **Existing result:** the statement and mechanism already appear in the
  cited source.
- **Direct rewrite/corollary:** obtained by transpose, renaming, a change of
  variables, or a short specialization of an existing result.
- **Application/audit tool:** useful in the present system but standard
  mathematics.
- **Candidate addition:** not found in the audited full texts.  This is a
  bounded literature claim, not an unconditional claim of priority.
- **Unresolved:** the primary full text was not available; no technical
  conclusion is inferred from metadata, an abstract, or a secondary citation.

## Executive classification

| Manuscript component | Closest prior art | Classification | Permitted positioning |
|---|---|---|---|
| Unconstrained non-identifiability and the uniformizing family | Vu Thanh--Gillis--Lecron (2023), Sec. IV-A | Existing result; exact reparameterization | Background limitation with explicit credit |
| Arbitrary local simplex-preserving gauges, entropy/argmax changes | Same SSMF change-of-basis mechanism | Direct extension/corollary | Consequences for router diagnostics, not foundational novelty |
| Product as a complete invariant on full-rank LoRA factors | Putterman et al. (2024), Lemma 3 / Theorem 3 | Existing result | Cite before using product-space invariants |
| Gauge-dependent factor optimization/aggregation/selection | LoRA Done RITE; Chen--Liu--Zhu; Singh; Aladrah et al. | Existing phenomenon | System-specific audit only |
| Loewner stability and trace-one PSD action selection | Classical Mahalanobis metric learning | Application/audit tool | Exact characterization for the ASLoRA decision, not new SDP geometry |
| Condition-number-bounded gauge metrics | Liao et al. (2015), Prop. 1 and Eqs. (2)--(4) | Existing machinery | Standard robustness restriction |
| ASLoRA running-average nearest-pair rule under a common gauge | ASLoRA arXiv v2 plus the above gauge/metric literature | System-specific application | Empirical decision audit; distinguish selection proxy from action cost |
| Exact Euclidean response stabilizer of the shared softmax-router model | Jacot and Marcotte et al. supply Jacobian--Gram/factor response; Shem-Ur et al. explicitly allow a function-preserving symmetry to change the NTK; RITE and LoRA-S establish low-rank chart dependence and invariant optimizer alternatives; Varre et al. the softmax block; Singh the Euclidean `GL -> O` step; Dorrell et al. an orthogonal-Gram plus tight-scattering route to permutation; Afsari the exact joint-diagonalizer-to-monomial step; Lau--Su and Tran et al. router/MoE permutation-shift symmetry; Nguyen et al. a density-level softmax inverse; Wang--Wang the Transformer function-gauge/quotient boundary; Goel et al. a softmax-attention-to-factor-loss reduction; van Oostrum et al. and Mishra et al. the metric counter-boundary | Candidate addition | Claim only the common-exact-product, complete-Euclidean-response finite-width iff theorem; not first function-equivalent NTK chart dependence, factor chart dependence, Gram-plus-scattering recovery, joint-diagonalizer uniqueness, or softmax/MoE label identifiability |
| Quantitative response-to-chart inverse | Nguyen et al. give a global softmax-MoE density-to-parameter inverse; Tu et al. give Gram-to-Procrustes stability; Afsari gives exact and local first-order joint-diagonalizer uniqueness/sensitivity; Bhaskara--Charikar--Vijayaraghavan give global robust Kruskal factor stability; generic perturbation tools are standard | Candidate addition with bounded-search wording | Only the complete model-specific common-product response chain on the named uniform nondegenerate set, not its Gram, joint-diagonalization, or robust tensor-factor components |
| Finite structured response tomography | Novak et al. (2022) NTK-vector reconstruction; Chiu--Demanet; Halikias--Townsend; Otto; Amsel et al. (2026) | Direct structured-probing corollary | Model-specific compression to `K` designed probes under `D-K>=L`; not a second central novelty |

## Numbered-result inventory

This table is exhaustive for the numbered mathematical statements in the
current manuscript.  **Direct** means that the result is a transpose,
specialization, or short textbook consequence; such a row is retained only to
make a boundary or an experimental estimand checkable.

| Manuscript label | Exact target | Closest result and variable map | Classification | Increment retained here |
|---|---|---|---|---|
| `thm:continuous_gauge` | Local simplex-preserving orbit of `(A,B)` | Vu Thanh--Gillis--Lecron Sec. IV-A under `X=Theta^T`, `W=B^T`, `H=A^T`; the unrestricted invertible rank-factor action is the ambient mechanism | Direct extension of existing SSMF non-identifiability | Router-facing dimension count and diagnostic consequences only |
| `thm:end_to_end_impossibility` | No end-to-end estimator can distinguish two worlds with the same `Theta` | Standard two-point observational-equivalence argument applied to the PA-1 gauge pair | Elementary direct corollary | Separates coordinate recovery from functional and historical recovery |
| `thm:anchor_recovery` and `lem:app_barycentric` | Recovery of separable/one-hot factors up to permutation | Standard separable SSMF/simplex vertex geometry; map layer rows of `Theta` to samples and rows of `B` to simplex vertices | Existing geometry / direct specialization | States the positive boundary in the manuscript's notation |
| `thm:noisy_recovery` | Exact one-hot partition under a `4 epsilon` separation | Triangle and reverse-triangle inequalities followed by connected components | Elementary stability lemma | Checkable sufficient margin; no novelty claim |
| `thm:probe_recovery` | Recovery of a distribution-relative functional partition | Uniform Bernstein concentration plus a union bound and standard linkage separation | Standard finite-sample clustering certificate | Fixes the observation law and distinguishes functional class count `K_f` from basis count `K` |
| `prop:factor_dynamics` | Effective gradient flow for unconstrained factors | Product rule for matrix factorization; coordinate-dependent low-rank optimization and gauge-induced stochastic selection are explicit in LoRA Done RITE, Singh, and Aladrah et al. | Direct calculation / existing phenomenon | Supplies the exact experimental control equation |
| `prop:app_softmax_dynamics` | Effective flow for jointly trained softmax rows and bases | Value--softmax flow in Varre--Rofin--Flammarion after mapping values to `B^T` and softmax weights to router rows, plus the shared-basis cross-row term | Direct model-specific calculation | Makes the complete response operator explicit |
| `prop:softmax_saturation` | Instantaneous softmax-logit attenuation | Standard softmax covariance Jacobian and `||C(a)|| <= tr C(a)` | Elementary bound | Explicitly denies a finite-time graph-preservation conclusion |
| `prop:app_symmetry_trap` | Uniform/equal initialization remains symmetric in deterministic flow | Standard invariance of a permutation-symmetric subspace | Elementary symmetry consequence | Identifies the required symmetry breaker; not a prevalence claim |
| `thm:response_stabilizer` | Product-equivalent full-rank charts have identical complete Euclidean response iff they differ by permutation | Jacot supplies Jacobian--Gram response; Marcotte et al. the direct matrix-factor response; Shem-Ur et al. the conceptual same-function/different-NTK boundary; Varre the softmax local block; Singh `GL -> O`; Dorrell et al. prove orthogonal-to-permutation under separate tight-scattering geometry; Afsari supplies the final exact joint-diagonalizer-to-monomial reduction; Lau--Su router label symmetries; Tran et al. dense-MoE function-level permutation/translation; van Oostrum et al. and Mishra et al. show the metric-specific boundary | Candidate addition after combination; not found as a theorem in the audited full texts | Exact finite-width stabilizer only for two jointly learned shared-softmax router/basis charts on one common exact product fiber under the named Euclidean law; the local `J(a)^2` blocks recover a diagonal family without scattering, while its monomial uniqueness is prior art |
| `cor:response_stability` and `thm:response_stability` | Linear inverse bound from response distance to the permutation orbit on a named nondegenerate set | Nguyen et al. give a global conditional-density-to-Voronoi-parameter inverse for Gaussian softmax MoE; Tu et al. give Gram-to-Procrustes stability; Afsari gives cost-specific local first-order sensitivity for approximate joint diagonalizers; Bhaskara--Charikar--Vijayaraghavan give global robust Kruskal stability for bounded tensor decompositions; ordinary perturbation tools are standard | Candidate addition, with bounded-search novelty wording | Explicit constants and degeneration families for the complete model-specific common-product Euclidean-response chain; not first softmax inverse, Procrustes, joint-diagonalization, or robust tensor-factor stability |
| `cor:response_tomography` | `K` probes designed from a common exact `Theta` determine the complete structured response when `D-K>=L`; equality then implies permutation | Novak et al. Sec. 3.3 Eqs. (7)--(9) reconstruct a full finite NTK with `O` identity NTK-vector products under `O -> LD`; general structured matvec recovery in Chiu--Demanet, Halikias--Townsend, Otto, and Amsel et al. | Direct structured operator-probing corollary, not central novelty | Explicit model-specific block separation and query compression from generic `LD` probes to `K` |
| `prop:response_tomography_noise` | Deterministic propagation of response errors through the displayed orthogonal reconstruction | Standard linear inverse perturbation; Chiu--Demanet Prop. 1.5 is the closest general conditioning/error precedent | Direct calculation | Makes the finite intervention numerically checkable; no noisy iff claim |
| `prop:response_tomography_gaussian` | `q=max{K,ceil(L/(D-K))}` Gaussian probes determine the structured response almost surely | Halikias--Townsend Defs. 1.1/2.1 and Lemma 2.2; Otto generic-identifying-set result; related rowwise Gaussian solves in structured recovery | Direct structured-probing corollary | Exact query count for this response block family; old eight-probe norms are not retroactively upgraded |
| `prop:response_gaussian_probes` | Small-ball detection of one fixed response alternative | One-dimensional Gaussian small-ball bound applied through a top singular vector | Standard probabilistic corollary | Clarifies that the released eight random probes test a fixed alternative and do not certify an arbitrary operator |
| `cor:response_inverse_gaussian` | Orbit separation plus the inverse bound yields an absolute-threshold Gaussian detection probability | Immediate combination of `cor:response_stability` with the preceding one-dimensional small-ball bound | Direct corollary | Gives a scoped interpretation for fixed alternatives; not a relative-threshold confidence certificate |

The structured finite-response statements passed proof, counterexample,
prior-art, and three-checkpoint implementation review on 2026-08-23.  Their
classification remains a direct structured-probing corollary; acceptance into
the manuscript does not upgrade them to a central novelty claim or
retroactively upgrade the older fixed-alternative Gaussian artifacts.

## Method inventory

| Method or empirical device | Closest existing method | Classification and allowed claim |
|---|---|---|
| ShareProbe / shared-input response clustering | Ordinary paired representation distances, agglomerative clustering, and the finite-sample certificate above | Diagnostic implementation.  It is not a new identification method unless it beats simple controls; the corrected paired sweep is reported even when it does not. |
| Effective-parameter distance baseline | Direct comparison in the quotient/product representation | Standard invariant baseline; applicable only when the effective tensors are observable. |
| ASLoRA gauge/action audit | ASLoRA's published running-average score plus standard `GL(r)` invariance and gauge-aware LoRA work | System-specific causal audit.  The broad claim that raw-factor merging is gauge dependent is already demonstrated by TwistedMerge. |
| Loewner/PSD pair-selectability code | Trace-one PSD Mahalanobis metric learning (Shen--Kim--Wang) and bounded-distortion metric learning (Liao et al.) | Audit tool, not a new SDP method. |
| Byte-LM partition folding | Effective-weight clustering/folding under a fixed parameter budget | Controlled existence witness only; one of three frozen searches succeeds, so it is not evidence of prevalence. |
| Frozen-checkpoint Euclidean response audit | Jacobian--vector/finite-difference validation of the response formula | Validation protocol for the candidate theorem, not a new automatic-differentiation method. |
| Four-probe response tomography audit | Standard structured operator probing applied to the response block form | Executable validation of complete-response access under exact common-product and rank/dimension assumptions; not an approximate-product theorem or a new probing method. |

## PA-1: unconstrained simplex-structured factorization

**Primary source.** Olivier Vu Thanh, Nicolas Gillis, and Fabian Lecron,
"Bounded Simplex-Structured Matrix Factorization: Algorithms,
Identifiability and Applications," *IEEE Transactions on Signal Processing*
71 (2023), 2434--2447,
[DOI](https://doi.org/10.1109/TSP.2023.3289704),
[arXiv full text](https://arxiv.org/abs/2209.12638).

**Exact location and result.** For nontrivial factor rank `r>=2`, Section
IV-A, printed p. 6, states that unconstrained SSMF is not identifiable and
constructs, for `alpha >= 0`,

```text
W(alpha) = W ((1+alpha) I - (alpha/r) 11^T),
H(alpha) = ((1/(1+alpha)) I
            + alpha/((1+alpha)r) 11^T) H.
```

The middle matrices are inverses, `H(alpha)` remains column stochastic, and
the new factors are not related merely by permutation/scaling when
`alpha > 0`.

**Exact variable map.** Let

```text
X = Theta^T,   W = B^T,   H = A^T,   r = K,
s = alpha/(1+alpha),
M_s = (1-s) I + (s/K) 11^T.
```

Then

```text
H(alpha) = M_s A^T = (A M_s)^T,
W(alpha) = B^T M_s^{-1} = (M_s^{-1} B)^T.
```

Thus the manuscript's uniformizing gauge
`(A,B) -> (A M_s, M_s^{-1} B)` is exactly this published construction after
transpose and substitution.  It is not a new non-identifiability theorem.

The same source's Theorem 1, printed pp. 6--7, summarizes minimum-volume SSMF
identifiability under `rank(W)=r` and a sufficiently scattered coefficient
matrix.  Separability/anchors and minimum-volume geometry therefore belong in
related work; elementary anchor recovery is not an adequate replacement for
a new central result.

**Classification.** Existing result.  The larger local family `I+tH` with
`H1=0`, entropy changes, and argmax changes are useful router-facing
consequences, but share the same change-of-basis mechanism and must not be
presented as the foundational novelty.

## PA-2: LoRA gauge invariance, complete invariants, and optimization

### Putterman et al. (2024)

Theo (Moe) Putterman, Derek Lim, Yoav Gelberg, Stefanie Jegelka, and Haggai
Maron, "Learning on LoRAs: GL-Equivariant Processing of Low-Rank Weight
Spaces for Large Finetuned Models," arXiv:2410.04207v2 (15 Oct. 2024),
[full text](https://arxiv.org/abs/2410.04207).

- Printed pp. 2--3 define the action
  `(U,V) -> (U R, V R^{-T})`.
- Theorems 1--2, printed p. 6, give invariant/equivariant processing results.
- Lemma 3 in Appendix C, printed p. 20, and the formal Theorem 3 on printed
  p. 19 show on the full-rank domain that the tuple of products
  `(U_i V_i^T)_i` separates `GL` orbits.

Map `U_i=B_i`, `V_i=A^T`, and `R=Q`.  ASLoRA's shared `A` restricts the
independent layer gauges to one common `Q`, but product completeness itself is
already known.  The source does not study a row-stochastic shared router,
ASLoRA's directed nearest-pair action, Loewner action stability, or the
Euclidean response stabilizer below.

### LoRA Done RITE (Yen et al., ICLR 2025)

Jui-Nan Yen et al., "LoRA Done RITE: Robust Invariant Transformation
Equilibration for LoRA Optimization," ICLR 2025,
[official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Abstract-Conference.html).

- Definition 1, printed p. 2, defines transformation invariance under
  `(A_2,B_2)=(A_1R,B_1R^{-T})`.
- Algorithm 1 and Theorem 2, printed p. 6, state invariance of the proposed
  optimization rule; Appendix A.4, printed pp. 15--16, proves it.

With their product `Z=A_L B_L^T`, set `A_L=A` and `B_L=B^T`; their action is
`(A,B)->(A R,R^{-1}B)`.  In the router model feasibility additionally requires
`R1=1` and `AR>0`.  This prior work shows that optimizer behavior can depend on
the factor chart and supplies an invariant alternative.  It does not
characterize which transformations preserve the ordinary Euclidean response
at a fixed full-rank shared-softmax factorization.

### LoRA-S (Zheng and Wu, ICLR 2026)

Jinyang Zheng and Tong Wu, "LoRA-S: An Efficient Low Rank Adaptation Scheme
via Sylvester Equation," ICLR 2026,
[official paper](https://openreview.net/forum?id=Guo2XGgxZA).

- Definition 2 identifies `(M,N)` with `(MR,NR^{-T})` at fixed
  `X=MN^T`.
- Equation (7) gives a quotient-invariant metric, Equation (9) computes the
  horizontal lift through a Sylvester equation, and Theorem 1 lifts an
  arbitrarily preconditioned optimizer to a transformation-invariant update.

With LoRA-S's `M_L=A`, `N_L=B^T`, and `R=M`, its factor gauge is exactly
`A'=AM,B'=M^{-1}B` before imposing router feasibility.  It therefore further
precludes novelty claims for factor-chart dependence or invariant optimizer
construction.  It changes the metric/lift so as to preserve the full factor
quotient; it does not classify the stabilizer of the *ordinary Euclidean*
`Z/B` response after the row-softmax map, nor give the present permutation
inverse.

An earlier nearby construction is Fangzhao Zhang and Mert Pilanci,
"Riemannian Preconditioned LoRA for Fine-Tuning Foundation Models," ICML
2024, [official PMLR paper](https://proceedings.mlr.press/v235/zhang24ax.html).
It derives small factor preconditioners from a Riemannian low-rank metric and
analyzes feature-learning stability.  This is additional precedent for
geometry-aware factor optimization, not an ordinary-Euclidean response
stabilizer on a row-simplex shared-router fiber.

### W2T (Han et al., 2026)

Xiaolong Han et al., "W2T: LoRA Weights Already Know What They Can Do,"
arXiv:2603.15990v1 (16 Mar. 2026),
[full text](https://arxiv.org/abs/2603.15990).

- Proposition 3.1, printed p. 3, proves invariance of its canonical
  decomposition under `(B,A)->(BG,G^{-1}A)`.
- Proposition 3.2, printed p. 3, shows that QR plus a small-core SVD recovers
  the same singular values/subspaces as a dense decomposition of `Delta W`.
- Appendix A, printed p. 12, explicitly retains sign ambiguities and rotations
  inside degenerate singular subspaces; deterministic post-processing is a
  numerical convention, not a proof of globally unique historical factors.

The paper uses invariant product features for adapter retrieval.  It contains
no nearest-pair action-set theorem, Loewner/semidefinite characterization,
ASLoRA directed tie, or Euclidean response-stabilizer theorem.

### Gauge-aware federated LoRA (Chen--Liu--Zhu, 2026)

Jinqian Chen, Chang Liu, and Jihua Zhu, "Beyond Factor Aggregation:
Gauge-Aware Low-Rank Server Representations for Federated LoRA,"
arXiv:2605.06733v1 (7 May 2026),
[full text](https://arxiv.org/abs/2605.06733).

- Definition 1, printed p. 3, defines gauge-equivalent factors.
- Definition 2, printed p. 4, defines a gauge-invariant server update.
- Propositions 1--2, printed p. 5, establish exact dense averaging when the
  server rank spans the union subspace and gauge invariance, respectively.

This paper already makes the general semantic point that raw-factor
aggregation can change with gauge.  Its action is federated aggregation, not
within-model nearest-pair layer tying; it contains no Loewner/SDP action set or
fixed-Euclidean shared-router response theorem.

### TwistedMerge (Gong--Xu, 2026)

Ting Gong and Shitan Xu, "TwistedMerge: Certified Higher-Order Diagnostics
and Abstention for Model Merging," arXiv:2607.20887v1 (23 Jul. 2026),
[full text](https://arxiv.org/abs/2607.20887).

Section 3.2, printed pp. 15--16, audits trained rank-4 residual adapters in
five groups of eight.  It applies orthogonal gauges, positive diagonal gauges
with condition number at most 8, and dense gauges with condition number at
most 30 (20 scrambles per family; 300 rows total).  Table 7, printed p. 16,
shows that naive factor averaging changes while whitened global
synchronization and deterministic full-update SVD remain stable.

This is strong prior evidence that naive factor averaging is gauge dependent.
Its target is averaging separate adapters, not choosing and executing an
ASLoRA lower-layer-uses-upper-layer tie.  The full text contains no Loewner
action ordering or nearest-pair merge tree.  Its discussion of regret on
printed p. 20 is a stated missing oracle-regret guarantee, not an unbounded
merge-regret theorem.  The limitations on printed p. 16 restrict the audit to
one residual LoRA-form feature layer over frozen ResNet-18 features and do not
claim a multilayer transformer result.

### Karuturi et al. workshop poster (unresolved technical audit)

Siddharth Karuturi, Kaustubh S. Bukkapatnam, Laksh Patel, and Tanush Ajay
Shastry, "The GL(r) Gauge Symmetry of LoRA: Principal Bundle Structure, Loss
Landscape Geometry, and Implications for Adapter Merging," ICML 2026 Workshop
on Weight-Space Symmetries poster,
[official OpenReview record](https://openreview.net/forum?id=YMsbYXDtqw).

The official record establishes the title, authors, venue, and poster status.
During this audit the primary PDF and API were blocked by an access challenge,
and no author-hosted full-text copy was located.  Consequently its coverage of
maximal invariants, nearest-pair selection, Loewner order, an SDP margin,
merge regret, ASLoRA, or the response theorem is **unresolved**.  No novelty
claim may rely on an inference from its title.  A later audit must obtain and
read the primary full text.

## PA-3: ASLoRA's actual rule and the coordinate-to-action estimand

### Version separation

The method details audited here come from Junyan Hu, **Xue Xiao**, Mengqi
Zhang, Yao Chen, Zhaochun Ren, Zhumin Chen, and Pengjie Ren, "ASLoRA: Adaptive
Sharing Low-Rank Adaptation Across Layers," arXiv:2412.10135v2 (16 Dec. 2024),
[full text](https://arxiv.org/abs/2412.10135).

The later formal record is Junyan Hu, **Jiao Xue**, Mengqi Zhang, Yao Chen,
Zhaochun Ren, **Shu Wu**, Zhumin Chen, and Pengjie Ren, *Pattern Recognition*
180 Part B (Dec. 2026), article 114072,
[DOI](https://doi.org/10.1016/j.patcog.2026.114072).  Its public preview
supports only the broad global-`A`/selective-`B` description; the journal full
text was not available to this audit.  Detailed equations, configuration, and
algorithm claims must therefore cite the arXiv-v2 key `hu2024aslorav2`, not
attribute unverified details to the formal version `hu2026aslora`.

### Exact method evidence in arXiv v2

- Equation (1), printed p. 3: `h=W_0 x+B A x`.
- Equation (2), printed pp. 3--4: one `A` is shared and layer-specific `B_i`
  factors remain.
- Equation (3), printed p. 4: historical running average
  `bar B_i^t=(1/t) sum_{k=1}^t B_i^k`.
- Equation (4), printed p. 4: raw Euclidean/Frobenius distances between the
  averaged `B_i` factors rank candidates.
- "Weight Merging," printed p. 4: after selection, the lower layer uses the
  upper layer's `B`.
- Algorithm 1, printed p. 4: merging is eligible when `t>T_s` at the stated
  interval.  The pseudocode scores all pairs, while the prose immediately
  below restricts the search to adjacent layers.
- Section 3.5, printed p. 5, says sharing `A` removes the influence of `A` and
  its initialization and gives more reasonable/consistent similarity
  calculations for `B`.  This motivates the decision audit, but it is not a
  claim to recover a historical ground-truth sharing graph.
- Table 5, printed p. 12, gives the MRPC setting: rank 8, alpha 16, learning
  rate `4e-4`, batch size 16, 30 epochs, maximum length 512, `T_s=320`, merge
  interval 240, `W_Q/W_V` adaptation, and an update ratio `lambda=0.5`.  The
  available method text never defines an operation using that ratio; the audit
  records the value in provenance but does not reverse-engineer a rule.

The source has operational ambiguities that an experiment must report rather
than silently resolve: all-pairs versus adjacent-pairs eligibility; whether
`Q` and `V` projections have separate or pooled graphs; and how a tied layer's
running history is updated.  The prose dimension for `A` is also incompatible
with the displayed `BA` product; the operational shape is `A in R^{r x d}`.

### Gauge map and exact quantities

Use the common gauge

```text
A'       = Q^{-1} A,
B_i'^k   = B_i^k Q,
B_i'^k A'= B_i^k A.
```

For candidate pair `p=(i,j)`, let

```text
D_p = bar B_i - bar B_j,    S_p = D_p^T D_p,    C = Q Q^T.
```

Then its post-gauge raw selection score is

```text
||D_p Q||_F^2 = <C,S_p>.
```

Orthogonal gauges (`kappa=1`) leave every raw distance unchanged and are a
required sanity control.  The historical product-space proxy
`||(bar B_i-bar B_j)A||_F` is gauge invariant, but it is still only a
running-average proxy.  It is not the disturbance caused by the current
directed merge.

For a current action in which lower layer `l` uses upper layer `u`, the exact
weight-update perturbation is

```text
||(B_l^t-B_u^t) A^t||_F.
```

An activation-weighted RMS disturbance additionally requires the second
moment of the input to the particular lower layer; moments from another layer
or from the historical average do not estimate the same action.  Full
validation loss/accuracy/F1 after each candidate tie is the direct downstream
measurement.  Without assumptions that connect historical averages to
current factors and control layer-specific activation moments, there is no
generic finite running-score-to-current-action regret bound, even in rank one.

**Classification.** The common-gauge ASLoRA nearest-pair audit is a concrete
coordinate-to-decision application.  It does not establish historical truth
recovery, and the running-average distance must not be mislabeled as current
merge cost.

## PA-4: Loewner stability and PSD action selection

For candidates with scatters `S_p`, pair `p` beats pair `q` under every SPD
gauge metric precisely when `S_q-S_p` is positive semidefinite (strict/tie
wording must match the chosen quantifiers).  The gauge-selectable set can be
tested with the trace-normalized program

```text
kappa_p = max_{C >= 0, tr(C)=1} min_{q != p} <C,S_q-S_p>.
```

Positive margin gives a separating PSD metric; requiring `C` to be strictly
positive definite may turn a maximum into a supremum at the boundary and must
be handled explicitly.  This convex duality is exact and useful, but it is not
new metric-learning geometry.

**Closest metric-learning source.** Chunhua Shen, Junae Kim, and Lei Wang,
"Scalable Large-Margin Mahalanobis Distance Metric Learning," *IEEE
Transactions on Neural Networks* 21(9), 1524--1530 (2010),
[DOI](https://doi.org/10.1109/TNN.2010.2052630),
[preprint](https://arxiv.org/abs/1003.0487).  Equation (4), printed p. 2,
maximizes a margin over a PSD matrix with `tr(X)=1` and linear triplet
inequalities/slack; the surrounding text identifies the semidefinite program
and explains that trace normalization removes scale ambiguity.  Map their
metric `X` to `C` and a triplet scatter difference to `S_q-S_p`.

**Condition-bounded source.** Renjie Liao, Jianping Shi, Ziyang Ma, Jun Zhu,
and Jiaya Jia, "Bounded-Distortion Metric Learning," arXiv:1505.02377 (2015),
[full text](https://arxiv.org/abs/1505.02377).  Proposition 1, printed p. 3,
identifies metric distortion with `cond(M)`; Eqs. (2)--(3), printed p. 4,
bound that condition number; Eq. (4), printed p. 5, uses
`alpha I <= M <= alpha K I`.  Map their `M` to the gauge metric `C`.

**Classification.** Application/audit tool.  A selectable-action enumeration
for ASLoRA may be useful, but the PSD margin and bounded-distortion LMI are
standard.  A worst-case action-cost maximum over selectable actions is a
definition once costs are supplied, not by itself a new regret theorem.

## PA-5: Euclidean tangent response of the shared softmax-router model

### Candidate theorem and exact target

Let `A=softmax_row(Z)>0`, `rank(A)=K`, `rank(B)=K`, and let both Euclidean
learning rates `eta_Z,eta_B` be positive.  For an effective gradient
`G in R^{n x d}`, the instantaneous effective-parameter response is

```text
R(G)_i = eta_B sum_j <a_i,a_j> G_j
         + eta_Z G_i B^T J_i^2 B,
J_i = diag(a_i)-a_i a_i^T.
```

Consider an equivalent feasible chart
`A'=AM`, `B'=M^{-1}B`, with `M1=1` and `AM>0`.  The exact candidate statement
is:

```text
R_(A,B)(G)=R_(A',B')(G) for every G
if and only if M is a permutation matrix.
```

The candidate contribution is the **combined fixed-factor, finite-width iff
stabilizer theorem**.  It is not the Jacobian-Gram identity, the observation
that an optimizer is chart dependent, the `GL -> O` isometry mechanism, or
softmax label symmetry, all of which have prior art below.

### Closest audited components

1. Arthur Jacot, Franck Gabriel, and Clement Hongler, "Neural Tangent Kernel:
   Convergence and Generalization in Neural Networks," NeurIPS 2018,
   [official paper](https://papers.nips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html).
   Section 4, printed p. 5, gives the finite-parameter Jacobian outer-product
   kernel and function-space gradient response.  Scale the `Z` and `B`
   Jacobian blocks by the square roots of their learning rates and vectorize
   `Theta`.  This derives the response-operator form; Jacot et al.'s Theorem 1
   is instead an infinite-width limit and must not be cited as the stabilizer
   theorem.

2. Sibylle Marcotte, Gabriel Peyre, and Remi Gribonval, "Intrinsic Training
   Dynamics of Deep Neural Networks," ICLR 2026,
   [official paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/184b2454410a88c280d29aa95fbac6e2-Abstract-Conference.html),
   [full text](https://arxiv.org/html/2508.07370).  Equations (3)--(4) give the
   lifted response `Dphi(theta) Dphi(theta)^T`; for
   `phi(U,V)=UV^T`, Eq. (26) gives
   `M[G]=UU^T G + G VV^T`.  Map `U->A`, `V->B^T`, and `z->Theta`.
   This is the direct general matrix-factor response precedent, but it has no
   row-softmax `J(a_i)^2` block, common-product-fiber stabilizer, or
   permutation inverse.

3. Aditya Varre, Mark Rofin, and Nicolas Flammarion, "Gradient Flow Polarizes
   Softmax Outputs towards Low-Entropy Solutions," arXiv:2603.06248v1
   (6 Mar. 2026), [full text](https://arxiv.org/abs/2603.06248).
   Their model `beta=V softmax(a)` and Eqs. (3)--(4), printed p. 3, give the
   parameter gradient flow.  Lemma D.2, printed p. 20, gives the one-row
   effective response `||s||^2 I + V J(s)^2 V^T`.  Map
   `s=a_i`, `V=B^T`, and `beta=Theta_i^T`.  This supplies the rowwise softmax
   term but not the shared-`B` cross-row block `eta_B A A^T G` or an iff
   stabilizer.

4. Devender Singh, "The Loss Does Not See the Basis, but Adam Does,"
   arXiv:2608.05136v1 (5 Aug. 2026),
   [full text](https://arxiv.org/abs/2608.05136).
   Lemma 3.2, printed p. 6 (proof Appendix B.1, printed p. 27), states that
   `(U,V)->(UT,VT^{-T})` preserves the product and preserves the Euclidean
   product metric for all factors iff `T` is orthogonal.  This is the closest
   `GL -> O` mechanism, but it is a global factor-space isometry and not the
   fixed shared-softmax response iff theorem.

5. Tim Tsz-Kit Lau and Weijie Su, "Symmetry-Compatible Principle for
   Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers,"
   arXiv:2605.18106v4 (22 Jun. 2026),
   [full text](https://arxiv.org/abs/2605.18106).
   Section 3.5, printed pp. 13--15, uses
   `p(x;W)=softmax(Wx)` and the router symmetry
   `W -> P W + 1_e a^T`.  Definition 3.4, printed p. 14, and Proposition 3.5,
   printed p. 15, formalize permutation equivariance, shared-row-shift
   invariance, and horizontality.  This establishes optimizer-relevant router
   symmetries but has no jointly learned shared basis or response-stabilizer
   theorem.

6. Nicola Aladrah et al., "Understanding and Inverse Design of Implicit Bias
   in Stochastic Learning: A Geometric Perspective," arXiv:2601.06597v2
   (4 Apr. 2026),
   [full text](https://arxiv.org/abs/2601.06597).  Its quotient-measure theorem
   and displayed effective loss derive an orbit-volume correction for noisy
   stochastic dynamics; its architecture examples include balanced matrix
   factorizations and attention rescalings.  The target is the stationary
   distribution or implicit preference along symmetry orbits, not the
   stabilizer of the deterministic, complete instantaneous response operator
   at two fixed shared-softmax charts.  It therefore strengthens the prior-art
   boundary for optimizer selection without implying the candidate iff
   theorem below.

7. Viet-Hoang Tran, Van Hoan Trinh, Khanh-Vinh Bui, and Tan M. Nguyen, "On
   Linear Mode Connectivity of Mixture-of-Experts Architectures," NeurIPS
   2025, [official paper](https://papers.nips.cc/paper_files/paper/2025/hash/fad7c708dda11f3e72cc1629bb130379-Abstract-Conference.html),
   [full text](https://arxiv.org/html/2509.11348v2).  Their Theorem 4.1 proves
   that equality of a dense MoE function on the input domain, pairwise
   distinct experts, and distinct gate-weight differences imply equality up
   to one expert permutation and one common affine gate translation.  This
   already supplies the same abstract label/shift group for a different
   function-level observable.  It rules out any claim of first generic
   softmax-MoE permutation-plus-shift identifiability, but it does not imply a
   shared `A B` chart stabilizer from `(Theta,R)`.

8. Huy Nguyen, TrungTin Nguyen, and Nhat Ho, "Demystifying Softmax Gating
   Function in Gaussian Mixture of Experts," NeurIPS 2023,
   [official paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0ef6ffcb85a2d238fc4761860c31ded4-Abstract-Conference.html),
   [full text](https://arxiv.org/html/2305.03288v2).  Proposition 1 identifies
   the exact-fitted conditional density up to labels and a common gate
   translation.  Their Theorem 1 gives a global linear lower bound from
   expected Hellinger distance to a translation-quotiented Voronoi parameter
   metric on a compact parameter set.  This is a genuine softmax-gated global
   inverse precedent, but its conditional-density observation and Gaussian
   expert assumptions do not yield the common-product Euclidean-response
   inverse here.

9. Stephen Tu, Ross Boczar, Max Simchowitz, Mahdi Soltanolkotabi, and Ben
   Recht, "Low-rank Solutions of Linear Matrix Equations via Procrustes
   Flow," ICML 2016, [official paper](https://proceedings.mlr.press/v48/tu16.html).
   Lemma 5.4 bounds Procrustes factor distance linearly by Gram error divided
   by the smallest factor singular value.  It directly precedes the
   Gram-to-orthogonal step of the quantitative proof; it contains neither the
   softmax `O(K)->S_K` step nor the complete response inverse.

10. Jesse van Oostrum, Johannes Muller, and Nihat Ay, "Invariance Properties
    of the Natural Gradient in Overparametrised Systems," *Information
    Geometry* 6(1):51--67 (2023),
    [official paper](https://doi.org/10.1007/s41884-022-00067-9), and Bamdev
    Mishra et al., "Fixed-Rank Matrix Factorizations and Riemannian Low-Rank
    Optimization," *Computational Statistics* 29:591--621 (2014),
    [full text](https://arxiv.org/html/1209.0430v2).  The first paper's
    Theorems 1--2 make the natural-gradient pushforward a tangent projection
    and invariant under diffeomorphic reparameterization; the second
    explicitly quotients `(G,H)~(GM^-1,HM^T)` under invariant metrics.  On a
    feasible full-rank `A'=AM,B'=M^-1B` fiber, the product tangent images
    agree, so a natural/quotient response preserves the full gauge rather than
    selecting permutations.  This is an exact counterexample to
    metric-independent extension of the candidate theorem.

11. Hong Wang and Kelly Wang, "Complete Characterization of Gauge Symmetries
    in Transformer Architectures" and "Gauge Fiber Bundle Geometry of
    Transformers," NeurReps 2025,
    [official record](https://neurips.cc/virtual/2025/136893),
    [geometry record](https://openreview.net/forum?id=sPCLRX1yOY).  The first
    characterizes the generic standard-attention function gauge as headwise
    query--key and value--output `GL` actions semidirect head permutations;
    the second studies the quotient/Fisher geometry and horizontal natural
    gradient.  These are direct Transformer gauge precedents, but the factors
    are Q/K and V/output projections and the observation is the function or
    quotient geometry, not equality of complete Euclidean response for a
    row-simplex shared-router `AB` chart.

12. Gautam Goel, Mahdi Soltanolkotabi, and Peter L. Bartlett, "Training
    Dynamics of Softmax Self-Attention: Fast Global Convergence via
    Preconditioning," arXiv:2603.01514v1 (2026),
    [full text](https://arxiv.org/html/2603.01514).  Their Theorem 1 writes the
    Gaussian population loss as
    `||A Sigma B^T Sigma^(1/2)-M Sigma^(1/2)||_F^2/2` and balances it with a
    regularizer; the resulting optimum manifold is orthogonally parameterized.
    This is a genuine softmax-attention-to-matrix-factor optimization
    precedent.  The Gaussian expectation cancels the softmax ratio before the
    factor loss is formed.  Its orthogonally parameterized optimum manifold
    does not classify the present row-simplex shared-router/common-product
    Euclidean-response stabilizer or give its inverse, so it does not imply
    the candidate theorem.

13. William Dorrell, Peter E. Latham, and James C. R. Whittington, "Convex
    Efficient Coding," ICLR 2026,
    [official record](https://openreview.net/forum?id=Se3YaqtjqE),
    arXiv:2601.10482v3.  Their Theorem 2 (printed p. 8; proof Appendix D.2,
    printed pp. 33--34) starts from two nonnegative representation matrices with
    the same sample Gram, writes `Z'=OZ`, and uses two tight-scattering /
    convex-hull assumptions to force the orthogonal matrix `O` to be a
    permutation.  The exact intermediate mapping is `Z=A^T`, `Z'=(A')^T`,
    and `O=M^T` after our cross-row response block has yielded
    `AA^T=A'A'^T`.  Strict positivity and `rank(A)=K` do not imply their
    scattering assumptions.  Their first condition requires the finite
    representation hull to contain
    `E_F={mu+x:x^T F^{-1}x=1}` for some `F>0` with
    `diag(F)=mu odot mu`.  For coordinate `j`, minimizing `x_j` on this
    ellipsoid gives `-sqrt(F_jj)=-mu_j`, so `E_F` touches `y_j=0`; meanwhile
    every point in the convex hull of strictly positive router rows has
    `y_j>=min_i A_ij>0`.  Thus strict positivity actually precludes this
    first scattering condition rather than implying it.  Our local
    softmax-covariance response blocks, rather than scattering, rule out the
    residual orthogonal ambiguity.  Theorem 2 states only this sufficient
    exact orthogonal-to-permutation condition and supplies no
    response-to-chart stability theorem.  It is
    therefore a direct orthogonal-to-permutation mechanism precedent, not a
    proof of the candidate response theorem.

14. Bijan Afsari, "Sensitivity Analysis for the Problem of Matrix Joint
    Diagonalization," *SIAM Journal on Matrix Analysis and Applications*
    30(3):1148--1171 (2008),
    [official article](https://doi.org/10.1137/060655997).  Theorem 2.3 gives
    essential uniqueness of a nonorthogonal exact joint diagonalizer up to
    permutation and diagonal scaling iff no two columns of the matrix of
    diagonal entries are collinear.  Once our proof has established
    `M^T Diag(a_i) M=Diag(a_i')` for every row, its diagonal-entry matrix is
    exactly `A`; full column rank implies Afsari's pairwise noncollinearity
    condition.  The resulting monomial gauge is already reduced to a
    permutation by `M 1=1`.  Section 4 and Theorems 4.1--4.2 additionally give local,
    first-order and cost-specific sensitivity of stationary approximate
    joint diagonalizers.  These results directly precede the final monomial
    reduction and its perturbative component.  They do not recover the
    router Gram and the family of diagonal congruences from `(Theta,R)`, nor
    do they give the manuscript's global compact-set inverse, conditioned on
    a common exact effective product, from response distance to the factor
    permutation orbit.

15. Aditya Bhaskara, Moses Charikar, and Aravindan Vijayaraghavan,
    "Uniqueness of Tensor Decompositions with Applications to Polynomial
    Identifiability," COLT 2014,
    [official PMLR record](https://proceedings.mlr.press/v35/bhaskara14a.html).
    Theorem 5 (printed p. 7) is a robust Kruskal theorem: two bounded rank-`R`
    order-three decompositions that are sufficiently close have factor
    matrices polynomially close up to one permutation and modewise diagonal
    rescalings when their robust Kruskal ranks sum to at least `2R+2`.
    Stack our approximate diagonal congruences as a tensor.  Its two
    decompositions are `[I,I,A']` and `[M^T,M^T,A]`; the reference robust
    Kruskal ranks are `K,K,K` under the router singular-value margin, and
    `3K>=2K+2` for `K>=2`.  Thus the global robust reduction from the close
    diagonal tensor family to a near permutation-and-scaling factorization is
    also prior art, not merely a local Afsari phenomenon.  This theorem starts
    from close tensor decompositions: it neither derives them from the full
    Euclidean response nor enforces the common exact effective product, and it
    does not supply the manuscript's tailored explicit linear constant for
    the simultaneous router/basis orbit.

16. Ori Shem-Ur, Jakub Vrabel, Robin Walters, and Yaron Oz, "Neural Tangent
    Kernel Perspective on Parameter-Space Symmetries," ICLR 2026 submission,
    [official OpenReview record](https://openreview.net/forum?id=InuV3Lh5bc).
    The indexed public PDF states in Theorem 4.1 (printed p. 6) that applying a
    function-preserving parameter-space symmetry at an arbitrary training step
    preserves the NTK linearization limit, explicitly "albeit with a different
    kernel."  This is a direct conceptual precedent that function-equivalent
    charts can have different Euclidean NTK/response geometry.  It gives no
    converse classification of charts with equal kernels, no shared-simplex
    `AB` calculation, and no response inverse.  At the audit date OpenReview's
    direct PDF endpoint returned a browser challenge, so only the official
    metadata and search-indexed theorem text were adjudicated; no uninspected
    technical claim is inferred from the title.

### Assumption and counterexample ledger

- `A>0` is automatic for finite softmax logits, but it is not needed by the
  exact algebra once full router and basis rank are imposed: singularity and
  positive semidefiniteness of each simplex covariance suffice.  A uniform
  lower bound `A_ij>=alpha` is needed by the quantitative inverse to keep the
  covariance gap away from zero.
- Full router rank cannot be dropped uniformly: at `K=3`, uniform/repeated
  rows admit non-permutation orthogonal rotations fixing `1` while preserving
  the response.  This counterexample does not assert necessity separately at
  every `K`.
- Full basis rank likewise cannot be dropped uniformly: at `K=3`, a positive
  full-rank router with a basis supported only in `span{1}` admits invisible
  non-permutation orthogonal rotations fixing `1`.
- `eta_Z>0` cannot be dropped uniformly for `K>=3`: if `eta_Z=0`, the response
  depends only on `AA^T`, so a small non-permutation orthogonal rotation fixing
  `1` remains invisible while preserving positivity.  At `K=2`, the
  orthogonal stabilizer fixing `1` contains only the two permutations, so this
  example does not establish necessity.
- `eta_B>0` is used by the current proof to expose the cross-row Gram block.
  With `eta_B=0`, permutation rigidity is proved for arbitrary signed gauges
  at `K=2` and for every `K` under an entrywise-nonnegative gauge.  The general
  signed `K>=3` case remains blocked; necessity must not be asserted.
- The theorem allows `n=K` and `d=K`; it does not require strict overwidth.
- `M` need not be entrywise nonnegative.  The feasibility assumptions are
  exactly `M1=1`, invertibility, and `AM>0`.
- Softmax logit row shifts remain a `Z`-level redundancy; the theorem concerns
  the router matrix `A` and the basis chart.
- Equality is quantified over **all** effective gradients.  One observed
  training direction or trajectory cannot establish operator equality.
- Equality of `Theta` is indispensable.  The response is quadratic in `B`, so
  `(A,B)` and `(A,-B)` have equal response but opposite effective products.

### Narrow novelty and interpretation statement

The audited full texts above do not contain the combined theorem that, on one
common exact product fiber, the complete Euclidean tangent-response stabilizer
of an interior, full-rank shared-softmax-router/basis factorization is exactly
the permutation group.
The manuscript may therefore say that it **derives this combined stabilizer
result and did not find it in the audited sources**.  It must not say that it
invented NTK or matrix-factor response, discovered optimizer coordinate
dependence, first identified softmax labels/shifts, first proved a global
softmax inverse, or recovered historical sharing truth.

The theorem identifies the current factor chart relative to a specified
Euclidean `Z/B` training law.  It does not cover AdamW without a separate
analysis, prove stable selection by finite training, recover a generating
sharing graph, or validate a downstream merge.  Gaussian probes detect any
fixed nonzero response difference with high probability; they do not give a
uniform finite-sample certificate over all arbitrarily close charts.

The manuscript also proves an explicit linear inverse bound from response
operator error to permutation-orbit distance on a uniformly interior,
full-rank, norm-bounded set.  Its proof was independently line-audited and
includes the required rank, boundary, exact-product, and observation
counterexamples.  No such quantitative inverse for this combined response
model on a common exact product fiber was found in the audited primary
sources.  This is a scoped novelty
assessment, not a claim that inverse-stability arguments, Sylvester equations,
or Gaussian fixed-alternative tests are themselves new.  The probe bound is
pointwise in a fixed alternative; fewer than `LD` probes cannot uniformly
certify an arbitrary response operator, while the known structured family
below can be determined from fewer probes.

### Finite structured observation: exact prior-art map

The manuscript's finite corollary assumes the exact common product and the
rank conditions above.  With `S=row(Theta)`, it writes the response as the
router Gram `Q=AA^T` plus local blocks supported on `S`.  Under `D-K>=L`, one
probe codes the `L` columns of `Q` in orthonormal directions of `S^perp`, and
the remaining `K-1` probes recover every local block on an orthonormal basis
of `S`.  This yields `K` explicit probes.  The construction is useful, but the
observation mechanism is established prior art:

1. Roman Novak, Jascha Sohl-Dickstein, and Samuel S. Schoenholz, "Fast Finite
   Width Neural Tangent Kernel," ICML 2022, PMLR 162:17018--17044,
   [official paper](https://proceedings.mlr.press/v162/novak22a.html).
   Section 3.3, Eqs. (7)--(9), states that `O` identity NTK-vector products
   reconstruct the complete `O x O` NTK.  Map `O -> LD`, the finite NTK to
   `J diag(eta_Z I,eta_B I) J^T`, and an identity column to `vec(G)`.  This is
   the direct general response-reconstruction precedent.  The manuscript adds
   only model-specific compression from generic `LD` probes to the displayed
   structured count.

2. Jiawei Chiu and Laurent Demanet, "Matrix Probing and its Conditioning,"
   *SIAM Journal on Numerical Analysis* 50(1):171--193 (2012),
   [DOI](https://doi.org/10.1137/110825972).  Their setup
   `A=sum_j c_j B_j`, Theorem 1.3, and Proposition 1.5 give coefficient
   recovery, conditioning, and error propagation from random matvec probes.
   Map their operator basis to the known block operators spanning
   `(Q,T_1,...,T_L)`.  Their result supplies the general method, not the
   response-specific orthogonal design or permutation conclusion.

3. Diana Halikias and Alex Townsend, "Structured Matrix Recovery from
   Matrix-Vector Products," *Numerical Linear Algebra with Applications*
   31(1):e2531 (2024), [DOI](https://doi.org/10.1002/nla.2531),
   [full text](https://arxiv.org/abs/2212.09841).  Definitions 1.1 and 2.1
   formalize exact query complexity and linear matrix families; Lemma 2.2
   gives the parameter-count lower bound.  Their cited generic-identifying-set
   mechanism is Samuel E. Otto, "A Note on Recovering Matrices in Linear
   Families from Generic Matrix-Vector Products" (2023),
   [primary record](https://doi.org/10.5281/zenodo.7916776).  These results
   support the almost-sure generic-probe corollary once an identifying design
   exists; they do not contain this response block family.

4. Noah Amsel et al., "Query Efficient Structured Matrix Learning," formally
   published at COLT 2026, PMLR 336:158--194,
   [official publication](https://proceedings.mlr.press/v336/amsel26a.html).
   Theorem 1 treats finite families, Theorem 4 extends through covering
   numbers, and Definition 2/Corollary 1 cover approximate learning of linear
   matrix families with access to the operator and its transpose.  Because the
   present response operator is self-adjoint, sidedness is not the substantive
   distinction.  This is broader approximate-query theory, not the same exact
   model-specific reconstruction or a chart-identification theorem.

The exact classification is therefore **direct structured operator-probing
corollary**.  The model-specific formulas and three-checkpoint, four-probe
analytic full-response audit make the all-gradient observable executable, but are not
promoted as a second central theoretical innovation.  The old eight Gaussian
probes remain only a pointwise fixed-alternative test because their artifacts
did not perform and record the rank reconstruction.

Finally, Daniel Stupariu and Andrei Manolache, "How the Optimizer Shapes
Learned Solutions in Equivariant Neural Networks," arXiv:2605.27662 (ICML
2026 Workshop on Weight-Space Symmetries), was audited from its
[primary full text](https://arxiv.org/abs/2605.27662).  Its Sections 2.1--2.3
are empirical Adam/Muon comparisons using performance, Hessian/loss-landscape,
and rank diagnostics.  It contains no response-isometry, stabilizer,
quantitative inverse, or tomography theorem.  This negative finding is based
on the full text, not poster metadata, and is not a priority claim.

## PA-6: reality and scope of the community claim

The literature audit does not support a broad claim that learned cross-layer
sharing papers commonly assert recovery of a historical or generating
sharing graph.  Most papers in the manuscript's corpus claim task quality,
parameter efficiency, compression, or an effective architecture.

### Additional full-text audit of recent systems

**E2LoRA (Li et al., ICLR 2026).**  The primary 27-page OpenReview paper,
"E2LoRA: Efficient and Effective Low-Rank Adaptation with Entropy-Guided
Adaptive Sharing," was audited from the official submission record
[`IQttyo0460`](https://openreview.net/forum?id=IQttyo0460).  Section 3.1,
printed pp. 4--5, obtains layer gradients on a small task-training subset
*before fine-tuning*, converts their elementwise spread into proxy entropy and
RMI, and greedily partitions consecutive layers.  Algorithm 1, printed p. 6,
then assigns a rank to each fixed interval before post-tuning.  Section 5,
printed p. 10, calls the allocation static and reusable; Appendix I, printed
pp. 23--24, reports identical interval/rank decisions for three sampled-subset
seeds and evaluates the resulting trained models.  The paper describes local
similarity as an intrinsic property and its decisions as structurally stable,
but contains no historical, ground-truth, or recovery target.  Most
importantly for scope, its decision variable is a pre-fine-tuning task-gradient
proxy, not a coordinate of jointly learned factors.  It is a relevant positive
example of a proxy-to-hard-allocation chain with stability/downstream checks,
not evidence for ambiguity or misuse of a learned `Theta=AB` router.

**RaSA (He et al., ICLR 2025).**  The official full text
([proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/b4fd162d3e2d015233486a2e313828a7-Paper-Conference.pdf)),
Section 2.1, printed pp. 2--3, defines a global rank pool assembled across all
layers plus layer-specific diagonal weights.  The cross-layer sharing pattern
is an architectural choice, not an inferred partition.  Sections 3--4 assess
matrix reconstruction and downstream performance.  It makes no historical
graph claim and cannot be counted as a learned-coordinate offender.

**LiSA (Mu et al., TACL 2026).**  The primary full text
([ACL Anthology PDF](https://aclanthology.org/2026.tacl-1.30.pdf)), Section 3,
printed pp. 659--662, measures current attention-score similarity across
layers and tasks.  Section 4 directly corrupts/shares adjacent attention
patterns to measure layer sensitivity, and Section 5 trains head-alignment and
low-rank difference-compensation modules before evaluating quality and
throughput.  The paper interprets similarity as a current model property and
compression opportunity, not a historical graph; its object is nonlinear,
input-dependent attention output, not a learned linear factor chart.  It also
supplies an example of measuring downstream consequences rather than treating
a heatmap as self-authenticating.

Concrete coordinate-to-action cases are narrower:

- ASLoRA uses a raw shared-factor distance to select an actual directed tie;
  it motivates the score as reasonable and consistent layer similarity, but
  does not establish historical truth recovery.
- Savarese and Maire (ICLR 2019) use learned template-coefficient geometry to
  infer approximate recurrence and then validate a folded network.
- Neural Parameter Allocation Search / ShapeShifter (ICLR 2022) converts
  learned factor coordinates or embeddings into hard parameter groups, but
  presents the result as budgeted architecture search rather than recovery of
  a planted graph.

The last case is a genuine action chain rather than a paper that merely plots
coefficients.  In the official ICLR text
([author-hosted proceedings PDF](https://htor.inf.ethz.ch/publications/img/plummer-npas.pdf)),
Section 3.1.1, printed pp. 4--5, the WAvg variant learns unconstrained
`alpha_i in R^K` and forms
`w_i=sum_k alpha_{ik} T_i^k`; the Emb variant first forms
`alpha_i=W_j phi_i+b_j`.  Section 3.2, printed p. 5, trains a one-group
low-budget model for 10--15% of the usual training time, then applies k-means
to the learned `alpha_i` or `phi_i` and uses the resulting clusters as the
hard parameter groups of the full model.  Table 7, Figure 4, and Appendix
Table 9 compare those executed groupings with single, random, manual, and
unshared alternatives on task metrics; the paper reports parity or small
improvements rather than treating the clustering as self-validating.
Consequently this is direct evidence that learned coordinates can affect a
structural allocation, but not evidence of historical-truth recovery.  It is
also not an exact instance of the present common-`B` simplex theorem in
general: the coordinates are unconstrained, and the templates are
layer-dependent slices/resizings of a shared parameter store, potentially
across heterogeneous weight shapes.  The ordinary factor gauge applies only
to compatible fixed-template subcases, so no blanket `Theta=AB` counterclaim
is made against the full ShapeShifter architecture.

Dynamic Layer Tying, Basis Sharing, FiPS, and predefined recursive-tying work
must not be cited as examples of a soft-coordinate truth-recovery error absent
paper-specific text.  Positive planted-recovery work likewise is not evidence
of the criticized mistake.  The defensible empirical narrative is therefore:

MASA / "Share Your Attention" (Zhussip et al., AAAI 2026) is a mixed boundary
case.  In the from-scratch path, Equations (1)--(2) give
`W_hat = D C`, with `D` and `C` jointly trained, scalar coefficients in
`R`, and no imposed atom constraint; the authors explicitly decline the atom
soft constraints just described (AAAI p. 29262).  During training a block
embedding and three-layer MLP produce each coefficient vector, after which the MLP and
embeddings are discarded and final `C` is retained (AAAI p. 29263).  The exact
map to our static factor notation is

```text
Theta = W_hat^T,    A = C^T,    B = D^T,
(D,C) -> (D R, R^{-1} C)  <=>  (A,B) -> (A R^{-T}, R^T B).
```

This is the ordinary unconstrained `GL(S)` factor gauge at inference, not an
SSMF/simplex-router model; because training uses the embedding--MLP
parameterization, it is also not the manuscript's direct Euclidean `Z/B`
response law.

The main paper says coefficients indicate atom contribution and shared atoms
capture cross-layer statistical regularities.  Crucially, the arXiv v2
supplement goes beyond a pure compression claim: supplementary p. 14,
Equation (18), interprets atom cosine as distinct/complementary versus
redundant components, and the immediately following "Visualization of Mixing
Coefficients" says raw-`C` heatmaps reveal specialization, redundancy, and
layer-wise adaptivity.  Those claims are representation-dependent under the
displayed `GL(S)` map.  They are nevertheless descriptive claims about the
current factor chart, not recovery of a historical/generative graph: MASA
defines no such target, reports no recovery metric, and derives no hard action
from `C`.

Its pretrained-model path is separate.  A vocabulary-output semantic probe and
consecutive-layer KL divergences choose contiguous shared-dictionary groups
(main-paper p. 4 / AAAI p. 29263; arXiv-v2 supplementary pp. 10--11), after
which orthonormal matrix PCA and local residual refinement are evaluated on
perplexity and downstream accuracy.  The action is therefore output-derived
and validated, not coefficient-derived.  Orthonormal PCA removes arbitrary
scale/shear but leaves an orthogonal basis freedom for the retained subspace,
so the scratch `GL(S)` statement must not be transferred verbatim to this
branch.  MASA establishes that weak coordinate interpretation occurs, while
remaining negative evidence for the stronger historical-recovery and
coordinate-to-hard-action narrative.  It does not alter the scoped ASLoRA
decision audit.

```text
some real systems make structural actions from factor coordinates;
the ASLoRA first-merge rule is a concrete auditable case;
the paper tests whether equivalent charts select different actions and whether
those actions have measurably different validation consequences.
```

This is a scoped decision-reliability claim, not a community-wide indictment.

Khilar, *Cross-Layer Subspace Coupling for LLM Compression* (arXiv:2605.30836,
2026), is a close empirical negative-neighbor for that decision-reliability
story.  Its Pythia 1.4B experiment reports a 37.2% improvement over per-layer
SVD-LLM in two-sided Frobenius reconstruction but perplexity 159.14 versus
50.17, with total per-layer activation error (5.76\times10^8) versus
(1.93\times10^8).  Thus the broad conclusion that a weight-space structural
proxy can mis-rank deployed compression decisions, and that activation or
downstream checks are necessary, is prior art.  Its action is selected from a
Grassmannian weight-reconstruction objective rather than a learned router
coordinate, and it does not hold the effective model fixed across alternative
charts.  It therefore does not imply the common-product gauge theorem or the
ASLoRA same-function--different-action audit.

CommonKV (Wang et al., arXiv:2508.16134, 2025) is another positive control for
the same reporting standard.  It obtains shared and layer-specific KV factors
by concatenated SVD, but selects which fixed adjacent groups to merge from
cosine similarity of the *latent cache activations*, then reports LongBench,
RULER, memory, and latency results.  It neither reads a historical graph from
the factors nor uses their raw coordinates as the decision score.  The author
repository is public at `github.com/rommel2021/CommonKV`; the ICLR 2026 record
was withdrawn, so this ledger treats it only as a public preprint/code
artifact, not as accepted-conference evidence.

## Citation and evidence discipline

1. Cite `vuthanh2023bounded` at the non-identifiability theorem/construction,
   not only in related work.
2. Cite `hu2024aslorav2` for ASLoRA equations, Algorithm 1, and Table 5; reserve
   `hu2026aslora` for metadata or claims visible in the formal record.
3. Distinguish exact effective products, historical product proxies, current
   directed weight cost, activation-weighted disturbance, and validation loss.
4. Do not infer a technical result from an abstract, title, poster listing, or
   inaccessible PDF.  Keep Karuturi et al. unresolved until the full text is
   obtained.
5. For every promoted theorem, record the closest theorem number/location,
   assumptions, conclusion, exact variable map, counterexamples when an
   assumption is removed, and what cannot be obtained by transpose or a
   one-line corollary.
