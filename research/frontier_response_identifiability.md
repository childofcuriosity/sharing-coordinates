# First-pass literature verification for the handover

Access date: 2026-09-24. Submission cutoff: unspecified.
Scope: selected direct precedents for parameter-factor ambiguity, coordinate-dependent
optimization, lifted dynamics, and robust identifiability. This is not an exhaustive
novelty audit. Existing research ledgers are useful leads, not independent verification.

| Work | Primary source opened | What was checked | Remaining work |
|---|---|---|---|
| Olivier Vu Thanh, Nicolas Gillis, Fabian Lecron, *Bounded Simplex-Structured Matrix Factorization: Algorithms, Identifiability and Applications*, TSP 71:2434–2447 (2023) | [University repository](https://orbi.umons.ac.be/handle/20.500.12907/45976), [author PDF](https://arxiv.org/pdf/2209.12638) | Metadata and full-text access; unconstrained simplex model and identifiability discussion located | Independently transcribe Section IV-A construction and check the exact change of variables |
| Jui-Nan Yen et al., *LoRA Done RITE: Robust Invariant Transformation Equilibration for LoRA Optimization*, ICLR 2025 | [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Abstract-Conference.html) | Title, authors, venue and abstract; factor-coordinate-dependent updates and invariant preconditioning are explicit | Read theorem assumptions and compare against the softmax-router response stabilizer; abstract alone does not settle overlap |
| Sibylle Marcotte, Gabriel Peyré, Rémi Gribonval, *Intrinsic training dynamics of deep neural networks*, ICLR 2026 | [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/184b2454410a88c280d29aa95fbac6e2-Abstract-Conference.html) | Metadata and abstract; lifted variables, intrinsic flows and relaxed balancing | Verify the exact matrix-factor formula and whether any converse covers this manuscript's observation model |
| Aditya Bhaskara, Moses Charikar, Aravindan Vijayaraghavan, *Uniqueness of Tensor Decompositions with Applications to Polynomial Identifiability*, COLT/PMLR 35:742–778 (2014) | [Official proceedings](https://proceedings.mlr.press/v35/bhaskara14a.html) | Metadata and abstract; robust Kruskal uniqueness with approximate decomposition recovery | Read Theorem 5 and reconstruct the diagonal-tensor reduction; do not claim equivalence based only on abstract |

Search queries used: `"Bounded Simplex-Structured Matrix Factorization" arxiv`;
`"softmax" "response" "identifiability" shared bases gradient`.
Exact known proceedings URLs were also opened directly. Search results unrelated to
the stated observation model were not used as novelty objections.

Local comparison priorities: Afsari joint diagonalization, Dorrell same-Gram
nonnegative identifiability, softmax mixture identifiability, and the stated
LoRA gauge/action literature. These leads remain to be independently checked in full.

For empirical review, the repository already contains effective-update and local-activation
ASLoRA comparators, plus effective-parameter k-means/Ward folding baselines.
First determine which claim each baseline evaluates; a request for more baselines
must identify the unresolved comparison rather than merely increase their number.

Current conclusion: existing work establishes substantial background mechanisms.
The manuscript's combined exact response-stabilizer and quantitative inverse claims
remain the candidate contribution. No exhaustive uniqueness-of-contribution claim is made.

## Full-text follow-up, 2026-09-24

The entries below supersede the metadata-only status above for the specified
passages, not for the entire papers.

- **SSMF, Section IV-A, PDF page 6:** checked the explicit uniformizing
  construction. Under X=Theta^T, W=B^T, H=A^T and t=alpha/(1+alpha),
  its transformation is exactly M=(1-t)I+t*11^T/K, A'=AM,
  B'=M^-1 B. Thus this path is prior art, including its product preservation;
  it cannot serve as the new central theorem. This comparison concerns the
  unconstrained SSMF subsection, not the bounded-basis identifiability theorem.
  [Primary text](https://arxiv.org/pdf/2209.12638).
- **LoRA Done RITE, Section 2, Definition 1 and equations (4)--(6), PDF
  pages 2--3:** checked the invertible-factor transformation and the finite
  effective-update invariance definition. The gradient-descent scaling example
  already establishes coordinate-dependent optimization. Our narrower target
  is the complete instantaneous response stabilizer for a softmax-constrained
  factor, with fixed block rates. This distinction is an observation-model
  comparison, not a claim that our theorem follows from their optimizer.
  [Primary text](https://proceedings.iclr.cc/paper_files/paper/2025/file/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Paper-Conference.pdf).
- **Bhaskara et al., Theorem 5, PDF page 7:** checked robust third-order
  uniqueness under k_A+k_B+k_C >= 2R+2, bounded factors, and robust Kruskal
  ranks. After our response-to-diagonal-algebra reduction, the exact tensor
  has factors (I,I,A), whose Kruskal ranks sum to 3K, sufficient for K>=2.
  Consequently, terminal tensor uniqueness is established background. A
  quantitative application still needs to track the response reduction,
  factor scalings and shared-product assumptions; citing the theorem alone
  does not supply our displayed constant.
  [Primary text](https://proceedings.mlr.press/v35/bhaskara14a.pdf).

Implication for presentation: emphasize the response-to-Gram and softmax
covariance reduction and its conditioning. Do not advertise generic gauge
ambiguity, optimizer coordinate dependence, or diagonal/tensor uniqueness as
new.

### Joint-diagonalization full-text follow-up

Retrieved the author's university-hosted manuscript after the publisher page
exposed only metadata: [Afsari, full text](https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf).
Read Theorem 2.3 (PDF page 5), Theorem 4.1 (page 13), and Theorem 4.2
(page 15). The uniqueness modulus excludes collinear diagonal profiles;
the two perturbation theorems concern specified optimization costs and
first-order stationary diagonalizers, with the latter assuming positive
diagonal matrices. The appendix's distinction between this intermediate
algebra and our response-to-factor chain is consistent with those passages.
Full rank of our diagonal-profile matrix implies the required noncollinearity.
This verifies the cited theorem-level comparison in the author manuscript;
the published typeset version has not been independently compared byte-for-byte.

### Gram and softmax identifiability comparisons

- **Dorrell et al., arXiv v3, Theorem 2 and Appendix D.2, pages 8 and
  33--34:** read the two tight-scattering hypotheses and the orthogonal-to-
  permutation argument. The ellipsoid touches the coordinate hyperplanes
  because its diagonal shape entries equal squared coordinate means.
  A finite strictly positive router cannot contain those touching points in
  its convex hull. Thus router positivity/full rank do not supply their
  scattering hypothesis; the local covariance response is genuinely extra
  information in our proof. This observation is our direct geometric
  comparison, not a claim about every theorem in that paper.
  [Primary text](https://arxiv.org/pdf/2601.10482v3).
- **Tran et al., Theorem 4.1 and Appendix A.1, pages 5 and 19:** checked
  equality of dense MoE functions, distinct experts and distinct within-model
  gate differences. Experts are ReLU networks; the conclusion matches expert
  functions, not their internally non-unique neural parameters. The manuscript
  now states that scope explicitly. The theorem is a prior function-level
  label/translation result, not an observation of our finite factor response.
  [Primary text](https://papers.nips.cc/paper_files/paper/2025/file/fad7c708dda11f3e72cc1629bb130379-Paper-Conference.pdf).
- **Nguyen et al., Proposition 1 and Theorem 1, pages 5--6:** verified
  translation identifiability and the exact-fitted Hellinger-to-Voronoi inverse
  bound; the model setup uses a compact parameter set and bounded covariates.
  The constant depends on the reference mixing measure and parameter domain.
  This supports the manuscript's attribution of an existing softmax inverse
  bound, while the conditional-density observable differs from our response
  operator. Only these comparisons were checked, not the full statistical proof.
  [Primary text](https://proceedings.neurips.cc/paper_files/paper/2023/file/0ef6ffcb85a2d238fc4761860c31ded4-Paper-Conference.pdf).

### Optimizer metric and operator probing: primary-source follow-up

- **van Oostrum et al., Theorems 1--2 and their proofs:** the pushed-forward
  natural parameter gradient is the projection onto the image of the parameter
  differential. Diffeomorphic reparameterization preserves that image. Our
  displayed tangent-space identity supplies the corresponding comparison for
  feasible full-rank router gauges; this application is our deduction. The
  claim concerns instantaneous model-space directions, not invariance of
  finite parameter steps. The publisher correction only adds editorial and
  conflict-of-interest information, not a mathematical correction.
  [Primary article](https://link.springer.com/article/10.1007/s41884-022-00067-9),
  [correction](https://link.springer.com/article/10.1007/s41884-023-00112-1).
- **Chiu and Demanet, Sections 1.1--1.3, Theorem 1.3 and Proposition 1.5:**
  read the known-linear-family coefficient solve, random-probe conditioning
  assumptions, approximation-error statement and stacked multi-probe system.
  This supports attribution of generic probing and conditioning. Their bound
  uses basis Gram conditioning and numerical ranks; it does not directly give
  our orthogonal-support extraction or our factor inverse constant. No claim
  of a better generic query bound is warranted.
  [Author-hosted full text](https://math.mit.edu/icg/papers/matrix-probing-theory.pdf).
- **Halikias and Townsend, Section 2, Definition 2.1, Lemma 2.2 and
  Sections 2.1.1--2.1.3:** verified the linear-family recovery setup, counting
  bound and explicit diagonal/block/tridiagonal designs. Their query model
  permits products with the matrix and its transpose. These results already
  establish that exploiting known structure can reduce queries substantially.
  Our response-specific construction is not a general matrix-recovery novelty.
  Their linear-family counting bound alone is not a lower bound for our
  nonlinear realizable response family with a known product.
  [Author preprint](https://arxiv.org/pdf/2212.09841).
- **Marcotte et al., equations (3)--(4), Theorem 3.9 and Appendix I,
  equations (26)--(27):** verified the lifted Jacobian-Gram flow and the
  two-factor operator `X -> UU^T X + X VV^T`. Their differential-kernel
  calculation also uses scalar-identity cancellation between the two terms;
  this algebraic maneuver is not independently novel. Theorem 3.9 concerns
  relaxed-balanced initialization and intrinsic dynamics along reachable
  trajectories, whereas our claim compares full responses at two fixed
  softmax-router charts. The checked text supports this distinction, not a
  claim that every possible consequence of their framework has been excluded.
  [Official full text](https://proceedings.iclr.cc/paper_files/paper/2026/file/184b2454410a88c280d29aa95fbac6e2-Paper-Conference.pdf).

These checks support the current restricted contribution wording; no theorem
or numerical result was changed. Remaining priority: recent softmax/optimizer
papers and precise claims about query complexity. Metadata inventory alone
still does not certify all cited works.

- **Otto, Proposition 1 (complete one-page technical note):** verified the
  determinant-polynomial proof: existence of one injective fixed-size probe
  design implies almost-everywhere injectivity for a known linear operator
  family. Downloaded the author-deposited PDF after the DOI resolver failed.
  For fixed support S, our enlarged family with arbitrary symmetric Q and
  arbitrary supported symmetric local blocks is linear. Our designed probes
  recover that family when D-K>=L; Otto therefore also implies generic K-probe
  uniqueness in that regime. Our direct rank proof additionally spells out
  the sufficient count max(K,ceil(L/(D-K))) when the complement is smaller.
  Neither argument makes this count necessary or proves good conditioning
  for a particular Gaussian draw. This is an application of existing generic
  recovery theory, consistent with the manuscript's secondary placement.
  [Author-deposited note](https://zenodo.org/records/7916776).

### Recent softmax and optimizer comparisons

Checked the cited versions rather than silently substituting newer versions.

- **Varre et al., v1, Appendix D.1, Lemma D.2 and Property D.3:** their
  single-output value-softmax model already yields the squared-covariance
  response. PDF page 20 was visually inspected: the derivative and intermediate
  product-rule line contain C(s)^2, while the next simplification and named M
  drop the square. This is an apparent algebraic typo, not a different
  optimizer; applying their displayed Jacobians independently gives
  `||s||^2 I + V C(s)^2 V^T`. Our appendix now attributes this local response
  explicitly to the derivation, rather than merely mentioning related flow
  terms. Their scalar-router setup does not supply our multirow full-rank
  common-product identifiability theorem. Section C.1 also distinguishes
  regression from the logistic one-hot limit; no universal polarization claim
  is imported into our manuscript.
  [Primary v1](https://arxiv.org/pdf/2603.06248v1).
- **Singh, v1, Definition 3.1 and Lemma 3.2, page 6:** verified the optimizer
  state-equivariance definition and maximal orthogonal subgroup preserving
  the unconstrained two-factor Euclidean metric for all factor pairs. This
  is a global metric-isometry statement, not the pointwise softmax response
  inverse. The manuscript already attributes this mechanism to prior work.
  This check does not endorse every empirical or optimizer-state claim in
  the preprint.
  [Primary v1](https://arxiv.org/pdf/2608.05136v1).
- **Lau and Su, v4, Section 3.5, Definition 3.4, pages 13--15:** verified
  expert-permutation/shared-logit-shift symmetry and the definition of
  horizontal compatible update maps for input-dependent MoE router weights.
  They design updates respecting those transformations. This does not prove
  necessity of permutation from equality of our effective responses under a
  fixed Euclidean metric. The comparison is with the checked definition and
  constructions, not a full audit of the 84-page manuscript.
  [Primary v4](https://arxiv.org/pdf/2605.18106v4).
- **Goel et al., v1, Sections 3--4, assumptions A1--A3 and Theorem 1:**
  verified the Gaussian attention setup and population objective's weighted
  matrix-factor form `||A Sigma B^T Sigma^(1/2)-M Sigma^(1/2)||_F^2/2`
  plus irreducible noise. This concerns an infinite-sample population model
  and a regularized/preconditioned algorithm, not a fixed learned simplex
  router's response operator. Our distinction is supported by the displayed
  objective; their finite-sample convergence proof was not independently
  audited in this pass.
  [Primary v1](https://arxiv.org/pdf/2603.01514v1).

### Structured approximation query bounds

**Amsel et al., COLT 2026, Theorem 1 (PDF page 4), Definition 7 and
Corollary 3 (page 11):** checked the official proceedings PDF obtained from
its linked PMLR GitHub asset. The finite-family theorem gives a (3+epsilon)
Frobenius approximation using adaptive products with the matrix and its
transpose. The linear-family corollary retains an additive alpha times the
target Frobenius norm; it is not an exact-recovery theorem at a finite query
count by setting alpha=0. Our response operator is self-adjoint, so transpose
access would coincide with forward access: two-sided querying is not itself
an obstruction to applying their result here. Our K-probe result instead
exploits a particular known support and block form for exact algebraic
reconstruction. Neither comparison proves K optimal. The current manuscript
correctly calls the result a structured-probing corollary and the packing
condition sufficient. No numerical query-complexity claim needs correction.
[Official record](https://proceedings.mlr.press/v336/amsel26a.html),
[full text](https://raw.githubusercontent.com/mlresearch/v336/main/assets/amsel26a/amsel26a.pdf).

### Attention symmetry and stochastic quotient bias

- **Wang and Wang, Complete Characterization, Table 1 and Theorem 2:**
  checked the official PMLR version (volume 282, pages 622--663). Its generic
  multi-head group comprises independent Q/K and V/output GL actions and
  head permutations. Conditions include canonical dimensions, projection
  ranks, head-form independence, full-rank output projection and the specified
  handling of biases. These conditions are not assumptions about our shared
  row-simplex router. This is a statement-level comparison and proof-sketch
  review, not an independent certification of the full maximality proof.
  [Official source](https://proceedings.mlr.press/v282/wang26a.html).
- **Wang and Wang, Gauge Fiber Bundle Geometry, Section 3 and Theorem 7:**
  checked the Fisher-null-space assumption and horizontal Riesz formulation
  in the official version (pages 664--684). The quotient metric argument
  assumes the Fisher kernel contains only gauge directions; finite data can
  otherwise add degeneracies. Added this qualification to our appendix. We
  cite the claimed construction as prior work without independently verifying
  all global bundle/connection assertions.
  [Official source](https://proceedings.mlr.press/v282/wang26b.html).

Both bibliography entries now use the 2026 proceedings publication year and
formal pages/volume; internal citation keys retain their old year for stability.
The 2025 workshop year remains explicit in the proceedings title. OpenReview
presented a browser challenge, so public publisher-linked PDFs were used.

- **Aladrah et al., v2, equations (1)--(3), (49)--(51):** checked the
  isotropic Langevin approximation, formal stationary-density qualification,
  and columnwise query/key rescaling example. The attention example studies
  a diagonal rescaling subgroup and balanced norms, not a full GL
  classification. The stationary density is a probability only when its
  partition function is finite. Our broad attribution of stochastic
  geometric corrections is supported; it supplies neither a general theorem
  for practical SGD/Adam noise nor our instantaneous response inverse.
  [Primary v2](https://arxiv.org/pdf/2601.06597v2).

### Empirical sharing and merging comparisons

- **TwistedMerge, v1, Section 3.2, PDF pages 15--16:** checked the trained
  adapter gauge audit, its five groups, rank-four residual layer, gauge
  conditioning limits and comparison against full-update SVD. The reported
  invariance contrast supports our attribution of a preexisting trained-factor
  merging consequence. The source explicitly limits this to a feature-layer
  corpus; it does not establish unrestricted GL robustness or accuracy
  superiority. We did not rerun its experiments or independently audit its
  higher-order certificate theory.
  [Primary v1](https://arxiv.org/pdf/2607.20887v1).
- **Khilar, v2, Tables 1--2 and Section 7:** verified the quoted 37.2%
  reconstruction improvement and WikiText-2 perplexities 159.14 versus 50.17
  for the Pythia 1.4B compression comparison. The reported results are a
  proxy/deployment mismatch, not a same-effective-product gauge experiment.
  We retain the wording that the source reports these measurements rather
  than treating them as our replications. Updated the bibliography note to
  v2, 6 June 2026, which contains the checked values; the old note identified
  the initial submission date. No claimed universal ranking follows.
  [Primary v2](https://arxiv.org/pdf/2605.30836v2).
- **MASA, arXiv v2, equations (4), (18), grouping paragraph and mixing-
  coefficient visualization paragraph:** verified orthonormal matrix PCA
  for the pretrained branch, grouping from adjacent layer vocabulary-space
  distributions, atom cosine correlations and descriptive coefficient plots.
  This supports separating its functional grouping action from coefficient
  interpretation. Our unconstrained-factor change-of-basis comparison is
  an algebraic deduction, not a claim that the pretrained PCA basis admits
  arbitrary GL changes while preserving its orthonormal constraint.
  [Primary v2](https://arxiv.org/pdf/2508.04581v2).

### LoRA factor-processing claims

- **Putterman et al., v2, Appendix C, Definition C.1, Proposition 2 and
  Lemma 3:** checked full-rank domain, product orbit separation and the
  compact-set factorization of continuous invariant functions. This supports
  the manuscript's full-rank qualifier; it is not an identifiability theorem
  for arbitrary rank-deficient factors or a recovery of layer history.
  [Primary v2](https://arxiv.org/pdf/2410.04207v2).
- **Chen et al., v1, Sections 3--4, Proposition 2 and Appendix A.2:**
  checked raw-factor gauge dependence and the projector-based consensus
  construction. The effective aggregate is the projection of the dense update
  average onto the retained consensus subspace. Equality with the unprojected
  average requires sufficient retained rank; the manuscript does not assert
  that stronger equality. Eigenvalue ties at a truncation boundary require
  a consistent subspace selection, so no unique coordinate-basis claim follows
  merely from an invariant projector sum.
  [Primary v1](https://arxiv.org/pdf/2605.06733v1).
- **Han et al., W2T v1, Appendix A.1--A.2 and Section 5.2:** checked the
  product invariance, QR-to-core-SVD construction and reported gauge tests.
  The singular decomposition retains sign and repeated-singular-value
  subspace ambiguities; the source invokes deterministic postprocessing.
  This supports product-based processing as prior art, but is not by itself
  a proof of unique raw singular-vector coordinates for every degenerate
  input. We do not claim to have replicated its retrieval experiments.
  [Primary v1](https://arxiv.org/pdf/2603.15990v1).

### Tangent responses and matrix-factor geometry

- **Jacot et al., Section 4, p. 5:** checked the finite-width definition as
  a sum of parameter-derivative outer products and the corresponding kernel
  gradient flow. The kernel can vary during training; constancy is a separate
  infinite-width result. Our packed factor-map Jacobian Gram, with fixed
  block learning rates, uses this established response construction rather
  than introducing a new general kernel principle. No infinite-width or
  constant-kernel assumption is needed for our fixed-chart theorem.
  [Primary proceedings PDF](https://papers.nips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf).
- **Novak et al., Section 3.3, equations (7)--(9):** checked NTK-vector
  products through a vector-Jacobian product followed by a Jacobian-vector
  product, and reconstruction using output-space identity columns. This
  directly supports the cited computation precedent. Our structured query
  count depends on the particular factor-response family; the cited general
  computation is not an optimal-query theorem for that family.
  [Primary proceedings PDF](https://proceedings.mlr.press/v162/novak22a/novak22a.pdf).
- **Tu et al., Lemma 5.4, p. 6:** checked the squared Procrustes-distance
  bound by squared Gram error divided by
  `2(sqrt(2)-1) * sigma_r(X)^2`. A positive smallest relevant singular value
  is needed for a finite bound. This is the Gram-to-orthogonal-factor
  perturbation antecedent cited in the manuscript; permutation identification
  using the softmax covariance blocks is a further model-specific step.
  This check covers the lemma statement, not an independent audit of its proof.
  [Primary proceedings PDF](https://proceedings.mlr.press/v48/tu16.pdf).
- **Mishra et al., Section 3.1, equations (5)--(6), Section 4.2 and
  Tables 2--3:** checked the full-rank factor equivalence
  `(G,H) -> (G M^{-1}, H M^T)` and quotient-compatible metric/horizontal
  constructions. This supports the boundary that suitable quotient geometry
  can remove coordinate dependence of optimization. It does not imply that
  unmodified Euclidean updates in our simplex chart are invariant under
  all product-preserving changes of basis.
  [Primary manuscript](https://arxiv.org/pdf/1209.0430).

### Static factorization recovery boundaries

- **Donoho and Stodden, Section 5, rules R1--R3, Theorem 1 and its
  corollary:** checked recovery up to scaling and permutation for their
  nonnegative generative model with separated support and complete factorial
  sampling. Their separability rule uses a distinguishing pixel for each
  part/articulation pair; it should not be silently replaced by our anchor-layer
  assumption. The manuscript cites this as condition-dependent NMF recovery,
  not a theorem for an unrestricted signed basis with positive soft routing.
  [Primary proceedings PDF](https://papers.nips.cc/paper_files/paper/2003/file/1843e35d41ccf6e63273495ba42df3c1-Paper.pdf).
- **Fu et al., Section III, Theorem 1, equation (3), and Table I:**
  checked full-rank and sufficiently-scattered assumptions and recovery by
  a determinant-minimizing identification criterion. Equation (3) normalizes
  columns of H, unlike the row normalization of equation (2); its objective
  is not simply the same enclosing-simplex volume interpretation. The result
  concerns an optimal solution of that criterion, not uniqueness of every
  unconstrained feasible factorization. Our collective citation to recovery
  under additional assumptions remains supported with this qualification.
  [Author-hosted published PDF](https://web.engr.oregonstate.edu/~fuxia/fu2018letter.pdf).
- **Abdolali and Gillis, Section 2 separability paragraph, Section 3,
  Assumption 1, Remark 1 and Theorem 1:** checked anchor columns and
  uniqueness among factorizations satisfying the facet-based conditions.
  These conditions include facet coverage and exclusion of equally populated
  spurious facets; they are not merely positivity or a universal consequence
  of separability. This supports both the static-boundary citation and the
  elementary convex-hull argument accompanying our separable recovery
  proposition. The latter explicitly restricts both competing factorizations
  to the separable model class and adds affine independence for coordinates.
  [Primary manuscript](https://arxiv.org/pdf/2007.11446).
- **Gillis and Rajko, Section 4.1, Theorem 6 and proof:** checked partial
  column recovery for rank-r exact NMF under a rank-(r-1) zero-region
  submatrix condition and a selective row in the other factor. Nonnegativity
  and rank force one corresponding column in every competing exact NMF,
  up to positive scaling. This is partial identifiability with support
  information, not recovery of every factor under simplex constraints alone.
  We do not use this passage as certification of the source's other theorems.
  [Primary manuscript](https://arxiv.org/pdf/2206.08022).

### Learned grouping, folding and explicit recovery targets

- **Savarese and Maire, Sections 3.2 and 4.3, Figure 4 and Appendix A:**
  checked absolute coefficient cosine similarity as the layer similarity
  matrix, extraction of recurrent patterns, and downstream evaluation after
  tying/folding. The cited description of a coordinate-to-structure action
  is supported. Their architectural-discovery language is accompanied by
  intervention measurements; it is not a claim of identifying a planted
  historical sharing graph. OpenReview required browser verification and the
  author PDF timed out, so this check used the arXiv manuscript carrying
  the ICLR 2019 publication header.
  [Primary manuscript](https://arxiv.org/pdf/1902.09701).
- **Plummer et al., Sections 2 and 3.2:** checked the fixed-parameter-budget
  objective and the short preliminary training with a single parameter group.
  The learned coefficients or embeddings are then clustered by k-means to
  form parameter groups for the full Shapeshifter Network. This supports
  our grouping-rule description; it does not establish that its clustering
  recovers a historical generator or that our gauge intervention reproduces
  their entire training procedure. OpenReview access required verification;
  the checked copy is the author-hosted ICLR 2022 paper.
  [Author-hosted proceedings PDF](https://htor.inf.ethz.ch/publications/img/plummer-npas.pdf).
- **David-Hay and Wolf, introduction and Algorithm 1:** checked the explicit
  layer-copy/independent-update actions and the reinforcement-learning policy
  with loss/perplexity feedback. This is a search over hard tying choices,
  supporting its treatment as a boundary case rather than an interpretation
  of jointly learned soft-factor coordinates. We have not replicated its
  reported parameter savings or training performance.
  [Primary proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/fce2d8a485746f76aac7b5650db2679d-Paper-Conference.pdf).
- **Yeh et al., Sections 4.3 and 5.1--5.3:** checked partition distance
  against an explicit target sharing scheme and the Gaussian shared-mean,
  permutation and shift experiments. Small cases have exact empirical
  recovery, while larger cases can have only partial recovery. Its Gaussian
  MSE analysis must not be described as a universal exact-partition theorem.
  Our current brief positive-boundary citation is supported: this work
  defines and evaluates the recovery target instead of inferring historical
  truth solely from a coefficient visualization. This passage check does not
  independently certify the source's theoretical bounds.
  [Primary proceedings PDF](https://proceedings.mlr.press/v151/yeh22b/yeh22b.pdf).

### Gradient allocation and attention-reuse controls

- **Li et al., E2LoRA, Figure 2, Sections 3.1--3.2, Algorithm 1 and
  Appendix I/Table 12:** checked task-subset gradients collected before
  fine-tuning, proxy entropy/RMI, consecutive sharing intervals and allocated
  ranks. Table 12 reports identical interval structures and ranks across
  seeds 0, 18 and 42 for CLIP-ViT-B/16 k-projection on EuroSAT. This supports
  the manuscript's source-attributed stability description, not universal
  subset invariance. The entropy and concatenated-gradient RMI are the
  source's proxies; this check does not certify them as true mutual information
  estimators. The decision is made from pretrained-model gradients, not a
  learned adapter's arbitrary factor chart. OpenReview required verification;
  the full official proceedings PDF was accessible.
  [Primary proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/7b434657e8b8c3bfbf1d61d6e0026f90-Paper-Conference.pdf).
- **He et al., RaSA, introduction, Section 3 and Theorem 3.1:** checked
  the global pool with k ranks contributed per layer, layer-specific weighting,
  and reconstruction-error comparison. The theorem constructs a feasible
  RaSA representation achieving LoRA's optimal reconstruction error; it is
  not a proof of a better trained task model or historical partition recovery.
  Pool connectivity is prescribed while its parameters are learned, which
  supports our fixed-sharing-topology boundary. The arXiv copy carries the
  ICLR 2025 publication header; OpenReview required browser verification.
  [Primary manuscript](https://arxiv.org/pdf/2503.12576).
- **Mu et al., LiSA, Sections 3--4, Section 5.1 and Figure 7:** checked
  attention similarity/sensitivity analyses, reuse of the preceding layer's
  pre-softmax score matrix, learned head alignment, and input-dependent
  low-rank difference compensation. These are attention computations followed
  by model evaluation, not a common linear parameter-factor chart or an
  assertion of historical sharing-graph recovery. This supports its use as
  an intervention-validation control. We have not reproduced its quality or
  throughput measurements. The official PDF also confirms the bibliography's
  TACL volume 14, pages 656--688 and May 2026 publication.
  [Primary journal PDF](https://aclanthology.org/2026.tacl-1.30.pdf).

### Compression-oriented sharing methods

- **Wang et al., Basis Sharing, Sections 3.1--3.3 and 4.1:** checked
  concatenated weight decomposition, scaled reconstruction-error comparisons
  for matrix types, adjacent-layer grouping and downstream evaluation. The
  method selects sharing to limit compression error; its coefficients are
  not presented as a recovered historical generator. This supports the
  manuscript's compression/reconstruction boundary, without independently
  reproducing the reported perplexity or reasoning results.
  [Primary proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/238c98450b1d9e8055f94d22f303bb57-Paper-Conference.pdf).
- **Uyuk et al., FiPS, Sections 2.2--2.3 and 3:** checked shared SVD
  initialization, sparse layer-specific projection factors, calibration-based
  activation reconstruction and optional global fine-tuning. Consecutive
  grouping is motivated by reconstruction and evaluated task quality; the
  text explicitly distinguishes increased rank from improved performance.
  This supports treating FiPS as compression evidence, not historical
  partition recovery. The arXiv full text carries the June 2026 TMLR header
  and the same OpenReview identifier as the bibliography. OpenReview itself
  required browser verification. No external benchmark was reproduced here.
  [Primary manuscript](https://arxiv.org/pdf/2411.09816).
- **Zhang et al., Geo-Sharing v1, Section 3.2, equations (9), (15)--(17)
  and Algorithm 1:** checked the local quadratic loss approximation and
  candidate-basis selection by high-curvature perturbation energy, followed
  by low-curvature projection and clipping. This supports the brief claim
  that the method uses curvature to select compression assignments. It is
  not a same-product softmax-factor stabilizer or historical-recovery result.
  The check does not certify a global optimum, actual loss preservation, or
  preservation of shared-factor representability after projection/clipping.
  PDF retrieval was intermittent; the primary arXiv HTML was accessible.
  [Primary v1 HTML](https://arxiv.org/html/2511.06786v1).

### Optimizer comparisons and horizontal lifting

- **Zheng and Wu, LoRA-S, Section 3.2, equations (7)--(9), Algorithm 1,
  Theorem 1 and Appendix A.5:** checked the full-rank quotient metric and
  Sylvester-based horizontal lift. The invariance proof compares factorizations
  using the same product-space descent vector. Thus its result must not be
  read as invariance of arbitrary independently preconditioned factor updates.
  Clarified the manuscript's former phrase “arbitrary preconditioned optimizers”
  to identify product-space directions and this consistency condition. This
  is an established invariant-optimizer counter-boundary to our fixed
  Euclidean chart result, not a proof of our response stabilizer. The check
  does not independently certify all efficient-feature-learning claims.
  [Primary proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/a6a90bcc2aa470c3871b2d39a67d26e8-Paper-Conference.pdf).
- **Stupariu and Manolache, v1, Section 2, Tables 1--3 and Section 3:**
  checked Adam/Muon comparisons, local loss curvature, and weight/representation
  ranks. The conclusions explicitly frame the mechanisms as unresolved and
  training-dynamics analysis as further work. These passages support our
  empirical-optimizer distinction, rather than a response-identification or
  tomography attribution. Mixed graph-task outcomes and the source's warning
  against equating smoother plots with lower curvature remain relevant scope
  limits. This follow-up checks those cited comparisons, not independent
  replication or a certificate covering every external claim.
  [Primary v1](https://arxiv.org/pdf/2605.27662v1).

### ASLoRA versions and inaccessible workshop full text

- **Hu et al., ASLoRA arXiv v2, Sections 3.2--3.5, equations (2)--(4)
  and Algorithm 1:** checked shared A, cumulative historical averages of B,
  raw Euclidean pair distance, and replacement of the lower layer's B by
  the upper layer's factor. The algorithm describes all pair distances while
  its following prose specifies adjacent layers. This supports preserving
  both candidate-set audits rather than claiming an unambiguous official
  implementation reproduction. Section 3.5 gives the shared-initialization
  rationale discussed in our manuscript. Author differences between v2 and
  the journal record are preserved in separate bibliography entries.
  [Primary v2](https://arxiv.org/pdf/2412.10135v2).
- **Hu et al., journal record, metadata/highlights only:** the indexed
  publisher record confirms Pattern Recognition 180, article 114072, the
  eight journal authors and global-A/adaptive-B description. Direct page
  access returned HTTP 403; full journal text was not available in this
  follow-up. Its December 2026 issue label does not establish when the paper
  first became available online. No detailed algorithm or theorem is
  attributed to this version on the basis of the abstract alone.
  [Publisher record](https://www.sciencedirect.com/science/article/abs/pii/S003132032601037X).
- **Karuturi et al., workshop item, unresolved full-text coverage:**
  rechecked the existing record in PRIOR_ART_REGISTRY.md, which identifies
  the official title/authors/poster entry. Current OpenReview forum/PDF
  requests hit browser verification, and the public API returned HTTP 403.
  The workshop website confirms its non-archival poster format but does not
  independently expose this item's technical content. This follow-up therefore
  retains metadata-only status from the historical registry and makes no new
  full-text or priority finding. The manuscript already states the limitation.
  [Item record](https://openreview.net/forum?id=YMsbYXDtqw);
  [workshop format](https://www.weightsymmetry.com/cfp).
