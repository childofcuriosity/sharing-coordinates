# Finite structured tomography of the Euclidean response

Audited: 2026-08-23.  This note turns the all-effective-gradient quantifier in
the response-stabilizer theorem into a finite, executable observation.  It is a
structured matrix-probing corollary, not a second central novelty claim.  The
observable remains the instantaneous response of a named Euclidean training
law; it is not an AdamW trajectory, a task-function oracle, or a historical
sharing graph.

## Setup

Let

\[
 A=\operatorname{softmax}_{\rm row}(Z/\tau)\in\mathbb R^{L\times K},
 \qquad B\in\mathbb R^{K\times D},\qquad \Theta=AB,
\]

where `tau`, `eta_Z`, and `eta_B` are known and strictly positive.  Assume

\[
 \operatorname{rank}(A)=\operatorname{rank}(B)
 =\operatorname{rank}(\Theta)=K. \tag{T1}
\]

For a column-form effective-gradient row `g_i in R^D`, the positive
instantaneous Euclidean response is

\[
 y_i=[\mathcal R_{A,B}(G)]_i
 =\eta_B\sum_{j=1}^L Q_{ij}g_j+H_i g_i, \tag{T2}
\]

where

\[
 Q=AA^\top,\qquad
 H_i={\eta_Z\over\tau^2}B^\top J_i^2B,
 \qquad J_i=\operatorname{diag}(a_i)-a_i a_i^\top. \tag{T3}
\]

Let `S=row(Theta)`.  Full rank in (T1) implies

\[
 S=\operatorname{row}(B),\qquad \dim S=K,
 \qquad \operatorname{range}(H_i)\subseteq S,
 \qquad H_i|_{S^\perp}=0. \tag{T4}
\]

The last two identities are the structure exploited below.

## Theorem T1: `K` deterministic probes determine the full response

Assume (T1) and the sufficient packing condition

\[
 D-K\ge L. \tag{T5}
\]

Choose an orthonormal basis `u_1,...,u_K` of `S` and orthonormal vectors
`v_1,...,v_L` in `S^perp`.  These choices use only the effective matrix
`Theta`, not a particular factor chart.  Define `K` probe matrices by their
rows:

\[
 g_j^{(1)}=v_j+u_1\quad(1\le j\le L), \tag{T6}
\]

and, for `2<=r<=K`,

\[
 g_j^{(r)}=u_r\quad(1\le j\le L). \tag{T7}
\]

Then the `K` responses `Y^(r)=R_(A,B)(G^(r))` determine `Q`, every `H_i`,
and hence the complete linear operator `R_(A,B)`.  Explicitly,

\[
 Q_{ij}={1\over\eta_B}\langle y_i^{(1)},v_j\rangle, \tag{T8}
\]

\[
 H_i u_1=y_i^{(1)}-
 \eta_B\sum_{j=1}^LQ_{ij}(v_j+u_1), \tag{T9}
\]

and, for `2<=r<=K`,

\[
 H_i u_r=y_i^{(r)}-
 \eta_B\left(\sum_{j=1}^LQ_{ij}\right)u_r. \tag{T10}
\]

Consequently, for two full-rank feasible charts `(A,B)` and `(A',B')` of
the same `Theta`, equality on the fixed probes (T6)--(T7) is equivalent to
equality of their complete response operators.  Combining this equivalence
with the separately proved exact response-stabilizer theorem gives

\[
 \mathcal R_{A,B}(G^{(r)})=\mathcal R_{A',B'}(G^{(r)})
 \quad\hbox{for every }r=1,\ldots,K
\]

if and only if there is a permutation matrix `P` with

\[
 A'=AP,\qquad B'=P^\top B. \tag{T11}
\]

The corresponding logits are still free under independent row shifts.

### Proof

Project (T2) for the first probe onto `S^perp`.  Equation (T4), the
orthogonality of `u_1`, and the orthonormality of the `v_j` give

\[
 \Pi_{S^\perp}y_i^{(1)}
 =\eta_B\sum_jQ_{ij}v_j.
\]

Taking inner products with `v_j` proves (T8).  Subtracting this now-known
shared-basis response from the unprojected first response leaves
`H_i(v_i+u_1)=H_i u_1`, proving (T9).  For probe `r>=2`, subtracting the
known shared response leaves `H_i u_r`, proving (T10).

For any `x in R^D`, (T4) yields

\[
 H_i x=\sum_{r=1}^K\langle u_r,x\rangle H_i u_r.
\]

Thus (T9)--(T10) determine every local block.  Equations (T2), (T8), and
the recovered local blocks determine the entire response.  The same probes
are used for both charts because their common rank-`K` product has the same
row space.  Equality on the probes is therefore equivalent to equality of
`Q` and all `H_i`, hence to equality of the whole response.  Applying the
exact stabilizer theorem proves (T11), and a simultaneous permutation
plainly gives the converse.  QED.

### Conditioning of the displayed reconstruction

The complement code matrix in (T8) has orthonormal rows, and the `S`
coordinates of probes (T6)--(T7) form exactly `I_K` for every layer.  Thus
the two coefficient solves in exact arithmetic have condition number one.
This does not make the recovered chart well-conditioned near a rank or
simplex-boundary degeneration; that separate inverse problem is controlled
by the margins in the quantitative stabilizer theorem.

## Proposition T2: deterministic observation-noise propagation

Suppose the observed response to probe `r` is `Y^(r)+E^(r)` with
`||E^(r)||_F<=epsilon_r`, and use (T8)--(T10) as the reconstruction.  Then

\[
 \|\widehat Q-Q\|_F\le {\epsilon_1\over\eta_B}. \tag{T12}
\]

Writing `H^(S)` for the tensor of all `K x K` local blocks in the `u` basis,

\[
 \|\widehat H^{(S)}-H^{(S)}\|_F
 \le\left[
 (1+\sqrt{L+1})^2\epsilon_1^2+
 \sum_{r=2}^K(\epsilon_r+\sqrt L\epsilon_1)^2
 \right]^{1/2}. \tag{T13}
\]

### Proof

Right multiplication by the orthonormal-row complement code is
nonexpansive, which proves (T12).  The first probe matrix has
`||G^(1)||_2=sqrt(L+1)` because its row Gram is `I_L+11^T`.  Hence the error
in the reconstructed first local column is at most

\[
 \epsilon_1+\eta_B\|\widehat Q-Q\|_F\|G^{(1)}\|_2
 \le(1+\sqrt{L+1})\epsilon_1.
\]

For each later probe, the erroneous shared subtraction contributes at most

\[
 \eta_B\|(\widehat Q-Q)\mathbf1\|_2
 \le\sqrt L\epsilon_1,
\]

in addition to `epsilon_r`.  Orthogonal projection into the `u` basis is
nonexpansive.  Squaring and summing the `K` column bounds proves (T13).
QED.

Equations (T12)--(T13) provide an error bar, not an exact noisy `iff`.
Two noisy charts are distinguished only when their reconstructed component
gap exceeds the sum of their applicable error radii.  No claim is made for
unknown, biased, or optimizer-state-dependent observation error.

## Corollary T3: generic Gaussian probes also determine the operator

Let `p=D-K>=1` and draw `q` effective-gradient matrices independently with
iid standard Gaussian entries.  If

\[
 q\ge\max\left\{K,\left\lceil {L\over p}\right\rceil\right\}, \tag{T14}
\]

then the full structured response is determined with probability one from
the complete response vectors.

To see this, concatenate all projected input rows in `S^perp` into an
`L x qp` Gaussian matrix `X_perp`.  Projected responses obey

\[
 Z_\perp=\eta_BQX_\perp.
\]

Condition `qp>=L` makes `X_perp` full row rank almost surely, recovering
`Q`.  After subtracting the shared term, the `q x K` matrix of `S`
coordinates for each layer is Gaussian and has full column rank almost
surely when `q>=K`; it recovers that layer's local block.  A finite union of
probability-zero rank failures still has probability zero.

For the released language checkpoints, `(L,K,D)=(12,4,787712)`, so four
Gaussian probes would be operator determining almost surely if their full
input and response vectors were retained and the rank solves were checked.
The older eight-probe artifacts report response norms rather than a complete
tomographic reconstruction, so this corollary must not be used retroactively
to relabel those summaries as rank-certified operator recovery.

The additive four-probe checkpoint audit computes each full-dimensional
response with the analytic block formula (T2).  It does **not** perform four
full-model autograd/JVP calls.  That formula implementation was independently
checked against autograd and native finite differences on the predeclared
`4 x 32` extraction in the older response audit.  The new artifact therefore
validates the structured reconstruction conditional on that separately tested
formula, rather than claiming four independent black-box response
interventions.

## Corollary T4: pointwise Gaussian detection from the inverse bound

This is a different, weaker use of Gaussian probes.  Let two fixed exact
charts belong to the same nondegenerate set on which the quantitative inverse
has one uniform constant `C`, and write

\[
 d_{\rm orbit}((A,B),(A',B'))\le
 C\|\mathcal D\|_{F\to F},\qquad
 \mathcal D=\mathcal R_{A,B}-\mathcal R_{A',B'}.
\]

If the pair is fixed before the probes and `d_orbit>=Delta`, then for iid
`x_l~N(0,I_(LD))`, every `t>0` satisfies

\[
 \Pr\left[
 \max_{1\le\ell\le q}\|\mathcal D x_\ell\|_2
 <{t\Delta\over C}
 \right]
 \le [2\Phi(t)-1]^q. \tag{T15}
\]

Indeed, `||D||_(F->F)>=Delta/C`.  If `v` is a top right singular vector,
then `||D x||>=||D|| |<v,x>|`, and the inner product is standard normal.
Independence proves (T15).  At `t=1/2,q=8`, the displayed miss bound is
approximately `4.62e-4`.

The quantifiers and scale in (T15) are essential:

- it is pointwise for one alternative fixed independently of the draws, not
  simultaneous over a continuum or valid after probe-based model selection;
- the threshold `t Delta/C` is absolute; it does not calibrate the released
  relative-response ratio without additional response-norm bounds;
- `C` is meaningful only after fixing the shared compact/nondegenerate set,
  learning rates, and temperature;
- with deterministic response error at most `epsilon` per observed
  difference, the observable lower threshold degrades to
  `t Delta/C-epsilon` when positive;
- approximate equality of `Theta` is outside the theorem unless a separate
  subspace/product perturbation result is proved.

Thus (T15) can explain why a handful of probes detects one predeclared,
frozen, well-separated chart, but it is not a confidence certificate for
equality.

## Boundary and counterexample ledger

1. **The packing condition is sufficient, not necessary.**  If `D-K<L`,
   the `L` orthonormal complement codes in (T6) do not exist, so this specific
   construction is undefined.  This is not a finite-identifiability lower
   bound.  For example, when `L=K=D=2`, `S^perp={0}`, yet the four canonical
   probes recover any `4 x 4` response operator column by column.  More
   efficient designs may exist between these extremes.

2. **`K=1`.**  The construction uses its single first probe when
   `D-1>=L`.  Row softmax is identically one and `J_i=H_i=0`; (T8) recovers
   the shared response.  There is no nontrivial basis permutation or router
   chart to identify.

3. **Rank.**  If `rank(Theta)<K`, its row space no longer supplies the
   `K`-dimensional common support in (T4).  Rank-deficient routers or bases
   also admit non-permutation response stabilizers documented in the main
   response proof ledger.  Neither the tomography theorem nor the exact
   chart conclusion survives unchanged.

4. **Known positive rates.**  Equation (T8) divides by `eta_B`; setting it
   to zero removes the complement-coded shared signal.  Setting `eta_Z=0`
   leaves non-permutation orthogonal gauges fixing `1` invisible.  Unknown
   relative block rates and temperature are not identified here.

5. **Exact common product.**  The common design uses
   `S=row(Theta)=row(B)=row(B')`.  A small product residual does not alone
   ensure a small row-space error without a singular-value margin, and the
   current proof supplies no approximate-product result.  Common-product
   information is also logically essential to the stabilizer theorem:
   `(A,B)` and `(A,-B)` have the same response but opposite products.
   Version 2 of the audit separates
   `reference_design_rank_dimension_pass`,
   `common_product_numerical_tolerance_pass`, and
   `exact_common_product_holds_by_symbolic_gauge_construction`.  It marks
   `common_design_operator_determining_theorem_applies` only when the first
   check and the symbolic exact-gauge attestation hold.  Falling below a
   floating-point product tolerance is never treated as a theorem premise.

6. **One passive task gradient can be blind.**  Suppose `D>K` and `L>2K`
   and take any two full-rank charts of the same product.  Choose nonzero
   `v in S^perp`.  Since
   `rank(AA^T-A'A'^T)<=2K`, choose nonzero
   `w in ker(AA^T-A'A'^T)`.  The nonzero rank-one gradient whose rows are
   `g_j=w_jv` obeys
   `(R_(A,B)-R_(A',B'))G=0`: the shared difference kills `w`, and every
   local block kills `v`.  This directly refutes any inference from one
   ordinary training direction to operator equality.

7. **Interventional meaning.**  An arbitrary probe `G` can be realized at a
   checkpoint as the gradient of the artificial linear objective
   `L_G(Theta)=<G,Theta>`, or applied by a Jacobian-vector implementation.
   The theorem is therefore executable when checkpoint parameters and such
   interventions are available.  It does not say that natural minibatch
   losses produce the designed directions, and it is not a passive
   behavioral probe of input/output function values.

8. **Optimizer and truth target.**  The reconstructed object is the
   instantaneous Euclidean `Z/B` response with known rates.  It does not
   reconstruct momentum, AdamW preconditioners, a finite-step trajectory,
   functional equivalence outside the linear weight-mixing model, or the
   historical graph that generated the checkpoint.

## Released version-2 artifact contract

The three raw files were regenerated from their checkpoints after the
exact-versus-numerical product distinction above was introduced; they were not
edited or rehashed in place.  They were regenerated once more after the shared
response helper corrected the even-sample median from an upper order statistic
to the conventional average of the two middle values.  Tomography itself does
not use that median, and all tomography measurements are unchanged; the rerun
updates the attested helper hash rather than silently rehashing old records.
The released hashes are:

- `src/response_tomography.py`:
  `88a4dd2c335e8d6e1cea0636705abc31331a60201f0a03ce7cba3548b066a6d0`;
- `experiments/run_response_tomography.py`:
  `aad2155138ab00a7b640455438f0d212de8835ce7d3cd93d5f0cd7fc4975f60c`;
- `experiments/summarize_response_tomography.py`:
  `4d44abc032bced9ed84645b39783ee62c844361d08847dbf525f7a52bb2cb9d9`;
- `src/response_identifiability.py`:
  `77e2a69c05e8690d735578f4e24389fd697fb6163bed2b9c36800e503ff995e6`;
- raw seed 0/1/2:
  `8fe17d12153cc5c4a251ca654fe23448fb8ed28dc8a1f283c9e9a521920752c9`,
  `03cecd7450149b1e49cc33f9769c93d1aebfdb4f36e999b60824a0384fe939cc`,
  and `59dda786953d93bd64f656300388ac9c61fc9f3a955748292d5c0ad75b21170e`;
- `results/response_tomography_summary.json`:
  `6578f06a467cacc2b3129173eb4925612e3c2e551afd2a6cdd1d4a4994dee7e5`.

The summarizer and summary hashes above include the cross-platform LF-output
fix.  The raw writer was not rewritten merely to normalize newlines; the three
raw records were recomputed from the checkpoints because a directly attested
helper changed.

All three designs have ranks `rank(A)=rank(B)=rank(Theta)=4`.  The worst
component reconstruction relative error is `6.853653630597927e-11`; the
largest permutation-control probe difference is
`9.43560526260365e-17`; and the smallest fixed-nonpermutation maximum
designed-probe difference is `0.9730757594220084`.  These values feed the
paper only through the separately validated summary and generated LaTeX
macros.

## Prior-art audit and exact classification

The finite probing theorem should be presented as a transparent structured
operator-recovery corollary.  The following primary sources already contain
the general mechanisms.

### General matrix probing

- Jiawei Chiu and Laurent Demanet, *Matrix Probing and its Conditioning*,
  SIAM J. Numer. Anal. 50(1), 2012,
  [DOI](https://doi.org/10.1137/110825972),
  [author full text](https://math.mit.edu/icg/papers/matrix-probing-theory.pdf).
  Their setup writes an unknown operator as
  `A=sum_(j=1)^p c_j B_j`, applies it to a Gaussian/Rademacher vector, and
  solves the resulting coefficient system.  Theorem 1.3 gives a
  high-probability conditioning result under a conditioned basis Gram matrix,
  high numerical rank of the basis matrices, and
  `n >= p(C kappa lambda log n)^2`; Proposition 1.5 propagates approximation
  error.  Map their unknown operator `A` to vectorized `R`, their basis
  matrices to the known block operators spanning `(Q,H_1,...,H_L)`, and
  their random vector to `vec(G)`.  This supplies the general probing
  framework, but not the response-specific orthogonal-complement separation,
  the explicit `K` probes, or the permutation-stabilizer conclusion.

- Diana Halikias and Alex Townsend, *Structured Matrix Recovery from
  Matrix-Vector Products*, Numerical Linear Algebra with Applications 31(1),
  2024, [journal](https://doi.org/10.1002/nla.2531),
  [arXiv full text](https://arxiv.org/pdf/2212.09841).  Definition 1.1 defines
  exact matvec query complexity.  Definition 2.1 writes a linearly
  parametrized family as `sum_i theta_i A_i`; Lemma 2.2 gives the parameter
  count lower bound `ceil(p/N)`.  The discussion immediately after Lemma 2.2
  invokes Otto's result: existence of one identifying set of `s` probes makes
  almost every `s`-probe set identifying.  Their Table 1 and Sections 2--4
  give explicit algorithms for diagonal, block-diagonal, tridiagonal,
  Toeplitz, low-rank, HSS, and HODLR families.  After the product row space
  `S` is known, (T2)--(T4) place the response in a linear operator family.
  Their results validate the general exact-recovery viewpoint, but do not
  list this block family or derive (T6)--(T10).

- Samuel E. Otto, *A Note on Recovering Matrices in Linear Families from
  Generic Matrix-Vector Products*, 2023,
  [primary record and full text](https://zenodo.org/records/7916776).
  The main result says that, for an unknown matrix in a known linear
  subspace, if any fixed numbers of right and left probes uniquely recover
  it, then failing probe pairs of those dimensions have Lebesgue measure
  zero.  Mapping the known subspace to the linear hull of (T2)--(T4) explains
  the probability-one character of generic probes once a deterministic
  identifying design exists.  It does not provide the response design,
  factor-chart stabilizer, or inverse-to-orbit bound.

- Noah Amsel et al., *Fixed-sparsity matrix approximation from
  matrix-vector products*, arXiv:2402.09379v3 (2024),
  [full text](https://arxiv.org/pdf/2402.09379).  Algorithm 1 performs
  row-wise least squares from iid Gaussian matvecs.  Proposition 1 states
  that a matrix with known row sparsity at most `s` is recovered exactly
  with probability one from `m>=s` probes; Theorem 1 gives the approximate
  Frobenius error for a non-sparse target.  In an `S plus S^perp` coordinate
  basis, our proof likewise performs small row/block solves, although the
  response is not simply a fixed-sparsity matrix in the ambient standard
  basis.  This is a close mechanism-level precedent for Corollary T3.

- Noah Amsel et al., *Query Efficient Structured Matrix Learning*, COLT
  2026, PMLR 336:158--194,
  [official publication](https://proceedings.mlr.press/v336/amsel26a.html),
  [full-text preprint](https://arxiv.org/html/2507.19290).  Theorem 1 treats
  approximation from finite matrix families; Theorem 4 extends it through
  covering numbers; Definition 2 and Corollary 1 give a
  `tilde O(sqrt(q))` approximation result for a `q`-dimensional linear matrix
  family, using access to both the operator and its transpose and allowing
  additive error.  Because the present response operator is self-adjoint,
  transpose access is not the substantive boundary.  It is broader
  approximate-query theory, not the same exact model-specific reconstruction
  or a chart-identification theorem.

### Optimizer/symmetry neighbors, not tomography precedents

- Roman Novak, Jascha Sohl-Dickstein, and Samuel S. Schoenholz, *Fast
  Finite Width Neural Tangent Kernel*, ICML 2022,
  [official paper](https://proceedings.mlr.press/v162/novak22a.html).
  Section 3.3, Eqs. (7)--(9), writes an NTK-vector product as a JVP composed
  with a VJP and states that applying it to the `O` columns of `I_O`
  reconstructs the full `O x O` NTK.  Map their one-example output dimension
  `O` to `LD`, their weighted empirical NTK to the Euclidean response
  `R=J diag(eta_Z I,eta_B I)J^T`, and an identity column to a vectorized
  effective-gradient probe.  This is the direct general precedent for
  reconstructing the complete response from response-vector products.  The
  only model-specific compression supplied here is that block structure
  reduces the generic `LD` identity probes to (T14) generic probes or, under
  (T5), the `K` explicit probes (T6)--(T7).
- Jacot, Gabriel, and Hongler (NeurIPS 2018) supply the finite Jacobian-Gram
  response identity; this makes (T2) NTK background.
- Singh (2026), Lemma 3.2, supplies the general matrix-factorization
  `GL(K)->O(K)` Euclidean-isometry step.  It has no shared-softmax local
  rigidity or finite response probes.
- Lau and Su (2026), Definition 3.4 and Proposition 3.5, formalize expert
  permutation and shared-logit-shift-compatible updates for MoE routers.
  They do not prove maximality for jointly learned shared bases or an inverse
  response theorem.
- Stupariu and Manolache, *How the Optimizer Shapes Learned Solutions in
  Equivariant Neural Networks*, ICML 2026 Workshop on Weight-Space
  Symmetries, [primary full text](https://arxiv.org/html/2605.27662), is an
  empirical Adam-versus-Muon study.  Sections 2.1--2.3 report performance,
  Hessian estimates/loss slices, and weight/representation ranks.  The paper
  contains no response-isometry, stabilizer, tomography, or inverse theorem.
  Poster/workshop metadata alone must not be treated as evidence of one.

### Novelty verdict

The deterministic probes (T6)--(T10), the Gaussian rank count (T14), and the
small-ball combination (T15) are useful for executability and interpretation,
but they are short consequences of the response block form plus standard
matrix/NTK probing.  They must not be promoted as the paper's central
theoretical innovation.  Relative to Novak et al.'s generic `LD` identity
NTK-vector products, the permissible claim is only the model-specific query
compression
`q=max{K,ceil(L/(D-K))}` for generic probes and especially `K` explicit,
condition-one probes under (T5).  The only narrowly supportable candidate for
central novelty remains the combined fixed-chart statement that the complete
Euclidean response stabilizer is exactly basis permutation, together with its
linear inverse on a named nondegenerate set.  The finite tomography result
removes the misleading impression that this statement intrinsically requires
`LD` or infinitely many observations; it does not broaden the optimizer or
truth target.
