# Optimizer-response identifiability: proof ledger

This note records a candidate positive result that is not implied by static
simplex-structured matrix factorization.  The observable is deliberately
stronger than the effective matrix alone: it includes the complete
instantaneous response of that matrix under a specified Euclidean training
law.  The result identifies the *current optimizer chart*, not a historical or
data-generating sharing graph.

## Model and response operator

Let

\[
 A=\operatorname{softmax}_{\rm row}(Z)\in\mathbb R^{n\times K},
 \qquad B\in\mathbb R^{K\times d},
 \qquad \Theta=AB.
\]

The loss depends on the parameters only through `Theta`.  For known positive
block learning rates, consider Euclidean gradient flow

\[
 \dot Z=-\eta_Z\nabla_Z L,\qquad
 \dot B=-\eta_B\nabla_B L,
 \qquad \eta_Z,\eta_B>0.
\]

For an arbitrary effective gradient
`G=nabla_Theta L in R^{n x d}`, define

\[
 \mathcal R_{A,B}(G)=-\dot\Theta.
\]

Write the transpose of router row `i` as `a_i`, the transpose of gradient row
`i` as `g_i`, and

\[
 J_i=\operatorname{diag}(a_i)-a_i a_i^\top.
\]

Direct differentiation gives

\[
 \dot B=-\eta_B A^\top G,
 \qquad
 \dot a_i=-\eta_ZJ_i^2B g_i,
\]

and hence

\[
 [\mathcal R_{A,B}(G)]_i
 =\eta_B\sum_{j=1}^n(a_i^\top a_j)g_j
  +\eta_ZB^\top J_i^2B g_i.
 \tag{R1}
\]

Thus the `(i,j)` block of the finite-dimensional tangent response is

\[
 [\mathcal R_{A,B}]_{ij}
 =\eta_B(a_i^\top a_j)I_d
 +\mathbf 1\{i=j\}\eta_ZB^\top J_i^2B.
\tag{R2}
\]

Equations (R1)--(R2) use unit softmax temperature.  If
`A=softmax_row(Z/tau)` with known `tau>0` and `J_i` is still defined by (R1)
without the outer temperature factor, replace `eta_Z` everywhere in the
response and in the constants below by `eta_Z/tau^2`.

This Jacobian-Gram interpretation is standard NTK calculus; equation (R1) is
included to fix every orientation and learning-rate convention.

## Theorem 1: exact stabilizer

Assume all entries of `A` are positive,

\[
 \operatorname{rank}(A)=K,
 \qquad \operatorname{rank}(B)=K.
 \tag{R3}
\]

Let `(A',B')` satisfy the same assumptions and `A'B'=AB`.  Then the following
are equivalent:

1. `R_{A,B}(G)=R_{A',B'}(G)` for every `G in R^{n x d}`;
2. there is a permutation matrix `P` such that
   `A'=AP` and `B'=P^T B`.

The logits remain unidentified under the independent row shifts
`z_i -> z_i+c_i 1`.

### Proof

Both factorizations have rank `K`, so standard full-rank factorization algebra
gives a unique `M in GL(K)` with

\[
 A'=AM,\qquad B'=M^{-1}B.
 \tag{R4}
\]

Since both routers have unit row sums and `A` has full column rank,
`M1=1`.

Equality for every `G` means equality of every response block.  Off-diagonal
blocks of (R2) give

\[
 a_i^\top a_j=(a_i')^\top a_j'\quad(i\ne j).
 \tag{R5}
\]

For a diagonal block, put

\[
 H_i=B^\top J_i^2B,
 \qquad H_i'=(B')^\top(J_i')^2B'.
\]

Equality of the diagonal response blocks says, more explicitly,

\[
 H_i-H_i'=-{\eta_B\over\eta_Z}
 \bigl(\|a_i\|_2^2-\|a_i'\|_2^2\bigr)I_d=:c_iI_d
 \tag{R6}
\]

For every simplex row, `J_i` is positive semidefinite and
`J_i1=0`.  Full row rank of `B` therefore gives

\[
 H_i\succeq0,\quad \operatorname{rank}(H_i)\le K-1<d,
\]

and the same holds for `H_i'`.  If `c_i>0`, (R6) makes `H_i` positive
definite; if `c_i<0`, it makes `H_i'` positive definite.  Both contradict
singularity.  Therefore `c_i=0`, and the diagonal router Gram entries are
equal as well.  Together with (R5),

\[
 AA^\top=A'A'^\top=AMM^\top A^\top.
\]

Left and right multiplication by a left inverse of `A` show
`MM^T=I`; hence `M` is orthogonal and `B'=M^TB`.

Now `H_i=H_i'` and a right inverse of `B` imply

\[
 J_i^2=M(J_i')^2M^\top.
 \tag{R7}
\]

Both `J_i` and `M J_i'M^T` are positive semidefinite, so uniqueness of the
positive-semidefinite square root gives

\[
 J_i=MJ_i'M^\top.
 \tag{R8}
\]

Using `a_i'=M^Ta_i`, expand the two softmax Jacobians in (R8).  Their rank-one
terms cancel and leave

\[
 \operatorname{diag}(a_i)=
 M\operatorname{diag}(a_i')M^\top.
 \tag{R9}
\]

The rows of `A'` span `R^K`.  For any distinct `r,s`, the `(r,s)` entry of
(R9) says

\[
 \sum_{\ell=1}^K M_{r\ell}M_{s\ell}a'_{i\ell}=0
 \quad\text{for every }i.
\]

Spanning forces `M_{r ell}M_{s ell}=0` for every `ell`.  Each column of the
orthogonal `M` consequently has exactly one nonzero entry, equal to `+1` or
`-1`: `M` is a signed permutation.  Finally `M1=1` removes every negative
sign.  Thus `M=P` is a permutation.  Conversely, simultaneous permutation
commutes with the row softmax Jacobians and leaves (R2) unchanged.  This proves
both directions.

## Necessity and target boundaries

- **Router rank.**  For `K=3`, let every row of `A` equal `(1/3,1/3,1/3)` and
  let `B=I`.  A non-permutation orthogonal matrix fixing `1` leaves both
  `AA^T` and every local softmax term unchanged.
- **Basis rank.**  Let `A=0.3 11^T+0.1I_3` and
  `B=1 e_1^T`.  The router is full rank but the basis has rank one.  The same
  non-permutation orthogonal transformation fixing `1` preserves the response.
- **Joint training.**  If `eta_Z=0`, only `eta_B AA^T G` remains; for
  `K>=3` the non-permutation orthogonal subgroup fixing `1` survives.  Thus a
  positive router-block rate cannot be dropped uniformly, although this
  counterexample does not establish necessity at `K=2`.  The proof uses
  `eta_B>0` to recover the router Gram matrix, but
  this example does not establish that the basis-block rate is logically
  necessary; no claim about the `eta_B=0` stabilizer is made here.
- **All-gradient quantifier.**  One task gradient, or even one trajectory, can
  lie in the kernel of a nonzero response difference.  Stable ordinary
  training does not establish operator equality.
- **Optimizer meaning.**  The result is for a named Euclidean metric and known
  block rates.  Natural gradient deliberately removes chart dependence; Adam,
  momentum state, parameter-space regularizers, router auxiliary losses, and
  finite steps require separate statements.
- **Truth target.**  `(Theta,R)` identifies the present coordinates modulo
  labels.  It does not identify which graph or factors generated the data in
  the past.

Although strict positivity is automatic for finite softmax logits, the exact
algebra above only used that each router row lies in the closed simplex:
`J_i` being PSD and singular is enough in (R6).  Thus the formula-defined
closed-simplex extension of Theorem 1 remains valid at boundary rows, provided
both routers and both factors retain the stated full ranks.  Positivity becomes
quantitatively material in Theorem 2, where it supplies a uniform spectral gap.

## Theorem 2: quantitative stability

Let `K>=2`, `n>=K`, and `d>=K`.  Suppose `A,A'` are row-stochastic and

\[
 \min_{ij}\{A_{ij},A'_{ij}\}\ge\alpha>0,
\]

\[
 \sigma_{\min}(A),\sigma_{\min}(A')\ge s_A,
 \quad
 \sigma_{\min}(B),\sigma_{\min}(B')\ge s_B,
\]

\[
 \|A\|_F,\|A'\|_F\le L_A,
 \quad
 \|B\|_F,\|B'\|_F\le L_B,
 \quad AB=A'B'.
 \tag{R10}
\]

For the same positive `eta_Z,eta_B`, define

\[
 \delta=\|\mathcal R_{A,B}-\mathcal R_{A',B'}\|_{F\to F}.
\]

Then an explicit finite constant

\[
 C=C(n,K,\alpha,s_A,s_B,L_A,L_B,\eta_Z,\eta_B)
\]

satisfies

\[
 \min_{P\in\mathfrak S_K}
 \left(\|A'-AP\|_F^2+\|B'-P^TB\|_F^2\right)^{1/2}
 \le C\delta.
 \tag{R11}
\]

One fully explicit (loose) choice is

\[
 L_M=\max\{1,L_A/s_A\},\quad
 c_G={n\over\eta_Bs_A^2},\quad
 c_H={2\over\eta_Zs_B^2},
\]

\[
 \mu_*=\alpha\left(1+{1\over4L_M^2}\right),\quad
 c_J={L_M^2(c_G+c_H)\over\mu_*},\quad
 c_D={\sqrt{nK}\over s_A}c_J,
\]

\[
 c_{\rm row}=\sqrt{Kc_D^2+(c_G+Kc_D^2)^2},
\]

\[
 \delta_0=\min\left\{1,{1\over2c_G},
 {1\over2\sqrt K c_{\rm row}}\right\},
\]

\[
 C_{\rm loc}=
 \sqrt{L_A^2+(L_ML_B)^2}\sqrt Kc_{\rm row},
 \qquad
 C=\max\left\{C_{\rm loc},
 {2\sqrt{L_A^2+L_B^2}\over\delta_0}\right\}.
 \tag{R12}
\]

### Proof audit

All unlabelled matrix norms in this proof are spectral norms; response-operator
norms are induced by the Frobenius norm, and every occurrence of `||.||_F` is
written explicitly.  The full proof has four quantitative steps.

1. Injecting a vector into input block `j` and projecting output block `i`
   shows that each response block differs by at most `delta`.  Off-diagonal
   blocks therefore give
   `|(AA^T-A'A'^T)_{ij}|<=delta/eta_B` for `i!=j`.  In a diagonal block, write
   
   \[
     E_i=t_iI_d+\eta_Z(H_i-H_i'),\qquad
     t_i=\eta_B(\|a_i\|_2^2-\|a_i'\|_2^2).
   \]
   
   Both `H_i,H_i'` are singular PSD.  If `t_i>=0`, a unit vector in
   `ker(H_i')` has quadratic form at least `t_i` under `E_i`; if `t_i<0`, a
   unit vector in `ker(H_i)` has quadratic form at most `t_i`.  Hence
   `|t_i|<=delta`, and the triangle inequality gives
   `||H_i-H_i'||<=2delta/eta_Z`.  The diagonal Gram entries consequently obey
   the same `delta/eta_B` bound as the off-diagonal entries, so
   
   \[
     \|AA^T-A'A'^T\|_F\le {n\delta\over\eta_B}.
   \]
   
   Full-rank factorization algebra gives `A'=AM`, `B'=M^{-1}B`, and `M1=1`.
   Moreover
   
   \[
     M=A^\dagger A',\qquad M^{-1}=(A')^\dagger A,qquad
     \|M\|_2,\|M^{-1}\|_2\le L_A/s_A\le L_M. \tag{R13a}
   \]
   
   Multiplying the Gram difference on both sides by `A^dagger` gives
   \[
     \|I-MM^T\|_2\le c_G\delta. \tag{R13}
   \]

2. Let `Q=B^T(BB^T)^{-1}` be the minimum-norm right inverse of `B`.  It has
   shape `d x K`, satisfies `BQ=I_K`, and obeys `||Q||_2<=1/s_B`.  Thus
   `Q^T(H_i-H_i')Q` gives
   \[
   \|J_i^2-M^{-T}(J_i')^2M^{-1}\|_2\le c_H\delta.
   \]
   Put `Y_i=M^T J_iM`.  The exact decomposition
   
   \[
   \begin{split}
     Y_i^2-(J_i')^2
      ={}&M^TJ_i(MM^T-I)J_iM\\
        &+M^T\{J_i^2-M^{-T}(J_i')^2M^{-1}\}M
   \end{split}
   \]
   
   together with `||J_i||_2<=1`, (R13), and (R13a) yields
   \[
    \|Y_i^2-(J_i')^2\|_2
    \le L_M^2(c_G+c_H)\delta. \tag{R14}
   \]
   Both `Y_i` and `J_i'` have the exact kernel `span{1}`: indeed
   `ker(Y_i)=span{M^{-1}1}=span{1}`.  Symmetry implies that they preserve
   `1^perp`.  For any probability vector with entries at least `alpha`,
   
   \[
     J(a)\succeq\alpha\Pi_{1^\perp}. \tag{R14a}
   \]
   
   To verify (R14a), write `a=alpha 1+r`; for `x` orthogonal to `1`, weighted
   Cauchy--Schwarz and `sum_j r_j=1-K alpha<=1` give
   `sum_j r_jx_j^2-(sum_jr_jx_j)^2>=0`.
   Hence `J_i'|_{1^perp}>=alpha I`.  For a unit `x` orthogonal to `1`,
   decompose `Mx=c1+y`, with `y` orthogonal to `1`.  Since
   `x=c1+M^{-1}y` and `x` is orthogonal to `1`,
   
   \[
     |c|\sqrt K\le L_M\|y\|_2,\qquad
     1\le |c|\sqrt K+L_M\|y\|_2,
   \]
   
   and therefore `||y||_2>=1/(2L_M)`.  Equation (R14a) then gives
   `Y_i|_{1^perp}>=alpha/(4L_M^2) I`.  On `1^perp`, put
   `D_i=Y_i-J_i'` and use the Sylvester equation
   \[
     Y_iD_i+D_iJ_i'=Y_i^2-(J_i')^2.
   \]
   Its inverse has the integral representation
   
   \[
     D_i=\int_0^\infty e^{-tY_i}
       \{Y_i^2-(J_i')^2\}e^{-tJ_i'}\,dt
   \]
   
   on that subspace, so its spectral operator norm is at most `1/mu_*`.
   Both sides vanish on `span{1}`; (R14) therefore proves
   `||M^T J_iM-J_i'||_2<=c_J delta` on the whole space.  The shared exact
   kernel is what makes this estimate linear rather than a generic
   square-root Hölder bound.

3. Since `a_i'=M^Ta_i`, expanding `J_i,J_i'` cancels their rank-one terms and
   gives
   
   \[
     \|M^T\operatorname{diag}(a_i)M-
       \operatorname{diag}(a_i')\|_2\le c_J\delta. \tag{R14b}
   \]
   
   Taking off-diagonal parts is contractive in Frobenius norm (not in spectral
   norm), and `||X||_F<=sqrt(K)||X||_2`.  With
   `X_j=Off(M^TE_{jj}M)`, (R14b) thus says
   
   \[
     \left\|\sum_{j=1}^K A_{ij}X_j\right\|_F
       \le\sqrt Kc_J\delta\quad(1\le i\le n).
   \]
   
   Apply row `j` of `A^dagger` in the direct-sum Frobenius space.  Its
   Euclidean norm is at most `1/s_A`, and Cauchy--Schwarz over the `n` rows
   yields
   \[
     \|\operatorname{Off}(M^TE_{jj}M)\|_F\le c_D\delta. \tag{R15}
   \]

4. `M^TE_jjM=r_jr_j^T`, where `r_j^T` is row `j` of `M`.  Let `q` maximize
   `|r_{jq}|`.  For `delta<=delta_0`, (R13) gives
   `||r_j||_2^2>=1/2`, whence `r_{jq}^2>=1/(2K)`.  Since the squared
   off-diagonal Frobenius norm contains
   `2r_{jq}^2 sum_{ell!=q}r_{jell}^2`, (R15) gives
   
   \[
     \sum_{\ell\ne q}r_{j\ell}^2\le Kc_D^2\delta^2,\qquad
     |r_{jq}^2-1|\le(c_G+Kc_D^2)\delta,
   \]
   
   where the second inequality also uses `delta<=1`.  Consequently row `j`
   is within `c_row delta` of a signed coordinate vector.  Collect these row
   choices into `S`.  Then
   `||M-S||_2<=||M-S||_F<=sqrt(K)c_row delta<=1/2`.  If two rows chose the same
   coordinate, `S` would be singular and hence `sigma_min(M)<=1/2`; this
   contradicts (R13), which gives `sigma_min(M)>=1/sqrt(2)`.  Thus the chosen
   coordinates are distinct.  Moreover a row of `M` has sum one.  Its distance
   from a negative coordinate vector is therefore at least `2/sqrt(K)`, while
   the chosen-row distance is at most `1/(2sqrt(K))`.  Every sign is positive,
   so `S=P` is a permutation and
   `||M-P||_F<=sqrt(K)c_row delta`.  Multiplication by `A`, and the identity
   `M^{-1}-P^T=M^{-1}(P-M)P^T`, give the local part of (R11).  For
   `delta>=delta_0`, the parameter-set diameter gives the second term in
   (R12).

The equality `AB=A'B'` is essential: `(A,B)` and `(A,-B)` have the same
response operator but opposite effective matrices.

### Explicit degeneration checks for the quantitative margins

The dependence on `alpha,s_A,s_B` is not merely an artifact of the displayed
proof.  The following full-rank families show that no uniform linear inverse
constant can ignore any one of them.  In all cases `K=n=d=3`, transformed
factors use `(A',B')=(A Q,Q^T B)` with an orthogonal `Q` fixing `1`, so the
effective product and router-Gram response agree exactly.

1. **Router rank.**  Let `U=11^T/3`,
   `A_eps=U+eps(I-U)`, `B=I`, and
   `Q=(1/3)[[2,-1,2],[-1,2,2],[2,2,-1]]`.
   Then `sigma_min(A_eps)=eps`, while the routers stay uniformly interior and
   the basis margins/norms stay fixed.  At `eps=0`, the uniform covariance
   `(I-U)/3` commutes with `Q`; hence the response gap is `O(eps)`, whereas
   the permutation-orbit parameter distance tends to
   `min_P ||Q-P^T||_F>0`.

2. **Basis rank.**  With the same `Q`, set
   `A=.3 11^T+.1I`, `B_eps=1e_1^T+eps I`.  The router margins are fixed and
   `sigma_min(B_eps)->0`.  Since every covariance kills `1`, all terms
   involving `1e_1^T` vanish:
   `B_eps^T J_i^2 B_eps=eps^2 J_i^2` and the primed analogue is
   `eps^2 Q(J_i')^2Q^T`.  Thus the response gap is `O(eps^2)`, while the
   chart distance has a positive limit because full-rank `A Q` is not in the
   permutation orbit of `A`.

3. **Simplex interior.**  Put
   `S=(1/sqrt(3))[[0,-1,1],[1,0,-1],[-1,1,0]]`,
   `Q_t=exp(tS)`, `A_t=t11^T+(1-3t)I`, and `B=I`.
   Since `S^T=-S` and `S1=0`, `Q_t` is orthogonal and fixes `1`.
   The off-diagonal first-order entries of `A_tQ_t` are
   `t(1 +/- 1/sqrt(3))+O(t^2)`, so both routers are feasible for small
   positive `t`.  Their smallest singular values are `1-3t`, and all basis
   singular values equal one; only the minimum router entry, `Theta(t)`,
   degenerates.  Covariances at the near-vertex rows are `O(t)`, so their
   squared local response gap is `O(t^2)`, while
   `||Q_t-I||_F=sqrt(2)t+O(t^2)` makes the chart distance `Theta(t)`.
   Hence the inverse ratio diverges like `1/t`.

These constructions are also executable regression tests in
`tests/test_response_quantitative_stability.py`.  They are stability
counterexamples after deleting a uniform margin, not violations of the exact
stabilizer at any positive `eps` or `t`.

### Independent small-`K` adversarial check

The following computation is a counterexample search, not part of the proof.
With PyTorch float64 and seed 20260823, for each `K in {2,3,4}` we generated
100 instances with `n=d=K+1`, Gaussian-softmax `A`, Gaussian full-row-rank
`B`, and
`M=I+tD`, where each Gaussian row of `D` was centered so that `D1=0`.
Starting at `t=0.03`, `t` was halved until `min(AM)>10^{-5}` and
`cond(M)<20`.  We materialized the full `nd`-by-`nd` block operator,
used `eta_Z=0.7, eta_B=1.3`, enumerated every permutation, and recomputed all
constants in (R12) from each pair.  Across 300 cases the largest observed
left-hand side of (R11) divided by `C delta` was
`1.286e-7`; no violation occurred.

To exercise the local rather than diameter branch, we also used
```text
A = [[0.8, 0.2], [0.2, 0.8]]
B = [[1.0, -0.3, 0.2], [0.1, 0.9, -0.7]]
D = [[1.0, -1.0], [-0.4, 0.4]]
```
and `M=I+tD` for `t in {10^{-8},10^{-9},10^{-10},10^{-11}}`.  Every case
had `delta<delta_0`.  The largest observed ratios of the left to the claimed
right side in (R13), the `Y_i-J_i'` bound, (R15), and the row-rounding bound
were respectively `0.4684`, `0.00913`, `0.00220`, and
`2.50e-6`.  These tests specifically exercise the `M` orientation,
singular common kernel, and spectral/Frobenius conversions, but of course
cannot establish the theorem.

## Finite Gaussian response probes

Let `m=nd` and represent the difference of two response operators by
`D in R^{m x m}`.  For independent `x_l~N(0,I_m)`:

- `q=m` noiseless probes recover an arbitrary `D` almost surely by
  `D=Y X^{-1}`.  With `q>=m`, use `D=YX^T(XX^T)^{-1}`.  Uniform recovery of an
  arbitrary response operator is impossible for `q<m`, because a nonzero
  operator can annihilate the probe span.
- For testing one fixed alternative, if `sigma=||D||_2`, an SVD argument gives
  `||Dx||>=sigma|Z|`, `Z~N(0,1)`.  Hence
  \[
  \Pr\left[\max_{\ell\le q}\|Dx_\ell\|<t\sigma\right]
  \le[2\Phi(t)-1]^q. \tag{R16}
  \]
  It is enough that
  \[
   q\ge{\log(1/\beta)\over-\log(2\Phi(t)-1)}.
  \]
  At `t=1/2`, this is about `1.04 log(1/beta)` probes.

Combining (R11) and (R16) tests a fixed factorization at distance at least
`Delta` from the permutation orbit.  This does not turn a handful of probes
into uniform recovery of every possible alternative.

The implementation also reports
`||Dx||_F/max{||R x||_F,||R' x||_F}` against a user-set numerical threshold.
That scale-free ratio is a useful empirical diagnostic, but (R16) is an
absolute small-ball bound and does not calibrate this relative threshold unless
one additionally bounds the two response norms.  It must not be reported as a
confidence level or a finite-probe equality certificate.

## Prior-art status

- Static `Theta=AB` non-identifiability and the uniformizing path are exact
  SSMF prior art after transposition (Vu Thanh, Gillis, and Lecron, 2023).
- Jacobian-Gram response operators are NTK background (Jacot et al., 2018).
- Marcotte, Peyre, and Gribonval (ICLR 2026) give the direct general
  matrix-factor response `G -> UU^T G + G VV^T` for the lift `UV^T`; they do
  not include the row-softmax blocks or classify a common-product fiber.
- LoRA Done RITE (Yen et al., ICLR 2025) and LoRA-S (Zheng and Wu, ICLR
  2026) directly show that ordinary low-rank factor optimization can depend
  on the chart and construct transformation-invariant updates.  With
  `U=A`, `V=B^T`, and `R=M`, their factor gauge is the present gauge before
  router feasibility is imposed.  They do not classify the ordinary
  row-softmax/shared-basis response stabilizer or its inverse.
- Zhang and Pilanci (ICML 2024) derive Riemannian low-rank preconditioners for
  LoRA and analyze their feature-learning stability.  This is another direct
  geometry-aware factor-optimization precedent, not a classification of the
  standard Euclidean response on the present shared-router fiber.
- Varre, Rofin, and Flammarion (2026) derive the corresponding value-softmax
  gradient-flow terms for one softmax mixture.
- Singh (2026), Lemma 3.2, identifies the orthogonal isometry subgroup inside
  the full matrix-factorization `GL(K)` gauge.  The `GL -> O` step alone is
  not new.
- Lau and Su (2026) identify expert permutation and logit-shift symmetries in
  softmax MoE routers.  Permutation symmetry alone is not new.
- Tran et al. (NeurIPS 2025, Theorem 4.1) identify a dense softmax MoE
  function up to expert permutation and common gate translation under
  distinctness conditions.  Nguyen, Nguyen, and Ho (NeurIPS 2023,
  Proposition 1 and Theorem 1) give the corresponding label/translation
  quotient and a global linear density-to-parameter inverse for an
  exact-fitted Gaussian softmax MoE.  These function/density observables and
  expert assumptions differ from `(Theta,R)`; they nevertheless preclude any
  claim that softmax permutation/translation identifiability or softmax
  inverse stability is new in general.
- Tu et al. (ICML 2016, Lemma 5.4) give the Gram-to-Procrustes linear
  stability step used inside the quantitative chain.
- Dorrell, Latham, and Whittington (ICLR 2026, Theorem 2) force an
  orthogonal same-Gram change of a nonnegative representation to be a
  permutation under additional tight-scattering/convex-hull conditions.  In
  the mapping `Z=A^T`, `Z'=A'^T`, and `O=M^T`, this is a direct
  orthogonal-to-permutation precedent.  A finite strictly positive router
  cannot satisfy their first ellipsoid-containment condition, so their theorem
  does not replace the present local softmax-covariance step.
- Afsari (2008, Theorem 2.3) shows that a family
  `M^T Diag(a_i) M` determines the joint diagonalizer up to permutation and
  diagonal scaling when the diagonal profiles are pairwise noncollinear;
  here the profile matrix is exactly `A`, and full column rank is stronger
  than that condition.  His Theorems 4.1--4.2 also give local first-order
  sensitivity for particular approximate joint-diagonalization costs.
- Bhaskara, Charikar, and Vijayaraghavan (COLT 2014, Theorem 5) give global
  robust Kruskal uniqueness.  Stacking our close diagonal families compares
  the bounded decompositions `[I,I,A']` and `[M^T,M^T,A]`, whose reference
  robust ranks are `K,K,K`; thus the robust diagonal-tensor-to-permutation
  step also has a direct polynomial-stability precedent.  Neither result
  derives the diagonal family from the response or enforces the common exact
  product.
- Shem-Ur, Vrabel, Walters, and Oz (ICLR 2026 submission, Theorem 4.1 as
  exposed by the official OpenReview search index) explicitly allow a
  function-preserving parameter symmetry to retain the NTK linearization
  regime while changing the kernel.  This is a direct conceptual precedent
  for same-function/different-Euclidean-response behavior, but it gives no
  equal-kernel converse, shared-simplex `AB` calculation, or response inverse.
  The direct PDF endpoint was browser-challenged at the audit date, so this
  entry is limited to official metadata and indexed theorem text.
- van Oostrum, Muller, and Ay (2023) prove natural-gradient pushforward
  reparameterization invariance, while Mishra et al. (2014) quotient the full
  low-rank factor gauge under invariant metrics.  Thus the stabilizer theorem
  is specific to the stated Euclidean `Z/B` law; it is false as a
  metric-independent claim.
- Wang and Wang (NeurReps 2025) characterize the generic function-preserving
  Q/K and V/output `GL` gauge of standard multi-head attention and study its
  quotient/Fisher geometry.  Their parameter blocks and function/quotient
  observable differ from the row-simplex shared-router factorization and do
  not give a complete Euclidean-response stabilizer on a common `AB` fiber.
- Goel, Soltanolkotabi, and Bartlett (arXiv:2603.01514, 2026) reduce a Gaussian
  softmax self-attention population loss to a weighted matrix-factorization
  loss and analyze preconditioned optimization.  The Gaussian expectation
  cancels the softmax normalization before their factor loss is formed; their
  results concern convergence to a regularized optimum manifold, not chart
  identification from equality of complete Jacobian--Gram response.

The candidate addition is restricted to the combined statement: for two
jointly learned shared softmax-router/basis charts on one common exact product
fiber, the complete finite-width Euclidean tangent-response stabilizer is
exactly simultaneous basis permutation under the named full-rank conditions,
with a linear quantitative inverse on a named nondegenerate set.  Independent
prior-art search and counterexample attempts have not found this exact chain.
It must not be described as the invention of tangent kernels, factor response,
optimizer coordinate dependence, orthogonal factor symmetry, router
permutation/translation identifiability, joint-diagonalizer uniqueness,
robust tensor-factor stability, or generic softmax inverse stability.
