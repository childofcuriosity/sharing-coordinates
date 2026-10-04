# Gauge-selectable structural decisions in shared-factor adapters

This note is a proof ledger, not manuscript prose.  It records the candidate
central result before any novelty or empirical claim is promoted.

## Setup and estimand

Let `L` layer adapters share an input-side factor

\[
  \Delta W_i=B_iA,
  \qquad B_i\in\mathbb R^{d_o\times r},
  \quad A\in\mathbb R^{r\times d_i}.
\]

The common change of coordinates

\[
  B_{i,Q}=B_iQ,\qquad A_Q=Q^{-1}A,qquad Q\in GL(r)
\]

leaves every effective update exactly fixed.  A running average of the
output-side factors transforms in the same way, because
`mean_t(B_i^t Q) = mean_t(B_i^t) Q` for a fixed counterfactual `Q`.

Let `P` be the finite set of directed merge actions eligible at one decision
time (all pairs or adjacent pairs must be specified).  Write
`p=(ell <- u)` when the lower layer will use the upper layer's *current*
factor.  The ASLoRA selection score instead uses historical running averages:

\[
 \bar D_p=\bar B_\ell-\bar B_u,
 \qquad S_p^{\rm score}=\bar D_p^\top\bar D_p\succeq0.
\]

ASLoRA's raw squared score in gauge `Q` is

\[
 s_p(Q)=\|\bar D_pQ\|_F^2
       =\langle C_Q,S_p^{\rm score}\rangle,
 \qquad C_Q=QQ^\top\succ0.
\]

Conversely every positive-definite `C` equals `QQ^T` for some invertible
`Q`.  Hence the gauge orbit of the raw rule is exactly the family of all
positive-definite Mahalanobis metrics on the shared rank space.  General LoRA
gauge non-uniqueness and the need for invariant downstream processing are
prior art; the candidate addition here is the exact *action set and risk
envelope* for a real cross-layer merge rule.

## Theorem A: exact order and action-set audit

For `p,q in P`:

1. `s_p(Q) <= s_q(Q)` for every `Q in GL(r)` if and only if
   `S_p \preceq S_q`.
2. The inequality is strict for every `Q` if and only if
   `S_p \preceq S_q` and `S_p != S_q`.
3. The order can be reversed by two gauges if and only if
   `S_q-S_p` is indefinite.

Consequently, `p` is the unique raw nearest action in every gauge if and only
if `S_p \preceq S_q` and `S_p != S_q` for every `q != p`.  This requires a
least element in Loewner order, not merely a minimal element.

For the more useful question “can a gauge make `p` the unique action?”, define

\[
 \kappa_p=
 \max_{C\succeq0,\ \operatorname{tr}C=1}
 \min_{q\ne p}\langle C,S_q-S_p\rangle.
 \tag{A1}
\]

Then

\[
 p\text{ is the unique nearest action for some }Q\in GL(r)
 \quad\Longleftrightarrow\quad \kappa_p>0.
 \tag{A2}
\]

Moreover, with `Delta(P)` the probability simplex over competitors,

\[
 \kappa_p=
 \min_{\lambda\in\Delta(P\setminus\{p\})}
 \lambda_{\max}\!\left(
    \sum_{q\ne p}\lambda_q(S_q-S_p)
 \right).
 \tag{A3}
\]

Thus any density matrix `C` supplies a rigorous lower bound

\[
 \min_{q\ne p}\langle C,S_q-S_p\rangle\le\kappa_p,
\]

and any simplex vector `lambda` supplies a rigorous upper bound through the
right side of (A3).  Approximate numerical optimization must report these as
bounds, not as an exact SDP solution.  A positive lower bound certifies that
an explicit gauge selects `p`; a non-positive upper bound certifies that no
gauge uniquely selects it.

### Proof

For parts 1--3, `QQ^T` ranges over the positive-definite cone.  If
`S_q-S_p` is positive semidefinite, its trace product with every
positive-definite matrix is nonnegative and is positive unless the difference
is zero.  Conversely, if a symmetric matrix `H` has a vector `v` with
`v^T H v<0`, then

\[
 C_\epsilon=vv^\top+\epsilon I\succ0
\]

has `tr(C_epsilon H)<0` for sufficiently small positive `epsilon`.  Applying
this construction to positive and negative eigendirections proves reversal
for an indefinite difference.

If a positive-definite, trace-normalized `C` uniquely selects `p`, finiteness
of `P` gives a strictly positive minimum gap, so `kappa_p>0`.  Conversely, if
the maximum in (A1) is positive but attained at a singular `C_0`, then

\[
 C_\epsilon=(1-\epsilon)C_0+\epsilon I/r
\]

is positive definite and preserves every strict gap for all sufficiently
small `epsilon`; a Cholesky factor supplies `Q`.

Finally, write the inner minimum as a minimization over the competitor
simplex.  The density-matrix set and the simplex are compact and convex and
the payoff is bilinear, so the finite-dimensional minimax theorem exchanges
max and min.  For fixed symmetric `H`,

\[
 \max_{C\succeq0,\operatorname{tr}C=1}\langle C,H\rangle
 =\lambda_{\max}(H),
\]

which gives (A3).

The PSD metric-learning literature already uses closely related triplet-margin
SDPs.  Novelty, if the literature audit continues to support it, is restricted
to interpreting the complete SPD cone as one shared-factor gauge orbit and
using (A1)--(A3) to enumerate the structural actions and risk that an unchanged
model can induce.

## Theorem B: score/action separation and the absence of a generic regret bound

At decision time `t`, define the *current* directed action difference and its
scatter by

\[
 D_p^{\rm act}=B_\ell^t-B_u^t,
 \qquad
 T_p^{\rm act}=(D_p^{\rm act})^\top D_p^{\rm act}.
 \tag{B1}
\]

Let `x_ell` be the actual input to the lower adapter before the merge and let

\[
 M_\ell=\mathbb E[x_\ell x_\ell^\top],
 \qquad G_p=A M_\ell A^\top.
\]

The uncentred second moment is used intentionally; a covariance alone would
omit the mean outer product.  Including the LoRA scale separately, the exact
immediate adapter-output mean-squared disturbance is

\[
 c_p^{\rm act}
 =\mathbb E\|(B_u^t-B_\ell^t)Ax_\ell\|_2^2
 =\langle G_p,T_p^{\rm act}\rangle.
 \tag{B2}
\]

It is invariant under the common gauge.  It is not a network-level validation
loss; that consequence must be measured after actually applying the action.

Without assumptions linking the running-average score scatter
`S_p^{score}`, the current action scatter `T_p^{act}`, and the action-specific
moment `G_p`, no finite multiplicative or additive regret guarantee exists --
even in rank one, where there is no gauge ranking ambiguity.

### Counterexample

Take scalar rank, `A=1`, and four layers observed at two times:

\[
 B^1=(2-M,0,3,0),\qquad B^2=(M,0,1,0).
\]

The running average is `(1,0,2,0)`, so the three adjacent squared scores are
`(1,4,4)` and the rule uniquely chooses `1 <- 2`.  At the current time the
three current squared action differences are `(M^2,1,1)`.  With identical
unit input moments, the selected action has cost `M^2` while an available
action has cost `1`.  Regret is unbounded as `M` grows although all scalar
metric condition numbers equal one.  Thus the earlier version of this ledger,
which substituted running-average scatter into the current merge cost, was
incorrect.

Layer-specific moments create a second independent obstruction.  Even when
`S_p^{score}=T_p^{act}`, two actions with scatters `1` and `4` are ranked in
that order by raw distance; assigning lower-layer moments `M^2` and `1` makes
their actual costs `M^2` and `4`.  There is no common action metric `G` to
which a single condition-number comparison applies.

### Conditional common-scatter lemma

The old spectral mismatch bound survives only under the explicit special
case

\[
 S_p^{\rm score}=T_p^{\rm act}=S_p,
 \qquad G_p=G\succ0\quad\text{for every }p.
 \tag{B3}
\]

If raw metric `C=QQ^T` selects `p` and `p_star` minimizes
`c_q=<G,S_q>`, then, whenever `c_{p_star}>0`,

\[
 {c_p\over c_{p_\star}}
 \le
 \operatorname{cond}(C^{-1/2}GC^{-1/2}).
 \tag{B4}
\]

Indeed, for `T=C^{1/2}SC^{1/2}` and
`H=C^{-1/2}GC^{-1/2}`,

\[
 \lambda_{\min}(H)\operatorname{tr}(CS)
 \le\operatorname{tr}(GS)
 \le\lambda_{\max}(H)\operatorname{tr}(CS),
\]

and the raw selected-pair inequality proves (B4).  With adverse tie-breaking
the constant is attained by rank-one scatters along the two extreme
eigendirections; under unique selection it is a supremum approached after an
arbitrarily small score perturbation.

## Corollary: whitening is only a special-case coordinate convention

Under the common-moment assumption in (B3), choosing `Q=G^{1/2}` gives

\[
 A_Q M A_Q^\top=I,
 \qquad \|(B_i-B_j)Q\|_F^2=\langle G,S_{ij}\rangle.
\]

This is an action-matched gauge convention.  Real cross-layer actions usually
have different lower-layer moments `G_p`, so one common gauge cannot whiten
them all.  Singular empirical moments require ridge regularization and then no
longer give the exact action cost.  In every case, whitening fixes coordinates;
it does not recover historical factors or a latent sharing graph.

For an unweighted current-parameter proxy, the invariant product distance
`||(B_i^t-B_j^t)A||_F` remains cheap to compute from the small Gram `AA^T`.
The corresponding distance using `bar B` is only a running-average selection
proxy, not the exact current action disturbance.

## Theorem C: unbounded structural regret over a model family

Rank `r>=2` is necessary for raw-distance reversal.  Consider three scalar
output adapters, adjacent candidate actions `(1,2)` and `(2,3)`, and

\[
 A=I_2,\quad B_1=(0,0),\quad B_2=(1,0),\quad
 B_3=(1,R),\qquad R>1.
\]

In the identity gauge, raw nearest merging chooses `(1,2)`, whose substitution
cost is `1`.  With

\[
 Q=\operatorname{diag}(1,(2R)^{-1}),
\]

all parent effective updates remain exactly the same, but raw distances become
`1` and `1/2`, so the rule chooses `(2,3)`, whose invariant cost is `R^2`.
The approximation ratio is `R^2` and the additive regret is `R^2-1`.

The unbounded statement is a supremum over this family of models.  For one
fixed checkpoint with finitely many eligible actions, invariant action costs
remain a finite set along its gauge orbit.  In rank one every raw distance is
multiplied by the same nonzero scalar, so reversal is impossible.

## Assumption-removal checks

- **Common factor and common gauge.**  The theorem requires one shared `A` and
  the same `Q` for every compared layer.  With independent `A_i,B_i,Q_i`, the
  difference is `B_iQ_i-B_jQ_j` and the scatter-matrix reduction fails.
- **Compatible shapes and scaling.**  Compared adapters must have the same
  rank and output shape and must use the same LoRA scale.  Otherwise the
  action must first be defined in effective-update space.
- **Full row rank.**  Invariance and Theorem A do not need full row rank of
  `A`; whitening and the finite condition-number bound need `G` positive
  definite.  If `G` is singular, raw differences in its nullspace can be
  functionally invisible.
- **Fixed decision time.**  A counterfactual gauge of a checkpoint is not a
  claim that ordinary Adam would generate both training traces.  The parent
  function is identical at the decision; later optimization is a separate
  source of coordinate dependence and must be reset/canonicalized in a fair
  repair experiment.
- **Running averages.**  A fixed post-hoc `Q` legitimately transforms the
  complete running average.  A time-varying gauge would introduce additional
  terms and is outside this result.
- **Task loss.**  `c_p` is exact adapter-output perturbation under the named
  covariance, not a universal bound on a nonlinear network's validation loss.
  Actual downstream loss must be measured.
- **Historical structure.**  Every conclusion is about whether a merge action
  is well-defined on the current function's gauge quotient.  No conclusion
  identifies a historical or data-generating sharing graph.
