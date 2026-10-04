# Audit: finite-sample safety certificates for structural actions

## 1. Objects that must not be conflated

Let `P` be a parameter space and let `~` denote equality of the deployed soft
model function.  Write `q=[theta]` for one equivalence orbit.  These are three
different objects:

1. A **coordinate rule** `r : P -> A` reads a particular parameterization and
   emits a hard action (for example, an argmax partition).
2. An **action function** `T_a(theta)` is the model deployed after action `a`.
   It is quotient-valid only if `T_a(theta)=T_a(theta')` whenever
   `theta ~ theta'`.
3. A **historical truth** is a latent statement about how the checkpoint was
   generated.  Neither low action risk nor quotient invariance identifies it.

For a finite action universe `A`, define the exact orbit image of a coordinate
rule

`S_r(q) = {r(theta') : theta' in q}`.

An implemented certificate may replace it by a known orbit-complete superset
`S(q)`.  Testing a few gauges does not establish orbit completeness.

## 2. The strongest clean finite-sample statement

### Theorem 1 (orbit-complete paired-loss certificate)

Fix integers `n>=1` and `K=|A|>=2`, confidence `delta in (0,1)`, and loss bound
`L>0`.  Assume:

- `T_a` is quotient-valid for every `a in A`;
- `S(q)` contains `S_r(q)`;
- `theta`, all action functions, `r`, and `S(q)` are frozen independently of
  the entire calibration sample;
- `Z_1,...,Z_n` are iid from the deployment distribution and independent of
  training/action construction;
- `0 <= ell(T_a(q),Z) <= L` almost surely for every action.

Set

`R_q(a) = E ell(T_a(q),Z)`

and, for ordered distinct actions,

`Dhat_ab = n^{-1} sum_i [ell(T_a(q),Z_i)-ell(T_b(q),Z_i)]`.

Let `M=|S(q)|(K-1)` and

`c_n = L sqrt(2 log(M/delta)/n)`.

Define the computable certificate

`C_n(S) = max(0, max_{a in S(q), b in A, b != a} (Dhat_ab+c_n))`.

Then, with probability at least `1-delta`, simultaneously for **every**
representative `theta' in q`,

`R_q(r(theta')) - min_{b in A} R_q(b) <= C_n(S)`.

Consequently, observing `C_n(S)<=epsilon` certifies that every coordinate
choice reachable on the entire orbit has regret at most `epsilon`.

If `S(q)` is allowed to depend on calibration labels, the same result remains
valid after replacing `M` by `K(K-1)` and constructing simultaneous bounds for
all ordered action pairs.  It does not remain valid if the action functions
themselves are fitted on the same calibration labels without a complexity or
data-splitting correction.

#### Proof

For fixed `(a,b)`, the paired difference lies in `[-L,L]`.  Hoeffding gives

`Pr{R_q(a)-R_q(b) > Dhat_ab+t} <= exp(-n t^2/(2L^2))`.

Substituting `t=c_n` and taking a union bound over the `M` ordered pairs gives,
with probability at least `1-delta`,

`R_q(a)-R_q(b) <= Dhat_ab+c_n`

for every `a in S(q)` and every `b != a`.  On that event, for any
`theta' in q`, orbit completeness gives `r(theta') in S(q)`, and

```
R_q(r(theta')) - min_b R_q(b)
  = max_b [R_q(r(theta'))-R_q(b)]
  <= C_n(S).
```

Quotient-validity is what makes `R_q(a)` well-defined independently of the
representative used to materialize action `a`.  QED.

### Proposition 2 (identity invariance is algebraic, not statistical)

For a coordinate rule `r`, the following are equivalent:

1. `r(theta)=r(theta')` whenever `theta~theta'`;
2. every exact orbit image `S_r(q)` is a singleton;
3. there exists a map `rbar : P/~ -> A` such that `r=rbar o pi`, where `pi` is
   the quotient map.

#### Proof

`1=>2` is immediate.  Under 2, define `rbar(q)` as the unique member of
`S_r(q)`, giving 3.  Condition 3 immediately gives 1.  QED.

Thus no confidence interval can turn a nonconstant coordinate rule into an
orbit-invariant rule.  Theorem 1 certifies *outcome safety despite varying
actions*; it does not certify common action identity.  Both desired properties
hold only if Proposition 2 is established separately and Theorem 1 accepts.

### Corollary 3 (quotient empirical-risk selector)

Suppose the action universe, action functions, and a deterministic tie order
are fixed on the quotient.  Let

`ahat = canonical argmin_{a in A} Rhat(a)`.

Then `ahat` is exactly the same for every representative of `q`.  With

`c_full = L sqrt(2 log(K(K-1)/delta)/n)`,

`R_q(ahat)-min_a R_q(a) <= c_full`

with probability at least `1-delta`.

#### Proof

All empirical action losses are quotient functions, so the deterministic
argmin is representation-invariant.  Apply the simultaneous ordered-pair event
from Theorem 1 to the data-dependent `ahat`.  Since
`Rhat(ahat)-Rhat(b)<=0` for every `b`, the regret is at most `c_full`.  QED.

This is the familiar finite-class holdout oracle inequality in quotient
notation.

### Variance-sensitive version

Let `Vhat_ab` be the unbiased sample variance of the paired differences.
Rescaling Maurer--Pontil's empirical Bernstein inequality from `[0,1]` to
`[-L,L]` and applying a union bound replaces `c_n` by the pair-specific radius

`sqrt(2 Vhat_ab log(2M/delta)/n) + 14 L log(2M/delta)/(3(n-1))`.

This is tighter when paired action losses are highly correlated.  It is a
direct application of an existing empirical Bernstein theorem, not a new
result.

### Fixed-corpus version matching the WikiText sampler

If the estimand is explicitly the average over a fixed population of `N`
disjoint corpus blocks, and `n` blocks are sampled uniformly without
replacement, Serfling's moment-generating-function bound gives the same theorem
with

`c_{n,N} = L sqrt(2 rho_n log(M/delta)/n)`,

where `rho_n=1-(n-1)/N`.  This avoids an iid language assumption but certifies
only the finite-corpus average.  It remains a direct union-bound application of
Serfling (1974), and raw NLL still needs a valid global bound or clipping.

## 3. Necessary assumptions and explicit counterexamples

### C1. Small regret does not imply invariant action identity

Let two distinct actions deploy exactly the same predictor and have zero loss.
Let `r` choose the first action at one representative and the second at another.
Orbit regret and risk diameter are zero, but `S_r(q)` has size two.  Therefore
no risk certificate proves a recovered graph or a common action.

### C2. Quotient-valid materialization is necessary

Parameterize the same scalar soft function by `(u,v)` with `uv=1`.  Give an
action label `a` the coordinate-dependent deployed predictor
`T_a(u,v)=constant(u)`.  Representatives `(1,1)` and `(2,1/2)` are soft-model
equivalent, yet action `a` deploys constants 1 and 2.  Calibration at the first
representative cannot certify the second.  Literal hardening after an inverse
gauge has precisely this type of coordinate dependence; common-effective
partition folding was introduced to avoid it.

### C3. Orbit completeness is necessary

Let `A={a0,a1}`, with risks 0 and 1.  Suppose the current representative emits
`a0`, another orbit representative emits `a1`, but the claimed orbit set is
`S={a0}`.  The computed certificate can be zero although the worst orbit action
has unit regret.  A finite random gauge search is evidence about sampled
gauges, not a proof that `S` covers the orbit.

### C4. Boundedness or a tail model is necessary

Let the comparator loss be zero.  Let the candidate loss be `H` with
probability `2 epsilon/H` and zero otherwise.  Its risk is `2 epsilon`, while
the probability that `n` calibration observations all show zero tends to one
as `H` grows.  Hence no nontrivial distribution-free finite-sample certificate
for unbounded NLL is possible.  Clipping NLL certifies clipped risk, not raw
cross-entropy.

### C5. Calibration/deployment agreement is necessary

If calibration puts all mass on an input where two actions agree and deployment
puts mass on an input where one fails, every calibration difference is zero
while deployment regret can be one.  Nonoverlapping blocks remove literal data
reuse; they do not by themselves establish iid sampling from a future language
distribution.

### C6. Exact optimal-action recovery needs a margin

For paired differences in `{-1,+1}` with means `+gamma` and `-gamma`, the two
data-generating laws become arbitrarily hard to distinguish when
`gamma=O(n^{-1/2})`.  No finite-sample selector uniformly identifies the true
better action without a risk gap assumption.  Epsilon-regret certification can
abstain and remains possible; exact optimal identity cannot be promised.

### C7. Historical truth remains unidentified

Two histories can produce the same current quotient object and identical
action functions but assign different latent sharing graphs.  All calibration
distributions are then identical.  Theorem 1 concerns present decision risk
only and contains no estimator of historical truth.

## 4. Prior-art mapping and novelty verdict

The statistical mechanism is already standard:

- finite-class simultaneous Hoeffding and variance-sensitive versions are
  covered directly by Maurer and Pontil, *Empirical Bernstein Bounds and Sample
  Variance Penalization* (COLT 2009), especially their finite-class corollary;
- validation-set oracle inequalities are classical model-selection results,
  including van der Vaart, Dudoit, and van der Laan (2006);
- high-confidence selection against a baseline is the premise of Thomas,
  Theocharous, and Ghavamzadeh, *High Confidence Policy Improvement* (ICML
  2015), and simultaneous evaluation of several policies is studied by Dann,
  Ghavamzadeh, and Marinov (AISTATS 2023);
- holdout calibration for risk control is central to Bates et al.,
  *Distribution-Free, Risk-Controlling Prediction Sets* (JACM 2021);
- finite-population concentration is Serfling (1974) and is sharpened by
  Bardenet and Maillard (Bernoulli 2015);
- invariant decision rules and orbit reductions long predate this application;
  see Eaton, *Group Invariance Applications in Statistics* (1989).

The exact mapping is: actions are the finite hypothesis/policy class, deployed
loss is hypothesis loss/negative policy value, the calibration sample is the
holdout set, and the orbit is a nuisance transformation group.  The proof adds
only a union bound plus the set-theoretic fact that invariant maps factor
through a quotient.  Pairing improves the numerical radius but not the
mechanism.

**Verdict:** correct and operationally useful, but **blocked as a central new
theoretical contribution**.  It can be retained as a safety protocol or a
formally delimited corollary with explicit attribution.  Presenting it as a new
identifiability theorem would be indefensible.

Moreover, the current WikiText experiment does not meet the iid/bounded-loss
version: it evaluates raw NLL on blocks of one corpus.  The finite-population
version is possible after declaring the fixed-corpus estimand and a predeclared
clipped loss or valid global loss bound.  The existing paired bootstrap is
descriptive, not this certificate.

## 5. Pivot: architecture-constrained positive functional recovery

The following avoids matrix factorization entirely, but its assumptions expose
why it is not yet a central result for the current architecture.

### Theorem 4 (functional equality graph from an isolated unisolvent probe interface)

Let `V=span{phi_1,...,phi_d}` be a known `d`-dimensional real vector space of
vector-valued layer-update functions.  Suppose every layer update has

`u_i(x)=sum_{j=1}^d c_{ij} phi_j(x)`.

Choose probes `x_1,...,x_m` and let `E` be the stacked evaluation operator from
coefficients to `(u(x_1),...,u(x_m))`.  Assume `rank(E)=d` and that the
architecture exposes each isolated update `u_i(x_s)`.  Then the current
functional equality graph is exactly recoverable:

`u_i=u_j  iff  E c_i=E c_j`.

With observations `y_i=E c_i+e_i`, `||e_i||_2<=eta`, let
`chat_i=E^dagger y_i` and `kappa=||E^dagger||_2`.  Equal functions satisfy

`||chat_i-chat_j||_2 <= 2 kappa eta`.

If every unequal pair has coefficient separation at least `Delta` and
`Delta>4 kappa eta`, then any threshold

`2 kappa eta < tau < Delta-2 kappa eta`

recovers the equality graph exactly.

#### Proof

Full column rank makes `E` injective, proving the noiseless equivalence.
Furthermore

`||chat_i-c_i|| <= ||E^dagger|| ||e_i|| <= kappa eta`.

The equal-pair upper bound and unequal-pair lower bound
`Delta-2 kappa eta` follow from the triangle inequality.  The stated gap admits
the threshold.  QED.

#### Necessity and scope

- If `rank(E)<d`, a nonzero coefficient vector in `ker(E)` gives distinct
  functions identical on every probe.
- Without a known finite-dimensional function class, any finite probe set is
  defeated even by a nonzero polynomial that vanishes on all probes.
- Without separation, bounded noise makes an equal and unequal pair
  indistinguishable.
- Without isolated layer hooks, full-model behavior need not identify the
  layer graph.  For scalar residual multipliers, the triples `(1,1,4)` and
  `(1,2,2)` have the same end-to-end product 4 but different equality
  partitions.
- The theorem recovers current functional equality only.  It recovers neither
  router coordinates nor historical tying.

This theorem is linear unisolvent interpolation.  More general finite-sample
neural identifiability and rank conditions already appear in Stock and
Gribonval (2021) and Bona-Pellissier, Malgouyres, and Bachoc (NeurIPS 2022),
while global parameter uniqueness modulo symmetries goes back at least to
Sussmann (1992) for minimal tanh networks.

**Pivot verdict:** this is a valid positive boundary result but also not a new
center.  A viable architecture-specific center would need a substantially
weaker, realistic observation model than isolated hooks plus a known
finite-dimensional function space, while proving recovery of the exact target
used in experiments.  No such theorem was established on this route.
