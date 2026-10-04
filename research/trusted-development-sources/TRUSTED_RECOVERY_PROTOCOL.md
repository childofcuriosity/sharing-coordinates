# Trusted noisy recovery: prospective protocol — 2026-09-25

Authorized unattended research goal: make noisy factor recovery assessable,
with separate mathematical local guarantees and empirically calibrated
rejection. The original source-bound studies are immutable. Known positive
Euclidean rates eta_Z=eta_B=tau=1; no unknown Adam state.

## Splits and independent units

- Development: new synthetic seeds 0..3 and existing byte-LM seed 10 only.
- Calibration: new synthetic seeds 100..119 and new byte-LM seeds 20,21.
- Held-out evaluation: synthetic seeds 200..239 and new byte-LM seeds 22..25.
- Fresh language models reuse the fixed width-128, L=12,K=4, 6000-step
  training configuration from train_enhancement.py. No training-setting
  search, early stopping, replacement or selection on recovery outcomes.
- Fixed synthetic factors L=6,K=3,D=16. Interior, rank contractions .1,.01,
  and boundary contraction .01. Random normal bases; seeds offset 3000000.
- The independent units are synthetic instances and trained checkpoints,
  not their many noise/probe variants. Report individual checkpoint results
  and seed-cluster bootstrap uncertainty on aggregate synthetic comparisons.

## Observation regimes

Synthetic Gaussian gradient directions, or actual cross-entropy effective
gradients for byte-LMs. Directions are normalized and known exactly. At each
checkpoint fix 12 fitting and 4 held-out directions, minibatches 2x32 tokens,
training-split sampling offset 3100000. Fit q=4 and q=12; the same last four
directions test response fit. Noise levels 0,1e-8,1e-6,1e-4,1e-3,1e-2.
Independent product/response perturbations have fixed relative Frobenius
norm, with common random directions across levels. Known relative upper bound
sigma is provided as observation metadata; the estimator may use the
observable absolute upper bound sigma/(1-sigma)*norm(observation), never the
actual hidden error vector or true factors. Roundoff floor is reported
separately from the mathematical noise model. Hold-out response observations
have noise too. No selection using clean hold-out responses.

Re-estimate a rank-K row subspace from noisy Theta. This introduces model
approximation error and must not be silently treated as the true row space.
Any certificate in the restricted subspace is conditional on that model;
full-factor truth need not satisfy it. Synthetic full-coordinate checks will
separately test local bounds without the subspace restriction.

## Methods and development budget

Compare existing spectral recovery; simplex-projected spectral initialization
with least-squares basis refit; and constrained joint fitting of observed
Theta and responses, using a fixed small number of starts and finite budget.
Fit constraints: row-simplex A; B represented in the estimated row subspace.
The fit objective is relative squared product plus response residual. Preserve
orthogonal residuals when reporting full-data fit; compression may remove
only candidate-independent terms. Development may set solver budget/scaling,
then record a freeze before calibration/evaluation. Maximum 3 solver variants
and 3 starts per variant; no test-set debugging other than recorded genuine
implementation corrections followed by complete reruns.

## Credibility diagnostics and predeclared outcomes

Observable candidates: residual relative to supplied noise budget, held-out
response fit, smallest singular value of the simplex-tangent measurement
Jacobian, factor feasibility/rank, estimated product subspace conditioning,
and disagreement between competitive solver starts after permutation.
Truth may be used only by the isolated evaluation/calibration scorer.

Primary evaluation error is combined relative Frobenius factor distance after
one common permutation. Good <=.05; bad >.10; intermediate (.05,.10] reported
separately. Solver failure is a rejected failure and remains in denominators.
Report recovery error distributions, accepted fraction, bad-output rejection
(recall), good-output false rejection, and bad fraction among accepted outputs.
An all-reject diagnostic is uninformative, not success.

Empirical threshold: choose on calibration only, maximizing accepted coverage
subject to at most 5% bad outputs among accepted calibration outputs and a
minimum of 20 accepted synthetic outputs. If infeasible, report failure; do
not relax constraints after viewing held-out outcomes. No population risk
guarantee follows from this calibration. Compare against residual-only and
condition-only diagnostics. Lock method, score and threshold JSON before
held-out recovery results are read.

## Mathematical scope

Derive an explicit Jacobian-Lipschitz local separation bound with observable
candidate-dependent constants, simplex-feasibility radius, residual and known
noise budget. Distinguish local feasible-component containment from a global
truth-error certificate. Numerical singular values without outward-rounded
enclosures are numerical evaluations of a real-arithmetic theorem, not
machine-verified proofs. Test whether the bound is useful; retain vacuous
outcomes. Generic nonlinear inverse and joint-diagonalization conditioning
theory must be attributed rather than presented as new foundations.

## Deliverable

Prospective source-bound records, frozen calibration, blind held-out results,
unit tests for algebra/compression/leakage and gate failure, clean-environment
checks, updated manuscript and local archive. Retain ICLR 2027 and author
placeholders. No public upload. Scientific-critical-thinking, paper-review
and ml-paper-writing guide design, evidence boundaries and presentation;
the existing Scientific Agent Skills attribution remains in the manuscript.
