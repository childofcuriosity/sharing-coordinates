# Focused noisy-recovery research review — 2026-09-25

## Outcome

The focused study is complete as a bounded research result. It improves
finite-noise factor recovery substantially, but does not establish a general
trustworthy-recovery procedure. In particular, the combined observable
rejection diagnostic misses its calibration target on held-out synthetic
data and does not uniformly beat residual-only rejection. We retain this
negative result rather than tune against the held-out outcomes.

The user-authorized scope was known positive Euclidean block rates and finite
noisy product/response observations. No Adam history, historical sharing graph,
general-purpose compression benefit, or public submission was added.

## Evidence obtained

- Six fresh fixed-budget byte-LM checkpoints: two calibration and four
  evaluation, each 6000 steps. Training uses local GPUs, with fixed data and
  source hashes. Development used a separate existing checkpoint.
- Twenty calibration and forty held-out synthetic seeds, with four regimes,
  six noise levels and two probe budgets. The 1920 synthetic held-out rows
  are 40 independent instances, not 1920 independent replications.
- A constrained joint estimator, projected-spectral comparator and original
  spectral comparator. Recovery interfaces do not receive true factors.
  Truth enters only scoring/calibration. Exact objective compression retains
  all orthogonal residual constants in the estimated noisy product subspace.
- The empirical threshold source and calibration data were frozen before
  held-out recovery. Scores and thresholds were not changed after evaluation.
  Local files provide execution provenance, not independent preregistration.
- An explicit model-specific curvature bound and a conditional local
  feasible-component proposition. Standard Taylor/Jacobian reasoning and
  joint-diagonalization sensitivity are attributed to prior work.
  Sixteen small full-coordinate sanity checks satisfy the local bound.
- A full independent CPU replay of the 1968 held-out cases, not just a small
  favorable subset. Good/intermediate/bad classifications all agree.
  Four spectral and two joint-fit acceptance decisions differ, and the largest
  joint-fit error difference is 0.0170033. No bitwise reproducibility claim.

## Main held-out findings

| Result | Projected spectral | Joint fit |
|---|---:|---:|
| Synthetic median factor error | 0.00333818 | 0.000100566 |
| Synthetic bad outputs (error >0.10) | 566/1920 | 227/1920 |
| Synthetic estimator failures | 3 | 0 returned-estimate failures |
| Byte-LM maximum factor error | 0.489855 | 0.0106343 |
| Byte-LM bad outputs | 1/48 | 0/48 |

Seventy selected synthetic joint fits exhaust the evaluation budget. A
returned estimate must not be described as certified convergence.

For the joint-fit combined diagnostic, synthetic coverage is 92.8125%,
bad-output rejection recall is 59.4714%, good-output false rejection is
0.1830%, and bad fraction among accepted outputs is 92/1782 = 5.1627%.
The seed-cluster 95% bootstrap interval for the last quantity is
[4.5043%, 5.7939%]. A calibration target of 5% is not a held-out guarantee.

Residual-only acceptance retains 1757/1920 and 68 bad outputs (3.8702%),
rejecting 70.0441% of bad outputs with the same good false-rejection rate.
The combined score has slightly greater coverage but is not uniformly better.
Condition-only rejection has very high good false rejection and rejects
every language output. The raw spectral feasibility gate also rejects every
language output. Those are uninformative coverage failures.

The language cohort has four independent checkpoints and no bad joint fits.
Consequently its bad-output rejection recall is undefined, not perfect.
All 48 cases are accurate and accepted within the tested noise grid only.

## Failure analysis

The rank-contracted regimes account for all 227 bad joint fits:
62 at contraction 0.1, and 165 at contraction 0.01. A pooled threshold hides
subgroup risk: the combined score accepts 55 bad outputs among 473 accepted
at contraction 0.1, and 37 among 352 at contraction 0.01. These rates are much
worse than the aggregate. Interior and near-one-hot regimes have no bad
joint fits on the present grid; that is not a general domain guarantee.

The largest error, 10.5414, occurs at synthetic seed 201, contraction 0.01,
noise 0.001, q=4. All three starts exhaust the budget and the selected
residual is 0.298, so this is not evidence that the global inverse itself
must have that error. It is a failure of the bounded recovery procedure.

The largest falsely accepted error, 0.81973, occurs at seed 210 in the same
regime/noise/probe condition. Training residual is 0.0906 against a supplied
noise budget 0.001416, held-out residual is 0.0947, and the selected run
exhausts its budget. The combined score 9.015 passes the calibrated threshold
14.791; residual-only rejection rejects it. This exposes how maximizing
pooled coverage can license poor individual cases. Agreement among a few
starts is not a global search certificate.

Only 140/1920 synthetic and 2/48 language estimates pass the numerical local
conditions. These are candidate-component statements with floating SVDs,
not interval-certified global truth-error bounds. The true full basis need
not lie in the estimated row subspace. A useful unconditional certificate
remains unresolved.

## Review disposition and next scientific boundary

The justified addition is a better scoped noisy estimator, exact objective
compression, an explicit local separation calculation, and an honest
calibration-transfer failure study. Do not advertise the combined score as
a superior or guaranteed trust mechanism. The paper, tables, and abstract
state this boundary.

A future study could test a conservative noise-consistency gate or
regime-aware calibration on entirely fresh development/calibration/evaluation
cohorts, with guarantees addressing subspace misspecification and distant
components. Those options have not been tested here and are not silently
substituted for the frozen primary diagnostic.

Scientific-critical-thinking informed the prospective split/confounder and
independent-unit design; paper-review informed comparison against simple
baselines and reporting negative outcomes; ml-paper-writing informed the
claim/evidence separation. They are procedural aids, not independent expert
review or validation. Source files and skill attribution remain available
in research/SKILL_REFERENCES.json and the manuscript disclosure.

## Artifacts

- Protocol and source freeze: TRUSTED_RECOVERY_PROTOCOL.md and
  TRUSTED_RECOVERY_FREEZE.json.
- Proof and literature boundary: TRUSTED_RECOVERY_PROOF.md.
- Thresholds: ../results/trusted_recovery/calibration_policy.json.
- Complete results: ../results/trusted_recovery/summary.json.
- Independent replay: TRUSTED_REPLAY_COMPARISON.json.
- Verification record: TRUSTED_RECOVERY_VALIDATION.json.
- Manuscript: ../paper/main.pdf; local arXiv package under ../dist/.
