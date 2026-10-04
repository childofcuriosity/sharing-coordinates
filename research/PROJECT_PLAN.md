# Sharing Coordinates: research-draft and enhancement plan

## Publication-readiness goal — 2026-09-25

The user requested a goal that ends only when recommending public release
is scientifically justified. The final acceptance evidence is in
[RELEASE_REVIEW.md](RELEASE_REVIEW.md), supported by a fresh mathematical
rebuild, primary-source comparison, and editorial claim/evidence review.
The recommendation concerns a theoretical arXiv preprint, with known domain
and measurement conditions, preserved negative results, and no actual upload.
The finite-noisy theorem is not presented as an empirical solver certificate.

## Precise theory goal — 2026-09-25

The user rejected treating experiments as completion of a theory objective.
The replacement acceptance contract is FINITE_NOISY_GOAL.md. The proof and
explicit constants are in FINITE_NOISY_THEOREM.md and the new manuscript
appendix; separate derivation, foundation audit and adversarial audit are
retained. Completion is based on the proved whole-feasible-set statement,
not on benchmarks or packaging. It is a conditional measurement-level
corollary, with novelty and solver limitations stated explicitly.

## Focused goal execution — 2026-09-25

The authorized unattended noisy-recovery work is recorded in
[TRUSTED_RECOVERY_REVIEW.md](TRUSTED_RECOVERY_REVIEW.md), with its prospective
protocol, immutable source/threshold freezes, held-out results and independent
CPU replay. The stronger unconditional trust guarantee remains unproved;
the negative calibration-transfer result is retained in the paper.


## Authorized enhancement — 2026-09-25

The user explicitly started a new goal after the preceding completed delivery.
ENHANCEMENT_PROTOCOL.md controls fresh training, equal-size paired actions,
natural gradients, precision and joint-noise diagnostics. ENHANCEMENT_REVIEW.md
records outcomes; ENHANCEMENT_VALIDATION.json records final artifact checks.
Earlier completion statements below describe the preceding milestone.

Owner: user; execution: Codex. Started 2026-09-24.
User priority: an arXiv-ready research presentation within several days; choose
a submission venue later. No public upload is implied by local preparation.
User clarification: retain the existing ICLR 2027 template and keep author
metadata as placeholders for now. Do not spend further effort changing the
layout. Deferred author details are not a blocker for this research deliverable.

## Completion criteria

1. Main identifiability and quantitative-stability arguments have explicit
   assumption-by-assumption checks, with unresolved points clearly marked.
2. Closest prior results are compared using primary full text, with novelty
   stated only at the granularity supported by that comparison.
3. Main empirical claims have source-bound artifacts and reproducible commands;
   negative results, finite-grid scope and initialization limitations remain visible.
4. Revised source and compiled PDF agree, figures/tables are checked, numerical
   summaries are derived, tests and manifest checks pass in the project environment.
5. An arXiv source bundle and status report are ready for author review. Author
   identity and actual public submission remain user-controlled.

## Work order

- Foundation: preserve original scientific artifacts, establish a verified baseline
  and an isolated Python environment plus local LaTeX build. Baseline, environment
  restoration, local compilation and the first additive-release verification completed.
- Theory: exact stabilizer, inverse stability, boundary cases, finite probes;
  primary-source novelty comparison and an independent derivation ledger.
- Evidence: full-dimensional automatic-differentiation tomography first;
  factor recovery/noise diagnostics and saved-record action analysis next if they
  answer a clear claim. No automatic expansion to costly unrelated benchmarks.
- Manuscript: clarify one central contribution, connect observations to conclusions,
  improve exposition and figures, retain explicit attribution and limitations.
- Release: clean source archive, regenerated PDF, no stale tables, reproducibility
  instructions and checked status of every intended claim.

## Resource policy

Eight RTX 5090 cards were visible and idle at initial inspection, with approximately
31.36 GiB available per card. Recheck before launching jobs; use distinct GPUs for
independent seeds. Start with a small implementation test and record source hashes
before full runs. Preserve originals and write new experiments to new directories.
Do not infer multi-seed success from a successful individual run.

The scientific success criterion is a defensible result, including informative
negative results. Neither acceptance nor a predetermined positive effect is a gate.

## Current milestone

Three full-dimensional autodiff tomography runs completed on GPUs 0,1,2 using
the original checkpoints; see `AUTOGRAD_TOMOGRAPHY_PROTOCOL.md` and additive
`results/autograd_tomography/seed*.json`. All predeclared numerical gates pass.
These are JVPs of the packed factor map, not the language-model forward loss.
The manuscript compiles locally in preprint-draft layout. Scoped exact,
quantitative, finite-probe and secondary theorem proof reviews are recorded in
`PROOF_REVIEW_2026-09-24.md`; theorem-level primary-source comparisons are in
`frontier_response_identifiability.md`. The empirical appendix audit and full
43-page visual pass are recorded in `EMPIRICAL_REVIEW_2026-09-24.md` and
`VISUAL_REVIEW.md`. Citation coverage now has 47 primary-text checks and 2 explicitly limited
dispositions; final release closure is recorded in `DELIVERY_STATUS.md` and `FINAL_REVIEW.md`. Constructive
factor recovery now has an explicit derivation, three synthetic autodiff tests
and a source-bound three-checkpoint pilot in `results/factor_recovery/`.
All noiseless pilot gates pass; noise diagnostics remain exploratory and do
not close the general noisy inverse problem. A 39-file arXiv draft archive
now compiles in a fresh directory with identical extracted PDF text; see
`ARXIV_PREPARATION.md`. Author metadata is intentionally deferred by the user;
human-author approval remains outside this agent's scientific self-review.

The authorized enhancement is now complete within its registered scope. Both
environments pass 156 tests; evidence and remaining scientific limits are in
ENHANCEMENT_REVIEW.md and ENHANCEMENT_VALIDATION.json. Public submission and
author identities remain deferred per the user.
