# Sharing Coordinates: Identifiability from Optimizer Response

The [current study review](research/LLM_REVIEW.md) recommends this as a scoped theoretical preprint with pretrained-language-model evidence. The manuscript retains the ICLR2027 template and author placeholder; nothing has been uploaded.

Read [paper/main.pdf](paper/main.pdf). The theory covers exact identification, a uniform inverse on a nondegenerate domain, and a whole-feasible-set finite-noisy consequence. New experiments test shared modules on **nine pretrained models from 160M to 8B**, with three adapter seeds and general-text/code/math gradient domains. All 1,080 sufficient-rank joint fits have factor error below 0.034; 540 insufficient-query cases are explicitly rejected. These controlled FP64 factor-response audits do not identify an AdamW trajectory or the whole LLM.

The structural result is deliberately bounded: recovered routers reproduce native partitions, but a simpler effective-parameter clustering control does too. We therefore do not claim that recovery improves structural decisions or compression. Noisy spectral failures and false acceptances remain visible.

See [LLM reproduction](research/LLM_REPRODUCIBILITY.md), [full accounting](results/llm/summary.json), [self-review](research/review_llm_extension.txt), and [current validation](research/LLM_VALIDATION.json). The earlier [theory release review](research/RELEASE_REVIEW.md) and [validation](research/RELEASE_VALIDATION.json) describe the preserved 54-page snapshot, not the newly expanded manuscript. That snapshot is retained in [research/releases/2026-09-25-theory](research/releases/2026-09-25-theory/).

The latest [theory result](research/FINITE_NOISY_THEOREM.md) proves an explicit
global feasible-set diameter bound from K unit-norm probes chosen using a
noisy product. The proof handles the entire feasible set without a true
subspace or local-component assumption, under stated nondegeneracy margins.
[Independent audits](research/FINITE_NOISY_ADVERSARIAL_AUDIT.md) record the
dependency checks and a corrected quantifier issue. This is a measurement-level
corollary of the existing full-operator inverse, not a solver guarantee.

The preceding [focused noisy-recovery study](research/TRUSTED_RECOVERY_REVIEW.md)
adds constrained joint fitting, a conditional local bound, and a frozen
calibration/evaluation experiment. Recovery improves, but the combined
rejection score misses its calibration target and does not uniformly beat
residual-only rejection. [Reproduction](research/TRUSTED_REPRODUCIBILITY.md)
includes the complete independent CPU replay and retained failures.

The [2026-09-25 enhancement review](research/ENHANCEMENT_REVIEW.md) reports
three new trained-router, equal-size folding audits, six natural-task-gradient
measurements, a small-product-perturbation stability extension and explicit
precision/conditioning failures. See [reproduction instructions](REPRODUCIBILITY.md)
and the prospective [protocol](research/ENHANCEMENT_PROTOCOL.md).

Earlier delivery snapshots and their limitations are retained in
[the historical review](research/FINAL_REVIEW.md) and
[delivery status](research/DELIVERY_STATUS.md); the release review above
supersedes their current-status judgments. The manuscript is
[paper/main.pdf](paper/main.pdf); rebuild the arXiv source draft using
[these instructions](research/ARXIV_PREPARATION.md).

This repository is the released artifact for a paper about learned cross-layer
parameter sharing. The model studied in the main theory is

\[
\Theta = AB,
\]

where every row of the router `A` lies in a simplex and the rows of `B` are
jointly learned bases. The repository deliberately separates three questions:

1. whether a static factorization is unique;
2. whether the parameterization has the same instantaneous training response;
3. whether a coordinate-derived structural action is well defined.

For `K >= 2`, the elementary static ambiguity is **not presented as a new theorem**. After
transposition it is the unconstrained simplex-structured matrix-factorization
ambiguity studied by Vu Thanh, Gillis, and Lecron (TSP 2023); the old
uniformizing construction is their construction under a change of variables.
The central theorem instead characterizes the full Euclidean effective-gradient
response of a finite softmax router with learned bases. Under strict router
positivity, full column/row rank, a fixed positive temperature, and the same
known positive Euclidean router/basis rates, two charts of the same exact
effective product have equal response for every effective gradient if and only
if they differ by a common basis permutation. These are stated sufficient hypotheses; in particular the
general stabilizer after setting the basis rate to zero remains unresolved for
signed gauges with three or more bases. The theorem identifies a current
parameter representation modulo labels; it does not recover a historical
sharing graph or certify an AdamW trajectory.

## Original audit inventory

The original empirical artifact has three non-interchangeable audits.
These records are retained as numerical or auxiliary decision evidence;
they do not define the final paper's theoretical contribution:

- **Frozen-checkpoint response audit.** At three byte-LM checkpoints, a fixed
  non-permutation gauge preserves the effective parameters to about `1.7e-16`
  relative error but changes the Euclidean instantaneous response by about
  `0.505` in median relative norm. All 24 checkpoint--probe evaluations
  (eight per checkpoint, using overlapping deterministic seed windows with
  10 unique Gaussian draws) detect this particular alternative; they are not
  a universal equality test. A separate
  structured-tomography audit uses four deterministic probes at each
  `(L,K,D)=(12,4,787712)` checkpoint. Under the exact common-product and rank
  conditions, those probes determine the complete structured response; the
  maximum component-reconstruction error is `6.85e-11`, permutation controls
  remain below `9.44e-17`, and the minimum across seeds of the per-seed maximum
  fixed non-permutation gap over the four designed probes is `0.9731` after
  rounding. The four
  full-dimensional responses are analytic evaluations of the response-block
  formula, which was independently checked against autograd and finite
  differences on the declared `4 x 32` extraction; they are not four full-model
  JVP or autograd measurements.
- **ASLoRA first-merge decision audit.** A documented reimplementation of the
  paper-defined MRPC settings freezes the model immediately before the first
  merge. A finite, deterministic bank of 10 gauges changes the raw-running-`B`
  action in 11/30 primary adjacent-mode instances and 14/30 primary all-pairs
  instances, with different validation outcomes after the selected layer-index
  action is executed in the unchanged canonical model. Generation sources and
  external inputs are byte-bound, and two complete deterministic passes over
  every selected canonical action agree exactly. Table 5's reported
  update ratio `lambda=0.5` has no defined operation in the available method
  text, so it is recorded but deliberately not reverse-engineered. This is not
  an official-code or official-checkpoint reproduction and is not a prevalence
  estimate.
- **Byte-LM partition decision audit.** At three trained checkpoints whose
  routers all meet the protocol's initialization-dominated flag, a predeclared
  5,000-trial positive-stochastic search finds a same-group-count partition
  change in only one checkpoint. In that controlled witness, refitting both
  partitions from the same effective tensors changes fold BPB from `2.1753`
  to `4.8149`; the paired interval for the difference is
  `[2.6287, 2.6510]`. At the successful seed, gauge-invariant k-means and Ward
  baselines both give `2.1753` BPB. The two failed-search records contain no
  fold or baseline evaluation. Signed-local and same-sorted-group-size
  searches were not executed; they are post-run non-primary protocol
  amendments. The supported conclusion is a controlled existence witness,
  not learned-router evidence or cross-seed replication.

The older planted factorization, distillation, gauge, saturation, and
ShareProbe sweeps remain available as diagnostic material. They establish
possibility and implementation checks, not real-system prevalence. The revised
ShareProbe sweep is a strict paired-common-random-number comparison: shared-
input and native signatures use the same initial probes, optional projection,
and fixed-scale iid Gaussian observation noise, while native states advance
only with clean updates. Mean ARI is `0.676` versus `0.675` for the MLP and
`0.501` versus `0.501` for the Transformer, with no consistent per-seed
advantage. A visible-effective-parameter-distance baseline reaches `1.000` in
  both planted architectures. That baseline does not test functional equivalence,
  but it shows that ShareProbe is unnecessary for this planted parameter-partition
  benchmark when the relevant effective block parameters are directly available.
  ShareProbe therefore remains an appendix diagnostic rather than a headline method.

## Artifact map

The 2026-09-24 additive audit in `results/autograd_tomography/` measures the
full packed-factor response with reverse-mode gradients and forward-mode JVPs.
All three checkpoints pass the prospective protocol in
`research/AUTOGRAD_TOMOGRAPHY_PROTOCOL.md`; the largest formula discrepancy is
`3.71e-16`, and reconstructed responses predict two unseen probes per chart
with maximum relative error `5.08e-14`. This is an independent observation
implementation for the shared-parameter map, not an end-to-end loss or AdamW
trajectory. Original analytic records are retained.

The active arXiv preparation plan is `research/PROJECT_PLAN.md`.
Build a minimal source draft with `.venv/bin/python scripts/package_arxiv.py`;
see `research/ARXIV_PREPARATION.md` for the isolated compilation check and
remaining author-review items. This command does not upload the paper.

- `paper/main.tex` and `paper/main.pdf`: manuscript source and compiled paper.
- `research/PRIOR_ART_REGISTRY.md`: theorem-by-theorem prior-art mapping.
- `research/ROUTE_REGISTRY.md`: mechanism-level route ledger with accepted,
  demoted, and explicitly blocked directions.
- `research/LITERATURE_EVIDENCE.md`: primary-source claim audit separating
  historical-structure claims from allocation, compression, and evaluated
  current-model decisions.
- `research/RESPONSE_IDENTIFIABILITY.md`: response theorem derivation and audit.
- `research/ETA_B_ZERO_BOUNDARY.md`: proofs for four closed partial
  zero-basis-rate subfamilies and the exact unresolved general signed-gauge
  gap for `K>=3`.
- `research/RESPONSE_TOMOGRAPHY.md`: proof, assumptions, counterexamples, and
  prior-art boundary for the finite structured response probes.
- `research/RESPONSE_CANONICAL_INDEPENDENT_AUDIT.md`: independent CPU
  regeneration of the released checkpoint response audit.
- `research/ROUTER_DECISION_CANONICAL_INDEPENDENT_AUDIT.md`: independent CPU
  regeneration of the released byte-LM search and folding consequence.
- `research/EMPIRICAL_PROTOCOL.md`: frozen primary byte-LM decision protocol,
  amendment ledger, execution status, and evidentiary boundary.
- `research/ASLORA_EMPIRICAL_PROTOCOL.md`: ASLoRA protocol and amendment
  chronology, including the explicit superseded-to-canonical transition.
- `research/ASLORA_CANONICAL_INDEPENDENT_AUDIT.md`: independent recalculation
  of the canonical ASLoRA counts, gates, and provenance bindings.
- `research/DECISION_THEORY.md`: decision definitions, limits, and counterexamples.
- `research/SHAREPROBE_PROTOCOL.md`: paired-probe protocol, amendment chronology,
  source/result hashes, and the negative baseline comparison.
- `results/response_summary.json`: validated response-audit aggregate.
- `results/response_tomography_seed{0,1,2}.json` and
  `results/response_tomography_summary.json`: source-attested structured
  tomography records and validated aggregate.
- `results/aslora_summary.json`: validated three-seed ASLoRA aggregate.
- `results/router_decisions_summary.json`: validated byte-LM decision aggregate.
- `RESULTS.md`: numerical ledger and claim boundaries.
- `REPRODUCIBILITY.md`: exact verification and regeneration commands.
- `MANIFEST.sha256`: hashes of the released sources, data, checkpoints, results,
  generated tables/figures, and paper.

## Quick verification

From the repository root:

```bash
python -m pip install -e ".[test]"
bash scripts/run_all.sh verify
```

`verify` reruns the tests, reconstructs all four primary summaries in a
temporary directory, compares them byte-for-byte with the canonical outputs,
checks the generated primary LaTeX tables and tomography macros, recomputes the
legacy aggregate in a temporary directory, and verifies `MANIFEST.sha256`. It
does not retrain the GPU models or overwrite results. See
`REPRODUCIBILITY.md` for the expensive raw regeneration modes and their
external dependency requirements.
