# Results and evidence boundaries

This ledger is derived from four strict summary JSON files covering three
primary evidence families. Numbers below were checked against those files; the
JSON remains authoritative. The old synthetic aggregate is retained separately
and is not pooled with the primary audits.

## Canonical primary records

| Evidence family | Raw records | Validated aggregate |
|---|---|---|
| Frozen byte-LM response | `results/primary_remote/response_audit_seed{0,1,2}.json` | `results/response_summary.json` |
| Structured response tomography | `results/response_tomography_seed{0,1,2}.json` | `results/response_tomography_summary.json` |
| ASLoRA MRPC first merge | `results/aslora_seed{0,1,2}/audit.json` plus adjacent provenance/diagnostics/snapshot | `results/aslora_summary.json` |
| Byte-LM partition decision | `results/primary_remote/router_decisions_seed{0,1,2}.json` | `results/router_decisions_summary.json` |

Independent reconstruction ledgers are provided for the
[response audit](research/RESPONSE_CANONICAL_INDEPENDENT_AUDIT.md),
[ASLoRA audit](research/ASLORA_CANONICAL_INDEPENDENT_AUDIT.md), and
[byte-LM decision audit](research/ROUTER_DECISION_CANONICAL_INDEPENDENT_AUDIT.md).
They are cross-checks; the validated canonical summaries above remain the
source of reported manuscript numbers.

The three 1,500-step byte-LM checkpoint hashes are recorded in
`REPRODUCIBILITY.md` and bound by `MANIFEST.sha256`. The matching per-seed
training JSON files are `results/primary_remote/language_seed{0,1,2}.json`.

## 1. Frozen-checkpoint Euclidean response

The audit extracts `A` and `B` from each trained byte-LM checkpoint, applies the
fixed positive-stochastic non-permutation gauge
`M = 0.65 I + 0.35 11^T/K`, and evaluates the analytic instantaneous response
on eight fixed Gaussian effective-gradient probes. A basis permutation is the
negative control.

| Seed | min entry of A | sigma_min(A) | sigma_min(B) | effective relative error | permutation max response difference | non-permutation median response difference | detected |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.883e-4 | 1.729 | 27.445 | 1.583e-16 | 8.820e-17 | 0.504500 | 8/8 |
| 1 | 2.717e-4 | 1.729 | 27.339 | 1.698e-16 | 8.820e-17 | 0.505131 | 8/8 |
| 2 | 2.613e-4 | 1.729 | 27.371 | 1.632e-16 | 8.824e-17 | 0.505791 | 8/8 |

Across seeds, the maximum effective relative error is
`1.697808166886258e-16`; all rank assumptions, analytic/autograd/finite-
difference formula checks, and permutation controls pass. All 24
checkpoint--probe evaluations detect this fixed alternative. There are eight
evaluations per checkpoint; the deterministic seed windows overlap, so these
24 evaluations use 10 unique Gaussian draws rather than 24 independent draws.

**Boundary.** The theorem concerns equality of the entire Euclidean response
operator for every effective gradient. Twenty-four successful checkpoint--probe
evaluations at three checkpoints validate the implementation and show a large
discrepancy for one
predeclared alternative; they do not certify operator inequality for every
possible alternative. The response is not the saved AdamW trajectory,
optimizer-selection stability, or historical sharing truth.

### Structured four-probe tomography

The additive tomography audit uses the common effective row space to design
four deterministic probes at each checkpoint. Here `(L,K,D)=(12,4,787712)`,
so the sufficient packing condition `D-K >= L` holds. The first probe encodes
the complete router Gram matrix in `L` orthogonal complement directions; the
remaining probes reconstruct every router-local block on the `K`-dimensional
effective row space.

| Seed | Probes | Maximum component reconstruction error | Maximum permutation-control response gap | Maximum fixed non-permutation designed-probe gap |
|---:|---:|---:|---:|---:|
| 0 | 4 | 5.71e-11 | 9.44e-17 | 0.9734 |
| 1 | 4 | 6.85e-11 | 9.26e-17 | 0.9731 |
| 2 | 4 | 5.06e-11 | 9.42e-17 | 0.9735 |

The common-product premise is supplied symbolically by the explicit invertible
gauge; the roughly `1.7e-16` computed product residuals are only numerical
checks. Under exact common product, full rank, known positive Euclidean block
rates, and the recorded dimension condition, equality on these designed probes
is equivalent to equality of the complete structured response. This is a
model-specific matrix-probing corollary, not a claim about arbitrary finite
probes, approximate products, passive task gradients, AdamW state, or
historical structure. Each full-dimensional response is evaluated with the
analytic response-block formula; that implementation was independently checked
against autograd and finite differences on the declared `4 x 32` extraction,
not by four full-model JVP or autograd measurements. The proof and prior-art
classification are in `research/RESPONSE_TOMOGRAPHY.md`.

## 2. ASLoRA coordinate-to-action chain

The audit is a documented reimplementation of the paper-defined
RoBERTa-base/MRPC settings, not an official reproduction. It freezes each seed at step 560
immediately before the first merge. The raw ASLoRA rule uses Euclidean distance
between cumulative running-average `B` factors. Ten deterministic gauges are
tested per seed: identity, an orthogonal control, and diagonal/dense transforms
with condition limits 2, 4, 8, and 30.

Appendix Table 5 of the audited arXiv version lists an update ratio
`lambda=0.5`, but its method text does not define an associated operation. The
reimplementation records it as `update_ratio_record_only=0.5` and does not
silently invent a rule. This unresolved method detail is another reason the
result is labeled a documented reimplementation rather than an official
reproduction.

The protocol amendment, superseded 12/30 and 16/30 pilot chronology,
source hashes, and deterministic rerun lock are recorded in
`research/ASLORA_EMPIRICAL_PROTOCOL.md`; the old numbers are not pooled with or
substituted for the canonical rerun.

The primary scheduling interpretation chooses query and value actions
separately (`per_projection`). Changes are relative to the identity-gauge
raw-`B` action.

| Candidate interpretation | Changed actions by seed | Changed / 30 gauge instances | Seeds with a change | Identity-relative validation-loss range | Accuracy range | F1 range |
|---|---:|---:|---:|---:|---:|---:|
| Adjacent pairs | 2, 6, 3 | 11/30 | 3/3 | [-0.053094, +0.008863] | [-0.002451, +0.009804] | [-0.002246, +0.005659] |
| All pairs | 4, 7, 3 | 14/30 | 3/3 | [-0.053094, +0.002724] | [-0.002451, +0.009804] | [-0.002246, +0.005659] |

For every seed and both primary candidate interpretations, at least one
gauge-equivalent running-factor representation selects a different action and
that action has a different validation outcome from the identity-gauge action.
The joint-scope sensitivity analysis is not uniform: adjacent mode changes in
5/30 instances in only 1/3 seeds, whereas all-pairs mode changes in 0/30
instances. This negative sensitivity result is retained rather than selecting
only the scheduling interpretation that changes.

The scientific equivalence path is deliberately indirect and exact at the
claimed level:

1. transform the frozen running factors in float64;
2. check product and declared-distance invariants and select only a layer-index
   action;
3. restore the canonical model exactly; and
4. execute and evaluate that action as a temporary tie in the unchanged model
   on all 408 MRPC validation examples.

Candidate coverage is complete for the declared families (204, 208, and 200
distinct actions in seeds 0, 1, and 2). Generation-time source hashes match the
released ASLoRA implementation. The local model directory and both MRPC
parquets are byte-inventoried, and each seed records a full initial attached
model-state digest. All canonical equivalence/restoration gates pass, and two
complete deterministic passes over every selected canonical action agree exactly. The
optional live factorized float32 re-execution diagnostic also passes in all
three seeds. It remains a finite-precision diagnostic rather than the
mathematical-equivalence gate.

Two invariant baselines are included: current effective-update Frobenius
distance (prevalidation) and current lower-layer local activation RMS
disturbance (unlabeled validation inputs, post hoc). The latter is a local
output disturbance, not network loss. A labeled validation oracle is also post
hoc; it enumerates all query-only, value-only, and joint-same-pair actions, not
the full query-pair-by-value-pair Cartesian space.

For the six seed-by-candidate-reading rows, the canonical validation-loss
difference from the identity raw-`B` action is negative/positive/exactly zero in
`2/2/2` rows for the effective-update rule and `2/1/3` rows for the local-
activation rule. The two negative local-activation differences occur in the
all-pairs reading for seeds 0 and 1; the adjacent reading has one positive
difference and two ties, while seed 2 is also a tie under all pairs. Thus
neither invariant rule uniformly improves across all seeds under both candidate
readings. Exact actions and loss differences are in
`paper/generated/aslora_baseline_table.tex`.

**Boundary.** The 10 gauges are a fixed finite sensitivity bank. Counts such as
11/30 and 14/30 are not estimates of how often arbitrary gauges, training runs,
datasets, or published systems fail. The audit establishes a concrete
same-effective-update -> different coordinate action -> different measured
outcome chain for this reimplementation. It does not claim the official
checkpoint would make the same choices, that the identity-gauge action is
correct, or that a historical graph is known.

## 3. Controlled byte-LM same-budget partition decision

The frozen, predeclared search uses 5,000 positive-stochastic router-only trials per
checkpoint, condition number at most 30, first-valid selection, and exactly
four nonempty groups. Both partitions are refit from the same effective tensors
before evaluation. The evaluation uses 256 fixed disjoint batches of shape
`8 x 128` and a 10,000-resample paired bootstrap.

All three routers are initialization-dominated under the protocol threshold:
their mean row-wise L1 movements round to `0.000313`--`0.000401`, well
below `0.01`. Therefore this audit is a controlled trained-model experiment,
not evidence about a training-selected router in a functioning LM.

| Seed | Search result | Soft test BPB | Router mean L1 movement | Partition changes / group-pass | Completed partition-only consequence |
|---:|---|---:|---:|---:|---|
| 0 | found at trial 500 | 2.1859 | 0.000335 | 1/3762 | fold BPB 2.1753 -> 4.8149; delta +2.6396 [2.6287, 2.6510] |
| 1 | none in 5,000 trials | 2.1878 | 0.000313 | 0/3790 | not evaluated |
| 2 | none in 5,000 trials | 2.1684 | 0.000401 | 0/3769 | not evaluated |

For seed 0 the selected transform has condition number 7.925 and changes group
sizes from `3/3/3/3` to `5/3/3/1`; normalized partition disagreement is 0.1212.
The reference and candidate refits have squared errors `5.47e-5` and `1868.09`,
respectively. The literal gauged-router hardening comparison is secondary
because it changes partition and basis coordinates jointly.

For the completed seed, the two gauge-invariant same-budget baselines are:

| Baseline | BPB | centroid-fit SSE |
|---|---:|---:|
| seeded k-means on common effective tensors | 2.175314515890271 | 0.00005468961745421999 |
| deterministic Ward on common effective tensors | 2.175314515890271 | 0.00005468961745421999 |

Seeds 1 and 2 exit after the failed finite search and store no fold, k-means,
or Ward evaluation; the missing entries are not baseline failures. The seed-0
candidate also changes the sorted group-size multiset from `3/3/3/3` to
`5/3/3/1`, so the effect does not isolate membership from partition balance.
The `signed_local` and same-sorted-group-size searches were not executed; they
are post-run non-primary protocol amendments and have no released outcomes.

The predeclared success fraction is 1/3. Consequently the aggregate records
`primary_claim_supported = false` and
`replicated_measurable_consequence = false`. This result supports a controlled
existence witness at one trained checkpoint, not a learned-router claim,
prevalence, or a stable replicated byte-LM effect.

## Legacy diagnostic material

The older canonical inputs remain in `results/`:

- 700 planted exact-factorization runs;
- 120 teacher/student distillation runs and 40 task-free references;
- 3,360 paired functional-probe records;
- 20 random gauge trials and 120 saturation runs; and
- 12 older byte-LM router-stability runs.

Together these are the 4,372 optimization/evaluation records validated by
`tests/test_canonical_results.py`; the analytic gauge object is counted
separately. They are useful for implementation checks and for showing that
loss/ARI or initialization effects can occur in controlled examples. They do
not show that published systems commonly infer a true historical graph, and
they do not complete a coordinate-to-decision consequence chain.

ShareProbe is specifically demoted. The revised 3,360-record sweep uses a
strict common-random-number coupling: counterfactual and native signatures see
the same initial probes, optional random projection, and realization of
`Y_i(H_s) = r_i(H_s) + sigma * epsilon_i_s`, with iid standard Gaussian
`epsilon` and an absolute (not batch-standard-deviation-scaled) `sigma`.
Native hidden states advance with clean updates only. Across the full fixed
grid, counterfactual versus native mean ARI is `0.6764` versus `0.6751` for the
MLP (paired mean difference `+0.00129`; 65/1,570/45 win/tie/loss cells) and
`0.5006` versus `0.5009` for the Transformer (difference `-0.00026`;
61/1,558/61). The seed-aggregated difference changes sign: six of ten MLP
seeds and three of ten Transformer seeds favor counterfactual signatures.

The added visible-effective-parameter-distance baseline obtains ARI `1.000`
in every planted MLP and Transformer cell, while counterfactual signatures
never beat it (they tie in 1,030/1,680 MLP and 745/1,680 Transformer cells).
This baseline is valid only when effective block tensors can be read directly;
parameter distance is not a test of functional equivalence because neural
parameter symmetries and distribution-restricted behavior remain. Even with
that boundary, the comparison shows no stable advantage or necessity for
ShareProbe in this benchmark. The saved empirical gap is computed against the
planted partition, but it is a finite-sample descriptive quantity rather than
a population certificate and says nothing outside the probe distribution.

## What may and may not be concluded

The combined artifact supports:

- for a nontrivial basis bank (`K >= 2`), static router coordinates can be
  ambiguous under the already-known SSMF gauge;
- the pair comprising a common exact effective model and its full Euclidean
  effective-gradient response is a strictly richer observation and, under the
  theorem's positivity/full-rank assumptions with the same known positive
  router/basis rates and temperature, fixes the current factorization up to
  basis labels;
- at the released byte-LM dimensions, four explicitly designed interventions
  reconstruct that complete structured response under the exact stated
  assumptions;
- one documented real-model reimplementation contains coordinate-derived merge
  decisions that vary across equivalent running-factor coordinates and have
  measurable canonical consequences; and
- one of three trained, initialization-dominated byte-LM checkpoints contains
  a consequential same-group-count partition witness under the fixed finite
  search.

It does **not** support:

- novelty of the static `Theta = AB` non-identifiability construction;
- a widespread-community-misinterpretation or prevalence claim;
- recovery of a historical/generative sharing graph from current coordinates;
- equivalence between optimizer stability, gauge fixing, current-coordinate
  identifiability, functional equivalence, and action reliability;
- extension from linear weight mixing to architectures that separately execute
  nonlinear experts and mix their outputs;
- a certificate from arbitrary finite probes, approximate products, or models
  outside the stated structured-response and rank conditions;
- replicated byte-LM decision harm across seeds; or
- superiority of ShareProbe over simple native signatures.
