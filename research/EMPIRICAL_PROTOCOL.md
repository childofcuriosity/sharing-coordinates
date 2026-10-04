# Frozen primary WikiText Router-Decision Audit and Amendment Ledger

## Claim being tested

The experiment tests a deliberately narrow claim: two exactly equivalent
soft-routed byte language models can induce different router partitions, and
using those partitions for the same predeclared, partition-only folding rule
can produce measurably different predictive performance.  It does **not** test
whether either router recovers a historical or ground-truth sharing graph, and
it does not estimate how often published systems or practitioners make this
decision.  A positive result establishes a representation-dependent decision
risk for this architecture and decision rule, not a community-wide prevalence
claim.

Equivalence is asserted only at the registered inference temperature (1.0 in
the primary runs).  Re-encoding `A M` as softmax logits need not preserve the
same gauge relation at other temperatures, during an annealing schedule, or
after either parameterization is retrained.

Literal argmax hardening of the original and inverse-gauged bases is secondary
only.  That comparison changes the router partition and the basis coordinates
simultaneously.  It therefore cannot isolate the consequence of choosing a
partition and is explicitly labeled
`secondary_coordinate_and_partition_confounding` in the JSON output.

## Checkpoints and exclusion rules

Train exactly three otherwise identical soft-routed byte-LM checkpoints with
training seeds 0, 1, and 2 for **1,500 optimizer steps**, with 500 linear-warmup
steps.  Use the final step-1,500 checkpoint; do not select a step, seed, or
checkpoint after seeing the decision audit.  Run every audit on the WikiText
test split whose SHA256 is stored in that checkpoint.

Protocol freeze: the 1,500-step budget was fixed on 2026-08-22 before any
router-decision result was inspected.  It matches the already established
WikiText sweep budget in `results/language_full.json` and the three blindly
completed checkpoint runs.  The earlier value 10,000 appeared only as a
software default and was not supported by a separate convergence analysis or
an empirical decision result; it is therefore not the primary budget.  A
future 10,000-step run must be registered as an extension and cannot replace
the three primary 1,500-step seeds.

Before structural analysis, report for every seed:

- final validation and test BPB;
- router mean row-wise L1 movement from initialization;
- final router probabilities and argmax partition;
- checkpoint-file SHA256, complete saved `LanguageConfig`, training metadata,
  corpus SHA256, PyTorch/CUDA versions, device, and hashes of the decision code.

A checkpoint is *weak-model flagged* if its test BPB does not improve by at
least 0.10 BPB over the add-one-smoothed byte-frequency unigram baseline fit on
the training split.  It is *initialization-dominated flagged* if mean row-wise router L1
movement is below 0.01.  Flagged seeds remain in all tables and cannot be
silently replaced.  If two or more seeds receive either flag, the experiment
cannot support a claim about a training-selected router in a functioning LM;
it remains only a controlled trained-model demonstration.  The released result
has all three routers below this movement threshold, so this failure
interpretation applies.

## Gauge generation and selection

The primary family is `positive_stochastic`.  Each candidate is

\[
M=(1-s)I+sP,\qquad s\sim U[0.02,1],
\]

where every entry of `P` is strictly positive and each row sums to one.  The
audit requires `cond(M) <= 30`, `min(A M) > 1e-8`, a nonempty transformed
partition with the same number of groups as the original partition, and at
least one changed pairwise co-membership edge.  For training seed `s`, fix
`search_seed = 620000 + s`, use 5,000 attempted candidates, and select the
first valid candidate.  Search sees router probabilities and partitions only;
it cannot see language loss, logits, effective weights, or evaluation batches.
Search failure is an observed failure and is never replaced by a new seed or a
larger post hoc trial budget.

The output must retain counts after every filter: `attempted`, `invertible`,
`condition_pass`, `simplex_pass`, `group_pass`, and `partition_change`, plus the
first-valid and selected candidate.  The fraction
`partition_change / group_pass` describes only this predeclared random gauge
family; it is not a population prevalence estimate for trained systems or
downstream harms.

**Post-run amendment, not executed in the released audit.**  A future
non-primary sensitivity may run `signed_local` with the same trial budget and
seed.  It permits negative entries in `M` but still requires row-sum
preservation, invertibility, the condition bound, and strict positivity of the
realized router `A M`.  The canonical raw records contain only
`positive_stochastic`; they contain no `signed_local`, `mixed`, or
`max_disagreement` result.  This paragraph was added as a protocol amendment
after the released primary runs and must not be described as a completed or
predeclared sensitivity analysis.  Any future result requires a separately
versioned artifact and cannot replace the primary records.

## Exact-equivalence gate

Disable CUDA matmul and cuDNN TF32 during each audit and restore their prior
process-global settings afterward.  Before making or evaluating any hard
decision, verify the soft original and gauged models on the fixed evaluation
batches.  A run is invalid and raises an error if any criterion fails:

| Quantity | Maximum |
|---|---:|
| relative L2 error over all effective `BasisLinear` weights and biases | `1e-5` |
| maximum absolute logit error | `1e-4` |
| maximum absolute paired-batch mean-NLL error | `1e-5` |
| absolute aggregate BPB error | `1e-5` |

Also require `max |M 1 - 1| <= 5e-6`, finite condition number, and
`min(A M) > 1e-8`.  These are numerical acceptance tolerances, not statements
that approximate equality in general implies identical finite-data behavior.

## Evaluation batches and decision budget

For training seed `s`, fix `batch_seed = 610000 + s`.  Use 256 batches of 8
sequences of length 128 from the test byte stream.  Sample blocks without
replacement from disjoint intervals of length 129, so no input/target byte is
reused across selected blocks.  Every model and decision for a seed receives
the identical stored starts.  Report per-batch mean NLL and BPB as well as the
token-weighted aggregate.

Let `C` be the number of nonempty groups in the original router argmax
partition.  Every compared folding has exactly `C` nonempty groups.  Report
the ordered group-size vectors; equal group count controls the number of stored
centroids but not partition balance.  As a stricter sensitivity analysis,
one could require a partition-changing candidate with the same sorted
group-size multiset.  **That stricter search was not executed in the released
audit and is a post-run, non-primary protocol amendment.**  The successful
primary candidate changes sizes from `3/3/3/3` to `5/3/3/1`; consequently the
released effect does not isolate membership changes from partition balance.
Any future same-size result must be reported separately and cannot replace the
primary candidate.

The materialized audit model retains unused tensor slots so that evaluation is
simple and paired.  The decision represents hard tying/folding to `C` stored
centroids; it does not by itself measure serialized checkpoint size, inference
latency, or memory savings.  Those quantities must not be claimed without a
separately materialized compressed implementation.

## Primary and baseline decisions

First compute the single common collection of effective tensors

\[
\Theta_i^{(m)}=\sum_k A_{ik}B_k^{(m)}
\]

from the original soft model, for every `BasisLinear` module and its optional
bias.  For each router partition, replace every layer in a group by the
arithmetic centroid of these same effective tensors, then materialize that
folded model with one-hot routing.  Thus the only input that differs between
the two primary folds is the partition.

The predeclared seed-level effect is

\[
\Delta_s = \operatorname{BPB}(\text{fold from gauged-router partition})
          - \operatorname{BPB}(\text{fold from original-router partition}).
\]

Report signed and absolute `Delta_s`, paired batch deltas, both centroid-fit
SSEs, normalized pairwise partition disagreement, group sizes, and condition
number.  Neither sign is privileged: the reliability concern is that an
arbitrary equivalent coordinate choice changes the quality of the same
structural decision.

Two gauge-invariant same-budget baselines operate directly on the common
effective tensors:

1. seeded multi-restart k-means (`kmeans_seed = 630000 + s`); and
2. deterministic Ward agglomeration with merge cost
   `|S||T|/(|S|+|T|) * ||mean(S)-mean(T)||^2`.

Both algorithms must return exactly `C` nonempty clusters, and their reported
objective must equal the actual centroid-fold SSE.  These baselines test
whether reading router coordinates is needed for the folding decision; they do
not claim recovery of a true graph.

## Uncertainty and success criteria

Within each seed, form the 256 paired batch differences in BPB and compute a
10,000-resample paired percentile bootstrap 95% interval using bootstrap seed
`640000 + s`.  This interval is descriptive for the fixed disjoint test blocks;
it is not obtained by treating individual tokens as independent.  Across the
three training seeds, report all three effects, their mean, standard deviation,
and the small-sample t interval

\[
\bar\Delta \pm 4.303\,s_\Delta/\sqrt{3}.
\]

Do not pool thousands of batch observations across checkpoints and present
them as independent training replicates.

Call the downstream consequence *replicated and measurable* only if at least
two of three seeds satisfy both `|Delta_s| >= 0.01 BPB` and a within-seed paired
95% interval excluding zero.  Otherwise report the exact estimates and narrow
the conclusion to representation-dependent partitions without established
performance consequence.  With three seeds, even the successful outcome is
evidence for this system, not a precise effect-size or prevalence estimate.

The following are predeclared failure interpretations:

- any failed equivalence or legality gate invalidates that run;
- no valid candidate within 5,000 trials is a search failure, not a seed to
  replace;
- a literal-hardening difference without a common-effective folding difference
  is a basis-coordinate/hardening artifact;
- an effect found only by `max_disagreement` but not by first-valid selection is
  adversarial existence evidence, not typicality evidence;
- two or more weak-model or initialization-dominated flags preclude a real
  learned-router-system claim;
- if neither router partition performs materially differently from the Ward or
  k-means baselines, do not claim that the invariant baselines are superior;
- if ShareProbe or another probe is later compared, it must beat these simple
  invariant baselines under the same checkpoints, partitions, budgets, and
  paired batches before being presented as necessary or advantageous.

All seeds, candidate counts, assignments, per-batch outcomes, invalid runs, and
flags are retained.  No checkpoint, gauge, evaluation batch, baseline restart,
or reporting threshold may be changed after inspecting primary BPB differences.

## Released execution status

Only the primary `positive_stochastic` first-valid search was executed, for
all three training seeds.  Seed 0 reached the decision stage; its stored
gauge-invariant same-budget baselines are:

| Baseline | BPB | centroid-fit SSE |
|---|---:|---:|
| seeded k-means on common effective tensors | 2.175314515890271 | 0.00005468961745421999 |
| deterministic Ward on common effective tensors | 2.175314515890271 | 0.00005468961745421999 |

Seeds 1 and 2 exhausted the finite search before the decision stage.  Their raw
records therefore store no partition fold, k-means, or Ward evaluation.  This
absence is not a failed baseline comparison.  Neither the `signed_local`
search nor the same-sorted-group-size search was executed; both are unexecuted
non-primary amendments with no numerical outcome in this release.
