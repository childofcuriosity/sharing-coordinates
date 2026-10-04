# ASLoRA first-merge audit protocol and amendment log

## Target and source boundary

The experiment is a documented reimplementation of the first merge described in
the publicly available `arXiv:2412.10135v2`, not an official reproduction.
No author code or checkpoint was located.  The available text is internally
ambiguous about all-pair versus adjacent-pair candidates and about whether
query and value share one decision, so both candidate sets are reported and
the per-projection interpretation supported by Figure 3 is primary.

The paper-configured run uses RoBERTa-base on MRPC, WQ/WV LoRA, rank 8,
alpha 16, learning rate `4e-4`, batch 16, weight decay 0.1, linear schedule,
warmup ratio 0.06, maximum length 512, `Ts=320`, and merge interval 240.  With
one-indexed updates and the strict condition in Algorithm 1, the first merge is
at optimizer step 560.  The reimplementation samples the current `B` after
each optimizer update, updates its cumulative average, stops immediately
before the merge, and saves both a model checkpoint and factor snapshot.

## Frozen counterfactual bank

Before any validation forward pass, independently for query and value, form
identity and orthogonal condition-one controls and deterministic diagonal and
dense common gauges with condition-number limits 2, 4, 8, and 30.  The bank is
a deterministic function only of rank and training seed.  For each gauge,
transform the frozen current and running factors in float64 and select the
paper's raw-running-`B` nearest action under both candidate interpretations.
No validation value, label, loss, logit, activation, or action consequence is
available to gauge generation or action selection.

Every selected directed layer-index action is then evaluated in the same,
unchanged canonical checkpoint by temporarily making the lower layer use the
upper layer's *current* `B`.  The parameter object and value are restored after
each complete MRPC validation pass.  This construction isolates

`same algebraic parent -> different coordinate-selected action -> different
canonical action consequence`.

## Estimands and baselines

The following are selection proxies, not current action costs:

- raw distance between running-average `B` factors (the audited rule);
- invariant distance between their running-average effective updates;
- a pooled-activation version of the running-average discrepancy.

The current directed action has two separate diagnostics:

- current effective-update Frobenius change;
- exact empirical RMS change in the lower layer's scaled LoRA output, using
  the current `B`, that lower layer's own non-padding-token second moment of
  `A x`, and scale `alpha/rank`.

The decisive network consequence is the full-validation loss, accuracy, and
F1 after the temporary canonical tie.  Query-only, value-only, and
joint-same-pair candidates are exhaustively evaluated.  Every composite action
actually selected by a declared gauge or baseline is also evaluated.  A
full `66 x 66` query/value Cartesian validation oracle is not claimed.

## Equivalence gate

The main gate requires, for every declared gauge:

1. frozen-snapshot effective products agree in float64 within the declared
   absolute/relative tolerance;
2. all theoretically invariant distance diagnostics agree after transforming
   their low-rank moments;
3. the canonical model's tensor values and parameter-object identities are
   restored exactly after every temporary action and numerical stress test.

Live float32 execution of the reparameterized factorized network is retained
as a separate numerical stress test.  It does not gate the canonical action
comparison and is never used to measure a selected action's consequence.
Finite-precision factor association can produce logit drift even when the
algebraic products agree; reporting that drift is useful, but treating it as
the structural action effect would be a confound.

## Amendment chronology

This release is the second complete three-seed run.  We record the superseded
run because silently replacing it would create a selective-reporting risk.

1. **Initial numerical-gate repair.**  The first seed-0 implementation made
   every live float32 re-execution check a run-level gate and raised only a
   generic error.  It finished training/candidate evaluation but returned no
   audit result.  Before using canonical action metrics, the runner was
   amended to persist diagnostics.  Even an identity `torch.linalg.solve(I,A)`
   path could move float32 logits although the frozen float64 products passed.
   We therefore made frozen-product/distance algebra plus exact canonical
   restoration the scientific gate and retained live factorized execution as
   a non-gating diagnostic.

2. **Superseded nondeterministic batch.**  A complete three-seed batch then
   used source hashes
   `796d4743b02cc14ecf37ac7cde93916720c92e1c7df74c3162aa5ac881ac2305`
   (runner),
   `c4a8fd1e0d74794de8b8053c1ac3f32742db378e78c4e749878932727f2eca07`
   (training), and
   `d57734071e18ab0f4384772a048faa0264813411069ceabe4fca82906d66bc64`
   (witness).  It reported 12/30 adjacent and 16/30 all-pairs primary action
   changes.  It did not force deterministic CUDA algorithms or repeat each
   selected canonical action.  A later final-artifact audit found small
   repeated-evaluation drift even for unchanged/identity paths, and the
   released source had necessarily changed when a deterministic repeat gate
   was added.  Those facts invalidate a claim that this batch was generated by
   the final protocol, irrespective of whether its headline counts were
   larger or smaller.

3. **Protocol lock before the replacement outcomes.**  Before running the
   replacement seeds, the runner was fixed to set
   `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enable PyTorch deterministic algorithms,
   disable cuDNN benchmarking and TF32, and require two exactly equal complete
   validation passes for the unmerged baseline and every distinct raw-
   running-`B`-selected canonical action.  It was also fixed to hash the three
   generation source files, both MRPC parquet files, every file in the local
   RoBERTa directory, and the complete initial attached model state.  The
   locked source hashes were
   `ab09259c092a9a3fab49e5aceb06aad7f56eb20b923b6edb3965bd8c940576d9`
   (runner),
   `69818d7c8eb7d20ec3c123f49b0c42ec3ffeb66b835bb79429083bcb4a646664`
   (training), and the unchanged witness hash above.  The locked seed-0 run was
   the fail-closed pilot for the exact repeat gate; because it passed without a
   protocol change, it is retained as canonical seed 0.  Seeds 1 and 2 were
   then run sequentially under the identical locked protocol.  No gauge, seed,
   candidate set, model hyperparameter, action rule, or validation metric was
   added after seeing a replacement outcome.

4. **Canonical replacement.**  All three deterministic runs passed exact
   repetition, source/input binding, candidate coverage, product/distance
   algebra, and restoration.  The replacement counts are 11/30 adjacent and
   14/30 all-pairs, with a changed consequential primary action in all three
   seeds under both candidate interpretations.  Live factorized float32
   re-execution also passes in this batch but remains diagnostic.  Only after
   the strict three-seed summarizer accepted the replacement were the old raw
   directories and duplicate summaries removed from the release.  Their old
   headline counts and generation hashes remain recorded here.

The deterministic gauge bank, training seeds, model/data configuration,
candidate families, selected layer-index rule, canonical action
materialization, and reported validation metrics were unchanged across the
two complete batches.  The replacement is a validity-driven protocol
amendment and lowers, rather than strengthens, the old action-change counts.

## Claim boundary

A positive result concerns this reimplementation, checkpoint family, and
first-merge decision.  It does not estimate prevalence, reproduce the full
ASLoRA merge tree, show that ASLoRA claimed a historical generator, or prove
that an invariant proxy is universally optimal.  A negative result is also
retained; it narrows the claim rather than authorizing a new gauge or seed.
