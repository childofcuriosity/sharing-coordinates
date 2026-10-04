# Independent byte-LM decision-artifact audit

Audited: 2026-08-23.  This audit regenerated the decision records from the
three released step-1500 checkpoints and the released WikiText-2 byte streams;
it did not use `results/router_decisions_summary.json` to determine the search
outcome.

## Exact released protocol

For each seed, the CPU regeneration used the recorded positive-stochastic
search family, 5,000 trials, condition-number limit 30, first-valid selection,
four nonempty groups, and seeds `610000 + seed`, `620000 + seed`, and
`630000 + seed` for batches, gauge search, and k-means respectively.  The
evaluation used 256 nonoverlapping batches of shape `8 x 128` from the released
test bytes.  Both candidate partitions were refit from the same effective
tensors.

The regenerated configuration objects and all 256 recorded batch-start arrays
are exactly equal to the canonical records.  For every seed, the search funnel
is also exactly equal:

| seed | attempted | simplex/condition pass | four-group pass | partition-changing | status |
|---:|---:|---:|---:|---:|---|
| 0 | 5,000 | 4,586 | 3,762 | 1 | found at trial 500 |
| 1 | 5,000 | 4,601 | 3,790 | 0 | exhausted |
| 2 | 5,000 | 4,581 | 3,769 | 0 | exhausted |

## Cross-device consequence check

Only seed 0 has a completed partition-fold consequence.  Its released GPU and
independently regenerated CPU values are:

| quantity | released GPU | CPU regeneration | absolute difference |
|---|---:|---:|---:|
| reference-fold BPB | 2.175314515890271 | 2.175314530670030 | 1.48e-8 |
| candidate-fold BPB | 4.814934889850295 | 4.814934825356801 | 6.45e-8 |
| candidate minus reference | 2.639620373960024 | 2.639620294686770 | 7.93e-8 |

The largest absolute per-batch BPB differences across the 256 batches are
`8.60e-7` for the reference fold and `1.38e-6` for the candidate fold.  These
are ordinary CPU/GPU floating-point differences and are negligible relative
to the reported effect.

The released seed-0 record also stores the two predeclared gauge-invariant
baselines.  Seeded k-means and deterministic Ward both have BPB
`2.175314515890271` and centroid-fit SSE `0.00005468961745421999`.  Seeds 1 and
2 exit before decision evaluation; their raw records contain no fold, k-means,
or Ward result.  The independent audit therefore treats those values as not
stored/evaluated, not as failed baselines.

## Interpretation

This regeneration confirms the finite search and the conditional consequence
for the released checkpoints.  It does not turn two exhausted 5,000-trial
searches into proofs of nonexistence, make the successful checkpoint
representative of a population, or identify either partition as historical
truth.  The canonical GPU JSON remains the manuscript's numerical source
because it is generation-source-attested; this file records an independent
cross-device consistency check.

All three routers satisfy the protocol's initialization-dominated flag.  The
successful candidate changes group sizes from `3/3/3/3` to `5/3/3/1`.
`signed_local` and same-sorted-group-size searches were not executed and are
post-run non-primary protocol amendments, so this audit makes no sensitivity
claim for either family.
