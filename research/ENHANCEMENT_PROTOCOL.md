# Prospective enhancement protocol — 2026-09-25

This additive extension does not replace any historical raw record. The user
authorized implementation, local experiments and manuscript preparation. The
ICLR 2027 template and author placeholders remain unchanged.

## Stage A: implementation/timing pilot

Use seed 90, random (unit-scale) router initialization, L=12, K=4, width=128,
MLP width=512, 8 heads, length=128, batch=16. Run 100 steps first on training
data only, with validation evaluation; no test evaluation or gauge-effect
selection. AdamW, base lr=3e-4, router lr=3e-3, weight decay=.1 for weights
and zero for router, no router regularizer, temperature=1, BF16 autocast.
The separate router rate addresses the previously observed near-frozen router;
it changes the experimental system and is not a continuation of old seeds.
The pilot determines feasibility and numerical correctness only. Any amended
configuration must be recorded before primary seeds run.

## Stage B: prospective trained-model extension

Fresh seeds 10,11,12, same architecture and optimizer as Stage A, fixed 6000
steps, 300-step warmup and cosine decay to .1 of starting rates. Save final
step only for the primary audit, log validation at steps 1,1000,...,6000.
Report router mean row L1 movement; <.01 is initialization-dominated. Report
validation trajectory and a byte-unigram test comparator; improvement <.10
BPB is weak-model flagged. Fixed budget is not a claim of convergence. Failed
or flagged seeds remain in results; never replace them based on outcome.

For each final model search two separately reported router-only families:
positive-stochastic and signed-local, 5000 candidates each, seeds 720000+s
and 730000+s. Use the existing family distributions, cond(M)<=30 and
min(A M)>1e-8. Require a changed co-membership partition with **identical sorted
nonempty group sizes** to the native argmax partition. Select the first legal
candidate without viewing losses. Retain all filter counts and failures.

Fold each selected partition from the same native effective tensors. Include
a deterministic effective-parameter baseline constrained to the same group
sizes. It minimizes centroid SSE over a fixed local-search procedure and is
not a globally optimal partition claim. Unconstrained Ward is descriptive only.
Freeze hard routes. Evaluate every action immediately and after 100 and 500
AdamW updates on identical training batches, same initialization of shared
nonbasis parameters, fresh optimizer state, no test-based selection. Use
lr=1e-4, weight decay=.1, gradient clipping=1, BF16 training and FP32 evaluation.
Stored inactive slots are not claimed as realized compression speedups.

Test uses 64 fixed disjoint batches of 8x128 tokens, seed 740000+s. Report all
per-batch outcomes and checkpoint hashes, signed and absolute gauge/native
BPB differences. Bootstrap intervals are conditional on the selected blocks;
training seeds, not probes/batches/gauges, are the independent replicates.
With three seeds report all seed outcomes and descriptive ranges; no population
prevalence or general algorithm-superiority claim. A search failure is missing
downstream evidence, not a zero effect. Router maturity and equal-size gates
are necessary for the strengthened learned-router interpretation.

## Stage C: task gradients and finite arithmetic

Start with all three original checkpoints (seeds 0,1,2), then assess new
checkpoints separately if useful. Compute effective-weight gradients through
the actual byte-LM cross-entropy with all nonbasis parameters held fixed.
Use 16 training minibatches of 2x32 tokens, seed 750000+s; never optimize the
input batches for separation. Compare native, common permutation and fixed
M=.65I+.35(11^T/K). Check effective-forward and chain-rule gradients first.
Measure Euclidean packed-map responses for all directions, their observed
span/singular values, separation from the alternative chart, and held-out
response prediction for q=1,4,8,12 training directions with four held-out
directions. Restrict conclusions to the observed span: finite natural gradients
do not establish factor identification or exhaustive gauge separation.

Compare explicit finite Euclidean steps h in {1e-1,1e-2,1e-3,1e-4,1e-5}
with the infinitesimal response, in FP64/FP32/BF16 on the same first task
gradient. Report relative parameter-step error, not optimizer equivalence.
These are block SGD experiments with known rates, not Adam trajectory results.

## Stage D: joint-noise feasibility

Use small synthetic full-rank interior factors, then systematically approach
rank/boundary degeneracy. Perturb both Theta and measured responses; recompute
the estimated row space from noisy Theta. Record failures and recovery errors
up to permutation, PSD clipping and eigengaps. A finite sweep is diagnostic,
not a new theorem. Only add a theorem after checking a complete proof.

## Review, attribution and release

Apply scientific-critical-thinking to confounding/independent units,
paper-review to claim/evidence/prior-art boundaries, ml-paper-writing to the
main narrative. Scientific Agent Skills: Timothy Kassis, Vinayak Agarwal,
Yuhuan He, Darshil Patel, Aubrey M. Brueckner (2026),
https://doi.org/10.48550/arXiv.2609.00065. Current arXiv record checked
2026-09-25 (v2, no journal reference); cite the unversioned DOI.

New code, source/protocol hashes, raw records and deterministic summaries must
be preserved. Run focused correctness tests before expensive experiments,
then clean-environment reproduction and release checks. Update the paper with
positive and negative findings; no public upload is authorized by this plan.
