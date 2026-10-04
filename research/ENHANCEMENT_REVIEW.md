# Enhancement review — 2026-09-25

Scoped scientific self-review and local artifact report, not independent peer
review or public submission. FINAL_REVIEW.md and FINAL_VALIDATION.json describe
the preceding delivery, not this extension.

## Claim–evidence assessment

1. **Learned-router equal-size decisions.** Three newly trained final-step
   checkpoints pass router-movement and unigram-improvement gates. All
   positive-stochastic searches find an equal-size partition change; all
   signed-local searches fail. Immediate signed differences are -0.00112,
   +0.87112,+0.14873 BPB, falling to +0.00009,+0.01823,+0.00343 after 500
   paired recovery updates. This removes the old initialization and size
   confounds but does not show universal or persistently large harm. Same-size
   effective-parameter local search matches the native action in two seeds and
   is slightly worse downstream in the third. Do not claim general advantage.
   Validation remains improving: trained and non-frozen is supported;
   converged is not.

2. **Natural directions under known structure.** At three original and three
   new checkpoints, four actual task-gradient directions meet complement/local
   numerical ranks. Worst held-out response error is below 1e-13; worst factor
   error is 3.03e-8. Original forward and factor gradients agree with an
   independent effective-weight implementation. Generic span extrapolation is
   not equivalent to the structural estimator. One direction does not identify
   the local blocks. These are controlled Euclidean JVPs at frozen checkpoints,
   not passive AdamW recovery. Exact product row space and FP64 are substantial
   assumptions; minibatches do not increase independent seed count.

3. **Finite arithmetic.** At h=1e-5, new-seed FP64 errors are 3.26e-8 to
   5.19e-8, FP32 errors .0208 to .0260, and BF16 errors 1.010 to 1.032.
   This is pure-cast factor-step subtraction, not ordinary BF16 autocast
   training with FP32 master weights. Do not claim BF16 training fails.

4. **Joint stability of valid nearby factors.** A projection bridge reduces
   nearby products to the existing same-product theorem. The corollary has an
   explicit smallness condition, enlarged domain and response Lipschitz bound.
   JOINT_STABILITY_PROOF.md checks: projection preserves row sums since 1 is
   in col(A); a right inverse controls projection error; full rank gives the
   exact intermediate product; positivity and singular values survive; the
   direct-sum response bound and quotient triangle inequality close the proof.
   Tests exercise the bridge but do not prove the theorem. This extends the
   existing bound; it is not a certified solver.

5. **Noisy algorithm remains exploratory.** The 450-cell synthetic grid
   perturbs products and responses and re-estimates row space. All runs return
   outputs, but near-rank-loss errors can exceed 60 at relative noise 1e-6.
   Returned factors need not satisfy the corollary's feasible-pair conditions.
   A constrained estimator must report feasibility and fit rather than infer
   success from lack of exceptions.

## Review scope

- scientific-critical-thinking: removed two confounds, separated independent
  seeds from conditional block intervals, retained search/precision failures.
- paper-review: separated identification, algorithm behavior and utility;
  retained simple invariant baselines and prior-art boundaries.
- ml-paper-writing: updated the abstract, theory, experiments, limitations,
  appendix and generated tables; retained ICLR 2027 and author placeholders.
- Scientific Agent Skills is attributed using current arXiv metadata and
  exported BibTeX, without claiming measured effectiveness of the skills.

Unknown-rate/Adam-state identification, historical graphs, optimal query count
and broad downstream utility remain open. Original ASLoRA training inputs and
optimizer checkpoints remain partly unbundled; saved-record auditing is not
complete reexecution. The new byte-LM extension bundles its corpus and final
checkpoints and does not inherit that missing-input issue.

## Completion checks

See ENHANCEMENT_VALIDATION.json for final tests, clean-environment scope,
source bindings, manuscript, archive and visual checks. All new measurements
are additive under results/enhancement; historical raw records are preserved.

The independent CPU replay passes 156 tests and exact table derivation.
Its 450 synthetic cells have matching status/configuration keys, but numerical
factor errors differ across backends. At relative noise 1e-6, maximum rank-
contraction error is 66.85 in the original run and 68.78 in the clean CPU
replay. Noiseless maximum errors remain below 4.3e-7 in both. The latter is a
sensitivity replication, not numerical identity. Re-estimated SVD/probe
geometry and ill-conditioned inversion can affect noise realization and
amplification; no causal decomposition of backend differences was attempted.
