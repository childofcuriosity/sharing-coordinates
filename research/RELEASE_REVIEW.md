# Publication-readiness review — 2026-09-25

**Recommendation: release the current manuscript as a scoped theoretical arXiv
preprint. No remaining scientific blocker was identified in the checks below.**
This is a judgment about the actual conditional results and their presentation,
not a prediction of ICLR acceptance, a claim of external human peer review, or
an assertion that every related paper has been ruled out. Nothing was uploaded.
The ICLR2027 template and deferred author information are retained by request.

The manuscript is **Sharing Coordinates: Identifiability from Optimizer
Response**. Its main text ends on page 7; references and the complete
supplement bring the PDF to 54 pages. Read `paper/main.pdf`; exact release
hashes and checks are recorded in `RELEASE_VALIDATION.json`. Earlier
`FINAL_REVIEW.md` and delivery reports describe historical snapshots and do
not supersede this disposition.

## What is now supported

The central contribution is a complete, model-specific inverse chain. For a
strictly positive, full-column-rank finite softmax router and full-row-rank
shared bases, two factorizations of the same effective product have the same
complete instantaneous response under common known positive Euclidean block
rates and temperature if and only if they differ by a common permutation.
Independent logit row shifts remain. This identifies current coordinates
relative to an optimizer metric, not a historical sharing graph.

The quantitative argument controls both original factors on a declared
interior, rank-conditioned, norm-bounded domain. It is global on that domain,
with explicit conservative constants and examples showing failure of uniform
linear control as each nondegeneracy margin is lost. The small-product-error
extension compares arbitrary domain members; it does not require that they
start in the same local solution branch.

The finite-noisy consequence uses K unit-Frobenius probes designed from the
observed product, when D−K≥L. For every feasible pair, best-rank-K residual
control bounds the part of each original basis outside the observed subspace.
Stable decoding of a shared-support operator family, followed by return to
the unprojected factors, gives a diameter bound for the entire feasible set.
The stated product-noise threshold is required; response noise has no
smallness restriction. No recovery algorithm enters this proof. These are
precise statements with a complete proof, not conclusions inferred from
successful numerical fits.

## Fresh scrutiny and corrections

The following agent checks were performed separately before the final
recommendation. They are automated internal reviews, not independent human
peer review or formal proof verification.

| Review | Evidence and result |
|---|---|
| `RELEASE_MATH_REBUILD.md` | Rebuilt response from the definition, then independently checked exact identification, singular-PSD separation, the common-kernel Sylvester step, global permutation rounding, joint perturbations, finite-noisy quantifiers, and all three degeneration families. No blocking mathematical error found. A second check covered the rewritten main proof sketches. |
| `RELEASE_PRIOR_ART.md` | Reopened the closest primary passages on SSMF, softmax flow/symmetry, joint diagonalization, robust tensor inversion, and structured probing. The combined response-specific result remains cautiously defensible; generic ingredients are prior work. |
| `RELEASE_EDITORIAL_AUDIT.md` | Read the full manuscript and supplement, recorded twelve issues, then checked their closure in the rewritten version. The final recommendation is scientific preprint release, not just typographic acceptance; final source and PDF-text hashes are included. |

Material corrections were made, rather than merely recording concerns:

1. **Corrected a false argmax statement.** The uniformizing path preserves
   coordinate ordering and argmax ties; other feasible nonuniform gauges can
   change assignments. The introduction and both relevant appendices now
   distinguish these facts.
2. **Replaced the three-way audit narrative with the actual theory.** The
   title, abstract, contributions, setup, proof sketches, results overview,
   and conclusion now lead through exact identification, stability, and the
   finite-noisy consequence. ASLoRA and folding are auxiliary decision
   witnesses, with complete results retained in the supplement.
3. **Separated different guarantees.** The small-product-perturbation
   pairwise bound, the whole-feasible-set diameter bound, the candidate-local
   component bound, and empirical acceptance are explicitly different.
4. **Separated theory and measurement protocols.** The existing experiments
   use the exact-product design or natural/Gaussian probes. They do not test
   the repeated-complement unit probes of the new theorem or its global
   diameter bound. Both the abstract and numerical overview state this.
5. **Corrected prior-art positioning.** Bhaskara et al. already provide
   linear robust inversion at fixed conditioning parameters. Neither the
   exponent nor global robust tensor uniqueness is claimed as an invention.
   The contribution is the stable response-to-diagonal reduction, its control
   of original factors, and the noisy-product measurement bridge.
6. **Reconciled old and new protocols.** The initial, initialization-dominated
   folding protocol is marked historical. The later learned-router,
   equal-size, matched-recovery experiment has its own protocol and results.
   Search failures and mixed-sign consequences remain visible.
7. **Tightened minor scope statements.** Gram insufficiency is qualified for
   K≥3; a fixed-checkpoint response map is distinguished from a passive
   cross-time trajectory rather than assigned an unconditional information
   ordering. Query counts explicitly refer to full matrix-valued outputs.
8. **Fixed publication consistency.** Removed a dangling cross-reference,
   restored table order, and removed old central/primary language for moved
   auxiliary material. The disclosure states actual AI involvement and human
   author responsibility without claiming completed external review or
   inventing an additional approval requirement.

The final mathematical text adds the explicit shared-term subtraction noted
by the math reviewer. Other post-review source differences are the scope and
cross-reference corrections above. The final editorial recheck and release
validation bind the resulting text, rather than pretending that an earlier
PDF hash covers later edits.

## Numerical evidence and remaining limits

Full-dimensional autodiff measurements at three checkpoints agree with the
response formula and predict unseen responses. Four natural task-gradient
directions meet the sufficient ranks at six checkpoints and yield FP64 factor
errors below 3.1e−8. These are controlled fixed-checkpoint measurements with
the known response family; finite-step FP32/BF16 failures are retained.

The frozen noisy-recovery study reports both improvements and failures.
Across 1,920 synthetic variants, joint fitting reduces the median factor
error to about 1.01e−4 but still has 227 errors above 0.10 and a maximum error
of 10.54. The combined diagnostic accepts 92 bad outputs among 1,782 accepted
outputs (5.16%; seed-cluster interval 4.50–5.79%). Residual-only rejection has
68 bad among 1,757 (3.87%). The combined rule has no demonstrated superiority,
and its interval includes the 5% calibration target. All 48 language-model
joint fits are good on the stated grid, which leaves bad-case rejection
recall unmeasured there. Local numerical passes do not rule out distant
feasible solutions or certify truth membership in a fitted subspace.

The following limitations are part of the published claim, not purportedly
resolved extensions:

- Domain margins and noise budgets are assumptions, not quantities certified
  by a successful fit. Constants can be very large; no practical sharpness,
  optimal query count, minimax rate, or efficient global solver is proved.
- The law is controlled Euclidean response with internal matrix access.
  Unknown AdamW state, arbitrary penalties, passive training histories, and
  nonlinear output mixtures are outside scope.
- Numerical experiments do not establish the new global theorem. Its proof
  is the evidence for the universal statement; no new training was needed
  merely to decorate that result.
- The ASLoRA evidence concerns a documented first-action reimplementation.
  Official code and some original inputs/checkpoints are not bundled; no
  full-training regret or official-method reproduction is asserted.
- Two citation checks retain limited technical access, including the
  inaccessible Karuturi workshop full text. The paper makes no first/only
  priority claim. The 51 cited keys are structurally complete; source-access
  and passage-check scope is recorded in `CITATION_COVERAGE.json` and the
  fresh prior-art review.
- Prior CPU replay agrees on return statuses and quality categories but
  differs on six acceptance decisions and is not bitwise identical. The
  discrepancy is preserved in `TRUSTED_REPLAY_COMPARISON.json`.

These restrictions do not negate the stated conditional inverse problem.
They would block presenting the work as a general trustworthy recovery
system, which the revised paper expressly does not do. The current proof,
explicit observation model, prior-art attribution, and failure disclosure
provide a substantive result suitable for public scrutiny.

## Artifact and presentation checks

All 54 rendered pages were inspected in contact sheets for layout; pages 1,
4–6, and 29–30 were also inspected individually for the rewritten main claims,
tables, constants, and proof. No clipping or overlap was found. Dense legacy
tables require zoom. Final text/log checks show no undefined or duplicate
references and no overfull boxes. The final editorial review binds PDF text
independently of build-time PDF metadata.

Source attestations and the five checked summarizers/replay comparison pass.
Raw observations, frozen experiment code, tests, figures, and generated
numerical tables were not changed in this release round. Relative to the
pre-round manifest, the only executable-file edit is descriptive status
metadata in the package report; the packaging script was then executed
successfully. The existing 162-test passes in the primary and isolated CPU
environments are inherited from `TRUSTED_RECOVERY_VALIDATION.json`, not
misreported as a new test run. No new numerical experiment was performed in
this manuscript/review round.

The 51-file source archive was extracted and compiled without shell escape
in a clean directory (three pdflatex passes). Its extracted PDF text equals
the working manuscript. The package has not been processed by arXiv itself.
The final manifest check covers the research sources and delivery records;
generated `dist/` files have separate exact hashes in the release validation.

The paper-review and ml-paper-writing skills guided the evidence audit and
narrative revision; their provenance is in `SKILL_REFERENCES.json`. They are
procedural aids, not scientific endorsements. The goal is completed on the
basis of the scientific recommendation above, with packaging serving only
to make that concrete manuscript reproducible and reviewable.
