# Integrated publication version — 2026-09-25

Current manuscript: [paper/main.pdf](../paper/main.pdf). The current decision and its exact validation record are [PUBLICATION_REVIEW.md](PUBLICATION_REVIEW.md) and [PUBLICATION_VALIDATION.json](PUBLICATION_VALIDATION.json). The 54-page and 61-page release judgments below and in older reports describe archived versions. The integrated paper retains the original proof chain and adds decision theory, ordinary trajectories, selected local real-model reproduction and controlled regression. No upload or push has occurred.

---

# Current release disposition — 2026-09-25

The current scientific recommendation is **publish as a scoped theoretical
arXiv preprint**, following the final [release review](RELEASE_REVIEW.md).
The proof chain, revised contribution boundary, and final claim/evidence
mapping were checked independently; the report states the limits of those
checks. This recommendation is not a venue-acceptance prediction or a claim
of external human peer review. No upload or remote push has occurred.

The current manuscript has a seven-page main text and 54 pages in total.
The ICLR2027 template and author placeholder remain as explicitly requested.
See [RELEASE_VALIDATION.json](RELEASE_VALIDATION.json) for the exact PDF,
source archive, and verification results. Source packaging is documented in
[ARXIV_PREPARATION.md](ARXIV_PREPARATION.md).

The older records below are historical snapshots. Their statements that a
jointly noisy-product theorem is absent, or that folding controls are
unexecuted, have been superseded by the subsequent theoretical and empirical
work. They are retained to document development, not outstanding blockers.

---

# Research draft status — 2026-09-24

Latest theoretical delivery: [FINITE_NOISY_THEOREM.md](FINITE_NOISY_THEOREM.md)
and [its adversarial audit](FINITE_NOISY_ADVERSARIAL_AUDIT.md). It establishes
an explicit global feasible-set diameter bound under the declared domain
conditions, using K unit-norm probes designed from noisy product observations.
The previous empirical study and its negative diagnostic findings remain valid.

Latest focused extension (2026-09-25):
[TRUSTED_RECOVERY_REVIEW.md](TRUSTED_RECOVERY_REVIEW.md) and
[TRUSTED_RECOVERY_VALIDATION.json](TRUSTED_RECOVERY_VALIDATION.json).
Its noisy estimator improves recovery, but its combined rejection diagnostic
does not establish general trustworthiness. The records below describe earlier
deliveries and are superseded on the added joint-noise/recovery topics.

The following is the earlier delivery. For the subsequently authorized
2026-09-25 extension, see [ENHANCEMENT_REVIEW.md](ENHANCEMENT_REVIEW.md) and
[ENHANCEMENT_VALIDATION.json](ENHANCEMENT_VALIDATION.json).

This is the scientific delivery record. The local research-draft deliverable is complete; human-author review remains pending.
Target: an arXiv presentation draft within several days, with venue selection
later. Keep the existing ICLR 2027 template and author placeholders, as the
user requested. These choices do not block research work.

## Claim-level status

| Claim or deliverable | Present support | Boundary / outstanding work |
|---|---|---|
| Exact optimizer-response identifiability up to common permutation | Assumption-by-assumption proof review; same exact product, positive full-rank router, full-rank basis and known positive Euclidean block rates | Does not identify a historical sharing graph or cover arbitrary optimizers |
| Quantitative inverse stability | Explicit constants and proof steps reviewed on uniformly interior, rank-conditioned, norm-bounded sets | Exact product is required; a jointly noisy-product theorem is not established |
| Structured finite-probe reconstruction | Derivation, deterministic observation-error bound and three new full-dimensional autodiff checkpoint audits | Packed factor-map response; neither LM-loss differentiation nor an AdamW training trajectory |
| Constructive factor recovery | Exact joint-diagonalization derivation, synthetic tests and three checkpoint pilots; noiseless orbit errors about 3.2–3.7e-9 | PSD clipping under noise is an exploratory estimator; no general noisy recovery guarantee |
| Coordinate-dependent first merge | Fixed-grid ASLoRA reimplementation audit: 11/30 adjacent and 14/30 all-pair per-projection changes | First-action consequences only; no full-training regret or official implementation reproduction |
| Byte-LM folding consequence | One successful search out of three; all outcomes retained | Initialization dominated; changed group-size balance; not cross-seed replication |
| Probe diagnostics | Paired grid summaries and empirical unit audit | No demonstrated general superiority; effective-parameter baseline is perfect on this planted grid |
| Novelty positioning | Selected closest primary theorems checked; static SSMF gauge and terminal diagonal/tensor uniqueness attributed to prior work | Candidate addition is the complete model-specific response-to-factor chain; 47 primary-text passage checks and 2 explicitly limited checks; workshop technical coverage remains unresolved |
| Manuscript presentation | All 43 rendered pages inspected; tables corrected without changing values or template | Dense supplementary tables require zoom; review applies to the PDF hash in VISUAL_REVIEW.md |
| Local reproducibility | Latest full verification passed: 148 tests, generated summaries/tables, attestations and manifest checks | Original training was not rerun; not all ASLoRA training inputs/checkpoints are bundled |
| arXiv source preparation | 39-file archive compiles in a clean directory; extracted PDF text matches | Not tested by arXiv itself and not submitted |

## Artifacts for review

- Manuscript: `paper/main.pdf`; editable source: `paper/main.tex` and sections.
- Generated submission draft: `dist/sharing-coordinates-arxiv-draft.tar.gz`.
- Clean-build preview: `dist/sharing-coordinates-preview.pdf`.
- Packaging hashes/checks: `dist/arxiv-package-report.json`.
- Proof, empirical and visual records: the dated reviews and VISUAL_REVIEW.md.
- Literature evidence: `frontier_response_identifiability.md`; structural
  citation inventory: `CITATION_INVENTORY.json`.
- New source-bound experiments: `results/autograd_tomography/` and
  `results/factor_recovery/`, with their protocols in `research/`.

## Final review disposition

The primary-source follow-up is closed at its stated scope: 47 cited keys have
primary-text passage checks; the ASLoRA journal record has metadata/abstract
coverage, and the Karuturi workshop paper remains technically unadjudicated.
See `CITATION_COVERAGE.json`. No priority inference is made from missing text.
The LoRA-S comparison was qualified to require the same product-space direction
across factor representations, then the archive was rebuilt and page 31 checked.

The requirement-to-evidence report is `FINAL_REVIEW.md`. The final sequential
release check passed (148 tests, one dependency warning, no skips). Research limitations in the table are deliverable
findings, not claimed completed extensions. No further full-training replication,
jointly noisy-product theorem, or optimizer generalization is asserted.

Author identity and human scientific approval remain deferred/user-controlled.
No public upload, remote push, author approval or acceptance is represented by
this record. The latest tests need not be rerun solely for this status document;
refresh the manifest after changing documentation.
