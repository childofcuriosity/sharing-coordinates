Current disposition: superseded by [RELEASE_REVIEW.md](RELEASE_REVIEW.md)
(2026-09-25). The document below records the earlier 2026-09-24 snapshot;
its absent-theorem and earlier-experiment limits are not a current inventory.

# Final scientific and artifact review — 2026-09-24

This report delivers the local research draft for author review. It does not
represent journal acceptance, independent peer review, or a public submission.
The user requested the existing ICLR 2027 template and author placeholders;
both are retained. Scientific success is bounded support, including negative
findings, rather than a predetermined positive result.

## Requirement-to-evidence audit

| Requirement | Authoritative evidence inspected | Disposition |
|---|---|---|
| Use the installed review and writing skills | SKILL_REFERENCES.json; scoped proof, empirical and frontier reviews | Applied evidence/claim separation, primary-source checks, explicit uncertainty and reproducibility review |
| Independently check exact response identification | RESPONSE_IDENTIFIABILITY.md; manuscript exact proof; PROOF_REVIEW_2026-09-24.md | Full-rank same-product argument, Gram extraction, PSD square root and diagonal-algebra reduction reviewed; no gap identified within stated assumptions |
| Check quantitative stability and finite observations | Manuscript four quantitative steps, constants, degenerating examples and tomography proof; proof ledger | Same exact product and interior/rank conditioning required; boundary trace lower bound and probe assumptions clarified |
| Establish contribution against prior work | frontier_response_identifiability.md; CITATION_COVERAGE.json; revised related-work appendix | 47 primary-text passage checks, 2 limited dispositions; static gauge, general response construction and terminal uniqueness attributed; workshop technical priority remains unresolved |
| Use local resources for traceable new evidence | Three seed records in each of results/autograd_tomography and results/factor_recovery; prospective protocols; source/checkpoint hash validators | Six additive full-dimensional checkpoint runs on local GPUs; no source-bound raw record replaced |
| Retain negative results and empirical units | EMPIRICAL_REVIEW_2026-09-24.md and raw/derived action records | Failed folding searches, initialization dominance, finite gauge grid and repeated checkpoint/probe units retained |
| Make numerical claims reproducible | REPRODUCIBILITY.md; strict summarizers; run_all.sh verify | Commands, environment, source attestations, derived tables and corruption-sensitive tests supplied; historical training not independently rerun |
| Source, PDF and figures agree | Clean arXiv archive build; dependency hashes; extracted-text comparison; VISUAL_REVIEW.md | 39 archived files; clean build succeeds; all source dependencies match report hashes; full visual pass plus targeted follow-ups |
| Check all legacy figures/tables | Fresh analyze_results run in .artifact-handover/final-derived | 9 TeX outputs and summary byte-identical; 6 PDFs render pixel-identically at 1600-pixel scale (PDF metadata may differ) |
| Deliver release artifacts and limitations | paper/main.pdf; dist source archive/preview/report; this report and DELIVERY_STATUS.md | Local author-review delivery; public upload and author identity remain user-controlled |

## Scientific assessment

The supported central result identifies full-rank simplex/basis factors up to
common permutation from the exact effective product and its complete response
under known positive Euclidean block rates. It does not identify a historical
sharing graph. The quantitative result strengthens this on uniformly interior,
rank-conditioned bounded sets. Structured probing and constructive recovery
make the response-to-factor chain inspectable; neither the general tangent
kernel nor joint diagonalization is itself new.

Three existing checkpoints pass the new autodiff tomography gates. The largest
held-out response prediction error is about 5.1e-14; noiseless factor-recovery
orbit errors are about 3.2–3.7e-9. These observations validate implementations
at those checkpoints, not population accuracy or a training-trajectory theorem.

The ASLoRA reimplementation demonstrates changed first choices on 11/30
adjacent-pair and 14/30 all-pair fixed-grid cases. It does not establish full
training regret or reproduce official code. The byte-LM fold witness succeeds
on only one of three searches, is initialization dominated and changes group
sizes. Probe results do not demonstrate general superiority; the effective
parameter baseline achieves perfect ARI on the planted grid.

## Unresolved limits delivered with the result

- No joint inverse theorem when both effective product and response are noisy;
  PSD-clipped noisy factor recovery remains exploratory.
- No theorem for arbitrary optimizers or unknown learning rates; signed
  eta_B=0, K>=3 remains outside the main positive-rate result.
- No optimal-query claim, broad end-to-end training validation, cross-seed
  folding replication, or general advantage for shared-input probes.
- Some original ASLoRA inputs/checkpoints are not bundled; original training
  execution is not certified by a stored result or hash alone.
- ASLoRA journal full text and the Karuturi workshop technical content remain
  unavailable. Metadata checks do not establish technical precedence.
- Human scientific approval, author details and actual arXiv processing are
  deferred. The existing draft disclosure states this accurately.

## Final local checks

The latest test execution passed 148 tests with one dependency deprecation
warning and no skips. Source attestations and primary/additive reconstruction
checks passed. A concurrent PDF rebuild changed the manifest input at the end
of that run; no raw scientific record was changed. The final sequential release
check is recorded below after refreshing the completed delivery inventory.

Manuscript SHA-256: `660e5a7af6408f2fbfd242607ffe43a5f81c635889e85aefed0026421403ae79`.

Archive SHA-256: `712e1c2bce65ff2393a36a913c1b31ef225ffa2f6cfaba1bbad437029c4e3659`.

Final sequential verification: **PASS**, 148 tests, one dependency warning,
no skips; source attestations, additive/primary derived records and manifest
all passed. Log SHA-256: `fa48c3d42c183331797f54ac8310bd0d7904cd199b2b26e0699bf4b2c01c4965`.
Final documentation-only edits were followed by a fresh manifest check.
No required agent implementation or review task remains; the unresolved
research extensions and external full-text limitations above remain disclosed.
