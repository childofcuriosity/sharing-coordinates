# Integrated preprint self-review — 2026-09-25

**Scientific recommendation: release this scoped theoretical preprint.** This recommendation concerns the integrated manuscript, not conference acceptance, a new compression method, broad natural-training prevalence, or demonstrated downstream gains. Author metadata remains intentionally deferred by the user; the ICLR2027 template is retained. Nothing has been uploaded or pushed. Exact mechanical validation and file identities are recorded in `PUBLICATION_VALIDATION.json`.

## What this paper establishes

The scientific thread is now explicit: effective weights need not identify optimizer coordinates; a specified Euclidean response can identify them; choosing a particular sharing decision may require less information; and distinct decision objectives can separate along training.

The core full-rank softmax/shared-basis stabilizer, nondegenerate inverse, joint-product perturbation bridge and finite-noisy whole-feasible-set result are retained. They require their stated metric, known rates, internal response observations and rank/interior/domain assumptions. The paper does not turn a noisy fit or a calibrated rejection rule into a theorem certificate.

The new decision analysis defines coefficient assignment, effective-weight clustering and router-vector clustering separately. Balanced Hilbert–Schmidt-response-optimal tying equals router clustering; off-diagonal router Gram entries already suffice. Thus full coordinate recovery is not generally necessary for that decision. Two-basis equivalence, tangent isotropy, exact strict interior conflicts, and equal-product/different-response-optimum examples establish both positive and negative boundaries. A fixed squared loss gives a transverse coefficient-boundary crossing while the global weight partition remains fixed, with an open set of forward initial states. Margin protection and the trajectory bound are conditional stability statements, not attraction or typicality theorems.

## Claim–evidence audit

| Claim | Evidence checked | What does not follow |
|---|---|---|
| Identifiability and finite-noisy consequences | Inherited proof chain read; five inherited theory appendix hashes match the archived release; prior proof audits retained | AdamW identification, hidden historical graph recovery, efficient global solver |
| Response-optimal balanced tying | Blockwise HS expansion; fixed diagonal/covariance terms; constant group sizes and identical basis rates | Task-gradient-optimal or AdamW-optimal partition |
| Strict decision conflict and equal-product impossibility | Exact integer/rational constructions, all 15 pair partitions; assignment dual prices; K=4 extension checked independently | Conflict on most natural checkpoints |
| Training can enter coefficient/weight disagreement | Fixed-target smooth ODE; negative transverse derivative; other strict gaps and rank/interior margins; continuous dependence on a compact interval | A natural-task probability or attraction mechanism; a theorem of router-clustering/weight crossing |
| Recovery in real pretrained modules | Preserved nine-model/27-checkpoint study; 1,080 sufficient-rank fits and 540 rejected rank-deficient cases | Full-backbone identification or downstream superiority |
| Natural training occurrence | All 18 exploratory runs; 4/9 AdamW coefficient entries, 9/9 response-objective entries, 0/9 unretuned SGD entries | Independent prevalence samples or optimizer-quality comparison |
| Selected real Euclidean reproduction | Original SmolLM2 update-70 state and next real batch; four step sizes; unchanged weight winner; dtype-preserving FP64 refinement | An unbiased new sample, exact ODE integration, generalization benefit |
| Constructed discrete mechanism | Eight primary runs, stationary control, forward ODE reference, all 60 perturbations including four initially ineligible cases | Natural-training replication |

The independent exact-certificate checker uses integer arithmetic and rational division rather than the floating pair-distance routine. The publication verifier recomputes 120 costs via projector traces, checks generation sources/protocols/checkpoints, checks all 106 archived dependencies, and resolves historical manuscript hashes against their own version. These checks complement, rather than replace, the analytical proofs.

## Issues resolved for this release

1. New theory and training results are integrated into the main paper and complete appendices; they are no longer only research notes or an experiment-only explanation.
2. The three grouping rules and their different centers/metrics are defined before results. Global enumeration is distinguished from the earlier Lloyd heuristic.
3. Full recovery is not oversold: the decision-specific sufficient Gram information is explicit. The matched original decision experiment found no advantage over effective-weight clustering and remains a negative finding.
4. The constructed-gradient theorem is stated as existential with a fixed loss and an open starting set. Real-model evidence uses natural language losses, but the local follow-up's selected state and batch are disclosed.
5. Ordinary SGD no-events, repeated-seed dependence, BF16/FP32 arithmetic, FP32 failed refinement and both precision follow-ups remain visible. AdamW observations are not labeled Euclidean-flow proofs.
6. Kernel k-means and Neural Tangent Transfer were checked in primary sources and cited. Gram clustering, kernel matching and Gronwall perturbation are credited precedents. Earlier static-SSMF, joint-diagonalization, robust-tensor and operator-probing boundaries remain intact.
7. The original PDF, package and validation dependencies were archived before integration. Historical validations are not rewritten to certify changed manuscript bytes. New deterministic figure/table generation and a version-aware read-only verifier are wired into the standard verification entry point.
8. Main text, proofs, supplementary protocols, complete outcomes and rendered pages were reviewed. A deferred all-runs table was moved near its discussion. The minimal TeX archive is rebuilt and tested independently of the repository layout.

## Remaining limitations that are disclosed, not hidden deliverables

This remains a specialized theory paper with controlled empirical support. The response objective assumes balanced fixed routes, matching basis learning rates and an isotropic full-operator norm. The trajectory bound assumes uniform response mismatch along the entire interval. The finite-noisy theorem needs a domain-dependent noise threshold and does not certify the practical solver. Full-rank/interior assumptions and conservative constants restrict scope.

The broad recovery study covers nine dense pretrained models up to 8B, but the trajectory grid covers three models up to 0.6B with a small shared module on 12 projections. Backbones are frozen. The local FP64 study uses one post hoc selected SmolLM2 state, one batch and an explicitly modified arithmetic path. Full recovery was not part of those native-factor dynamics measurements. No result shows that the proposed observations improve compression, latency, generation accuracy or downstream loss relative to a simpler decision control.

The manuscript is deliberately long because it retains full proofs and the complete historical/negative evidence. A future venue submission would benefit from a narrower presentation and venue-specific page budgeting; that is not required for this preprint. One earlier workshop source remained metadata-only in the inherited literature audit, so no priority assertion is based on its inaccessible full text. Searches are bounded, not an exhaustive novelty certificate.

## Review provenance

This is an AI-assisted self-review using `ml-paper-writing` and `paper-review` skills, with direct source/proof/result inspection and executable artifact checks. It is not new external human peer review or a formal proof-assistant certification. Earlier independent-agent audit reports are historical records, not newly commissioned reviewers for this integration. Human author information and responsibility remain to be supplied before actual submission, as explicitly deferred by the user.
