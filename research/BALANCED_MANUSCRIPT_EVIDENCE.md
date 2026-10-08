# Current manuscript: evidence and chapter plan

Updated 2026-10-09. Active manuscript: `paper/main.tex`, including `paper/sections/balanced_stage.tex`. This is a working ledger, not another manuscript version.

## Evidence map

| Claim | Evidence | Scope |
| --- | --- | --- |
| Research motivation and human interventions | Author's conversation, recorded in the research-history section | Attribution of direction and corrections; no general conclusion about AI capability |
| Balance conservation | Du, Hu, Lee (2018), Theorem 2.2; differentiation in text | Equal-rate Euclidean flow, free factors; convergence is conditional |
| Dictionary-column and coefficient-row geometry | Expansion of the balance identity in text | Exact balance; basis comparisons, not layer comparisons |
| Coefficient Gram identity | Complete algebraic proof in text | Any balanced product, including rank-deficient products |
| Minimum factor norm and orthogonal freedom | Proofs using compact SVD and product-preserving variations | Complete orthogonal parameterization stated for rank(W)=r |
| Second-stage SGD attraction | Existing project derivation SECOND_STAGE_SELECTION.md ?8; Li?Wang?Arora (2022), Theorem 4.6 and Corollary 5.2; proof reproduced in manuscript | Fixed isotropic matrix observations, minimal width, small-step limit, arbitrary entry on optimal fiber; weighted balance |
| MASA | arXiv:2508.04581v2, method, experimental setup, Figures 3–10 | Joint training and pretrained PCA distinguished; displayed axes checked |
| NPAS | arXiv:2006.10598v4, Sections 3.1–3.2, Appendix B.3 | Direct coefficients versus affine embeddings; template resizing |
| ASLoRA | arXiv:2412.10135v2, Sections 3.2–3.5, Figure 3 | Historical matrix averages; allocation plot |
| SuperWeights | WACV 2024, Section 2.2, Table 3 | Gradient compatibility differs from static factor geometry |
| Prior identifiability work | Carrington, Bharath, Preston (2019), full text | Similarity evaluation and orthogonal freedom already studied |
| Prior SVD grouping work | Shamrai, arXiv:2601.11626v1 | Existing error-controlled clustering |

Primary URLs and bibliographic entries are in the manuscript. Independent citation review accompanies this update.

## Chapter plan

1. Motivation: equivalent heatmaps versus actual-training relevance.
2. Setting: parameter-saving shared basis, unique optimal product, factorization fiber.
3. Results: balance, D–C and W–C relations, remaining coordinate freedom.
4. Optimization: classical sufficient case and the unproved practical extension.
5. Applications: match each relation to what authors actually plot or cluster.
6. Research history: human decisions that determined the scientific question.
7. Future work: actual-training probability and concrete downstream decisions.

The unresolved link is whether actual application optimizers and parameterizations produce sufficient balance, with useful error and probability bounds. Uniform rotation, mixing time, rarity of coordinate changes, and improved downstream performance are not established results in this version. Earlier exploratory experiments are not repurposed as proofs of these claims.

The SGD model result is included as established within its assumptions: attraction from every fiber entry point and a sequential-time/step-size asymptotic probability statement. The open practical question must not erase this result or reopen first-stage convergence, which is outside the selected scope.
