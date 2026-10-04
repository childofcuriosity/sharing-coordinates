# Release editorial source checks

Date: 2026-09-25. Scope: limited claim-alignment audit of the supplied 55-page manuscript, not a fresh full novelty review. No historical review conclusions were read. Submission cutoff is unspecified; the paper identifies itself as a September 2026 preprint.

Primary pages accessed:

- Bhaskara, Charikar, Vijayaraghavan, *Uniqueness of Tensor Decompositions with Applications to Polynomial Identifiability*, COLT/PMLR 35, 742–778 (2014): https://proceedings.mlr.press/v35/bhaskara14a.html . Official abstract confirms a robust Kruskal uniqueness antecedent. This check does not independently verify the manuscript's specific Theorem 5 reduction.
- Novak, Sohl-Dickstein, Schoenholz, *Fast Finite Width Neural Tangent Kernel*, ICML/PMLR 162, 17018–17044 (2022): https://proceedings.mlr.press/v162/novak22a.html . Official abstract confirms Jacobian-derived finite NTK computation as established background. No empirical performance comparisons are inferred.
- Hu et al., *ASLoRA: Adaptive Sharing Low-Rank Adaptation Across Layers*, arXiv:2412.10135v2 (16 December 2024): https://arxiv.org/abs/2412.10135 . Official page confirms title, version date and the 2024 author list. This metadata check does not resolve candidate-set ambiguity or the undefined update-ratio operation reported by the manuscript.

The editorial audit's mathematical correction about the uniformizing path is direct algebra from the manuscript, not a novelty claim requiring literature authority. Its query-mismatch finding compares the manuscript's own Corollary 1, Appendix D Eq.116, and Appendix M protocols. Existing references were not added to a separate final-reference list. All broader latest-literature coverage remains outside this focused audit; no superiority, first-result or exhaustive-coverage claim is made.

## Final-pass addition

Opened the primary Bhaskara et al. PDF at https://proceedings.mlr.press/v35/bhaskara14a.pdf and read Theorem 5 (PDF page 7, printed page 7). Its tensor tolerance is epsilon-prime divided by a polynomial in fixed dimensions/conditioning/bounds, and its factor conclusion is epsilon-prime up to permutation and diagonal scaling. This directly supports the revised manuscript's acknowledgement that linear error dependence at fixed conditioning is already known. It does not alone establish the manuscript's response-to-tensor reduction, which was checked as part of the manuscript's own argument.
