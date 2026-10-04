# Decision-dynamics prior-art boundary (2026-09-25)

This focused search supplements the inherited full-text audit in `paper/sections/appendix_literature.tex`, `RELEASE_PRIOR_ART.md` and `LLM_CITATION_AUDIT.md`. It does not claim exhaustive coverage or priority over inaccessible work. Searches targeted kernel/Gram clustering, response-preserving compression, and training-kernel matching; primary sources were read before narrowing the manuscript's claims.

| Closest source | What is already established | Boundary of this manuscript |
|---|---|---|
| Dhillon, Guan & Kulis, KDD 2004, *Kernel k-means, Spectral Clustering and Normalized Cuts* | Gram/trace formulations of clustering; section 3 | Equal-capacity softmax-response expansion makes the local covariance contribution partition independent; the generic clustering identity is credited |
| Liu & Zenke, ICML 2020, *Finding trainable sparse networks through Neural Tangent Transfer* | Initial-output and tangent-kernel matching for sparse-network training dynamics; section 3 | No novelty claim for matching dynamics during compression or generic trajectory perturbation; this paper separates the specific balanced sharing objectives |
| Jacot et al., NeurIPS 2018, *Neural Tangent Kernel* | Jacobian-Gram training response | The model-specific common-product stabilizer and finite-noisy inverse, not the existence of a response kernel |
| Vu Thanh, Gillis & Lecron, TSP 2023, bounded simplex-structured matrix factorization | Static simplex-factor ambiguity, including the uniformizing transformation | Same-product nonuniqueness itself is prior art; fixed Euclidean response supplies an additional observable |
| Afsari 2008, *Sensitivity analysis for the problem of matrix joint diagonalization* | Uniqueness and sensitivity of joint diagonalization | The full response-to-factor chain is distinguished from this inherited intermediate mechanism |

New citation provenance:

- NTT publisher record and supplied BibTeX: https://proceedings.mlr.press/v119/liu20o.html ; primary full text https://proceedings.mlr.press/v119/liu20o/liu20o.pdf ; independently corroborated author/title/ICML metadata at https://arxiv.org/abs/2006.08228 .
- Kernel k-means author-lab record and supplied BibTeX: https://bigdata.oden.utexas.edu/publication/kernel-k-means-spectral-clustering-and-normalized-cuts/ ; primary paper read through https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Dhillon-2004-spectral.pdf ; corroborated in the author's publication list https://www.cs.utexas.edu/~dml/inderjit_publications_by_topic.html .
- Raw fetched BibTeX and HTML SHA-256 values: `publication-sources/CITATION_EXTRACTION.json`. Bibliographic metadata was extracted from supplied records, not generated from memory. A separate local download of the kernel paper failed; the browser full-text read succeeded. No claim depends on a partial local PDF.

Resulting contribution language: exact response-optimal tying specialization; decision-specific sufficient information; strict interior conflicts, including equal-product/different-response optima; explicit fixed-loss transverse crossing with an open initial set. The Gronwall bound is standard and conditional. No broad “first dynamics-preserving compression” or “full recovery necessary” claim remains.
