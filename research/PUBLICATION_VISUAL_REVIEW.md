# Integrated manuscript visual inspection — 2026-09-25

Rendered the manuscript with Poppler, inspected every page in contact-sheet overview, and inspected the main dynamics figure and affected appendix tables at larger scale. The final PDF has 72 pages; the main discussion ends on page 11, followed by references and complete appendices. This is a preprint layout, not an ICLR page-limit certification.

The first build had 73 pages. An all-runs table restricted to a float-only page delayed subsequent tables to the end of the supplement. Changed that table to permit ordinary placement; it now appears on page 22, and the local precision table on page 23, next to their discussions. Regenerated scientific notation in the precision table. The numerical values were not changed.

Figure 2 is readable at manuscript size: score axes, update/time distinctions and convergence axes are labeled; separate line styles supplement color; the convergence reference is a slope guide rather than a confidence interval. The controlled-regression figure preserves the unsuccessful FP32 refinement. Exact partition certificates and all-run outcomes fit their columns. Overview found no clipped content, accidental blank pages or overlaps. Some historical supplementary tables remain dense and some appendix floats occupy dedicated pages; their complete values also exist as machine-readable artifacts.

Clean-build logs have no undefined references, overfull boxes or pending cross-reference reruns. The independent archive rebuild has identical extracted text to the working manuscript. This is a local visual/TeX check, not a prediction of arXiv's renderer.
