# Prospective structured reconstruction from task gradients — 2026-09-25

Registered before any task-gradient outcome. This supplements Stage C of the
enhancement protocol with a structural estimator, alongside the generic span
predictor. Known rates and tau are one in these checkpoint runs.

Given row basis U of the exact product, write G_perp=G-(GU^T)U and similarly
R_perp. The response formula implies R_perp=Q G_perp, Q=A A^T. Pool the first
q natural directions and solve Q=(sum R_perp G_perp^T)(sum G_perp G_perp^T)^+.
For each row i, solve r_i U^T-(QG)_i U^T=T_i (g_i U^T) over those q directions.
The estimator uses observations and U only, never the true A,B. Pseudoinverses
use relative cutoff 1e-12. Full row rank of the pooled complement design and
rank K of every local design suffice for algebraic uniqueness, not guaranteed
conditioning. Report ranks, singular values and all failures.

For q=1,4,8,12 report component errors against truth (evaluation only),
prediction on the last four directions, and existing constructive factor
recovery with fixed weights seed 520000. No selection over weights/cutoffs.
The generic predictor assumes only linearity; the structured predictor also
assumes the paper's known response form. The 16 training minibatches at one
checkpoint are not independent training replicates. Finite-step precision
sweeps remain distinct from exact-response recovery.
