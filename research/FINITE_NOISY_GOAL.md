# Theory-only goal and acceptance contract — 2026-09-25

User approved the precise finite-noisy-observation goal after rejecting the
previous substitution of experimental deliverables for theoretical progress.
This record fixes the mathematical target, not a benchmark score.

For known positive Euclidean rates and temperature, fixed L>=K>=2 and D-K>=L,
let the parameter domain contain row-stochastic A with A_ij>=alpha>0,
sigma_min(A)>=s_A>0, and B with sigma_min(B)>=s_B>0 and ||B||_F<=M.
Only the observed noisy product may determine the K probes, each with
Frobenius norm at most one. Responses are full L-by-D matrices.

The required output is a complete proof (or a strict impossibility result for
the actual existential-design/universal-feasible-set statement) that the
entire set of domain members fitting the product and response observations
within e_theta and e_R has permutation-quotient diameter at most
C_theta e_theta+C_R e_R, under an explicit noise threshold. Constants must
be expanded. No true row-space oracle, local truth neighborhood, common
connected component, or unproved equivalent-strength lemma is allowed.

Completion requires:
- Exact quantifiers, assumptions, observation norm, probe normalization,
  constants, threshold and zero/singleton/empty-set boundary interpretation.
- Proof of all newly used bridges, with older inverse theorems individually
  audited rather than treated as trusted facts.
- Independent adversarial scrutiny of dimensions, norms, factors of two and
  sqrt(K), subspace misspecification, remote feasible pairs and circularity.
- Honest classification as an explicit corollary if it composes existing
  results; no unsupported novelty, optimality or solver-convergence claim.

Experiments, local Taylor inequalities, code packaging, partial reductions,
and a plausible missing lemma do not satisfy the goal. A research time budget
does not turn an unfinished proof into a completed goal.

## Approach registry

A. Direct observed-subspace factor projection: use B -> B P_hat and
Eckart--Young plus A's left inverse. Root and independent derivation reached
this bridge without requiring a singular-vector perturbation theorem.

B. Principal-angle/operator-block compression: adversarial reviewer derives
a second route through projector error and P_hat T_i P_hat. Valid but yields
a looser basis-conditioning dependence; retained in its audit.

C. Obstructions: examine signed basis ambiguity, rank/boundary degeneration,
scalar counting, singular-value ties, and K=1. No admissible counterexample
to the fixed-domain statement found. Counterexamples to weaker observations
are not promoted into impossibility results for the stated design.

D. Foundation audit: inspect the complete-response theorem's Gram extraction,
covariance Sylvester inverse, diagonal-algebra conditioning and global
permutation rounding; then inspect the joint-product intermediate factors.
This dependency audit is required independently of routes A and B.

The repeated-complement probe variant was selected after the independent
route exposed its shorter norm bound. The construction uses the complement
code in every query and averages its decoding across the K responses.
