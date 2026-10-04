# Independent mathematical rebuild of the release theory

Date: 2026-09-25. Scope: the definitions in `paper/sections/problem.tex` and `theory.tex`, followed by `appendix_theory.tex`, `appendix_enhancement_theory.tex`, and `appendix_finite_noisy_theory.tex`. I additionally read the auxiliary statements in `appendix_secondary_theory.tex` to check the hypotheses of their proofs. I did not consult previous audit reports or their conclusions, run experiments, or search the web. This report checks the mathematics as written, not novelty, bibliographic accuracy, empirical validity, or solver performance. No paper file was changed.

**Result:** I found no blocking mathematical error in the exact-response → same-product linear stability → joint product/response stability → entire finite-noisy feasible-set diameter chain. The explicit constants cover the operations actually used. The three quantitative degeneration families are valid. The conclusion is conditional on the stated domain, complete controlled Euclidean responses, and known metric; it is not a statement about passive optimization histories or successful numerical inversion.

## 1. Rebuilding the response and exact identification from definitions

Write router rows as columns. For `theta_i = B^T a_i` and `a_i = softmax(z_i/tau)`, the Jacobian is `J_i = C_i/tau`, where `C_i = diag(a_i)-a_i a_i^T`. The chain rule gives

`dot a_i = -(eta_Z/tau^2) C_i^2 B g_i`,

`dot B = -eta_B sum_j a_j g_j^T`.

Differentiating `theta_i` therefore gives exactly the response blocks in the paper. There is no missing temperature factor or transpose. Arbitrary effective gradients are implementable as the gradients of linear losses on the effective product.

Independently, the identification argument is:

1. Full rank of both factors makes the common product rank K. Its column space gives `A'=AM`, `B'=M^{-1}B`, with M invertible. Stochasticity and injectivity of A imply `M1=1`.
2. Off-diagonal response blocks identify the off-diagonal entries of `AA^T`. Each diagonal block is `eta_B ||a_i||^2 I + lambda H_i`, where `H_i=B^T C_i^2 B` is PSD and singular. The inequality `rank(H_i)<=K-1<D` still holds at D=K. Two singular PSD matrices cannot differ by a nonzero scalar identity: the matrix on the positive side would become positive definite. Thus diagonal response equality also identifies the Gram diagonal and H_i separately.
3. Gram equality and a left inverse of A give `MM^T=I`. Cancelling B using its right inverse gives `C_i^2=M(C_i')^2M^T`. Both potential square roots are PSD, so uniqueness of the PSD root gives `C_i=MC_i'M^T`.
4. Because `a_i'=M^T a_i`, the rank-one parts cancel, leaving `diag(a_i)=M diag(a_i') M^T`. Each off-diagonal entry is a linear equation in a row of A'. Full column rank of A' implies `M_{rj}M_{sj}=0` for all r≠s and j. Orthogonality makes M a signed permutation; `M1=1` eliminates negative signs.
5. Direct substitution verifies the converse for a simultaneous permutation of A and B. Independent logit row shifts persist.

This does not assume that positivity plus Gram equality already force a permutation. They do not; the covariance blocks and spanning step do the essential extra work. Exact uniqueness also extends algebraically to boundary simplex rows with full factor ranks: this particular proof uses PSD, not a positive covariance gap.

## 2. Same-product quantitative inverse: fragile steps checked

**Block extraction and singular PSD separation.** Injecting one input block and projecting one output block are contractions, so each response-difference block has spectral norm at most delta. For a diagonal block `E=tI+lambda(H-H')`, a vector in `ker H'` handles t≥0 and a vector in `ker H` handles t<0. These yield `|t|<=delta` without assuming a shared kernel of H and H'. Consequently `||H-H'||<=2delta/lambda` and `||AA^T-A'A'^T||_F<=L delta/eta_B`. The manuscript uses the correct kernels and signs.

**Singular-value inversions.** The identities `M=A†A'`, `M^{-1}=(A')†A` give both norm bounds by `L_A/s_A`. Cancelling the Gram difference costs `s_A^{-2}`; cancelling the local basis quadratic form costs `s_B^{-2}`. These are the stated c_G and c_H. No unsupported singular-value product inequality is used. The right inverse of B is valid at D=K and D>K.

**The covariance square-root step.** Set `Y=M^T C M`, not `M C M^T`; expansion of Y² produces precisely the extra `MM^T-I` term in the paper. This orientation matters and is correct. The simplex inequality `C(a)>=alpha Pi_{1-perp}` follows by writing `a=alpha 1+r`, applying weighted Cauchy–Schwarz, and using `sum r<=1`.

Both Y and C' have exactly the same kernel `span(1)`: `M1=1` is indispensable here. On `1-perp`, the proof's bound `Y>=alpha/(4L_M²) I` is valid. Indeed its decomposition `Mx=c1+y` and `x=c1+M^{-1}y` proves `||y||>=1/(2L_M)`; even the stronger `x=Pi M^{-1}y` gives `||y||>=1/L_M`. This possible sharpening is unnecessary. The written smaller gap is safe.

On that invariant subspace the Sylvester identity is exactly

`Y(Y-C')+(Y-C')C' = Y²-(C')²`.

The exponential integral bounds its inverse in spectral norm by the reciprocal of the sum of the two gaps, without requiring commutativity of Y and C'. Extending by zero on the shared kernel is valid. Thus the stated c_J is a linear, rather than square-root/Hölder, bound. A generic PSD square-root perturbation theorem would not suffice here, but the manuscript does not rely on one.

**Diagonal algebra and rounding.** Covariance expansion again cancels the rank-one terms exactly. Passing to Frobenius norm costs sqrt(K), stacking the L equations costs sqrt(L), and a pseudoinverse row costs at most 1/s_A. This gives c_D as stated.

For a row r of M, approximate orthogonality gives `||r||²>=1/2`; its largest coordinate has squared magnitude at least 1/(2K). The off-diagonal outer-product bound then gives tail squared norm at most `K c_D² delta²`. The bound on its peak square converts to a bound on its peak magnitude since `||r_q|-1|<=|r_q²-1|`. These justify c_row. The Frobenius distance to the collected signed coordinate rows is at most 1/2. Repeated selected columns would make that matrix singular, contradicting `sigma_min(M)>=1/sqrt(2)`. A row summing to one is at least `2/sqrt(K)` from any negative coordinate vector, whereas the selected distance is at most `1/(2sqrt(K))`. All signs are therefore positive. Boundary equalities at delta=delta_0 leave strict numerical separation in both contradiction arguments.

The parameter conversion uses the valid identity `M^{-1}-P^T=M^{-1}(P-M)P^T`. For delta≥delta_0 the parameter-domain diameter supplies the second branch of C. This is a genuine global inverse on the specified bounded same-product set, not merely a local branch assertion.

## 3. Explicit degeneration and observation counterexamples

All three quantitative families preserve exact products and router Gram matrices, and their changed factors remain outside permutation equivalence for every sufficiently small positive parameter.

- **Router rank.** The displayed Q is symmetric orthogonal, fixes 1, and is not a permutation. `A_epsilon=U+epsilon(I-U)` has singular values 1, epsilon, epsilon. `A_epsilon Q=U+epsilon(Q-U)` remains uniformly positive for epsilon≤1/4. Bases retain unit singular values. At epsilon=0 the covariance is `(I-U)/3` and commutes with Q, so response difference is O(epsilon), while basis distance from the finite permutation set stays positive.
- **Basis rank.** `B_epsilon=1 e_1^T+epsilon I` is full rank for epsilon>0 (its determinant is `epsilon²(epsilon+1)`), with a singular-value margin tending to zero. `A=.3 11^T+.1I` and `AQ=.3 11^T+.1Q` have uniformly positive entries and full rank. Covariances kill 1, eliminating all constant and cross terms in the local quadratic responses, leaving an O(epsilon²) response gap. The router distance from its permutation orbit stays strictly positive because A is injective and Q is not a permutation.
- **Interior.** S is skew symmetric, kills 1, and has Frobenius norm sqrt(2); thus `Q_t=exp(tS)` is orthogonal and fixes 1. The off-diagonal first-order entries of `A_tQ_t` are `t(1±1/sqrt(3))`, so positivity holds. Its singular values equal those of A_t, namely 1 and `1-3t` twice. Both covariances are O(t). For a first-row perturbation of a vertex with off-vertex rates x,y, the leading covariance-square trace is `4x²+4y²+2xy`. At x=y=1 this is 10; at x=1−1/sqrt(3), y=1+1/sqrt(3) it is 12. Trace difference bounds spectral norm below by one third its absolute value, confirming a Theta(t²) response gap. The finite permutation set makes identity nearest for sufficiently small t; the basis displacement alone is sqrt(2)t+O(t²). The factor/response ratio therefore diverges as 1/t with rank and norm margins fixed.

The other displayed boundaries also check out: B→−B preserves response but generally changes product; uniform-router rotations violate only router rank; identical basis rows kill the local response and permit Gram-preserving rotations; eta_Z=0 leaves those rotations when K≥3; and `G=w v^T` with v perpendicular to the common basis row space and w in the Gram-difference kernel gives a nonzero blind gradient when D>K and L>2K. The stationary-gradient example is valid but weaker.

The partial eta_B=0 discussion is correctly limited. For nonnegative M the transformed columns are probability vectors at pairwise distance sqrt(2), forcing distinct vertices. For K=2, subtraction of the two positive-row equations yields equal sums of the old/new first coordinates, and the displayed `(1-d)[S(2-S)+d Delta²]=0` identity follows. No conclusion for general signed K≥3 gauges is proved or claimed.

## 4. Joint perturbations

The projected intermediate router `A_tilde=P_A A'` preserves row sums because 1 belongs to col(A). Multiplication of the product residual by the right inverse of B' bounds its displacement by epsilon_Theta/s_B. Entrywise positivity and singular values survive the stipulated threshold. Full column rank inside col(A) makes the two column spaces equal, so `A_tilde B_tilde=AB` exactly.

The identity `B_tilde-B'=A_tilde† P_A(AB-A'B')` follows from `A_tilde†P_A A'=I`, and gives the stated `2epsilon_Theta/s_A` bound. It preserves the basis singular margin s_B/2. Enlarged norm bounds are safe: orthogonal projection does not increase the router Frobenius norm, while the basis displacement is at most s_B/2≤L_B/2. The argument applies the previous theorem to valid enlarged-domain factors.

The Lipschitz bound uses `||C-C'||<=3||a-a'||` and `||C²-(C')²||<=6||a-a'||`, with simplex row norms at most one. The Gram term is bounded by `2eta_B sqrt(L)||Delta A||_F`; the local direct sum is bounded by its largest block norm. Cauchy–Schwarz in the two factor displacements gives exactly the stated L_R. The triangle inequality on permutation orbits is valid since the finite group acts by isometries. Thus the joint constant and the factor-of-two rule for two feasible candidates are justified.

## 5. Finite noisy observations and their quantifiers

The finite-noisy theorem does not covertly assume the observed row space is the true row space.

1. For every feasible factor, its rank-K product is a competitor for the best rank-K approximation of the same observed product. This gives an observed residual at most epsilon_Theta, then an actual product tail at most 2epsilon_Theta, and a basis tail at most `2epsilon_Theta/s_A`. This is uniform over the entire feasible set, even if a chosen top-K frame is discontinuous.
2. Projection of B changes only the local response and costs at most `t=4lambda M epsilon_Theta/s_A` in full operator norm. Projected bases need not meet the rank margins; they are used only in the larger linear response class.
3. The repeated complement code gives `E_p W^T=c Delta Q` for every p, so `||Delta Q||_F<=E/(c sqrt(K))`. Projection onto U gives every row of every local Delta T_i. The stacked shared contamination has norm `sqrt(K)||Delta Q 1||`, bounded by `sqrt(L)E/c`. Hence the stated h bounds the whole operator, even without symmetric local matrices.
4. Unit-Frobenius probes cost at most t each under projection, or sqrt(K)t for the stack. Two candidates therefore give `||R-R'||<=2h epsilon_R+2t(h sqrt(K)+1)`, reproducing beta exactly.
5. The original, unprojected factor products differ by at most 2epsilon_Theta. The stated threshold is precisely half the joint theorem threshold. Substitution `(alpha/2,s_A/2,s_B/2,2sqrt(L),2M)` reproduces every expanded c_G, c_H, c_D, L_M, C_* and L_R. No original-space basis component is discarded in the final distance.

All candidates share the same observed frame and probes. The proof is deterministic for any fixed observations satisfying feasibility; noise may depend on the observed product or probe selection. It proves a diameter bound for every pair, then takes the supremum; there is no union bound over candidates or illicit fixed-alternative-to-uniform transition. Large epsilon_R is covered by the earlier global diameter branch. Empty and singleton feasible sets are harmless. The theorem does not assert feasibility or furnish an algorithm finding a member. The K=1 bound follows directly from `||1_L(B-B')||_F=sqrt(L)||B-B'||_F`.

## 6. Remaining local checks and release limits

The exact structured tomography formulas, deterministic noise propagation, constructive PSD-root/joint-diagonalizer reconstruction, Gaussian rank conditions, and fixed-alternative Gaussian small-ball estimate are algebraically consistent. The latter explicitly fixes the alternative before drawing probes; its probability is not a simultaneous noisy certificate. The exact eigensystem reconstruction correctly removes eigenvector signs by reading conjugated diagonals and only claims an almost-sure noncollision for continuous random weights. Its numerical clipping implementation has no stability guarantee from this argument, as the text explicitly says.

Auxiliary proofs checked against their statements: local stochastic gauge, entropy increase, the numeric argmax example (determinant .5 and the four displayed transformed rows), affine barycentric uniqueness, two-sided separability, noisy one-hot clustering, Bernstein/union-bound partition recovery, softmax saturation, and symmetric initialization invariance are consistent. The composition example gives identity in both histories. The partition theorem's population gap remains an external hypothesis; existence of concentration constants is not a measured population certificate.

No mathematical correction is required by this rebuild. An optional sharpening of the covariance lower bound would only improve loose constants and is unnecessary for release. The mathematical chain supports the paper's conditional identification and stability claims; this report alone does not decide whether the empirical package, novelty framing, or arXiv submission as a whole is ready.

## 7. Final-main recheck

Date: 2026-09-25. This follow-up is limited to the rewritten `problem.tex`, `theory.tex`, and the abstract/assembly in `main.tex`, plus the small-product wording and bridge statement in `appendix_enhancement_theory.tex`. It compares the new proof sketches with the mathematical derivation above; it does not repeat the full appendix review. **No blocking mathematical inconsistency found.**

- The revised observation definition retains fixed positive Euclidean rates, temperature, arbitrary linear effective-gradient tests, and the product-only loss assumption. The permutation distance and operator norm agree with the appendices. The current-factor identification target remains distinct from historical or functional partition recovery.
- The exact proof sketch's smallest-eigenvalue explanation is valid: each individual diagonal response block is a scalar identity plus a singular PSD matrix, so its minimum eigenvalue is exactly the scalar. This remains true at D=K because the local rank is at most K−1. Gram recovery, orthogonal gauge reduction, PSD roots, and the diagonal-spanning step match the full proof.
- The linear-inverse explanation correctly invokes a shared nullspace and a positive gap on its complement. Precisely, the matrices compared by the Sylvester equation are the transformed covariance `M^T C_i M` and `C_i'`; both have kernel span(1). The summary's language is consistent with that step. Globality follows from the permutation rounding plus domain-diameter branch, not from an unstated local invertibility assertion.
- The joint bridge reproduces the exact small-product threshold, kappa, enlarged domain for C_*, and response/product coefficients. It does not require small response error or prior proximity of the original factors. Replacing “locally” by “for small product perturbations” in the appendix is accurate and removes a potentially misleading neighborhood implication.
- The finite-noisy statement retains the original-space feasible set, stacked response norm, observed-product-dependent probes, exactly K unit-Frobenius queries, and half-size product threshold. The new sketch gives the correct uniform basis tail `2 epsilon_Theta/s_A`, projection cost `4 lambda M epsilon_Theta/s_A`, h, beta, and pairwise full-operator bound. The subsequent product discrepancy is `2 epsilon_Theta`, exactly what the joint bridge requires. Arbitrary deterministic/adversarial probe-dependent noise is compatible with the deterministic feasible-set proof. No uniform-probability assertion or solver guarantee was added.
- The abstract correctly includes the noisy-product threshold, nondegenerate-domain premise, complete controlled response, and absence of a candidate-local premise. Its “unit-norm” queries are made precise as unit-Frobenius in the theorem. The K=1 discussion and the distinction from a candidate-local numerical certificate are consistent with the earlier proof.

One optional exposition refinement, not a theorem error: after decoding the shared Gram term using `W^T`, local blocks are recovered using `U^T` **after subtracting that shared term**. The present sentence abbreviates this subtraction. The full appendix spells it out and the preceding decoded-shared context makes the intended step identifiable. Adding “after subtracting the shared term” would make the main sketch self-contained at that point.

Reviewed snapshot SHA-256 values (the proof appendices are recorded as dependency snapshots, not as a claim of an additional full reread):

```text
b6a0b93737f044df95a04668a546d9aed093d98fa0ff784c4eaf82656352e427  paper/sections/problem.tex
6d73b4655d92e182ef8bb20f1f318ba1549f7563d78056d66c8322753f2b8a5e  paper/sections/theory.tex
d831fefcf5b0709eb29171cbf63c6d9ce8d8a715fc940d925bab1e3574db71b3  paper/main.tex
4ac846c9e423a4d79e1ac5afe365b8afc1242e8c49bb79276ccda5559841bffd  paper/sections/appendix_theory.tex
bd0ca56c5ca41b41370a3c170457c40f8342facf6a2374bd325673be83ace3dd  paper/sections/appendix_enhancement_theory.tex
edfc698ea8469b97bf259510057661f12a27ad53f7b43574866acefefc463293  paper/sections/appendix_finite_noisy_theory.tex
```
