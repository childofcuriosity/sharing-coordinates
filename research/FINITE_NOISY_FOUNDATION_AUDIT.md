# Independent audit of the existing inverse-stability foundation

Date: 2026-09-25. Scope: mathematical proof audit only; no experiments, manuscript edits, or formal proof-assistant verification.

Audited source snapshots:

- `paper/sections/appendix_theory.tex`, SHA256 `4ac846c9e423a4d79e1ac5afe365b8afc1242e8c49bb79276ccda5559841bffd`.
- `paper/sections/appendix_enhancement_theory.tex`, SHA256 `853b38c7683f27f5206fa5bd818e69d210355e2ef40bf0d3cc9a2d0f4f3c85ea`.

## Verdict

I found no fatal gap, circular argument, or missing dimension factor in the same-product quantitative stability theorem or its joint-product corollary. They can serve as foundations for a finite-noisy-probe theorem, provided the new theorem explicitly establishes a uniform bound from finite observations to the **complete** response-operator discrepancy. The old results alone do not establish that step.

The old inverse theorem is genuinely global on its specified parameter domain, despite a local rounding argument in one branch. It does not assume proximity to a true factorization, membership in one connected component, or a successful numerical solver.

## Same-product theorem: checks by proof step

1. **Extracting individual blocks (lines 313–346).** Input injection and output projection have norm one in the Frobenius-induced operator norm. Hence each block discrepancy has spectral norm at most delta. For a diagonal block `E=t I+lambda(H-H')`, PSD singularity permits the sign-dependent kernel test: if `t>=0`, use a unit vector in `ker H'`; otherwise use one in `ker H`. This proves `|t|<=delta`, and subsequently `||H-H'||_2<=2 delta/lambda`. The singularity is valid even at `D=K`, since `rank H<=K-1<D`. The Gram Frobenius bound is `L delta/eta_B`, since there are `L^2` scalar entries. No unsupported extraction of a diagonal scalar from a generic matrix is used.

2. **The factorization gauge (lines 349–366).** Equal rank-K products and full ranks give the unique invertible gauge `A'=AM`, `B'=M^{-1}B`; simplex row sums give `M 1=1`. The bounds on both `M` and `M^{-1}` follow from the two router pseudoinverses. Multiplication of the Gram discrepancy by `A^dagger` on both sides gives `||I-MM^T||_2 <= L delta/(eta_B s_A^2)`. Replacing the Gram Frobenius norm by its spectral upper bound is valid.

3. **Covariance-square cancellation (lines 369–392).** The right inverse of B has norm at most `1/s_B`. The identity for `Y_i=M^T C_i M` expands exactly: the two terms sum to `M^T C_i MM^T C_i M-C_i'^2`. The stated bound `L_M^2(c_G+c_H) delta` follows without an extra power of `L_M`.

4. **Uniform positive covariance gap (lines 395–427).** For a probability vector with entries at least alpha, write `a=alpha 1+r` and apply weighted Cauchy–Schwarz with `sum r<=1`. This yields `C(a)>=alpha Pi_{1-perp}`. Because `M^{-1}1=1`, `ker(M^T C_i M)=span{1}`, exactly matching the kernel of `C_i'`. The argument proving `||Pi_{1-perp} Mx||>=1/(2L_M)` is valid; indeed a stronger `1/L_M` bound follows directly by orthogonally projecting `x=c1+M^{-1}y`. The weaker stated constant remains correct.

5. **Linear square-root step (lines 430–445).** Restricted to the common orthogonal complement, the Sylvester equation has the displayed exponentially decaying integral solution. Spectral submultiplicativity gives its inverse norm bound `1/[alpha+alpha/(4L_M^2)]`. Both matrices and their difference annihilate the common kernel, so extension to the full space is valid. The argument does not incorrectly apply a globally Lipschitz matrix square root at zero.

6. **Diagonal-algebra estimate (lines 449–478).** Rank-one terms cancel because `a_i'=M^T a_i`. Removing a diagonal is only claimed contractive in Frobenius norm, which is correct. The spectral-to-Frobenius conversion contributes `sqrt(K)`, and solving the L-row system by a pseudoinverse row contributes `sqrt(L)/s_A`. Thus `c_D=sqrt(LK)c_J/s_A` includes the necessary dimension factors. Off-diagonal projection need not be contractive in spectral norm, but the proof never assumes it is.

7. **Rounding (lines 481–519).** For `delta<=delta_0`, almost orthogonality gives squared row norm at least one half. The two off-diagonal terms involving a maximal coordinate give the row-tail bound. The peak discrepancy uses `delta^2<=delta`, explicitly permitted by `delta_0<=1`. Turning squared peak error into absolute peak error uses `||x|-1|<=|x^2-1|`, so no denominator is missing. The threshold ensures `||M-S||_2<=1/2`; a singular S contradicts `sigma_min(M)>=1/sqrt(2)`. Finally row sums rule out negative coordinate vectors: their distance from a vector of row sum one is at least `2/sqrt(K)`. The per-row rounding error is at most `1/(2sqrt(K))`, as needed.

8. **Factor bounds and global branch (lines 521–540).** The identity `M^{-1}-P^T=M^{-1}(P-M)P^T` is correct. Using Frobenius upper bounds for A and B is conservative. The large-delta branch uses the finite ambient parameter diameter `2 sqrt(L_A^2+L_B^2)`, divided by the positive explicit threshold. This closes the argument for every pair in the stated domain, including distant pairs. No compactness-only inverse modulus or unproved local-to-global assertion is hidden here.

## Joint-product corollary: checks

1. Set `Atilde=P_A A'`. Since `P_A 1=1`, row sums are preserved. The product residual and `B'^dagger` give `||Atilde-A'||_F<=epsilon_Theta/s_B`. The alpha threshold preserves positivity. The second threshold even gives `sigma_min(Atilde)>=3s_A/4`, stronger than the claimed half margin.

2. Full column rank of Atilde and inclusion in `col A` imply equality of their column spaces. Thus `Btilde=Atilde^dagger AB` satisfies the exact product identity. Expanding `Atilde B' = P_A A'B'` proves the displayed displacement formula, yielding `||Btilde-B'||_F<=2 epsilon_Theta/s_A` and the half B rank margin.

3. The enlarged norm bounds, asserted briefly in the manuscript, are justified explicitly as follows. The second noise threshold gives `||Atilde-A'||_F<=s_A/4<=L_A/4` and `||Btilde-B'||_F<=s_B/2<=L_B/2`. Therefore the conservative upper bounds `2L_A,2L_B` hold. The inequalities `s_A<=L_A` and `s_B<=L_B` follow from a nonempty admissible domain.

4. For simplex rows, `||C(a)-C(a')||_2<=3||a-a'||_2` and `||C(a)^2-C(a')^2||_2<=6||a-a'||_2` follow by direct expansion. With both basis norms at most M, each local response block differs by at most `lambda(6M^2||Delta a_i||+2M||Delta B||_F)`. The local response is a direct sum: its induced norm is a maximum, not a sum over layers. Combining it with the Gram bound `2 eta_B sqrt(L)||Delta A||_F`, followed by scalar Cauchy–Schwarz, gives precisely the stated `L_R`. No missing `sqrt(L)` occurs in the local-block term.

5. Applying the audited same-product theorem to `(A,B)` and `(Atilde,Btilde)`, then adding the displacement from the second original pair, proves the stated joint bound. The quotient distance has the required triangle inequality because simultaneous permutation acts isometrically and the group is finite.

6. “Relaxed locally” in the introductory prose refers only to the allowed **product residual size**. The factor-pair conclusion is global; it must not be reinterpreted as requiring a common local branch. The corollary's last paragraph uses `2e_R` only when the supplied response observations control the complete operator norm. For finitely many responses, that substitution requires an additional theorem.

## Requirements for using these results in the new goal

- Specify `K>=2`, or separately handle K=1. The audited theorem explicitly assumes K at least two.
- Use `L_A=sqrt(L)` for the proposed simplex domain, and `L_B=M` for its prescribed basis Frobenius bound. Do not substitute measured candidate singular values for domain-wide lower bounds.
- Expand or reproduce the explicit constant definitions; a bare unspecified inverse constant would fail the goal even though the existing theorem supplies one.
- Normalize each designed probe. The old exact probes have Frobenius norms `sqrt(2L)` and `sqrt(L)`, so their unscaled decoding formulas cannot be reused under a unit-norm query constraint without corresponding noise amplification factors.
- Control the whole feasible set uniformly. A subspace estimate made from observations is common to all feasible pairs, which permits such a uniform argument; selecting a true row space separately for each pair would not define one observable query design.
- State the diameter of an empty feasible set by convention or restrict the theorem to a nonempty set. Neither inverse result establishes existence or computational findability of feasible factors.

## A possible simplification of the missing subspace step

Let P be the rank-K right singular projector of the observed product, with any fixed tie-breaking rule. For any feasible pair `Theta=AB`, optimality of truncated SVD gives `||Theta_hat(I-P)||_F<=epsilon_Theta`. Hence, uniformly over the feasible set,

`||B(I-P)||_F <= ||A^dagger||_2 ||Theta(I-P)||_F <= 2 epsilon_Theta/s_A`.

This is a direct residual bound, requiring neither a true subspace oracle nor a particular Davis–Kahan constant. Replacing B by BP preserves the router and changes a complete response by at most `2 lambda_Z M ||B(I-P)||_F`, since both basis norms are bounded by M. If `2 epsilon_Theta/s_A<=s_B/2`, the projected basis retains rank K and its row space is exactly the common observed K-space. Thus exact designed-probe decoding applies to the projected response family. A new proof must still account for: both projection errors, normalized-query measurement errors, the projected pair's weakened rank margin when invoking the joint corollary, and the resulting explicit small-noise threshold. This observation is a proposed route, not a substitute for that complete proof.

## Scope of this assessment

The review uses direct symbolic checks, not experimental agreement. The procedural scientific-critical-thinking skill informed the distinction between a proved inverse property, an observational theorem, and solver behavior; it does not validate a mathematical result. Its library attribution is already present in the project bibliography as `kassis2026scientificagentskillslibrary`. No claim of novelty, optimal constants, formal verification, or independent human peer review follows from this audit.

## Second independent review: complete finite-noisy candidate

Reviewed `research/FINITE_NOISY_INDEPENDENT_ROUTE.md`, in its version using `G_p=(W+1_L u_p)/sqrt(2L)` for **every** probe p and applying the joint-product corollary to the **original** factor pairs. This differs materially from the projected-factor corollary application suggested above: the candidate needs no rank margin at all for BP. The following review checks the actual algebra rather than treating the existence of the source note as a proof.

**Verdict:** I find no unresolved mathematical gap in this version of the candidate theorem, subject to the explicit positive domain constants, known positive rates and temperature, and dimensional hypotheses stated there. Its foundational dependency is the separately audited full-operator inverse theorem. The candidate expands that dependency's constants and does not conceal the finite-probe question in an unproved inverse lemma. This is a proof audit, not formal machine verification or a novelty certification.

### Observed-subspace projection

The top-K right singular projector P minimizes `||Theta_hat(I-P)||_F` over rank-K right projections. Since the feasible matrix AB has rank K, its row projector is a valid comparison, giving the desired residual bound. Consequently `B(I-P)=A^dagger AB(I-P)` proves the uniform `2e_theta/s_A` basis bound. Repeated singular values and discontinuous deterministic basis choices do not invalidate this pointwise residual argument. The same projector and probes are used by every pair in the fixed feasible set.

The quadratic response expansion can be written explicitly with `D=B-BP` as

`B^T C_i^2 B-(BP)^T C_i^2(BP) = D^T C_i^2 B+(BP)^T C_i^2 D`.

Both original and projected basis spectral norms are at most M; therefore its spectral norm is at most `2M||D||_F`. Taking the maximum over local blocks yields the candidate's `t=4lambda M e_theta/s_A`. Neither positivity nor full rank of BP appears in these estimates. The projected operator belongs to the broader linear response class even if BP is rank deficient, which is all that Lemma 2 requires.

### Shared-support linear-class norming lemma

Let `c=1/sqrt(2L)`. The orthogonality identities `WW^T=I_L`, `WU^T=0`, and `UU^T=I_K` yield the two stated projection equations exactly. In particular, repeating W in every probe is essential to the `sqrt(K)` gain:

`K c^2 ||Delta Q||_F^2 = sum_p ||E_p W^T||_F^2 <= sum_p ||E_p||_F^2 = E^2`.

The stacked local-coordinate term contains every row of every `Delta T_i` exactly once, so its norm is `sqrt(sum_i||Delta T_i||_F^2)`. The stacked shared term has norm `sqrt(K)||Delta Q 1_L||_2`. The triangle inequality and `||Delta Q1_L||<=sqrt(L)||Delta Q||_F` give

`sqrt(sum_i||Delta T_i||_F^2) <= E/c+sqrt(KL)||Delta Q||_F <= (1+sqrt(L))E/c`.

For an arbitrary full ambient gradient G, right projection to U is contractive, its embedding back through U is isometric, and the layerwise block operator has norm `max_i||Delta T_i||_2`. Hence

`||Delta Rbar|| <= ||Delta Q||_2+max_i||Delta T_i||_2 <= [1/sqrt(K)+1+sqrt(L)]E/c`.

Thus the displayed h is correct. There is no spectral/Frobenius norm confusion, missing factor of K, or requirement for symmetry, PSD local blocks, or factor full rank in this norming result. It holds on a larger linear class than the factor-induced class, which is legitimate.

### Finite residual to complete response

Each original-to-projected response contributes at most t on each unit-Frobenius probe. Its K-response stack norm is therefore at most `sqrt(K)t`. The pair contributes twice that amount, in addition to the original feasible-pair response-stack discrepancy `2e_R`. Applying the norming lemma and adding the two full-operator projection errors proves

`||R_x-R_x'|| <= 2h e_R+2(h sqrt(K)+1)t`.

Substituting t gives exactly `beta=(8lambda M/s_A)(h sqrt(K)+1)`. No small-noise or true-subspace-membership hypothesis was inserted at this stage.

### Complete constant substitution and global step

For the original domain take router norm bound `sqrt(L)` and basis Frobenius bound M. The corollary's enlarged domain is precisely `(alpha/2,s_A/2,s_B/2,2sqrt(L),2M)`. Direct expansion gives:

- `L_M=max(1,4sqrt(L)/s_A)`;
- `c_G=4L/(eta_B s_A^2)` and `c_H=8/(lambda s_B^2)`;
- `mu=(alpha/2)(1+1/(4L_M^2))`;
- `c_J=L_M^2(c_G+c_H)/mu` and `c_D=2sqrt(LK)c_J/s_A`;
- `c_row=sqrt(K c_D^2+(c_G+K c_D^2)^2)`;
- `delta_0=min(1,1/(2c_G),1/(2sqrt(K)c_row))`;
- `C_local=2sqrt(L+L_M^2 M^2)sqrt(K)c_row`;
- `C_star=max(C_local,4sqrt(L+M^2)/delta_0)`.

These exactly match the candidate's definitions through its aliases. In the separate forward-response Lipschitz constant, the enlarged basis bound `2M` produces `6lambda(2M)^2=24lambda M^2` and `2lambda(2M)=4lambda M`, also matching the candidate. The original-domain displacement coefficient remains `kappa=sqrt(s_B^{-2}+4s_A^{-2})`; it must not be recomputed with the enlarged margins, and the candidate correctly does not do so.

Every pair in the feasible set has product discrepancy at most `2e_theta`. The candidate threshold `min(alpha s_B/4,s_A s_B/8)` therefore gives exactly the hypothesis of the joint-product corollary on the **original** pair. The norming projection is only an analysis device; there is no illicit application of the rank-conditioned inverse to BP. Combining the corollary with the finite-response bound yields the stated coefficients `C_R=2hC_star` and `C_theta=C_star beta+2kappa(1+C_star L_R)`.

The theorem is uniform over arbitrary pairs in the whole feasible set. The inherited large-response-error diameter branch remains present in C_star, so no small e_R assumption or local connected-component membership is needed. The finite group quotient triangle inequality used by the joint corollary has already been checked above. Empty and singleton sets are explicitly covered by convention. K=1 is handled separately and correctly, since A is fixed and product error alone bounds B discrepancy by `2e_theta/sqrt(L)`.

### Remaining limitations, not proof gaps

The theorem establishes an information property conditional on a specified admissible domain and bounded errors. It does not establish that an arbitrary numerical optimizer finds a feasible point, that constants are practically useful, that the probe count is optimal, or that domain margins can be certified from a fitted candidate. None of those stronger claims is required in the approved goal or used in this proof. The measurement-layer result is a new composition here, but the global algebraic inverse machinery is inherited; the candidate's restrained novelty classification is appropriate.
