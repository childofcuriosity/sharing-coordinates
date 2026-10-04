# Adversarial audit: finite noisy observations and the whole feasible set

Status: independent analytical audit, 2026-09-25. This document does not claim an impossibility result. No numerical experiment is used as proof.

## Target and quantifiers

The desired statement is existential in probe design, universal in the observed product and in all pairs of feasible factors. The design may depend on the observed product and declared dimensions/constants, but not on true factors. Noise is adversarial within the stated norm bounds. A single failing design, a failing numerical algorithm, or a failure of a local certificate cannot refute this statement.

The output responses are full L-by-D matrices. A count of K probes is therefore not a count of K scalar observations. Scalar-information counting is not a valid obstruction here.

## Main finding

No obstruction was found under the stated fixed interior, rank and norm margins and D-K >= L. The existing packed construction can be made uniformly robust to product noise. The relevant object is the top-K projector of the observed product, not a continuously chosen singular-vector basis. Repeated singular values within the selected subspace do not obstruct the proof.

Write m=s_A s_B. For any feasible product Theta=AB and a top-K right singular projector P_hat of Theta_hat,

    ||Theta (I-P_hat)||_2 <= 2 epsilon_Theta.

Indeed, Theta is a rank-K competitor for the best rank-K approximation of Theta_hat; its residual in spectral norm is at most epsilon_Theta. A triangle inequality supplies the displayed factor 2. Since row(Theta)=row(B), Theta^dagger Theta=P, and ||Theta^dagger||_2 <= 1/m,

    ||P(I-P_hat)||_2 <= 2 epsilon_Theta/m.

Equal projector ranks imply ||P-P_hat||_2 equals this sine-of-angle norm. This is uniform over every feasible factor pair. It does not assume that the true B lies in the estimated subspace.

For H_i=B^T C(a_i)^2 B, H_i=P H_i P and ||H_i||_2 <= M^2. Consequently

    ||H_i-P_hat H_i P_hat||_2
      <= 2 M^2 ||P-P_hat||_2
      <= 4 M^2 epsilon_Theta/m.

The Gram part eta_B AA^T G does not change. Compressing only the local response blocks therefore produces a common-support structured operator at a uniformly bounded operator-norm cost. It need not itself arise from a valid factor pair. That is harmless: the finite-observation reconstruction lemma applies to this larger linear structured class; the existing factor theorem is applied only after bounding differences of the original valid operators.

This route avoids the invalid claim that a noisy low-dimensional fit automatically contains the truth.

## Potential failure points in a proof

1. **Probe normalization.** The first packed probe has rows v_i+u_1 and Frobenius norm sqrt(2L); subsequent probes have rows u_r and norm sqrt(L). Every response/noise bound must account for division by these norms. An argument using the old unnormalized probes without this adjustment does not prove the proposed target.
2. **Operator norm versus one measurement.** For q unit-norm probes, an operator modeling error rho contributes at most sqrt(q) rho to the stacked response norm. Omitting sqrt(q) is invalid in general.
3. **Two feasible explanations.** Product and stacked-response discrepancies between two feasible pairs are bounded by twice the respective observation budgets. A single-candidate error budget cannot be inserted directly into a pairwise stability theorem.
4. **Projector versus basis.** Singular-vector bases can rotate or flip signs. The proof must be invariant under such choices, or specify a deterministic tie-breaking convention for defining the actual queries. No Lipschitz singular-vector basis is needed.
5. **Artificial support compression.** Do not apply the nonlinear factor stability theorem directly to the compressed structured operators. They need not correspond to valid softmax factors. First obtain a bound for the original full operator difference by triangle inequalities.
6. **Adaptive observations.** All responses must be measured at the probes computed from the same observed product. The deterministic bounds allow arbitrary noise dependence on the product and design, so independence is unnecessary. Responses at probes selected from an unavailable clean product would change the experiment.
7. **Uniform feasible-set conclusion.** A reconstruction algorithm's local solution cannot stand in for the whole feasible set. The projector argument above is pairwise universal because each feasible product is within epsilon_Theta of the same Theta_hat.
8. **Smallness conditions.** When using the existing joint-product theorem, its threshold applies to pairwise product discrepancy 2 epsilon_Theta. In particular its conditions become epsilon_Theta <= min(alpha s_B/4, s_A s_B/8). Any other threshold arising from a chosen proof must also be listed.
9. **Constants.** A citation to an unspecified inverse constant is insufficient. Existing same-product constants may be used only if displayed and the enlarged domain substitutions are explicit. Their potentially severe conditioning should not be marketed as a useful numerical accuracy prediction.
10. **Novelty.** This route is a composition of explicit structured tomography, elementary subspace perturbation and a previously proved global factor inverse. It solves the stated missing observation-to-factor implication if completed. It should not be presented as a new general theory of inverse problems or as an estimator guarantee.

## Edge cases and attempted obstructions

- K=1: row stochasticity fixes A=1_L. For any two feasible products, sqrt(L)||B-B'||_F <= 2 epsilon_Theta. Thus the full feasible-set diameter is already at most 2 epsilon_Theta/sqrt(L), with no response needed. Do not invoke a K>=2 theorem here.
- Empty parameter domains: K>L is incompatible with the required full column rank of A. For K>1, alpha>=1/K is incompatible with positive s_A (alpha=1/K forces every row to be uniform). M<sqrt(K)s_B is incompatible with the B rank margin. A universal statement on an empty domain is formally vacuous, not a recovery result. A nonempty feasible set should be an explicit condition when interpreting the diameter.
- Noisy matrices may have rank greater than K: use the top-K projector, not the full row space of Theta_hat. With a nonempty feasible set, sigma_K(Theta_hat)>=s_A s_B-epsilon_Theta and sigma_{K+1}(Theta_hat)<=epsilon_Theta; epsilon_Theta<s_A s_B/2 ensures separation at the cut. Internal multiplicities do not matter.
- Signed basis symmetry: (A,B) and (A,-B) have identical responses, so responses alone cannot identify the factors. The product observation rules out this ambiguity at the required small noise because ||AB||_2>=s_A s_B. This is not a counterexample to the joint observation target.
- Boundary or rank degeneration can destroy uniform stability, but a sequence with alpha, s_A or s_B tending to zero violates the fixed-domain quantifier. Such examples can establish necessary constant dependence, not refute the stated theorem.
- Constant-factor gauges preserving AB are precisely the difficult alternatives controlled by the existing same-product theorem. If the theorem were false, a counterexample would have to survive its interior/rank margins. Reading its Gram separation, covariance Sylvester inverse, approximate diagonal algebra and permutation rounding revealed no immediate invalid step.
- An all-reject diagnostic or a failed least-squares estimator says nothing about the existence of this deterministic diameter bound.

## Audit boundary

This audit supports feasibility and identifies a proof route; it is not by itself the final theorem. Final acceptance requires line-by-line checking of the chosen normalized finite-probe inverse inequality, expanded constants, joint-product substitutions and the pairwise feasible-set conclusion. No universal impossibility claim is warranted by the attempted obstructions above.

## Audit of the candidate proof communicated after the initial audit

The candidate uses a sharper Frobenius estimate and a symmetric packed design. Both survive independent checking.

Let U have orthonormal rows spanning a top-K right singular subspace of Theta_hat and put P=U^T U. Choose W in R^{L x D} with WW^T=I_L and WU^T=0, possible because D-K>=L. Let u_p be row p of U, viewed as a column. Define every probe by

    G_p = (W + 1_L u_p^T)/sqrt(2L),  p=1,...,K.

The two terms are orthogonal in Frobenius inner product and each has squared norm L, hence every probe has norm exactly one. The design uses only the observed product, dimensions and fixed choices of orthonormal bases.

For any feasible pair, Eckart–Young in Frobenius norm gives

    ||Theta_hat(I-P)||_F <= epsilon_Theta,
    ||AB(I-P)||_F <= 2 epsilon_Theta,
    ||B(I-P)||_F <= 2 epsilon_Theta/s_A.

Set B_bar=BP. Using ||B_bar||_2<=||B||_2<=M and ||C(a_i)||_2<=1 gives

    ||R_{A,B}-R_{A,B_bar}||_{F->F}
       <= 2 lambda M ||B-B_bar||_F
       <= b := 4 lambda M epsilon_Theta/s_A.

No rank bound on B_bar is invoked. This estimate improves the initial projector route and avoids needing any explicit principal-angle theorem.

### Normalized finite-probe inverse

Consider an arbitrary common-support structured operator S(G)=Gamma G+rowwise(H_i g_i), with H_i=P H_i P. Neither positivity nor factor realizability is necessary. Write E=(sum_p ||S(G_p)||_F^2)^{1/2} and Y_p=sqrt(2L) S(G_p). Then

    Y_p W^T = Gamma,
    ||Gamma||_F <= sqrt(2L/K) E.

In the factor model Gamma includes eta_B: Gamma=eta_B AA^T. Calling Gamma Q without this convention would require an extra eta_B in the reconstruction formula.

Projecting Y_p to U coordinates and subtracting (Gamma 1_L)e_p^T recovers column p of each H_i in that basis. Stacking all recovered columns gives

    (sum_i ||H_i||_F^2)^{1/2}
       <= sqrt(2L) E + sqrt(K)||Gamma 1_L||_2
       <= sqrt(2L)(1+sqrt(L)) E.

Thus

    ||S||_{F->F} <= h E,
    h=sqrt(2L)(1/sqrt(K)+1+sqrt(L)).

For two arbitrary original feasible explanations, their measured stack difference is at most 2 epsilon_R. Each support-compression error contributes at most sqrt(K)b to that stack. Therefore

    ||R_1-R_2||_{F->F}
       <= 2 h epsilon_R + 2 b(h sqrt(K)+1).

The two final b terms are necessary because the desired difference concerns the original operators, not their support-compressed versions.

### Composition with the global joint-product theorem

Let kappa=sqrt(s_B^{-2}+4s_A^{-2}), and let C_* and L_R be the previously established explicit joint-product constants evaluated on the enlarged domain. The pairwise product discrepancy is at most 2 epsilon_Theta. With

    epsilon_Theta <= min(alpha s_B/4, s_A s_B/8),

composition yields

    d_perm <= C_Theta epsilon_Theta + C_R epsilon_R,
    C_R = 2 C_* h,
    C_Theta = 8 C_* lambda M(h sqrt(K)+1)/s_A
                  + 2 kappa(1+C_* L_R).

These coefficients were independently checked, including the two-feasible-pair factors. There is no small-response-noise assumption: the older global theorem uses a diameter fallback for larger response discrepancies.

### Quantifier verdict

The construction and compression bound apply to every factor pair in the same feasible set, including distant components. The proof does not choose a component, fit a model, assume a nearby true solution, or assume the estimated row space contains any true B. The linear common-support inverse is applied only to support-compressed operators; the nonlinear global factor theorem is applied only to original valid factors. This separation is essential and is respected by the candidate route.

A boundary tie among singular values does not invalidate Eckart–Young for any selected top-K minimizer. Moreover, with a nonempty feasible set and the stated threshold the K-th and (K+1)-th singular values are separated anyway. Internal repeated singular values only change the basis and hence the probes; the constants are uniform over those choices. K=1 should still be handled by the direct product-only result above.

Verdict: no counterexample or gap found in the communicated candidate route. This is a conditional review of that exact proof structure, not a claim that every later transcription is automatically correct. The final written theorem must display the previously defined constants or give fully explicit substitutions; an unexplained C_* would not satisfy the agreed goal.

## Second review: complete independent-route document

Read the complete `research/FINITE_NOISY_INDEPENDENT_ROUTE.md` line by line after the candidate discussion. Reviewed SHA-256: `9c43d8e2ada39f7c361f0548dfe04087dfc3f9b6014690ca0ec745464bbaa07b`. Dependencies examined: `paper/sections/appendix_theory.tex` at SHA-256 `4ac846c9e423a4d79e1ac5afe365b8afc1242e8c49bb79276ccda5559841bffd`, and `research/JOINT_STABILITY_PROOF.md` at SHA-256 `e42deec39872b6d7deb921d7464912492d89cc0de1c5da68bfde702de01acdc9`.

All constants in the complete route match the candidate computation above. In particular, substituting basis norm 2M into the joint Lipschitz constant gives 24 lambda M^2 in the router coefficient and 4 lambda M in the basis coefficient. The enlarged same-product parameters are substituted consistently. The declared positive rates ensure all denominators are positive. Positivity and both singular margins of the intermediate factorization follow at the non-strict stated threshold; there is no hidden strict-inequality gap.

Additional boundary checks requested by the primary researcher:

- If e_theta=0 but e_R>0, every feasible B has the exact common observed row space, t=0, and the bridge reduces to ||Delta R||<=2h e_R. The factor diameter is C_R e_R, with no local assumption.
- If both budgets vanish, the argument recovers zero permutation-orbit diameter. Distinct permutations are not erroneously treated as distinct identifiable factors.
- For arbitrary Theta_hat, the chosen top-K subspace is defined even if the feasible set is empty. A nonempty feasible set is the only premise needed to derive the rank-K approximation residual bound. No clean-product row space is used in query selection.
- Product noise may be statistically dependent on every response error, and response errors may depend on the probes. Every inequality is deterministic; the proof uses no stochastic independence.
- The result is universal over every pair in F, not just every pair in a selected connected component. The original same-product theorem's diameter fallback removes the last possible large-response-error/locality concern.
- The K=1 paragraph correctly bypasses the K>=2 inverse constants and gives the direct product-only result. A zero probe is allowed by the stated norm-at-most-one budget; if an exactly-unit convention is desired, its suggested normalized probe also exists under the dimensional condition.

Final adversarial verdict on this reviewed version: no algebraic gap, quantifier gap, hidden local hypothesis, or valid counterexample found. The theoretical goal is met as an explicit corollary of the repository's previously proved full-operator inverse theorem, not as a newly independent inverse principle. This review is a mathematical checking record, not a formal proof-assistant certificate. No numerical validation was substituted for proof.

## LaTeX transcription review

Reviewed `paper/sections/appendix_finite_noisy_theory.tex` at SHA-256 `a0131fbaec5618a78d0855e3e00fb6154429670dba19df5eaf43c2f72528762b`. The full-space feasible-set definition, all normalized probe constants, fully expanded inherited inverse constant, shared coefficient Q=eta_B AA^T, pairwise budget factors and K=1 boundary agree with the reviewed independent proof. No mathematical transcription error was found in this appendix version.

The first reviewed main-text version, `paper/sections/theory.tex` at SHA-256 `02ecd6377621fa7b9483b0eb0f0fdf2cfa07e52dea84d066a463ab2f0b6445be`, said that F should "contain every domain member" satisfying the residual bounds. Literally this permits arbitrary supersets, whose diameter need not obey the conclusion. I requested replacing this with "be the set of all domain members" or "consist of all domain members". This is a quantifier repair to the short corollary, not a defect in the appendix proof, which already defines F by equality. I also recommended stating positivity of alpha,M and nonnegativity of the budgets explicitly in the short corollary for self-contained readability. Final main-text acceptance is recorded below after correction.

Final corrected LaTeX acceptance: `paper/sections/appendix_finite_noisy_theory.tex` SHA-256 `edfc698ea8469b97bf259510057661f12a27ad53f7b43574866acefefc463293`; `paper/sections/theory.tex` SHA-256 `e4c2ab1507e5144456b51a696f9aa555b75aeb4fce138f49e3c5632688f937b2`. The main statement now says "consist of all domain members", explicitly assumes alpha>0,M>0, and declares nonnegative error budgets. The appendix's final-constant display was split across lines without changing any expression. These corrected versions pass the mathematical and quantifier transcription audit within the scope stated above. No experiment or typesetting check is part of this acceptance.
