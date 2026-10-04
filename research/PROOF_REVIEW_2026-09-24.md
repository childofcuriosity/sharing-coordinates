# Scoped proof review, 2026-09-24

Scope: direct rereading of the exact and quantitative proofs in
`RESPONSE_IDENTIFIABILITY.md`, including the displayed constants (R12).
This is an algebraic review, not independent certification or a novelty verdict.
The manuscript synchronization follow-up is recorded below.

## Exact response stabilizer

1. The response formula follows from the differential of row softmax and the
   two Euclidean block gradients. A non-unit temperature contributes tau^-2
   only to the router term; rates and temperature must match across charts.
2. Equal products with both factors full rank give a unique invertible M.
   Unit row sums and full column rank force M1=1.
3. Off-diagonal operator blocks identify off-diagonal router Gram entries.
   Diagonal blocks have a scalar identity plus a singular PSD local term.
   Two singular PSD matrices cannot differ by a nonzero scalar identity:
   either sign would make one positive definite. This also covers D=K,
   since the local term has rank at most K-1.
4. Equal full Gram matrices and full router rank force MM^T=I.
   Pulling the local block back with a basis right inverse and using the
   unique PSD square root is justified only after establishing orthogonality.
5. The rank-one terms cancel. Router rows span the full diagonal algebra,
   so every column of M has at most one nonzero entry. Orthogonality gives
   signed permutations and M1=1 fixes the signs. The converse follows by
   direct permutation covariance.

No gap identified in this chain. Strict positivity is not needed for the
formula-defined closed-simplex exact extension; full ranks still are. The
observable includes both the product and the full response, not one task gradient.

## Quantitative inverse

- Each block norm is bounded by the full induced operator norm. Kernel vectors
  of the singular PSD local terms control the diagonal scalar without a
  cancellation assumption. The Gram Frobenius bound uses n entries per axis,
  hence n rather than sqrt(n).
- Both M and its inverse are bounded by L_A/s_A. Pullback through the
  minimum-norm right inverse of B contributes s_B^-2, as stated.
- Y=M^T J M has exact kernel span(1), because M1=1. It need not be an
  orthogonal conjugate. The proof correctly works on the shared complement.
- J(a) >= alpha times the orthogonal projector onto 1-perp follows by writing
  a=alpha*1+r and applying weighted Cauchy--Schwarz. The stated weaker gap
  alpha/(4 L_M^2) for Y follows from projecting Mx onto that complement.
- The Sylvester inverse integral is used on the complement only; its two
  positive gaps justify a linear bound. Applying a generic PSD square-root
  estimate on the entire space would not give this conclusion.
- Extracting off-diagonal entries is contractive in Frobenius norm, not
  spectral norm. The displayed sqrt(K) conversion and sqrt(n)/s_A inversion
  factors account for this distinction.
- The rounding threshold makes the selected signed coordinate matrix within
  1/2 of M, whereas M has minimum singular value at least 1/sqrt(2).
  Repeated selected coordinates are therefore impossible. Unit row sums
  exclude negative signs. The global diameter branch covers larger errors.

No algebraic gap identified in these steps or the displayed constants. The
constants are loose; observed numerical success does not establish practical
conditioning. The exact same-product assumption is essential. The proof does
not yet provide a joint inverse bound when both product and response are noisy.

## Finite-probe and observation-noise review

Read the finite tomography subsection of `paper/sections/appendix_theory.tex`
against `src/response_tomography.py` and the additive autodiff runner.
Full ranks imply row(Theta)=row(B), so the probe design uses observable Theta
and known K only. Under D-K>=L, complement codes separate every Gram column
in the first response. Subtracting that shared response yields each local
block's action on a basis of row(Theta); the block vanishes on its complement.
This establishes operator recovery without presupposing known factors.

The stated noise radii follow from orthonormal projection, the first probe's
spectral norm sqrt(L+1), and the row-sum error bounded by sqrt(L) times the
Gram error. Projection in the implementation only improves those upper bounds.
The generic Gaussian result additionally needs q(D-K)>=L and q>=K; its finite
union of full-rank events is valid. Neither result is a minimal-query claim.

The new noise measurements compare against the noiseless reconstructed
components, so they isolate propagation error rather than total error against
exact factors. Numerical roundoff slack is explicit. The summarizer now
recomputes both bounds from recorded noise norms; it does not rerun the JVPs.
Exact Theta and its row space are assumed throughout: noise in Theta is not
covered by these radii. No algebraic discrepancy was identified in this review.

## Secondary appendix follow-up

Read the statements in `appendix_secondary_theory.tex` and their proofs in
`appendix_theory.tex`: local row-sum gauge, entropy and argmax examples,
observational impossibility, anchor recovery, noisy one-hot recovery,
finite-probe clustering, unconstrained/softmax dynamics, saturation and the
symmetric initialization trap. Also checked the partial eta_B=0 argument.

- The local gauge dimension is K(K-1) because H1=0 imposes K independent
  linear constraints and full router rank makes the induced directions
  distinct. Affine independence is correctly preserved by the augmented basis.
- Anchor recovery restricts both competing factorizations to the separable
  class. The noisy one-hot threshold follows from 2*epsilon within-class
  distance and gamma-2*epsilon between-class distance.
- In the finite-probe theorem, inserting delta/4 gives exactly the constants
  32*nu^2/delta^2 and 8*c/delta. The union bound does not need independence
  across layer pairs. Linkage recovery follows inductively from strict
  within/between separation. Added explicit L>=2, 0<eta<1 and at least two
  target classes so the minimum between-class gap is defined.
- The bounded special case does require independence across probe draws;
  this was implicit and is now explicit. Its Hoeffding coefficient 8 is
  correct. `src/probes.py` uses independent Gaussian draws across probes;
  `src/models.py` normalizes both residual branches, consistent with the
  fixed-network bounded-clean-update argument. No population gap or uniform
  Bernstein constants are estimated by the released empirical sweep.
- The saturation bound is pointwise, not a finite-trajectory theorem.
  Corrected the symmetry discussion: shared minibatch randomness alone
  cannot break a symmetry that every sampled product-only loss preserves.
  Parameter-specific stochastic perturbations can do so. The proposition's
  deterministic invariant-manifold calculation itself is unchanged.
- For eta_B=0, the nonnegative-gauge reduction gives simplex columns at the
  maximum pairwise distance, hence distinct vertices. The K=2 subtraction
  and addition identities are consistent. Signed K>=3 remains an explicitly
  unresolved extension, not a missing assumption in the main theorem.

No further algebraic gap identified in these reviewed auxiliary claims.
These checks do not certify the external papers' proofs or historical
experiment execution beyond their available source-bound artifacts.

## Remaining review and status

Follow-up: directly compared the manuscript's exact proof and all four
quantitative steps, including its constants, against the reviewed ledger.
The n-to-L notation change and lambda_Z=eta_Z/tau^2 scaling agree. Also
checked the three degenerating families. For the interior example, the
manuscript previously gave only an O(t^2) upper bound before asserting a
ratio of order 1/t. Added the missing matching lower bound: in row 1, the
two covariance-square traces are 10t^2+O(t^3) and 12t^2+O(t^3).
Trace invariance and |trace(E)| <= 3||E||_2 give a nonzero quadratic gap.
Thus the claimed order follows, rather than merely divergence at least as
fast as 1/t. This is a proof-detail clarification, not a change in theorem.

- Compare the diagonal-algebra step with primary joint-diagonalization and
  robust tensor-identifiability results before deciding the novelty wording:
  the specific theorem comparisons are now recorded in the frontier ledger.
- The secondary theorem statements and their proofs have now been reviewed
  as described above. Random counterexample searches remain diagnostics,
  not proofs. The empirical appendix and citation coverage are separate completed scoped
  checks; see EMPIRICAL_REVIEW_2026-09-24.md and CITATION_COVERAGE.json.
- Human-author review is pending; the preprint disclosure must not imply it
  has already happened.
