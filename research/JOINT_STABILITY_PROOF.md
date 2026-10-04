# Joint product/response stability: derivation and checks — 2026-09-25

Scope: pairs of valid full-rank interior factors with common known positive
Euclidean block rates. This does not certify arbitrary output of the existing
PSD-clipped estimator. Uses the manuscript's existing same-product theorem.

Let both pairs (A,B),(A',B') satisfy min entries alpha, singular lower bounds
s_A,s_B, and Frobenius upper bounds L_A,L_B as in Theorem response_stability,
but allow epsilon=||AB-A'B'||_F>0. Let epsilon <= min(alpha*s_B/2,s_A*s_B/4).
Set P=A A^dagger (orthogonal projection), A_t=P A', B_t=A_t^dagger AB.

1. P 1=1 because A 1=1, so A_t 1=1. Since
   (I-P)A'=(I-P)(A'B'-AB) B'^dagger,
   ||A_t-A'||_F <= epsilon/s_B. Consequently min A_t >= alpha/2,
   sigma_min(A_t)>=s_A-epsilon/s_B>=3s_A/4>=s_A/2.
   Its column space equals col(A), hence A_t B_t=AB exactly.

2. A_t^dagger A_t=I and AB=P AB imply
   B_t-B'=A_t^dagger P(AB-A'B'),
   ||B_t-B'||_F <= 2epsilon/s_A. Consequently
   sigma_min(B_t)>=s_B/2. Both pairs (A,B),(A_t,B_t) lie in the enlarged
   domain alpha/2,s_A/2,s_B/2,2L_A,2L_B. Nonnegative row stochasticity is
   established before using any simplex covariance bounds.

3. Set kappa=sqrt(s_B^{-2}+4s_A^{-2}); combined factor displacement from
   (A',B') to (A_t,B_t) <= kappa epsilon.
   On this enlarged domain let M=2L_B, lambda=eta_Z/tau^2,
   L_R=sqrt((2 eta_B sqrt(L)+6 lambda M^2)^2+(2 lambda M)^2).
   For simplex rows, ||a||_2<=1, ||C(a)||_2<=1 and
   ||C(a)-C(a')||_2<=3||a-a'||_2, so ||C(a)^2-C(a')^2||_2<=6||a-a'||_2.
   The Gram term is bounded by 2eta_B sqrt(L)||delta A||_F; each diagonal
   block by lambda(6M^2||delta a_i||_2+2M||delta B||_F). The direct-sum
   block norm is its maximum, bounded by the global displacement. This gives
   ||R(A',B')-R(A_t,B_t)||_{F->F} <= L_R kappa epsilon.

4. Let C_* be the explicit constant of the same-product theorem evaluated on
   the enlarged domain. Its application gives orbit distance from (A,B) to
   (A_t,B_t) <= C_*(epsilon_R+L_R kappa epsilon). Permutation actions preserve
   the combined Frobenius metric. Triangle inequality yields

   d_perm((A,B),(A',B')) <= C_* epsilon_R + kappa(1+C_* L_R) epsilon.

All dimensions and full-rank inverses have been checked. No assumption of a
common column/row space for the original two products is needed; the projected
intermediate factorization provides the common product. The small-product-
error condition is essential to this argument. Constants are very loose.

Observation corollary: if two valid feasible pairs each fit the same noisy
product and response within e_Theta,e_R, substitute 2e_Theta,2e_R, subject to
the smallness condition. This is conditional identifiability of feasible pairs,
not an algorithmic guarantee that a feasible pair exists or is returned.

Numerical tests exercise non-common product spaces, exact row sums, exact
intermediate product, displacement and response bounds, plus the smallness
condition. Such tests support implementation checks, not proof validity.
