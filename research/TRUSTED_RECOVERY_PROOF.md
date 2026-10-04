# Observable local separation for finite noisy responses

This is a self-contained specialization of standard Jacobian/Taylor local
inverse reasoning, not a new generic inverse theorem or a global solver
guarantee. Rates and temperature are fixed to one in the experiment.

## Exact compression, with all residuals preserved

Fix the orthonormal rows U of the estimated rank-K right singular subspace
of the noisy product. Candidates have B=C U. Set X_p=G_p U^T,
Gperp_p=G_p-X_p U, Y_p=Robs_p U^T, Rperp_p=Robs_p-Y_p U.
With Q=A A^T and S_i=diag(a_i)-a_i a_i^T,
the predicted parallel responses are Q X_p + [C^T S_i^2 C X_{p,i}]_i,
and the orthogonal responses are Q Gperp_p.

Let H=sum_p Gperp_p Gperp_p^T, T=sum_p Rperp_p Gperp_p^T.
When H is positive definite, Qls=T H^{-1}. Orthogonality of the least-squares
residual gives, for every Q (including the restricted Q=A A^T),

sum_p ||Q Gperp_p-Rperp_p||_F^2
 = ||Q H^{1/2}-T H^{-1/2}||_F^2
   + sum_p ||Qls Gperp_p-Rperp_p||_F^2.

Similarly ||A C U-Thetaobs||_F^2
 = ||A C-Thetaobs U^T||_F^2 + ||Thetaobs(I-U^T U)||_F^2.
Normalize the two product/response blocks by their observed Frobenius norms.
The implemented objective includes both constant terms. The compressed
candidate-dependent map has exactly the same differences and Jacobian
singular values as the full restricted map: the compression is an isometry
on candidate-dependent differences, not a removal of inconvenient residuals.

A subspace estimated from noisy Theta is generally NOT the exact row space
of the true B. Therefore small compressed local bounds need not bound full
true-factor error. Independent scoring includes ||Btrue(I-U^T U)||_F^2.

## Candidate-centered coordinates and curvature

At a strictly simplex-interior candidate (A0,C0), write
A=A0+sA Z V^T, C=C0+sC W, where V is an orthonormal basis for
the zero-sum subspace, sA=||A0||_F, sC=||C0||_F, and x=vec(Z,W).
The local metric is the combined relative distance using candidate norms;
it is not identical to evaluation distance using truth norms.
For r <= min(A0)/(2sA), the entire radius-r ball is strictly feasible.
Let b=sC(1+r), c=min(1,max_i ||S_i(A0)||_2+3sA r).

Along this ball, ||S_i||_2 <= c, ||DS_i[h]||_2 <= 3||h||_2,
and ||D^2S_i[h,k]||_2 <= 2||h||_2 ||k||_2.
For E_i=S_i C, differentiating E_i^T E_i twice gives a bilinear
Hessian bound represented by the symmetric nonnegative 2x2 matrix

  [(18+4c)b^2 sA^2, 12 c b sA sC;
    12 c b sA sC,     2 c^2 sC^2].

Its spectral norm equals its largest eigenvalue (nonnegative entries and
symmetric matrix; the negative eigenvalue cannot have larger magnitude).
Call this h. Including the A A^T G term, the response Hessian norm is
at most (2sA^2+h)*sqrt(sum_p ||G_p||_F^2)/||Robs||_F.
A conservative product Hessian bound is 2sA sC/||Thetaobs||_F.
Their Euclidean combination M(r) bounds the measurement Hessian on the ball,
hence the Jacobian Lipschitz constant. This also applies in full coordinates
with C replaced by B, independent of the subspace compression.

## Conditional local feasible-component proposition

Let F be the normalized measurement map, y its observations, rho=||F(0)-y||,
and mu=sigma_min(DF(0))>0. Constant residual coordinates can be appended to
F and y without changing derivatives. Suppose ||D^2F||<=M throughout the
closed radius-r ball, and mu-Mr/2>0. For every ||x||<=r,

  ||F(x)-F(0)|| >= mu||x|| - M||x||^2/2
                    >= (mu-Mr/2)||x||.

Proof: integrate the Jacobian along t x, bounding the Taylor remainder
by M||x||^2/2; then apply the smallest-singular-value inequality.

For an observation noise upper bound nu, define the feasible set
S={x: ||F(x)-y||<=nu} over the simplex-affine domain. If rho<=nu and

  rho+nu < mu r - M r^2/2,

then the connected component of S containing zero lies strictly inside
the radius-r ball. Indeed, no point of S lies on that sphere by the triangle
inequality and the displayed lower bound. A connected set containing zero
and a point outside the sphere would have a connected norm image crossing r.
Every point in this component consequently satisfies

  ||x|| <= (rho+nu)/(mu-Mr/2).

This argument also directly bounds any feasible point already known to lie
inside the ball, without requiring a connectedness assertion.

The implementation starts from the feasibility radius r0, chooses
r=min(r0,mu/M(r0)), and recomputes M(r), so M(r)r<=mu because the bound is
nondecreasing in r. Candidate norms, residuals, and singular values are
observable. With both relative product/response noise <=sigma<1,
nu=sqrt(2)*sigma/(1-sigma) is an observable normalized upper bound.
The separate numerical floor used in empirical scores is NOT included in
this theorem's noise budget.

## What is not proved

- No exclusion of distant feasible components or permutation copies.
- No demonstration that the true factors belong to this component.
- No guarantee that a restricted noisy-subspace model contains the truth.
- No solver convergence or global optimality assertion.
- Floating-point SVD/curvature values are not outward-rounded enclosures.
- A calibration threshold is an empirical heuristic, not a coverage/risk
  theorem. Correlated noise variants are not independent replications.
- An all-reject diagnostic is not a useful recovery procedure.

## Prior-art boundary and sources checked 2026-09-25

Afsari, "Sensitivity Analysis for the Problem of Matrix Joint
Diagonalization", SIAM J. Matrix Anal. Appl. 30(3), 2008:
https://isr.umd.edu/Labs/ISL/ICA2006/Sensitivity_Final.pdf
Sections 1 and 4 explicitly connect approximate diagonalization sensitivity
with conditioning and uniqueness. We do not claim that connection as new.

Behling, Goncalves and Santos, "Local convergence analysis of the
Levenberg-Marquardt framework for nonzero-residue nonlinear least-squares
problems under an error bound condition", author preprint, 23 August 2018:
https://optimization-online.org/wp-content/uploads/2018/08/6779.pdf
Section 2, equations (5)-(6), gives the standard Lipschitz-Jacobian Taylor
remainder used here. Their solver convergence results have different
hypotheses and are NOT invoked as a guarantee for our SciPy solver.

The specialized contributions here are the response-model compression,
explicit candidate-scale curvature accounting, and a bounded experiment
separating local numerical separation from empirical failure rejection.
