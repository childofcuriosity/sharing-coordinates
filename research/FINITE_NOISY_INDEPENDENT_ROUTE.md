# Independent finite noisy probe route

Status: a proved finite-observation-to-full-operator bridge, composed with the repository's explicit full-operator inverse theorem. The composition is global over the feasible set, not a local candidate certificate. The existing inverse theorem and its joint-product extension were read line by line; their algebra is consistent with this composition. This note is not a claim of an optimal constant or a new general inverse-function theorem.

## Setup and explicit probes

Write lambda = eta_Z/tau^2 > 0. All matrix errors below are Frobenius errors; response-operator errors use the induced F-to-F norm. K >= 2, L >= K, D-K >= L. Let P be the orthogonal projector onto *any fixed deterministic choice* of the top K right singular vectors of the observed product Theta_hat. Write U for the K-by-D orthonormal-row matrix and u_p for its p-th row. Choose W in R^{L-by-D} with WW^T=I_L and WU^T=0. Such W exists because D-K >= L. Neither selection uses any unknown factor. Singular-value ties can be resolved deterministically; no gap is needed in the argument.

Set c=1/sqrt(2L) and, for p=1,...,K,

    G_p = c (W + 1_L u_p).

Orthogonality gives ||G_p||_F^2=c^2(L+L)=1. The probes are fixed once Theta_hat is observed, before looking at responses or at candidate factors.

## Lemma 1: every feasible factor lies close to the observable subspace

For any feasible (A,B), let Bbar=BP. Since AB has rank K and ||AB-Theta_hat||_F <= e_theta, the best-rank-K approximation property gives

    ||Theta_hat(I-P)||_F <= e_theta,
    ||AB(I-P)||_F <= 2 e_theta,
    ||B-Bbar||_F <= 2 e_theta/s_A.

The last inequality uses B(I-P)=A^dagger AB(I-P). This argument is uniform over all feasible pairs; it neither assumes the true row space is known nor assumes it equals the selected subspace. No full-rank property is required of Bbar.

Let C_i=Diag(a_i)-a_i a_i^T. For simplex a_i, ||C_i||_2 <= 1, and ||Bbar||_2 <= ||B||_2 <= M. Therefore

    ||B^T C_i^2 B - Bbar^T C_i^2 Bbar||_2
      <= 2 M ||B-Bbar||_F.

The Gram response term is unchanged. The remaining change is a block diagonal operator, so

    ||R_(A,B) - R_(A,Bbar)||_(F->F)
      <= t := 4 lambda M e_theta/s_A.                     (1)

## Lemma 2: K explicit probes stably norm the shared-subspace family

Consider any two operators of the form

    Rbar(G)=Q G + local_T(G U^T) U,

where Q is an L-by-L matrix, each T_i is a K-by-K matrix, and local_T acts on row i by right multiplication by T_i. Symmetry is not needed here. Set E_p=Delta Rbar(G_p), and E=(sum_p ||E_p||_F^2)^(1/2). Projection gives

    E_p W^T = c Delta Q,

hence ||Delta Q||_F <= E/(c sqrt(K)). Projection onto U gives

    E_p U^T = c Delta Q 1_L e_p^T
                  + c [row i = e_p^T Delta T_i].

Consequently, after stacking p,

    (sum_i ||Delta T_i||_F^2)^(1/2)
      <= E/c + sqrt(K)||Delta Q 1_L||_2
      <= (1+sqrt(L)) E/c.

For every G,

    ||Delta Rbar(G)||_F
      <= (||Delta Q||_2 + max_i ||Delta T_i||_2) ||G||_F.

Thus

    ||Delta Rbar||_(F->F) <= h E,
    h = sqrt(2L) (1/sqrt(K) + 1 + sqrt(L)).                (2)

For actual factors Q=eta_B AA^T and T_i=lambda U Bbar^T C_i^2 Bbar U^T. Rate constants need not enter h because the lemma norms the operator itself.

## Lemma 3: observable finite residuals control the original full operators

Take arbitrary two feasible factor pairs x,x'. Their response stacks differ by at most 2 e_R. Equations (1), ||G_p||_F=1, and the triangle inequality yield

    ||stack Delta Rbar(G_p)||_F <= 2 e_R + 2 sqrt(K) t.

Applying (2) and adding the two original-to-projected operator errors gives

    ||R_x-R_x'||_(F->F)
      <= 2 h e_R + beta e_theta,
    beta = (8 lambda M/s_A) (h sqrt(K)+1).                (3)

All pairs use the same U,W, probes. No local branch, unknown Jacobian, oracle row space, or probe-conditioning hypothesis appears. Projection is an analysis device; the final pair remains the original pair in the original ambient D-dimensional space.

## Global diameter theorem and all constants

Let the feasible set be the one in the user-approved goal, with A entries >= alpha, row sums one, singular values bounded below by s_A,s_B, and ||B||_F <= M. Assume

    e_theta <= e_0 := min(alpha s_B/4, s_A s_B/8).

Define a_star=alpha/2, a_s=s_A/2, b_s=s_B/2, L_A=2sqrt(L), L_B=2M, and

    L_M=max(1,L_A/a_s),
    c_G=L/(eta_B a_s^2),
    c_H=2/(lambda b_s^2),
    mu=a_star (1+1/(4L_M^2)),
    c_J=L_M^2(c_G+c_H)/mu,
    c_D=sqrt(LK)c_J/a_s,
    c_row=sqrt(K c_D^2 + (c_G+K c_D^2)^2),
    delta_0=min(1,1/(2c_G),1/(2sqrt(K)c_row)),
    C_local=sqrt(L_A^2+(L_M L_B)^2) sqrt(K)c_row,
    C_star=max(C_local,2sqrt(L_A^2+L_B^2)/delta_0),
    kappa=sqrt(s_B^(-2)+4s_A^(-2)),
    L_R=sqrt((2eta_B sqrt(L)+24lambda M^2)^2+(4lambda M)^2),
    C_R=2 h C_star,
    C_theta=C_star beta+2 kappa(1+C_star L_R).

Then

    diam_(d_perm)(F) <= C_theta e_theta + C_R e_R.         (4)

There is no smallness assumption on e_R. Empty or singleton feasible sets satisfy the diameter bound under the usual zero-diameter convention.

Proof: any two feasible pairs have product difference <= 2 e_theta. To avoid any reliance on a local connected component, construct Atilde=P_A A', Btilde=Atilde^dagger AB. Exactly as in the existing joint-product proof,

    ||Atilde-A'||_F <= (2e_theta)/s_B,
    ||Btilde-B'||_F <= (4e_theta)/s_A.

The e_0 condition ensures positivity >=alpha/2, singular values >=s_A/2,s_B/2. Row sums are preserved because 1_L lies in col(A). The intermediate pair has exactly product AB and lies in the enlarged domain used above. The existing explicit same-product inverse theorem therefore applies with C_star. The straight segments between the two valid simplex router matrices remain in the simplex; basis norms remain <=2M. The response is L_R-Lipschitz on this region, by ||C(a)-C(a')||_2<=3||a-a'||_2. The triangle inequality gives

    d_perm(x,x')
       <= C_star ||R_x-R_x'||_(F->F)
          + kappa(1+C_star L_R)||AB-A'B'||_F.

Insert (3) and the product bound to obtain (4). This holds for each arbitrary pair in F, hence for its entire diameter. No supposition that any candidate is close to truth was used. QED, contingent only on the separately stated and proved full-operator inverse theorem, whose constant has been expanded above rather than hidden.

## Existing-theorem dependency audit

I read the existing same-product proof in paper/sections/appendix_theory.tex. The full-operator norm bounds each response block; singular positive semidefinite local blocks separate the scalar Gram diagonal. Equal products imply A'=AN, B'=N^{-1}B with N1=1. Gram proximity bounds NN^T-I. The shared kernel span{1} permits a *linear*, rather than generic square-root Holder, covariance perturbation bound via the Sylvester integral on 1-perp. Router full rank then separates the diagonal algebra. Small-error row rounding gives a permutation; large errors are handled by the bounded domain diameter. The claimed constant explicitly includes that large-error case, which is essential for using it globally here. The joint-product extension's projected router preserves row sums and positivity at the stated threshold. I found no additional local-branch premise in either proof.

## What this does and does not establish

This proves an information property of a specified query protocol. A true domain member whose errors satisfy the declared bounds lies in F, so every feasible estimator has factor error at most (4). It does not furnish an efficient algorithm to find a feasible point, guarantee feasibility for misspecified data, establish near-optimal constants, or apply to passive task gradients or unknown AdamW states. Constants are dimension-explicit but large. New mathematical work here is the uniform finite-noisy bridge through the observed product subspace and the consequent global diameter theorem; the general algebraic inverse machinery is inherited and must be cited as such.

## K=1 boundary case

When K=1, row stochasticity fixes A=1_L, hence C(a_i)=0. For any two feasible pairs, ||B-B'||_F=||AB-AB'||_F/sqrt(L)<=2e_theta/sqrt(L). One may choose the single probe G_1=0 (norm at most one), or the same displayed normalized probe if desired. Thus C_theta=2/sqrt(L), C_R=0, with no small-noise threshold. The response carries no extra basis information in this case. The K>=2 constants above should not be used to obscure this trivial boundary case.

## Novelty classification

The result should be called an explicit measurement-layer corollary of the existing global full-operator inverse theorem. Its content is that exactly K normalized probes chosen from the noisy product suffice, with a fully quantified error transfer uniform over the whole feasible set. It is not an independent general inverse theorem, and no novelty or optimality claim about K probes follows from this proof alone.
