# Balanced sharing decisions: geometry, gradient flow, and the theory–practice gap

Date: 2026-09-25. Additive theoretical investigation; the earlier frozen LLM study is unchanged. These are derivations for the specified model, not a literature-priority claim. All equations use plain text for terminal reading. Numerical certificates are in `results/routing_dynamics/analysis.json`; numbers below use one-based layer indices.

## 1. Three decisions that must not be conflated

There are L = K m layers, K bases, and D effective coordinates per layer. A is L by K, strictly positive with row sum one; B is K by D. The effective weights are Theta = A B. The softmax parametrization is A_i = softmax(Z_i / tau). A balanced assignment H is a zero-one L by K matrix with one entry per row and m entries per column. Let Q_H = H H^T / m, the orthogonal projector that replaces rows by their group mean.

The original experiment maximizes S(H) = <A,H>. This is equivalent to minimizing ||A-H||_F^2: it assigns to fixed simplex vertices subject to capacity. Call it **coefficient assignment**. It is not k-means with refitted centers.

Direct weight clustering minimizes J_Theta(H) = ||(I-Q_H) Theta||_F^2. In router coordinates this is clustering with metric G = B B^T. The existing implementation uses balanced Lloyd iteration and is a heuristic, not a global optimizer.

Router-vector clustering minimizes J_A(H) = ||(I-Q_H) A||_F^2, allowing arbitrary group means. Section 4 gives this third objective an exact instantaneous-response interpretation. Recovering A is a measurement operation; choosing any of these objectives is a separate decision.

## 2. An exact equivalence, and its limit

**Proposition 1 (two bases).** If K=2, group sizes are equal, and b1 != b2, the sets of globally optimal unlabeled partitions for coefficient assignment and weight clustering coincide, including ties.

Proof. Write a_i=(t_i,1-t_i). An assignment is a subset U of m layers allocated to the first basis; its score is a constant plus 2 sum_U t_i, hence top-m sorting is optimal. Also theta_i=b2+t_i(b1-b2). Set s=sum_U t_i and T=sum_all t_i. Minimizing within-group SSE is equivalent to maximizing s^2+(T-s)^2, or |s-T/2|. The largest and smallest feasible s are respectively the top-m and bottom-m sums, and these complementary subsets give the same unlabeled partitions. This argument also characterizes all ties. This is a global-objective result, not a guarantee for arbitrary Lloyd stationary points. It holds along any trajectory while the stated conditions persist.

**Proposition 2 (simplex-isotropic bases).** Let P=I-11^T/K. If P G P=lambda P, lambda>0, then J_Theta(H)=lambda J_A(H) for every H.

Proof. Every row of X=(I-Q_H)A sums to zero, so XP=X. Therefore tr(X G X^T)=lambda tr(X X^T). This equates the two free-center clustering objectives, not coefficient assignment.

For K=3 this distinction is already strict. Let B=[I3,0] with D=9 and

    A = (1/23) * [ 8 11  4
                   8  5 10
                  10  9  4
                   5 11  7
                   6 10  7
                  11  5  7 ].

Coefficient assignment uniquely gives {1,4}, {2,5}, {3,6}, score 60/23; its nearest labeled competitor scores 59/23. Global weight clustering uniquely gives {1,3}, {2,6}, {4,5}, SSE 14/529; the next partition costs 31/529. The coefficient partition costs 41/529. All 90 labeled assignments and 15 unlabeled partitions can be checked with integer arithmetic.

This is an interior, full-rank example: min(A)=4/23, smallest singular value of A approximately .244, and all singular values of B are one. Strict margins give an open neighborhood of disagreement. Mixing A with the uniform matrix by any positive factor preserves both winning partitions; thus merely being close to uniform does not guarantee agreement (the rank margin decreases as that factor tends to zero).

The recorded certificate also embeds the example into K=4, L=8, D=12. Its integer numerator has first six rows [16*N_i+23,23] and last two rows [23,23,23,391], denominator 460; B=[I4,0]. The old pairs persist, with {7,8} added. This matches the study's number of bases without claiming a natural distribution over such examples.

## 3. Actual Euclidean training dynamics

Let ell(Theta) be a differentiable loss, g_i its effective-weight gradient (as a column), and C_i=diag(a_i)-a_i a_i^T. With positive learning-rate constants eta_B, eta_Z:

    Zdot_i = -(eta_Z/tau) C_i B g_i
    adot_i = -(eta_Z/tau^2) C_i^2 B g_i
    Bdot   = -eta_B A^T grad ell
    Thetadot = -R_soft grad ell.

The LD by LD response operator has blocks

    R_soft[i,j] = eta_B (a_i dot a_j) I_D
                 + indicator(i=j) (eta_Z/tau^2) B^T C_i^2 B.

These follow by chain rule through Theta=AB and softmax, including both factor updates. In particular, B changes during the dynamics; treating its metric as fixed generally misses part of the question. These are Euclidean gradient-flow equations. AdamW, optimizer states, stochastic gradients, and weight decay require different equations.

## 4. Which partition actually best preserves instantaneous dynamics?

For fixed hard assignment H, optimize its shared bases with the same eta_B. Its response is R_H = eta_B (H H^T) tensor I_D. Refitting the base values does not change this response.

**Theorem 3 (response-optimal balanced partition).** For Hilbert–Schmidt error of the full operator,

    ||R_soft-R_H||_HS^2 = constant + 2 eta_B^2 D m J_A(H),

where the constant is independent of H. Thus the globally response-optimal balanced partition is exactly global router-vector k-means, not coefficient assignment.

Proof. Put gamma=eta_Z/tau^2 and T_i=B^T C_i^2 B. Direct blockwise expansion gives

    eta_B^2 D ||AA^T-HH^T||_F^2
    + 2 eta_B gamma sum_i (||a_i||^2-1) tr(T_i)
    + gamma^2 sum_i ||T_i||_F^2.

The last two terms do not depend on H because (HH^T)_ii=1. Also ||HH^T||_F^2=Lm and <AA^T,HH^T>=||H^T A||_F^2. Finally J_A=||A||_F^2-||H^T A||_F^2/m. Substitution proves the identity.

Important boundaries: capacities are equal; learning rates match; hard routing is fixed; the metric is full-operator Hilbert–Schmidt, corresponding to isotropic gradient queries. It is not necessarily the optimum for operator norm, actual task gradients, AdamW, or final language-model loss. The router diagonal term may be large but is partition-independent in this specific comparison.

**Less information than full recovery suffices.** The only partition-dependent quantity is the sum of within-group off-diagonal entries of AA^T. Off-diagonal response blocks directly reveal these entries up to eta_B. Hence response-Gram clustering is sufficient for this objective. Full reconstruction of A is not necessary; the recovery paper must not claim otherwise.

**An explicit conflict.** Keep the A above and use B=[diag(4,1,1),0]. The response optimum remains {1,3},{2,6},{4,5}; the weight optimum becomes {1,2},{3,6},{4,5}.

| Partition | Weight SSE | Router SSE |
|---|---:|---:|
| Response optimum | 119/529 | 14/529 |
| Weight optimum | 65/529 | 50/529 |

For eta_B=1, D=9, m=2, choosing the weight optimum increases full squared response error by 1296/529. Choosing the response optimum increases initial squared weight error by 54/529. Neither number alone proves a downstream benefit.

**Theorem 4 (weights alone do not determine the response-optimal decision).** There are two strictly positive, full-rank factorizations with exactly the same Theta and different unique response-optimal partitions.

Proof by certificate. Start with the isotropic example and set

    M=(1/10)*[2 1 7; 4 4 2; 5 2 3],
    A'=AM, B'=M^-1 B.

M is invertible and row-stochastic, so A' remains positive and normalized, and A'B'=AB exactly. For A the unique router k-means optimum is {1,3},{2,6},{4,5}. For A' it is {1,2},{3,6},{4,5}, cost 200/52900 versus runner-up 212/52900. Enumerating integer pair sums verifies uniqueness. By Theorem 3 these are different unique response optima. Any rule given only Theta has identical input in the two worlds, so cannot always choose the unique optimum in both. The Euclidean parameter metric is part of each world: this is not a passive coordinate change with a correspondingly transformed optimizer.

This proves a possible need for response information, not a need for full A recovery or a frequent benefit on language-model tasks.

## 5. A disagreement that gradient flow actually enters

A static factorization counterexample alone does not answer training reachability. The following construction gives a fixed, smooth, bounded-below loss and a transverse crossing during its gradient flow.

Take B0=[I3,0], D=9, and

    A0=(1/92)*[40 24 28
               24 44 24
               32 28 32
               43 37 12
               24 20 48
               37 31 24].

Let labeled assignments p=(1,2,3,2,3,1), q=(1,2,3,1,3,2), and E=H_p-H_q. Exactly these two assignments tie for best coefficient score 238/92; the next distinct score is 228/92. Weight clustering uniquely selects partition p, with cost 506/8464, runner-up 578/8464. These statements have finite integer certificates.

At this point define G0_i=B0^T C_i^2 E_i and fix once and for all

    T=A0 B0-G0,
    ell(Theta)=0.5 ||Theta-T||_F^2.

Use eta_B=eta_Z=tau=1, with Z0=log(A0). At time zero grad ell=G0. Therefore

    d/dt [S_p(A)-S_q(A)] at zero
      = -sum_i ||B0^T C_i^2 E_i||^2 < 0.

Proof of strictness: positivity makes ker(C_i)=span(1); nonzero E_i has row sum zero; B0^T is injective. At least one E_i is nonzero. The derivative is approximately -0.1069216982, but strict negativity follows analytically.

Smooth ODE local existence gives a trajectory on a two-sided interval around zero. Strict gaps to all other assignments and all competing weight partitions persist by continuity, as do positivity and full rank. For sufficiently small negative time, coefficient assignment selects p; for sufficiently small positive time, it selects q. Weight clustering stays at p throughout. Choose any sufficiently close negative-time point as the initial state and run the same fixed loss forward: training enters disagreement from agreement. Backward integration is only a way of specifying that earlier initial state.

This phenomenon is robust to small perturbations: choose two endpoints with strict opposite score signs and preserved remaining margins. Continuous dependence of the flow on initial conditions and the fixed target preserves those endpoint decisions, and hence a crossing between them. It is not confined to starting exactly on a tie surface.

A general local reachability lemma helps delimit the claim. At a positive A and full-row-rank B, any desired tangent velocity v_i with 1^T v_i=0 can be induced by choosing

    g_i=-(tau^2/eta_Z) B^T (BB^T)^-1 (C_i^2)^+ v_i.

Substitution into adot verifies the claim because C_i^2 times its pseudoinverse projects onto the zero-sum tangent space. A fixed squared loss with target Theta-G realizes this gradient at the point. Bdot is induced, not independently prescribed. Thus there is no universal router-direction constraint from this parametrization alone. Natural language losses constrain their available gradients and are not covered by an arbitrary-target typicality assertion.

The numerical ODE trace verifies the construction at times +/- .01 and +/- .1, but the sign-and-continuity argument above is the proof.

## 6. When agreement is stable, rather than inevitable

**Proposition 5 (near-hard balanced clusters).** Suppose a balanced partition c satisfies a_i,c(i)>=1-delta, delta<1/2. Let d be the smallest basis separation, D_B the largest, and epsilon=delta D_B. If d>4 epsilon, coefficient assignment and global balanced weight clustering uniquely recover c.

Proof. The own coefficient strictly exceeds every alternative, so the balanced pointwise best assignment is feasible and unique. Each theta_i lies within epsilon of b_c(i), since it is a convex combination. Within-group distances are at most 2 epsilon and between-group distances at least d-2 epsilon>2 epsilon. For every equal-sized partition SSE is (1/m) times the sum of squared within-group pair distances. A different partition loses and gains the same number of pairs; each new between-group pair costs more than any lost within-group pair. Its SSE is strictly larger. The same argument gives a directly checkable sufficient certificate: min between-group squared distance > max within-group squared distance.

This separation also makes farthest-first seeding choose one point per true group: every unrepresented group is farther from existing centers than any represented group's points are from their own center. Nearest-center assignment is then pointwise correct and balanced. Updated centroids stay inside the corresponding epsilon balls, so subsequent Lloyd steps preserve it.

**Proposition 6 (finite-time margin protection).** Suppose the coefficient winner has labeled score gap Gamma_A>0, and the weight-clustering winner has global SSE gap Gamma_Theta>0, and their unlabeled partitions agree initially. Let X0 be the globally centered initial Theta. If changes obey

    sqrt(2L) ||Delta A||_F < Gamma_A,
    2 ||X0||_F ||Delta Theta||_F + ||Delta Theta||_F^2 < Gamma_Theta,

the two winners remain unchanged.

Proof. Any difference of two assignment matrices has Frobenius norm at most sqrt(2L), giving the first bound by Cauchy–Schwarz. A difference of two orthogonal projections has operator norm at most one. Expanding each competing quadratic cost difference around X0 bounds its change by 2||X0|| epsilon+epsilon^2. Global centering is allowed because all partition projections preserve the all-ones vector, and centering cannot increase the perturbation norm.

Along Euclidean flow assume ||B||_2<=beta and ||grad ell||_F<=M on the interval. Since ||C_i||_2<=1/2 (Gershgorin bound 2a_i(1-a_i)<=1/2), and ||A||_2^2<=L,

    ||Adot||_F <= v_A = eta_Z beta M/(4 tau^2),
    ||Thetadot||_F <= v_Theta = (eta_B L+eta_Z beta^2/(4 tau^2)) M.

Replace the two perturbations above by v_A t and v_Theta t to obtain an explicit sufficient time interval of agreement. This is conditional finite-time stability, not attraction to agreement, and says nothing without the initial margins and interval bounds.

## 7. How grouping errors enter future training trajectories

Let soft and hard effective weights follow the same loss, with

    Theta_s dot = -R_s(t) grad ell(Theta_s),
    Theta_h dot = -R_H grad ell(Theta_h).

Assume the gradient is L_ell-Lipschitz between the compared states, ||grad ell(Theta_s)||<=M, and ||R_s(t)-R_H||_op<=delta throughout the interval. Since ||R_H||_op=eta_B m, set c=eta_B m L_ell. Then

    ||Theta_s(t)-Theta_h(t)||
      <= exp(c t) ||Theta_s(0)-Theta_h(0)||
         + M delta (exp(c t)-1)/c.

At c=0 the last term is M delta t. Proof: subtract the ODEs, splitting the drift into -(R_s-R_H) grad ell(Theta_s) and -R_H[grad ell(Theta_s)-grad ell(Theta_h)]. The norm's upper Dini derivative is bounded by M delta+c times the current distance. Integrating this scalar inequality proves the bound.

If the hard model begins at group means, the initial squared distance is J_Theta. The second term instead depends on response mismatch. Theorem 3 optimizes its instantaneous Hilbert–Schmidt upper bound, not the uniform-in-time delta in this theorem. Small initial operator mismatch alone does not certify small mismatch later.

This is the precise connection between the static and dynamic objectives: they control different terms of trajectory error. The anisotropic example demonstrates a tradeoff. Neither objective is universally optimal for future task loss; the displayed upper bound is not a lower bound or a proof of superiority.

## 8. What existing trained models establish, and what they do not

The added audit is explicitly post hoc. It reads all 27 already-trained checkpoints (9 models, 3 seeds), changes no training, and introduces no new downstream-loss comparisons. Seven checkpoints have different coefficient-assignment and frozen weight-Lloyd partitions: Pythia-160M seeds 0/1, Pythia-2.8B seed 0, Qwen-0.6B seeds 0/1/2, and Qwen-1.7B seed 2. The earlier nine prespecified downstream-decision checkpoints still have matching original partitions. These are different scopes, not contradictory results.

Router-vector Lloyd and weight Lloyd differ on 4/27 checkpoints; these are heuristic outputs, not four certified global conflicts. Only 1/27 passes the sufficient pairwise-separation condition for the coefficient partition. Failure of a sufficient condition is not proof of disagreement.

For Pythia-160M we enumerate all 15,400 unlabeled balanced partitions (12 layers, 4 groups of 3), so local clustering failure can be separated from global objective differences:

| Seed | Coefficient partition excess weight SSE | Global weight vs global response optimum | Weight optimum gap | Router optimum gap |
|---|---:|---|---:|---:|
| 0 | .365105 | Different | .007875 | .014359 |
| 1 | .364774 | Same | .061496 | .047384 |
| 2 | .198803 | Different | .141154 | .106921 |

All three coefficient partitions are suboptimal for weight SSE. Seed 2's coefficient partition is optimal for router SSE (roundoff residual approximately -1e-16). Seeds 0 and 2 provide trained-checkpoint instances of the static/response objective conflict. These finite floating-point enumerations have large positive runner-up margins but are not interval-arithmetic certificates for model bytes. Constructed small cases above additionally have exact integer certificates.

These observations show that the distinction is not limited to hand-written A matrices within this project's training setup. They do not measure prevalence across arbitrary LLM training, show that the optimizer crossed from agreement during training, establish a downstream gain, or explain AdamW trajectories by an Euclidean theorem. No uniform-hardness or separation assumption has been shown to hold for most checkpoints.

## 9. Remaining mathematical gap and next research target

The theory now establishes special-case equivalence, interior counterexamples, a realizable and robust gradient-flow crossing, conditional agreement stability, a response-optimal partition characterization, an information limitation for weight-only decisions, and a trajectory-error bound. It does not establish typicality under natural training.

A next theoretical target is to derive decision-boundary crossing or protection under a specified gradient distribution/optimizer, with explicit assumptions on its projection onto the boundary normal. Such assumptions must be measured or justified for the actual training protocol; arbitrary squared-loss targets cannot substitute for that justification. A task-weighted response objective may be more appropriate than isotropic Hilbert–Schmidt error, but it generally loses the exact router-k-means reduction and must be derived separately. Full recovery must also be compared against the sufficient response-Gram decision rule, not only against weight clustering.

Validation code: `src/routing_geometry.py`, `experiments/analyze_routing_dynamics.py`, and `tests/test_routing_geometry.py`. The analysis JSON records source and checkpoint hashes. This note is an additive research result, not yet incorporated into the frozen manuscript or claimed as an externally reviewed publication.
