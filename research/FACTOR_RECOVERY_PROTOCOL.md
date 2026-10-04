# Constructive factor recovery: derivation and prospective checks

Written 2026-09-24 before executing the implementation below. This is an
internal protocol, not an external preregistration. The aim is to test whether
the response-identification proof supplies a usable reconstruction, rather
than just a uniqueness certificate. Joint diagonalization itself is background
linear algebra, not a proposed additional novelty.

## Inputs and exact construction

Inputs are Theta, its orthonormal row basis U (rows), the measured Gram Q,
the reconstructed local blocks T_i in U coordinates, and lambda=eta_Z/tau^2.
No true A or B is supplied to reconstruction.

Factor Q=X X^T with X of full column rank K, then C=X^dagger Theta U^T.
In exact arithmetic X=A O for some orthogonal O, and C=O^T B U^T is invertible.
Thus H_i=C^-T (T_i/lambda) C^-1 = O^T J_i^2 O. Its PSD square root is
O^T J_i O. With x_i the transpose of row i of X, define
D_i=sqrt(H_i)+x_i x_i^T=O^T diag(a_i) O.

Diagonalize sum_i w_i D_i. Distinct eigenvalues identify the shared eigenvectors
up to signs and permutation; signs cancel in diagonal extraction. Set row i
of Ahat to diag(V^T D_i V), then Bhat=Ahat^dagger Theta. Full column rank
of A makes its columns distinct, so continuous random weights give distinct
combined eigenvalues almost surely. This is an exact-arithmetic statement;
the implementation checks the realized gap and reports numerical residuals.

## Numerical conventions and scope

Symmetrize measured Gram/local pullbacks and clip negative eigenvalues to
zero when taking PSD square roots. Record the discarded negative magnitude;
clipping is a numerical estimator, not a theorem about noisy recovery.
Use the top K positive Gram eigenvalues. Return raw recovered factors without
simplex projection; report positivity, row sums, product residual and joint
diagonalization residual. Degenerate Gram or combined eigenspectrum is an
explicit failure, not a reason to tune weights until success.

Initial synthetic tests: K=2,3,4, L=K+2, D=2K+3, float64; fixed generators
510000+K for factors, weights seed 520000. Use autodiff observations, rates
eta_Z=.7, eta_B=1.3, tau=.8. Compare with truth only after recovery, enumerating
permutations for evaluation. Exact-observation relative factor tolerance 1e-6.
Also test an intentionally degenerate weight vector (all zeros) fails clearly.

After the synthetic checks, a checkpoint experiment may reuse the three
original checkpoints and four full-dimensional autodiff probes. Any such
run must save source/checkpoint hashes, fixed weight seeds, residuals and
failures to new files. Noise experiments must retain failures and cannot be
described as a general noisy-factor-recovery guarantee.

Checkpoint pilot (fixed before execution): native chart only, seeds 0,1,2;
weights 520000 for every checkpoint, no retries. Use original tau and unit
block rates; four fresh autodiff observations. Noise grid 0,1e-10,1e-8,1e-6,
1e-4, with independent normalized Gaussian observation noise seeded 540000+seed.
Report the minimum over permutations of the root-sum-squared relative router
and basis errors. Noiseless error <=1e-5 is the pilot gate; noisy rows are
descriptive, including exceptions. They do not set a new theoretical threshold.
