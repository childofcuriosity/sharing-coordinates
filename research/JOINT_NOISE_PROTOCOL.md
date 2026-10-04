# Joint-noise diagnostic grid — 2026-09-25

Prospective synthetic diagnostic, not a theorem validation or population study.
CPU FP64, L=6,K=3,D=16, seeds 0..9, standard normal B and softmax normal A.
Regimes: interior (unchanged), A-rank contraction toward uniform, and boundary
contraction toward cycling one-hot rows. Contraction levels 1,.1,.01,.001;
interior uses only 1. Noise levels 0,1e-8,1e-6,1e-4,1e-2, independently on
Theta and each observed response with prescribed relative Frobenius norm.
At each noisy Theta compute its best rank-K truncation and design the usual
K probes from that estimate. Query the true autodiff response, add noise,
and reconstruct using the estimated geometry. Fixed Gaussian diagonalization
weights seed 520000, no retries or positivity projection; retain failures,
raw rank/positivity/PSD diagnostics and factor errors up to permutation.
Boundary A=level*softmax(normal)+(1-level)*cycling_one_hot remains positive.
Rank contraction A=level*softmax(normal)+(1-level)/K approaches rank one.

A separate analytic argument will examine stability between *valid* nearby
factorizations when both products and responses differ. That inverse-property
statement must not be presented as an error guarantee for the unconstrained,
PSD-clipped reconstruction algorithm or arbitrary noisy observations.
