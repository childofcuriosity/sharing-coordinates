# Full-dimensional automatic-differentiation response audit

Defined 2026-09-24 before the new checkpoint runs. This is a prospective local
analysis plan, not an externally preregistered protocol. Original results and
generation sources remain frozen.

Question: can the K designed interventions reconstruct the structured response
when the observations come from differentiating the complete packed map
`(Z,B) -> softmax(Z/tau) @ B`, instead of the analytic response formula?
This map covers all shared parameters; it is not the end-to-end language-model
loss, an AdamW update, or an identification of training history.

Inputs: all three released `results/checkpoints/language_seed{0,1,2}.pt` files.
Use their full L,K,D dimensions, float64, recorded final temperature, and
`eta_z=eta_b=1`. There is no model training or checkpoint selection.

Observation oracle: reverse-mode differentiation of the linear loss `<Theta,G>`
produces the Euclidean Z/B gradients. A forward-mode `torch.func.jvp` of the
effective-parameter map pushes those rate-weighted gradients forward. The oracle
must not call the analytic response routine or hand-code its softmax Jacobian.

Controls: native chart; cyclic basis permutation; the original positive-stochastic
gauge `M=.65 I+.35 11^T/K`. All charts use a single probe design obtained from
the native effective matrix. Log-probability re-encoding into finite logits is
numerical: record its residual rather than asserting bitwise product equality.

Measurements per chart:

1. Four designed response observations (K probes generally).
2. Agreement with the separate analytic implementation.
3. Router-Gram and local-block reconstruction from the observations.
4. Prediction of two unseen Gaussian probe responses, independently observed by
   the same autodiff oracle, using seeds `910000+training_seed` and
   `920000+training_seed`.
5. Product residual, permutation response control and non-permutation gaps.

Noise sensitivity on the native chart only: add fixed-norm Gaussian observation
noise at relative norms `0,1e-10,1e-8,1e-6,1e-4` independently to each designed
response. The scale is each corresponding noiseless response Frobenius norm.
Use device-local RNG seed `930000+training_seed`, report actual absolute noise
norms, Gram/local reconstruction errors, and the manuscript's deterministic
noise bounds. This is a sensitivity grid, not a calibrated probability model.
Compare noise-induced component errors against the noiseless reconstructed
components; separately report the noiseless reconstruction error versus formula
truth. This separation avoids attributing baseline rounding to injected noise.

Predeclared numerical gates: relative analytic/oracle discrepancy <=1e-9;
unseen-response relative error <=1e-8; permutation gap <=1e-9;
realized-product relative residual <=1e-10. A non-permutation maximum designed
gap >1e-3 is recorded as fixed-alternative separation, not a universal lower bound.
Noise bounds are checked with `1e-8*max(1, norm of noiseless component)` absolute
floating-point slack, recorded explicitly. Any failed gate is retained and
investigated; do not remove seeds or retune the grid to obtain a pass.

Outputs: additive JSON under `results/autograd_tomography/`, source hashes,
checkpoint hash, software/device metadata, wall time and peak allocated memory.
Small synthetic CPU tests must establish oracle/formula agreement, unseen-probe
prediction, and sensitivity to corrupted observations before checkpoint runs.
Run training seeds on separate available GPUs; numerical scheduling is not
multi-agent delegation.

Interpretation: successful runs strengthen the implementation and measurement
evidence for structured operator reconstruction. They neither constitute a
proof, establish practical factor recovery from noisy measurements, nor validate
the theorem for adaptive optimizers.
