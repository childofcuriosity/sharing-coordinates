# Independent response-artifact audit

Audited: 2026-08-23.  This check was run from the released language
checkpoints, not from `results/response_summary.json`.

## Procedure

For seeds 0, 1, and 2, rerun `experiments.run_response_audit` on CPU with the
released checkpoint, the recorded probe seed `740000 + seed`, eight probes,
unit Euclidean block rates, and gauge strength `0.35`.  The independent CPU
audit deliberately used formula-validation seed `750000 + seed`; the
canonical GPU records used the constant formula-validation seed `750000` for
all three checkpoints.  Thus the seed-1 and seed-2 formula checks exercise
different random directions and are tolerance-consistency checks, not exact
cross-device replays of that diagnostic.  The recomputation outputs were
written to the temporary `.audit_response_recompute` directory and were not
substituted for the canonical GPU artifacts.

## Cross-device comparison

The CPU recomputation reproduces the substantive GPU results.  Differences in
the last digits are expected from CPU/GPU linear algebra and finite-difference
evaluation.

| seed | GPU median nonpermutation response difference | CPU recomputation | GPU minimum singular value of A | CPU recomputation | GPU minimum singular value of B | CPU recomputation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5044997862109962 | 0.5044997862109923 | 1.7290647316696526 | 1.7290647316696524 | 27.445178327622700 | 27.445178327622640 |
| 1 | 0.5051310351033897 | 0.5051310351033886 | 1.7290606240764381 | 1.7290606240764383 | 27.339246469611272 | 27.339246469611830 |
| 2 | 0.5057914296078900 | 0.5057914296078824 | 1.7290149091303781 | 1.7290149091303781 | 27.371451682684242 | 27.371451682684330 |

All 24 checkpoint--probe evaluations detect a response change in both runs.
There are eight evaluations per checkpoint; because the deterministic seed
windows overlap, they use 10 unique Gaussian draws rather than 24 independent
draws.
The effective-product residuals remain below `1.77e-16`; permutation-control
relative differences remain below `8.83e-17`.  The worst CPU transformed-chart
finite-difference error is `7.04e-13`, while the released GPU maximum is
`9.57e-13`; both pass the declared tolerance.  Those two maxima are consistency
checks over their respective directions, not a pairwise exact replay.  All
analytic-versus-autograd checks pass at the declared tolerance.

## Interpretation

This is an independent regeneration check for one fixed nonpermutation chart
at three checkpoints.  It does not turn eight probes into a uniform operator
certificate, identify an AdamW trajectory, or identify historical sharing.
The canonical summary remains the source for manuscript numbers because it is
bound to the released generation-time source hashes; the CPU recomputation is
a cross-device consistency audit.

The generation source in both runs includes
`src/response_identifiability.py` at SHA256
`77e2a69c05e8690d735578f4e24389fd697fb6163bed2b9c36800e503ff995e6`.
