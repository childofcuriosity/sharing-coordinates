# Empirical claim and unit audit

Read `paper/sections/appendix.tex` against the strict summaries, relevant raw
record fields, source-attestation checks and the bootstrap implementation.
This review does not rerun the original training or certify historical execution.

| Claim | Checked evidence | Interpretation |
|---|---|---|
| Gaussian response detection | `response_summary.json` protocol: 24 checkpoint/probe evaluations, overlapping windows, 10 unique draws | Fixed alternatives only; not 24 independent random directions |
| Original structured tomography | Three analytic full-dimensional response records with extraction-level formula checks | Kept distinct from the new autodiff measurements |
| New autodiff reconstruction | Three new source-bound records; three charts, four designed and two held-out directions per chart | Additional measurement validation at existing checkpoints, not new training seeds |
| Constructive factor recovery | Three source-bound pilot records; one weight seed and five noise levels | Numerical estimator diagnostics with exact effective products, no general noisy inverse guarantee |
| ASLoRA primary actions | Per-seed adjacent changes 2,6,3; all-pairs changes 4,7,3; fixed ten-gauge grid | 11/30 and 14/30 describe that finite grid, not population frequencies |
| ASLoRA coverage/restoration | 204,208,200 canonical evaluations; required union equals actual; no missing/extra actions; restoration gates pass | First-action audit only; not full training regret or official-code reproduction |
| Byte-LM fold consequence | One successful search, two exhausted searches; all three initialization-dominated | Controlled existence witness, not cross-seed replication |
| Byte-LM effect interval | Seed 0 raw paired batch differences; 256 batches; 10,000 bootstrap resamples in `paired_bootstrap_interval` | Conditional evaluation uncertainty, not uncertainty over training or search |
| Probe diagnostic | `summary.json`: 1,680 grid records per architecture; 10 training/configuration seeds; paired means and win/tie/loss counts match text | Grid cells reuse seeds and conditions; no stable superiority claim |

The successful fold changes group sizes from 3/3/3/3 to 5/3/3/1. Its baseline
and candidate BPB are 2.1753145 and 4.8149349. The two failed searches store
no fold/baseline evaluation; they are not zero-effect observations. The stored
cross-seed interval and standard deviation are null, consistent with that scope.
The original primary eligibility flag is false and remains visible.

The fixed effective-parameter baseline obtains ARI 1 across the planted probe
grid. Shared-input/native mean ARIs match the rounded manuscript values, and
the seed-averaged sign varies. Thus retaining this as a conditional diagnostic
rather than a proposed generally superior method is supported by the records.

Changes made: added the two new experiment families to the appendix inventory,
identified the shared checkpoint count, and made the bootstrap conditioning
explicit. The main numerical claims and original negative results are unchanged.

Remaining limits: the original ASLoRA training inputs/checkpoint are not all
bundled, and the byte-LM decision summary has weaker generation provenance
than the source-attested response records. Those limitations remain disclosed.
No stored outcome is taken as proof that a different dataset, optimizer,
search budget or longer training trajectory behaves the same way.
