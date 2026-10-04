# ShareProbe paired-comparison protocol and amendment log

This note records the audit and rerun of the secondary ShareProbe experiment.
It is intentionally separate from the primary response, ASLoRA, and byte-LM
decision audits. ShareProbe is retained only as a planted functional diagnostic.

## Audit of the superseded implementation

The previous `src/probes.py` did not support a controlled comparison:

- shared-input and native signatures used different input seeds (`40000+s`
  versus `41000+s`);
- they used different projection seeds (`50000+s` versus `51000+s`);
- observation noise was drawn from successive global RNG states rather than a
  shared realization;
- the nominal noise level was multiplied by each realized batch's
  `update.std()`, so it was not a fixed observation model;
- only the native-signature baseline was reported; and
- the saved "empirical gap" was calculated using the predicted clustering,
  rather than the planted target partition used by the finite-probe theorem.

The old aggregate ARIs were MLP `0.955157/0.945921` and Transformer
`0.989123/0.990018` for shared-input/native signatures. Those numbers are
superseded because the comparison and noise models differed between methods.

## Outcome-blind revised design

Before running the revised sweep, the following design was fixed in source and
covered by focused tests:

1. Each configuration draws one initial probe tensor with seed `40000+s`, one
   optional projection with seed `50000+s`, and one per-layer/per-probe/per-
   output iid standard-Gaussian noise tensor with seed `60000+s`.
2. Both functional methods receive those same three objects (common random
   numbers). Each marginal observation is
   `Y_i(H_s) = r_i(H_s) + sigma * epsilon_i_s` with absolute standard deviation
   `sigma`; no batch statistic rescales it.
3. Native states advance only with the clean residual update.
4. Average-linkage receives the planted number of bases. The empirical gap is
   evaluated against the planted assignment, not the predicted clusters.
5. A simple parameter-access baseline concatenates all instantiated
   `BasisLinear` effective weights and biases. It receives neither projection
   nor observation noise.
6. The original full grid was retained unchanged: four patterns, ten seeds,
   seven probe counts, and six noise levels, or 1,680 records per architecture.

The parameter baseline is not a functional-equivalence oracle: parameter
symmetries can make different vectors implement the same function, and
distribution-restricted functional equality need not imply parameter equality.
The finite empirical gap is not a population certificate.

## Rerun chronology

The first complete paired rerun used:

- `src/probes.py` SHA256
  `a12dc25635d9f3a2b019b8f7a9f40abd16328c7c2dd12390879944c11f325a76`;
- `experiments/run_probes.py` SHA256
  `b1888f2c21bc1dfab8ea5b3e0ed1a878a54f86bc202a4107e7f7a4a163468cad`;
- MLP raw SHA256
  `779a1ac67fca5d9e7800090f235ca1b8d6cc93bd560a021bc06e39d4416cface`
  (58.945 seconds); and
- Transformer raw SHA256
  `524504291ddb7f49e244c985f2d5ea9a5291547f3c6979af1b0c0b10b414976d`
  (78.973 seconds).

An artifact audit then requested a standalone hash of the invariant protocol
semantics. No random seed, grid point, observation equation, model operation,
metric, clustering rule, or baseline was changed. The only source changes
factored the already-recorded invariant fields into
`PROBE_PROTOCOL_SPEC`, added its canonical JSON SHA256, and wrote that payload
and hash into the artifact envelope. Both full sweeps were rerun rather than
editing raw JSON in place.

That metadata-fingerprint rerun had raw hashes
`3e10f4c5120832223dfd81292b0dc829566c2d8b663c3c1d9f3bd85ec8f4ef62`
(MLP) and
`cbcc15e7ba5e604356875fc05d906e11c9aa65aba00fe02547c1028bcf139960`
(Transformer), with runner hash
`945154269a20b08fa560cafbd8a49cfb7dfe6015a78dac1cd3092c4ea82c4c28`.
A final release audit observed that the source envelope named only the runner
and `src/probes.py`, although the calculation also imports the repository's
model, pattern, and metric modules.  The source envelope was expanded without
changing any protocol, seed, grid, operation, or metric, and both complete
sweeps were rerun again.  Their 1,680 scientific result records are exactly
equal to the preceding rerun; only timestamps, elapsed time, and the expanded
source map differ.

The final released bindings are:

- protocol fingerprint SHA256
  `915e00102458bea94bcda01931bab95d958cf4366e2ae71892d3d2295ab4b061`;
- `src/probes.py` SHA256
  `e9fe07226bee6699253bfd82e2dfe7410ce80b12df64f9a2d2c67dd516df81d8`;
- `src/models.py` SHA256
  `2de707405452dcf04159afef79ad3c717d34f355df6774986ccde82d6ea549d8`;
- `src/patterns.py` SHA256
  `a9f2a75dd3148759bc68760e1f4493f2628244e20c00ab967c6d23f4edb2b4c2`;
- `src/metrics.py` SHA256
  `0922d66d1912377d119320efe37876c7b4d010e77c9f855b3d8b29cf65602b51`;
- `experiments/run_probes.py` SHA256
  `780eaee4c23612870b88ccf33e8f32f6ec3c7c497112166388120f08b8adaa7f`;
- `results/probes_mlp.json` SHA256
  `9bb7d3b8668a4e59fb9fbea834f164f39f72becb61b47529cf5141a4ec634898`
  (1,680 records; 64.231 seconds); and
- `results/probes_transformer.json` SHA256
  `592c1052b862507cf274cbf8bf497bb0d43000faa8f2ac3434890db9e95df28e`
  (1,680 records; 79.902 seconds).

The final rerun reproduces the first paired rerun's reported aggregate and
hardest-cell outcomes. Superseded raw files were overwritten only after their
hashes, elapsed times, and aggregates above were recorded.

## Final comparison

Across the fixed full grid:

| Architecture | Shared-input ARI | Native ARI | Effective-parameter ARI | Shared minus native | cell W/T/L | positive/zero/negative seed means |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 0.676373 | 0.675084 | 1.000000 | +0.001289 | 65/1570/45 | 6/0/4 |
| Transformer | 0.500606 | 0.500865 | 1.000000 | -0.000259 | 61/1558/61 | 3/0/7 |

For the hardest configured cell (`n=1`, `sigma=2`, 40 pattern/seed records),
MLP shared/native mean ARI is `0.011448/0.012451` and Transformer is
`-0.003027/-0.012991`; neither functional method achieves perfect recovery in
any of the 40 records. The effective-parameter baseline achieves 40/40 perfect
recoveries in both architectures.

These results provide no stable advantage for ShareProbe. They do show why a
shared-input diagnostic can target a named functional distribution when
parameters are hidden, but they do not establish necessity, natural-system
superiority, or a population recovery certificate.

## Commands

The final remote commands were:

```bash
/root/autodl-tmp/v2_1_decision/.venv/bin/python -m experiments.run_probes \
  --output results/probes_mlp.json --architecture mlp --device cuda \
  --sweep --sweep-seeds 10 --projection-dimension 0

/root/autodl-tmp/v2_1_decision/.venv/bin/python -m experiments.run_probes \
  --output results/probes_transformer.json --architecture transformer \
  --device cuda --sweep --sweep-seeds 10 --projection-dimension 0

python -m experiments.analyze_results
```

`tests/test_probes.py`, `tests/test_analysis.py`, and
`tests/test_canonical_results.py` check the CRN coupling, absolute-noise
semantics, clean native evolution, target-relative gap, both parameter
extractors, protocol assertions, fingerprint, record counts, generation source
hashes, and recomputed recovery metrics.
