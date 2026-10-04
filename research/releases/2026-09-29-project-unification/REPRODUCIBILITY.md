# Integrated publication version — 2026-09-25

Start with [PUBLICATION_REPRODUCIBILITY.md](research/PUBLICATION_REPRODUCIBILITY.md) for the current proof/decision/training evidence and verification commands. `bash scripts/run_all.sh verify` now includes regenerated dynamics figures/tables and version-aware provenance checks. Old manuscript hashes are checked against their immutable pre-dynamics archive; raw experiment sources/checkpoints remain checked at their original paths. The instructions below retain the earlier studies and their distinct protocols.

# Reproducibility

## Latest focused extension

For frozen noisy recovery, calibration/evaluation splits, source gates,
complete CPU replay and failure reporting, see
[TRUSTED_REPRODUCIBILITY.md](research/TRUSTED_REPRODUCIBILITY.md).
The shared `verify` entry point includes these checks.


## Prospective enhancement (2026-09-25)

New training checkpoints and final-step records are in
`results/enhancement/seed{10,11,12}/`. These are separate 6,000-step,
width-128, random-router runs, not continuations of historical checkpoints.
Raw records bind source, protocol, checkpoint and corpus hashes.

```bash
.venv/bin/python -m experiments.summarize_enhancement --check

# Reexecute into a NEW directory, one GPU and sequential independent seeds.
PYTHON="$PWD/.venv/bin/python" DEVICE=cuda:0 \
  bash scripts/reproduce_enhancement.sh /tmp/sharing-enhancement-fresh
```

The GPU runs use Python 3.12, NVIDIA PyTorch 2.7.0a0 and CUDA 12.8, BF16
autocast for training and FP64 response measurements. PyTorch with
`torch.func.jvp` and a BF16-capable GPU are required; legacy package minimum
dependencies do not describe this runner. Timing, serialization hashes and
floating-point values may differ across backends. Existing output directories
are refused. Natural directions use actual cross-entropy gradients but
controlled Euclidean responses at fixed checkpoints, not AdamW trajectories.
Bootstrap intervals condition on 64 blocks, not a training population.

An independent CPU environment uses Python 3.12, PyTorch 2.7.1+cpu,
NumPy 1.26.4, matplotlib 3.10.1, pytest 8.1.1, transformers 4.36.2,
datasets 2.16.1 and pyarrow 16.1.0. The final validation record states its
tested scope: CPU tests, raw-result derivation and synthetic replay differ
from rerunning all GPU training. See `research/ENHANCEMENT_REVIEW.md`.

All commands below are run from the repository root. The checked-in raw
records are the source of numerical claims; summary JSON, CSV, Markdown, and
LaTeX files are derived from them. A release is valid only when the strict
summarizers, tests, paper build, and manifest check all succeed.

The release inventory binds file bytes, including line endings. The seven
legacy CRLF files named in `.gitattributes` must retain CRLF in the working
tree, even on Linux. Git stores their normalized LF representation and restores
CRLF at checkout. Do not normalize those files with an editor or regenerate
their attestations merely to suppress a mismatch. The 2026-09-24 handover
restored those seven files to their existing manifest hashes; no numerical
results, source-attestation hashes, or checkpoint bytes were changed.

## Environment

The base package requires Python 3.8+, NumPy, PyTorch, and Matplotlib:

```bash
python -m pip install -e ".[test]"
```

The ASLoRA runner additionally requires Hugging Face Transformers and Datasets:

```bash
python -m pip install -e ".[test,aslora]"
```

The staged entry point assumes Bash plus standard Unix utilities (`cmp`,
`mktemp`, and `rm`); the corpus downloader also needs `curl` and `sha256sum`.
`release` requires `latexmk`, or both `pdflatex` and `bibtex`. The ordinary
`verify` mode does not compile LaTeX. Set `PYTHON=python3` when the interpreter
is not available under the name `python`.

The primary GPU records were produced on an NVIDIA RTX 3090. The byte-LM and
response records used Python 3.8.10 and PyTorch 1.11.0+cu113. The ASLoRA
provenance files additionally record Transformers 4.36.2, Datasets 2.16.1,
Tokenizers 0.15.2, NumPy 1.22.4, and CUDA 11.3. These are provenance facts, not
minimum-package constraints. The byte-LM runs were seeded but did not force
deterministic-algorithm mode. The strengthened ASLoRA runner does force it and
requires two exactly matching complete-validation re-evaluations of every
raw-`B`-selected canonical action. Exact retraining bits across hardware and
library stacks are nevertheless not promised.

The additive 2026-09-24 audit was tested with Python 3.12.3 and the host's
PyTorch 2.7.0a0+7c8ec84dab.nv25.03 (CUDA 12.8). Its local environment preserves
the host GPU build while pinning the optional Hugging Face dependencies:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install transformers==4.36.2 datasets==2.16.1 pyarrow==16.1.0
PYTHON="$PWD/.venv/bin/python" bash scripts/run_all.sh verify
```

All 143 tests passed in this environment, including the optional local RoBERTa
attachment test. This is a tested host configuration, not a portable lockfile;
the host also contains an unrelated `lightning-thunder`/`dill` dependency conflict.
That package is not used by these tests or experiments. The new autodiff runner
requires `torch.func.jvp`, beyond the base package's historical PyTorch minimum.

## Non-destructive release verification

An additional constructive-recovery pilot is specified in
`research/FACTOR_RECOVERY_PROTOCOL.md`. Its three source-bound records are in
`results/factor_recovery/`; rerun one without overwriting the originals with:

```bash
.venv/bin/python -m experiments.run_factor_recovery --seed 0 --device cuda:0 --output /tmp/factor-recovery-seed0.json
.venv/bin/python -m experiments.summarize_factor_recovery --check
```

Repeat with seeds 1 and 2 as needed. The summarizer validates the source and
checkpoint bindings and preserves recorded noisy failures in derived output;
it does not rerun GPU observations. `verify` includes this check. Numerical
PSD clipping and fixed-weight diagonalization are exploratory estimators,
with no general noisy-recovery guarantee. Original factors enter only the
observation oracle and post-recovery evaluation, not the reconstruction call.

```bash
bash scripts/run_all.sh verify
```

This mode performs no training and writes only to a temporary directory. It:

- runs the full test suite;
- compares generation-time source hashes in every primary raw record with the
  released implementation, not merely with the other seeds;
- validates and reconstructs the Gaussian response, structured-tomography,
  ASLoRA, and byte-LM decision summaries, then compares all released derived
  outputs byte-for-byte;
- regenerates the six audit LaTeX tables (three compact main tables and three
  appendix detail tables) plus the tomography macro file in the temporary
  directory and compares them with `paper/generated`;
- recomputes the legacy aggregate `results/summary.json` in the temporary
  directory; and
- checks every entry in `MANIFEST.sha256`.

This is the command designed for routine artifact checking. It does not imply
that the expensive end-to-end retraining command was executed on the current
host.

## Primary byte-LM checkpoints

### Additive full-dimensional autodiff audit

The 2026-09-24 follow-up measures the response of the full packed factor map
using reverse-mode factor gradients followed by forward-mode JVPs. It does not
retrain the language model or overwrite the original analytic audit. With a
PyTorch build supporting `torch.func.jvp`, run each seed on an available GPU:

```bash
python -m experiments.run_autograd_tomography \
  --checkpoint results/checkpoints/language_seed0.pt \
  --device cuda:0 --output /tmp/autograd-tomography-seed0.json
```

Repeat with seeds 1 and 2 and distinct output paths. Existing outputs are never
overwritten. The released run used Python 3.12.3, PyTorch
2.7.0a0+7c8ec84dab.nv25.03 (CUDA 12.8), float64, and RTX 5090 GPUs; peak allocated
memory was approximately 1.89 GiB per process. The observation and comparison
code, protocol, and checkpoints are hash-bound in every new record.

`python -m experiments.summarize_autograd_tomography --check` validates the
released records and compares their summary and LaTeX table. This is now part
of `verify`. The new runner requires a newer PyTorch than the original core
package's minimum; the published runtime above is the tested follow-up setup.

### Original training artifacts

The release contains the exact WikiText-2 byte streams and three frozen
checkpoints. The corpus hashes are:

| Split | SHA-256 |
|---|---|
| train | `9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f` |
| validation | `f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8` |
| test | `d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0` |

The checkpoint hashes for seeds 0, 1, and 2 are respectively
`6f9fce092e9a7dba6a24de9178736425cbf27d62b2e385ecce0b2210a217324d`,
`ae09aaf044be8bbcdc7acf08827d96a47423a4422bff361e2d0b82513ab7a4d5`,
and `77006e2d6d5c024ddca274f3cd029c9d4672a0c53ed1d99ac7528453521c2fe4`.
The raw JSON preserves the original remote paths, while every consumer accepts
the released local files and verifies content hashes.

To regenerate the corpus from its pinned public source:

```bash
bash scripts/download_wikitext2.sh
```

To retrain the three structured-cycle checkpoints and rerun the Gaussian
response, structured-tomography, and partition-decision audits:

```bash
DEVICE=cuda bash scripts/run_all.sh byte-lm
```

The response runner evaluates a hypothetical Euclidean gradient-flow response
at the frozen checkpoint. It neither reconstructs the checkpoint's AdamW state
nor asserts a historical graph. Its eight fixed probes per seed test the
predeclared alternative `M = 0.65 I + 0.35 11^T/K`; finite probe success is not
an equality certificate. Across the three checkpoints there are 24
checkpoint--probe evaluations but only 10 unique Gaussian draws because the
deterministic seed windows overlap. The canonical response records use probe
seed `740000 + seed` but deliberately hold the auxiliary analytic/autograd/finite-
difference formula-check seed fixed at `750000` for all three checkpoints;
`scripts/run_all.sh byte-lm` passes both policies explicitly.

The structured-tomography runner is deterministic and uses no probe seed. From
the exact shared product it constructs an orthonormal basis of the effective
row space and `L` orthogonal complement codes. At the released
`(L,K,D)=(12,4,787712)` dimensions, four designed effective-gradient probes
reconstruct the complete response because `D-K >= L`. The explicit gauge
construction supplies exact common-product equality; the floating-point
product residual is only a consistency check. The three raw records are
`results/response_tomography_seed{0,1,2}.json`, and the strict aggregate is
`results/response_tomography_summary.json`. See
`research/RESPONSE_TOMOGRAPHY.md` for the proof, noise bounds, and conditions.
The four full-dimensional responses are analytic evaluations of the structured
response-block formula. That implementation was independently checked against
autograd and finite differences on the declared `4 x 32` extraction; the audit
does not claim four full-model JVP or autograd measurements.

The partition-decision runner uses 5,000 positive-stochastic,
condition-at-most-30 trials, a first-valid rule, four groups, and 256 disjoint
`8 x 128` test batches. Search exhaustion remains a recorded negative result;
the script does not enlarge the search post hoc.

## ASLoRA first-merge audit

This is a **documented reimplementation of the paper-defined settings**, not
an official-code/checkpoint reproduction. It follows the reported
RoBERTa-base/MRPC configuration: rank 8,
alpha 16, learning rate `4e-4`, batch size 16, 30 epochs, weight decay 0.1,
linear schedule with 0.06 warmup, and query/value LoRA with one shared `A` per
projection. It stops at one-indexed optimizer step 560 before committing a
merge. Appendix Table 5 also lists an update ratio `lambda=0.5`, but the
available method text defines no operation that uses it. The value is retained
in configuration/provenance as `update_ratio_record_only`; the reimplementation
does not invent an update rule. The available paper is ambiguous between
adjacent and all-pairs candidates and between per-projection and joint
scheduling, so both candidate modes and both scopes are recorded;
per-projection is primary.

The strict canonical summarizer requires local MRPC parquet paths ending in
`glue/mrpc/train-00000-of-00001.parquet` and
`glue/mrpc/validation-00000-of-00001.parquet`. Set:

```bash
export ASLORA_TRAIN_PARQUET=/absolute/path/glue/mrpc/train-00000-of-00001.parquet
export ASLORA_VALIDATION_PARQUET=/absolute/path/glue/mrpc/validation-00000-of-00001.parquet
export ASLORA_MODEL=/absolute/path/to/roberta-base
export ASLORA_MODEL_REVISION=main             # provenance field for this local snapshot
export ASLORA_LOCAL_FILES_ONLY=1              # canonical release setting
DEVICE=cuda bash scripts/run_all.sh aslora
```

The staged release entry point requires `ASLORA_MODEL` to be a local directory
so the strict summarizer can bind its complete file inventory. It therefore
does not accept a Hub identifier for this mode. `ASLORA_LOCAL_FILES_ONLY`
defaults to the canonical value `1`, prohibiting fallback downloads while
loading that directory; `ASLORA_CACHE_DIR` changes the workspace-local cache.
The runner also creates a
large `pre_first_merge_checkpoint.pt` containing optimizer state. That
transient file was retained on the training host but is not required to audit
the released results and is not included here; the compact frozen-factor
snapshot, complete audit JSON, configuration, invariance diagnostics, and
provenance are included for each seed. The staged script removes the transient
checkpoint after a successful run unless `ASLORA_KEEP_CHECKPOINT=1` is set.

The MRPC parquet files and RoBERTa base weights are external dependencies, not
manifest entries. Each released run nevertheless byte-binds both parquet files,
every file in the local model directory, and the complete initial attached
model state in provenance. Independent end-to-end regeneration requires
external files matching those recorded inventories. Retain the byte-identical
local model directory; `ASLORA_MODEL_REVISION` is recorded alongside it but is
not a substitute for the inventory. These bindings identify the inputs; they
do not create an exact cross-hardware retraining-bit guarantee.

The deterministic gauge bank has exactly 10 entries per seed: identity, an
orthogonal control, and diagonal/dense gauges at condition limits 2, 4, 8, and
30. It is a finite sensitivity grid, not a random prevalence sample and not an
optimization over validation loss. Raw-running-`B` actions are fixed before
validation. Gauge-transformed float64 factors select an action, then that
layer-index action is evaluated as a temporary tie in the unchanged canonical
model.

All float64 product/distance and exact-restoration gates pass. Two complete
deterministic repetitions of every raw-`B`-selected canonical action agree
exactly. Live factorized float32 re-execution also passes in all three seeds;
because finite-precision agreement is not mathematical equivalence, it remains
a diagnostic and is never the causal-consequence gate.

## Paired functional-probe diagnostic

The legacy target reruns two 1,680-record planted sweeps. The revised protocol
is `paired-crn-fixed-gaussian-v1`. For each configuration, the shared-input and
native estimators receive the identical initial Gaussian probe batch, identical
optional random projection, and identical per-layer/per-probe/per-output iid
standard-Gaussian noise array. Observations are
`Y_i(H_s) = r_i(H_s) + sigma * epsilon_i_s`, where `sigma` is the configured
absolute standard deviation; it is never multiplied by a realized update
standard deviation. Native hidden states are advanced using the clean update,
not the noisy observation.

The JSON stores all three random seeds and these protocol assertions in every
record, plus generation-time hashes of the probe runner and its direct local
dependencies (`src/probes.py`, `src/models.py`, `src/patterns.py`, and
`src/metrics.py`) at artifact level. The analyzer rejects legacy,
unpaired, scale-mismatched, or overclaiming records. The reported empirical gap
uses the planted assignment, not the predicted clusters. It is a descriptive
finite-sample gap: the artifact does not estimate the theorem's population gap
or verify its uniform tail assumption.

The effective-parameter comparison concatenates the instantiated weights and
biases of every `BasisLinear` module and clusters their Euclidean distances. It
uses no projection or observation noise. This is a simple baseline available
in the present weight-mixing implementation; it is not labeled a functional-
equivalence test and need not identify functionally equivalent neural
parameterizations in general.

The outcome-blind redesign, metadata-only fingerprint amendment, both complete
reruns, exact source/result hashes, and final comparisons are recorded in
`research/SHAREPROBE_PROTOCOL.md`.

## Derived outputs and release build

To rebuild derived files after intentionally changing raw records:

```bash
bash scripts/run_all.sh derive
```

This also validates the three tomography raw records, rebuilds their summary,
and regenerates `paper/generated/response_tomography_numbers.tex`.

To regenerate the older planted/synthetic diagnostic sweeps:

```bash
DEVICE=cuda bash scripts/run_all.sh legacy
```

To compile the manuscript, run all tests, and intentionally rewrite the
manifest after reviewed changes:

```bash
bash scripts/run_all.sh release
```

`release` is mutating: it rewrites derived summaries/tables/figures,
`paper/main.pdf`, and `MANIFEST.sha256`. It does not retrain raw GPU models.
Use `byte-lm` and `aslora` first when raw primary records themselves must be
regenerated.

## Integrity and schema gates

The primary summaries fail closed on protocol drift:

- `experiments/summarize_response_audit.py` requires seeds 0--2, the frozen
  1,500-step protocol, rank/positivity checks, formula validation, and the
  permutation control.
- `experiments/summarize_response_tomography.py` requires the three frozen
  checkpoints, four-probe design/rank gates, exact symbolic common-product
  attestations, permutation controls, and fixed non-permutation separation.
- `experiments/summarize_router_decisions.py` retains both exhausted searches
  and completed searches and applies the frozen, predeclared replication rule.
  Its JSON key `preregistered_search_seed_count` is retained only for schema
  compatibility; it counts seeds in that predeclared finite search and does
  not assert an externally time-stamped registration.
- `experiments/summarize_aslora_audit.py` requires all seeds, exact first-merge
  timing, complete 408-example validation evaluation, candidate coverage,
  deterministic execution, repeated selected-action evaluation, canonical
  restoration, and matching generation-time source hashes.

`scripts/verify_source_attestation.py` additionally matches the tomography
source map in all three raw records and the summary against the five released
generation sources, checks every raw-result digest recorded by the summary,
and binds each raw record to its released frozen checkpoint.

All canonical non-CSV derived text (JSON, Markdown, and LaTeX) is written with
explicit LF endings, so `verify` performs the same byte comparison on Windows
and Linux; CSV retains the standard `csv` dialect's CRLF records. The raw
response/tomography JSON writer is deliberately left frozen because its source
hash is attested by the released raw records. `verify` never rewrites those raw
records and disables Python bytecode while rebuilding only inside its temporary
directory.

`MANIFEST.sha256` includes the WikiText text files and frozen byte-LM
checkpoints, in addition to code, raw/derived results, figures, manuscript
sources, and PDF. Verify it directly with:

```bash
python scripts/write_manifest.py --check
```

### Independent CPU replay and container isolation

Create the environment on local disk. In this container the inherited
LD_LIBRARY_PATH includes the globally installed NVIDIA PyTorch libraries;
a newly installed CPU wheel would otherwise load incompatible shared objects.
The verified CPU commands remove that path and the inherited PYTHONPATH only
for the invoked subprocess:

    python -m venv /tmp/sharing-clean
    env -u PIP_CONSTRAINT /tmp/sharing-clean/bin/python -m pip install \
      torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
    env -u PIP_CONSTRAINT /tmp/sharing-clean/bin/python -m pip install \
      -r research/clean_cpu_requirements.txt --index-url https://pypi.org/simple
    env -u PIP_CONSTRAINT /tmp/sharing-clean/bin/python -m pip install \
      -e . --no-deps --no-build-isolation
    env -u LD_LIBRARY_PATH -u PYTHONPATH /tmp/sharing-clean/bin/python -m pytest -q
    env -u LD_LIBRARY_PATH -u PYTHONPATH /tmp/sharing-clean/bin/python \
      -m experiments.summarize_enhancement --check
    env -u LD_LIBRARY_PATH -u PYTHONPATH /tmp/sharing-clean/bin/python \
      -m experiments.run_joint_noise --output /tmp/fresh-joint-noise.json

Use the configured network proxy for installation; no proxy is required for
tests or replay. The CPU wheel must be installed before applying the complete
freeze. The new environment inherits no system site packages. All 156 tests
passed; dependency deprecation warnings remain. The deterministic table
derivation agrees exactly. All 450 synthetic grid cells replay with matching
statuses, but noisy factor errors are not numerically identical across
PyTorch/linear-algebra backends, especially in ill-conditioned cells.
Both runs reproduce severe rank-degeneracy sensitivity. The CPU replay is
not a claim of bitwise GPU training reproduction.
