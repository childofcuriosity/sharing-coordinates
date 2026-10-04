# Reproducing the pretrained-model study

The new study extends the theoretical artifact with frozen-backbone shared modules on nine public pretrained models, three adapter seeds each. The main theorem remains an internal Euclidean-response statement. Model training uses AdamW, but response observations use controlled FP64 factor differentiation; these are different objects.

## Inspect and verify existing evidence

Start in the repository root. The new environment is `.cache/llm/venv`; the original `.venv` and its older Transformers installation are preserved. The tested versions and GPU/driver are recorded in `LLM_ENVIRONMENT.json`. A new environment needs a working PyTorch/CUDA build for its GPU, then the pinned `[llm]` extra. On this host we used a system-site-packages venv to retain its working NVIDIA PyTorch build:

```bash
python -m venv --system-site-packages .cache/llm/venv
# On this host PIP_CONSTRAINT points at an unrelated NVIDIA constraints file.
env -u PIP_CONSTRAINT .cache/llm/venv/bin/python -m pip install -e '.[llm,test]'
.cache/llm/venv/bin/python scripts/restore_llm_corpora.py
.cache/llm/venv/bin/python -m pytest -q tests/test_llm_shared.py tests/test_llm_observations.py tests/test_llm_protocol.py
.cache/llm/venv/bin/python -m experiments.summarize_llm_study
MPLCONFIGDIR=.cache/matplotlib .cache/llm/venv/bin/python -m experiments.generate_llm_paper
```

The summarizer requires all 27 checkpoints, complete query/noise grids, all nine decision units, matching checkpoint/observation/policy hashes, and disjoint probe/loss documents. It does not silently omit failed or missing units. It deterministically regenerates derived JSON/CSV files and tables; raw records are not overwritten. `check_freeze()` checks all 17 frozen source/config/data files plus raw corpus hashes at each main stage. `scripts/prepare_llm_corpora.py` is the original *development pinning* script; reproduction uses `restore_llm_corpora.py`, which fetches the existing immutable URLs without resolving current branches.

The data lock, model inventory and model-byte inventory are separate: `LLM_CORPUS_LOCK.json`, `LLM_ACCESS_INVENTORY.json`, `LLM_MODEL_BYTES.json`. Downloads use public revision-pinned endpoints and `token=False`. Llama3.2's 401 access failures are recorded; no Llama results are claimed. Set the user-provided proxy variables for network access as needed. No credentials or gate bypass are required for the nine accessible models.

## Regenerate training in a separate working copy

Preserve this delivery's evidence. In a separate copy with identical source and input locks, archive/remove the copy's existing `results/llm/calibration`, `results/llm/evaluation`, `research/LLM_ACCEPTANCE_POLICY.json` and derived study outputs before a fresh reproduction. The original delivery must stay intact. Main commands refuse to overwrite completed stage records or partial recovery logs. Merely rerunning a campaign in the delivered copy skips completed stages; that is not an independent reproduction.

```bash
.cache/llm/venv/bin/python scripts/download_llm_models.py --workers 3
.cache/llm/venv/bin/python scripts/restore_llm_corpora.py
.cache/llm/venv/bin/python scripts/run_llm_campaign.py --stage gpu --split calibration --gpus 0 1 2 3
.cache/llm/venv/bin/python scripts/run_llm_campaign.py --stage recover --split calibration --workers 4
.cache/llm/venv/bin/python -m experiments.calibrate_llm_recovery
.cache/llm/venv/bin/python scripts/run_llm_campaign.py --stage gpu --split evaluation --gpus 0 1 2 3 4 5 6 7
.cache/llm/venv/bin/python scripts/run_llm_campaign.py --stage recover --split evaluation --workers 12
.cache/llm/venv/bin/python scripts/run_llm_campaign.py --stage decisions --split evaluation --gpus 0 1 2 3 4 5 6 7
.cache/llm/venv/bin/python -m experiments.stress_llm_conditioning
.cache/llm/venv/bin/python -m experiments.check_llm_action_repeat
.cache/llm/venv/bin/python -m experiments.summarize_llm_study
.cache/llm/venv/bin/python -m experiments.generate_llm_paper
```

The runner sets CPU thread limits, one main job per selected GPU, and offline model loading after downloads. GPU observation jobs drop the backbone before auditing factor responses. The 8B model's maximum allocated footprint was 19.08 GiB; runtime and memory depend on environment. A frozen budget is 256 adaptation steps and 64 decision recovery steps, not a convergence guarantee. The three seeds change adapters and batches, not pretrained backbone weights.

## Independent arithmetic replay

`experiments.verify_llm_replay` selects three query conditions at every evaluation checkpoint (81 cases, 243 estimator outputs), using saved observations and a separately installed CPU PyTorch. The existing successful replay used PyTorch 2.7.1+cpu in `/tmp/sharing-clean-20260925`; its dependencies and numerical results are in `results/llm/cpu-replay.json`. On this NVIDIA host, inherited library paths must be unset for that environment:

```bash
env -u LD_LIBRARY_PATH -u PYTHONPATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /tmp/sharing-clean-20260925/bin/python -m experiments.verify_llm_replay
```

For a new replay, use a separate working copy or archive that copy's existing `cpu-replay.json`: the script refuses overwrite. This checks estimator arithmetic and three full-dimensional residual identities; it does not independently repeat backbone training. BF16 training is not bitwise deterministic. Three identical Pythia actions per seed quantify a small execution variation rather than implying structural improvements.

## Files and distribution scope

- `results/llm/{calibration,evaluation}/`: final factor checkpoints, training data identities and loss traces, compact observations, per-case recovery records; evaluation decision units also have action records.
- `results/llm/summary.json`, `recovery-cases.csv`, `decision-cases.csv`: strict derived accounting, not selected examples.
- `results/llm/conditioning-stress.json`: 72 artificial-factor stress cases, separately registered and excluded from the count of 27 trained checkpoints.
- `research/LLM_STUDY_FREEZE.json`, `LLM_ACCEPTANCE_POLICY.json`, `LLM_STRESS_PROTOCOL.json`: source/data/config freeze, calibrated thresholds and separate stress registration.
- `research/LLM_VALIDATION.json` and `LLM_REVIEW.md`: delivery checks and bounded scientific judgment.
- `.cache/llm/hub`: about 40 GiB of revision-pinned model files, excluded from the research/source release.
- `.cache/llm/gradients`: about 17 GiB of original full-dimensional gradients, hash-bound in training records and excluded from the small release. They are available locally and regenerated by the GPU stage. Compact observations suffice for recovery replay; full-space checks also need the relevant raw gradients.
- `.artifact-llm`: operational logs; key development failures and job completion evidence are copied into `research/llm-run-logs/` for durable provenance.

The manuscript-only archive is produced by `scripts/package_arxiv.py`; model weights, optimizer caches and datasets are not bundled with that arXiv source archive. Raw corpora retain their upstream provenance and terms. The fixed ICLR2027 template and author placeholder are intentional. No upload occurs.
