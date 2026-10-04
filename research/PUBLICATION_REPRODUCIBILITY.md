# Reproducing the integrated preprint

Run from the `sharing-coordinates` repository root. The manuscript retains ICLR2027 styling and the requested author placeholder. It is a preprint, with no conference page-limit claim. Upstream pretrained weights are separately downloaded from the pinned inventories; they are not in the TeX archive.

## Read-only verification and paper rebuild

The tested numerical environment is `.cache/llm/venv` (PyTorch 2.7 NVIDIA build, Transformers 4.57.6). The local `.venv` Python 3.12 also supplies the packaging utility. Detailed dependency and hardware records remain in `LLM_ENVIRONMENT.json` and `LLM_REPRODUCIBILITY.md`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=.cache/llm/venv/bin/python bash scripts/run_all.sh verify
.cache/llm/venv/bin/python -m scripts.check_decision_certificates
.cache/llm/venv/bin/python -m experiments.generate_dynamics_paper --check
.venv/bin/python scripts/package_arxiv.py
```

`verify` runs the complete unit suite, prior source/freeze gates, deterministic summaries, independent projector objectives, new publication figures/tables and repository SHA-256 inventory. Packaging rebuilds TeX, selects actual recorder dependencies, extracts the resulting archive, compiles without shell escape, rejects undefined references/overfull boxes, and compares extracted PDF text. It does not run pretrained-model training or certify scientific validity.

`MANIFEST.sha256` covers the research artifact; `dist/arxiv-package-report.json` covers the separately generated source archive. After intentional edits, regenerate affected derived artifacts and rebuild the PDF/package before running `scripts/write_manifest.py`; do not refresh hashes to conceal unexplained differences. The historical `experiments.verify_routing_training_records` records its original pre-integration deliverable and is superseded for current read-only checks by `scripts.verify_publication_records`. Original historical reports are not rewritten to claim validation of a new PDF.

## Evidence map

| Evidence | Immutable inputs/results | Reader-facing interpretation |
|---|---|---|
| Exact decision constructions | `scripts/check_decision_certificates.py`; `paper/generated/decision_certificates.tex` | Integer exhaustive SSE and labeled assignment certificates; proofs are in Appendix A |
| Controlled SGD | `results/routing_training/training.json` | Eight primary settings, all 60 perturbations, ODE reference and stationary control |
| Natural trajectories | `results/llm_routing_trajectories/*/trajectory.json` | All 18 runs, every update, failed/no-event outcomes retained |
| Event replay | `results/routing_training/summary.json` and study logs | Saved optimizer replay and fresh-backbone backward replay have different scopes |
| Real local flow | `results/llm_local_flow*/local_flow.json` | Selected update-70 state; FP32 failure, outer FP64 and dtype-preserving FP64 all retained |
| Earlier recovery | `results/llm/summary.json`; `LLM_STUDY_FREEZE.json` | Nine models, 27 adapter checkpoints, fixed controlled response measurements |
| Historical release | `releases/2026-09-25-pre-dynamics/SNAPSHOT.json` | Original 61-page PDF and 106 versioned dependencies, separate from the integrated release |

## Rerunning training

The numerical raw generation sources remain byte-identical to their recorded source hashes. Read `ROUTING_TRAINING_PROTOCOL.md`, `LLM_ROUTING_TRAJECTORY_PROTOCOL.md` and the three `LLM_LOCAL_FLOW*_PROTOCOL.md` documents before rerunning. Training refuses to overwrite existing outputs. Use a separate copy of the artifact with fresh output locations, retaining the original records; scripts with fixed paths require those output directories to be absent in that copy. Do not delete this release's evidence to rerun it.

The controlled CPU experiment accepts an independent output path:

```bash
.cache/llm/venv/bin/python -m experiments.train_routing_crossing --output /tmp/crossing-rerun.json
```

In a fresh study copy, the 18-run grid is the Cartesian product of `--model pythia160m|qwen06b|smol360m`, `--seed 0|1|2`, and `--optimizer adamw|sgd`:

```bash
.cache/llm/venv/bin/python -m experiments.train_llm_routing_trajectory --model smol360m --seed 1 --optimizer adamw --device cuda:0
```

The local precision follow-up reads `smol360m-adamw-seed1/router_boundary.pt`, its exact batch IDs and the pinned pretrained model. `python -m experiments.train_llm_local_flow_fp64_norm` uses the frozen script's `cuda:5`; run on the original eight-GPU host or explicitly document a device-only source change in a new reproduction. It overrides RMS normalization to preserve input dtype and removes the shared-injection downcast. This is an intentional numerical protocol, not evidence that standard mixed-precision training has FP64 convergence.

The ordinary grid uses BF16 frozen backbones with FP32 factors and clipped scheduled updates. Local flow uses one selected fixed real batch, equal-rate unclipped SGD and FP64 arithmetic. Neither is full-model training. Repeated router seeds across models are dependent; repeated precisions are not new natural-training replicates. See `ROUTING_TRAINING_REVIEW.md` for the complete execution history and limits.
