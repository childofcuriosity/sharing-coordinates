# Local integration — 2026-10-04

Integrated `handoff/unified-project-20260929` (`30a51c6696f38437a7a99eba81eb4569b38139c2`) into the local main release (`64258be6dbbe65107976c210d245b1dbbfe502d4`). The original release remains in Git history at `64258be6dbbe65107976c210d245b1dbbfe502d4`. The temporary archive branch was removed during branch cleanup on 2026-10-05.

## Included and preserved

The updated manuscript, recovery and decision code, experiments, tests, research records, and `studies/task-weighted-response/` are included. All 12 files deleted by the handoff branch were restored from the original release: WikiText train/validation/test data, three ASLoRA factor snapshots, three byte-LM checkpoints, two historical probe results, and the paper style archive. Their bytes match the original commit exactly.

The retained-file list and hashes are in `research/releases/2026-10-04-local-integration/`. This integration does not change experimental conclusions or rerun training.

## Inventory boundaries

- Root `MANIFEST.sha256` describes the current integrated files and is generated with `python -X utf8 scripts/write_manifest.py`; verify with the same command plus `--check`.
- Original and handoff root manifests are preserved as `original-MANIFEST.sha256` and `handoff-MANIFEST.sha256` in the integration archive.
- `HANDOFF.json`, `HANDOFF_VALIDATION.json`, `HANDOFF_EXCLUDED_FILES.json`, and `FULL_RESEARCH_MANIFEST.sha256` remain historical records of the September 29 delivery. Their counts and paths are not a certification of this working tree.
- 465 paths in the historical full-research inventory are still absent locally. They are listed in `retained-files-validation.json`. Restoring the original 12 files does not restore all newer raw gradients, model checkpoints, trajectories, or caches. Full experiment replay requires the external full archive and model environment described in the historical handoff.

## Validation performed

Environment: Windows, Python 3.13.11, PyTorch 2.11.0+cpu, SciPy 1.17.0, pytest 9.1.1.

- Main suite: 180 passed, 1 skipped across the initial run and targeted retry. The initial run passed 160 tests; 20 setup errors came from access to the system pytest temporary directory. All 20 passed with an independent project-local temporary directory.
- Task-weighted original study: 5 passed; batch-correction study: 5 passed. Each suite ran separately in its own directory.
- `scripts/verify_source_attestation.py`: passed for the original response, router-decision, ASLoRA, probes, and tomography generation sources.
- `python -X utf8 -m scripts.check_decision_certificates`: exact integer certificates, four-basis extension, and boundary dual/rank checks passed.
- All 12 restored files match the original Git blob bytes.

Main-suite logs are archived beside the manifests. Upstream whitespace and line endings in frozen evidence were retained to preserve source/hash bindings. No full GPU training, complete experimental replay, or new PDF compilation was performed.

For a fresh main-suite run, create `.tmp` and use a new unused child path for pytest:

```powershell
New-Item -ItemType Directory -Force .tmp | Out-Null
python -X utf8 -m pytest -q -p no:cacheprovider --basetemp .tmp/integration-fresh tests
```

The standard full-release verification also needs newer raw inputs; passing source tests and the current manifest does not establish full replay availability.
