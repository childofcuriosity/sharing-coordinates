# Reproducing the focused noisy-recovery study

Run commands from the repository root. Python 3.12 was used; the primary
environment has torch 2.7.0a0+7c8ec84dab.nv25.03, SciPy 1.15.2 and NumPy 1.26.4.
The independent CPU environment uses torch 2.7.1+cpu and the same SciPy/NumPy.
The existing clean_cpu_requirements.txt pins the CPU environment; install
torch's CPU wheel from its official index and install this checkout with
pip install -e . --no-deps. The requirements do not fetch an old remote
editable checkout. Set the user's proxy variables if network is needed.
Do not overwrite global libraries.

## Read-only release verification

    PYTHON=.venv/bin/python bash scripts/run_all.sh verify

This checks historical attestations, tests, new frozen-source bindings,
calibration and evaluation inventories, checkpoint hashes, regenerated
summaries/tables, replay comparison, recomputed saved-estimate errors and
residuals, and MANIFEST.sha256. It does not retrain.

The focused checks are:

    .venv/bin/python -m experiments.summarize_trusted_recovery --check
    .venv/bin/python -m experiments.compare_trusted_replay --check
    .venv/bin/python -m scripts.verify_trusted_records
    .venv/bin/python -m pytest tests/test_trusted_recovery.py

Raw records preserve execution paths for provenance. The release summarizer
resolves observations/checkpoints by their registered repository-relative
identities and verifies their hashes, so the checkout can move directories.

## Replay a frozen observation without replacing the primary result

    .venv/bin/python -m experiments.run_trusted_recovery \
      --observations results/trusted_recovery/evaluation/synthetic200.pt \
      --output /tmp/trusted-synthetic200-replay.json

The output must not exist. Budget, split, source and threshold gates remain
active. No true factors enter estimate(); score_truth() runs afterward.
The primary records and independent CPU replay are both retained.

For the isolated CPU environment on this host:

    env -u LD_LIBRARY_PATH -u PYTHONPATH \
      /tmp/sharing-clean-20260925/bin/python -m experiments.run_trusted_recovery \
      --observations results/trusted_recovery/evaluation/language22.pt \
      --output /tmp/trusted-language22-cpu-replay.json

Unsetting these variables avoids the system NVIDIA PyTorch shared libraries.
This is specific to the local host, not a universal dependency workaround.

## Re-extract observations

    .venv/bin/python -m experiments.build_trusted_observations \
      --kind language --split evaluation --seed 22 --device cuda:0 \
      --checkpoint results/trusted_recovery/checkpoints/seed22/checkpoint.pt \
      --output /tmp/trusted-language22-observations.pt

The corpus hash, checkpoint seed and 6000-step budget are checked. Synthetic
observations need no GPU or external data. The stored compressed observations
are sufficient for all released recovery and scoring runs. Full task-gradient
re-extraction additionally requires the bundled checkpoints and training text.

## Cohort execution order

The original order was:
1. Training seeds 20..25 with the unchanged train_enhancement configuration.
2. Development seeds 0..3 and language seed 10 only.
3. Source freeze, then run_trusted_batch --split calibration.
4. calibrate_trusted_recovery writes calibration_policy.json once.
5. run_trusted_batch --split evaluation, then summaries and independent replay.

Batch commands refuse overwrites and primary execution checks the frozen
sources and policy. A fresh full study belongs in a separate working copy
with a new clearly labeled protocol/seed cohort and new freezes; deleting
the released primary results or retrospectively editing its thresholds is
not a faithful reproduction. The old development source versions are retained
in trusted-development-sources/ because split enforcement and full-coordinate
diagnostic support were finalized after the exploratory pilot.

## Numerical and statistical limits

The full clean-CPU replay has identical quality categories but six changed
combined-score acceptance decisions (four spectral, two joint fit).
Do not demand bitwise optimizer equality or claim it was observed.
Known controlled noise, estimated-subspace fitting, finite budgets, only four
language training replications, and local non-interval bounds limit the claims.
Calibration/evaluation variants within a seed are correlated. Confidence
intervals resample seeds, not individual grid cells.

## Manuscript packaging

    .venv/bin/python scripts/package_arxiv.py

This compiles the existing ICLR 2027 manuscript, builds a minimal source archive,
extracts and compiles it without shell escape, and compares extracted PDF text.
It never uploads. Run packaging before refreshing MANIFEST.sha256 because
the PDF bytes can change. Author information remains a placeholder.
