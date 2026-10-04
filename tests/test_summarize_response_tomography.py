import copy
import json
from pathlib import Path

import pytest

from experiments.summarize_response_tomography import summarize_response_tomography


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _released_payloads():
    paths = [
        PROJECT_ROOT / "results" / f"response_tomography_seed{seed}.json"
        for seed in range(3)
    ]
    return paths, [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_released_tomography_results_form_a_valid_three_seed_summary():
    paths, payloads = _released_payloads()
    summary = summarize_response_tomography(payloads, raw_paths=paths)
    assert [row["seed"] for row in summary["seed_rows"]] == [0, 1, 2]
    assert summary["probe_count_K"] == 4
    assert summary["across_seed"]["all_permutation_controls_pass"]
    assert summary["across_seed"]["all_fixed_nonpermutations_distinguished"]
    assert summary["across_seed"][
        "maximum_component_reconstruction_relative_error"
    ] < 1e-10
    assert len(summary["raw_result_sha256"]) == 3


def test_tomography_summary_rejects_a_mixed_source_snapshot():
    _, payloads = _released_payloads()
    mixed = copy.deepcopy(payloads)
    mixed[1]["source_sha256"]["src/models.py"] = "0" * 64
    with pytest.raises(ValueError, match="different source snapshots"):
        summarize_response_tomography(mixed)

