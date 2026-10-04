import copy
import json
from pathlib import Path

import pytest

from experiments.generate_response_tomography_latex import generate, render


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT_ROOT / "results" / "response_tomography_summary.json"


def _summary():
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_released_summary_generates_bound_tomography_macros(tmp_path):
    output = tmp_path / "numbers.tex"
    generate(SUMMARY, output)
    text = output.read_text(encoding="utf-8")
    assert "\\ResponseTomographyProbeCount}{4}" in text
    assert "\\ResponseTomographyDimension}{787,712}" in text
    assert "6.85\\!\\times\\!10^{-11}" in text
    assert "0.9731" in text


def test_generator_rejects_failed_or_drifted_summary():
    payload = copy.deepcopy(_summary())
    payload["across_seed"]["all_permutation_controls_pass"] = False
    with pytest.raises(ValueError, match="required control"):
        render(payload, SUMMARY)

    payload = copy.deepcopy(_summary())
    payload["probe_count_K"] = 5
    with pytest.raises(ValueError, match="drifted"):
        render(payload, SUMMARY)
