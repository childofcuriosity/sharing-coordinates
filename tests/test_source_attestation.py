import copy
import json
from pathlib import Path

import pytest

from scripts.verify_source_attestation import (
    ROOT,
    _require_source_map,
    verify_family,
    verify_release_sources,
    verify_tomography_bindings,
)


def test_response_source_attestation_matches_current_release():
    records = []
    for seed in range(3):
        records.append(
            (
                ROOT / "results" / "primary_remote" / f"response_audit_seed{seed}.json",
                ("source_sha256",),
            )
        )
    verified = verify_family("response", records)
    assert set(verified) == {
        "experiments/run_response_audit.py",
        "src/response_identifiability.py",
        "src/language.py",
        "src/models.py",
    }


def test_probe_source_attestation_matches_current_release():
    records = [
        (
            ROOT / "results" / "probes_mlp.json",
            ("generation", "source_sha256"),
        ),
        (
            ROOT / "results" / "probes_transformer.json",
            ("generation", "source_sha256"),
        ),
    ]
    verified = verify_family("paired probes", records)
    assert set(verified) == {
        "src/probes.py",
        "src/models.py",
        "src/patterns.py",
        "src/metrics.py",
        "experiments/run_probes.py",
    }


def test_tomography_sources_raw_results_and_checkpoints_are_bound():
    raw_paths = [
        ROOT / "results" / f"response_tomography_seed{seed}.json"
        for seed in range(3)
    ]
    summary = ROOT / "results" / "response_tomography_summary.json"
    records = [(path, ("source_sha256",)) for path in raw_paths]
    records.append((summary, ("source_sha256",)))
    verified = verify_family("response tomography", records)
    assert set(verified) == {
        "experiments/run_response_tomography.py",
        "src/response_tomography.py",
        "src/response_identifiability.py",
        "src/language.py",
        "src/models.py",
    }
    verify_tomography_bindings(raw_paths, summary)
    assert "response_tomography" in verify_release_sources()


def test_source_attestation_rejects_digest_drift():
    payload = json.loads(
        (
            ROOT
            / "results"
            / "primary_remote"
            / "response_audit_seed0.json"
        ).read_text(encoding="utf-8")
    )
    source = copy.deepcopy(payload["source_sha256"])
    source["src/models.py"] = "0" * 64
    with pytest.raises(ValueError, match="source hash mismatch"):
        _require_source_map(source, "tampered response")


def test_source_attestation_rejects_paths_outside_repository():
    with pytest.raises(ValueError, match="non-repository source path"):
        _require_source_map({"../outside.py": "0" * 64}, "unsafe")
