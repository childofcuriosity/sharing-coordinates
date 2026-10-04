import json
import hashlib
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from src.metrics import normalized_entropy, recovery_metrics


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _records(name):
    payload = _load(name)
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return payload


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, list):
        for item in value:
            _assert_finite(item)
    elif isinstance(value, dict):
        for item in value.values():
            _assert_finite(item)


def _assert_unique_configs(records):
    keys = [json.dumps(record["config"], sort_keys=True) for record in records]
    assert len(keys) == len(set(keys))


def _assert_metrics(record, probabilities):
    truth = torch.tensor(record["true_assignment"], dtype=torch.long)
    computed = recovery_metrics(truth, probabilities)
    for name, value in computed.items():
        assert math.isclose(float(record[name]), float(value), rel_tol=1e-6, abs_tol=1e-6)


def test_canonical_inventory_is_complete_unique_and_finite():
    expected_counts = {
        "factorization_full.json": 700,
        "teacher_student_mlp.json": 60,
        "teacher_student_transformer.json": 60,
        "teacher_baselines.json": 40,
        "probes_mlp.json": 1680,
        "probes_transformer.json": 1680,
        "language_full.json": 12,
    }
    for filename, count in expected_counts.items():
        records = _records(filename)
        assert len(records) == count
        _assert_unique_configs(records)
        _assert_finite(records)

    gauge = _load("gauge_full.json")
    assert len(gauge["gauge_trials"]) == 20
    assert len(gauge["saturation"]) == 120
    _assert_finite(gauge)


def test_saved_planted_recovery_metrics_recompute_exactly():
    for filename in (
        "factorization_full.json",
        "teacher_student_mlp.json",
        "teacher_student_transformer.json",
    ):
        for record in _records(filename):
            probabilities = torch.tensor(record["probabilities"], dtype=torch.float32)
            _assert_metrics(record, probabilities)

    for filename in ("probes_mlp.json", "probes_transformer.json"):
        for record in _records(filename):
            num_bases = int(record["config"]["num_bases"])
            for mode in ("counterfactual", "native", "effective_parameter"):
                assignment = torch.tensor(record[f"{mode}_assignment"], dtype=torch.long)
                probabilities = F.one_hot(assignment, num_classes=num_bases).float()
                combined = {"true_assignment": record["true_assignment"], **record[mode]}
                _assert_metrics(combined, probabilities)


def test_language_router_summaries_recompute_from_saved_probabilities():
    for record in _load("language_full.json"):
        initial = torch.tensor(record["initial_probabilities"], dtype=torch.float32)
        final = torch.tensor(record["final_probabilities"], dtype=torch.float32)
        movement = float((final - initial).abs().sum(dim=-1).mean())
        assert math.isclose(
            movement,
            float(record["router_mean_l1_movement"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        assert final.argmax(dim=-1).tolist() == record["router_argmax"]
        assert math.isclose(
            normalized_entropy(final),
            float(record["router_entropy"]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )


def test_probe_artifacts_bind_the_paired_protocol_and_generation_sources():
    root = Path(__file__).resolve().parents[1]
    expected_sources = {
        "src/probes.py": _sha256(root / "src/probes.py"),
        "src/models.py": _sha256(root / "src/models.py"),
        "src/patterns.py": _sha256(root / "src/patterns.py"),
        "src/metrics.py": _sha256(root / "src/metrics.py"),
        "experiments/run_probes.py": _sha256(root / "experiments/run_probes.py"),
    }
    for filename, architecture in (
        ("probes_mlp.json", "mlp"),
        ("probes_transformer.json", "transformer"),
    ):
        payload = _load(filename)
        assert payload["schema_version"] == "probe-sweep-v2"
        assert payload["protocol_version"] == "paired-crn-fixed-gaussian-v1"
        fingerprint_bytes = json.dumps(
            payload["protocol_fingerprint_payload"],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        assert payload["protocol_fingerprint_sha256"] == hashlib.sha256(
            fingerprint_bytes
        ).hexdigest()
        assert payload["generation"]["architecture"] == architecture
        assert payload["generation"]["record_count"] == 1680
        assert payload["generation"]["source_sha256"] == expected_sources
        for record in payload["results"]:
            protocol = record["protocol"]
            for field, value in payload["protocol_fingerprint_payload"].items():
                assert protocol[field] == value
            assert protocol["same_initial_probes_across_functional_modes"] is True
            assert protocol["same_observation_noise_across_functional_modes"] is True
            assert protocol["same_projection_across_functional_modes"] is True
            assert protocol["native_state_evolution"] == "clean_update_only"
            assert protocol["observation_noise_scale_is_absolute"] is True
            assert protocol["empirical_gap_is_population_certificate"] is False
            assert (
                protocol["effective_parameter_baseline_is_functional_equivalence_test"]
                is False
            )
