"""Validate and summarize the three-seed ASLoRA first-merge audit.

The scientific path in the raw artifacts is the canonical action audit: a
gauge is applied in float64 only to select a layer-index action, and that
action is evaluated by a temporary tie in the unchanged canonical model.
Factorized float32 re-execution is retained as a numerical diagnostic and is
never promoted to an equivalence gate here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SEEDS = (0, 1, 2)
MODES = ("adjacent", "all")
SCOPES = ("per_projection", "joint")
TARGETS = ("query", "value")
GAUGE_NAMES = (
    "identity",
    "orthogonal_k1",
    "diagonal_k2",
    "dense_k2",
    "diagonal_k4",
    "dense_k4",
    "diagonal_k8",
    "dense_k8",
    "diagonal_k30",
    "dense_k30",
)
SOURCE_PATHS = (
    "experiments/run_aslora_mrpc.py",
    "src/aslora_training.py",
    "src/aslora_witness.py",
)
METRICS = ("loss", "accuracy", "f1")
DEFAULT_INPUTS = tuple(
    PROJECT_ROOT / "results" / ("aslora_seed%d" % seed) / "audit.json"
    for seed in PRIMARY_SEEDS
)


def _portable_input_path(path: Path) -> str:
    """Serialize release inputs without binding a summary to one checkout."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("%s must be finite" % name)
    return converted


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_input_inventory(value: object, name: str) -> Mapping[str, object]:
    inventory = _property_map(value, name)
    files = _sequence(inventory.get("files"), "%s.files" % name)
    _require(
        inventory.get("kind") in {"file", "directory"}
        and int(inventory.get("file_count", -1)) == len(files)
        and len(files) > 0
        and _is_sha256(inventory.get("inventory_sha256")),
        "%s has an invalid artifact inventory" % name,
    )
    for index, raw in enumerate(files):
        item = _property_map(raw, "%s.files[%d]" % (name, index))
        _require(
            bool(item.get("relative_path"))
            and int(item.get("size_bytes", -1)) >= 0
            and _is_sha256(item.get("sha256")),
            "%s.files[%d] is not byte-bound" % (name, index),
        )
    _require(
        inventory.get("inventory_sha256") == _canonical_json_sha256(files),
        "%s inventory digest does not match its file entries" % name,
    )
    return inventory


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "%s is not a JSON object" % path)
    return payload


def _property_map(value: object, name: str) -> Mapping[str, object]:
    _require(isinstance(value, dict), "%s must be an object" % name)
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    _require(isinstance(value, list), "%s must be an array" % name)
    return value


def _action_key(action: Mapping[str, object]) -> str:
    pieces = []
    for target in TARGETS:
        if target not in action:
            continue
        pair = action[target]
        _require(
            isinstance(pair, list) and len(pair) == 2,
            "action.%s must be a two-element layer pair" % target,
        )
        lower, upper = int(pair[0]), int(pair[1])
        _require(lower < upper, "action pair must be ordered lower-to-upper")
        pieces.append("%s:%d-%d" % (target, lower, upper))
    _require(bool(pieces), "action must select at least one target")
    return "|".join(pieces)


def _pair_is_adjacent(pair: Sequence[object]) -> bool:
    return int(pair[1]) - int(pair[0]) == 1


def _validate_action_for_mode(
    action: Mapping[str, object], mode: str, scope: Optional[str] = None
) -> None:
    _require(set(action).issubset(TARGETS), "action contains an unknown target")
    if mode == "adjacent":
        _require(
            all(_pair_is_adjacent(action[target]) for target in action),
            "adjacent mode selected a non-adjacent pair",
        )
    if scope == "per_projection":
        _require(set(action) == set(TARGETS), "per_projection action lacks Q or V")
    elif scope == "joint":
        _require(set(action) == set(TARGETS), "joint action lacks Q or V")
        _require(
            action["query"] == action["value"],
            "joint action must use the same layer pair for Q and V",
        )


def _metric_payload(value: object, name: str) -> Dict[str, float]:
    record = _property_map(value, name)
    return {metric: _finite(record.get(metric), "%s.%s" % (name, metric)) for metric in METRICS}


def _metric_difference(
    first: Mapping[str, float], second: Mapping[str, float]
) -> Dict[str, float]:
    return {metric: float(first[metric]) - float(second[metric]) for metric in METRICS}


def _metrics_close(
    first: Mapping[str, object], second: Mapping[str, object], tolerance: float = 1e-10
) -> bool:
    return all(
        math.isclose(
            _finite(first.get(metric), "first metric"),
            _finite(second.get(metric), "second metric"),
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
        for metric in METRICS
    )


def _range(values: Iterable[float]) -> Dict[str, float]:
    materialized = [float(value) for value in values]
    _require(bool(materialized), "cannot summarize an empty range")
    return {"min": min(materialized), "max": max(materialized)}


def _protocol_projection(
    audit: Mapping[str, object], provenance: Mapping[str, object]
) -> Dict[str, object]:
    config = _property_map(audit.get("config"), "config")
    provenance_config = _property_map(provenance.get("config"), "provenance.config")
    _require(config == provenance_config, "audit and provenance configs disagree")
    gauge_generation = _property_map(audit.get("gauge_generation"), "gauge_generation")
    families = _sequence(gauge_generation.get("families"), "gauge_generation.families")
    dataset = _property_map(provenance.get("dataset"), "provenance.dataset")
    inputs = _property_map(
        provenance.get("input_artifacts"), "provenance.input_artifacts"
    )
    return {
        "audit_format": audit.get("format"),
        "audit_status": audit.get("status"),
        "provenance_format": provenance.get("format"),
        # Seed is the sole intentional config variation. Hash every other
        # recorded field so an unanticipated drift cannot be hidden by a
        # hand-selected protocol projection.
        "config": {
            name: config[name] for name in sorted(config) if name != "seed"
        },
        "decision_protocol": provenance.get("decision_protocol"),
        "implementation_choices": provenance.get("implementation_choices"),
        "source_ambiguities": provenance.get("source_ambiguities"),
        "timeline": audit.get("timeline"),
        "distance_baseline_data_usage": audit.get("distance_baseline_data_usage"),
        "gauge_generation": {
            "performed_before_validation": gauge_generation.get(
                "performed_before_validation"
            ),
            "deterministic_bank_uses_only_snapshot_shapes_and_seed": gauge_generation.get(
                "deterministic_bank_uses_only_snapshot_shapes_and_seed"
            ),
            "external_witness_count": gauge_generation.get("external_witness_count"),
            "validation_used_for_raw_action_selection": gauge_generation.get(
                "validation_used_for_raw_action_selection"
            ),
            "families": [
                {
                    "name": family.get("name"),
                    "family": family.get("family"),
                    "requested_condition_limit": family.get(
                        "requested_condition_limit"
                    ),
                    "source": family.get("source"),
                }
                for family in families
            ],
        },
        # Hugging Face transformation fingerprints include execution details and
        # differ across these seeded runs despite the same parquet inputs.  They
        # are recorded separately, but are not a protocol identity field.
        "dataset": {
            name: dataset.get(name)
            for name in (
                "config",
                "name",
                "requested_revision",
                "source",
                "train_examples",
                "train_parquet",
                "validation_examples",
                "validation_parquet",
            )
        },
        "generation_input_inventory_sha256": {
            name: _property_map(inputs.get(name), "input_artifacts.%s" % name).get(
                "inventory_sha256"
            )
            for name in ("local_model", "train_parquet", "validation_parquet")
        },
    }


def _validate_expected_protocol(
    audit: Mapping[str, object], provenance: Mapping[str, object], seed: int
) -> str:
    config = _property_map(audit.get("config"), "config")
    expected = {
        "seed": seed,
        "a_scope": "per_projection",
        "alpha": 16.0,
        "candidate_modes": list(MODES),
        "dataset_config": "mrpc",
        "dataset_name": "glue",
        "dataset_revision": "main",
        "decision_scopes": list(SCOPES),
        "deterministic_algorithms": True,
        "effective_batch_size": 16,
        "epochs": 30,
        "eval_batch_size": 16,
        "first_merge_step": 560,
        "gauge_atol": 5e-5,
        "gauge_condition_limits": [2.0, 4.0, 8.0, 30.0],
        "gauge_rtol": 5e-5,
        "gradient_accumulation_steps": 1,
        "learning_rate": 4e-4,
        "max_length": 512,
        "merge_count": 7,
        "merge_interval": 240,
        "model_revision": "main",
        "paper_deviations": {},
        "primary_decision_scope": "per_projection",
        "primary_action_evaluation_repeats": 2,
        "rank": 8,
        "start_merge_step": 320,
        "target_names": list(TARGETS),
        "train_batch_size": 16,
        "update_ratio_record_only": 0.5,
        "warmup_ratio": 0.06,
        "weight_decay": 0.1,
    }
    for name, expected_value in expected.items():
        _require(
            config.get(name) == expected_value,
            "seed %d violates protocol: config.%s=%r, expected %r"
            % (seed, name, config.get(name), expected_value),
        )
    _require(
        str(config.get("model_name", "")).rstrip("/\\").endswith("roberta-base"),
        "seed %d did not use RoBERTa-base" % seed,
    )
    _require(
        audit.get("format") == "aslora-roberta-mrpc-first-merge-audit-v1"
        and audit.get("status")
        == "completed_pre_first_merge_audit_no_merge_committed",
        "seed %d has an unexpected audit format or status" % seed,
    )
    _require(
        provenance.get("format") == "aslora-mrpc-provenance-v1"
        and int(provenance.get("seed", -1)) == seed,
        "seed %d has invalid provenance identity" % seed,
    )
    execution = _property_map(audit.get("execution"), "execution")
    timeline = _property_map(audit.get("timeline"), "timeline")
    _require(
        int(execution.get("optimizer_step", -1)) == 560
        and int(execution.get("micro_batches_seen", -1)) == 560
        and int(timeline.get("first_merge_optimizer_step", -1)) == 560
        and timeline.get("condition") == "t>Ts and (t-Ts)%m==0"
        and timeline.get("running_average_observation")
        == "after_each_optimizer_update",
        "seed %d was not stopped at the declared first-merge boundary" % seed,
    )
    baseline = _property_map(audit.get("baseline_validation"), "baseline_validation")
    _require(
        int(baseline.get("examples", -1)) == 408
        and int(baseline.get("batches", -1)) == 26,
        "seed %d did not evaluate the complete MRPC validation split" % seed,
    )
    dataset = _property_map(provenance.get("dataset"), "provenance.dataset")
    _require(
        dataset.get("name") == "glue"
        and dataset.get("config") == "mrpc"
        and dataset.get("requested_revision") == "main"
        and dataset.get("source") == "local_parquet"
        and int(dataset.get("train_examples", -1)) == 3668
        and int(dataset.get("validation_examples", -1)) == 408
        and str(dataset.get("train_parquet", "")).replace("\\", "/").endswith(
            "/glue/mrpc/train-00000-of-00001.parquet"
        )
        and str(dataset.get("validation_parquet", "")).replace("\\", "/").endswith(
            "/glue/mrpc/validation-00000-of-00001.parquet"
        )
        and bool(dataset.get("train_fingerprint"))
        and bool(dataset.get("validation_fingerprint")),
        "seed %d has incomplete or unexpected dataset provenance" % seed,
    )
    inputs = _property_map(
        provenance.get("input_artifacts"), "provenance.input_artifacts"
    )
    _require(
        _is_sha256(inputs.get("initial_attached_model_state_sha256")),
        "seed %d lacks an initial full-model state digest" % seed,
    )
    model_inventory = _validate_input_inventory(
        inputs.get("local_model"), "input_artifacts.local_model"
    )
    train_inventory = _validate_input_inventory(
        inputs.get("train_parquet"), "input_artifacts.train_parquet"
    )
    validation_inventory = _validate_input_inventory(
        inputs.get("validation_parquet"), "input_artifacts.validation_parquet"
    )
    _require(
        model_inventory.get("kind") == "directory"
        and train_inventory.get("kind") == "file"
        and validation_inventory.get("kind") == "file",
        "seed %d has unexpected input artifact kinds" % seed,
    )
    normalized = lambda value: str(value).replace("\\", "/").rstrip("/")
    _require(
        normalized(model_inventory.get("resolved_path"))
        == normalized(config.get("model_name"))
        and normalized(train_inventory.get("resolved_path"))
        == normalized(config.get("train_parquet"))
        and normalized(validation_inventory.get("resolved_path"))
        == normalized(config.get("validation_parquet")),
        "seed %d input inventories do not bind the configured paths" % seed,
    )
    model_names = {
        str(_property_map(item, "model input file").get("relative_path"))
        for item in _sequence(model_inventory.get("files"), "local_model.files")
    }
    _require(
        "config.json" in model_names
        and bool({"model.safetensors", "pytorch_model.bin"} & model_names)
        and bool({"tokenizer.json", "vocab.json"} & model_names),
        "seed %d local model inventory lacks config, weights, or tokenizer" % seed,
    )
    gauge_generation = _property_map(audit.get("gauge_generation"), "gauge_generation")
    families = _sequence(gauge_generation.get("families"), "gauge families")
    _require(
        [family.get("name") for family in families] == list(GAUGE_NAMES),
        "seed %d has the wrong deterministic gauge bank" % seed,
    )
    _require(
        gauge_generation.get("performed_before_validation") is True
        and gauge_generation.get("deterministic_bank_uses_only_snapshot_shapes_and_seed")
        is True
        and int(gauge_generation.get("external_witness_count", -1)) == 0
        and gauge_generation.get("validation_used_for_raw_action_selection") is False,
        "seed %d gauge bank was not validation-independent" % seed,
    )
    data_usage = _property_map(
        audit.get("distance_baseline_data_usage"),
        "distance_baseline_data_usage",
    )
    expected_data_usage = {
        "running_raw_B_selection_proxy": "prevalidation_snapshot_only_ASLoRA_rule",
        "current_effective_update_frobenius": "prevalidation_snapshot_only",
        "current_lower_local_activation_rms": (
            "complete_validation_inputs_no_labels_posthoc_diagnostic"
        ),
        "validation_loss_oracles": (
            "complete_validation_inputs_and_labels_posthoc_oracle_never_used_for_raw_gauge_selection"
        ),
    }
    _require(
        all(data_usage.get(name) == value for name, value in expected_data_usage.items()),
        "seed %d has unexpected baseline/oracle data usage" % seed,
    )
    return _canonical_json_sha256(_protocol_projection(audit, provenance))


def _candidate_metrics(
    candidates: Mapping[str, object], key: str
) -> Dict[str, float]:
    _require(key in candidates, "candidate evaluation is missing %s" % key)
    evaluation = _property_map(candidates[key], "candidate_evaluations.%s" % key)
    return _metric_payload(evaluation.get("metrics"), "candidate metrics")


def _expected_family_keys(layer_count: int) -> set:
    keys = set()
    for lower in range(layer_count):
        for upper in range(lower + 1, layer_count):
            keys.add("query:%d-%d" % (lower, upper))
            keys.add("value:%d-%d" % (lower, upper))
            keys.add(
                "query:%d-%d|value:%d-%d" % (lower, upper, lower, upper)
            )
    return keys


def _selected_action_keys(audit: Mapping[str, object]) -> set:
    selected = set()
    distance_baselines = _property_map(
        audit.get("distance_baselines"), "distance_baselines"
    )
    for mode in MODES:
        selections = _property_map(
            _property_map(distance_baselines.get(mode), "distance mode").get(
                "selections"
            ),
            "distance selections",
        )
        for method in selections.values():
            method_scopes = _property_map(method, "baseline selection")
            for scope in SCOPES:
                selected.add(
                    str(
                        _property_map(method_scopes.get(scope), "baseline scope").get(
                            "action_key"
                        )
                    )
                )
    for gauge in _sequence(audit.get("gauge_audits"), "gauge_audits"):
        modes = _property_map(
            _property_map(gauge, "gauge audit").get("modes"), "gauge modes"
        )
        for mode in MODES:
            decisions = _property_map(
                _property_map(modes.get(mode), "gauge mode").get("decisions"),
                "gauge decisions",
            )
            for scope in SCOPES:
                selected.add(
                    str(
                        _property_map(decisions.get(scope), "gauge decision").get(
                            "action_key"
                        )
                    )
                )
    return selected


def _validate_candidate_coverage(
    audit: Mapping[str, object], seed: int
) -> Dict[str, object]:
    candidates = _property_map(
        audit.get("candidate_evaluations"), "candidate_evaluations"
    )
    _require(
        int(audit.get("candidate_evaluation_count", -1)) == len(candidates),
        "seed %d candidate_evaluation_count is stale" % seed,
    )
    activation = _property_map(audit.get("activation_weighting"), "activation_weighting")
    per_layer_counts = _property_map(
        activation.get("per_layer_counts"), "activation per_layer_counts"
    )
    layer_count = len(_sequence(per_layer_counts.get("query"), "query counts"))
    _require(
        layer_count == 12
        and len(_sequence(per_layer_counts.get("value"), "value counts"))
        == layer_count,
        "seed %d has an unexpected layer count" % seed,
    )
    expected_family = _expected_family_keys(layer_count)
    selected = _selected_action_keys(audit)
    required = expected_family | selected
    missing = sorted(required - set(candidates))
    _require(not missing, "seed %d candidate coverage misses %s" % (seed, missing[:3]))
    unexpected = sorted(set(candidates) - required)
    _require(
        not unexpected,
        "seed %d has undeclared candidate evaluations %s"
        % (seed, unexpected[:3]),
    )

    baseline = _metric_payload(audit.get("baseline_validation"), "baseline validation")
    for key, raw_evaluation in candidates.items():
        evaluation = _property_map(raw_evaluation, "candidate %s" % key)
        action = _property_map(evaluation.get("action"), "candidate action")
        _require(_action_key(action) == key, "seed %d candidate key/action mismatch" % seed)
        metrics = _metric_payload(evaluation.get("metrics"), "candidate metrics")
        full_metrics = _property_map(evaluation.get("metrics"), "candidate metrics")
        _require(
            int(full_metrics.get("examples", -1)) == 408
            and int(full_metrics.get("batches", -1)) == 26,
            "seed %d candidate %s was not evaluated on the complete validation split"
            % (seed, key),
        )
        delta = _metric_payload(
            evaluation.get("delta_from_unmerged"), "candidate delta"
        )
        for metric in METRICS:
            _require(
                math.isclose(
                    metrics[metric] - baseline[metric],
                    delta[metric],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ),
                "seed %d candidate %s has a stale %s delta"
                % (seed, key, metric),
            )

    oracles = _property_map(
        audit.get("validation_loss_oracles"), "validation_loss_oracles"
    )
    for mode in MODES:
        records = _property_map(oracles.get(mode), "validation oracle mode")
        _require(
            set(records) == {"query_only", "value_only", "joint_same_pair"},
            "seed %d has incomplete validation oracles for %s" % (seed, mode),
        )
        for record in records.values():
            oracle = _property_map(record, "validation oracle")
            key = str(oracle.get("action_key"))
            _require(key in candidates, "seed %d oracle action was not evaluated" % seed)
            _require(
                _metrics_close(
                    _property_map(oracle.get("validation_metrics"), "oracle metrics"),
                    _property_map(
                        _property_map(candidates[key], "candidate").get("metrics"),
                        "candidate metrics",
                    ),
                ),
                "seed %d oracle metrics disagree with candidate evaluation" % seed,
            )
    coverage = _property_map(
        audit.get("validation_loss_oracle_coverage"),
        "validation_loss_oracle_coverage",
    )
    _require(
        set(_sequence(coverage.get("enumerated_exactly"), "oracle coverage"))
        == {
            "every_query_only_pair",
            "every_value_only_pair",
            "every_joint_same_pair",
        },
        "seed %d overstates or omits oracle coverage" % seed,
    )
    return {
        "pass": True,
        "layer_count": layer_count,
        "required_family_action_count": len(expected_family),
        "selected_action_count": len(selected),
        "required_union_action_count": len(required),
        "actual_candidate_evaluation_count": len(candidates),
        "extra_selected_composite_count": len(set(candidates) - expected_family),
        "missing_action_count": 0,
        "unexpected_action_count": 0,
    }


def _validate_equivalence_and_restoration(
    audit: Mapping[str, object], seed: int
) -> Dict[str, object]:
    gate = _property_map(audit.get("equivalence_gate"), "equivalence_gate")
    _require(
        gate.get("canonical_action_audit_all_pass") is True,
        "seed %d failed the canonical action audit" % seed,
    )
    _require(
        gate.get("primary_canonical_action_repeatability_pass") is True,
        "seed %d failed deterministic canonical-action repeatability" % seed,
    )
    deterministic = _property_map(
        audit.get("deterministic_execution"), "deterministic_execution"
    )
    _require(
        deterministic.get("requested") is True
        and deterministic.get("torch_deterministic_algorithms_enabled") is True
        and deterministic.get("cudnn_deterministic") is True
        and deterministic.get("cudnn_benchmark") is False
        and deterministic.get("cuda_matmul_allow_tf32") is False
        and deterministic.get("cudnn_allow_tf32") is False,
        "seed %d did not use the declared deterministic inference protocol" % seed,
    )
    repeatability = _property_map(
        audit.get("primary_canonical_action_repeatability"),
        "primary_canonical_action_repeatability",
    )
    _require(
        repeatability.get("all_pass") is True
        and repeatability.get("exact_repeat_gate") is True
        and int(repeatability.get("required_complete_passes", -1)) == 2
        and int(repeatability.get("selected_action_count", 0)) > 0,
        "seed %d lacks an exact repeated primary-action evaluation gate" % seed,
    )
    _require(
        "factorized_float32_reexecution_all_pass" in gate,
        "seed %d omits the float32 diagnostic" % seed,
    )
    gauges = _sequence(audit.get("gauge_audits"), "gauge_audits")
    _require(len(gauges) == len(GAUGE_NAMES), "seed %d does not have ten gauges" % seed)
    for raw_gauge in gauges:
        gauge = _property_map(raw_gauge, "gauge audit")
        name = str(_property_map(gauge.get("gauge"), "gauge metadata").get("name"))
        required_flags = (
            "canonical_action_audit_pass",
            "restoration_exact",
            "parameter_identity_restored",
            "product_invariance_pass",
            "all_required_invariances_pass",
        )
        _require(
            all(gauge.get(flag) is True for flag in required_flags),
            "seed %d gauge %s failed a canonical invariance/restoration gate"
            % (seed, name),
        )
        _require(
            gauge.get("selection_audit_dtype") == "float64"
            and gauge.get("selection_inputs") == "raw_running_average_B_only"
            and gauge.get("validation_used_for_selection") is False,
            "seed %d gauge %s used the wrong selection path" % (seed, name),
        )
        distance_audit = _property_map(
            gauge.get("distance_gauge_audit"), "distance_gauge_audit"
        )
        _require(
            distance_audit.get("all_expected_invariances_pass") is True,
            "seed %d gauge %s failed a declared distance invariance" % (seed, name),
        )
        for mode in MODES:
            records = _property_map(
                _property_map(distance_audit.get("modes"), "distance modes").get(
                    mode
                ),
                "distance mode",
            )
            for invariant_name in (
                "current_effective_update_frobenius",
                "current_lower_local_activation_rms",
            ):
                invariant = _property_map(records.get(invariant_name), invariant_name)
                _require(
                    invariant.get("expected_invariant") is True
                    and invariant.get("invariance_pass") is True,
                    "seed %d gauge %s broke invariant baseline %s"
                    % (seed, name, invariant_name),
                )

    restoration = _property_map(audit.get("restoration"), "restoration")
    _require(
        restoration.get("no_merge_committed") is True
        and restoration.get("original_premerge_model_unchanged") is True
        and restoration.get("trainable_digest_before")
        == restoration.get("trainable_digest_after")
        and restoration.get("structural_parameter_identity_before")
        == restoration.get("structural_parameter_identity_after"),
        "seed %d did not restore the canonical pre-merge model exactly" % seed,
    )
    float32_pass = bool(gate.get("factorized_float32_reexecution_all_pass"))
    return {
        "pass": True,
        "canonical_action_audit_all_pass": True,
        "product_and_declared_distance_invariances_all_pass": True,
        "restoration_all_pass": True,
        "deterministic_primary_action_repeatability_all_pass": True,
        "factorized_float32_reexecution_all_pass": float32_pass,
        "factorized_float32_role": "diagnostic_only_not_scientific_gate",
    }


def _normalize_source_map(value: Mapping[str, object]) -> Dict[str, str]:
    normalized = {}
    for raw_name, raw_digest in value.items():
        name = str(raw_name).replace("\\", "/")
        matches = [path for path in SOURCE_PATHS if name.endswith(path)]
        if not matches:
            continue
        normalized[matches[0]] = str(raw_digest).lower()
    return normalized


def _source_hash_validation(
    provenances: Sequence[Mapping[str, object]], project_root: Path
) -> Dict[str, object]:
    local = {
        relative: _sha256_file(project_root / relative) for relative in SOURCE_PATHS
    }
    bundle_sha256 = _canonical_json_sha256(local)
    recorded = []
    for provenance in provenances:
        raw = provenance.get("source_sha256", provenance.get("source_hashes"))
        recorded.append(None if raw is None else _normalize_source_map(_property_map(raw, "source hashes")))
    availability = [item is not None for item in recorded]
    _require(
        all(availability) or not any(availability),
        "generation source hashes are present for only a subset of seeds",
    )
    if not any(availability):
        return {
            "generation_source_attestation_available": False,
            "generation_source_attestation_pass": False,
            "local_current_source_sha256": local,
            "local_current_source_bundle_sha256": bundle_sha256,
            "limitation": (
                "The three artifacts do not embed generation-time source hashes. "
                "These local hashes are reported for handoff only and do not prove "
                "which source bytes generated the artifacts."
            ),
        }
    expected_keys = set(SOURCE_PATHS)
    for index, item in enumerate(recorded):
        _require(
            set(item) == expected_keys,
            "seed %d generation source hash set is incomplete" % PRIMARY_SEEDS[index],
        )
        _require(
            item == local,
            "seed %d generation source hashes do not match local source"
            % PRIMARY_SEEDS[index],
        )
    _require(
        all(item == recorded[0] for item in recorded[1:]),
        "generation source hashes differ across seeds",
    )
    return {
        "generation_source_attestation_available": True,
        "generation_source_attestation_pass": True,
        "local_current_source_sha256": local,
        "local_current_source_bundle_sha256": bundle_sha256,
        "recorded_source_sha256": recorded[0],
    }


def _oracle_record(
    audit: Mapping[str, object], mode: str, scope: str
) -> Dict[str, object]:
    records = _property_map(
        _property_map(audit.get("validation_loss_oracles"), "oracles").get(mode),
        "oracle mode",
    )
    if scope == "joint":
        oracle = _property_map(records.get("joint_same_pair"), "joint oracle")
        return {
            "coverage": "exact_over_declared_joint_same_pair_candidate_family",
            "action_key": oracle.get("action_key"),
            "validation_metrics": _metric_payload(
                oracle.get("validation_metrics"), "joint oracle metrics"
            ),
        }
    components = {}
    for name in ("query_only", "value_only"):
        oracle = _property_map(records.get(name), "%s oracle" % name)
        components[name] = {
            "action_key": oracle.get("action_key"),
            "validation_metrics": _metric_payload(
                oracle.get("validation_metrics"), "%s metrics" % name
            ),
        }
    return {
        "coverage": "componentwise_query_and_value_only_not_full_cartesian",
        "components": components,
        "not_claimed": "exact oracle over all query-pair by value-pair composites",
    }


def _seed_mode_scope_row(
    audit: Mapping[str, object], seed: int, mode: str, scope: str
) -> Dict[str, object]:
    candidates = _property_map(audit.get("candidate_evaluations"), "candidates")
    gauge_audits = _sequence(audit.get("gauge_audits"), "gauge_audits")
    records = []
    for raw_gauge in gauge_audits:
        gauge = _property_map(raw_gauge, "gauge audit")
        name = str(_property_map(gauge.get("gauge"), "gauge").get("name"))
        modes = _property_map(gauge.get("modes"), "gauge modes")
        decision = _property_map(
            _property_map(
                _property_map(modes.get(mode), "gauge mode").get("decisions"),
                "gauge decisions",
            ).get(scope),
            "gauge decision scope",
        )
        action = _property_map(decision.get("action"), "gauge action")
        key = str(decision.get("action_key"))
        _require(_action_key(action) == key, "gauge decision action/key mismatch")
        _validate_action_for_mode(action, mode, scope)
        metrics = _candidate_metrics(candidates, key)
        embedded = _property_map(
            decision.get("original_gauge_candidate_evaluation"),
            "embedded candidate evaluation",
        )
        canonical_evaluation = _property_map(candidates[key], "candidate")
        _require(
            embedded == canonical_evaluation,
            "seed %d gauge %s embeds a stale canonical candidate evaluation"
            % (seed, name),
        )
        records.append({"gauge": name, "action_key": key, "metrics": metrics})
    identity_records = [record for record in records if record["gauge"] == "identity"]
    _require(len(identity_records) == 1, "seed %d lacks one identity gauge" % seed)
    identity = identity_records[0]
    changed = [record for record in records if record["action_key"] != identity["action_key"]]
    for record in records:
        record["delta_from_identity_action"] = _metric_difference(
            record["metrics"], identity["metrics"]
        )

    baseline_mode = _property_map(
        _property_map(audit.get("distance_baselines"), "distance baselines").get(
            mode
        ),
        "distance baseline mode",
    )
    selections = _property_map(baseline_mode.get("selections"), "baseline selections")

    def baseline_action(method: str) -> str:
        method_scopes = _property_map(selections.get(method), method)
        selection = _property_map(method_scopes.get(scope), "%s scope" % method)
        action = _property_map(selection.get("action"), "%s action" % method)
        _validate_action_for_mode(action, mode, scope)
        key = str(selection.get("action_key"))
        _require(_action_key(action) == key, "%s selection key is stale" % method)
        return key

    raw_action = baseline_action("running_raw_B_selection_proxy")
    _require(
        raw_action == identity["action_key"],
        "seed %d identity gauge does not reproduce the raw-B rule" % seed,
    )
    effective_action = baseline_action("current_effective_update_frobenius")
    activation_action = baseline_action("current_lower_local_activation_rms")
    raw_action_metrics = _candidate_metrics(candidates, raw_action)
    _require(
        _metrics_close(raw_action_metrics, identity["metrics"], tolerance=0.0),
        "seed %d identity raw-B metrics do not match the canonical candidate" % seed,
    )
    effective_action_metrics = _candidate_metrics(candidates, effective_action)
    activation_action_metrics = _candidate_metrics(candidates, activation_action)

    ranges = {
        "selected_action_validation_metrics": {
            metric: _range(record["metrics"][metric] for record in records)
            for metric in METRICS
        },
        "delta_from_identity_action": {
            metric: _range(
                record["delta_from_identity_action"][metric] for record in records
            )
            for metric in METRICS
        },
        "changed_gauges_delta_from_identity_action": (
            {
                metric: _range(
                    record["delta_from_identity_action"][metric]
                    for record in changed
                )
                for metric in METRICS
            }
            if changed
            else None
        ),
    }
    nonzero_consequence = any(
        abs(record["delta_from_identity_action"][metric]) > 1e-12
        for record in changed
        for metric in METRICS
    )
    return {
        "seed": seed,
        "candidate_mode": mode,
        "decision_scope": scope,
        "gauge_count": len(records),
        "identity_action_key": identity["action_key"],
        "action_change_count": len(changed),
        "changed_gauges": [record["gauge"] for record in changed],
        "unique_action_count": len({record["action_key"] for record in records}),
        "unique_actions": sorted({record["action_key"] for record in records}),
        "gauge_records": records,
        "consequence_ranges": ranges,
        "at_least_one_changed_action_has_nonzero_validation_consequence": nonzero_consequence,
        "baselines": {
            "raw_B_rule": {
                "name": "running_raw_B_selection_proxy",
                "action_key": raw_action,
                "selected_action_validation_metrics": raw_action_metrics,
                "delta_from_raw_B_action": {
                    metric: 0.0 for metric in METRICS
                },
                "input": "cumulative running-average B only",
                "data_usage": "prevalidation snapshot only",
                "gauge_property": "not invariant under a general GL(rank) gauge",
            },
            "gauge_invariant_current_effective_baseline": {
                "name": "current_effective_update_frobenius",
                "action_key": effective_action,
                "selected_action_validation_metrics": effective_action_metrics,
                "delta_from_raw_B_action": _metric_difference(
                    effective_action_metrics, raw_action_metrics
                ),
                "estimand": "Frobenius distance between current effective LoRA updates B_i A",
                "data_usage": "prevalidation snapshot only",
            },
            "gauge_invariant_current_local_activation_baseline": {
                "name": "current_lower_local_activation_rms",
                "action_key": activation_action,
                "selected_action_validation_metrics": activation_action_metrics,
                "delta_from_raw_B_action": _metric_difference(
                    activation_action_metrics, raw_action_metrics
                ),
                "estimand": "lower-layer local RMS LoRA output disturbance using current B and the lower layer activation moment",
                "data_usage": "complete validation inputs without labels, post hoc diagnostic",
            },
            "validation_loss_oracle": _oracle_record(audit, mode, scope),
        },
    }


def _aggregate_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    aggregates = []
    for mode in MODES:
        for scope in SCOPES:
            selected = [
                row
                for row in rows
                if row["candidate_mode"] == mode and row["decision_scope"] == scope
            ]
            _require(len(selected) == 3, "aggregate cell lacks three seeds")
            all_records = [
                record for row in selected for record in row["gauge_records"]
            ]
            aggregate = {
                "candidate_mode": mode,
                "decision_scope": scope,
                "seed_count": 3,
                "total_gauge_instances": len(all_records),
                "total_action_change_count": sum(
                    int(row["action_change_count"]) for row in selected
                ),
                "action_change_counts_by_seed": {
                    str(row["seed"]): row["action_change_count"] for row in selected
                },
                "seeds_with_action_change": sum(
                    int(row["action_change_count"] > 0) for row in selected
                ),
                "seeds_with_nonzero_validation_consequence": sum(
                    bool(
                        row[
                            "at_least_one_changed_action_has_nonzero_validation_consequence"
                        ]
                    )
                    for row in selected
                ),
                "unique_actions_across_seeds": sorted(
                    {
                        action
                        for row in selected
                        for action in row["unique_actions"]
                    }
                ),
                "delta_from_seed_identity_action": {
                    metric: _range(
                        record["delta_from_identity_action"][metric]
                        for record in all_records
                    )
                    for metric in METRICS
                },
            }
            aggregate["chain_holds_in_all_three_seeds"] = (
                aggregate["seeds_with_action_change"] == 3
                and aggregate["seeds_with_nonzero_validation_consequence"] == 3
            )
            aggregates.append(aggregate)
    return aggregates


def summarize_aslora_audits(
    audits: Sequence[Mapping[str, object]],
    provenances: Sequence[Mapping[str, object]],
    project_root: Path = PROJECT_ROOT,
    input_records: Optional[Sequence[Mapping[str, object]]] = None,
) -> Dict[str, object]:
    """Strictly validate three raw audits and compute seed/mode/scope summaries."""

    _require(len(audits) == 3, "the ASLoRA summary requires exactly three audits")
    _require(
        len(provenances) == len(audits),
        "every audit requires its adjacent provenance.json",
    )
    indexed: Dict[int, Tuple[Mapping[str, object], Mapping[str, object]]] = {}
    protocol_hashes: Dict[str, str] = {}
    candidate_gates: Dict[str, object] = {}
    equivalence_gates: Dict[str, object] = {}
    for audit, provenance in zip(audits, provenances):
        config = _property_map(audit.get("config"), "config")
        seed = int(config.get("seed", -1))
        _require(seed in PRIMARY_SEEDS, "unexpected ASLoRA seed %d" % seed)
        _require(seed not in indexed, "duplicate ASLoRA seed %d" % seed)
        protocol_hashes[str(seed)] = _validate_expected_protocol(audit, provenance, seed)
        candidate_gates[str(seed)] = _validate_candidate_coverage(audit, seed)
        equivalence_gates[str(seed)] = _validate_equivalence_and_restoration(
            audit, seed
        )
        indexed[seed] = (audit, provenance)
    _require(
        tuple(sorted(indexed)) == PRIMARY_SEEDS,
        "audits must contain seeds 0, 1, and 2",
    )
    _require(
        len(set(protocol_hashes.values())) == 1,
        "protocol projections differ across seeds",
    )
    ordered_audits = [indexed[seed][0] for seed in PRIMARY_SEEDS]
    ordered_provenances = [indexed[seed][1] for seed in PRIMARY_SEEDS]
    source_gate = _source_hash_validation(ordered_provenances, project_root)
    rows = [
        _seed_mode_scope_row(indexed[seed][0], seed, mode, scope)
        for seed in PRIMARY_SEEDS
        for mode in MODES
        for scope in SCOPES
    ]
    aggregates = _aggregate_rows(rows)
    aggregate_index = {
        (record["candidate_mode"], record["decision_scope"]): record
        for record in aggregates
    }
    primary_cells = [aggregate_index[(mode, "per_projection")] for mode in MODES]
    primary_chain = all(record["chain_holds_in_all_three_seeds"] for record in primary_cells)
    joint_cells = [aggregate_index[(mode, "joint")] for mode in MODES]
    joint_chain = all(record["chain_holds_in_all_three_seeds"] for record in joint_cells)
    canonical_gate = all(record["pass"] for record in equivalence_gates.values())
    coverage_gate = all(record["pass"] for record in candidate_gates.values())
    result_integrity_gate = canonical_gate and coverage_gate
    fully_attested = (
        result_integrity_gate
        and bool(source_gate["generation_source_attestation_available"])
        and bool(source_gate["generation_source_attestation_pass"])
    )
    source_conclusion = (
        "The generation-time source hashes embedded in all three artifacts "
        "match one another and the released source snapshot."
        if fully_attested
        else (
            "Generation-time source hashes are absent or mismatched, so the "
            "observed chain is not a fully source-attested reproduction."
        )
    )
    dataset_processing_fingerprints = {
        str(seed): {
            "train_fingerprint": _property_map(
                indexed[seed][1].get("dataset"), "provenance.dataset"
            ).get("train_fingerprint"),
            "validation_fingerprint": _property_map(
                indexed[seed][1].get("dataset"), "provenance.dataset"
            ).get("validation_fingerprint"),
        }
        for seed in PRIMARY_SEEDS
    }
    generation_input_bindings = {
        str(seed): {
            "initial_attached_model_state_sha256": _property_map(
                indexed[seed][1].get("input_artifacts"),
                "provenance.input_artifacts",
            ).get("initial_attached_model_state_sha256"),
            **{
                "%s_inventory_sha256" % name: _property_map(
                    _property_map(
                        indexed[seed][1].get("input_artifacts"),
                        "provenance.input_artifacts",
                    ).get(name),
                    "input_artifacts.%s" % name,
                ).get("inventory_sha256")
                for name in ("local_model", "train_parquet", "validation_parquet")
            },
        }
        for seed in PRIMARY_SEEDS
    }
    summary: Dict[str, object] = {
        "format": "aslora-three-seed-first-merge-summary-v1",
        "protocol": {
            "seeds": list(PRIMARY_SEEDS),
            "candidate_modes": list(MODES),
            "decision_scopes": list(SCOPES),
            "primary_decision_scope": "per_projection",
            "gauge_count_per_seed": len(GAUGE_NAMES),
            "gauge_names": list(GAUGE_NAMES),
            "canonical_protocol_sha256": next(iter(protocol_hashes.values())),
            "per_seed_protocol_sha256": protocol_hashes,
            "cross_seed_protocol_hash_pass": True,
            "per_seed_dataset_processing_fingerprints": dataset_processing_fingerprints,
            "generation_input_artifact_bindings": generation_input_bindings,
            "dataset_fingerprint_note": (
                "Hugging Face processing fingerprints are retained as secondary "
                "metadata. The local model directory and both parquet files are "
                "also byte-bound at generation; their inventory SHA256 values "
                "are part of the cross-seed protocol identity."
            ),
        },
        "validation": {
            "candidate_coverage": candidate_gates,
            "candidate_coverage_all_pass": coverage_gate,
            "equivalence_and_restoration": equivalence_gates,
            "canonical_scientific_gate_all_pass": canonical_gate,
            "source_hashes": source_gate,
            "result_integrity_gate_without_generation_source_attestation": result_integrity_gate,
            "fully_attested_reproducibility_gate": fully_attested,
        },
        "definitions": {
            "raw_B_rule": (
                "ASLoRA's first action minimizes Euclidean distance between "
                "cumulative running-average layer-specific B factors."
            ),
            "gauge_invariant_current_effective_baseline": (
                "Frobenius distance between current effective updates B_i A."
            ),
            "gauge_invariant_current_local_activation_baseline": (
                "Exact lower-layer local RMS output disturbance under the current "
                "B substitution and lower-layer activation moment; it is not "
                "network validation loss."
            ),
            "validation_loss_oracle": (
                "Post-hoc complete labeled validation enumeration over every "
                "query-only pair, value-only pair, and joint same-pair action. "
                "It is never used for raw gauge selection, and it is not a full "
                "query-pair by value-pair Cartesian oracle."
            ),
            "canonical_action_audit": (
                "Gauge-transformed float64 factors select a layer-index action; "
                "that action is mapped to and evaluated in the unchanged canonical model."
            ),
            "factorized_float32_reexecution": (
                "Numerical diagnostic only; finite-precision reassociation may alter "
                "logits and it is not the equivalence gate."
            ),
        },
        "seed_mode_scope_rows": rows,
        "across_seed_mode_scope": aggregates,
        "chain_assessment": {
            "primary_per_projection_chain_holds_for_both_candidate_modes_in_all_three_seeds": primary_chain,
            "joint_sensitivity_chain_holds_for_both_candidate_modes_in_all_three_seeds": joint_chain,
            "canonical_scientific_chain_observed_across_seeds": (
                primary_chain and result_integrity_gate
            ),
            "fully_source_attested_chain": primary_chain and fully_attested,
            "conclusion": (
                "Across all three seeds, in the primary per-projection scope and "
                "under both adjacent and all-pairs candidate interpretations, "
                "gauge-equivalent running factors select different raw-B actions "
                "and at least one changed action has a different canonical validation "
                "outcome from the identity-gauge action. The joint sensitivity result "
                "is not uniform across both modes. " + source_conclusion
            ),
        },
    }
    if input_records is not None:
        summary["inputs"] = list(input_records)
    return summary


def summarize_paths(
    input_paths: Sequence[Path], project_root: Path = PROJECT_ROOT
) -> Dict[str, object]:
    _require(len(input_paths) == 3, "exactly three audit.json paths are required")
    audits = []
    provenances = []
    records = []
    for raw_path in input_paths:
        path = raw_path.resolve()
        _require(path.is_file(), "missing audit file %s" % path)
        provenance_path = path.parent / "provenance.json"
        _require(provenance_path.is_file(), "missing %s" % provenance_path)
        audit = _read_json(path)
        provenance = _read_json(provenance_path)
        snapshot_path = path.parent / "pre_first_merge_snapshot.pt"
        records.append(
            {
                "seed": int(_property_map(audit.get("config"), "config").get("seed")),
                "audit_path": _portable_input_path(path),
                "audit_sha256": _sha256_file(path),
                "provenance_path": _portable_input_path(provenance_path),
                "provenance_sha256": _sha256_file(provenance_path),
                "snapshot_path": _portable_input_path(snapshot_path),
                "snapshot_sha256": (
                    _sha256_file(snapshot_path) if snapshot_path.is_file() else None
                ),
            }
        )
        audits.append(audit)
        provenances.append(provenance)
    return summarize_aslora_audits(
        audits,
        provenances,
        project_root=project_root,
        input_records=records,
    )


def _csv_rows(summary: Mapping[str, object]) -> List[Dict[str, object]]:
    flattened = []
    for raw_row in summary["seed_mode_scope_rows"]:
        row = _property_map(raw_row, "summary row")
        ranges = _property_map(row["consequence_ranges"], "consequence ranges")
        deltas = _property_map(ranges["delta_from_identity_action"], "deltas")
        baselines = _property_map(row["baselines"], "baselines")
        oracle = _property_map(baselines["validation_loss_oracle"], "oracle")
        flattened.append(
            {
                "seed": row["seed"],
                "candidate_mode": row["candidate_mode"],
                "decision_scope": row["decision_scope"],
                "gauge_count": row["gauge_count"],
                "identity_action_key": row["identity_action_key"],
                "action_change_count": row["action_change_count"],
                "changed_gauges": ";".join(row["changed_gauges"]),
                "unique_action_count": row["unique_action_count"],
                "unique_actions": ";".join(row["unique_actions"]),
                "loss_delta_min": deltas["loss"]["min"],
                "loss_delta_max": deltas["loss"]["max"],
                "accuracy_delta_min": deltas["accuracy"]["min"],
                "accuracy_delta_max": deltas["accuracy"]["max"],
                "f1_delta_min": deltas["f1"]["min"],
                "f1_delta_max": deltas["f1"]["max"],
                "raw_B_action": baselines["raw_B_rule"]["action_key"],
                "current_effective_action": baselines[
                    "gauge_invariant_current_effective_baseline"
                ]["action_key"],
                "current_effective_loss_delta_from_raw_B": baselines[
                    "gauge_invariant_current_effective_baseline"
                ]["delta_from_raw_B_action"]["loss"],
                "current_local_activation_action": baselines[
                    "gauge_invariant_current_local_activation_baseline"
                ]["action_key"],
                "current_local_activation_loss_delta_from_raw_B": baselines[
                    "gauge_invariant_current_local_activation_baseline"
                ]["delta_from_raw_B_action"]["loss"],
                "validation_oracle_coverage": oracle["coverage"],
                "validation_oracle_actions": (
                    oracle.get("action_key")
                    if "action_key" in oracle
                    else ";".join(
                        value["action_key"] for value in oracle["components"].values()
                    )
                ),
            }
        )
    return flattened


def markdown_report(summary: Mapping[str, object]) -> str:
    validation = summary["validation"]
    source = validation["source_hashes"]
    source_note = (
        "Generation-time source bundle SHA256: `%s`; it matches the current "
        "released source snapshot."
        % source["local_current_source_bundle_sha256"]
        if source["generation_source_attestation_pass"]
        else (
            "Local current source bundle SHA256: `%s`. This is not a "
            "generation-time attestation for the present artifacts."
            % source["local_current_source_bundle_sha256"]
        )
    )
    lines = [
        "# ASLoRA three-seed first-merge audit",
        "",
        "Scientific gate: canonical action audit, candidate coverage, and restoration all pass: **%s**."
        % str(validation["result_integrity_gate_without_generation_source_attestation"]).lower(),
        "Generation-time source hashes embedded in all three artifacts: **%s**."
        % str(source["generation_source_attestation_available"]).lower(),
        "Factorized float32 re-execution is diagnostic only and is not used as the equivalence gate.",
        "",
        "| Seed | Candidates | Scope | Changed / 10 gauges | Unique actions | Identity raw-B action | Loss delta range | Accuracy delta range | F1 delta range |",
        "|---:|:---|:---|---:|---:|:---|:---|:---|:---|",
    ]
    for row in summary["seed_mode_scope_rows"]:
        delta = row["consequence_ranges"]["delta_from_identity_action"]
        lines.append(
            "| {seed} | {mode} | {scope} | {changed}/10 | {unique} | `{identity}` | [{lmin:+.6f}, {lmax:+.6f}] | [{amin:+.6f}, {amax:+.6f}] | [{fmin:+.6f}, {fmax:+.6f}] |".format(
                seed=row["seed"],
                mode=row["candidate_mode"],
                scope=row["decision_scope"],
                changed=row["action_change_count"],
                unique=row["unique_action_count"],
                identity=row["identity_action_key"],
                lmin=delta["loss"]["min"],
                lmax=delta["loss"]["max"],
                amin=delta["accuracy"]["min"],
                amax=delta["accuracy"]["max"],
                fmin=delta["f1"]["min"],
                fmax=delta["f1"]["max"],
            )
        )
    lines.extend(
        [
            "",
            "## Across seeds",
            "",
            "| Candidates | Scope | Changed / 30 gauge instances | Seeds with change | Global loss delta | Global accuracy delta | Global F1 delta | Chain in all seeds |",
            "|:---|:---|---:|---:|:---|:---|:---|:---|",
        ]
    )
    for record in summary["across_seed_mode_scope"]:
        delta = record["delta_from_seed_identity_action"]
        lines.append(
            "| {mode} | {scope} | {changed}/30 | {seeds}/3 | [{lmin:+.6f}, {lmax:+.6f}] | [{amin:+.6f}, {amax:+.6f}] | [{fmin:+.6f}, {fmax:+.6f}] | {chain} |".format(
                mode=record["candidate_mode"],
                scope=record["decision_scope"],
                changed=record["total_action_change_count"],
                seeds=record["seeds_with_action_change"],
                lmin=delta["loss"]["min"],
                lmax=delta["loss"]["max"],
                amin=delta["accuracy"]["min"],
                amax=delta["accuracy"]["max"],
                fmin=delta["f1"]["min"],
                fmax=delta["f1"]["max"],
                chain=str(record["chain_holds_in_all_three_seeds"]).lower(),
            )
        )
    lines.extend(
        [
            "",
            "## Baseline meanings",
            "",
            "- **Raw-B rule:** ASLoRA's cumulative running-average `B` Euclidean rule; prevalidation and gauge dependent.",
            "- **Current effective baseline:** current `B_i A` Frobenius distance; gauge invariant and prevalidation.",
            "- **Current local activation baseline:** lower-layer RMS output disturbance on unlabeled validation inputs; gauge invariant, post hoc, and not network loss.",
            "- **Validation oracle:** labeled post-hoc enumeration of query-only, value-only, and joint-same-pair families. Per-projection Cartesian composites are not exhaustively enumerated.",
            "",
            "## Conclusion",
            "",
            summary["chain_assessment"]["conclusion"],
            "",
            source_note,
        ]
    )
    return "\n".join(lines) + "\n"


def write_summary_outputs(
    summary: Mapping[str, object],
    json_output: Path,
    csv_output: Path,
    markdown_output: Path,
) -> None:
    for path in (json_output, csv_output, markdown_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    with json_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
    rows = _csv_rows(summary)
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with markdown_output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_report(summary))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="three audit.json files; defaults to results/aslora_seed{0,1,2}",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "aslora_summary.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "aslora_summary.csv",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=PROJECT_ROOT / "results" / "aslora_summary.md",
    )
    args = parser.parse_args(argv)
    paths = tuple(args.inputs) if args.inputs else DEFAULT_INPUTS
    summary = summarize_paths(paths)
    write_summary_outputs(
        summary, args.json_output, args.csv_output, args.markdown_output
    )
    print(
        "wrote %s, %s, and %s"
        % (args.json_output, args.csv_output, args.markdown_output)
    )


if __name__ == "__main__":
    main()
