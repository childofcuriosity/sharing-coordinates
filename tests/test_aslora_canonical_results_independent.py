"""Independent adversarial checks for the released canonical ASLoRA records.

These tests deliberately do not import ``summarize_aslora_audit`` or any of the
three generation modules whose hashes are recorded by the artifacts.  The raw
JSON and frozen tensor snapshots are treated as the inputs, and the reported
rankings, action counts, consequence ranges, coverage, and byte bindings are
reconstructed here from first principles.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
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

Pair = Tuple[int, int]
Action = Dict[str, Pair]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _seed_paths(seed: int) -> Tuple[Path, Path, Path]:
    directory = ROOT / "results" / ("aslora_seed%d" % seed)
    return (
        directory / "audit.json",
        directory / "provenance.json",
        directory / "pre_first_merge_snapshot.pt",
    )


def _load_snapshot(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 1.11, used to generate the records.
        return torch.load(path, map_location="cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pairs(mode: str, layer_count: int = 12) -> Tuple[Pair, ...]:
    if mode == "adjacent":
        return tuple((index, index + 1) for index in range(layer_count - 1))
    if mode == "all":
        return tuple(itertools.combinations(range(layer_count), 2))
    raise AssertionError("unknown mode %s" % mode)


def _pair_text(pair: Pair) -> str:
    return "%d-%d" % pair


def _action_from_json(value: Mapping[str, Sequence[int]]) -> Action:
    return {target: (int(pair[0]), int(pair[1])) for target, pair in value.items()}


def _action_key(action: Mapping[str, Pair]) -> str:
    return "|".join(
        "%s:%d-%d" % (target, pair[0], pair[1])
        for target, pair in sorted(action.items())
    )


def _argmin(values: Mapping[Pair, float]) -> Pair:
    return min(values, key=lambda pair: (values[pair], pair))


def _raw_distances(running_b: torch.Tensor, pairs: Iterable[Pair]) -> Dict[Pair, float]:
    result: Dict[Pair, float] = {}
    for lower, upper in pairs:
        difference = running_b[lower] - running_b[upper]
        result[(lower, upper)] = float(torch.linalg.vector_norm(difference))
    return result


def _assert_metric_record(record: Mapping[str, Any], baseline: Mapping[str, float]) -> None:
    metrics = record["metrics"]
    assert metrics["examples"] == 408
    assert metrics["batches"] == 26
    for metric in ("loss", "accuracy", "f1"):
        assert math.isfinite(float(metrics[metric]))
        assert record["delta_from_unmerged"][metric] == pytest.approx(
            metrics[metric] - baseline[metric], abs=1e-15
        )


def test_released_file_source_and_recorded_input_bindings() -> None:
    """Rehash every released input and every inventory manifest independently."""

    summary = _read_json(ROOT / "results" / "aslora_summary.json")
    summary_inputs = {int(record["seed"]): record for record in summary["inputs"]}
    common_external_inventories: Dict[str, Dict[str, Any]] = {}
    initial_state_hashes = set()

    for seed in SEEDS:
        audit_path, provenance_path, snapshot_path = _seed_paths(seed)
        audit = _read_json(audit_path)
        provenance = _read_json(provenance_path)
        summary_record = summary_inputs[seed]

        assert summary_record["audit_sha256"] == _sha256(audit_path)
        assert summary_record["provenance_sha256"] == _sha256(provenance_path)
        assert summary_record["snapshot_sha256"] == _sha256(snapshot_path)

        assert audit["source_sha256"] == provenance["source_sha256"]
        assert set(audit["source_sha256"]) == set(SOURCE_PATHS)
        for relative_path in SOURCE_PATHS:
            assert audit["source_sha256"][relative_path] == _sha256(ROOT / relative_path)

        inputs = provenance["input_artifacts"]
        initial_hash = inputs["initial_attached_model_state_sha256"]
        assert len(initial_hash) == 64
        int(initial_hash, 16)
        initial_state_hashes.add(initial_hash)
        for name in ("local_model", "train_parquet", "validation_parquet"):
            inventory = inputs[name]
            files = inventory["files"]
            assert inventory["file_count"] == len(files)
            assert files
            relative_paths = [entry["relative_path"] for entry in files]
            assert relative_paths == sorted(relative_paths)
            assert len(relative_paths) == len(set(relative_paths))
            for entry in files:
                assert set(entry) == {"relative_path", "sha256", "size_bytes"}
                assert entry["size_bytes"] >= 0
                assert len(entry["sha256"]) == 64
                int(entry["sha256"], 16)
            # Recompute rather than trust the stored digest; the production
            # summarizer now enforces the same tamper-evident relation.
            assert inventory["inventory_sha256"] == _canonical_json_sha256(files)
            if name not in common_external_inventories:
                common_external_inventories[name] = inventory
            else:
                assert inventory == common_external_inventories[name]

            summary_binding = summary["protocol"]["generation_input_artifact_bindings"][
                str(seed)
            ][name + "_inventory_sha256"]
            assert summary_binding == inventory["inventory_sha256"]
        assert summary["protocol"]["generation_input_artifact_bindings"][str(seed)][
            "initial_attached_model_state_sha256"
        ] == initial_hash

    # Different training seeds must not silently share one attached initial
    # state digest, even though the external base model and data bytes agree.
    assert len(initial_state_hashes) == len(SEEDS)


def test_snapshot_recomputes_every_raw_distance_and_selected_action() -> None:
    """Rebuild all 3,960 serialized raw distances without generator imports."""

    minimum_gap = math.inf
    for seed in SEEDS:
        audit_path, _, snapshot_path = _seed_paths(seed)
        audit = _read_json(audit_path)
        snapshot = _load_snapshot(snapshot_path)
        assert snapshot["format"] == "aslora-first-merge-v1"
        assert tuple(sorted(snapshot["targets"])) == TARGETS
        assert [record["gauge"]["name"] for record in audit["gauge_audits"]] == list(
            GAUGE_NAMES
        )

        for gauge_index, gauge_record in enumerate(audit["gauge_audits"]):
            gauge = gauge_record["gauge"]
            direct: Dict[str, Dict[Pair, float]] = {}
            for target in TARGETS:
                state = snapshot["targets"][target]
                assert tuple(state["shared_a"].shape) == (8, 768)
                assert tuple(state["layer_b"].shape) == (12, 768, 8)
                assert tuple(state["running_b"].shape) == (12, 768, 8)
                assert state["running_count"] == 560

                q = torch.tensor(gauge["matrices_Q"][target], dtype=torch.float64)
                assert tuple(q.shape) == (8, 8)
                condition_number = float(torch.linalg.cond(q))
                assert condition_number == pytest.approx(
                    gauge["condition_numbers"][target], rel=2e-13, abs=2e-13
                )
                assert condition_number <= gauge["requested_condition_limit"] * (
                    1.0 + 2e-13
                )
                if gauge_index == 0:
                    assert torch.equal(q, torch.eye(8, dtype=torch.float64))
                elif gauge_index == 1:
                    assert torch.allclose(
                        q.T @ q,
                        torch.eye(8, dtype=torch.float64),
                        atol=2e-14,
                        rtol=2e-14,
                    )

                running = state["running_b"].to(dtype=torch.float64)
                direct[target] = _raw_distances(running @ q, _pairs("all"))

                # A cheap but complete rank-space check of the product gauge:
                # Q solve(Q,A) must recover every entry of A.
                shared_a = state["shared_a"].to(dtype=torch.float64)
                recovered_a = q @ torch.linalg.solve(q, shared_a)
                assert torch.allclose(recovered_a, shared_a, atol=2e-14, rtol=2e-13)

            for mode in MODES:
                pairs = _pairs(mode)
                serialized = gauge_record["modes"][mode]["distance_table"]
                for target in TARGETS:
                    table = serialized["per_projection"][target][
                        "running_raw_B_selection_proxy"
                    ]
                    assert set(table) == {_pair_text(pair) for pair in pairs}
                    for pair in pairs:
                        assert table[_pair_text(pair)] == pytest.approx(
                            direct[target][pair], rel=2e-12, abs=2e-12
                        )

                joint_direct = {
                    pair: math.hypot(direct["query"][pair], direct["value"][pair])
                    for pair in pairs
                }
                joint_table = serialized["joint"]["running_raw_B_selection_proxy"]
                for pair in pairs:
                    assert joint_table[_pair_text(pair)] == pytest.approx(
                        joint_direct[pair], rel=2e-12, abs=2e-12
                    )

                expected_per_projection = {
                    target: _argmin({pair: direct[target][pair] for pair in pairs})
                    for target in TARGETS
                }
                expected_joint_pair = _argmin(joint_direct)
                expected_joint = {target: expected_joint_pair for target in TARGETS}
                decisions = gauge_record["modes"][mode]["decisions"]
                for scope, expected in (
                    ("per_projection", expected_per_projection),
                    ("joint", expected_joint),
                ):
                    assert _action_from_json(decisions[scope]["action"]) == expected
                    assert decisions[scope]["action_key"] == _action_key(expected)

                # Record the closest first/second ranking.  The weakest released
                # margin is still >4.6e-4, far above the CPU/GPU float64 replay
                # discrepancy checked above.
                for values in (
                    {pair: direct["query"][pair] for pair in pairs},
                    {pair: direct["value"][pair] for pair in pairs},
                    joint_direct,
                ):
                    ordered = sorted(values.values())
                    minimum_gap = min(minimum_gap, ordered[1] - ordered[0])

    assert minimum_gap == pytest.approx(0.0004627883962863566, rel=2e-10, abs=2e-12)


def test_identity_baseline_candidate_execution_coverage_and_repeatability() -> None:
    """Audit identity controls, action materialization, oracles, and repeats."""

    all_pairs = _pairs("all")
    base_actions: Dict[str, Action] = {}
    for pair in all_pairs:
        for action in (
            {"query": pair},
            {"value": pair},
            {"query": pair, "value": pair},
        ):
            base_actions[_action_key(action)] = action
    assert len(base_actions) == 198

    for seed, expected_count in zip(SEEDS, (204, 208, 200)):
        audit_path, _, _ = _seed_paths(seed)
        audit = _read_json(audit_path)
        candidates = audit["candidate_evaluations"]
        baseline = audit["baseline_validation"]

        expected_actions = dict(base_actions)
        raw_selected_keys = set()
        for gauge_record in audit["gauge_audits"]:
            per_gauge_keys = set()
            for mode in MODES:
                for scope in SCOPES:
                    decision = gauge_record["modes"][mode]["decisions"][scope]
                    action = _action_from_json(decision["action"])
                    key = _action_key(action)
                    assert key == decision["action_key"]
                    expected_actions[key] = action
                    raw_selected_keys.add(key)
                    per_gauge_keys.add(key)
                    assert decision["original_gauge_candidate_evaluation"] == candidates[key]
            assert set(gauge_record["selected_fixed_action_audits"]) == per_gauge_keys

        for mode in MODES:
            baseline_record = audit["distance_baselines"][mode]
            identity_decisions = audit["gauge_audits"][0]["modes"][mode]["decisions"]
            raw_baseline = baseline_record["selections"][
                "running_raw_B_selection_proxy"
            ]
            for scope in SCOPES:
                # This is the exact identity action against which the summary
                # computes its gauge-relative changes.
                assert raw_baseline[scope] == {
                    key: identity_decisions[scope][key] for key in ("action", "action_key")
                }
            for metric_records in baseline_record["selections"].values():
                for selection in metric_records.values():
                    action = _action_from_json(selection["action"])
                    assert selection["action_key"] == _action_key(action)
                    expected_actions[selection["action_key"]] = action

        assert audit["candidate_evaluation_count"] == expected_count
        assert len(candidates) == expected_count
        assert set(candidates) == set(expected_actions)
        for key, record in candidates.items():
            action = _action_from_json(record["action"])
            assert key == _action_key(action)
            assert action == expected_actions[key]
            _assert_metric_record(record, baseline)

        # Recompute each declared validation-loss oracle over its exact family.
        for mode in MODES:
            pairs = set(_pairs(mode))
            families = {
                "query_only": [
                    _action_key({"query": pair}) for pair in pairs
                ],
                "value_only": [
                    _action_key({"value": pair}) for pair in pairs
                ],
                "joint_same_pair": [
                    _action_key({"query": pair, "value": pair}) for pair in pairs
                ],
            }
            for family, keys in families.items():
                best = min(keys, key=lambda key: (candidates[key]["metrics"]["loss"], key))
                oracle = audit["validation_loss_oracles"][mode][family]
                assert oracle["action_key"] == best
                assert oracle["action"] == candidates[best]["action"]
                assert oracle["validation_metrics"] == candidates[best]["metrics"]

        repeats = audit["primary_canonical_action_repeatability"]
        assert repeats["required_complete_passes"] == 2
        assert repeats["all_pass"] is True
        assert repeats["exact_repeat_gate"] is True
        assert repeats["deterministic_algorithms_required"] is True
        assert set(repeats["selected_action_keys"]) == raw_selected_keys
        assert repeats["selected_action_count"] == len(raw_selected_keys)
        assert len(repeats["records"]) == 1
        for record in repeats["records"]:
            assert record["exact_repeat_pass"] is True
            for difference in (
                [record["baseline_difference"]]
                + list(record["selected_action_differences"].values())
            ):
                assert set(difference.values()) == {0.0}

        restoration = audit["restoration"]
        assert restoration["no_merge_committed"] is True
        assert restoration["original_premerge_model_unchanged"] is True
        assert restoration["trainable_digest_before"] == restoration["trainable_digest_after"]
        assert restoration["structural_parameter_identity_before"] == restoration[
            "structural_parameter_identity_after"
        ]


def test_independent_primary_counts_chain_and_metric_ranges_match_summary() -> None:
    """Recompute 11/30, 14/30, both 3/3 chains, and every range."""

    summary = _read_json(ROOT / "results" / "aslora_summary.json")
    indexed_summary = {
        (row["candidate_mode"], row["decision_scope"]): row
        for row in summary["across_seed_mode_scope"]
    }
    audits = {
        seed: _read_json(_seed_paths(seed)[0])
        for seed in SEEDS
    }

    recomputed: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for mode in MODES:
        for scope in SCOPES:
            per_seed_changes: Dict[str, int] = {}
            seeds_with_consequence = 0
            all_deltas = {metric: [] for metric in ("loss", "accuracy", "f1")}
            unique_actions = set()
            for seed in SEEDS:
                audit = audits[seed]
                gauges = audit["gauge_audits"]
                identity_key = gauges[0]["modes"][mode]["decisions"][scope]["action_key"]
                identity_metrics = audit["candidate_evaluations"][identity_key]["metrics"]
                changed_with_consequence = False
                change_count = 0
                for gauge in gauges:
                    key = gauge["modes"][mode]["decisions"][scope]["action_key"]
                    unique_actions.add(key)
                    metrics = audit["candidate_evaluations"][key]["metrics"]
                    deltas = {
                        metric: metrics[metric] - identity_metrics[metric]
                        for metric in all_deltas
                    }
                    for metric, value in deltas.items():
                        all_deltas[metric].append(value)
                    if key != identity_key:
                        change_count += 1
                        if any(value != 0.0 for value in deltas.values()):
                            changed_with_consequence = True
                per_seed_changes[str(seed)] = change_count
                seeds_with_consequence += int(changed_with_consequence)

            record = {
                "action_change_counts_by_seed": per_seed_changes,
                "total_action_change_count": sum(per_seed_changes.values()),
                "seeds_with_action_change": sum(value > 0 for value in per_seed_changes.values()),
                "seeds_with_nonzero_validation_consequence": seeds_with_consequence,
                "chain_holds_in_all_three_seeds": (
                    all(value > 0 for value in per_seed_changes.values())
                    and seeds_with_consequence == 3
                ),
                "delta_from_seed_identity_action": {
                    metric: {"min": min(values), "max": max(values)}
                    for metric, values in all_deltas.items()
                },
                "unique_actions_across_seeds": sorted(unique_actions),
            }
            recomputed[(mode, scope)] = record

            reported = indexed_summary[(mode, scope)]
            for key in (
                "action_change_counts_by_seed",
                "total_action_change_count",
                "seeds_with_action_change",
                "seeds_with_nonzero_validation_consequence",
                "chain_holds_in_all_three_seeds",
                "unique_actions_across_seeds",
            ):
                assert reported[key] == record[key]
            for metric in all_deltas:
                for endpoint in ("min", "max"):
                    assert reported["delta_from_seed_identity_action"][metric][endpoint] == (
                        pytest.approx(
                            record["delta_from_seed_identity_action"][metric][endpoint],
                            abs=1e-15,
                        )
                    )

    adjacent = recomputed[("adjacent", "per_projection")]
    all_pairs = recomputed[("all", "per_projection")]
    assert adjacent["action_change_counts_by_seed"] == {"0": 2, "1": 6, "2": 3}
    assert adjacent["total_action_change_count"] == 11
    assert adjacent["chain_holds_in_all_three_seeds"] is True
    assert adjacent["delta_from_seed_identity_action"]["loss"] == pytest.approx(
        {"min": -0.0530944957452662, "max": 0.00886300556799946}, abs=1e-15
    )
    assert all_pairs["action_change_counts_by_seed"] == {"0": 4, "1": 7, "2": 3}
    assert all_pairs["total_action_change_count"] == 14
    assert all_pairs["chain_holds_in_all_three_seeds"] is True
    assert all_pairs["delta_from_seed_identity_action"]["loss"] == pytest.approx(
        {"min": -0.0530944957452662, "max": 0.002723619049670667}, abs=1e-15
    )

    # The negative sensitivity result is part of the anti-selection audit.
    assert recomputed[("adjacent", "joint")]["action_change_counts_by_seed"] == {
        "0": 0,
        "1": 5,
        "2": 0,
    }
    assert recomputed[("all", "joint")]["total_action_change_count"] == 0


def test_canonical_protocol_and_paper_numbers_are_not_stale() -> None:
    """Pin the declared execution boundary and prevent old-run prose regressions."""

    for seed in SEEDS:
        audit_path, provenance_path, _ = _seed_paths(seed)
        audit = _read_json(audit_path)
        provenance = _read_json(provenance_path)
        config = audit["config"]
        assert config == provenance["config"]
        assert config["seed"] == seed
        assert config["rank"] == 8
        assert config["alpha"] == 16.0
        assert config["epochs"] == 30
        assert config["train_batch_size"] == 16
        assert config["eval_batch_size"] == 16
        assert config["first_merge_step"] == 560
        assert config["start_merge_step"] == 320
        assert config["merge_interval"] == 240
        assert config["candidate_modes"] == list(MODES)
        assert config["decision_scopes"] == list(SCOPES)
        assert config["primary_decision_scope"] == "per_projection"
        assert config["primary_action_evaluation_repeats"] == 2
        assert config["deterministic_algorithms"] is True
        assert config["local_files_only"] is True
        assert provenance["dataset"]["train_examples"] == 3668
        assert provenance["dataset"]["validation_examples"] == 408
        assert audit["execution"]["optimizer_step"] == 560
        assert len(audit["execution"]["step_trace"]) == 560
        assert [step["optimizer_step"] for step in audit["execution"]["step_trace"]] == list(
            range(1, 561)
        )
        assert audit["gauge_generation"]["performed_before_validation"] is True
        assert audit["gauge_generation"]["external_witness_count"] == 0
        assert audit["gauge_generation"]["validation_used_for_raw_action_selection"] is False
        determinism = audit["deterministic_execution"]
        assert determinism["cublas_workspace_config"] == ":4096:8"
        assert determinism["requested"] is True
        assert determinism["torch_deterministic_algorithms_enabled"] is True
        assert determinism["cudnn_deterministic"] is True
        assert determinism["cudnn_benchmark"] is False
        assert determinism["cuda_matmul_allow_tf32"] is False
        assert determinism["cudnn_allow_tf32"] is False
        assert audit["status"] == "completed_pre_first_merge_audit_no_merge_committed"

    canonical_claim_prose = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "paper/main.tex",
            "paper/sections/introduction.tex",
            "paper/sections/discussion.tex",
            "paper/generated/aslora_table.tex",
        )
    )
    assert "12/30" not in canonical_claim_prose
    assert "16/30" not in canonical_claim_prose
    assert "11/30" in canonical_claim_prose
    assert "14/30" in canonical_claim_prose

    appendix = (ROOT / "paper/sections/appendix.tex").read_text(encoding="utf-8")
    assert "202, 207, and 206" not in appendix
    assert "204, 208, and 200" in appendix
    assert "also passes its tolerance in all three deterministic runs" in appendix
