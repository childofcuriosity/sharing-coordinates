import copy
import csv
import json
from pathlib import Path

import pytest

from experiments.summarize_aslora_audit import (
    DEFAULT_INPUTS,
    summarize_aslora_audits,
    summarize_paths,
    write_summary_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def raw_artifacts():
    audits = []
    provenances = []
    for seed in (0, 1, 2):
        result_dir = PROJECT_ROOT / "results" / ("aslora_seed%d" % seed)
        audits.append(
            json.loads((result_dir / "audit.json").read_text(encoding="utf-8"))
        )
        provenances.append(
            json.loads((result_dir / "provenance.json").read_text(encoding="utf-8"))
        )
    return audits, provenances


@pytest.fixture(scope="module")
def real_summary(raw_artifacts):
    audits, provenances = raw_artifacts
    return summarize_aslora_audits(audits, provenances, PROJECT_ROOT)


def _aggregate_index(summary):
    return {
        (row["candidate_mode"], row["decision_scope"]): row
        for row in summary["across_seed_mode_scope"]
    }


def _seed_row_index(summary):
    return {
        (row["seed"], row["candidate_mode"], row["decision_scope"]): row
        for row in summary["seed_mode_scope_rows"]
    }


def test_real_three_seed_summary_uses_canonical_gate_and_all_seeds(real_summary):
    validation = real_summary["validation"]
    assert validation["candidate_coverage_all_pass"]
    assert validation["canonical_scientific_gate_all_pass"]
    assert validation[
        "result_integrity_gate_without_generation_source_attestation"
    ]
    assert validation["fully_attested_reproducibility_gate"]
    assert validation["source_hashes"][
        "generation_source_attestation_available"
    ]
    assert validation["source_hashes"]["generation_source_attestation_pass"]
    assert all(
        record["factorized_float32_reexecution_all_pass"]
        for record in validation["equivalence_and_restoration"].values()
    )
    assert all(
        record["deterministic_primary_action_repeatability_all_pass"]
        for record in validation["equivalence_and_restoration"].values()
    )
    assert [
        validation["candidate_coverage"][str(seed)][
            "actual_candidate_evaluation_count"
        ]
        for seed in (0, 1, 2)
    ] == [204, 208, 200]

    aggregate = _aggregate_index(real_summary)
    assert aggregate[("adjacent", "per_projection")][
        "total_action_change_count"
    ] == 11
    assert aggregate[("all", "per_projection")][
        "total_action_change_count"
    ] == 14
    assert aggregate[("adjacent", "joint")]["total_action_change_count"] == 5
    assert aggregate[("all", "joint")]["total_action_change_count"] == 0
    assert aggregate[("adjacent", "per_projection")][
        "chain_holds_in_all_three_seeds"
    ]
    assert aggregate[("all", "per_projection")][
        "chain_holds_in_all_three_seeds"
    ]
    assert not aggregate[("adjacent", "joint")][
        "chain_holds_in_all_three_seeds"
    ]
    assert not aggregate[("all", "joint")]["chain_holds_in_all_three_seeds"]
    assert real_summary["chain_assessment"][
        "primary_per_projection_chain_holds_for_both_candidate_modes_in_all_three_seeds"
    ]
    assert real_summary["chain_assessment"]["fully_source_attested_chain"]


def test_real_summary_reports_identity_relative_consequences_and_baselines(
    real_summary,
):
    rows = _seed_row_index(real_summary)
    seed_two_all = rows[(2, "all", "per_projection")]
    deltas = seed_two_all["consequence_ranges"]["delta_from_identity_action"]
    assert seed_two_all["identity_action_key"] == "query:0-1|value:0-1"
    assert seed_two_all["action_change_count"] == 3
    assert seed_two_all["unique_action_count"] == 3
    assert deltas["loss"]["min"] == pytest.approx(-0.0530944957452662)
    assert deltas["loss"]["max"] == pytest.approx(0.0)
    assert deltas["accuracy"]["min"] == pytest.approx(-0.002451002597808838)
    assert deltas["f1"]["max"] == pytest.approx(0.0008254447742320759)

    for row in rows.values():
        baselines = row["baselines"]
        assert baselines["raw_B_rule"]["action_key"] == row[
            "identity_action_key"
        ]
        assert baselines["gauge_invariant_current_effective_baseline"][
            "data_usage"
        ] == "prevalidation snapshot only"
        assert "without labels" in baselines[
            "gauge_invariant_current_local_activation_baseline"
        ]["data_usage"]
        oracle = baselines["validation_loss_oracle"]
        if row["decision_scope"] == "per_projection":
            assert oracle["coverage"] == (
                "componentwise_query_and_value_only_not_full_cartesian"
            )
            assert "not_claimed" in oracle
        else:
            assert oracle["coverage"] == (
                "exact_over_declared_joint_same_pair_candidate_family"
            )

    expected_primary_loss_deltas = {
        (0, "adjacent"): (-0.03364973383791303, 0.0),
        (0, "all"): (-0.007450254232275688, -0.0007690188347124205),
        (1, "adjacent"): (0.001765942456675551, 0.00013882856743008265),
        (1, "all"): (0.0027515245418922585, -0.0004822126790589021),
        (2, "adjacent"): (0.0, 0.0),
        (2, "all"): (0.0, 0.0),
    }
    for (seed, mode), (effective_delta, activation_delta) in (
        expected_primary_loss_deltas.items()
    ):
        baselines = rows[(seed, mode, "per_projection")]["baselines"]
        assert baselines["gauge_invariant_current_effective_baseline"][
            "delta_from_raw_B_action"
        ]["loss"] == pytest.approx(effective_delta, abs=1e-15)
        assert baselines["gauge_invariant_current_local_activation_baseline"][
            "delta_from_raw_B_action"
        ]["loss"] == pytest.approx(activation_delta, abs=1e-15)


def test_summary_rejects_failed_scientific_gate_restoration_and_coverage(
    raw_artifacts,
):
    audits, provenances = raw_artifacts

    failed_gate = copy.deepcopy(audits)
    failed_gate[0]["equivalence_gate"]["canonical_action_audit_all_pass"] = False
    with pytest.raises(ValueError, match="failed the canonical action audit"):
        summarize_aslora_audits(failed_gate, provenances, PROJECT_ROOT)

    failed_restoration = copy.deepcopy(audits)
    failed_restoration[1]["restoration"]["trainable_digest_after"] = "not-restored"
    with pytest.raises(ValueError, match="restore the canonical pre-merge model"):
        summarize_aslora_audits(failed_restoration, provenances, PROJECT_ROOT)

    incomplete = copy.deepcopy(audits)
    incomplete[2]["candidate_evaluations"].pop("query:0-1")
    incomplete[2]["candidate_evaluation_count"] -= 1
    with pytest.raises(ValueError, match="candidate coverage misses"):
        summarize_aslora_audits(incomplete, provenances, PROJECT_ROOT)


def test_summary_rejects_tampered_or_misdirected_input_bindings(raw_artifacts):
    audits, provenances = raw_artifacts

    tampered = copy.deepcopy(provenances)
    tampered[0]["input_artifacts"]["train_parquet"]["files"][0][
        "sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="digest does not match"):
        summarize_aslora_audits(audits, tampered, PROJECT_ROOT)

    misdirected = copy.deepcopy(provenances)
    misdirected[1]["input_artifacts"]["local_model"][
        "resolved_path"
    ] = "/different/roberta-base"
    with pytest.raises(ValueError, match="configured paths"):
        summarize_aslora_audits(audits, misdirected, PROJECT_ROOT)


def test_writers_emit_json_csv_and_markdown_from_validated_summary(
    tmp_path, real_summary
):
    json_path = tmp_path / "aslora_summary.json"
    csv_path = tmp_path / "aslora_summary.csv"
    markdown_path = tmp_path / "aslora_summary.md"
    write_summary_outputs(real_summary, json_path, csv_path, markdown_path)

    written_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert written_json["format"] == "aslora-three-seed-first-merge-summary-v1"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["seed"] for row in rows} == {"0", "1", "2"}
    assert {
        "current_effective_loss_delta_from_raw_B",
        "current_local_activation_loss_delta_from_raw_B",
    }.issubset(rows[0])
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "11/30" in markdown
    assert "14/30" in markdown
    assert "Generation-time source hashes embedded" in markdown


def test_default_path_loader_hashes_each_input_artifact():
    summary = summarize_paths(DEFAULT_INPUTS)
    assert [record["seed"] for record in summary["inputs"]] == [0, 1, 2]
    assert all(len(record["audit_sha256"]) == 64 for record in summary["inputs"])
    assert all(
        len(record["provenance_sha256"]) == 64 for record in summary["inputs"]
    )
