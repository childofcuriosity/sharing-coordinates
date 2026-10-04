import ast
import json
from pathlib import Path

import pytest

from experiments.analyze_results import (
    _require_bound_probe_artifact,
    _require_finite_metrics,
    _require_paired_probe_protocol,
    analyze_results,
    init_relation,
)


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _recovery(ari):
    return {
        "adjusted_rand": ari,
        "assignment_accuracy": (ari + 1.0) / 2.0,
        "graph_accuracy": (ari + 1.0) / 2.0,
        "normalized_entropy": 0.1,
    }


def _probe_protocol(seed, noise, projection_dimension=0):
    return {
        "version": "paired-crn-fixed-gaussian-v1",
        "comparison_design": "paired_common_random_numbers",
        "initial_probe_seed": 40_000 + seed,
        "projection_seed": 50_000 + seed,
        "observation_noise_seed": 60_000 + seed,
        "observation_noise_scale": noise,
        "observation_noise_scale_is_absolute": True,
        "projection_dimension": projection_dimension,
        "same_initial_probes_across_functional_modes": True,
        "same_observation_noise_across_functional_modes": True,
        "same_projection_across_functional_modes": True,
        "native_state_evolution": "clean_update_only",
        "empirical_gap_is_population_certificate": False,
        "effective_parameter_baseline_is_functional_equivalence_test": False,
    }


def test_analysis_generates_required_artifacts_and_skips_missing_input(tmp_path):
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    generated = tmp_path / "generated"
    summary_path = results / "summary.json"

    factorization = []
    for seed, error, ari in [(0, 0.01, -0.2), (1, 0.03, 0.8)]:
        factorization.append(
            {
                "config": {
                    "method": "soft",
                    "init": "neutral",
                    "pattern": "cycle",
                    "seed": seed,
                },
                "relative_reconstruction_error": error,
                "history": [],
                **_recovery(ari),
            }
        )
    factorization.extend(
        [
            {
                "config": {
                    "method": "soft",
                    "init": "pattern_cycle",
                    "pattern": "cycle",
                    "seed": 2,
                },
                "relative_reconstruction_error": 0.02,
                "history": [],
                **_recovery(0.9),
            },
            {
                "config": {
                    "method": "soft",
                    "init": "pattern_cycle",
                    "pattern": "contiguous",
                    "seed": 3,
                },
                "relative_reconstruction_error": 0.02,
                "history": [],
                **_recovery(-0.1),
            },
            {
                "config": {
                    "method": "soft",
                    "init": "pattern_cycle",
                    "pattern": "palindrome",
                    "seed": 4,
                },
                "relative_reconstruction_error": 0.025,
                "history": [],
                **_recovery(0.0),
            },
            {
                "config": {
                    "method": "soft",
                    "init": "random",
                    "pattern": "cycle",
                    "seed": 5,
                },
                "relative_reconstruction_error": 0.04,
                "history": [],
                **_recovery(0.2),
            },
            {
                "config": {
                    "method": "kmeans",
                    "init": "neutral",
                    "pattern": "cycle",
                    "seed": 6,
                },
                "relative_reconstruction_error": 0.005,
                "history": [],
                **_recovery(1.0),
            },
            {
                "config": {
                    "method": "oracle",
                    "init": "neutral",
                    "pattern": "contiguous",
                    "seed": 7,
                },
                "relative_reconstruction_error": 0.0,
                "history": [],
                **_recovery(1.0),
            },
        ]
    )
    _write(results / "factorization_full.json", factorization)

    _write(
        results / "teacher_student_mlp.json",
        [
            {
                "config": {
                    "architecture": "mlp",
                    "pattern": "cycle",
                    "router_init": "pattern_cycle",
                    "hard": False,
                    "hidden_weight": 0.0,
                    "vertex_strength": 0.0,
                    "usage_strength": 0.0,
                },
                "test_task_loss": 0.02,
                "test_hidden_loss": 0.4,
                "history": [{"step": 0, "task_loss": 0.2}],
                **_recovery(0.1),
            },
            {
                "config": {
                    "architecture": "mlp",
                    "pattern": "contiguous",
                    "router_init": "pattern_cycle",
                    "hard": False,
                    "hidden_weight": 0.0,
                    "vertex_strength": 0.0,
                    "usage_strength": 0.0,
                },
                "test_task_loss": 0.021,
                "test_hidden_loss": 0.4,
                "history": [],
                **_recovery(0.0),
            },
            {
                "config": {
                    "architecture": "mlp",
                    "pattern": "cycle",
                    "router_init": "neutral",
                    "hard": False,
                    "hidden_weight": 0.0,
                    "vertex_strength": 0.0,
                    "usage_strength": 0.0,
                },
                "test_task_loss": 0.019,
                "test_hidden_loss": 0.4,
                "history": [],
                **_recovery(0.2),
            },
        ],
    )
    _write(
        results / "teacher_student_transformer.json",
        [
            {
                "config": {
                    "architecture": "transformer",
                    "pattern": "contiguous",
                    "router_init": "pattern_cycle",
                    "hard": False,
                    "hidden_weight": 0.0,
                    "vertex_strength": 0.0,
                    "usage_strength": 0.0,
                },
                "test_task_loss": 0.01,
                "test_hidden_loss": 0.3,
                "history": [],
                **_recovery(0.9),
            },
            {
                "config": {
                    "architecture": "transformer",
                    "pattern": "cycle",
                    "router_init": "pattern_cycle",
                    "hard": False,
                    "hidden_weight": 0.0,
                    "vertex_strength": 0.0,
                    "usage_strength": 0.0,
                },
                "test_task_loss": 0.011,
                "test_hidden_loss": 0.3,
                "history": [],
                **_recovery(0.8),
            },
            {
                "config": {
                    "architecture": "transformer",
                    "pattern": "contiguous",
                    "router_init": "neutral",
                    "hard": False,
                    "hidden_weight": 0.0,
                    "vertex_strength": 0.0,
                    "usage_strength": 0.0,
                },
                "test_task_loss": 0.012,
                "test_hidden_loss": 0.3,
                "history": [],
                **_recovery(0.1),
            },
        ],
    )

    for filename, architecture in [("probes_mlp.json", "mlp"), ("probes_transformer.json", "transformer")]:
        probe_records = []
        for count in (1, 4):
            for noise in (0.0, 0.2):
                ari = 1.0 - noise - 1.0 / (count + 1.0)
                probe_records.append(
                    {
                        "config": {
                            "architecture": architecture,
                            "num_probes": count,
                            "observation_noise": noise,
                            "seed": 0,
                        },
                        "protocol": _probe_protocol(0, noise),
                        "counterfactual": {**_recovery(ari), "empirical_gap": count - noise},
                        "native": {**_recovery(ari - 0.2), "empirical_gap": count - noise - 0.5},
                        "effective_parameter": {
                            **_recovery(1.0),
                            "empirical_gap": count + 1.0,
                        },
                    }
                )
        _write(results / filename, probe_records)

    _write(
        results / "gauge_full.json",
        {
            "exact_argmax_flip": {
                "effective_max_abs_error": 0.0,
                "argmax_before": [1, 1, 1, 0],
                "argmax_after": [1, 0, 0, 0],
            },
            "gauge_trials": [
                {
                    "effective_max_abs_error": 1e-8,
                    "original_entropy": 0.4,
                    "transformed_entropy": 0.6,
                    "argmax_agreement": 0.5,
                    "coassignment_relative_change": 0.2,
                    "initial_loss_difference": 0.0,
                    "post_step_effective_relative_difference": 0.01,
                        "original_logit_gradient_norm": 0.2,
                        "transformed_logit_gradient_norm": 0.4,
                        "alpha": [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]],
                        "transformed_alpha": [[0.9, 0.1], [0.2, 0.8], [0.1, 0.9]],
                }
            ],
            "saturation": [
                {
                    "temperature": 1.0,
                    "margin": 2.0,
                    "initial_gradient_norm": 0.01,
                    "step_to_90pct": None,
                    "final_mse": 1e-4,
                    **_recovery(-0.1),
                }
            ],
        },
    )

    summary = analyze_results(results, figures, generated, summary_path)

    assert (figures / "factorization_loss_vs_ari.pdf").stat().st_size > 0
    assert (figures / "distillation_loss_vs_ari.pdf").stat().st_size > 0
    assert (figures / "probe_phase_diagram.pdf").stat().st_size > 0
    assert (generated / "factorization_table.tex").is_file()
    assert (generated / "factorization_full_table.tex").is_file()
    assert (generated / "distillation_table.tex").is_file()
    assert (generated / "distillation_full_table.tex").is_file()
    assert (generated / "key_numbers.tex").is_file()
    assert summary_path.is_file()
    assert summary["inputs"]["language_full.json"]["status"] == "missing"
    factor_groups = {
        group["init_relation"]: group
        for group in summary["factorization"]["groups"]
        if group["method"] == "soft"
    }
    assert set(factor_groups) == {"matched", "mismatched", "neutral", "random", "structured_other"}
    assert factor_groups["matched"]["raw_initializations"] == ["pattern_cycle"]
    assert factor_groups["matched"]["teacher_patterns"] == ["cycle"]
    assert factor_groups["mismatched"]["raw_initializations"] == ["pattern_cycle"]
    assert factor_groups["mismatched"]["teacher_patterns"] == ["contiguous"]
    assert factor_groups["structured_other"]["teacher_patterns"] == ["palindrome"]
    factor_stat = factor_groups["neutral"]["metrics"]["adjusted_rand"]
    assert factor_stat["n"] == 2
    assert factor_stat["mean"] == 0.30000000000000004
    assert summary["distillation"]["n_records"] == 6
    distillation_relations = {group["init_relation"] for group in summary["distillation"]["groups"]}
    assert distillation_relations == {"matched", "mismatched", "neutral"}
    assert summary["gauge"]["saturation"]["groups"][0]["metrics"]["step_to_90pct"]["n"] == 0
    for group in summary["probes"]["groups"]:
        assert set(group["modes"]) == {
            "counterfactual",
            "native",
            "effective_parameter",
        }
        comparison = group["paired_adjusted_rand_comparisons"][0]
        assert comparison["first"] == "counterfactual"
        assert comparison["second"] == "native"
        assert comparison["wins"] == 4
        assert comparison["ties"] == 0
        assert comparison["losses"] == 0

    factor_table = (generated / "factorization_table.tex").read_text(encoding="utf-8")
    factor_full_table = (generated / "factorization_full_table.tex").read_text(encoding="utf-8")
    assert "Init relation" in factor_table
    assert "matched" in factor_table
    assert "one-hot" in factor_table
    assert r"structured\_other" not in factor_table
    assert "palindrome" not in factor_table
    assert "structured-other" in factor_full_table
    assert "palindrome" in factor_full_table
    assert len(factor_table.splitlines()) < len(factor_full_table.splitlines())
    distillation_table = (generated / "distillation_table.tex").read_text(encoding="utf-8")
    distillation_full_table = (generated / "distillation_full_table.tex").read_text(encoding="utf-8")
    assert "Architecture & Init relation" in distillation_table
    assert len(distillation_table.splitlines()) < len(distillation_full_table.splitlines())
    probe_table = (generated / "probe_table.tex").read_text(encoding="utf-8")
    assert "effective parameter" in probe_table
    assert "not a test of functional equivalence" in probe_table
    key_numbers = (generated / "key_numbers.tex").read_text(encoding="utf-8")
    assert r"\newcommand{\GaugeExactError}" in key_numbers
    assert r"\newcommand{\GaugeExactArgmaxFlips}{2}" in key_numbers
    assert r"\newcommand{\GaugeTrialMeanPartitionARI}" in key_numbers
    assert r"\newcommand{\GaugeTrialChangedPartitions}{1}" in key_numbers
    assert r"\newcommand{\FactorizationRunCount}{8}" in key_numbers
    assert r"\newcommand{\TeacherStudentMLPRunCount}{3}" in key_numbers
    assert r"\newcommand{\TeacherStudentTransformerRunCount}{3}" in key_numbers
    assert r"\newcommand{\LanguageRunCount}" not in key_numbers


def test_init_relation_is_teacher_pattern_relative():
    assert init_relation({"config": {"pattern": "cycle", "init": "neutral"}}, "init") == "neutral"
    assert init_relation({"config": {"pattern": "cycle", "init": "random"}}, "init") == "random"
    assert init_relation({"config": {"pattern": "cycle", "init": "pattern_cycle"}}, "init") == "matched"
    assert (
        init_relation({"config": {"pattern": "contiguous", "init": "pattern_cycle"}}, "init")
        == "mismatched"
    )
    assert (
        init_relation({"config": {"pattern": "palindrome", "init": "pattern_cycle"}}, "init")
        == "structured_other"
    )


def test_analysis_source_parses_with_python38_grammar():
    source_path = Path(__file__).resolve().parents[1] / "experiments" / "analyze_results.py"
    ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path), feature_version=(3, 8))


def test_canonical_metrics_cannot_be_silently_dropped():
    with pytest.raises(ValueError, match="missing or non-finite"):
        _require_finite_metrics("example", [{"score": float("nan")}], ("score",))


def test_probe_protocol_rejects_unpaired_or_mislabelled_records():
    record = {
        "config": {"seed": 2, "observation_noise": 0.25},
        "protocol": _probe_protocol(2, 0.25),
    }
    _require_paired_probe_protocol([record])

    unpaired = json.loads(json.dumps(record))
    unpaired["protocol"]["same_projection_across_functional_modes"] = False
    with pytest.raises(ValueError, match="does not attest"):
        _require_paired_probe_protocol([unpaired])

    overclaim = json.loads(json.dumps(record))
    overclaim["protocol"]["empirical_gap_is_population_certificate"] = True
    with pytest.raises(ValueError, match="mislabels the empirical gap"):
        _require_paired_probe_protocol([overclaim])


def test_probe_artifact_rejects_invalid_protocol_fingerprint():
    payload = {
        "results": [],
        "protocol_fingerprint_payload": {"version": "paired"},
        "protocol_fingerprint_sha256": "0" * 64,
        "generation": {"record_count": 0, "source_sha256": {}},
    }
    with pytest.raises(ValueError, match="fingerprint does not verify"):
        _require_bound_probe_artifact("probe.json", payload)
