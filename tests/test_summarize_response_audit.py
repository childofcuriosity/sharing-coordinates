import copy

import pytest

from experiments.summarize_response_audit import summarize_response_audits


def _payload(seed):
    source = {
        "experiments/run_response_audit.py": "a" * 64,
        "src/response_identifiability.py": "b" * 64,
        "src/language.py": "c" * 64,
        "src/models.py": "d" * 64,
    }
    return {
        "format": "router-response-audit-v1",
        "status": "completed_current_coordinate_response_audit",
        "protocol": {
            "probe_count": 8,
            "probe_seed": 740_000 + seed,
            "eta_z": 1.0,
            "eta_b": 1.0,
            "gauge_strength": 0.35,
            "gauge_strength_is_predeclared_default": True,
            "minimum_probability": 1e-12,
            "small_ball_relative_threshold": 1e-8,
            "operator_materialized": False,
        },
        "checkpoint": {
            "sha256": "%064d" % seed,
            "language_config": {
                "seed": seed,
                "steps": 1500,
                "warmup_steps": 500,
                "hard": False,
                "router_trainable": True,
            },
            "metadata": {"extra": {
                "saved_at_step": 1500,
                "checkpoint_kind": "final_predeclared_training_step",
            }},
            "primary_checks": {"all_pass": True},
        },
        "source_sha256": source,
        "factor_diagnostics": {
            "theorem_rank_assumptions_pass": True,
            "A_full_column_rank": True,
            "B_full_row_rank": True,
            "minimum_A_entry": 1e-3,
            "A_sigma_min": 1.2,
            "B_sigma_min": 2.3,
        },
        "packing": {"reconstruction_pass": True},
        "formula_validation": {
            "pass": True,
            "autograd_relative_tolerance": 1e-8,
            "finite_difference_relative_tolerance": 1e-6,
            "native_extracted_factor_model": {
                "rows": 4,
                "columns": 32,
                "best_finite_difference_relative_l2_error": 1e-10,
                "analytic_vs_autograd_relative_l2_error": 1e-12,
            },
            "nonpermutation_extracted_factor_model": {
                "rows": 4,
                "columns": 32,
                "best_finite_difference_relative_l2_error": 2e-10 + seed * 1e-11,
                "analytic_vs_autograd_relative_l2_error": 2e-12,
            },
        },
        "permutation_control": {
            "pass": True,
            "control_tolerance": 5e-11,
            "probe_audit": {
                "detected_probe_count": 0,
                "maximum_relative_response_difference": 1e-16,
            },
        },
        "positive_stochastic_nonpermutation": {
            "is_nonpermutation": True,
            "detected_by_at_least_one_fixed_probe": True,
            "equivalence": {"effective_relative_l2_error": 2e-16},
            "probe_audit": {
                "detected_probe_count": 8,
                "probe_count": 8,
                "median_relative_response_difference": 0.5 + seed * 0.001,
                "minimum_relative_response_difference": 0.49,
            },
        },
    }


def test_three_seed_response_summary_checks_controls_and_sources():
    summary = summarize_response_audits([_payload(2), _payload(0), _payload(1)])
    assert [row["seed"] for row in summary["seed_rows"]] == [0, 1, 2]
    assert summary["across_seed"]["all_fixed_nonpermutation_probes_detected"]
    assert summary["across_seed"]["maximum_effective_relative_l2_error"] == 2e-16
    assert summary["across_seed"]["maximum_formula_fd_relative_error"] == pytest.approx(2.2e-10)
    assert summary["seed_rows"][0]["formula_worst_fd_relative_error"] == 2e-10
    assert summary["protocol"]["checkpoint_probe_evaluations"] == 24
    assert summary["protocol"]["unique_gaussian_draws"] == 10
    assert summary["protocol"]["probe_seed_starts"] == [740_000, 740_001, 740_002]
    assert summary["protocol"]["probe_windows_overlap"] is True


def test_response_summary_rejects_mixed_source_snapshots_and_failed_controls():
    payloads = [_payload(0), _payload(1), _payload(2)]
    mixed = copy.deepcopy(payloads)
    mixed[1]["source_sha256"]["src/models.py"] = "e" * 64
    with pytest.raises(ValueError, match="different source snapshots"):
        summarize_response_audits(mixed)
    broken = copy.deepcopy(payloads)
    broken[2]["permutation_control"]["pass"] = False
    with pytest.raises(ValueError, match="permutation control failed"):
        summarize_response_audits(broken)
