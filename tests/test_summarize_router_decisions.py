import copy

import pytest

from experiments.summarize_router_decisions import (
    markdown_table,
    paired_bootstrap_interval,
    summarize_router_decision_results,
)


def _decision(bpb, groups=2, squared_error=1.0, role=None):
    output = {
        "num_groups": groups,
        "instantaneous_bpb": bpb,
        "batch_bpb": [bpb] * 256,
        "effective_fit": {"squared_error": squared_error},
    }
    if role is not None:
        output["causal_role"] = role
    return output


def _payload(seed, delta, movement=0.2, test_bpb=6.0, unigram_bpb=8.0):
    reference_bpb = 6.5
    candidate_bpb = reference_bpb + delta
    reference = _decision(reference_bpb, squared_error=3.0)
    reference["common_effective_partition_refit"] = True
    candidate = _decision(candidate_bpb, squared_error=4.0)
    candidate["common_effective_partition_refit"] = True
    literal_original = _decision(
        6.6,
        role="secondary_coordinate_and_partition_confounding",
    )
    literal_gauged = _decision(
        6.7,
        role="secondary_coordinate_and_partition_confounding",
    )
    kmeans = _decision(6.4, squared_error=2.0)
    kmeans["kmeans"] = {"inertia": 2.0}
    ward = _decision(6.45, squared_error=2.5)
    ward["ward"] = {"inertia": 2.5}
    starts = [
        [(batch * 8 + item) * 129 for item in range(8)]
        for batch in range(256)
    ]
    return {
        "config": {
            "temperature": 1.0,
            "search_family": "positive_stochastic",
            "gauge_selection": "first_valid",
            "batch_sampling": "nonoverlap",
            "search_trials": 5000,
            "max_condition": 30.0,
            "min_probability": 1e-8,
            "batch_size": 8,
            "sequence_length": 128,
            "eval_batches": 256,
            "max_effective_relative_l2": 1e-5,
            "max_logit_abs_error": 1e-4,
            "max_batch_nll_error": 1e-5,
            "max_bpb_equivalence_error": 1e-5,
            "disable_tf32": True,
            "batch_seed": 610_000 + seed,
            "search_seed": 620_000 + seed,
            "kmeans_seed": 630_000 + seed,
            "kmeans_restarts": 8,
            "kmeans_iterations": 50,
        },
        "checkpoint": {
            "file_sha256": "%064d" % seed,
            "corpus_split": "test",
            "add_one_byte_unigram_bpb": unigram_bpb,
            "language_config": {
                "seed": seed,
                "steps": 1500,
                "warmup_steps": 500,
            },
            "training_metadata": {
                "saved_at_step": 1500,
                "checkpoint_kind": "final_predeclared_training_step",
                "test_bpb": test_bpb,
                "router_mean_l1_movement": movement,
            },
        },
        "equivalence": {
            "passed": True,
            "checks": {
                "effective_relative_l2": {"value": 1e-7, "threshold": 1e-5},
                "max_abs_logit_error": {"value": 1e-6, "threshold": 1e-4},
                "max_abs_batch_nll_error": {"value": 1e-7, "threshold": 1e-5},
                "absolute_bpb_error": {"value": 1e-7, "threshold": 1e-5},
            },
        },
        "numeric_controls": {"tf32_disabled_during_audit": True},
        "evaluation_batches": {
            "count": 256,
            "batch_size": 8,
            "sequence_length": 128,
            "sampling": "nonoverlap",
            "starts": starts,
        },
        "gauge_search": {
            "search_family": "positive_stochastic",
            "selection_rule": "first_valid",
            "minimum_gauge_entry": 0.01,
            "condition_number": 2.0,
            "selected_trial": 5,
            "first_valid_trial": 5,
            "counts": {
                "attempted": 5000,
                "invertible": 4999,
                "condition_pass": 4900,
                "simplex_pass": 4800,
                "group_pass": 4000,
                "partition_change": 500,
            },
        },
        "gauge": {
            "row_sum_error": 1e-8,
            "minimum_transformed_probability": 0.01,
        },
        "group_budget": 2,
        "primary_causal_contrast": {
            "name": "common_effective_partition_folds",
            "reference_decision": "partition_fold_original",
            "candidate_decision": "partition_fold_gauged",
            "common_effective_tensors": True,
            "reference_group_sizes": [6, 6],
            "candidate_group_sizes": [5, 7],
            "normalized_pairwise_partition_disagreement": 0.25,
            "batch_bpb_delta_candidate_minus_reference": [delta] * 256,
            "instantaneous_bpb_delta_candidate_minus_reference": delta,
        },
        "decisions": {
            "literal_argmax_original": literal_original,
            "literal_argmax_gauged": literal_gauged,
            "partition_fold_original": reference,
            "partition_fold_gauged": candidate,
            "effective_theta_kmeans": kmeans,
            "effective_theta_ward": ward,
        },
    }


def test_three_seed_summary_applies_predeclared_success_rule():
    summary = summarize_router_decision_results(
        [_payload(0, 0.02), _payload(1, -0.015), _payload(2, 0.001)],
    )
    assert summary["protocol"]["training_steps"] == 1500
    assert summary["across_seed"]["measurable_seed_count"] == 2
    assert summary["across_seed"]["replicated_measurable_consequence"] is True
    assert summary["across_seed"]["learned_system_eligibility"] is True
    assert summary["across_seed"]["primary_claim_supported"] is True
    assert summary["protocol"]["executed_search_families"] == [
        "positive_stochastic"
    ]
    assert summary["protocol"]["executed_sensitivity_analyses"] == []
    assert summary["protocol"]["unexecuted_nonprimary_protocol_amendments"] == [
        "signed_local_search",
        "same_sorted_group_size_multiset_search",
    ]
    assert all(row["decision_evaluations_stored"] for row in summary["seed_rows"])
    assert all(
        row["gauge_invariant_baselines_stored"] for row in summary["seed_rows"]
    )
    table = markdown_table(summary)
    assert "Literal-hardening BPB is intentionally omitted" in table
    assert "| 0 |" in table


def test_summary_rejects_posthoc_training_budget_change():
    payloads = [_payload(0, 0.02), _payload(1, 0.02), _payload(2, 0.02)]
    payloads[1] = copy.deepcopy(payloads[1])
    payloads[1]["checkpoint"]["language_config"]["steps"] = 10000
    with pytest.raises(ValueError, match="expected 1500"):
        summarize_router_decision_results(payloads)


def test_summary_keeps_exhausted_searches_as_negative_results():
    completed = _payload(0, 0.02)
    failures = []
    for seed in (1, 2):
        payload = _payload(seed, 0.02)
        payload["status"] = "no_partition_changing_gauge_found"
        payload["gauge_search"] = {
            "status": "no_partition_changing_gauge_found",
            "message": "predeclared search exhausted",
            "seed": 620_000 + seed,
            "trials": 5000,
            "search_family": "positive_stochastic",
            "selection_rule": "first_valid",
            "counts": {
                "attempted": 5000,
                "invertible": 5000,
                "condition_pass": 4500,
                "simplex_pass": 4500,
                "group_pass": 3700,
                "partition_change": 0,
            },
            "max_condition": 30.0,
            "min_probability": 1e-8,
            "required_groups": 2,
            "original_groups": 2,
            "original_assignment": [0, 0, 1, 1],
            "search_inputs": "router_probabilities_only",
        }
        payload["numeric_controls"] = {
            "tf32_disabled_during_audit": True,
            "search_failed_before_model_equivalence_or_decision_evaluation": True,
        }
        failures.append(payload)
    summary = summarize_router_decision_results([completed] + failures)
    aggregate = summary["across_seed"]
    assert aggregate["partition_changing_gauge_found_count"] == 1
    assert aggregate["partition_changing_gauge_not_found_count"] == 2
    assert aggregate["existence_chain_observed"] is True
    assert aggregate["replicated_measurable_consequence"] is False
    assert aggregate["t_interval_lower"] is None
    assert summary["seed_rows"][0]["gauge_invariant_baselines_stored"] is True
    assert all(
        row["decision_evaluations_stored"] is False
        and row["gauge_invariant_baselines_stored"] is False
        for row in summary["seed_rows"][1:]
    )
    table = markdown_table(summary)
    assert "succeeded for 1/3 seeds" in table
    assert "not evaluated" in table
    assert "k-means 6.4000; Ward 6.4500" in table
    assert "not stored/evaluated (search exited first)" in table
    assert "signed-local and same-sorted-group-size" in table


def test_paired_bootstrap_is_seeded_and_rejects_empty_input():
    first = paired_bootstrap_interval([-0.1, 0.0, 0.1], seed=9, resamples=100)
    second = paired_bootstrap_interval([-0.1, 0.0, 0.1], seed=9, resamples=100)
    assert first == second
    with pytest.raises(ValueError, match="at least one"):
        paired_bootstrap_interval([], seed=9, resamples=100)
