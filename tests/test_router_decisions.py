import json
import math

import pytest
import torch

from src.language import LanguageConfig, build_language_model
from src.router_decisions import (
    GaugeSearchFailure,
    RouterDecisionConfig,
    apply_common_router_gauge,
    compare_effective_tensors,
    effective_basis_tensors,
    effective_theta_kmeans,
    effective_theta_ward,
    make_fixed_language_batches,
    materialize_partition_folding,
    run_router_decision_audit,
    search_partition_changing_gauge,
)


def _tiny_model(depth=4, num_bases=2):
    config = LanguageConfig(
        depth=depth,
        num_bases=num_bases,
        dimension=8,
        hidden_dimension=16,
        num_heads=2,
        sequence_length=8,
        batch_size=2,
        steps=0,
        amp=False,
        seed=3,
    )
    return build_language_model(config)


def test_common_gauge_preserves_every_effective_tensor_and_logits():
    model = _tiny_model(depth=4, num_bases=3)
    probabilities = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.20, 0.65, 0.15],
            [0.15, 0.20, 0.65],
            [0.45, 0.35, 0.20],
        ]
    )
    with torch.no_grad():
        model.router.logits.copy_(probabilities.log())
    gauge = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.20, 0.70, 0.10],
            [0.10, 0.20, 0.70],
        ]
    )
    reference = effective_basis_tensors(model)
    transformed, metadata = apply_common_router_gauge(
        model, gauge, temperature=1.0, min_probability=1e-8
    )
    candidate = effective_basis_tensors(transformed)
    error = compare_effective_tensors(reference, candidate)
    assert len(reference) == 6  # qkv/output/up/down weights and up/down biases
    assert error["max_abs"] < 2e-6
    assert metadata["condition_number"] < 10.0
    tokens = torch.randint(0, 256, (2, 8))
    with torch.no_grad():
        first = model(tokens, temperature=1.0, hard=False)
        second = transformed(tokens, temperature=1.0, hard=False)
    assert torch.max(torch.abs(first - second)) < 2e-5


def test_router_only_search_changes_partition_at_fixed_group_budget():
    probabilities = torch.tensor(
        [[0.90, 0.10], [0.60, 0.40], [0.40, 0.60], [0.10, 0.90]]
    )
    gauge, metadata = search_partition_changing_gauge(
        probabilities,
        seed=17,
        trials=1000,
        max_condition=20.0,
        min_probability=1e-8,
        search_family="positive_stochastic",
        selection_rule="first_valid",
    )
    transformed = probabilities @ gauge
    assert torch.all(transformed > 0)
    assert metadata["search_inputs"] == "router_probabilities_only"
    assert metadata["original_groups"] == metadata["transformed_groups"] == 2
    assert metadata["pairwise_partition_disagreements"] > 0
    assert metadata["original_assignment"] != metadata["transformed_assignment"]
    assert metadata["minimum_gauge_entry"] > 0.0
    assert metadata["selected_trial"] == metadata["first_valid_trial"]
    counts = metadata["counts"]
    assert counts["attempted"] == 1000
    assert (
        counts["invertible"]
        >= counts["condition_pass"]
        >= counts["simplex_pass"]
        >= counts["group_pass"]
        >= counts["partition_change"]
        >= 1
    )
    _, signed_metadata = search_partition_changing_gauge(
        probabilities,
        seed=17,
        trials=100,
        max_condition=20.0,
        min_probability=1e-8,
        search_family="signed_local",
        selection_rule="first_valid",
    )
    assert signed_metadata["search_family"] == "signed_local"
    assert signed_metadata["selected_trial"] == signed_metadata["first_valid_trial"]
    assert signed_metadata["counts"]["partition_change"] >= 1


def test_router_only_search_failure_preserves_stage_counts():
    probabilities = torch.tensor(
        [[0.999, 0.001], [0.999, 0.001], [0.001, 0.999], [0.001, 0.999]]
    )
    with pytest.raises(GaugeSearchFailure) as captured:
        search_partition_changing_gauge(
            probabilities,
            seed=31,
            trials=25,
            max_condition=1.0,
            search_family="positive_stochastic",
            selection_rule="first_valid",
        )
    metadata = captured.value.metadata
    assert metadata["status"] == "no_partition_changing_gauge_found"
    assert metadata["counts"]["attempted"] == 25
    assert metadata["counts"]["partition_change"] == 0
    assert metadata["original_groups"] == metadata["required_groups"] == 2


def test_small_cpu_router_decision_chain_reports_equivalence_and_nlls():
    model = _tiny_model(depth=4, num_bases=2)
    probabilities = torch.tensor(
        [[0.90, 0.10], [0.60, 0.40], [0.40, 0.60], [0.10, 0.90]]
    )
    with torch.no_grad():
        model.router.logits.copy_(probabilities.log())
    stream = torch.arange(2048, dtype=torch.long) % 256
    result = run_router_decision_audit(
        model,
        stream,
        RouterDecisionConfig(
            temperature=1.0,
            batch_size=2,
            sequence_length=8,
            eval_batches=2,
            search_seed=17,
            search_trials=1000,
            max_condition=20.0,
            kmeans_restarts=2,
            kmeans_iterations=10,
        ),
    )
    assert result["equivalence"]["effective_tensors"]["max_abs"] < 2e-6
    assert result["equivalence"]["fixed_batch_logits"]["max_abs_logit_error"] < 2e-5
    assert result["assignments"]["original"] != result["assignments"]["gauged"]
    assert result["group_budget"] == 2
    assert set(result["decisions"]) == {
        "literal_argmax_original",
        "literal_argmax_gauged",
        "partition_fold_original",
        "partition_fold_gauged",
        "effective_theta_kmeans",
        "effective_theta_ward",
    }
    for decision in result["decisions"].values():
        assert len(decision["batch_nll"]) == 2
        assert math.isfinite(decision["instantaneous_bpb"])
        assert decision["num_groups"] == 2
    assert result["equivalence"]["passed"] is True
    assert result["evaluation_batches"]["sampling"] == "nonoverlap"
    assert result["primary_causal_contrast"]["name"] == "common_effective_partition_folds"
    assert result["primary_causal_contrast"]["common_effective_tensors"] is True
    assert (
        result["decisions"]["literal_argmax_gauged"]["causal_role"]
        == "secondary_coordinate_and_partition_confounding"
    )
    assert (
        result["decisions"]["partition_fold_gauged"]["causal_role"]
        == "primary_common_effective_partition_fold"
    )
    json.dumps(result, allow_nan=False)
    kmeans_error = result["decisions"]["effective_theta_kmeans"]["effective_fit"]
    kmeans_inertia = result["decisions"]["effective_theta_kmeans"]["kmeans"]["inertia"]
    assert math.isclose(
        kmeans_inertia,
        kmeans_error["squared_error"],
        rel_tol=1e-5,
        abs_tol=1e-8,
    )


def test_nonoverlap_batches_are_disjoint_and_reproducible():
    stream = torch.arange(4096, dtype=torch.long) % 256
    batches, metadata = make_fixed_language_batches(
        stream,
        batch_size=3,
        sequence_length=8,
        count=4,
        seed=91,
        sampling="nonoverlap",
    )
    repeated, repeated_metadata = make_fixed_language_batches(
        stream,
        batch_size=3,
        sequence_length=8,
        count=4,
        seed=91,
        sampling="nonoverlap",
    )
    starts = [start for row in metadata["starts"] for start in row]
    assert len(starts) == len(set(starts))
    assert all(start % 9 == 0 for start in starts)
    assert metadata == repeated_metadata
    assert all(
        torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
        for first, second in zip(batches, repeated)
    )


def test_effective_theta_baselines_are_nonempty_and_ward_is_deterministic():
    model = _tiny_model(depth=5, num_bases=3)
    common = effective_basis_tensors(model)
    kmeans_assignment, kmeans_metadata = effective_theta_kmeans(
        common,
        num_clusters=3,
        seed=11,
        restarts=3,
        iterations=5,
    )
    _, kmeans_fold = materialize_partition_folding(model, common, kmeans_assignment)
    ward_first, ward_metadata = effective_theta_ward(common, num_clusters=3)
    ward_second, ward_repeated = effective_theta_ward(common, num_clusters=3)
    _, ward_fold = materialize_partition_folding(model, common, ward_first)
    assert torch.unique(kmeans_assignment).numel() == 3
    assert torch.unique(ward_first).numel() == 3
    assert torch.equal(ward_first, ward_second)
    assert ward_metadata == ward_repeated
    assert math.isclose(
        kmeans_metadata["inertia"],
        kmeans_fold["effective_fit"]["squared_error"],
        rel_tol=1e-5,
        abs_tol=1e-8,
    )
    assert math.isclose(
        ward_metadata["inertia"],
        ward_fold["effective_fit"]["squared_error"],
        rel_tol=1e-5,
        abs_tol=1e-8,
    )


def test_exact_equivalence_threshold_failure_is_fatal_and_tf32_is_restored():
    model = _tiny_model(depth=4, num_bases=2)
    probabilities = torch.tensor(
        [[0.90, 0.10], [0.60, 0.40], [0.40, 0.60], [0.10, 0.90]]
    )
    with torch.no_grad():
        model.router.logits.copy_(probabilities.log())
    stream = torch.arange(2048, dtype=torch.long) % 256
    previous_matmul = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        with pytest.raises(ValueError, match="exact-equivalence audit failed"):
            run_router_decision_audit(
                model,
                stream,
                RouterDecisionConfig(
                    batch_size=2,
                    sequence_length=8,
                    eval_batches=2,
                    search_seed=17,
                    search_trials=1000,
                    max_condition=20.0,
                    max_effective_relative_l2=0.0,
                    max_logit_abs_error=0.0,
                    max_batch_nll_error=0.0,
                    max_bpb_equivalence_error=0.0,
                    disable_tf32=True,
                ),
            )
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul
        torch.backends.cudnn.allow_tf32 = previous_cudnn
