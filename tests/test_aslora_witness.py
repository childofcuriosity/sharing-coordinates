import math

import pytest
import torch

from src.aslora_witness import (
    SharedALoRALinear,
    apply_common_gl_gauge,
    attach_aslora_to_roberta_classifier,
    candidate_pairs,
    effective_pair_distances,
    effective_updates,
    evaluate_candidate_merges,
    gauge_condition_number,
    is_orthogonal_gauge,
    load_first_merge_snapshot,
    make_orthogonal_gauge,
    make_three_layer_flip_snapshot,
    merge_lower_uses_upper,
    raw_pair_distances,
    save_first_merge_snapshot,
    select_minimum_pair,
    snapshot_selected_pairs,
    update_running_average,
)


def test_fixed_gl_gauge_preserves_products_and_flips_raw_first_merge():
    snapshot, gauge = make_three_layer_flip_snapshot(epsilon=0.2, delta=0.05)
    state = snapshot.targets["query"]
    before_products = effective_updates(state.shared_a, state.layer_b)
    before_raw = raw_pair_distances(state.running_b, mode="adjacent")
    before_invariant = effective_pair_distances(
        state.shared_a, state.running_b, mode="adjacent"
    )

    transformed_a, transformed_b, transformed_running = apply_common_gl_gauge(
        state.shared_a, state.layer_b, state.running_b, gauge
    )
    after_products = effective_updates(transformed_a, transformed_b)
    after_raw = raw_pair_distances(transformed_running, mode="adjacent")
    after_invariant = effective_pair_distances(
        transformed_a, transformed_running, mode="adjacent"
    )

    assert select_minimum_pair(before_raw) == (0, 1)
    assert select_minimum_pair(after_raw) == (1, 2)
    assert torch.allclose(before_products, after_products, atol=1e-12, rtol=1e-12)
    for pair in candidate_pairs(3, "adjacent"):
        assert torch.allclose(
            before_invariant[pair], after_invariant[pair], atol=1e-12, rtol=1e-12
        )


def test_condition_one_orthogonal_gauge_is_raw_distance_sanity_check():
    generator = torch.Generator().manual_seed(11)
    shared_a = torch.randn(3, 5, generator=generator, dtype=torch.float64)
    layer_b = torch.randn(4, 6, 3, generator=generator, dtype=torch.float64)
    running_b = torch.randn(4, 6, 3, generator=generator, dtype=torch.float64)
    gauge = make_orthogonal_gauge(3, seed=9, dtype=torch.float64)

    transformed_a, transformed_b, transformed_running = apply_common_gl_gauge(
        shared_a, layer_b, running_b, gauge
    )
    before_raw = raw_pair_distances(running_b)
    after_raw = raw_pair_distances(transformed_running)
    before_invariant = effective_pair_distances(shared_a, running_b)
    after_invariant = effective_pair_distances(transformed_a, transformed_running)

    assert is_orthogonal_gauge(gauge)
    assert math.isclose(gauge_condition_number(gauge), 1.0, rel_tol=1e-10, abs_tol=1e-10)
    assert torch.allclose(
        effective_updates(shared_a, layer_b),
        effective_updates(transformed_a, transformed_b),
        atol=1e-10,
        rtol=1e-10,
    )
    for pair in before_raw:
        assert torch.allclose(before_raw[pair], after_raw[pair], atol=1e-10, rtol=1e-10)
        assert torch.allclose(
            before_invariant[pair], after_invariant[pair], atol=1e-10, rtol=1e-10
        )


def test_running_average_and_lower_uses_upper_semantics():
    average = torch.zeros(3, 1, 2, dtype=torch.float64)
    first = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]], dtype=torch.float64
    )
    second = first + 2.0
    average, count = update_running_average(average, first, previous_count=0)
    average, count = update_running_average(average, second, previous_count=count)
    assert count == 2
    assert torch.equal(average, first + 1.0)

    merged = merge_lower_uses_upper(first, (0, 2))
    assert torch.equal(merged[0], first[2])
    assert torch.equal(merged[1], first[1])
    assert torch.equal(merged[2], first[2])
    assert torch.equal(first[0], torch.tensor([[1.0, 2.0]], dtype=torch.float64))


def test_snapshot_roundtrip_and_each_candidate_evaluation_hook(tmp_path):
    snapshot, _ = make_three_layer_flip_snapshot()
    destination = tmp_path / "first_merge.pt"
    save_first_merge_snapshot(snapshot, destination)
    restored = load_first_merge_snapshot(destination)

    assert snapshot_selected_pairs(restored) == {"query": (0, 1)}
    visited = []

    def evaluator(pair, shared_a, merged_b):
        visited.append(pair)
        updates = effective_updates(shared_a, merged_b)
        lower, upper = pair
        return {"post_merge_pair_gap": float(torch.linalg.norm(updates[lower] - updates[upper]))}

    records = evaluate_candidate_merges(
        restored.targets["query"], evaluator=evaluator, mode="adjacent"
    )
    assert visited == [(0, 1), (1, 2)]
    assert [record["pair"] for record in records] == visited
    assert all(record["metrics"]["post_merge_pair_gap"] == 0.0 for record in records)


def test_optional_transformers_roberta_wq_wv_attachment_without_download():
    transformers = pytest.importorskip("transformers")
    config = transformers.RobertaConfig(
        vocab_size=31,
        hidden_size=12,
        num_hidden_layers=3,
        num_attention_heads=3,
        intermediate_size=16,
        num_labels=2,
    )
    model = transformers.RobertaForSequenceClassification(config)
    handle = attach_aslora_to_roberta_classifier(
        model,
        rank=2,
        alpha=4.0,
        a_scope="per_projection",
        seed=4,
    )

    assert set(handle.modules) == {"query", "value"}
    assert all(isinstance(module, SharedALoRALinear) for module in handle.modules["query"])
    query_a_ids = {id(module.shared_a) for module in handle.modules["query"]}
    value_a_ids = {id(module.shared_a) for module in handle.modules["value"]}
    assert len(query_a_ids) == 1
    assert len(value_a_ids) == 1
    assert query_a_ids != value_a_ids
    assert len({id(module.b) for module in handle.modules["query"]}) == 3

    with torch.no_grad():
        for index, module in enumerate(handle.modules["query"]):
            module.b.fill_(float(index + 1))
    handle.update_running_averages()
    snapshot = handle.capture_first_merge_snapshot(
        candidate_mode="adjacent", decision_scope="per_projection"
    )
    assert snapshot.targets["query"].running_count == 1
    assert snapshot.targets["query"].layer_b.shape == (3, 12, 2)

    original = handle.modules["query"][0].b
    upper = handle.modules["query"][1].b
    with handle.temporary_lower_uses_upper((0, 1), targets=("query",)):
        assert handle.modules["query"][0].b is upper
    assert handle.modules["query"][0].b is original

