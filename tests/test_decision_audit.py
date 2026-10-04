import math

import pytest
import torch

from src.decision_audit import (
    approximate_selectability_margin,
    audit_all_gauge_order,
    audit_pair_distances,
    fixed_metric_selectability_witness,
    gauge_from_metric,
    metric_distance,
    r2_ranking_flip_example,
    r2_unbounded_gauge_family,
    running_average_bases,
    selectability_bounds,
    verify_gl_gauge,
)


DTYPE = torch.float64


def test_running_average_and_all_three_pair_distances():
    averaged = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 2.0]],
            [[3.0, -1.0]],
        ],
        dtype=DTYPE,
    )
    history = torch.stack((torch.zeros_like(averaged), 2.0 * averaged, 9.0 * averaged))
    A = torch.diag(torch.tensor([1.0, 2.0], dtype=DTYPE))
    covariance = torch.diag(torch.tensor([4.0, 0.25], dtype=DTYPE))

    assert torch.allclose(running_average_bases(history, steps=2), averaged)
    audit = audit_pair_distances(
        history,
        A,
        [(1, 0)],
        running_average=True,
        average_steps=2,
        activation_covariance=covariance,
    )
    score = audit.scores[(0, 1)]

    assert torch.allclose(
        score.S_p, torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=DTYPE)
    )
    assert score.raw_distance == pytest.approx(math.sqrt(5.0))
    assert score.effective_distance == pytest.approx(math.sqrt(17.0))
    assert score.activation_weighted_distance == pytest.approx(math.sqrt(8.0))
    assert audit.used_running_average
    assert audit.running_average_steps == 2
    assert audit.a_rank == 2
    assert audit.a_full_row_rank


def test_rank_deficient_A_is_reported_and_effective_distance_is_a_seminorm():
    bases = torch.tensor([[[0.0, 0.0]], [[0.0, 3.0]]], dtype=DTYPE)
    A = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    audit = audit_pair_distances(bases, A, [(0, 1)])

    assert audit.scores[(0, 1)].raw_distance == pytest.approx(3.0)
    assert audit.scores[(0, 1)].effective_distance == pytest.approx(0.0)
    assert audit.a_rank == 1
    assert not audit.a_full_row_rank


def test_gl_gauge_preserves_products_and_invariant_distances_but_changes_raw_score():
    example = r2_ranking_flip_example()
    before = example["before"]
    after = example["after"]
    gauge = example["gauge"]
    p, q = example["pairs"]

    assert gauge.products_close
    assert gauge.max_absolute_error < 1e-12
    assert before.scores[p].raw_distance < before.scores[q].raw_distance
    assert after.scores[p].raw_distance > after.scores[q].raw_distance
    for pair in (p, q):
        assert after.scores[pair].effective_distance == pytest.approx(
            before.scores[pair].effective_distance
        )
        assert after.scores[pair].activation_weighted_distance == pytest.approx(
            before.scores[pair].activation_weighted_distance
        )
        expected_metric_distance = metric_distance(
            before.scores[pair].S_p, gauge.metric_C
        )
        assert after.scores[pair].raw_distance == pytest.approx(expected_metric_distance)


def test_verify_gl_gauge_does_not_require_full_rank_A():
    bases = torch.tensor([[[1.0, 2.0]], [[-1.0, 4.0]]], dtype=DTYPE)
    A = torch.tensor([[1.0], [0.0]], dtype=DTYPE)
    R = torch.tensor([[2.0, 0.5], [0.0, 0.25]], dtype=DTYPE)
    result = verify_gl_gauge(bases, A, R, atol=1e-12, rtol=1e-12)
    assert result.products_close
    assert result.max_absolute_error < 1e-12


def test_all_gauge_order_uses_the_exact_psd_boundary():
    S_p = torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE))
    S_q = torch.diag(torch.tensor([0.0, 4.0], dtype=DTYPE))
    flippable = audit_all_gauge_order(S_p, S_q)
    assert flippable.classification == "order_flippable_by_gauge"
    assert not flippable.p_no_farther_for_all_gauges
    assert not flippable.q_no_farther_for_all_gauges

    dominating_q = torch.diag(torch.tensor([2.0, 1.0], dtype=DTYPE))
    ordered = audit_all_gauge_order(S_p, dominating_q)
    assert ordered.classification == "p_no_farther_for_all_gauges"
    assert ordered.p_no_farther_for_all_gauges
    assert not ordered.q_no_farther_for_all_gauges

    tied = audit_all_gauge_order(S_p, S_p.clone())
    assert tied.classification == "tie_for_all_gauges"
    assert tied.p_no_farther_for_all_gauges
    assert tied.q_no_farther_for_all_gauges


def test_singular_fixed_C_with_positive_gap_produces_checkable_spd_gauge_witness():
    p, q = (0, 1), (0, 2)
    scatters = {
        p: torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE)),
        q: torch.diag(torch.tensor([0.0, 4.0], dtype=DTYPE)),
    }
    singular_C = torch.diag(torch.tensor([0.0, 1.0], dtype=DTYPE))
    witness = fixed_metric_selectability_witness(scatters, p, singular_C)

    assert witness.lower_bound == pytest.approx(4.0)
    assert not witness.metric_is_spd
    assert witness.certifies_selectable
    assert witness.spd_metric_C is not None
    assert witness.spd_lower_bound > 0.0
    assert torch.linalg.eigvalsh(witness.spd_metric_C).min() > 0.0
    reconstructed_R = gauge_from_metric(witness.spd_metric_C)
    inverse_R = torch.linalg.inv(reconstructed_R)
    assert torch.allclose(
        inverse_R @ inverse_R.T, witness.spd_metric_C, atol=1e-11, rtol=1e-11
    )


def test_selectability_bounds_can_certify_both_selectable_and_nonselectable_cases():
    p, q, s = (0, 1), (0, 2), (1, 2)
    selectable = {
        p: torch.zeros(2, 2, dtype=DTYPE),
        q: torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE)),
        s: torch.diag(torch.tensor([0.0, 1.0], dtype=DTYPE)),
    }
    positive = selectability_bounds(selectable, p)
    assert positive.lower_bound == pytest.approx(0.5)
    assert positive.upper_bound == pytest.approx(0.5)
    assert positive.certifies_selectable
    assert not positive.certifies_not_selectable

    nonselectable = {
        p: torch.eye(2, dtype=DTYPE),
        q: torch.diag(torch.tensor([0.0, 1.0], dtype=DTYPE)),
        s: torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE)),
    }
    negative = selectability_bounds(nonselectable, p)
    assert negative.lower_bound == pytest.approx(-0.5)
    assert negative.upper_bound == pytest.approx(-0.5)
    assert not negative.certifies_selectable
    assert negative.certifies_not_selectable


def test_torch_search_is_explicitly_approximate_but_returns_valid_certificates():
    p, q, s = (0, 1), (0, 2), (1, 2)
    scatters = {
        p: torch.zeros(2, 2, dtype=DTYPE),
        q: torch.diag(torch.tensor([1.0, 0.0], dtype=DTYPE)),
        s: torch.diag(torch.tensor([0.0, 1.0], dtype=DTYPE)),
    }
    result = approximate_selectability_margin(
        scatters, p, iterations=100, learning_rate=0.25
    )

    assert result.lower_bound <= result.upper_bound + 1e-10
    assert result.lower_bound == pytest.approx(0.5, abs=1e-8)
    assert result.upper_bound == pytest.approx(0.5, abs=1e-8)
    assert result.certifies_selectable
    assert torch.linalg.eigvalsh(result.metric_C).min() >= -1e-12
    assert torch.trace(result.metric_C) == pytest.approx(1.0)
    assert result.mixture_weights.min() >= -1e-12
    assert result.mixture_weights.sum() == pytest.approx(1.0)


def test_r2_family_has_unbounded_regret_with_fixed_products_within_each_model():
    records = r2_unbounded_gauge_family([2.0, 10.0, 100.0])
    for record in records:
        assert record.raw_distance_p_before == pytest.approx(1.0)
        assert record.raw_distance_q_before == pytest.approx(record.scale)
        assert record.raw_distance_p_after == pytest.approx(1.0)
        assert record.raw_distance_q_after == pytest.approx(0.5)
        assert record.selected_pair_before == (0, 1)
        assert record.selected_pair_after == (1, 2)
        assert record.effective_distance_p == pytest.approx(1.0)
        assert record.effective_distance_q == pytest.approx(record.scale)
        assert record.invariant_cost_p == pytest.approx(1.0)
        assert record.invariant_cost_q == pytest.approx(record.scale ** 2)
        assert record.approximation_ratio == pytest.approx(record.scale ** 2)
        assert record.additive_regret == pytest.approx(record.scale ** 2 - 1.0)
        assert record.max_product_error < 1e-12
    assert records[-1].approximation_ratio > 1000.0 * records[0].approximation_ratio


def test_invalid_shapes_singular_gauge_and_non_psd_covariance_are_rejected():
    bases = torch.zeros(2, 1, 2, dtype=DTYPE)
    A = torch.eye(2, dtype=DTYPE)
    with pytest.raises(ValueError, match="invertible"):
        verify_gl_gauge(bases, A, torch.zeros(2, 2, dtype=DTYPE))
    with pytest.raises(ValueError, match="positive semidefinite"):
        audit_pair_distances(
            bases,
            A,
            [(0, 1)],
            activation_covariance=torch.diag(torch.tensor([1.0, -1.0], dtype=DTYPE)),
        )
    with pytest.raises(ValueError, match="history"):
        running_average_bases(bases)
