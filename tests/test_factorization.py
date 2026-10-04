import math

from src.factorization import FactorizationConfig, run_factorization


def test_oracle_recovers_planted_graph():
    result = run_factorization(
        FactorizationConfig(depth=8, num_bases=4, dimension=16, pattern="cycle", method="oracle", steps=1),
        device="cpu",
    )
    assert result["assignment_accuracy"] == 1.0
    assert result["adjusted_rand"] == 1.0
    assert result["relative_reconstruction_error"] < 1e-6


def test_frozen_true_bases_make_router_recoverable():
    result = run_factorization(
        FactorizationConfig(
            depth=8,
            num_bases=4,
            dimension=32,
            pattern="random_balanced",
            method="soft_frozen",
            steps=600,
            lr=0.05,
            seed=1,
        ),
        device="cpu",
    )
    assert result["assignment_accuracy"] == 1.0
    assert result["relative_reconstruction_error"] < 0.05


def test_hard_methods_are_evaluated_with_hard_probabilities():
    result = run_factorization(
        FactorizationConfig(
            depth=8,
            num_bases=4,
            dimension=16,
            pattern="cycle",
            method="straight_through",
            init="pattern_cycle",
            steps=1,
        ),
        device="cpu",
    )
    probabilities = result["probabilities"]
    assert all(min(abs(value), abs(value - 1.0)) < 1e-6 for row in probabilities for value in row)
    assert all(abs(sum(row) - 1.0) < 1e-6 for row in probabilities)
