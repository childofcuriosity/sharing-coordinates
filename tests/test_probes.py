import math

import pytest
import torch

import src.probes as probes
from src.models import BasisResidualMLP, CausalBasisTransformer
from src.probes import (
    PairedProbeRandomness,
    ProbeConfig,
    effective_parameter_signatures,
    run_probe_recovery,
    separation_certificate,
)


def test_noiseless_counterfactual_probes_recover_tying_and_parameter_baseline():
    result = run_probe_recovery(
        ProbeConfig(
            architecture="mlp",
            pattern="random_balanced",
            depth=8,
            num_bases=4,
            dimension=16,
            hidden_dimension=32,
            num_probes=4,
            projection_dimension=0,
            observation_noise=0.0,
            seed=1,
        ),
        device="cpu",
    )
    assert result["counterfactual"]["adjusted_rand"] == 1.0
    assert result["effective_parameter"]["adjusted_rand"] == 1.0
    assert result["protocol"]["comparison_design"] == "paired_common_random_numbers"
    assert result["protocol"]["empirical_gap_is_population_certificate"] is False
    assert (
        result["protocol"][
            "effective_parameter_baseline_is_functional_equivalence_test"
        ]
        is False
    )


def test_empirical_gap_uses_supplied_target_and_squared_distances():
    signatures = torch.tensor([[0.0], [2.0], [10.0]])
    assignment = torch.tensor([0, 0, 1])
    gap = separation_certificate(signatures, assignment)
    assert gap["max_within_distance"] == 4.0
    assert gap["min_between_distance"] == 64.0
    assert gap["empirical_gap"] == 60.0

    # Supplying a different target changes the descriptive finite-sample gap;
    # it is not silently computed on the predicted clustering.
    alternate = separation_certificate(signatures, torch.tensor([0, 1, 0]))
    assert alternate["max_within_distance"] == 100.0
    assert alternate["min_between_distance"] == 4.0
    assert alternate["empirical_gap"] == -96.0


def test_native_measurement_noise_does_not_change_next_layer_input():
    class Router:
        @staticmethod
        def probabilities(temperature=1.0, hard=True):
            return torch.ones(2, 1)

    class IdentityResidual:
        router = Router()
        residual_scale = 1.0

        @staticmethod
        def residual_update(x, alpha):
            return x

    config = ProbeConfig(
        architecture="mlp",
        depth=2,
        num_bases=1,
        dimension=3,
        hidden_dimension=3,
        num_probes=2,
        projection_dimension=0,
        observation_noise=0.5,
        seed=7,
    )
    initial = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    standard_noise = torch.stack(
        [
            torch.arange(6, dtype=torch.float32).reshape(2, 3),
            -torch.arange(6, dtype=torch.float32).reshape(2, 3),
        ]
    )
    randomness = PairedProbeRandomness(initial, standard_noise, None)

    signatures = probes.native_trajectory_signatures(
        IdentityResidual(), config, torch.device("cpu"), randomness
    )
    first_observed = initial + 0.5 * standard_noise[0]
    second_clean = 2.0 * initial
    second_observed = second_clean + 0.5 * standard_noise[1]
    expected = torch.stack(
        [first_observed.flatten(), second_observed.flatten()]
    ) / math.sqrt(2.0)
    assert torch.allclose(signatures, expected)


def test_observation_noise_has_fixed_absolute_scale():
    noise = torch.tensor([1.0, -2.0, 0.5])
    small = probes._observe(torch.tensor([0.0, 1.0, 2.0]), noise, 0.25)
    large = probes._observe(torch.tensor([0.0, 100.0, 200.0]), noise, 0.25)
    assert torch.equal(small - torch.tensor([0.0, 1.0, 2.0]), 0.25 * noise)
    assert torch.equal(large - torch.tensor([0.0, 100.0, 200.0]), 0.25 * noise)
    with pytest.raises(ValueError, match="nonnegative"):
        probes._observe(torch.zeros(3), noise, -0.1)


def test_paired_randomness_is_reproducible_and_shared_in_run(monkeypatch):
    config = ProbeConfig(
        architecture="mlp",
        depth=4,
        num_bases=2,
        dimension=8,
        hidden_dimension=8,
        num_probes=3,
        projection_dimension=5,
        observation_noise=0.2,
        seed=9,
    )
    first = probes.paired_probe_randomness(config, torch.device("cpu"))
    second = probes.paired_probe_randomness(config, torch.device("cpu"))
    assert torch.equal(first.initial_probes, second.initial_probes)
    assert torch.equal(first.standard_observation_noise, second.standard_observation_noise)
    assert torch.equal(first.projection, second.projection)

    seen = []
    original_counterfactual = probes.counterfactual_signatures
    original_native = probes.native_trajectory_signatures

    def counterfactual_spy(model, cfg, device, randomness=None):
        seen.append(("counterfactual", id(randomness)))
        return original_counterfactual(model, cfg, device, randomness)

    def native_spy(model, cfg, device, randomness=None):
        seen.append(("native", id(randomness)))
        return original_native(model, cfg, device, randomness)

    monkeypatch.setattr(probes, "counterfactual_signatures", counterfactual_spy)
    monkeypatch.setattr(probes, "native_trajectory_signatures", native_spy)
    run_probe_recovery(config, device="cpu")
    assert seen[0][0] == "counterfactual"
    assert seen[1][0] == "native"
    assert seen[0][1] == seen[1][1]


@pytest.mark.parametrize("architecture", ["mlp", "transformer"])
def test_effective_parameter_baseline_supports_both_architectures(architecture):
    common = dict(
        dimension=8,
        hidden_dimension=12,
        depth=4,
        num_bases=2,
        router_init="planted",
        router_trainable=False,
        router_pattern="cycle",
        seed=3,
    )
    if architecture == "mlp":
        model = BasisResidualMLP(**common)
    else:
        model = CausalBasisTransformer(
            vocab_size=16,
            max_length=4,
            num_heads=2,
            **common,
        )
    signatures = effective_parameter_signatures(model)
    assert signatures.shape[0] == 4
    assert torch.equal(signatures[0], signatures[2])
    assert torch.equal(signatures[1], signatures[3])
    assert not torch.equal(signatures[0], signatures[1])
