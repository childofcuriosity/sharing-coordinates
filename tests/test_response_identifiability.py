import json

import pytest
import torch

from experiments.run_response_audit import build_parser, run
from src.language import (
    LanguageConfig,
    build_language_model,
    save_language_checkpoint,
)
from src.models import CausalBasisTransformer
from src.response_identifiability import (
    apply_euclidean_response,
    audit_fixed_gaussian_probes,
    autograd_response_validation,
    cyclic_permutation,
    deterministic_positive_stochastic_gauge,
    matrix_diagnostics,
    pack_basis_matrix,
    reconstruct_effective_tensors,
    transform_factors,
)
from src.router_decisions import effective_basis_tensors


def _full_rank_factors():
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=torch.float64
    )
    bases = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.2]], dtype=torch.float64
    )
    return probabilities, bases


def test_analytic_response_matches_autograd_and_finite_difference():
    probabilities, bases = _full_rank_factors()
    result = autograd_response_validation(
        probabilities,
        bases,
        rows=3,
        columns=3,
        seed=41,
        eta_z=0.7,
        eta_b=1.3,
    )
    assert result["analytic_vs_autograd_max_abs_error"] < 1e-12
    assert result["analytic_vs_autograd_relative_l2_error"] < 1e-12
    assert result["best_finite_difference_relative_l2_error"] < 1e-8


def test_permutation_control_has_identical_response_for_every_tested_gradient():
    probabilities, bases = _full_rank_factors()
    permutation = cyclic_permutation(2)
    transformed_a, transformed_b, metadata = transform_factors(
        probabilities, bases, permutation, min_probability=0.0
    )
    assert metadata["effective_max_abs_error"] < 1e-14
    generator = torch.Generator().manual_seed(4)
    for _ in range(5):
        gradient = torch.randn(3, 3, generator=generator, dtype=torch.float64)
        reference = apply_euclidean_response(
            probabilities, bases, gradient, eta_z=0.4, eta_b=1.7
        )
        candidate = apply_euclidean_response(
            transformed_a, transformed_b, gradient, eta_z=0.4, eta_b=1.7
        )
        assert torch.allclose(reference, candidate, atol=1e-12, rtol=1e-12)


def test_positive_stochastic_nonpermutation_changes_full_rank_response():
    probabilities, bases = _full_rank_factors()
    gauge = deterministic_positive_stochastic_gauge(2, strength=0.35)
    assert not torch.equal(gauge, torch.eye(2, dtype=torch.float64))
    transformed_a, transformed_b, metadata = transform_factors(
        probabilities, bases, gauge, min_probability=0.0
    )
    diagnostics = matrix_diagnostics(probabilities, bases)
    assert diagnostics["theorem_rank_assumptions_pass"]
    assert metadata["minimum_transformed_probability"] > 0
    gradient = torch.tensor(
        [[0.2, -0.4, 0.7], [0.9, 0.1, -0.3], [-0.6, 0.8, 0.5]],
        dtype=torch.float64,
    )
    difference = apply_euclidean_response(
        probabilities, bases, gradient
    ) - apply_euclidean_response(transformed_a, transformed_b, gradient)
    assert float(difference.norm()) > 1e-3


def test_even_probe_count_uses_the_standard_two_middle_value_median():
    probabilities, bases = _full_rank_factors()
    gauge = deterministic_positive_stochastic_gauge(2, strength=0.35)
    transformed_a, transformed_b, _ = transform_factors(
        probabilities, bases, gauge, min_probability=0.0
    )
    audit = audit_fixed_gaussian_probes(
        probabilities,
        bases,
        transformed_a,
        transformed_b,
        probe_count=4,
        probe_seed=123,
    )
    values = sorted(
        record["relative_response_difference"] for record in audit["probes"]
    )
    expected = (values[1] + values[2]) / 2.0
    assert audit["median_relative_response_difference"] == pytest.approx(expected)
    assert audit["median_relative_response_difference"] != values[2]


def test_rank_deficient_basis_counterexample_hides_nonpermutation_response():
    gauge = torch.tensor(
        [[2.0, -1.0, 2.0], [-1.0, 2.0, 2.0], [2.0, 2.0, -1.0]],
        dtype=torch.float64,
    ) / 3.0
    uniform = torch.ones(3, 3, dtype=torch.float64) / 3.0
    probabilities = 0.9 * uniform + 0.1 * torch.eye(3, dtype=torch.float64)
    bases = torch.ones(3, 1, dtype=torch.float64) @ torch.tensor(
        [[1.0, 0.0, 0.0]], dtype=torch.float64
    )
    transformed_a, transformed_b, metadata = transform_factors(
        probabilities, bases, gauge, min_probability=0.0
    )
    assert metadata["minimum_transformed_probability"] > 0
    diagnostics = matrix_diagnostics(probabilities, bases)
    assert diagnostics["A_full_column_rank"]
    assert not diagnostics["B_full_row_rank"]
    gradient = torch.randn(3, 3, generator=torch.Generator().manual_seed(9), dtype=torch.float64)
    reference = apply_euclidean_response(probabilities, bases, gradient)
    candidate = apply_euclidean_response(transformed_a, transformed_b, gradient)
    assert torch.allclose(reference, candidate, atol=1e-12, rtol=1e-12)


def test_rank_deficient_router_counterexample_hides_nonpermutation_response():
    gauge = torch.tensor(
        [[2.0, -1.0, 2.0], [-1.0, 2.0, 2.0], [2.0, 2.0, -1.0]],
        dtype=torch.float64,
    ) / 3.0
    probabilities = torch.ones(1, 3, dtype=torch.float64) / 3.0
    bases = torch.eye(3, dtype=torch.float64)
    transformed_a, transformed_b, _ = transform_factors(
        probabilities, bases, gauge, min_probability=0.0
    )
    diagnostics = matrix_diagnostics(probabilities, bases)
    assert not diagnostics["A_full_column_rank"]
    assert diagnostics["B_full_row_rank"]
    gradient = torch.tensor([[0.4, -0.2, 0.7]], dtype=torch.float64)
    reference = apply_euclidean_response(probabilities, bases, gradient)
    candidate = apply_euclidean_response(transformed_a, transformed_b, gradient)
    assert torch.allclose(reference, candidate, atol=1e-12, rtol=1e-12)


def test_packed_basis_reconstructs_every_layer_effective_tensor():
    model = CausalBasisTransformer(
        vocab_size=17,
        max_length=8,
        dimension=4,
        hidden_dimension=8,
        num_heads=2,
        depth=3,
        num_bases=2,
        seed=12,
    ).double()
    packed = pack_basis_matrix(model, temperature=1.0)
    reconstructed = reconstruct_effective_tensors(packed)
    direct = effective_basis_tensors(
        model, probabilities=packed.router, temperature=1.0
    )
    assert list(reconstructed) == list(direct)
    for key in direct:
        assert torch.allclose(reconstructed[key], direct[key], atol=1e-12, rtol=1e-12)
    assert [item.key for item in packed.layout] == [
        "attention.qkv.weight",
        "attention.output.weight",
        "up.weight",
        "up.bias",
        "down.weight",
        "down.bias",
    ]


def test_checkpoint_runner_produces_auditable_current_coordinate_result(tmp_path):
    config = LanguageConfig(
        train_path="unused-train",
        validation_path="unused-valid",
        test_path="unused-test",
        depth=3,
        num_bases=2,
        dimension=4,
        hidden_dimension=8,
        num_heads=2,
        sequence_length=8,
        batch_size=2,
        steps=1500,
        warmup_steps=500,
        seed=2,
        amp=False,
    )
    model = build_language_model(config)
    checkpoint = tmp_path / "language_seed2_step1500.pt"
    save_language_checkpoint(
        str(checkpoint),
        model,
        config,
        {"train": "a", "validation": "b", "test": "c", "combined": "d"},
        extra={
            "checkpoint_kind": "final_predeclared_training_step",
            "saved_at_step": 1500,
        },
    )
    output = tmp_path / "response.json"
    args = build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--probe-count",
            "3",
            "--autograd-columns",
            "8",
        ]
    )
    result = run(args)
    assert result["status"] == "completed_current_coordinate_response_audit"
    assert result["checkpoint"]["primary_checks"]["all_pass"]
    assert result["permutation_control"]["pass"]
    assert result["positive_stochastic_nonpermutation"][
        "detected_by_at_least_one_fixed_probe"
    ]
    assert result["formula_validation"]["pass"]
    assert set(result["source_sha256"]) == {
        "experiments/run_response_audit.py",
        "src/response_identifiability.py",
        "src/language.py",
        "src/models.py",
    }
    assert output.is_file()
    assert json.loads(output.read_text(encoding="utf-8"))["format"] == "router-response-audit-v1"


def test_cli_dry_run_does_not_open_checkpoint(tmp_path):
    missing = tmp_path / "missing.pt"
    args = build_parser().parse_args(
        ["--checkpoint", str(missing), "--dry-run"]
    )
    result = run(args)
    assert result["status"] == "dry_run_checkpoint_not_opened"
    assert not missing.exists()
