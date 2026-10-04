import json

import pytest
import torch

from experiments.run_response_tomography import build_parser, run
from src.language import LanguageConfig, build_language_model, save_language_checkpoint
from src.response_identifiability import (
    apply_euclidean_response,
    cyclic_permutation,
    deterministic_positive_stochastic_gauge,
    transform_factors,
)
from src.response_tomography import (
    _reconstruct_components,
    audit_structured_tomography,
    build_tomography_design,
    design_diagnostics,
    tomography_probe,
)


def _factors():
    probabilities = torch.tensor(
        [
            [0.70, 0.20, 0.10],
            [0.15, 0.70, 0.15],
            [0.10, 0.25, 0.65],
            [0.40, 0.35, 0.25],
        ],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(20260823)
    bases = torch.randn(3, 10, generator=generator, dtype=torch.float64)
    assert int(torch.linalg.matrix_rank(bases)) == 3
    return probabilities, bases


def test_design_uses_only_theta_and_has_the_required_geometry():
    probabilities, bases = _factors()
    theta = probabilities @ bases
    design = build_tomography_design(theta, num_bases=3)
    diagnostics = design_diagnostics(design, theta)
    assert design.probe_count == 3
    assert diagnostics["dimension_condition_D_minus_K_ge_L"]
    assert diagnostics["theta_rank"] == 3
    assert diagnostics["row_basis_orthonormal_max_abs_error"] < 1e-12
    assert diagnostics["complement_codes_orthonormal_max_abs_error"] < 1e-12
    assert diagnostics["row_complement_cross_max_abs_error"] < 1e-12
    assert diagnostics["theta_rowspace_relative_residual"] < 1e-12
    first = tomography_probe(design, 0)
    assert first.shape == theta.shape
    for index in range(1, 3):
        probe = tomography_probe(design, index)
        assert torch.allclose(
            probe,
            design.row_basis[index].expand_as(probe),
            atol=0.0,
            rtol=0.0,
        )


def test_K_probes_reconstruct_components_and_separate_nonpermutation():
    probabilities, bases = _factors()
    gauge = deterministic_positive_stochastic_gauge(3, strength=0.35)
    transformed_a, transformed_b, _ = transform_factors(
        probabilities, bases, gauge, min_probability=0.0
    )
    audit = audit_structured_tomography(
        probabilities,
        bases,
        transformed_a,
        transformed_b,
        eta_z=0.7,
        eta_b=1.3,
        temperature=0.8,
        exact_common_product_holds_by_symbolic_gauge_construction=True,
    )
    assert audit["reference_design_rank_dimension_pass"]
    assert audit["common_product_numerical_tolerance_pass"]
    assert audit["exact_common_product_holds_by_symbolic_gauge_construction"]
    assert audit["common_design_operator_determining_theorem_applies"]
    assert len(audit["probe_records"]) == 3
    assert audit["response_operators_distinguished_within_tolerance"]
    assert not audit[
        "response_equal_on_operator_determining_probes_within_tolerance"
    ]
    reconstruction = audit["reconstruction"]
    for key in (
        "reference_router_gram_relative_error",
        "candidate_router_gram_relative_error",
        "reference_local_blocks_relative_error",
        "candidate_local_blocks_relative_error",
    ):
        assert reconstruction[key] < 1e-11
    assert reconstruction["local_blocks_pair_relative_difference"] > 1e-4


def test_reconstructed_components_predict_an_unseen_response():
    """The K observations determine the operator, not only fitted summaries."""

    probabilities, bases = _factors()
    eta_z, eta_b, temperature = 0.7, 1.3, 0.8
    design = build_tomography_design(probabilities @ bases, num_bases=3)
    responses = [
        apply_euclidean_response(
            probabilities,
            bases,
            tomography_probe(design, index),
            eta_z=eta_z,
            eta_b=eta_b,
            temperature=temperature,
        )
        for index in range(design.probe_count)
    ]
    gram, local_coordinates = _reconstruct_components(
        responses, design, eta_b
    )
    generator = torch.Generator().manual_seed(20260824)
    unseen = torch.randn(
        probabilities.shape[0],
        bases.shape[1],
        generator=generator,
        dtype=torch.float64,
    )
    row_coordinates = unseen @ design.row_basis.transpose(0, 1)
    # The explicit einsum below is easier to audit than materializing every
    # D-by-D local block: row_coordinates[i] @ H_i^{(S)} @ row_basis.
    reconstructed_local_coordinates = torch.einsum(
        "lr,lrs->ls", row_coordinates, local_coordinates
    )
    reconstructed = eta_b * (gram @ unseen) + (
        reconstructed_local_coordinates @ design.row_basis
    )
    direct = apply_euclidean_response(
        probabilities,
        bases,
        unseen,
        eta_z=eta_z,
        eta_b=eta_b,
        temperature=temperature,
    )
    assert torch.allclose(reconstructed, direct, atol=2e-12, rtol=2e-12)


def test_K_probes_accept_exact_basis_permutation_control():
    probabilities, bases = _factors()
    transformed_a, transformed_b, _ = transform_factors(
        probabilities, bases, cyclic_permutation(3), min_probability=0.0
    )
    audit = audit_structured_tomography(
        probabilities,
        bases,
        transformed_a,
        transformed_b,
        exact_common_product_holds_by_symbolic_gauge_construction=True,
    )
    assert audit[
        "response_equal_on_operator_determining_probes_within_tolerance"
    ]
    assert not audit["response_operators_distinguished_within_tolerance"]
    assert (
        audit["reconstruction"]["maximum_probe_relative_response_difference"]
        < 1e-12
    )


def test_dimension_condition_is_a_named_design_requirement_not_a_rank_claim():
    probabilities = torch.tensor(
        [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]], dtype=torch.float64
    )
    bases = torch.tensor(
        [[1.0, 0.0, 0.5], [0.0, 1.0, -0.2]], dtype=torch.float64
    )
    theta = probabilities @ bases
    assert int(torch.linalg.matrix_rank(theta)) == 2
    with pytest.raises(ValueError, match="effective_dimension-num_bases"):
        build_tomography_design(theta, num_bases=2)


def test_rank_one_case_uses_one_probe_and_zero_router_local_block():
    probabilities = torch.ones(2, 1, dtype=torch.float64)
    bases = torch.tensor([[1.0, -2.0, 0.5, 0.2]], dtype=torch.float64)
    audit = audit_structured_tomography(
        probabilities,
        bases,
        probabilities.clone(),
        bases.clone(),
        exact_common_product_holds_by_symbolic_gauge_construction=True,
    )
    assert audit["design"]["probe_count"] == 1
    assert audit[
        "response_equal_on_operator_determining_probes_within_tolerance"
    ]
    assert audit["reconstruction"]["reference_local_blocks_relative_error"] < 1e-12


def test_numerically_close_product_does_not_activate_exact_common_design_theorem():
    probabilities, bases = _factors()
    candidate_bases = bases.clone()
    candidate_bases[0, 0] += 1e-12
    audit = audit_structured_tomography(
        probabilities,
        bases,
        probabilities.clone(),
        candidate_bases,
        equality_relative_tolerance=5e-11,
    )
    assert audit["reference_design_rank_dimension_pass"]
    assert audit["common_product_numerical_tolerance_pass"]
    assert not audit[
        "exact_common_product_holds_by_symbolic_gauge_construction"
    ]
    assert not audit[
        "common_design_operator_determining_theorem_applies"
    ]


def test_tomography_runner_writes_additive_source_attested_result(tmp_path):
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
    output = tmp_path / "tomography.json"
    args = build_parser().parse_args(
        ["--checkpoint", str(checkpoint), "--output", str(output)]
    )
    result = run(args)
    assert result["status"] == "completed_operator_determining_response_tomography"
    assert result["probe_count_K"] == 2
    assert result["all_checks_pass"]
    assert result["permutation_control"]["tomography"][
        "response_equal_on_operator_determining_probes_within_tolerance"
    ]
    assert result["positive_stochastic_nonpermutation"]["tomography"][
        "response_operators_distinguished_within_tolerance"
    ]
    assert set(result["source_sha256"]) == {
        "experiments/run_response_tomography.py",
        "src/response_tomography.py",
        "src/response_identifiability.py",
        "src/language.py",
        "src/models.py",
    }
    assert json.loads(output.read_text(encoding="utf-8"))["all_checks_pass"]
