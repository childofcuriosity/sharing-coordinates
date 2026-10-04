"""Finite structured tomography for the Euclidean router response.

The complete response operator in :mod:`src.response_identifiability` has a
special block form.  Once the effective matrix ``Theta = A @ B`` has rank
``K``, every chart in its full-rank factorization orbit has the same
``K``-dimensional row space.  The router-local response blocks are supported
on that row space, whereas the shared-basis response acts isotropically in
the effective coordinate.  This separation lets ``K`` explicitly designed
effective-gradient probes determine the whole response when the orthogonal
complement has room for one code vector per layer.

This module is deliberately separate from the source-attested response audit.
It adds a structured finite-observation certificate without changing the
released Gaussian-probe generator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch

from src.response_identifiability import (
    apply_euclidean_response,
    softmax_jacobian_squares,
)


@dataclass(frozen=True)
class TomographyDesign:
    """Compact description of the deterministic response probes.

    ``row_basis`` has orthonormal rows spanning ``row(Theta)`` and
    ``complement_codes`` has one orthonormal row per effective layer, all in
    the orthogonal complement.  The probes themselves are generated lazily so
    a caller never has to retain ``K`` matrices of shape ``[L,D]``.
    """

    row_basis: torch.Tensor
    complement_codes: torch.Tensor
    theta_rank: int
    theta_singular_values: torch.Tensor

    @property
    def num_bases(self) -> int:
        return int(self.row_basis.shape[0])

    @property
    def depth(self) -> int:
        return int(self.complement_codes.shape[0])

    @property
    def effective_dimension(self) -> int:
        return int(self.row_basis.shape[1])

    @property
    def probe_count(self) -> int:
        return self.num_bases


def _validate_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


def _row_space_basis(
    theta: torch.Tensor,
    num_bases: int,
    relative_rank_tolerance: float,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return an orthonormal row basis using only the effective matrix."""

    _validate_matrix("theta", theta)
    if num_bases < 1:
        raise ValueError("num_bases must be positive")
    if not 0 < relative_rank_tolerance < 1:
        raise ValueError("relative_rank_tolerance must lie in (0,1)")
    depth, dimension = theta.shape
    if num_bases > min(depth, dimension):
        raise ValueError("num_bases cannot exceed min(depth, dimension)")

    # The thin SVD exposes only min(L,D) right singular vectors and therefore
    # avoids a D-by-D allocation even for packed language-model parameters.
    _, singular_values, right_vectors = torch.linalg.svd(
        theta, full_matrices=False
    )
    largest = float(singular_values[0].detach().cpu())
    if largest == 0.0:
        raise ValueError("theta must be nonzero")
    threshold = relative_rank_tolerance * largest
    rank = int((singular_values > threshold).sum().detach().cpu())
    if rank != num_bases:
        raise ValueError(
            "theta must have declared rank num_bases; "
            f"observed rank {rank}, expected {num_bases}"
        )
    return right_vectors[:num_bases], singular_values, rank


def _coordinate_complement_codes(
    row_basis: torch.Tensor,
    count: int,
    tolerance: float,
) -> torch.Tensor:
    """Choose deterministic orthonormal complement codes by greedy scanning.

    Among any first ``K+L`` coordinate vectors, their projections onto the
    complement of a ``K``-dimensional subspace span at least ``L`` dimensions.
    Modified Gram--Schmidt therefore finds the requested codes without forming
    a full null-space basis.
    """

    _validate_matrix("row_basis", row_basis)
    if count < 1:
        raise ValueError("count must be positive")
    if not tolerance > 0:
        raise ValueError("tolerance must be positive")
    k, dimension = row_basis.shape
    if dimension - k < count:
        raise ValueError(
            "finite K-probe design requires effective_dimension-num_bases "
            ">= depth"
        )
    codes: List[torch.Tensor] = []
    scan_limit = k + count
    for coordinate in range(scan_limit):
        candidate = torch.zeros(
            dimension, dtype=row_basis.dtype, device=row_basis.device
        )
        candidate[coordinate] = 1.0
        candidate = candidate - row_basis[:, coordinate] @ row_basis
        # Reorthogonalize twice.  This is inexpensive for L small and keeps the
        # realized design auditable close to machine precision.
        for _ in range(2):
            for code in codes:
                candidate = candidate - torch.dot(candidate, code) * code
        norm = torch.linalg.vector_norm(candidate)
        if float(norm.detach().cpu()) > tolerance:
            codes.append(candidate / norm)
            if len(codes) == count:
                break
    if len(codes) != count:
        raise RuntimeError("failed to construct the guaranteed complement codes")
    return torch.stack(codes, dim=0)


def build_tomography_design(
    theta: torch.Tensor,
    num_bases: int,
    relative_rank_tolerance: float = 1e-10,
    orthogonalization_tolerance: float = 1e-10,
) -> TomographyDesign:
    """Build the ``K``-probe design from an exact rank-``K`` effective matrix.

    The sufficient packing condition is ``D-K >= L``.  It is not claimed to
    be necessary for finite response identification; it makes one probe able
    to encode all ``L`` router-Gram columns in mutually orthogonal directions.
    """

    row_basis, singular_values, rank = _row_space_basis(
        theta, int(num_bases), float(relative_rank_tolerance)
    )
    codes = _coordinate_complement_codes(
        row_basis,
        int(theta.shape[0]),
        float(orthogonalization_tolerance),
    )
    return TomographyDesign(
        row_basis=row_basis,
        complement_codes=codes,
        theta_rank=rank,
        theta_singular_values=singular_values,
    )


def tomography_probe(design: TomographyDesign, index: int) -> torch.Tensor:
    """Materialize one of the ``K`` deterministic effective gradients."""

    if not 0 <= int(index) < design.probe_count:
        raise IndexError("tomography probe index is out of range")
    if int(index) == 0:
        return design.complement_codes + design.row_basis[0].unsqueeze(0)
    return design.row_basis[int(index)].unsqueeze(0).expand(
        design.depth, -1
    ).clone()


def design_diagnostics(
    design: TomographyDesign,
    theta: torch.Tensor,
) -> Dict[str, Any]:
    """Return compact, independently checkable rank/orthogonality residuals."""

    u = design.row_basis
    v = design.complement_codes
    identity_u = torch.eye(
        design.num_bases, dtype=u.dtype, device=u.device
    )
    identity_v = torch.eye(design.depth, dtype=v.dtype, device=v.device)
    theta_residual = theta - (theta @ u.transpose(0, 1)) @ u
    return {
        "construction": "deterministic_projected_coordinate_gram_schmidt",
        "probe_count": design.probe_count,
        "depth_L": design.depth,
        "num_bases_K": design.num_bases,
        "effective_dimension_D": design.effective_dimension,
        "complement_dimension_D_minus_K": (
            design.effective_dimension - design.num_bases
        ),
        "dimension_condition_D_minus_K_ge_L": (
            design.effective_dimension - design.num_bases >= design.depth
        ),
        "theta_rank": design.theta_rank,
        "theta_singular_values": [
            float(value) for value in design.theta_singular_values.detach().cpu()
        ],
        "row_basis_orthonormal_max_abs_error": float(
            (u @ u.transpose(0, 1) - identity_u).abs().max().detach().cpu()
        ),
        "complement_codes_orthonormal_max_abs_error": float(
            (v @ v.transpose(0, 1) - identity_v).abs().max().detach().cpu()
        ),
        "row_complement_cross_max_abs_error": float(
            (u @ v.transpose(0, 1)).abs().max().detach().cpu()
        ),
        "theta_rowspace_relative_residual": float(
            theta_residual.double().norm().detach().cpu()
            / theta.double().norm().detach().cpu().clamp_min(
                torch.finfo(torch.float64).tiny
            )
        ),
    }


def _component_truth(
    probabilities: torch.Tensor,
    bases: torch.Tensor,
    design: TomographyDesign,
    eta_z: float,
    temperature: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    gram = probabilities @ probabilities.transpose(0, 1)
    basis_in_row_coordinates = bases @ design.row_basis.transpose(0, 1)
    jacobian_squares = softmax_jacobian_squares(probabilities)
    local = (eta_z / (temperature * temperature)) * torch.einsum(
        "rk,ikm,ms->irs",
        basis_in_row_coordinates.transpose(0, 1),
        jacobian_squares,
        basis_in_row_coordinates,
    )
    return gram, local


def _reconstruct_components(
    responses: Sequence[torch.Tensor],
    design: TomographyDesign,
    eta_b: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(responses) != design.probe_count:
        raise ValueError("one response is required for every tomography probe")
    for response in responses:
        if response.shape != (design.depth, design.effective_dimension):
            raise ValueError("a tomography response has the wrong shape")
    first_probe = tomography_probe(design, 0)
    first_response = responses[0]
    gram = (
        first_response @ design.complement_codes.transpose(0, 1)
    ) / float(eta_b)
    local = torch.empty(
        design.depth,
        design.num_bases,
        design.num_bases,
        dtype=first_response.dtype,
        device=first_response.device,
    )
    first_residual = first_response - float(eta_b) * (gram @ first_probe)
    local[:, :, 0] = first_residual @ design.row_basis.transpose(0, 1)
    row_sums = gram.sum(dim=1)
    for index in range(1, design.num_bases):
        shared_direction = design.row_basis[index]
        residual = responses[index] - (
            float(eta_b) * row_sums.unsqueeze(1) * shared_direction.unsqueeze(0)
        )
        local[:, :, index] = residual @ design.row_basis.transpose(0, 1)
    return gram, local


def _relative_error(observed: torch.Tensor, expected: torch.Tensor) -> float:
    difference = observed - expected
    numerator = difference.double().norm().detach().cpu()
    denominator = expected.double().norm().detach().cpu()
    # A K=1 softmax has an exactly zero local block.  In that case a relative
    # error is undefined, so report the absolute residual on the same field;
    # for every nonzero component the quantity is the ordinary relative error.
    if float(denominator) == 0.0:
        return float(numerator)
    return float(numerator / denominator)


def audit_structured_tomography(
    probabilities: torch.Tensor,
    bases: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    candidate_bases: torch.Tensor,
    *,
    eta_z: float = 1.0,
    eta_b: float = 1.0,
    temperature: float = 1.0,
    equality_relative_tolerance: float = 5e-11,
    exact_common_product_holds_by_symbolic_gauge_construction: bool = False,
) -> Dict[str, Any]:
    """Run and reconstruct the designed ``K`` probes for a pair.

    A small computed product residual is only a numerical consistency check;
    it cannot establish exact equality.  The common-design theorem is marked
    applicable only when the caller attests exact equality from a symbolic
    construction (the released runner uses an explicit invertible gauge).
    This keeps approximate-product experiments from inheriting an exact
    theorem merely by falling below a floating-point tolerance.
    """

    for name, value in (
        ("probabilities", probabilities),
        ("bases", bases),
        ("candidate_probabilities", candidate_probabilities),
        ("candidate_bases", candidate_bases),
    ):
        _validate_matrix(name, value)
    if probabilities.shape != candidate_probabilities.shape:
        raise ValueError("the routers must have the same shape")
    if bases.shape != candidate_bases.shape:
        raise ValueError("the basis matrices must have the same shape")
    if probabilities.shape[1] != bases.shape[0]:
        raise ValueError("A and B have incompatible latent dimensions")
    if any(
        value.device != probabilities.device
        for value in (bases, candidate_probabilities, candidate_bases)
    ):
        raise ValueError("all factor matrices must share one device")
    if any(
        value.dtype != probabilities.dtype
        for value in (bases, candidate_probabilities, candidate_bases)
    ):
        raise ValueError("all factor matrices must share one dtype")
    if not math.isfinite(eta_z) or eta_z <= 0:
        raise ValueError("eta_z must be finite and positive")
    if not math.isfinite(eta_b) or eta_b <= 0:
        raise ValueError("eta_b must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if equality_relative_tolerance < 0:
        raise ValueError("equality_relative_tolerance must be nonnegative")

    theta = probabilities @ bases
    candidate_theta = candidate_probabilities @ candidate_bases
    product_relative_error = _relative_error(candidate_theta, theta)
    common_product_numerical_tolerance_pass = (
        product_relative_error <= equality_relative_tolerance
    )
    if not common_product_numerical_tolerance_pass:
        raise ValueError(
            "structured tomography requires the same effective matrix; "
            f"relative discrepancy is {product_relative_error:.3e}"
        )
    k = int(probabilities.shape[1])
    design = build_tomography_design(theta, k)
    diagnostics = design_diagnostics(design, theta)
    reference_design_rank_dimension_pass = bool(
        diagnostics["dimension_condition_D_minus_K_ge_L"]
        and diagnostics["theta_rank"] == k
    )
    common_design_theorem_applies = bool(
        reference_design_rank_dimension_pass
        and exact_common_product_holds_by_symbolic_gauge_construction
    )

    reference_responses: List[torch.Tensor] = []
    candidate_responses: List[torch.Tensor] = []
    probe_records: List[Dict[str, Any]] = []
    for index in range(design.probe_count):
        gradient = tomography_probe(design, index)
        reference = apply_euclidean_response(
            probabilities,
            bases,
            gradient,
            eta_z=eta_z,
            eta_b=eta_b,
            temperature=temperature,
        )
        candidate = apply_euclidean_response(
            candidate_probabilities,
            candidate_bases,
            gradient,
            eta_z=eta_z,
            eta_b=eta_b,
            temperature=temperature,
        )
        difference = candidate - reference
        reference_norm = float(reference.double().norm().detach().cpu())
        candidate_norm = float(candidate.double().norm().detach().cpu())
        difference_norm = float(difference.double().norm().detach().cpu())
        denominator = max(
            reference_norm,
            candidate_norm,
            torch.finfo(torch.float64).tiny,
        )
        probe_records.append(
            {
                "index": index,
                "construction": (
                    "orthogonal_complement_layer_codes_plus_row_basis_1"
                    if index == 0
                    else f"shared_row_basis_{index + 1}"
                ),
                "gradient_frobenius_norm": float(
                    gradient.double().norm().detach().cpu()
                ),
                "reference_response_frobenius_norm": reference_norm,
                "candidate_response_frobenius_norm": candidate_norm,
                "response_difference_frobenius_norm": difference_norm,
                "relative_response_difference": difference_norm / denominator,
            }
        )
        reference_responses.append(reference)
        candidate_responses.append(candidate)

    reference_gram, reference_local = _reconstruct_components(
        reference_responses, design, eta_b
    )
    candidate_gram, candidate_local = _reconstruct_components(
        candidate_responses, design, eta_b
    )
    true_reference_gram, true_reference_local = _component_truth(
        probabilities, bases, design, eta_z, temperature
    )
    true_candidate_gram, true_candidate_local = _component_truth(
        candidate_probabilities,
        candidate_bases,
        design,
        eta_z,
        temperature,
    )
    maximum_probe_relative_difference = max(
        record["relative_response_difference"] for record in probe_records
    )
    response_equal_within_tolerance = (
        maximum_probe_relative_difference <= equality_relative_tolerance
    )
    reconstruction = {
        "reference_router_gram_relative_error": _relative_error(
            reference_gram, true_reference_gram
        ),
        "candidate_router_gram_relative_error": _relative_error(
            candidate_gram, true_candidate_gram
        ),
        "reference_local_blocks_relative_error": _relative_error(
            reference_local, true_reference_local
        ),
        "candidate_local_blocks_relative_error": _relative_error(
            candidate_local, true_candidate_local
        ),
        "router_gram_pair_relative_difference": _relative_error(
            candidate_gram, reference_gram
        ),
        "local_blocks_pair_relative_difference": _relative_error(
            candidate_local, reference_local
        ),
        "maximum_probe_relative_response_difference": (
            maximum_probe_relative_difference
        ),
    }
    return {
        "format": "structured-response-tomography-v2",
        "same_effective_product_relative_error": product_relative_error,
        "common_product_numerical_tolerance_pass": (
            common_product_numerical_tolerance_pass
        ),
        "exact_common_product_holds_by_symbolic_gauge_construction": bool(
            exact_common_product_holds_by_symbolic_gauge_construction
        ),
        "design": diagnostics,
        "probe_records": probe_records,
        "reconstruction": reconstruction,
        "reference_design_rank_dimension_pass": (
            reference_design_rank_dimension_pass
        ),
        "common_design_operator_determining_theorem_applies": (
            common_design_theorem_applies
        ),
        "response_equal_on_operator_determining_probes_within_tolerance": (
            response_equal_within_tolerance
        ),
        "response_operators_distinguished_within_tolerance": (
            not response_equal_within_tolerance
        ),
        "equality_relative_tolerance": float(equality_relative_tolerance),
        "interpretation": (
            "The reference design satisfies the recorded rank/dimension checks. "
            "These K probes determine both complete structured Euclidean responses "
            "under the additional exact-common-product construction attestation. "
            "The product tolerance is only an implementation check, not an "
            "approximate-product theorem, noisy-observation confidence interval, "
            "or historical-graph certificate."
        ),
    }
