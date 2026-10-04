"""Instantaneous Euclidean-response audits for shared-router factor models.

The estimand is deliberately narrow.  For a current factorization
``Theta = A @ B`` with ``A = row_softmax(Z / temperature)``, this module
computes the response of the *layer-effective parameter vectors* to an
arbitrary effective gradient ``G`` under hypothetical Euclidean gradient flow
in ``Z`` and ``B``.  It does not model AdamW state and it does not identify a
historical or generative sharing graph.

The implementation applies the response directly to probes and never
materializes its ``(n*d) x (n*d)`` matrix.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from src.models import BasisLinear


@dataclass(frozen=True)
class BasisSlice:
    """One contiguous tensor in the canonical packed basis matrix."""

    key: str
    module_name: str
    parameter_name: str
    start: int
    end: int
    parameter_shape_without_basis: Tuple[int, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["parameter_shape_without_basis"] = list(
            self.parameter_shape_without_basis
        )
        return payload


@dataclass
class PackedRouterFactors:
    """Router probabilities and all mixed BasisLinear parameters."""

    router: torch.Tensor
    bases: torch.Tensor
    layout: List[BasisSlice]
    temperature: float

    @property
    def depth(self) -> int:
        return int(self.router.shape[0])

    @property
    def num_bases(self) -> int:
        return int(self.router.shape[1])

    @property
    def effective_dimension(self) -> int:
        return int(self.bases.shape[1])


def basis_linear_modules(model: torch.nn.Module) -> List[Tuple[str, BasisLinear]]:
    """Return BasisLinear modules in deterministic registration order."""

    modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, BasisLinear)
    ]
    if not modules:
        raise ValueError("model contains no BasisLinear modules")
    return modules


def pack_basis_matrix(
    model: torch.nn.Module,
    temperature: float = 1.0,
) -> PackedRouterFactors:
    """Pack BasisLinear weight/bias tensors into ``B[K,d]``.

    Ordering is model registration order; within each BasisLinear, weight is
    followed by bias when present.  Consequently every row of ``A @ B`` can be
    unpacked into the exact effective weight/bias tensors used at that layer.
    """

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    try:
        router = model.router.probabilities(temperature, hard=False).detach()
    except AttributeError as error:
        raise TypeError("model must expose router.probabilities") from error
    if router.ndim != 2:
        raise ValueError("router probabilities must have shape [depth, K]")
    k = int(router.shape[1])
    pieces: List[torch.Tensor] = []
    layout: List[BasisSlice] = []
    offset = 0
    for module_name, module in basis_linear_modules(model):
        for parameter_name, parameter in (
            ("weight", module.weight),
            ("bias", module.bias),
        ):
            if parameter is None:
                continue
            if parameter.shape[0] != k:
                raise ValueError("BasisLinear rank disagrees with router width")
            flat = parameter.detach().reshape(k, -1)
            width = int(flat.shape[1])
            key = "%s.%s" % (module_name, parameter_name)
            pieces.append(flat)
            layout.append(
                BasisSlice(
                    key=key,
                    module_name=module_name,
                    parameter_name=parameter_name,
                    start=offset,
                    end=offset + width,
                    parameter_shape_without_basis=tuple(parameter.shape[1:]),
                )
            )
            offset += width
    bases = torch.cat(pieces, dim=1)
    if bases.shape != (k, offset):
        raise AssertionError("packed basis shape does not match its layout")
    return PackedRouterFactors(router, bases, layout, float(temperature))


def unpack_effective_vectors(
    effective_vectors: torch.Tensor,
    layout: Sequence[BasisSlice],
) -> Dict[str, torch.Tensor]:
    """Invert the canonical packing for a matrix with shape ``[depth,d]``."""

    if effective_vectors.ndim != 2:
        raise ValueError("effective_vectors must have shape [depth,d]")
    outputs: Dict[str, torch.Tensor] = {}
    expected_start = 0
    for item in layout:
        if item.start != expected_start or item.end <= item.start:
            raise ValueError("basis layout must be contiguous and nonempty")
        shape = (effective_vectors.shape[0],) + item.parameter_shape_without_basis
        outputs[item.key] = effective_vectors[:, item.start : item.end].reshape(shape)
        expected_start = item.end
    if expected_start != effective_vectors.shape[1]:
        raise ValueError("basis layout does not cover the effective vector")
    return outputs


def reconstruct_effective_tensors(
    packed: PackedRouterFactors,
) -> Dict[str, torch.Tensor]:
    return unpack_effective_vectors(packed.router @ packed.bases, packed.layout)


def audit_packing_reconstruction(
    model: torch.nn.Module,
    packed: PackedRouterFactors,
) -> Dict[str, Any]:
    """Compare packed reconstruction against every module's direct mixture."""

    reconstructed = reconstruct_effective_tensors(packed)
    direct: Dict[str, torch.Tensor] = {}
    for module_name, module in basis_linear_modules(model):
        direct[module_name + ".weight"] = torch.tensordot(
            packed.router, module.weight, dims=([1], [0])
        )
        if module.bias is not None:
            direct[module_name + ".bias"] = torch.tensordot(
                packed.router, module.bias, dims=([1], [0])
            )
    if list(reconstructed) != list(direct):
        raise RuntimeError("packed and direct effective tensor orders disagree")
    max_abs = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    per_tensor: Dict[str, float] = {}
    for key, reference in direct.items():
        difference = reconstructed[key] - reference
        error = float(difference.detach().double().abs().max().cpu())
        per_tensor[key] = error
        max_abs = max(max_abs, error)
        squared_error += float(difference.detach().double().square().sum().cpu())
        squared_reference += float(reference.detach().double().square().sum().cpu())
    relative = math.sqrt(squared_error / max(squared_reference, 1e-300))
    return {
        "max_abs_error": max_abs,
        "relative_l2_error": relative,
        "per_tensor_max_abs_error": per_tensor,
    }


def softmax_jacobian_squares(probabilities: torch.Tensor) -> torch.Tensor:
    """Return ``J_i^2`` for every row-softmax Jacobian."""

    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [n,K]")
    diagonal = torch.diag_embed(probabilities)
    jacobians = diagonal - probabilities.unsqueeze(2) * probabilities.unsqueeze(1)
    return jacobians @ jacobians


def apply_euclidean_response(
    probabilities: torch.Tensor,
    bases: torch.Tensor,
    effective_gradient: torch.Tensor,
    eta_z: float = 1.0,
    eta_b: float = 1.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Apply ``-dot(Theta)`` to one effective-gradient probe.

    The returned positive response is

    ``eta_b A A^T G + (eta_z/tau^2) row_i[G_i B^T J_i^2 B]``.

    Here ``J_i = diag(a_i) - a_i a_i^T`` omits the outer ``1/tau``;
    the two temperature factors are included explicitly in the coefficient.
    """

    if probabilities.ndim != 2 or bases.ndim != 2 or effective_gradient.ndim != 2:
        raise ValueError("A, B, and G must all be matrices")
    n, k = probabilities.shape
    if bases.shape[0] != k:
        raise ValueError("A and B have incompatible latent dimensions")
    if effective_gradient.shape != (n, bases.shape[1]):
        raise ValueError("G must have the same shape as A @ B")
    if probabilities.device != bases.device or bases.device != effective_gradient.device:
        raise ValueError("A, B, and G must be on the same device")
    if probabilities.dtype != bases.dtype or bases.dtype != effective_gradient.dtype:
        raise ValueError("A, B, and G must have the same dtype")
    if not math.isfinite(eta_z) or eta_z <= 0:
        raise ValueError("eta_z must be finite and positive")
    if not math.isfinite(eta_b) or eta_b <= 0:
        raise ValueError("eta_b must be finite and positive")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")

    # Associate as A(A^T G): O(n K d), rather than first materializing A A^T
    # and paying O(n^2 d).  Neither route materializes the nd-by-nd operator.
    basis_response = eta_b * (
        probabilities @ (probabilities.transpose(0, 1) @ effective_gradient)
    )
    jacobian_squares = softmax_jacobian_squares(probabilities)
    gradient_basis_coordinates = effective_gradient @ bases.transpose(0, 1)
    routed_coordinates = torch.einsum(
        "nk,nkl->nl", gradient_basis_coordinates, jacobian_squares
    )
    router_response = (eta_z / (temperature * temperature)) * (
        routed_coordinates @ bases
    )
    return basis_response + router_response


def cyclic_permutation(num_bases: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    if num_bases < 1:
        raise ValueError("num_bases must be positive")
    permutation = torch.zeros(num_bases, num_bases, dtype=dtype)
    for row in range(num_bases):
        permutation[row, (row + 1) % num_bases] = 1
    return permutation


def deterministic_positive_stochastic_gauge(
    num_bases: int,
    strength: float = 0.35,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """A preregisterable, validation-free non-permutation gauge."""

    if num_bases < 2:
        raise ValueError("a non-permutation gauge requires at least two bases")
    if not 0 < strength < 1:
        raise ValueError("strength must lie strictly between zero and one")
    identity = torch.eye(num_bases, dtype=dtype)
    uniform = torch.ones(num_bases, num_bases, dtype=dtype) / float(num_bases)
    return (1.0 - strength) * identity + strength * uniform


def transform_factors(
    probabilities: torch.Tensor,
    bases: torch.Tensor,
    gauge: torch.Tensor,
    min_probability: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Return ``A'=A M, B'=M^{-1}B`` after strict legality checks."""

    gauge = torch.as_tensor(
        gauge, dtype=probabilities.dtype, device=probabilities.device
    )
    k = probabilities.shape[1]
    if gauge.shape != (k, k):
        raise ValueError("gauge has the wrong shape")
    ones = torch.ones(k, dtype=gauge.dtype, device=gauge.device)
    row_sum_error = float((gauge @ ones - ones).abs().max().detach().cpu())
    if row_sum_error > 1e-9:
        raise ValueError("gauge does not satisfy M 1 = 1")
    condition = float(torch.linalg.cond(gauge.detach().double()).cpu())
    if not math.isfinite(condition):
        raise ValueError("gauge must be invertible")
    transformed_probabilities = probabilities @ gauge
    minimum = float(transformed_probabilities.min().detach().cpu())
    if minimum <= min_probability:
        raise ValueError("A M is not in the requested strict simplex interior")
    transformed_bases = torch.linalg.solve(gauge, bases)
    reference = probabilities @ bases
    candidate = transformed_probabilities @ transformed_bases
    difference = candidate - reference
    relative = float(
        difference.double().norm().cpu()
        / reference.double().norm().clamp_min(torch.finfo(torch.float64).tiny).cpu()
    )
    return transformed_probabilities, transformed_bases, {
        "condition_number": condition,
        "row_sum_max_abs_error": row_sum_error,
        "minimum_transformed_probability": minimum,
        "effective_max_abs_error": float(difference.abs().max().detach().cpu()),
        "effective_relative_l2_error": relative,
    }


def matrix_diagnostics(probabilities: torch.Tensor, bases: torch.Tensor) -> Dict[str, Any]:
    """Report exact theorem assumptions and conditioning diagnostics."""

    a = probabilities.detach().double().cpu()
    b = bases.detach().double().cpu()
    singular_a = torch.linalg.svdvals(a)
    singular_b = torch.linalg.svdvals(b)
    rank_a = int(torch.linalg.matrix_rank(a))
    rank_b = int(torch.linalg.matrix_rank(b))
    k = int(a.shape[1])
    sigma_min_a = float(singular_a.min())
    sigma_min_b = float(singular_b.min())
    sigma_max_a = float(singular_a.max())
    sigma_max_b = float(singular_b.max())
    return {
        "depth_n": int(a.shape[0]),
        "num_bases_K": k,
        "effective_dimension_d": int(b.shape[1]),
        "minimum_A_entry": float(a.min()),
        "maximum_A_entry": float(a.max()),
        "A_frobenius_norm": float(a.norm()),
        "B_frobenius_norm": float(b.norm()),
        "A_sigma_min": sigma_min_a,
        "B_sigma_min": sigma_min_b,
        "A_sigma_max": sigma_max_a,
        "B_sigma_max": sigma_max_b,
        "A_condition_number": (
            None if sigma_min_a == 0 else sigma_max_a / sigma_min_a
        ),
        "B_condition_number": (
            None if sigma_min_b == 0 else sigma_max_b / sigma_min_b
        ),
        "rank_A": rank_a,
        "rank_B": rank_b,
        "A_full_column_rank": rank_a == k,
        "B_full_row_rank": rank_b == k,
        "theorem_rank_assumptions_pass": rank_a == k and rank_b == k,
    }


def _probe_record(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    gradient: torch.Tensor,
    seed: int,
    threshold: float,
) -> Dict[str, Any]:
    difference = candidate - reference
    reference_norm = float(reference.double().norm().detach().cpu())
    candidate_norm = float(candidate.double().norm().detach().cpu())
    difference_norm = float(difference.double().norm().detach().cpu())
    gradient_norm = float(gradient.double().norm().detach().cpu())
    denominator = max(
        reference_norm,
        candidate_norm,
        torch.finfo(torch.float64).tiny,
    )
    relative = difference_norm / denominator
    return {
        "seed": int(seed),
        "gradient_frobenius_norm": gradient_norm,
        "reference_response_frobenius_norm": reference_norm,
        "candidate_response_frobenius_norm": candidate_norm,
        "response_difference_frobenius_norm": difference_norm,
        "response_difference_over_gradient_norm": difference_norm
        / max(gradient_norm, torch.finfo(torch.float64).tiny),
        "relative_response_difference": relative,
        "small_ball_detected": relative > threshold,
    }


def audit_fixed_gaussian_probes(
    probabilities: torch.Tensor,
    bases: torch.Tensor,
    transformed_probabilities: torch.Tensor,
    transformed_bases: torch.Tensor,
    probe_count: int = 8,
    probe_seed: int = 740_000,
    eta_z: float = 1.0,
    eta_b: float = 1.0,
    temperature: float = 1.0,
    small_ball_relative_threshold: float = 1e-8,
) -> Dict[str, Any]:
    """Stream fixed CPU-generated Gaussian probes through two responses."""

    if probe_count < 1:
        raise ValueError("probe_count must be positive")
    if small_ball_relative_threshold < 0:
        raise ValueError("small-ball threshold must be nonnegative")
    records: List[Dict[str, Any]] = []
    squared_differences: List[float] = []
    for index in range(probe_count):
        seed = int(probe_seed + index)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        gradient_cpu = torch.randn(
            probabilities.shape[0],
            bases.shape[1],
            generator=generator,
            dtype=probabilities.dtype,
            device="cpu",
        )
        gradient = gradient_cpu.to(device=probabilities.device)
        reference = apply_euclidean_response(
            probabilities, bases, gradient, eta_z, eta_b, temperature
        )
        candidate = apply_euclidean_response(
            transformed_probabilities,
            transformed_bases,
            gradient,
            eta_z,
            eta_b,
            temperature,
        )
        record = _probe_record(
            reference,
            candidate,
            gradient,
            seed,
            small_ball_relative_threshold,
        )
        records.append(record)
        squared_differences.append(record["response_difference_frobenius_norm"] ** 2)
        del gradient, gradient_cpu, reference, candidate
    relatives = sorted(record["relative_response_difference"] for record in records)
    detected = sum(bool(record["small_ball_detected"]) for record in records)
    return {
        "probe_distribution": "iid_standard_Gaussian_entries_generated_on_CPU",
        "probe_seed_start": int(probe_seed),
        "probe_count": int(probe_count),
        "small_ball_relative_threshold": float(small_ball_relative_threshold),
        "detected_probe_count": int(detected),
        "empirical_detection_rate": detected / float(probe_count),
        "minimum_relative_response_difference": relatives[0],
        "median_relative_response_difference": statistics.median(relatives),
        "maximum_relative_response_difference": relatives[-1],
        "mean_squared_response_difference": sum(squared_differences)
        / float(probe_count),
        "interpretation": (
            "empirical fixed-probe detection only; not a finite-sample recovery "
            "guarantee or a historical-structure claim"
        ),
        "probes": records,
    }


def autograd_response_validation(
    probabilities: torch.Tensor,
    bases: torch.Tensor,
    rows: int = 4,
    columns: int = 32,
    seed: int = 750_000,
    eta_z: float = 1.0,
    eta_b: float = 1.0,
    temperature: float = 1.0,
    finite_difference_steps: Sequence[float] = (1e-2, 1e-3, 1e-4, 1e-5),
) -> Dict[str, Any]:
    """Validate the formula on a small extracted abstract factor model."""

    selected_rows = min(int(rows), int(probabilities.shape[0]))
    selected_columns = min(int(columns), int(bases.shape[1]))
    if selected_rows < 1 or selected_columns < 1:
        raise ValueError("autograd validation needs at least one row and column")
    input_probabilities = probabilities[:selected_rows].detach().double().cpu()
    b = bases[:, :selected_columns].detach().double().cpu()
    z = (float(temperature) * input_probabilities.log()).requires_grad_(True)
    # A float32 checkpoint row can sum to 1 only up to rounding.  The response
    # being checked is that of the actual logits, so use their exactly realized
    # float64 softmax here and in the analytic formula.
    a = F.softmax(z.detach() / float(temperature), dim=-1)
    b_parameter = b.clone().requires_grad_(True)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gradient = torch.randn(
        selected_rows,
        selected_columns,
        generator=generator,
        dtype=torch.float64,
    )

    def effective(logits: torch.Tensor, basis_values: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / float(temperature), dim=-1) @ basis_values

    theta = effective(z, b_parameter)
    loss = (theta * gradient).sum()
    gradient_z, gradient_b = torch.autograd.grad(loss, (z, b_parameter))
    velocity_z = -float(eta_z) * gradient_z.detach()
    velocity_b = -float(eta_b) * gradient_b.detach()
    _, autograd_theta_dot = torch.autograd.functional.jvp(
        effective,
        (z.detach(), b_parameter.detach()),
        (velocity_z, velocity_b),
        create_graph=False,
        strict=True,
    )
    analytic_negative_dot = apply_euclidean_response(
        a,
        b,
        gradient,
        eta_z=eta_z,
        eta_b=eta_b,
        temperature=temperature,
    )
    analytic_theta_dot = -analytic_negative_dot
    analytic_error = autograd_theta_dot - analytic_theta_dot
    analytic_relative = float(
        analytic_error.norm()
        / autograd_theta_dot.norm().clamp_min(torch.finfo(torch.float64).tiny)
    )

    finite_records: List[Dict[str, float]] = []
    for step in finite_difference_steps:
        epsilon = float(step)
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError("finite-difference steps must be positive and finite")
        plus = effective(z.detach() + epsilon * velocity_z, b + epsilon * velocity_b)
        minus = effective(z.detach() - epsilon * velocity_z, b - epsilon * velocity_b)
        finite_dot = (plus - minus) / (2.0 * epsilon)
        difference = finite_dot - autograd_theta_dot
        finite_records.append(
            {
                "epsilon": epsilon,
                "max_abs_error": float(difference.abs().max()),
                "relative_l2_error": float(
                    difference.norm()
                    / autograd_theta_dot.norm().clamp_min(
                        torch.finfo(torch.float64).tiny
                    )
                ),
            }
        )
    return {
        "rows": selected_rows,
        "columns": selected_columns,
        "seed": int(seed),
        "linear_loss_effective_gradient": "fixed_G",
        "analytic_vs_autograd_max_abs_error": float(analytic_error.abs().max()),
        "analytic_vs_autograd_relative_l2_error": analytic_relative,
        "finite_difference": finite_records,
        "best_finite_difference_relative_l2_error": min(
            record["relative_l2_error"] for record in finite_records
        ),
    }


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(str(temporary), str(destination))
