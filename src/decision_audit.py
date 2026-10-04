"""Gauge audits for merge decisions in shared-``A`` LoRA models.

Shape and multiplication convention
-----------------------------------
There are ``L`` layer-specific factors ``B_i`` with shape ``(o, r)`` and one
shared factor ``A`` with shape ``(r, d)``.  The effective update at layer
``i`` is

``Delta_i = B_i @ A``.

The stacked ``bases`` tensor therefore has shape ``(L, o, r)``.  A history of
such tensors has shape ``(T, L, o, r)`` and must be explicitly reduced with
``running_average_bases`` (or by passing ``running_average=True`` to
``audit_pair_distances``).  The convention is deliberately explicit because
the LoRA literature sometimes swaps the names of the two factors.

For an invertible ``R`` of shape ``(r, r)``, the legal GL(r) gauge is

``A' = R @ A`` and ``B_i' = B_i @ inv(R)``.  This is the same convention as
``B_i'=B_i Q, A'=Q^{-1}A`` used in the
decision-theory ledger after the variable substitution ``Q=R^{-1}``.

It preserves every ``B_i @ A``.  For a candidate pair ``p=(i,j)``, define

``D_p = B_i - B_j`` and ``S_p = D_p.T @ D_p``.

The squared raw factor distance after the gauge is ``<C,S_p>``, where
``C = inv(R) @ inv(R).T`` and ``<X,Y> = trace(X.T @ Y)``.  In contrast,
``||(B_i-B_j)A||_F`` and its activation-weighted analogue are gauge invariant.

Rank boundary
-------------
Gauge invariance only needs ``R`` to be invertible; it does *not* require
``A`` to have full row rank.  Full row rank is needed for the effective-update
distance to be a genuine metric on the ``B_i`` factors.  If ``rank(A) < r``,
it is only a seminorm: different ``B_i`` can differ in a null direction and
still have identical products.  ``audit_pair_distances`` reports this boundary
rather than silently assuming it away.

Selectability
-------------
For a finite candidate set, the exact mathematical margin for selecting pair
``p`` under some gauge is

``kappa_p = max_{C >= 0, trace(C)=1} min_{q != p} <C, S_q-S_p>``.

This is a semidefinite program.  This module intentionally has no SDP
dependency and never labels a numerical search result as the exact kappa.
``fixed_metric_selectability_witness`` verifies a supplied certificate;
``selectability_bounds`` returns rigorous feasible lower/upper bounds; and
``approximate_selectability_margin`` merely searches for better feasible
certificates with projected subgradient iterations in Torch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


Pair = Tuple[int, int]


@dataclass
class PairScore:
    """Distances for one (canonically ordered) layer pair."""

    pair: Pair
    S_p: torch.Tensor
    raw_distance: float
    effective_distance: float
    activation_weighted_distance: Optional[float]


@dataclass
class PairDistanceAudit:
    """Pair scores together with the rank boundary of the shared factor."""

    scores: Dict[Pair, PairScore]
    bases: torch.Tensor
    used_running_average: bool
    running_average_steps: Optional[int]
    a_rank: int
    a_full_row_rank: bool
    rank_tolerance: float

    @property
    def scatter_matrices(self) -> Dict[Pair, torch.Tensor]:
        return {pair: score.S_p for pair, score in self.scores.items()}


@dataclass
class GaugeProductAudit:
    """A gauge-transformed factorization and its numerical product check."""

    transformed_bases: torch.Tensor
    transformed_A: torch.Tensor
    inverse_R: torch.Tensor
    metric_C: torch.Tensor
    products_before: torch.Tensor
    products_after: torch.Tensor
    max_absolute_error: float
    relative_frobenius_error: float
    products_close: bool
    condition_number_R: float


@dataclass
class AllGaugeOrderAudit:
    """Exact (up to ``eigenvalue_tolerance``) order classification for two pairs."""

    classification: str
    p_no_farther_for_all_gauges: bool
    q_no_farther_for_all_gauges: bool
    eigenvalues_Sq_minus_Sp: torch.Tensor
    eigenvalue_tolerance: float


@dataclass
class SelectabilityWitness:
    """Verification of one feasible density-matrix selectability witness."""

    target: Pair
    competitors: Tuple[Pair, ...]
    metric_C: torch.Tensor
    gaps: Dict[Pair, float]
    lower_bound: float
    minimum_eigenvalue_C: float
    metric_is_spd: bool
    certifies_selectable: bool
    spd_metric_C: Optional[torch.Tensor]
    spd_lower_bound: Optional[float]
    spd_gauge_R: Optional[torch.Tensor]


@dataclass
class SelectabilityBounds:
    """Certified bounds obtained from feasible primal and dual witnesses."""

    target: Pair
    competitors: Tuple[Pair, ...]
    metric_C: torch.Tensor
    mixture_weights: torch.Tensor
    lower_bound: float
    upper_bound: float
    certifies_selectable: bool
    certifies_not_selectable: bool


@dataclass
class ApproximateSelectabilityResult:
    """Output of a numerical certificate search, not an exact SDP solution."""

    target: Pair
    competitors: Tuple[Pair, ...]
    iterations: int
    metric_C: torch.Tensor
    mixture_weights: torch.Tensor
    lower_bound: float
    upper_bound: float
    certificate_gap: float
    certifies_selectable: bool
    certifies_not_selectable: bool


@dataclass
class UnboundedFamilyRecord:
    """One finite member of the analytic r=2 unbounded-regret model family."""

    scale: float
    raw_distance_p_before: float
    raw_distance_q_before: float
    raw_distance_p_after: float
    raw_distance_q_after: float
    selected_pair_before: Pair
    selected_pair_after: Pair
    effective_distance_p: float
    effective_distance_q: float
    invariant_cost_p: float
    invariant_cost_q: float
    approximation_ratio: float
    additive_regret: float
    max_product_error: float


def _floating_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.is_complex():
        raise ValueError("%s must be real" % name)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("%s must contain only finite values" % name)
    return tensor


def _canonical_pair(pair: Sequence[int], num_layers: Optional[int] = None) -> Pair:
    if len(pair) != 2:
        raise ValueError("each candidate pair must contain exactly two layer indices")
    i, j = int(pair[0]), int(pair[1])
    if i == j:
        raise ValueError("a candidate pair must contain two different layers")
    if i < 0 or j < 0 or (num_layers is not None and (i >= num_layers or j >= num_layers)):
        raise ValueError("candidate pair %r is outside the layer range" % ((i, j),))
    return (i, j) if i < j else (j, i)


def _canonical_pairs(pairs: Iterable[Sequence[int]], num_layers: int) -> Tuple[Pair, ...]:
    result: List[Pair] = []
    seen = set()
    for pair in pairs:
        canonical = _canonical_pair(pair, num_layers)
        if canonical in seen:
            raise ValueError("duplicate candidate pair %r" % (canonical,))
        seen.add(canonical)
        result.append(canonical)
    if not result:
        raise ValueError("at least one candidate pair is required")
    return tuple(result)


def running_average_bases(history: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
    """Return the average of history entries ``0, ..., steps-1``.

    ``history`` has shape ``(T,L,o,r)``.  ``steps=None`` uses the complete
    history.  This is the arithmetic running average used by ASLoRA at the
    selected audit time; it does not modify the input history.
    """

    history = _floating_tensor(history, "history")
    if history.ndim != 4:
        raise ValueError("history must have shape (T, L, o, r)")
    total_steps = int(history.shape[0])
    if total_steps == 0:
        raise ValueError("history must contain at least one step")
    used_steps = total_steps if steps is None else int(steps)
    if used_steps < 1 or used_steps > total_steps:
        raise ValueError("steps must lie between 1 and the history length")
    return history[:used_steps].mean(dim=0)


def _rank_information(A: torch.Tensor, rank_tolerance: Optional[float]) -> Tuple[int, bool, float]:
    singular_values = torch.linalg.svdvals(A)
    largest = float(singular_values.max()) if singular_values.numel() else 0.0
    if rank_tolerance is None:
        eps = torch.finfo(A.dtype).eps
        tolerance = max(A.shape) * eps * largest
    else:
        tolerance = float(rank_tolerance)
        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("rank_tolerance must be finite and nonnegative")
    rank = int((singular_values > tolerance).sum().item())
    return rank, rank == int(A.shape[0]), tolerance


def _activation_covariance(
    d: int,
    reference: torch.Tensor,
    activation_covariance: Optional[torch.Tensor],
    activations: Optional[torch.Tensor],
    psd_tolerance: float,
) -> Optional[torch.Tensor]:
    if activation_covariance is not None and activations is not None:
        raise ValueError("pass activation_covariance or activations, not both")
    covariance: Optional[torch.Tensor]
    if activations is not None:
        values = torch.as_tensor(activations, dtype=reference.dtype, device=reference.device)
        if values.ndim != 2 or values.shape[1] != d or values.shape[0] == 0:
            raise ValueError("activations must have shape (n, d) with n > 0")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("activations must contain only finite values")
        covariance = values.transpose(0, 1) @ values / int(values.shape[0])
    elif activation_covariance is not None:
        covariance = torch.as_tensor(
            activation_covariance, dtype=reference.dtype, device=reference.device
        )
        if covariance.shape != (d, d):
            raise ValueError("activation_covariance must have shape (d, d)")
        if not bool(torch.isfinite(covariance).all()):
            raise ValueError("activation_covariance must contain only finite values")
        asymmetry = float((covariance - covariance.transpose(0, 1)).abs().max())
        if asymmetry > psd_tolerance:
            raise ValueError("activation_covariance must be symmetric")
        covariance = 0.5 * (covariance + covariance.transpose(0, 1))
        minimum = float(torch.linalg.eigvalsh(covariance).min())
        if minimum < -psd_tolerance:
            raise ValueError("activation_covariance must be positive semidefinite")
    else:
        covariance = None
    return covariance


def audit_pair_distances(
    bases: torch.Tensor,
    A: torch.Tensor,
    pairs: Iterable[Sequence[int]],
    *,
    running_average: bool = False,
    average_steps: Optional[int] = None,
    activation_covariance: Optional[torch.Tensor] = None,
    activations: Optional[torch.Tensor] = None,
    rank_tolerance: Optional[float] = None,
    psd_tolerance: float = 1e-7,
) -> PairDistanceAudit:
    """Compute ``S_p`` and three distances for candidate merge pairs.

    The raw distance is ``||B_i-B_j||_F``.  The effective distance is
    ``||(B_i-B_j)A||_F``.  Given an uncentred activation second moment
    ``Sigma`` (or activation rows from which it is computed), the weighted
    distance is

    ``sqrt(trace((B_i-B_j) A Sigma A.T (B_i-B_j).T))``.

    The latter is RMS output discrepancy under the empirical activations.
    No activation-weighted value is reported when neither activation input is
    supplied.
    """

    bases = _floating_tensor(bases, "bases")
    if running_average:
        selected_bases = running_average_bases(bases, average_steps)
        used_steps = int(bases.shape[0]) if average_steps is None else int(average_steps)
    else:
        if average_steps is not None:
            raise ValueError("average_steps is only valid with running_average=True")
        if bases.ndim != 3:
            raise ValueError("bases must have shape (L, o, r)")
        selected_bases = bases
        used_steps = None

    if selected_bases.ndim != 3:
        raise ValueError("averaged bases must have shape (L, o, r)")
    num_layers, _, rank_dimension = selected_bases.shape
    if num_layers < 2 or rank_dimension < 1:
        raise ValueError("bases need at least two layers and a positive rank dimension")

    A = torch.as_tensor(A, dtype=selected_bases.dtype, device=selected_bases.device)
    if A.ndim != 2 or A.shape[0] != rank_dimension:
        raise ValueError("A must have shape (r, d), matching the final dimension of bases")
    if not bool(torch.isfinite(A).all()):
        raise ValueError("A must contain only finite values")
    canonical_pairs = _canonical_pairs(pairs, int(num_layers))
    covariance = _activation_covariance(
        int(A.shape[1]), selected_bases, activation_covariance, activations, psd_tolerance
    )
    a_rank, full_row_rank, used_rank_tolerance = _rank_information(A, rank_tolerance)

    scores: Dict[Pair, PairScore] = {}
    for pair in canonical_pairs:
        i, j = pair
        difference = selected_bases[i] - selected_bases[j]
        scatter = difference.transpose(0, 1) @ difference
        raw_squared = torch.trace(scatter).clamp_min(0)
        effective_difference = difference @ A
        effective_squared = (effective_difference * effective_difference).sum().clamp_min(0)
        weighted: Optional[float] = None
        if covariance is not None:
            weighted_squared = torch.trace(
                effective_difference @ covariance @ effective_difference.transpose(0, 1)
            ).clamp_min(0)
            weighted = float(torch.sqrt(weighted_squared))
        scores[pair] = PairScore(
            pair=pair,
            S_p=scatter,
            raw_distance=float(torch.sqrt(raw_squared)),
            effective_distance=float(torch.sqrt(effective_squared)),
            activation_weighted_distance=weighted,
        )
    return PairDistanceAudit(
        scores=scores,
        bases=selected_bases,
        used_running_average=running_average,
        running_average_steps=used_steps,
        a_rank=a_rank,
        a_full_row_rank=full_row_rank,
        rank_tolerance=used_rank_tolerance,
    )


def apply_gl_gauge(
    bases: torch.Tensor, A: torch.Tensor, R: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ``A'=R A, B'=B R^{-1}`` and return ``(B', A', R^{-1})``.

    ``bases`` may have shape ``(..., o, r)``, so the same operation also
    applies pointwise to a complete running-average history.  Invertibility of
    ``R`` is checked by solving a linear system.  No rank assumption is made
    about ``A``.
    """

    bases = _floating_tensor(bases, "bases")
    if bases.ndim < 2:
        raise ValueError("bases must have shape (..., o, r)")
    rank_dimension = int(bases.shape[-1])
    A = torch.as_tensor(A, dtype=bases.dtype, device=bases.device)
    R = torch.as_tensor(R, dtype=bases.dtype, device=bases.device)
    if A.ndim != 2 or A.shape[0] != rank_dimension:
        raise ValueError("A must have shape (r, d)")
    if R.shape != (rank_dimension, rank_dimension):
        raise ValueError("R must have shape (r, r)")
    if not bool(torch.isfinite(A).all()) or not bool(torch.isfinite(R).all()):
        raise ValueError("A and R must contain only finite values")
    identity = torch.eye(rank_dimension, dtype=bases.dtype, device=bases.device)
    try:
        inverse_R = torch.linalg.solve(R, identity)
    except RuntimeError as error:
        raise ValueError("R must be invertible") from error
    if not bool(torch.isfinite(inverse_R).all()):
        raise ValueError("R is numerically singular")
    transformed_bases = torch.matmul(bases, inverse_R)
    transformed_A = R @ A
    return transformed_bases, transformed_A, inverse_R


def verify_gl_gauge(
    bases: torch.Tensor,
    A: torch.Tensor,
    R: torch.Tensor,
    *,
    atol: float = 1e-8,
    rtol: float = 1e-6,
) -> GaugeProductAudit:
    """Apply a GL gauge and numerically verify every effective product."""

    bases = _floating_tensor(bases, "bases")
    A = torch.as_tensor(A, dtype=bases.dtype, device=bases.device)
    transformed_bases, transformed_A, inverse_R = apply_gl_gauge(bases, A, R)
    products_before = torch.matmul(bases, A)
    products_after = torch.matmul(transformed_bases, transformed_A)
    error = products_after - products_before
    max_absolute = float(error.abs().max()) if error.numel() else 0.0
    denominator = float(torch.linalg.norm(products_before))
    relative = float(torch.linalg.norm(error)) / max(denominator, torch.finfo(bases.dtype).tiny)
    R_tensor = torch.as_tensor(R, dtype=bases.dtype, device=bases.device)
    return GaugeProductAudit(
        transformed_bases=transformed_bases,
        transformed_A=transformed_A,
        inverse_R=inverse_R,
        metric_C=inverse_R @ inverse_R.transpose(0, 1),
        products_before=products_before,
        products_after=products_after,
        max_absolute_error=max_absolute,
        relative_frobenius_error=relative,
        products_close=bool(torch.allclose(products_before, products_after, atol=atol, rtol=rtol)),
        condition_number_R=float(torch.linalg.cond(R_tensor)),
    )


def _symmetric_scatter(scatter: torch.Tensor, name: str, tolerance: float) -> torch.Tensor:
    scatter = _floating_tensor(scatter, name)
    if scatter.ndim != 2 or scatter.shape[0] != scatter.shape[1]:
        raise ValueError("%s must be a square matrix" % name)
    asymmetry = float((scatter - scatter.transpose(0, 1)).abs().max())
    if asymmetry > tolerance:
        raise ValueError("%s must be symmetric" % name)
    return 0.5 * (scatter + scatter.transpose(0, 1))


def metric_distance(
    S_p: torch.Tensor, C: torch.Tensor, *, validation_tolerance: float = 1e-7
) -> float:
    """Return ``sqrt(<C,S_p>)`` after validating an SPD/PSD metric ``C``."""

    S_p = _symmetric_scatter(S_p, "S_p", validation_tolerance)
    C = torch.as_tensor(C, dtype=S_p.dtype, device=S_p.device)
    C = _symmetric_scatter(C, "C", validation_tolerance)
    if C.shape != S_p.shape:
        raise ValueError("C and S_p must have the same shape")
    if float(torch.linalg.eigvalsh(C).min()) < -validation_tolerance:
        raise ValueError("C must be positive semidefinite")
    squared = torch.sum(C * S_p).clamp_min(0)
    return float(torch.sqrt(squared))


def audit_all_gauge_order(
    S_p: torch.Tensor,
    S_q: torch.Tensor,
    *,
    eigenvalue_tolerance: float = 1e-9,
) -> AllGaugeOrderAudit:
    """Classify the raw order of two pairs under *all* GL gauges.

    Pair ``p`` is no farther than ``q`` for every gauge iff ``S_q-S_p`` is
    PSD.  The reverse order holds for every gauge iff it is NSD.  Otherwise
    the difference is indefinite and suitable SPD gauges realize both strict
    orders.  This is an exact eigenvalue characterization in exact arithmetic;
    ``eigenvalue_tolerance`` states the numerical boundary used here.
    """

    if eigenvalue_tolerance < 0:
        raise ValueError("eigenvalue_tolerance must be nonnegative")
    S_p = _symmetric_scatter(S_p, "S_p", eigenvalue_tolerance)
    S_q = torch.as_tensor(S_q, dtype=S_p.dtype, device=S_p.device)
    S_q = _symmetric_scatter(S_q, "S_q", eigenvalue_tolerance)
    if S_p.shape != S_q.shape:
        raise ValueError("S_p and S_q must have the same shape")
    eigenvalues = torch.linalg.eigvalsh(S_q - S_p)
    p_all = float(eigenvalues.min()) >= -eigenvalue_tolerance
    q_all = float(eigenvalues.max()) <= eigenvalue_tolerance
    if p_all and q_all:
        classification = "tie_for_all_gauges"
    elif p_all:
        classification = "p_no_farther_for_all_gauges"
    elif q_all:
        classification = "q_no_farther_for_all_gauges"
    else:
        classification = "order_flippable_by_gauge"
    return AllGaugeOrderAudit(
        classification=classification,
        p_no_farther_for_all_gauges=p_all,
        q_no_farther_for_all_gauges=q_all,
        eigenvalues_Sq_minus_Sp=eigenvalues,
        eigenvalue_tolerance=eigenvalue_tolerance,
    )


def _validated_scatter_mapping(
    scatter_matrices: Mapping[Pair, torch.Tensor], tolerance: float
) -> Tuple[Tuple[Pair, ...], Dict[Pair, torch.Tensor]]:
    if len(scatter_matrices) < 2:
        raise ValueError("selectability requires at least two candidate pairs")
    validated: Dict[Pair, torch.Tensor] = {}
    keys: List[Pair] = []
    reference: Optional[torch.Tensor] = None
    for raw_pair, raw_scatter in scatter_matrices.items():
        pair = _canonical_pair(raw_pair)
        if pair in validated:
            raise ValueError("duplicate candidate pair %r" % (pair,))
        if reference is None:
            scatter = _symmetric_scatter(raw_scatter, "S_%r" % (pair,), tolerance)
            reference = scatter
        else:
            scatter = torch.as_tensor(raw_scatter, dtype=reference.dtype, device=reference.device)
            scatter = _symmetric_scatter(scatter, "S_%r" % (pair,), tolerance)
            if scatter.shape != reference.shape:
                raise ValueError("all S_p matrices must have the same shape")
        minimum = float(torch.linalg.eigvalsh(scatter).min())
        if minimum < -tolerance:
            raise ValueError("each S_p must be positive semidefinite")
        validated[pair] = scatter
        keys.append(pair)
    return tuple(keys), validated


def _density_matrix(C: torch.Tensor, reference: torch.Tensor, tolerance: float) -> torch.Tensor:
    C = torch.as_tensor(C, dtype=reference.dtype, device=reference.device)
    C = _symmetric_scatter(C, "C", tolerance)
    if C.shape != reference.shape:
        raise ValueError("C must have the same shape as S_p")
    eigenvalues = torch.linalg.eigvalsh(C)
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("C must be positive semidefinite")
    trace = float(torch.trace(C))
    if abs(trace - 1.0) > tolerance:
        raise ValueError("C must have trace one")
    # Return an actually feasible floating-point witness.  Tiny negative
    # eigenvalues within the declared validation tolerance are clipped rather
    # than leaked into a purported PSD certificate.
    if float(eigenvalues.min()) < 0.0:
        eigenvalues, eigenvectors = torch.linalg.eigh(C)
        eigenvalues = eigenvalues.clamp_min(0)
        C = (eigenvectors * eigenvalues.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    return C / torch.trace(C)


def _target_and_differences(
    scatter_matrices: Mapping[Pair, torch.Tensor], target: Sequence[int], tolerance: float
) -> Tuple[Pair, Tuple[Pair, ...], torch.Tensor, List[torch.Tensor]]:
    keys, validated = _validated_scatter_mapping(scatter_matrices, tolerance)
    target_pair = _canonical_pair(target)
    if target_pair not in validated:
        raise ValueError("target pair is not in scatter_matrices")
    competitors = tuple(pair for pair in keys if pair != target_pair)
    target_scatter = validated[target_pair]
    differences = [validated[pair] - target_scatter for pair in competitors]
    return target_pair, competitors, target_scatter, differences


def _inner_product(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.sum(left * right)


def _safe_spd_perturbation(
    C: torch.Tensor,
    differences: Sequence[torch.Tensor],
    lower_bound: float,
    gap_tolerance: float,
) -> Tuple[Optional[torch.Tensor], Optional[float]]:
    """Turn a positive-gap PSD density witness into an explicit SPD witness."""

    if lower_bound <= gap_tolerance:
        return None, None
    # Remove only validation-scale negative eigenvalues before mixing.  The
    # gaps are recomputed, so the returned witness remains directly checkable.
    eigenvalues, eigenvectors = torch.linalg.eigh(0.5 * (C + C.transpose(0, 1)))
    projected_values = eigenvalues.clamp_min(0)
    projected = (eigenvectors * projected_values.unsqueeze(0)) @ eigenvectors.transpose(0, 1)
    projected = projected / torch.trace(projected)
    projected_gaps = torch.stack([_inner_product(projected, diff) for diff in differences])
    projected_lower = float(projected_gaps.min())
    if projected_lower <= gap_tolerance:
        return None, None

    rank_dimension = int(C.shape[0])
    identity_density = torch.eye(rank_dimension, dtype=C.dtype, device=C.device) / rank_dimension
    identity_lower = float(
        torch.stack([_inner_product(identity_density, diff) for diff in differences]).min()
    )
    # For epsilon in (0,1), (1-epsilon)C + epsilon I/r is SPD.  If the
    # identity direction has a negative gap, remain safely below the exact
    # zero-crossing.  A modest epsilon also makes the witness numerically SPD.
    epsilon = 1e-4
    if identity_lower < projected_lower:
        crossing = projected_lower / (projected_lower - identity_lower)
        epsilon = min(epsilon, 0.5 * crossing)
    epsilon = max(epsilon, 64.0 * torch.finfo(C.dtype).eps)
    if epsilon >= 1.0:
        return None, None
    spd = (1.0 - epsilon) * projected + epsilon * identity_density
    spd_gaps = torch.stack([_inner_product(spd, diff) for diff in differences])
    spd_lower = float(spd_gaps.min())
    if spd_lower <= gap_tolerance or float(torch.linalg.eigvalsh(spd).min()) <= 0:
        return None, None
    return spd, spd_lower


def gauge_from_metric(C: torch.Tensor, *, validation_tolerance: float = 1e-9) -> torch.Tensor:
    """Construct ``R`` satisfying ``C=inv(R)@inv(R).T`` for an SPD ``C``."""

    C = _floating_tensor(C, "C")
    C = _symmetric_scatter(C, "C", validation_tolerance)
    if abs(float(torch.trace(C)) - 1.0) > validation_tolerance:
        raise ValueError("C must have trace one")
    if float(torch.linalg.eigvalsh(C).min()) <= validation_tolerance:
        raise ValueError("C must be numerically positive definite")
    lower = torch.linalg.cholesky(C)
    identity = torch.eye(C.shape[0], dtype=C.dtype, device=C.device)
    # inv(R)=lower, hence R=inv(lower).
    return torch.linalg.solve(lower, identity)


def fixed_metric_selectability_witness(
    scatter_matrices: Mapping[Pair, torch.Tensor],
    target: Sequence[int],
    C: torch.Tensor,
    *,
    validation_tolerance: float = 1e-7,
    gap_tolerance: float = 1e-9,
) -> SelectabilityWitness:
    """Verify a fixed feasible ``C`` as a strict selectability certificate.

    A positive returned ``lower_bound`` is a rigorous feasible lower bound on
    ``kappa_p`` (subject only to the stated floating-point tolerances).  If the
    supplied ``C`` is singular, a positive gap is preserved by an explicit
    perturbation toward ``I/r``; the returned ``spd_metric_C`` and
    ``spd_gauge_R`` can be checked directly.
    """

    target_pair, competitors, reference, differences = _target_and_differences(
        scatter_matrices, target, validation_tolerance
    )
    C = _density_matrix(C, reference, validation_tolerance)
    gap_values = [_inner_product(C, difference) for difference in differences]
    gaps = {pair: float(value) for pair, value in zip(competitors, gap_values)}
    lower_bound = min(gaps.values())
    minimum_eigenvalue = float(torch.linalg.eigvalsh(C).min())
    metric_is_spd = minimum_eigenvalue > validation_tolerance
    spd_metric: Optional[torch.Tensor]
    spd_lower: Optional[float]
    if lower_bound > gap_tolerance:
        if metric_is_spd:
            spd_metric, spd_lower = C, lower_bound
        else:
            spd_metric, spd_lower = _safe_spd_perturbation(
                C, differences, lower_bound, gap_tolerance
            )
    else:
        spd_metric, spd_lower = None, None
    spd_gauge = None
    if spd_metric is not None:
        # The constructed metric can have a very small eigenvalue by design;
        # use a scale-aware machine threshold rather than the user validation
        # threshold when recovering its Cholesky factor.
        spd_gauge = gauge_from_metric(
            spd_metric, validation_tolerance=8.0 * torch.finfo(spd_metric.dtype).eps
        )
    return SelectabilityWitness(
        target=target_pair,
        competitors=competitors,
        metric_C=C,
        gaps=gaps,
        lower_bound=lower_bound,
        minimum_eigenvalue_C=minimum_eigenvalue,
        metric_is_spd=metric_is_spd,
        # A positive-gap PSD density matrix already proves existence of an SPD
        # witness by continuity, even in an extreme numerical case where an
        # explicit, well-conditioned Cholesky witness cannot be represented.
        certifies_selectable=lower_bound > gap_tolerance,
        spd_metric_C=spd_metric,
        spd_lower_bound=spd_lower,
        spd_gauge_R=spd_gauge,
    )


def _mixture_tensor(
    mixture_weights: Optional[torch.Tensor],
    num_competitors: int,
    reference: torch.Tensor,
    tolerance: float,
) -> torch.Tensor:
    if mixture_weights is None:
        return torch.full(
            (num_competitors,),
            1.0 / num_competitors,
            dtype=reference.dtype,
            device=reference.device,
        )
    weights = torch.as_tensor(mixture_weights, dtype=reference.dtype, device=reference.device)
    if weights.shape != (num_competitors,):
        raise ValueError("mixture_weights must have one entry per competitor")
    if not bool(torch.isfinite(weights).all()) or float(weights.min()) < -tolerance:
        raise ValueError("mixture_weights must be finite and nonnegative")
    if abs(float(weights.sum()) - 1.0) > tolerance:
        raise ValueError("mixture_weights must sum to one")
    # As for density matrices, make the returned dual witness genuinely
    # feasible after accepting validation-scale roundoff.
    weights = weights.clamp_min(0)
    return weights / weights.sum()


def selectability_bounds(
    scatter_matrices: Mapping[Pair, torch.Tensor],
    target: Sequence[int],
    *,
    C: Optional[torch.Tensor] = None,
    mixture_weights: Optional[torch.Tensor] = None,
    validation_tolerance: float = 1e-7,
    certificate_tolerance: float = 1e-9,
) -> SelectabilityBounds:
    """Return certified lower and upper bounds on the selectability margin.

    Any density matrix ``C`` gives the lower bound

    ``min_q <C,S_q-S_p> <= kappa_p``.

    Any simplex vector ``lambda`` gives the upper bound

    ``kappa_p <= lambda_max(sum_q lambda_q (S_q-S_p))``.

    The second inequality follows by replacing a minimum with a convex
    combination and maximizing the resulting linear functional over density
    matrices.  Minimizing this upper bound over ``lambda`` is the SDP dual
    (equivalently, the finite-dimensional minimax form), but optimality is not
    assumed here.  Defaults use ``I/r`` and a uniform competitor mixture.
    """

    target_pair, competitors, reference, differences = _target_and_differences(
        scatter_matrices, target, validation_tolerance
    )
    rank_dimension = int(reference.shape[0])
    if C is None:
        C_tensor = torch.eye(
            rank_dimension, dtype=reference.dtype, device=reference.device
        ) / rank_dimension
    else:
        C_tensor = _density_matrix(C, reference, validation_tolerance)
    weights = _mixture_tensor(
        mixture_weights, len(competitors), reference, validation_tolerance
    )
    gaps = torch.stack([_inner_product(C_tensor, difference) for difference in differences])
    lower_bound = float(gaps.min())
    mixture_matrix = torch.zeros_like(reference)
    for weight, difference in zip(weights, differences):
        mixture_matrix = mixture_matrix + weight * difference
    upper_bound = float(torch.linalg.eigvalsh(mixture_matrix).max())
    if lower_bound > upper_bound + 10.0 * validation_tolerance:
        raise RuntimeError("numerical failure: certified lower bound exceeds upper bound")
    return SelectabilityBounds(
        target=target_pair,
        competitors=competitors,
        metric_C=C_tensor,
        mixture_weights=weights,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        certifies_selectable=lower_bound > certificate_tolerance,
        # An upper bound must be genuinely nonpositive to rule out the strict
        # condition kappa_p > 0.  A small *positive* upper bound is not such a
        # certificate, even when it lies inside the reporting tolerance.
        certifies_not_selectable=upper_bound <= 0.0,
    )


def _project_simplex(vector: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of one vector onto the probability simplex."""

    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(1, vector.numel() + 1, dtype=vector.dtype, device=vector.device)
    positive = sorted_values - cumulative / indices > 0
    rho = int(torch.nonzero(positive, as_tuple=False)[-1, 0])
    threshold = cumulative[rho] / float(rho + 1)
    projected = (vector - threshold).clamp_min(0)
    return projected / projected.sum()


def _project_density(matrix: torch.Tensor) -> torch.Tensor:
    matrix = 0.5 * (matrix + matrix.transpose(0, 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    projected_values = _project_simplex(eigenvalues)
    return (eigenvectors * projected_values.unsqueeze(0)) @ eigenvectors.transpose(0, 1)


def approximate_selectability_margin(
    scatter_matrices: Mapping[Pair, torch.Tensor],
    target: Sequence[int],
    *,
    iterations: int = 1000,
    learning_rate: float = 0.5,
    validation_tolerance: float = 1e-7,
    certificate_tolerance: float = 1e-9,
) -> ApproximateSelectabilityResult:
    """Search for tight feasible selectability certificates with Torch.

    This is *not* an exact SDP solver and the returned values are *not* called
    ``kappa_p``.  Projected subgradient ascent searches density matrices for a
    large lower bound, while projected subgradient descent searches competitor
    mixtures for a small valid upper bound.  Every retained iterate is
    feasible, so the final lower/upper values remain independently checkable
    certificates even if the optimization has not converged.
    """

    if iterations < 1:
        raise ValueError("iterations must be positive")
    if learning_rate <= 0 or not math.isfinite(learning_rate):
        raise ValueError("learning_rate must be finite and positive")
    target_pair, competitors, reference, differences = _target_and_differences(
        scatter_matrices, target, validation_tolerance
    )
    rank_dimension = int(reference.shape[0])
    num_competitors = len(competitors)
    C = torch.eye(rank_dimension, dtype=reference.dtype, device=reference.device) / rank_dimension
    weights = torch.full(
        (num_competitors,),
        1.0 / num_competitors,
        dtype=reference.dtype,
        device=reference.device,
    )
    best_C = C.clone()
    best_lower = float("-inf")
    best_weights = weights.clone()
    best_upper = float("inf")
    average_C = torch.zeros_like(C)
    average_weights = torch.zeros_like(weights)

    with torch.no_grad():
        for iteration in range(iterations):
            step_size = learning_rate / math.sqrt(iteration + 1.0)

            gap_values = torch.stack([_inner_product(C, difference) for difference in differences])
            lower = float(gap_values.min())
            if lower > best_lower:
                best_lower, best_C = lower, C.clone()
            worst_index = int(torch.argmin(gap_values))
            C = _project_density(C + step_size * differences[worst_index])

            mixture_matrix = torch.zeros_like(reference)
            for weight, difference in zip(weights, differences):
                mixture_matrix = mixture_matrix + weight * difference
            eigenvalues, eigenvectors = torch.linalg.eigh(mixture_matrix)
            upper = float(eigenvalues[-1])
            if upper < best_upper:
                best_upper, best_weights = upper, weights.clone()
            top_vector = eigenvectors[:, -1]
            weight_gradient = torch.stack(
                [top_vector @ difference @ top_vector for difference in differences]
            )
            weights = _project_simplex(weights - step_size * weight_gradient)

            # Ergodic averages often stabilize nonsmooth subgradient paths.
            average_C = average_C + (C - average_C) / float(iteration + 1)
            average_weights = average_weights + (weights - average_weights) / float(iteration + 1)
            average_gaps = torch.stack(
                [_inner_product(average_C, difference) for difference in differences]
            )
            average_lower = float(average_gaps.min())
            if average_lower > best_lower:
                best_lower, best_C = average_lower, average_C.clone()
            average_mixture = torch.zeros_like(reference)
            for weight, difference in zip(average_weights, differences):
                average_mixture = average_mixture + weight * difference
            average_upper = float(torch.linalg.eigvalsh(average_mixture).max())
            if average_upper < best_upper:
                best_upper, best_weights = average_upper, average_weights.clone()

    certified = selectability_bounds(
        scatter_matrices,
        target_pair,
        C=best_C,
        mixture_weights=best_weights,
        validation_tolerance=validation_tolerance,
        certificate_tolerance=certificate_tolerance,
    )
    return ApproximateSelectabilityResult(
        target=target_pair,
        competitors=competitors,
        iterations=iterations,
        metric_C=certified.metric_C,
        mixture_weights=certified.mixture_weights,
        lower_bound=certified.lower_bound,
        upper_bound=certified.upper_bound,
        certificate_gap=certified.upper_bound - certified.lower_bound,
        certifies_selectable=certified.certifies_selectable,
        certifies_not_selectable=certified.certifies_not_selectable,
    )


def r2_ranking_flip_example() -> Dict[str, object]:
    """Return a full-rank, exact-product r=2 raw-ranking flip.

    With the adjacent candidates ``p=(0,1)`` and ``q=(1,2)``, the original
    raw distances are ``1`` and ``2``.  For ``R=diag(1,4)``, they become
    ``1`` and ``1/2`` while
    all effective updates (and therefore all activation-weighted distances)
    remain unchanged.
    """

    dtype = torch.float64
    A = torch.eye(2, dtype=dtype)
    bases = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 2.0]],
        ],
        dtype=dtype,
    )
    pairs = ((0, 1), (1, 2))
    activation_covariance = torch.tensor([[2.0, 0.25], [0.25, 0.5]], dtype=dtype)
    before = audit_pair_distances(
        bases, A, pairs, activation_covariance=activation_covariance
    )
    R = torch.diag(torch.tensor([1.0, 4.0], dtype=dtype))
    gauge = verify_gl_gauge(bases, A, R, atol=1e-12, rtol=1e-12)
    after = audit_pair_distances(
        gauge.transformed_bases,
        gauge.transformed_A,
        pairs,
        activation_covariance=activation_covariance,
    )
    return {
        "A": A,
        "bases": bases,
        "R": R,
        "pairs": pairs,
        "before": before,
        "after": after,
        "gauge": gauge,
    }


def r2_unbounded_gauge_family(scales: Iterable[float]) -> List[UnboundedFamilyRecord]:
    """Evaluate the analytic r=2 family with unbounded structural regret.

    For each model scale ``t>1``, use ``B_0=(0,0)``, ``B_1=(1,0)``, and
    ``B_2=(1,t)`` with adjacent actions ``p=(0,1)`` and ``q=(1,2)``.  The
    identity gauge selects ``p`` from raw distances ``1`` and ``t``.  The
    counterfactual ``Q=diag(1,1/(2t))`` (equivalently, this module's
    ``R=Q^{-1}=diag(1,2t)``) leaves all products fixed but changes the raw
    distances to ``1`` and ``1/2``, selecting ``q``.  With ``A=I``, their
    invariant squared costs are ``1`` and ``t^2``.  Thus the approximation
    ratio is ``t^2`` and the additive regret is ``t^2-1``.

    The unbounded statement is across this *family of models*.  For any one
    fixed checkpoint with finitely many actions, its invariant action costs
    form a finite set along the gauge orbit.
    """

    dtype = torch.float64
    A = torch.eye(2, dtype=dtype)
    pairs = ((0, 1), (1, 2))
    records: List[UnboundedFamilyRecord] = []
    for raw_scale in scales:
        scale = float(raw_scale)
        if scale <= 1 or not math.isfinite(scale):
            raise ValueError("every model scale must be finite and greater than one")
        bases = torch.tensor(
            [
                [[0.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, scale]],
            ],
            dtype=dtype,
        )
        before = audit_pair_distances(bases, A, pairs)
        R = torch.diag(torch.tensor([1.0, 2.0 * scale], dtype=dtype))
        gauge = verify_gl_gauge(bases, A, R, atol=1e-11, rtol=1e-11)
        after = audit_pair_distances(gauge.transformed_bases, gauge.transformed_A, pairs)
        p, q = pairs
        selected_before = min(pairs, key=lambda pair: before.scores[pair].raw_distance)
        selected_after = min(pairs, key=lambda pair: after.scores[pair].raw_distance)
        cost_p = before.scores[p].effective_distance ** 2
        cost_q = before.scores[q].effective_distance ** 2
        records.append(
            UnboundedFamilyRecord(
                scale=scale,
                raw_distance_p_before=before.scores[p].raw_distance,
                raw_distance_q_before=before.scores[q].raw_distance,
                raw_distance_p_after=after.scores[p].raw_distance,
                raw_distance_q_after=after.scores[q].raw_distance,
                selected_pair_before=selected_before,
                selected_pair_after=selected_after,
                effective_distance_p=after.scores[p].effective_distance,
                effective_distance_q=after.scores[q].effective_distance,
                invariant_cost_p=cost_p,
                invariant_cost_q=cost_q,
                approximation_ratio=cost_q / cost_p,
                additive_regret=cost_q - cost_p,
                max_product_error=gauge.max_absolute_error,
            )
        )
    if not records:
        raise ValueError("scales must contain at least one value")
    return records
