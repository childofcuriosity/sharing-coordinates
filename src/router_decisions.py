"""Gauge-controlled structural decisions for the byte language model.

The search routine in this module receives router probabilities only.  It
cannot inspect task loss, logits, basis tensors, or evaluation batches.  Loss
is evaluated only after a partition-changing gauge has been fixed.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn import functional as F

from .models import BasisLinear, CausalBasisTransformer


@dataclass
class RouterDecisionConfig:
    temperature: float = 1.0
    batch_size: int = 4
    sequence_length: int = 32
    eval_batches: int = 4
    batch_seed: int = 610_000
    search_seed: int = 620_000
    search_trials: int = 2000
    max_condition: float = 30.0
    min_probability: float = 1e-8
    search_family: str = "positive_stochastic"
    gauge_selection: str = "first_valid"
    max_effective_relative_l2: float = 1e-5
    max_logit_abs_error: float = 1e-4
    max_batch_nll_error: float = 1e-5
    max_bpb_equivalence_error: float = 1e-5
    disable_tf32: bool = True
    batch_sampling: str = "nonoverlap"
    kmeans_seed: int = 630_000
    kmeans_restarts: int = 8
    kmeans_iterations: int = 50


class GaugeSearchFailure(RuntimeError):
    """Structured negative result from a finite, router-only gauge search."""

    def __init__(self, message: str, metadata: Mapping[str, object]):
        super().__init__(message)
        self.metadata = dict(metadata)


def _basis_modules(model: CausalBasisTransformer) -> Dict[str, BasisLinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, BasisLinear)
    }


def effective_basis_tensors(
    model: CausalBasisTransformer,
    probabilities: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Return every layer's effective weight and bias for every BasisLinear."""

    if probabilities is None:
        probabilities = model.router.probabilities(temperature, hard=False)
    outputs: Dict[str, torch.Tensor] = {}
    for name, module in _basis_modules(model).items():
        outputs[name + ".weight"] = torch.tensordot(
            probabilities,
            module.weight,
            dims=([1], [0]),
        )
        if module.bias is not None:
            outputs[name + ".bias"] = torch.tensordot(
                probabilities,
                module.bias,
                dims=([1], [0]),
            )
    return outputs


def compare_effective_tensors(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> Dict[str, float]:
    if set(reference) != set(candidate):
        raise ValueError("effective tensor collections have different keys")
    max_abs = 0.0
    squared_error = 0.0
    squared_reference = 0.0
    for name in reference:
        difference = candidate[name] - reference[name]
        if difference.numel():
            max_abs = max(max_abs, float(difference.detach().abs().max()))
        squared_error += float(difference.detach().double().square().sum())
        squared_reference += float(reference[name].detach().double().square().sum())
    return {
        "max_abs": max_abs,
        "relative_l2": math.sqrt(squared_error / max(squared_reference, 1e-300)),
        "squared_error": squared_error,
    }


def apply_common_router_gauge(
    model: CausalBasisTransformer,
    gauge: torch.Tensor,
    temperature: float = 1.0,
    min_probability: float = 0.0,
    in_place: bool = False,
) -> Tuple[CausalBasisTransformer, Dict[str, float]]:
    """Apply ``A'=A M, B'=M^{-1}B`` to all BasisLinear tensors."""

    transformed = model if in_place else copy.deepcopy(model)
    probabilities = transformed.router.probabilities(temperature, hard=False).detach()
    gauge = gauge.to(device=probabilities.device, dtype=probabilities.dtype)
    k = probabilities.shape[1]
    if tuple(gauge.shape) != (k, k):
        raise ValueError("gauge must have shape (%d, %d)" % (k, k))
    ones = torch.ones(k, device=gauge.device, dtype=gauge.dtype)
    row_sum_error = float((gauge @ ones - ones).abs().max())
    if row_sum_error > 5e-6:
        raise ValueError("gauge does not preserve row sums")
    condition = float(torch.linalg.cond(gauge.detach().double()))
    if not math.isfinite(condition):
        raise ValueError("gauge is singular")
    transformed_probabilities = probabilities @ gauge
    minimum = float(transformed_probabilities.min())
    if minimum <= min_probability:
        raise ValueError(
            "gauge leaves the strict simplex interior: min(A M)=%.3e" % minimum
        )

    with torch.no_grad():
        transformed.router.logits.copy_(
            temperature * transformed_probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        )
        for module in _basis_modules(transformed).values():
            flat_weight = module.weight.reshape(k, -1)
            module.weight.copy_(torch.linalg.solve(gauge, flat_weight).reshape_as(module.weight))
            if module.bias is not None:
                flat_bias = module.bias.reshape(k, -1)
                module.bias.copy_(torch.linalg.solve(gauge, flat_bias).reshape_as(module.bias))

    realized = transformed.router.probabilities(temperature, hard=False).detach()
    probability_error = float((realized - transformed_probabilities).abs().max())
    return transformed, {
        "condition_number": condition,
        "row_sum_error": row_sum_error,
        "minimum_transformed_probability": minimum,
        "router_realization_max_abs_error": probability_error,
    }


def _partition_disagreement(first: torch.Tensor, second: torch.Tensor) -> int:
    first_edges = first[:, None] == first[None, :]
    second_edges = second[:, None] == second[None, :]
    return int(torch.triu(first_edges != second_edges, diagonal=1).sum())


def search_partition_changing_gauge(
    probabilities: torch.Tensor,
    seed: int,
    trials: int,
    max_condition: float,
    min_probability: float = 1e-8,
    required_groups: Optional[int] = None,
    search_family: str = "mixed",
    selection_rule: str = "max_disagreement",
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Search using router partitions only, with deterministic RNG and bounds.

    ``mixed`` preserves the historical low-level search behavior.  The
    preregistered high-level audit passes ``positive_stochastic`` explicitly.
    """

    if trials <= 0:
        raise ValueError("trials must be positive")
    if search_family not in {"positive_stochastic", "signed_local", "mixed"}:
        raise ValueError("unknown search family: %s" % search_family)
    if selection_rule not in {"first_valid", "max_disagreement"}:
        raise ValueError("unknown gauge selection rule: %s" % selection_rule)
    source_device = probabilities.device
    source_dtype = probabilities.dtype
    alpha = probabilities.detach().to(device="cpu", dtype=torch.float64)
    depth, k = alpha.shape
    original = alpha.argmax(dim=1)
    original_groups = int(torch.unique(original).numel())
    target_groups = original_groups if required_groups is None else required_groups
    generator = torch.Generator(device="cpu").manual_seed(seed)
    identity = torch.eye(k, dtype=torch.float64)
    best: Optional[Tuple[int, float, int, torch.Tensor, torch.Tensor]] = None
    first_valid: Optional[Tuple[int, float, torch.Tensor, torch.Tensor]] = None
    counts = {
        "attempted": 0,
        "invertible": 0,
        "condition_pass": 0,
        "simplex_pass": 0,
        "group_pass": 0,
        "partition_change": 0,
    }

    for trial in range(trials):
        counts["attempted"] += 1
        use_positive = search_family == "positive_stochastic" or (
            search_family == "mixed" and trial % 2 == 0
        )
        if use_positive:
            raw = torch.rand((k, k), generator=generator, dtype=torch.float64) + 0.01
            stochastic = raw / raw.sum(dim=1, keepdim=True)
            strength = 0.02 + 0.98 * float(torch.rand((), generator=generator))
            gauge = (1.0 - strength) * identity + strength * stochastic
        else:
            direction = torch.randn((k, k), generator=generator, dtype=torch.float64)
            direction = direction - direction.mean(dim=1, keepdim=True)
            direction = direction / direction.norm().clamp_min(1e-12)
            log_scale = -3.5 + 4.0 * float(torch.rand((), generator=generator))
            gauge = identity + math.exp(log_scale) * direction

        row_error = float((gauge.sum(dim=1) - 1.0).abs().max())
        if row_error > 1e-10:
            continue
        condition = float(torch.linalg.cond(gauge))
        if not math.isfinite(condition):
            continue
        counts["invertible"] += 1
        if condition > max_condition:
            continue
        counts["condition_pass"] += 1
        transformed = alpha @ gauge
        if not torch.isfinite(transformed).all() or float(transformed.min()) <= min_probability:
            continue
        counts["simplex_pass"] += 1
        assignment = transformed.argmax(dim=1)
        groups = int(torch.unique(assignment).numel())
        if groups != target_groups:
            continue
        counts["group_pass"] += 1
        disagreement = _partition_disagreement(original, assignment)
        if disagreement <= 0:
            continue
        counts["partition_change"] += 1
        if first_valid is None:
            first_valid = (trial, condition, gauge.clone(), assignment.clone())
        candidate = (disagreement, -condition, -trial, gauge.clone(), assignment.clone())
        if best is None or candidate[:3] > best[:3]:
            best = candidate

    if best is None:
        message = (
            "no partition-changing gauge with %d groups and condition <= %.3g "
            "was found in %d router-only trials"
            % (target_groups, max_condition, trials)
        )
        raise GaugeSearchFailure(
            message,
            {
                "status": "no_partition_changing_gauge_found",
                "message": message,
                "seed": seed,
                "trials": trials,
                "search_family": search_family,
                "selection_rule": selection_rule,
                "counts": counts,
                "max_condition": max_condition,
                "min_probability": min_probability,
                "required_groups": target_groups,
                "original_groups": original_groups,
                "original_assignment": original.tolist(),
                "search_inputs": "router_probabilities_only",
            },
        )

    assert first_valid is not None
    if selection_rule == "first_valid":
        selected_trial, selected_condition, gauge, assignment = first_valid
        disagreement = _partition_disagreement(original, assignment)
    else:
        disagreement, negative_condition, negative_trial, gauge, assignment = best
        selected_trial = -negative_trial
        selected_condition = -negative_condition
    first_trial, first_condition, _, first_assignment = first_valid
    first_matrix = first_valid[2]
    return gauge.to(device=source_device, dtype=source_dtype), {
        "seed": seed,
        "trials": trials,
        "search_family": search_family,
        "selection_rule": selection_rule,
        "counts": counts,
        "first_valid_trial": first_trial,
        "first_valid_condition_number": first_condition,
        "first_valid_assignment": first_assignment.tolist(),
        "selected_trial": selected_trial,
        "condition_number": selected_condition,
        "max_condition": max_condition,
        "minimum_transformed_probability": float((alpha @ gauge).min()),
        "pairwise_partition_disagreements": disagreement,
        "original_assignment": original.tolist(),
        "transformed_assignment": assignment.tolist(),
        "original_groups": original_groups,
        "transformed_groups": int(torch.unique(assignment).numel()),
        "matrix": gauge.tolist(),
        "minimum_gauge_entry": float(gauge.min()),
        "search_inputs": "router_probabilities_only",
        "first_valid": {
            "trial": first_trial,
            "condition_number": first_condition,
            "assignment": first_assignment.tolist(),
            "matrix": first_matrix.tolist(),
        },
        "selected": {
            "trial": selected_trial,
            "condition_number": selected_condition,
            "assignment": assignment.tolist(),
            "matrix": gauge.tolist(),
        },
    }


def canonicalize_partition(assignments: torch.Tensor) -> torch.Tensor:
    values = assignments.detach().cpu().tolist()
    mapping: Dict[int, int] = {}
    canonical: List[int] = []
    for value in values:
        integer = int(value)
        if integer not in mapping:
            mapping[integer] = len(mapping)
        canonical.append(mapping[integer])
    return torch.tensor(canonical, dtype=torch.long, device=assignments.device)


def literal_argmax_effective_tensors(
    model: CausalBasisTransformer,
    temperature: float = 1.0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    probabilities = model.router.probabilities(temperature, hard=False)
    assignments = probabilities.argmax(dim=1)
    one_hot = F.one_hot(assignments, num_classes=model.num_bases).to(probabilities.dtype)
    return effective_basis_tensors(model, one_hot), assignments


def materialize_partition_folding(
    model: CausalBasisTransformer,
    common_effective: Mapping[str, torch.Tensor],
    assignments: torch.Tensor,
) -> Tuple[CausalBasisTransformer, Dict[str, object]]:
    """Refit group centroids from common effective tensors, then hard-route."""

    folded = copy.deepcopy(model)
    canonical = canonicalize_partition(assignments).to(folded.router.logits.device)
    groups = int(canonical.max()) + 1
    if groups > folded.num_bases:
        raise ValueError("partition uses more groups than the model basis budget")
    modules = _basis_modules(folded)
    with torch.no_grad():
        for name, module in modules.items():
            for field in ("weight", "bias"):
                parameter = getattr(module, field)
                if parameter is None:
                    continue
                values = common_effective[name + "." + field].to(parameter.device)
                parameter.zero_()
                for group in range(groups):
                    members = canonical == group
                    parameter[group].copy_(values[members].mean(dim=0))
        folded.router.logits.fill_(-20.0)
        folded.router.logits.scatter_(1, canonical[:, None], 20.0)

    folded_effective = effective_basis_tensors(
        folded,
        F.one_hot(canonical, num_classes=folded.num_bases).to(folded.router.logits.dtype),
    )
    error = compare_effective_tensors(common_effective, folded_effective)
    return folded, {
        "assignment": canonical.detach().cpu().tolist(),
        "num_groups": groups,
        "effective_fit": error,
    }


def _effective_matrix(common_effective: Mapping[str, torch.Tensor]) -> torch.Tensor:
    keys = sorted(common_effective)
    if not keys:
        raise ValueError("no effective tensors were provided")
    depth = common_effective[keys[0]].shape[0]
    parts = []
    for key in keys:
        value = common_effective[key]
        if value.shape[0] != depth:
            raise ValueError("effective tensors disagree on depth")
        parts.append(value.reshape(depth, -1))
    return torch.cat(parts, dim=1)


def _repair_empty_clusters(
    assignment: torch.Tensor,
    distances: torch.Tensor,
    num_clusters: int,
) -> torch.Tensor:
    """Deterministically keep the requested partition budget nonempty."""

    repaired = assignment.clone()
    counts = torch.bincount(repaired, minlength=num_clusters)
    losses = distances.gather(1, repaired[:, None]).squeeze(1)
    order = torch.argsort(losses, descending=True).detach().cpu().tolist()
    for empty in torch.nonzero(counts == 0, as_tuple=False).flatten().tolist():
        selected = None
        for index in order:
            source = int(repaired[index])
            if int(counts[source]) > 1:
                selected = index
                break
        if selected is None:
            raise RuntimeError("could not make all k-means clusters nonempty")
        source = int(repaired[selected])
        repaired[selected] = empty
        counts[source] -= 1
        counts[empty] += 1
    return repaired


def effective_theta_kmeans(
    common_effective: Mapping[str, torch.Tensor],
    num_clusters: int,
    seed: int,
    restarts: int = 8,
    iterations: int = 50,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """K-means on concatenated effective tensors with a fixed group budget."""

    matrix = _effective_matrix(common_effective).detach()
    depth = matrix.shape[0]
    if not 1 <= num_clusters <= depth:
        raise ValueError("num_clusters must be between one and depth")
    if num_clusters == 1:
        assignment = torch.zeros(depth, dtype=torch.long, device=matrix.device)
        centroid = matrix.mean(dim=0, keepdim=True)
        inertia = float((matrix - centroid).double().square().sum())
        return assignment, {"inertia": inertia, "restarts": restarts, "iterations": 0}

    generator = torch.Generator(device="cpu").manual_seed(seed)
    squared_norm = matrix.double().square().sum(dim=1, keepdim=True)
    best_assignment: Optional[torch.Tensor] = None
    best_inertia = float("inf")
    selected_iterations = 0
    for _ in range(max(1, restarts)):
        first = int(torch.randint(depth, (1,), generator=generator))
        chosen = [first]
        centroids = [matrix[first].clone()]
        while len(centroids) < num_clusters:
            center_matrix = torch.stack(centroids)
            distances = (
                squared_norm
                + center_matrix.double().square().sum(dim=1)[None]
                - 2.0 * matrix.double() @ center_matrix.double().t()
            ).clamp_min_(0.0)
            closest = distances.min(dim=1).values.detach().cpu()
            closest[chosen] = 0.0
            if float(closest.sum()) == 0.0:
                next_index = next(index for index in range(depth) if index not in chosen)
            else:
                next_index = int(torch.multinomial(closest, 1, generator=generator))
            chosen.append(next_index)
            centroids.append(matrix[next_index].clone())
        center_matrix = torch.stack(centroids)
        previous: Optional[torch.Tensor] = None
        for iteration in range(max(1, iterations)):
            distances = (
                squared_norm
                + center_matrix.double().square().sum(dim=1)[None]
                - 2.0 * matrix.double() @ center_matrix.double().t()
            ).clamp_min_(0.0)
            assignment = _repair_empty_clusters(
                distances.argmin(dim=1), distances, num_clusters
            )
            if previous is not None and torch.equal(assignment, previous):
                break
            previous = assignment.clone()
            for group in range(num_clusters):
                members = assignment == group
                if members.any():
                    center_matrix[group] = matrix[members].mean(dim=0)
                else:
                    farthest = int(distances.min(dim=1).values.argmax())
                    center_matrix[group] = matrix[farthest]
        restart_iterations = iteration + 1
        final_distances = (
            squared_norm
            + center_matrix.double().square().sum(dim=1)[None]
            - 2.0 * matrix.double() @ center_matrix.double().t()
        ).clamp_min_(0.0)
        assignment = _repair_empty_clusters(
            final_distances.argmin(dim=1), final_distances, num_clusters
        )
        # Empty-cluster repair can change an assignment after ``center_matrix``
        # was last updated.  Recompute the centroids and the actual objective of
        # the returned partition instead of reporting stale nearest-center loss.
        fitted_centers = torch.stack(
            [matrix[assignment == group].mean(dim=0) for group in range(num_clusters)]
        )
        residual = matrix - fitted_centers[assignment]
        inertia = float(residual.detach().double().square().sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_assignment = assignment.clone()
            selected_iterations = restart_iterations

    assert best_assignment is not None
    canonical = canonicalize_partition(best_assignment)
    return canonical, {
        "inertia": best_inertia,
        "restarts": max(1, restarts),
        "iterations": selected_iterations,
    }


def effective_theta_ward(
    common_effective: Mapping[str, torch.Tensor],
    num_clusters: int,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Deterministic Ward clustering on gauge-invariant effective tensors.

    The merge cost is ``|S||T|/(|S|+|T|) * ||mean(S)-mean(T)||^2``.  Ties are
    resolved lexicographically by cluster membership, so the result has no
    optimizer or initialization dependence.
    """

    matrix = _effective_matrix(common_effective).detach().double()
    depth = matrix.shape[0]
    if not 1 <= num_clusters <= depth:
        raise ValueError("num_clusters must be between one and depth")
    clusters: List[Tuple[int, ...]] = [(index,) for index in range(depth)]
    merge_costs: List[float] = []
    while len(clusters) > num_clusters:
        best_key: Optional[Tuple[float, Tuple[int, ...], Tuple[int, ...]]] = None
        best_pair: Optional[Tuple[int, int]] = None
        for first_index in range(len(clusters)):
            first = clusters[first_index]
            first_mean = matrix[list(first)].mean(dim=0)
            for second_index in range(first_index + 1, len(clusters)):
                second = clusters[second_index]
                second_mean = matrix[list(second)].mean(dim=0)
                scale = len(first) * len(second) / float(len(first) + len(second))
                cost = float(scale * (first_mean - second_mean).square().sum())
                key = (cost, first, second)
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (first_index, second_index)
        assert best_key is not None and best_pair is not None
        first_index, second_index = best_pair
        merged = tuple(sorted(clusters[first_index] + clusters[second_index]))
        clusters = [
            cluster
            for index, cluster in enumerate(clusters)
            if index not in {first_index, second_index}
        ]
        clusters.append(merged)
        clusters.sort()
        merge_costs.append(best_key[0])

    assignment = torch.empty(depth, dtype=torch.long, device=matrix.device)
    for group, members in enumerate(sorted(clusters)):
        assignment[list(members)] = group
    centroids = torch.stack(
        [matrix[assignment == group].mean(dim=0) for group in range(num_clusters)]
    )
    inertia = float((matrix - centroids[assignment]).square().sum())
    return assignment, {
        "method": "deterministic_ward_effective_theta",
        "inertia": inertia,
        "merge_costs": merge_costs,
        "num_clusters": num_clusters,
    }


def make_fixed_language_batches(
    stream: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    count: int,
    seed: int,
    sampling: str,
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], Dict[str, object]]:
    """Create reproducible evaluation batches and record their exact starts."""

    if batch_size <= 0 or sequence_length <= 0 or count <= 0:
        raise ValueError("batch_size, sequence_length, and count must be positive")
    if len(stream) <= sequence_length:
        raise ValueError("stream is too short for the requested sequence length")
    if sampling not in {"with_replacement", "nonoverlap"}:
        raise ValueError("unknown batch sampling scheme: %s" % sampling)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if sampling == "with_replacement":
        starts = torch.randint(
            0,
            len(stream) - sequence_length,
            (count, batch_size),
            generator=generator,
        )
    else:
        span = sequence_length + 1
        available = len(stream) // span
        required = count * batch_size
        if required > available:
            raise ValueError(
                "nonoverlap sampling needs %d disjoint blocks but only %d are available"
                % (required, available)
            )
        block_ids = torch.randperm(available, generator=generator)[:required]
        starts = (block_ids * span).reshape(count, batch_size)

    offsets = torch.arange(sequence_length, device=stream.device)
    batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for batch_starts in starts:
        device_starts = batch_starts.to(stream.device)
        inputs = stream[device_starts[:, None] + offsets[None]]
        targets = stream[device_starts[:, None] + offsets[None] + 1]
        batches.append((inputs, targets))
    return batches, {
        "sampling": sampling,
        "seed": seed,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "count": count,
        "starts": starts.tolist(),
    }


def fixed_language_batches(
    stream: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    count: int,
    seed: int,
    sampling: str = "with_replacement",
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Compatibility wrapper; historical callers sample with replacement."""

    batches, _ = make_fixed_language_batches(
        stream,
        batch_size,
        sequence_length,
        count,
        seed,
        sampling,
    )
    return batches


@torch.no_grad()
def evaluate_fixed_batches(
    model: CausalBasisTransformer,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    temperature: float,
    hard: bool,
) -> Dict[str, object]:
    was_training = model.training
    model.eval()
    batch_nll: List[float] = []
    batch_bpb: List[float] = []
    total_nll = 0.0
    total_tokens = 0
    for inputs, targets in batches:
        logits = model(inputs, temperature=temperature, hard=hard)
        nll_sum = F.cross_entropy(
            logits.reshape(-1, 256),
            targets.reshape(-1),
            reduction="sum",
        )
        tokens = targets.numel()
        mean_nll = float(nll_sum) / tokens
        batch_nll.append(mean_nll)
        batch_bpb.append(mean_nll / math.log(2.0))
        total_nll += float(nll_sum)
        total_tokens += tokens
    model.train(was_training)
    aggregate_nll = total_nll / total_tokens
    return {
        "batch_nll": batch_nll,
        "batch_bpb": batch_bpb,
        "mean_nll": aggregate_nll,
        "instantaneous_bpb": aggregate_nll / math.log(2.0),
        "tokens": total_tokens,
    }


@torch.no_grad()
def compare_fixed_batch_logits(
    reference: CausalBasisTransformer,
    candidate: CausalBasisTransformer,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    temperature: float,
) -> Dict[str, object]:
    reference_training = reference.training
    candidate_training = candidate.training
    reference.eval()
    candidate.eval()
    maximum = 0.0
    reference_nll: List[float] = []
    candidate_nll: List[float] = []
    tokens_per_batch: List[int] = []
    for inputs, targets in batches:
        first = reference(inputs, temperature=temperature, hard=False)
        second = candidate(inputs, temperature=temperature, hard=False)
        maximum = max(maximum, float((first - second).abs().max()))
        reference_nll.append(
            float(F.cross_entropy(first.reshape(-1, 256), targets.reshape(-1)))
        )
        candidate_nll.append(
            float(F.cross_entropy(second.reshape(-1, 256), targets.reshape(-1)))
        )
        tokens_per_batch.append(targets.numel())
    reference.train(reference_training)
    candidate.train(candidate_training)
    reference_total = sum(value * count for value, count in zip(reference_nll, tokens_per_batch))
    candidate_total = sum(value * count for value, count in zip(candidate_nll, tokens_per_batch))
    total_tokens = sum(tokens_per_batch)
    return {
        "max_abs_logit_error": maximum,
        "reference_batch_nll": reference_nll,
        "candidate_batch_nll": candidate_nll,
        "max_abs_batch_nll_error": max(
            abs(first - second) for first, second in zip(reference_nll, candidate_nll)
        ),
        "reference_instantaneous_bpb": reference_total / total_tokens / math.log(2.0),
        "candidate_instantaneous_bpb": candidate_total / total_tokens / math.log(2.0),
    }


def _group_sizes(assignments: torch.Tensor) -> List[int]:
    canonical = canonicalize_partition(assignments)
    groups = int(canonical.max()) + 1
    return [int((canonical == group).sum()) for group in range(groups)]


def _assert_exact_equivalence(
    effective: Mapping[str, float],
    logits: Mapping[str, object],
    config: RouterDecisionConfig,
) -> Dict[str, object]:
    bpb_error = abs(
        float(logits["reference_instantaneous_bpb"])
        - float(logits["candidate_instantaneous_bpb"])
    )
    checks = {
        "effective_relative_l2": (
            float(effective["relative_l2"]),
            config.max_effective_relative_l2,
        ),
        "max_abs_logit_error": (
            float(logits["max_abs_logit_error"]),
            config.max_logit_abs_error,
        ),
        "max_abs_batch_nll_error": (
            float(logits["max_abs_batch_nll_error"]),
            config.max_batch_nll_error,
        ),
        "absolute_bpb_error": (bpb_error, config.max_bpb_equivalence_error),
    }
    failures = []
    for name, (value, threshold) in checks.items():
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("equivalence threshold %s must be finite and nonnegative" % name)
        if not math.isfinite(value) or value > threshold:
            failures.append("%s=%.6g > %.6g" % (name, value, threshold))
    if failures:
        raise ValueError("exact-equivalence audit failed: " + "; ".join(failures))
    return {
        "passed": True,
        "checks": {
            name: {"value": value, "threshold": threshold}
            for name, (value, threshold) in checks.items()
        },
    }


@torch.no_grad()
def _run_router_decision_audit_impl(
    model: CausalBasisTransformer,
    stream: torch.Tensor,
    config: RouterDecisionConfig,
) -> Dict[str, object]:
    if config.sequence_length > model.max_length:
        raise ValueError("audit sequence length exceeds checkpoint maximum")
    probabilities = model.router.probabilities(config.temperature, hard=False).detach()
    gauge, search = search_partition_changing_gauge(
        probabilities,
        seed=config.search_seed,
        trials=config.search_trials,
        max_condition=config.max_condition,
        min_probability=config.min_probability,
        search_family=config.search_family,
        selection_rule=config.gauge_selection,
    )
    gauged, gauge_metadata = apply_common_router_gauge(
        model,
        gauge,
        temperature=config.temperature,
        min_probability=config.min_probability,
    )
    common_effective = effective_basis_tensors(model, probabilities)
    gauged_effective = effective_basis_tensors(gauged, temperature=config.temperature)
    effective_equivalence = compare_effective_tensors(common_effective, gauged_effective)
    batches, batch_metadata = make_fixed_language_batches(
        stream,
        config.batch_size,
        config.sequence_length,
        config.eval_batches,
        config.batch_seed,
        config.batch_sampling,
    )
    logit_equivalence = compare_fixed_batch_logits(
        model,
        gauged,
        batches,
        config.temperature,
    )
    equivalence_audit = _assert_exact_equivalence(
        effective_equivalence,
        logit_equivalence,
        config,
    )

    original_literal, original_assignment = literal_argmax_effective_tensors(
        model, config.temperature
    )
    gauged_literal, gauged_assignment = literal_argmax_effective_tensors(
        gauged, config.temperature
    )
    original_groups = int(torch.unique(original_assignment).numel())
    gauged_groups = int(torch.unique(gauged_assignment).numel())
    if original_groups != gauged_groups:
        raise AssertionError("router-only search did not preserve the group budget")

    original_fold, original_fold_metadata = materialize_partition_folding(
        model, common_effective, original_assignment
    )
    gauged_fold, gauged_fold_metadata = materialize_partition_folding(
        model, common_effective, gauged_assignment
    )
    kmeans_assignment, kmeans_metadata = effective_theta_kmeans(
        common_effective,
        original_groups,
        seed=config.kmeans_seed,
        restarts=config.kmeans_restarts,
        iterations=config.kmeans_iterations,
    )
    kmeans_fold, kmeans_fold_metadata = materialize_partition_folding(
        model, common_effective, kmeans_assignment
    )
    ward_assignment, ward_metadata = effective_theta_ward(
        common_effective,
        original_groups,
    )
    ward_fold, ward_fold_metadata = materialize_partition_folding(
        model, common_effective, ward_assignment
    )

    original_literal_metrics = evaluate_fixed_batches(
        model, batches, config.temperature, hard=True
    )
    gauged_literal_metrics = evaluate_fixed_batches(
        gauged, batches, config.temperature, hard=True
    )
    original_fold_metrics = evaluate_fixed_batches(
        original_fold, batches, config.temperature, hard=True
    )
    gauged_fold_metrics = evaluate_fixed_batches(
        gauged_fold, batches, config.temperature, hard=True
    )
    kmeans_fold_metrics = evaluate_fixed_batches(
        kmeans_fold, batches, config.temperature, hard=True
    )
    ward_fold_metrics = evaluate_fixed_batches(
        ward_fold, batches, config.temperature, hard=True
    )

    decisions = {
        "literal_argmax_original": {
            "assignment": original_assignment.detach().cpu().tolist(),
            "num_groups": original_groups,
            "effective_fit": compare_effective_tensors(common_effective, original_literal),
            "causal_role": "secondary_coordinate_and_partition_confounding",
            "common_effective_partition_refit": False,
            **original_literal_metrics,
        },
        "literal_argmax_gauged": {
            "assignment": gauged_assignment.detach().cpu().tolist(),
            "num_groups": gauged_groups,
            "effective_fit": compare_effective_tensors(common_effective, gauged_literal),
            "causal_role": "secondary_coordinate_and_partition_confounding",
            "common_effective_partition_refit": False,
            **gauged_literal_metrics,
        },
        "partition_fold_original": {
            **original_fold_metadata,
            "causal_role": "primary_common_effective_partition_fold",
            "common_effective_partition_refit": True,
            **original_fold_metrics,
        },
        "partition_fold_gauged": {
            **gauged_fold_metadata,
            "causal_role": "primary_common_effective_partition_fold",
            "common_effective_partition_refit": True,
            **gauged_fold_metrics,
        },
        "effective_theta_kmeans": {
            **kmeans_fold_metadata,
            "kmeans": kmeans_metadata,
            "causal_role": "gauge_invariant_same_budget_baseline",
            "common_effective_partition_refit": True,
            **kmeans_fold_metrics,
        },
        "effective_theta_ward": {
            **ward_fold_metadata,
            "ward": ward_metadata,
            "causal_role": "gauge_invariant_same_budget_baseline",
            "common_effective_partition_refit": True,
            **ward_fold_metrics,
        },
    }
    batch_nll_delta = [
        gauged - original
        for original, gauged in zip(
            original_fold_metrics["batch_nll"],
            gauged_fold_metrics["batch_nll"],
        )
    ]
    batch_bpb_delta = [value / math.log(2.0) for value in batch_nll_delta]
    total_pairs = model.depth * (model.depth - 1) // 2
    pairwise_disagreements = _partition_disagreement(
        original_assignment,
        gauged_assignment,
    )
    primary_contrast = {
        "name": "common_effective_partition_folds",
        "reference_decision": "partition_fold_original",
        "candidate_decision": "partition_fold_gauged",
        "isolated_factor": "router_induced_partition",
        "common_effective_tensors": True,
        "same_group_budget": original_groups,
        "reference_group_sizes": _group_sizes(original_assignment),
        "candidate_group_sizes": _group_sizes(gauged_assignment),
        "pairwise_partition_disagreements": pairwise_disagreements,
        "normalized_pairwise_partition_disagreement": (
            pairwise_disagreements / total_pairs if total_pairs else 0.0
        ),
        "batch_nll_delta_candidate_minus_reference": batch_nll_delta,
        "batch_bpb_delta_candidate_minus_reference": batch_bpb_delta,
        "mean_nll_delta_candidate_minus_reference": (
            float(gauged_fold_metrics["mean_nll"])
            - float(original_fold_metrics["mean_nll"])
        ),
        "instantaneous_bpb_delta_candidate_minus_reference": (
            float(gauged_fold_metrics["instantaneous_bpb"])
            - float(original_fold_metrics["instantaneous_bpb"])
        ),
    }
    primary_contrast["absolute_instantaneous_bpb_delta"] = abs(
        float(primary_contrast["instantaneous_bpb_delta_candidate_minus_reference"])
    )
    return {
        "config": asdict(config),
        "gauge_search": search,
        "gauge": gauge_metadata,
        "equivalence": {
            **equivalence_audit,
            "effective_tensors": effective_equivalence,
            "fixed_batch_logits": logit_equivalence,
        },
        "evaluation_batches": batch_metadata,
        "assignments": {
            "original": original_assignment.detach().cpu().tolist(),
            "gauged": gauged_assignment.detach().cpu().tolist(),
            "effective_theta_kmeans": kmeans_assignment.detach().cpu().tolist(),
            "effective_theta_ward": ward_assignment.detach().cpu().tolist(),
        },
        "group_budget": original_groups,
        "primary_causal_contrast": primary_contrast,
        "literal_hardening_warning": (
            "literal original-versus-gauged hardening changes both the router "
            "partition and inverse-gauged basis coordinates; it is not the primary "
            "partition-only causal contrast"
        ),
        "decisions": decisions,
    }


def run_router_decision_audit(
    model: CausalBasisTransformer,
    stream: torch.Tensor,
    config: RouterDecisionConfig,
) -> Dict[str, object]:
    """Run the equivalence -> partition -> decision -> outcome chain.

    CUDA TF32 flags are process-global, so they are saved and restored even if
    a gauge or equivalence check fails.
    """

    previous_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)
    try:
        if config.disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        result = _run_router_decision_audit_impl(model, stream, config)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
    result["numeric_controls"] = {
        "tf32_disabled_during_audit": config.disable_tf32,
        "matmul_allow_tf32_during_audit": (
            False if config.disable_tf32 else previous_matmul_tf32
        ),
        "cudnn_allow_tf32_during_audit": (
            False if config.disable_tf32 else previous_cudnn_tf32
        ),
        "matmul_allow_tf32_prior_and_restored": previous_matmul_tf32,
        "cudnn_allow_tf32_prior_and_restored": previous_cudnn_tf32,
    }
    return result
