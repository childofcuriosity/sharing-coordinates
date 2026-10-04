"""Faithful first-merge witnesses for ASLoRA-style cross-layer sharing.

The tensor-level utilities in this module have no dependency on Hugging Face
Transformers.  The optional RoBERTa adapter works by inspecting the standard
``RobertaForSequenceClassification`` module layout and therefore also imports
without Transformers being installed.

The implementation deliberately isolates *one merge decision at a saved
checkpoint*.  ASLoRA continues ordinary optimization between merges, and an
ordinary optimizer need not be equivariant under a LoRA change of basis.  A
single-snapshot witness can consequently make an exact product-preservation
claim without silently assuming equivariant training dynamics.
"""

from __future__ import annotations

import contextlib
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F


Pair = Tuple[int, int]


ASLORA_SOURCE_NOTES: Dict[str, str] = {
    "factorization": (
        "ASLoRA arXiv:2412.10135v2 Eq. (2) uses h_i = W_i x + B_i A x: A is shared across layers "
        "and B_i is layer specific before merging."
    ),
    "running_average": (
        "ASLoRA arXiv:2412.10135v2 Eq. (3) defines the similarity input as the cumulative average "
        "of each B_i over all observed training steps; it does not specify "
        "whether a step is sampled immediately before or after the optimizer update."
    ),
    "candidate_ambiguity": (
        "ASLoRA arXiv:2412.10135v2 Algorithm 1 says to calculate and sort all S_{i,j}, while the prose "
        "immediately below it says pairwise similarity between adjacent layers. "
        "Both 'all' and 'adjacent' are therefore exposed explicitly."
    ),
    "similarity_wording": (
        "ASLoRA arXiv:2412.10135v2 Equation (4) makes smaller L2 distance mean higher similarity and "
        "Algorithm 1 selects the minimum distance.  Later prose says 'lowest "
        "similarity'; this implementation follows the equation and algorithm."
    ),
    "merge_direction": (
        "The available ASLoRA arXiv:2412.10135v2 states that, when a pair is merged, the lower layer uses the "
        "B matrix of the upper layer.  Layer indices here increase from lower to upper."
    ),
    "qv_scope_ambiguity": (
        "The available ASLoRA arXiv:2412.10135v2 experiments adapt W_Q and W_V, but the text does not unambiguously "
        "state whether one A and one merge graph are shared jointly across the Q/V "
        "families.  The adapter defaults to one cross-layer A per projection and "
        "supports a joint layer-pair decision as an explicit option."
    ),
}


def _validate_factor_shapes(
    shared_a: torch.Tensor,
    layer_b: torch.Tensor,
    running_b: Optional[torch.Tensor] = None,
) -> None:
    if shared_a.ndim != 2:
        raise ValueError("shared_a must have shape [rank, in_features]")
    if layer_b.ndim != 3:
        raise ValueError("layer_b must have shape [layers, out_features, rank]")
    if layer_b.shape[-1] != shared_a.shape[0]:
        raise ValueError("the latent rank of layer_b and shared_a must agree")
    if running_b is not None and running_b.shape != layer_b.shape:
        raise ValueError("running_b must have the same shape as layer_b")
    if layer_b.device != shared_a.device:
        raise ValueError("shared_a and layer_b must be on the same device")
    if layer_b.dtype != shared_a.dtype:
        raise ValueError("shared_a and layer_b must have the same dtype")


def candidate_pairs(num_layers: int, mode: str = "all") -> List[Pair]:
    """Return deterministic first-merge candidates.

    ``all`` follows Algorithm 1 literally.  ``adjacent`` follows the prose below
    the algorithm.  Pairs are always ordered ``(lower, upper)``.
    """

    if num_layers < 2:
        raise ValueError("at least two layers are required for a merge")
    if mode == "adjacent":
        return [(index, index + 1) for index in range(num_layers - 1)]
    if mode == "all":
        return [
            (lower, upper)
            for lower in range(num_layers)
            for upper in range(lower + 1, num_layers)
        ]
    raise ValueError("candidate mode must be 'all' or 'adjacent'")


def update_running_average(
    previous_average: torch.Tensor,
    current_value: torch.Tensor,
    previous_count: int,
) -> Tuple[torch.Tensor, int]:
    """Append one observation to the exact cumulative average from ASLoRA Eq. (3)."""

    if previous_count < 0:
        raise ValueError("previous_count must be non-negative")
    if previous_average.shape != current_value.shape:
        raise ValueError("previous_average and current_value must have the same shape")
    next_count = previous_count + 1
    if previous_count == 0:
        return current_value.detach().clone(), next_count
    updated = previous_average + (current_value.detach() - previous_average) / float(next_count)
    return updated, next_count


def effective_updates(shared_a: torch.Tensor, layer_b: torch.Tensor) -> torch.Tensor:
    """Return the dense effective LoRA updates ``B_i A`` for every layer."""

    _validate_factor_shapes(shared_a, layer_b)
    return torch.matmul(layer_b, shared_a)


def raw_pair_distances(
    running_b: torch.Tensor,
    pairs: Optional[Sequence[Pair]] = None,
    mode: str = "all",
    squared: bool = False,
) -> Dict[Pair, torch.Tensor]:
    """ASLoRA's raw running-average B distances at one decision snapshot."""

    if running_b.ndim != 3:
        raise ValueError("running_b must have shape [layers, out_features, rank]")
    chosen_pairs = list(pairs) if pairs is not None else candidate_pairs(running_b.shape[0], mode)
    result: Dict[Pair, torch.Tensor] = {}
    for lower, upper in chosen_pairs:
        _validate_pair((lower, upper), running_b.shape[0])
        distance_sq = (running_b[lower] - running_b[upper]).square().sum()
        result[(lower, upper)] = distance_sq if squared else distance_sq.sqrt()
    return result


def effective_pair_distances(
    shared_a: torch.Tensor,
    running_b: torch.Tensor,
    pairs: Optional[Sequence[Pair]] = None,
    mode: str = "all",
    squared: bool = False,
) -> Dict[Pair, torch.Tensor]:
    """Gauge-invariant distances ``||(bar B_i-bar B_j) A||_F``."""

    _validate_factor_shapes(shared_a, running_b)
    chosen_pairs = list(pairs) if pairs is not None else candidate_pairs(running_b.shape[0], mode)
    result: Dict[Pair, torch.Tensor] = {}
    for lower, upper in chosen_pairs:
        _validate_pair((lower, upper), running_b.shape[0])
        difference = torch.matmul(running_b[lower] - running_b[upper], shared_a)
        distance_sq = difference.square().sum()
        result[(lower, upper)] = distance_sq if squared else distance_sq.sqrt()
    return result


def aggregate_pair_distances(
    targets: Mapping[str, "FactorSnapshot"],
    mode: str = "all",
    invariant: bool = False,
) -> Dict[Pair, torch.Tensor]:
    """Aggregate WQ/WV distances with a root-sum-of-squares convention.

    This is an explicit implementation choice for a joint Q/V decision, not a
    claim that the ASLoRA paper uniquely specifies this aggregation.
    """

    if not targets:
        raise ValueError("at least one target is required")
    layer_counts = {snapshot.layer_b.shape[0] for snapshot in targets.values()}
    if len(layer_counts) != 1:
        raise ValueError("all targets must have the same number of layers")
    pairs = candidate_pairs(next(iter(layer_counts)), mode)
    totals = {pair: None for pair in pairs}  # type: Dict[Pair, Optional[torch.Tensor]]
    for snapshot in targets.values():
        distances = (
            effective_pair_distances(snapshot.shared_a, snapshot.running_b, pairs=pairs, squared=True)
            if invariant
            else raw_pair_distances(snapshot.running_b, pairs=pairs, squared=True)
        )
        for pair, value in distances.items():
            totals[pair] = value if totals[pair] is None else totals[pair] + value
    return {pair: value.sqrt() for pair, value in totals.items() if value is not None}


def select_minimum_pair(distances: Mapping[Pair, torch.Tensor]) -> Pair:
    """Select the minimum-distance pair with lexicographic tie breaking."""

    if not distances:
        raise ValueError("at least one candidate distance is required")
    return min(distances, key=lambda pair: (float(distances[pair].detach().cpu()), pair))


def _validate_pair(pair: Pair, num_layers: int) -> None:
    lower, upper = pair
    if not (0 <= lower < upper < num_layers):
        raise ValueError("merge pair must satisfy 0 <= lower < upper < num_layers")


def merge_lower_uses_upper(layer_b: torch.Tensor, pair: Pair) -> torch.Tensor:
    """Return a tensor view of the paper's lower-uses-upper merge action."""

    if layer_b.ndim != 3:
        raise ValueError("layer_b must have shape [layers, out_features, rank]")
    _validate_pair(pair, layer_b.shape[0])
    lower, upper = pair
    merged = layer_b.clone()
    merged[lower] = merged[upper]
    return merged


def apply_common_gl_gauge(
    shared_a: torch.Tensor,
    layer_b: torch.Tensor,
    running_b: torch.Tensor,
    gauge: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply ``A' = Q^{-1}A`` and ``B_i' = B_i Q`` at a fixed snapshot.

    Transforming the cumulative average as ``bar B_i' = bar B_i Q`` is exact
    when the same fixed gauge is applied to every historical B observation.
    """

    _validate_factor_shapes(shared_a, layer_b, running_b)
    rank = shared_a.shape[0]
    if gauge.shape != (rank, rank):
        raise ValueError("gauge must be a square matrix of size rank")
    gauge = gauge.to(device=shared_a.device, dtype=shared_a.dtype)
    try:
        transformed_a = torch.linalg.solve(gauge, shared_a)
    except RuntimeError as error:
        raise ValueError("gauge must be invertible") from error
    transformed_b = torch.matmul(layer_b, gauge)
    transformed_running = torch.matmul(running_b, gauge)
    return transformed_a, transformed_b, transformed_running


def gauge_condition_number(gauge: torch.Tensor) -> float:
    singular_values = torch.linalg.svdvals(gauge)
    smallest = float(singular_values.min().detach().cpu())
    if smallest == 0.0:
        return math.inf
    return float((singular_values.max() / singular_values.min()).detach().cpu())


def is_orthogonal_gauge(gauge: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-6) -> bool:
    if gauge.ndim != 2 or gauge.shape[0] != gauge.shape[1]:
        return False
    identity = torch.eye(gauge.shape[0], dtype=gauge.dtype, device=gauge.device)
    return bool(torch.allclose(gauge.transpose(0, 1) @ gauge, identity, atol=atol, rtol=rtol))


def make_orthogonal_gauge(
    rank: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if rank < 1:
        raise ValueError("rank must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sample = torch.randn(rank, rank, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(sample)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    q = q @ torch.diag(signs)
    return q.to(device=device) if device is not None else q


@dataclass
class FactorSnapshot:
    """One projection family's pre-merge ASLoRA factors."""

    shared_a: torch.Tensor
    layer_b: torch.Tensor
    running_b: torch.Tensor
    running_count: int
    target_name: str = "query"

    def __post_init__(self) -> None:
        _validate_factor_shapes(self.shared_a, self.layer_b, self.running_b)
        if self.running_count < 0:
            raise ValueError("running_count must be non-negative")

    def detached_cpu_clone(self) -> "FactorSnapshot":
        return FactorSnapshot(
            shared_a=self.shared_a.detach().cpu().clone(),
            layer_b=self.layer_b.detach().cpu().clone(),
            running_b=self.running_b.detach().cpu().clone(),
            running_count=self.running_count,
            target_name=self.target_name,
        )

    def gauged(self, gauge: torch.Tensor) -> "FactorSnapshot":
        a, b, running = apply_common_gl_gauge(
            self.shared_a, self.layer_b, self.running_b, gauge
        )
        return FactorSnapshot(a, b, running, self.running_count, self.target_name)


@dataclass
class FirstMergeSnapshot:
    """Serializable whole-model state immediately before the first hard merge."""

    targets: Dict[str, FactorSnapshot]
    candidate_mode: str = "all"
    decision_scope: str = "per_projection"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_mode not in {"all", "adjacent"}:
            raise ValueError("candidate_mode must be 'all' or 'adjacent'")
        if self.decision_scope not in {"per_projection", "joint"}:
            raise ValueError("decision_scope must be 'per_projection' or 'joint'")
        if not self.targets:
            raise ValueError("snapshot must contain at least one target")

    def detached_cpu_clone(self) -> "FirstMergeSnapshot":
        return FirstMergeSnapshot(
            targets={name: state.detached_cpu_clone() for name, state in self.targets.items()},
            candidate_mode=self.candidate_mode,
            decision_scope=self.decision_scope,
            metadata=copy.deepcopy(self.metadata),
        )


def save_first_merge_snapshot(snapshot: FirstMergeSnapshot, path: Path) -> None:
    """Save a versioned, CPU-only snapshot before any merge mutation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    clean = snapshot.detached_cpu_clone()
    payload = {
        "format": "aslora-first-merge-v1",
        "candidate_mode": clean.candidate_mode,
        "decision_scope": clean.decision_scope,
        "metadata": clean.metadata,
        "source_notes": dict(ASLORA_SOURCE_NOTES),
        "targets": {
            name: {
                "shared_a": state.shared_a,
                "layer_b": state.layer_b,
                "running_b": state.running_b,
                "running_count": state.running_count,
                "target_name": state.target_name,
            }
            for name, state in clean.targets.items()
        },
    }
    torch.save(payload, destination)


def load_first_merge_snapshot(path: Path) -> FirstMergeSnapshot:
    payload = torch.load(Path(path), map_location="cpu")
    if payload.get("format") != "aslora-first-merge-v1":
        raise ValueError("unrecognized ASLoRA snapshot format")
    targets = {
        name: FactorSnapshot(
            shared_a=state["shared_a"],
            layer_b=state["layer_b"],
            running_b=state["running_b"],
            running_count=int(state["running_count"]),
            target_name=state.get("target_name", name),
        )
        for name, state in payload["targets"].items()
    }
    metadata = dict(payload.get("metadata", {}))
    metadata.setdefault("source_notes", payload.get("source_notes", dict(ASLORA_SOURCE_NOTES)))
    return FirstMergeSnapshot(
        targets=targets,
        candidate_mode=payload["candidate_mode"],
        decision_scope=payload["decision_scope"],
        metadata=metadata,
    )


def snapshot_distances(
    snapshot: FirstMergeSnapshot,
    invariant: bool = False,
) -> Dict[str, Dict[Pair, torch.Tensor]]:
    """Compute per-projection or joint distances according to snapshot config."""

    if snapshot.decision_scope == "joint":
        return {
            "joint": aggregate_pair_distances(
                snapshot.targets, snapshot.candidate_mode, invariant=invariant
            )
        }
    result: Dict[str, Dict[Pair, torch.Tensor]] = {}
    for name, state in snapshot.targets.items():
        result[name] = (
            effective_pair_distances(
                state.shared_a, state.running_b, mode=snapshot.candidate_mode
            )
            if invariant
            else raw_pair_distances(state.running_b, mode=snapshot.candidate_mode)
        )
    return result


def snapshot_selected_pairs(snapshot: FirstMergeSnapshot, invariant: bool = False) -> Dict[str, Pair]:
    return {
        name: select_minimum_pair(distances)
        for name, distances in snapshot_distances(snapshot, invariant=invariant).items()
    }


def apply_snapshot_gauges(
    snapshot: FirstMergeSnapshot,
    gauges: Mapping[str, torch.Tensor],
) -> FirstMergeSnapshot:
    """Return a new snapshot after fixed per-projection common GL gauges."""

    unknown = set(gauges) - set(snapshot.targets)
    if unknown:
        raise ValueError("gauges supplied for unknown targets: %s" % sorted(unknown))
    targets = {
        name: state.gauged(gauges[name]) if name in gauges else state.detached_cpu_clone()
        for name, state in snapshot.targets.items()
    }
    metadata = copy.deepcopy(snapshot.metadata)
    metadata["gauge_condition_numbers"] = {
        name: gauge_condition_number(gauge) for name, gauge in gauges.items()
    }
    return FirstMergeSnapshot(
        targets=targets,
        candidate_mode=snapshot.candidate_mode,
        decision_scope=snapshot.decision_scope,
        metadata=metadata,
    )


def evaluate_candidate_merges(
    state: FactorSnapshot,
    evaluator: Callable[[Pair, torch.Tensor, torch.Tensor], Mapping[str, float]],
    mode: str = "all",
) -> List[Dict[str, Any]]:
    """Evaluate every immediate lower-uses-upper counterfactual.

    ``evaluator`` receives ``(pair, shared_a, merged_layer_b)``.  This hook can
    compute tensor losses without Transformers, or be connected to a model-level
    evaluation loop by the caller.
    """

    raw = raw_pair_distances(state.running_b, mode=mode)
    invariant = effective_pair_distances(state.shared_a, state.running_b, mode=mode)
    records: List[Dict[str, Any]] = []
    for pair in candidate_pairs(state.layer_b.shape[0], mode):
        merged_b = merge_lower_uses_upper(state.layer_b, pair)
        metrics = dict(evaluator(pair, state.shared_a, merged_b))
        records.append(
            {
                "pair": pair,
                "raw_running_distance": float(raw[pair].detach().cpu()),
                "invariant_effective_distance": float(invariant[pair].detach().cpu()),
                "metrics": metrics,
            }
        )
    return records


def search_ranking_flip_gauge(
    state: FactorSnapshot,
    mode: str = "all",
    max_condition: float = 100.0,
    trials: int = 512,
    seed: int = 0,
) -> Optional[torch.Tensor]:
    """Search bounded random SPD-shaped gauges for a raw nearest-pair flip.

    Failure to find a flip is not a certificate of stability.  Exact stability
    can instead be audited through the pair Gram matrices and Loewner order.
    """

    if max_condition < 1.0:
        raise ValueError("max_condition must be at least one")
    if trials < 1:
        raise ValueError("trials must be positive")
    original = select_minimum_pair(raw_pair_distances(state.running_b, mode=mode))
    rank = state.shared_a.shape[0]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    log_bound = math.log(max_condition)
    for _ in range(trials):
        sample = torch.randn(rank, rank, generator=generator, dtype=torch.float64)
        rotation, _ = torch.linalg.qr(sample)
        log_scales = torch.empty(rank, dtype=torch.float64).uniform_(
            -0.5 * log_bound, 0.5 * log_bound, generator=generator
        )
        log_scales -= log_scales.mean()
        gauge = rotation @ torch.diag(log_scales.exp()) @ rotation.transpose(0, 1)
        gauge = gauge.to(device=state.shared_a.device, dtype=state.shared_a.dtype)
        transformed = state.gauged(gauge)
        selected = select_minimum_pair(raw_pair_distances(transformed.running_b, mode=mode))
        if selected != original:
            return gauge
    return None


def make_three_layer_flip_snapshot(
    epsilon: float = 0.2,
    delta: float = 0.05,
    dtype: torch.dtype = torch.float64,
) -> Tuple[FirstMergeSnapshot, torch.Tensor]:
    """Return a deterministic adjacent-pair witness used by tests and the CLI."""

    if not (0.0 < delta < epsilon < 1.0):
        raise ValueError("the witness requires 0 < delta < epsilon < 1")
    shared_a = torch.eye(2, dtype=dtype)
    layer_b = torch.tensor(
        [[[0.0, 0.0]], [[epsilon, 0.0]], [[epsilon, 1.0]]], dtype=dtype
    )
    state = FactorSnapshot(
        shared_a=shared_a,
        layer_b=layer_b,
        running_b=layer_b.clone(),
        running_count=1,
        target_name="query",
    )
    snapshot = FirstMergeSnapshot(
        targets={"query": state},
        candidate_mode="adjacent",
        decision_scope="per_projection",
        metadata={
            "kind": "deterministic-three-layer-witness",
            "epsilon": epsilon,
            "delta": delta,
            "source_notes": dict(ASLORA_SOURCE_NOTES),
        },
    )
    return snapshot, torch.diag(torch.tensor([1.0, delta], dtype=dtype))


class SharedALoRALinear(nn.Module):
    """A frozen linear layer plus a shared-A, layer-specific-B LoRA update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        shared_a: nn.Parameter,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("ASLoRA currently expects torch.nn.Linear query/value modules")
        if shared_a.shape != (rank, base_layer.in_features):
            raise ValueError("shared A has an incompatible shape")
        self.base_layer = base_layer
        self.shared_a = shared_a
        self.b = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(rank)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        low_rank = F.linear(F.linear(inputs, self.shared_a), self.b)
        return base + self.scaling * low_rank

    def effective_update(self) -> torch.Tensor:
        return self.scaling * (self.b @ self.shared_a)


class RobertaASLoRAHandle:
    """Training/evaluation handle for a patched RoBERTa classifier.

    Call :meth:`update_running_averages` once after each chosen optimizer step.
    Immediately before the first merge, call :meth:`capture_first_merge_snapshot`
    and save it.  Model-level candidate evaluation is non-destructive and uses
    the exact lower-uses-upper parameter tying rule.
    """

    def __init__(
        self,
        model: nn.Module,
        modules: Mapping[str, Sequence[SharedALoRALinear]],
        a_scope: str,
        alpha: float,
        a_init_std: float,
    ) -> None:
        self.model = model
        self.modules = {name: list(values) for name, values in modules.items()}
        self.a_scope = a_scope
        self.alpha = alpha
        self.a_init_std = a_init_std
        self.running_b = {
            name: torch.stack([module.b.detach().clone() for module in values])
            for name, values in self.modules.items()
        }
        self.running_count = {name: 0 for name in self.modules}

    def align_running_state_to_parameters(self) -> None:
        """Move non-parameter running averages after ``model.to(...)`` calls."""

        for name, modules in self.modules.items():
            reference = modules[0].b
            self.running_b[name] = self.running_b[name].to(
                device=reference.device, dtype=reference.dtype
            )

    def update_running_averages(self) -> None:
        """Record the current B tensors as one Eq. (3) observation."""

        for name, modules in self.modules.items():
            current = torch.stack([module.b.detach() for module in modules])
            updated, count = update_running_average(
                self.running_b[name], current, self.running_count[name]
            )
            self.running_b[name] = updated
            self.running_count[name] = count

    def capture_first_merge_snapshot(
        self,
        candidate_mode: str = "all",
        decision_scope: str = "per_projection",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> FirstMergeSnapshot:
        targets: Dict[str, FactorSnapshot] = {}
        for name, modules in self.modules.items():
            layer_b = torch.stack([module.b.detach().clone() for module in modules])
            running = self.running_b[name].detach().clone()
            if self.running_count[name] == 0:
                # This makes an untrained inspection well defined while recording
                # that no historical optimizer observations were supplied.
                running = layer_b.clone()
            targets[name] = FactorSnapshot(
                shared_a=modules[0].shared_a.detach().clone(),
                layer_b=layer_b,
                running_b=running,
                running_count=self.running_count[name],
                target_name=name,
            )
        snapshot_metadata = {
            "architecture": self.model.__class__.__name__,
            "target_modules": list(self.modules),
            "a_scope": self.a_scope,
            "alpha": self.alpha,
            "a_init_std": self.a_init_std,
            "source_notes": dict(ASLORA_SOURCE_NOTES),
            "warning": (
                "This is a first-decision snapshot.  A full merge-tree gauge claim "
                "would additionally require equivariant intervening optimization."
            ),
        }
        if metadata:
            snapshot_metadata.update(dict(metadata))
        return FirstMergeSnapshot(
            targets=targets,
            candidate_mode=candidate_mode,
            decision_scope=decision_scope,
            metadata=snapshot_metadata,
        )

    @contextlib.contextmanager
    def temporary_lower_uses_upper(
        self,
        pair: Pair,
        targets: Optional[Iterable[str]] = None,
    ) -> Iterator[None]:
        selected_targets = list(targets) if targets is not None else list(self.modules)
        originals: List[Tuple[SharedALoRALinear, nn.Parameter]] = []
        try:
            for name in selected_targets:
                if name not in self.modules:
                    raise ValueError("unknown target %r" % name)
                modules = self.modules[name]
                _validate_pair(pair, len(modules))
                lower, upper = pair
                originals.append((modules[lower], modules[lower].b))
                modules[lower].b = modules[upper].b
            yield
        finally:
            for module, parameter in originals:
                module.b = parameter

    def evaluate_immediate_candidate_merges(
        self,
        evaluator: Callable[[nn.Module], Mapping[str, float]],
        candidate_mode: str = "all",
        targets: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run an external classification evaluator once per immediate merge."""

        selected_targets = list(targets) if targets is not None else list(self.modules)
        if not selected_targets:
            raise ValueError("at least one target is required")
        num_layers = len(self.modules[selected_targets[0]])
        records: List[Dict[str, Any]] = []
        for pair in candidate_pairs(num_layers, candidate_mode):
            with self.temporary_lower_uses_upper(pair, selected_targets):
                metrics = dict(evaluator(self.model))
            records.append({"pair": pair, "metrics": metrics})
        return records

    def commit_lower_uses_upper(
        self,
        pair: Pair,
        targets: Optional[Iterable[str]] = None,
    ) -> None:
        """Permanently tie the lower layer's B parameter to the upper layer's B.

        Callers must rebuild an optimizer created before this operation so its
        parameter groups do not retain the replaced lower-layer parameter.
        """

        selected_targets = list(targets) if targets is not None else list(self.modules)
        for name in selected_targets:
            modules = self.modules[name]
            _validate_pair(pair, len(modules))
            lower, upper = pair
            modules[lower].b = modules[upper].b


def attach_aslora_to_roberta_classifier(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_names: Sequence[str] = ("query", "value"),
    a_scope: str = "per_projection",
    a_init_std: float = 0.02,
    freeze_backbone: bool = True,
    train_classifier: bool = True,
    seed: int = 0,
) -> RobertaASLoRAHandle:
    """Patch standard RoBERTa WQ/WV modules with ASLoRA factors.

    No model is downloaded and no Transformers import occurs here.  ``model``
    must expose ``model.roberta.encoder.layer[*].attention.self.{query,value}``.
    ``per_projection`` (default) shares one A across depth for WQ and another
    across depth for WV. ``global`` shares one A across both families when their
    input dimensions agree; this option exposes the paper's Q/V-scope ambiguity.
    """

    if rank < 1:
        raise ValueError("rank must be positive")
    if a_scope not in {"per_projection", "global"}:
        raise ValueError("a_scope must be 'per_projection' or 'global'")
    try:
        layers = list(model.roberta.encoder.layer)
    except AttributeError as error:
        raise TypeError("model does not expose the standard RoBERTa encoder layout") from error
    if not layers:
        raise ValueError("RoBERTa encoder has no layers")
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    shared_by_name: Dict[str, nn.Parameter] = {}
    global_shared: Optional[nn.Parameter] = None
    modules: Dict[str, List[SharedALoRALinear]] = {}

    for target_name in target_names:
        base_layers: List[nn.Linear] = []
        for layer in layers:
            try:
                base = getattr(layer.attention.self, target_name)
            except AttributeError as error:
                raise TypeError("RoBERTa layer lacks attention.self.%s" % target_name) from error
            if not isinstance(base, nn.Linear):
                raise TypeError("attention.self.%s is not torch.nn.Linear" % target_name)
            base_layers.append(base)
        input_sizes = {base.in_features for base in base_layers}
        output_sizes = {base.out_features for base in base_layers}
        if len(input_sizes) != 1 or len(output_sizes) != 1:
            raise ValueError("all layers in a projection family must have matching shapes")
        input_size = next(iter(input_sizes))
        reference = base_layers[0]
        if a_scope == "global" and global_shared is not None:
            if global_shared.shape != (rank, input_size):
                raise ValueError("global A cannot span target modules with different input sizes")
            shared_a = global_shared
        else:
            sample = torch.randn(rank, input_size, generator=generator, dtype=torch.float32)
            sample = sample.to(device=reference.weight.device, dtype=reference.weight.dtype)
            shared_a = nn.Parameter(sample * a_init_std)
            if a_scope == "global":
                global_shared = shared_a
        shared_by_name[target_name] = shared_a
        wrapped: List[SharedALoRALinear] = []
        for layer, base in zip(layers, base_layers):
            wrapper = SharedALoRALinear(base, shared_a, rank=rank, alpha=alpha)
            setattr(layer.attention.self, target_name, wrapper)
            wrapped.append(wrapper)
        modules[target_name] = wrapped

    if train_classifier and hasattr(model, "classifier"):
        for parameter in model.classifier.parameters():
            parameter.requires_grad_(True)

    # Explicitly re-enable the shared A and all layer-specific B parameters.
    for shared_a in set(shared_by_name.values()):
        shared_a.requires_grad_(True)
    for wrapped in modules.values():
        for module in wrapped:
            module.b.requires_grad_(True)

    return RobertaASLoRAHandle(
        model=model,
        modules=modules,
        a_scope=a_scope,
        alpha=alpha,
        a_init_std=a_init_std,
    )
