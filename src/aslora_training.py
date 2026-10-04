"""Controlled training and first-merge audits for ASLoRA on MRPC.

The core utilities in this module depend only on PyTorch.  Hugging Face
``transformers`` and ``datasets`` are imported lazily by :func:`require_hf`
so tensor, scheduling, and mock-model tests remain usable without downloading
models or datasets.

The implementation stops at the first ASLoRA merge opportunity.  Optimizer
steps are one-indexed, the cumulative B average is updated immediately after
each optimizer update, and the first event is therefore ``t=560`` for
``Ts=320`` and ``m=240``.  The pre-merge model is never committed to a merge:
every structural action is evaluated through temporary parameter tying and is
restored before the next candidate.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.aslora_witness import (
    ASLORA_SOURCE_NOTES,
    FactorSnapshot,
    FirstMergeSnapshot,
    Pair,
    RobertaASLoRAHandle,
    SharedALoRALinear,
    aggregate_pair_distances,
    apply_snapshot_gauges,
    candidate_pairs,
    effective_pair_distances,
    effective_updates,
    gauge_condition_number,
    make_orthogonal_gauge,
    raw_pair_distances,
    save_first_merge_snapshot,
    select_minimum_pair,
)


PAPER_EVIDENCE: Dict[str, str] = {
    "architecture": (
        "ASLoRA arXiv:2412.10135v2 Section 4.1 uses RoBERTa-base on GLUE and fine-tunes W_Q, W_V, "
        "and the classification head; the shared A is cross-layer and B is "
        "layer-specific before merging."
    ),
    "mrpc_hyperparameters": (
        "ASLoRA arXiv:2412.10135v2 Appendix Table 5 gives MRPC: learning rate 4e-4, batch size 16, "
        "30 epochs, weight decay 0.1, max length 512, Ts=320, merge interval "
        "240, rank 8, alpha 16, linear scheduler, and warmup ratio 0.06."
    ),
    "first_merge_timing": (
        "ASLoRA arXiv:2412.10135v2 Algorithm 1 indexes t=1,...,T and merges when t>Ts and "
        "(t-Ts)%m==0; for MRPC this makes the first event t=560, not t=320."
    ),
    "running_average_timing": (
        "ASLoRA arXiv:2412.10135v2 Algorithm 1 updates the cumulative B average before testing the merge "
        "condition.  The paper does not state whether B^t is sampled before or "
        "after the optimizer update; this runner declares after-update sampling."
    ),
    "candidate_ambiguity": ASLORA_SOURCE_NOTES["candidate_ambiguity"],
    "decision_scope": (
        "ASLoRA arXiv:2412.10135v2 Figure 3 reports separate query and value sharing configurations, so "
        "per-projection selection is the primary interpretation.  Algorithm 1 "
        "does not explicitly define multi-target scheduling; a joint root-sum-"
        "of-squares interpretation is reported as sensitivity analysis."
    ),
    "undefined_update_ratio": (
        "ASLoRA arXiv:2412.10135v2 Appendix Table 5 lists update ratio lambda=0.5 but the available method "
        "text does not define an operation using it.  It is recorded in "
        "provenance and deliberately not reverse-engineered into the algorithm."
    ),
}


PAPER_DEFAULTS: Dict[str, Any] = {
    "rank": 8,
    "alpha": 16.0,
    "learning_rate": 4e-4,
    "train_batch_size": 16,
    "epochs": 30,
    "weight_decay": 0.1,
    "max_length": 512,
    "start_merge_step": 320,
    "merge_interval": 240,
    "merge_count": 7,
    "warmup_ratio": 0.06,
    "gradient_accumulation_steps": 1,
}


def first_scheduled_merge_step(start_merge_step: int, merge_interval: int) -> int:
    """Return the first positive t satisfying Algorithm 1's strict condition."""

    start = int(start_merge_step)
    interval = int(merge_interval)
    if start < 0:
        raise ValueError("start_merge_step must be nonnegative")
    if interval < 1:
        raise ValueError("merge_interval must be positive")
    return start + interval


def is_scheduled_merge_step(
    optimizer_step: int,
    start_merge_step: int,
    merge_interval: int,
    merges_completed: int = 0,
    merge_count: int = 1,
) -> bool:
    """Implement Algorithm 1 with one-indexed optimizer steps."""

    step = int(optimizer_step)
    start = int(start_merge_step)
    interval = int(merge_interval)
    if step < 1:
        return False
    if interval < 1:
        raise ValueError("merge_interval must be positive")
    if merges_completed < 0 or merge_count < 0:
        raise ValueError("merge counts must be nonnegative")
    return (
        merges_completed < merge_count
        and step > start
        and (step - start) % interval == 0
    )


@dataclass
class ASLoRAMRPCConfig:
    """Paper-faithful defaults plus explicitly recorded runtime choices."""

    model_name: str = "FacebookAI/roberta-base"
    model_revision: str = "main"
    dataset_name: str = "glue"
    dataset_config: str = "mrpc"
    dataset_revision: str = "main"
    train_parquet: Optional[str] = None
    validation_parquet: Optional[str] = None
    local_files_only: bool = False
    rank: int = 8
    alpha: float = 16.0
    learning_rate: float = 4e-4
    train_batch_size: int = 16
    eval_batch_size: int = 16
    epochs: int = 30
    weight_decay: float = 0.1
    max_length: int = 512
    start_merge_step: int = 320
    merge_interval: int = 240
    merge_count: int = 7
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    target_names: Tuple[str, ...] = ("query", "value")
    a_scope: str = "per_projection"
    candidate_modes: Tuple[str, ...] = ("adjacent", "all")
    primary_decision_scope: str = "per_projection"
    decision_scopes: Tuple[str, ...] = ("per_projection", "joint")
    gauge_condition_limits: Tuple[float, ...] = (2.0, 4.0, 8.0, 30.0)
    gauge_atol: float = 5e-5
    gauge_rtol: float = 5e-5
    seed: int = 0
    num_workers: int = 0
    device: str = "cuda"
    update_ratio_record_only: float = 0.5
    deterministic_algorithms: bool = True
    primary_action_evaluation_repeats: int = 2

    @property
    def first_merge_step(self) -> int:
        return first_scheduled_merge_step(self.start_merge_step, self.merge_interval)

    @property
    def effective_batch_size(self) -> int:
        return self.train_batch_size * self.gradient_accumulation_steps

    def paper_deviations(self) -> Dict[str, Dict[str, Any]]:
        deviations: Dict[str, Dict[str, Any]] = {}
        for name, expected in PAPER_DEFAULTS.items():
            actual = getattr(self, name)
            if actual != expected:
                deviations[name] = {"paper": expected, "actual": actual}
        if tuple(self.target_names) != ("query", "value"):
            deviations["target_names"] = {
                "paper": ["query", "value"],
                "actual": list(self.target_names),
            }
        if self.a_scope != "per_projection":
            deviations["a_scope"] = {
                "paper_interpretation": "per_projection",
                "actual": self.a_scope,
            }
        return deviations

    def validate(self, allow_paper_config_deviation: bool = False) -> None:
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.alpha <= 0 or self.learning_rate <= 0:
            raise ValueError("alpha and learning_rate must be positive")
        if self.train_batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive")
        if self.epochs < 1 or self.max_length < 1:
            raise ValueError("epochs and max_length must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.primary_action_evaluation_repeats < 2:
            raise ValueError("primary_action_evaluation_repeats must be at least two")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must lie in [0,1)")
        if self.weight_decay < 0 or self.max_grad_norm < 0:
            raise ValueError("weight_decay and max_grad_norm must be nonnegative")
        if self.a_scope != "per_projection":
            raise ValueError(
                "the controlled gauge audit requires an independent cross-layer A "
                "for each projection family"
            )
        if not self.target_names or set(self.target_names) != {"query", "value"}:
            raise ValueError("the controlled MRPC run requires W_Q and W_V")
        if not self.candidate_modes or any(
            mode not in {"adjacent", "all"} for mode in self.candidate_modes
        ):
            raise ValueError("candidate_modes must contain adjacent and/or all")
        if len(set(self.candidate_modes)) != len(self.candidate_modes):
            raise ValueError("candidate_modes must not contain duplicates")
        if (self.train_parquet is None) != (self.validation_parquet is None):
            raise ValueError(
                "train_parquet and validation_parquet must be supplied together"
            )
        if set(self.decision_scopes) != {"per_projection", "joint"}:
            raise ValueError("both per_projection and joint interpretations are required")
        if self.primary_decision_scope != "per_projection":
            raise ValueError("Figure 3 supports per_projection as the primary interpretation")
        if any(limit < 1 or not math.isfinite(limit) for limit in self.gauge_condition_limits):
            raise ValueError("gauge condition limits must be finite and at least one")
        if self.first_merge_step != 560 and not allow_paper_config_deviation:
            raise ValueError("strict MRPC configuration must first merge at optimizer step 560")
        deviations = self.paper_deviations()
        if deviations and not allow_paper_config_deviation:
            raise ValueError("paper hyperparameter deviations require explicit opt-in: %r" % deviations)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for name in (
            "target_names",
            "candidate_modes",
            "decision_scopes",
            "gauge_condition_limits",
        ):
            payload[name] = list(payload[name])
        payload["first_merge_step"] = self.first_merge_step
        payload["effective_batch_size"] = self.effective_batch_size
        payload["paper_deviations"] = self.paper_deviations()
        return payload


@dataclass
class NamedGauge:
    name: str
    family: str
    matrices: Dict[str, torch.Tensor]
    requested_condition_limit: float
    source: str = "deterministic_prevalidation"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def condition_numbers(self) -> Dict[str, float]:
        return {
            target: gauge_condition_number(matrix)
            for target, matrix in self.matrices.items()
        }

    def to_dict(self, include_matrices: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "family": self.family,
            "requested_condition_limit": self.requested_condition_limit,
            "condition_numbers": self.condition_numbers(),
            "source": self.source,
            "metadata": self.metadata,
        }
        if include_matrices:
            payload["matrices_Q"] = {
                target: matrix.detach().cpu().tolist()
                for target, matrix in self.matrices.items()
            }
        return payload


def _orthogonal_from_generator(
    rank: int,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    sample = torch.randn(rank, rank, generator=generator, dtype=dtype)
    q, r = torch.linalg.qr(sample)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q @ torch.diag(signs)


def make_bounded_gauge(
    rank: int,
    family: str,
    condition_limit: float,
    seed: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate Q for ``B'=BQ, A'=Q^{-1}A`` without validation access."""

    if rank < 1:
        raise ValueError("rank must be positive")
    limit = float(condition_limit)
    if limit < 1 or not math.isfinite(limit):
        raise ValueError("condition_limit must be finite and at least one")
    if family == "identity":
        result = torch.eye(rank, dtype=dtype)
    elif family == "orthogonal":
        result = make_orthogonal_gauge(rank, seed=seed, dtype=dtype)
    elif family in {"diagonal", "dense"}:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        if rank == 1:
            singular_values = torch.ones(1, dtype=dtype)
        else:
            exponents = torch.linspace(0.0, 1.0, rank, dtype=dtype)
            singular_values = torch.exp(-math.log(limit) * exponents)
            permutation = torch.randperm(rank, generator=generator)
            singular_values = singular_values[permutation]
        diagonal = torch.diag(singular_values)
        if family == "diagonal":
            result = diagonal
        else:
            left = _orthogonal_from_generator(rank, generator, dtype)
            right = _orthogonal_from_generator(rank, generator, dtype)
            result = left @ diagonal @ right.transpose(0, 1)
    else:
        raise ValueError("family must be identity, orthogonal, diagonal, or dense")
    if device is not None:
        result = result.to(device=device)
    actual = gauge_condition_number(result)
    tolerance = 1e-8 * max(1.0, limit)
    if actual > limit + tolerance:
        raise RuntimeError("constructed gauge exceeds its condition limit")
    return result


def build_deterministic_gauge_bank(
    snapshot: FirstMergeSnapshot,
    condition_limits: Sequence[float] = (2.0, 4.0, 8.0, 30.0),
    seed: int = 0,
) -> List[NamedGauge]:
    """Create identity, condition-one sanity, diagonal, and dense gauges."""

    targets = sorted(snapshot.targets)
    ranks = {name: int(snapshot.targets[name].shared_a.shape[0]) for name in targets}
    dtype = torch.float64

    def matrices(family: str, limit: float, offset: int) -> Dict[str, torch.Tensor]:
        return {
            name: make_bounded_gauge(
                ranks[name], family, limit, seed + offset + 1009 * index, dtype=dtype
            )
            for index, name in enumerate(targets)
        }

    bank = [
        NamedGauge("identity", "identity", matrices("identity", 1.0, 0), 1.0),
        NamedGauge(
            "orthogonal_k1",
            "orthogonal",
            matrices("orthogonal", 1.0, 17),
            1.0,
        ),
    ]
    for index, limit_value in enumerate(condition_limits):
        limit = float(limit_value)
        bank.append(
            NamedGauge(
                "diagonal_k%s" % ("%g" % limit),
                "diagonal",
                matrices("diagonal", limit, 101 + index * 23),
                limit,
            )
        )
        bank.append(
            NamedGauge(
                "dense_k%s" % ("%g" % limit),
                "dense",
                matrices("dense", limit, 503 + index * 29),
                limit,
            )
        )
    return bank


def _matrix_from_external_entry(
    entry: Mapping[str, Any],
    rank: int,
    convention: str,
) -> torch.Tensor:
    if "metric_C" in entry:
        metric = torch.as_tensor(entry["metric_C"], dtype=torch.float64)
        if metric.shape != (rank, rank):
            raise ValueError("external metric_C has the wrong rank")
        metric = 0.5 * (metric + metric.transpose(0, 1))
        try:
            return torch.linalg.cholesky(metric)
        except RuntimeError as error:
            raise ValueError("external metric_C must be positive definite") from error
    raw = entry.get("Q", entry.get("R", entry.get("matrix")))
    if raw is None:
        raise ValueError("external gauge entry needs Q, R, matrix, or metric_C")
    matrix = torch.as_tensor(raw, dtype=torch.float64)
    if matrix.shape != (rank, rank):
        raise ValueError("external gauge matrix has the wrong rank")
    used_convention = str(entry.get("convention", convention)).upper()
    if "R" in entry and "convention" not in entry:
        used_convention = "R"
    if "Q" in entry and "convention" not in entry:
        used_convention = "Q"
    if used_convention == "R":
        return torch.linalg.inv(matrix)
    if used_convention != "Q":
        raise ValueError("external convention must be Q or R")
    return matrix


def load_external_gauge_bank(
    path: Path,
    snapshot: FirstMergeSnapshot,
    max_condition_number: float = 30.0,
) -> List[NamedGauge]:
    """Load explicit Q/R or decision-audit metric-C witnesses from JSON."""

    source_path = Path(path)
    source_bytes = source_path.read_bytes()
    payload = json.loads(source_bytes.decode("utf-8"))
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    root_convention = str(payload.get("convention", "Q")) if isinstance(payload, dict) else "Q"
    entries: Any = payload.get("gauges", [payload]) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("external gauge JSON must contain a list")
    targets = sorted(snapshot.targets)
    result: List[NamedGauge] = []
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ValueError("each external gauge must be an object")
        raw_matrices = item.get("matrices")
        matrices: Dict[str, torch.Tensor] = {}
        if isinstance(raw_matrices, Mapping):
            for target in targets:
                if target not in raw_matrices:
                    raise ValueError("external gauge is missing target %s" % target)
                target_entry = raw_matrices[target]
                wrapped = target_entry if isinstance(target_entry, Mapping) else {"matrix": target_entry}
                matrices[target] = _matrix_from_external_entry(
                    wrapped,
                    int(snapshot.targets[target].shared_a.shape[0]),
                    root_convention,
                )
        else:
            target_name = item.get("target")
            for target in targets:
                rank = int(snapshot.targets[target].shared_a.shape[0])
                if target_name is not None and target != target_name:
                    matrices[target] = torch.eye(rank, dtype=torch.float64)
                else:
                    matrices[target] = _matrix_from_external_entry(
                        item, rank, root_convention
                    )
        conditions = {name: gauge_condition_number(value) for name, value in matrices.items()}
        if any(value > max_condition_number + 1e-8 for value in conditions.values()):
            raise ValueError("external gauge exceeds max_condition_number")
        result.append(
            NamedGauge(
                name=str(item.get("name", "external_%d" % index)),
                family="external_witness",
                matrices=matrices,
                requested_condition_limit=max(conditions.values()),
                source="external_prevalidation_json",
                metadata={
                    "path": str(source_path.resolve()),
                    "sha256": source_sha256,
                    "declared_convention": item.get("convention", root_convention),
                },
            )
        )
    return result


def action_key(action: Mapping[str, Pair]) -> str:
    if not action:
        return "unmerged"
    return "|".join(
        "%s:%d-%d" % (target, pair[0], pair[1])
        for target, pair in sorted(action.items())
    )


def action_to_json(action: Mapping[str, Pair]) -> Dict[str, List[int]]:
    return {target: [int(pair[0]), int(pair[1])] for target, pair in sorted(action.items())}


@contextlib.contextmanager
def temporary_target_action(
    handle: RobertaASLoRAHandle,
    action: Mapping[str, Pair],
) -> Iterator[None]:
    """Temporarily apply possibly different query/value lower-to-upper ties."""

    with ExitStack() as stack:
        for target, pair in sorted(action.items()):
            stack.enter_context(handle.temporary_lower_uses_upper(pair, targets=(target,)))
        yield


def structural_identity_signature(handle: RobertaASLoRAHandle) -> Dict[str, List[int]]:
    return {
        target: [id(module.b) for module in modules]
        for target, modules in sorted(handle.modules.items())
    }


def trainable_tensor_digest(model: nn.Module) -> str:
    """Hash trainable tensor values and names without touching frozen weights."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass
class LiveGaugeRestorationAudit:
    restoration_exact: bool = False
    identity_restored: bool = False


@contextlib.contextmanager
def temporary_live_gauges(
    handle: RobertaASLoRAHandle,
    matrices: Mapping[str, torch.Tensor],
) -> Iterator[LiveGaugeRestorationAudit]:
    """Apply fixed common Q gauges in place and restore exact tensors/objects."""

    if set(matrices) != set(handle.modules):
        raise ValueError("one gauge matrix is required for every target family")
    original_identity = structural_identity_signature(handle)
    original_a: Dict[str, torch.Tensor] = {}
    original_b: Dict[str, List[torch.Tensor]] = {}
    original_running = {
        target: values.detach().clone() for target, values in handle.running_b.items()
    }
    shared_parameter_ids: Dict[int, str] = {}
    audit = LiveGaugeRestorationAudit()
    try:
        with torch.no_grad():
            for target, modules in sorted(handle.modules.items()):
                q = torch.as_tensor(
                    matrices[target],
                    dtype=modules[0].shared_a.dtype,
                    device=modules[0].shared_a.device,
                )
                rank = modules[0].shared_a.shape[0]
                if q.shape != (rank, rank):
                    raise ValueError("live gauge has the wrong rank for %s" % target)
                shared_id = id(modules[0].shared_a)
                if shared_id in shared_parameter_ids:
                    raise ValueError(
                        "independent projection gauges require distinct shared A parameters"
                    )
                shared_parameter_ids[shared_id] = target
                original_a[target] = modules[0].shared_a.detach().clone()
                original_b[target] = [module.b.detach().clone() for module in modules]
                transformed_a = torch.linalg.solve(q, modules[0].shared_a.detach())
                modules[0].shared_a.copy_(transformed_a)
                for module in modules:
                    module.b.copy_(module.b.detach() @ q)
                handle.running_b[target] = handle.running_b[target] @ q
        yield audit
    finally:
        with torch.no_grad():
            for target, modules in sorted(handle.modules.items()):
                modules[0].shared_a.copy_(original_a[target])
                for module, value in zip(modules, original_b[target]):
                    module.b.copy_(value)
                handle.running_b[target] = original_running[target]
        audit.identity_restored = structural_identity_signature(handle) == original_identity
        audit.restoration_exact = audit.identity_restored and all(
            torch.equal(handle.running_b[target], original_running[target])
            and torch.equal(handle.modules[target][0].shared_a, original_a[target])
            and all(
                torch.equal(module.b, value)
                for module, value in zip(handle.modules[target], original_b[target])
            )
            for target in handle.modules
        )


class ProjectedActivationCollector:
    """Collect uncentred moments of ``z=A x`` for every target and layer.

    Layer-specific moments are required for the immediate ``lower <- upper``
    disturbance: the replacement occurs at the lower layer and must be weighted
    by that layer's own pre-merge inputs.  Aggregating over layers is retained
    only for a clearly labelled running-average selection proxy.
    """

    def __init__(self, handle: RobertaASLoRAHandle) -> None:
        self.handle = handle
        self.grams: Dict[str, List[Optional[torch.Tensor]]] = {
            target: [None for _ in modules]
            for target, modules in handle.modules.items()
        }
        self.counts: Dict[str, List[int]] = {
            target: [0 for _ in modules]
            for target, modules in handle.modules.items()
        }
        self._mask: Optional[torch.Tensor] = None
        self._hooks: List[Any] = []

    def __enter__(self) -> "ProjectedActivationCollector":
        for target, modules in self.handle.modules.items():
            for layer_index, module in enumerate(modules):
                hook = module.register_forward_pre_hook(
                    self._make_hook(target, layer_index)
                )
                self._hooks.append(hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._mask = None

    def begin_batch(self, attention_mask: Optional[torch.Tensor]) -> None:
        self._mask = None if attention_mask is None else attention_mask.detach()

    def end_batch(self) -> None:
        self._mask = None

    def _make_hook(self, target: str, layer_index: int) -> Any:
        def hook(module: SharedALoRALinear, inputs: Tuple[torch.Tensor, ...]) -> None:
            if not inputs:
                return
            values = inputs[0].detach()
            projected = F.linear(values, module.shared_a.detach()).float()
            if projected.ndim == 3 and self._mask is not None:
                mask = self._mask.to(device=projected.device, dtype=torch.bool)
                selected = projected[mask]
            else:
                selected = projected.reshape(-1, projected.shape[-1])
            if selected.numel() == 0:
                return
            gram = (selected.transpose(0, 1) @ selected).double().cpu()
            if self.grams[target][layer_index] is None:
                self.grams[target][layer_index] = torch.zeros_like(gram)
            self.grams[target][layer_index] += gram
            self.counts[target][layer_index] += int(selected.shape[0])

        return hook

    def covariances(self) -> Dict[str, torch.Tensor]:
        """Return count-weighted, across-layer moments for proxy diagnostics."""

        result: Dict[str, torch.Tensor] = {}
        for target, counts in self.counts.items():
            total_count = sum(counts)
            available = [gram for gram in self.grams[target] if gram is not None]
            if total_count > 0 and available:
                result[target] = sum(available[1:], available[0].clone()) / float(
                    total_count
                )
        return result

    def per_layer_covariances(self) -> Dict[str, torch.Tensor]:
        """Return ``[layer, rank, rank]`` moments, omitting incomplete targets."""

        result: Dict[str, torch.Tensor] = {}
        for target, counts in self.counts.items():
            grams = self.grams[target]
            if all(count > 0 and gram is not None for count, gram in zip(counts, grams)):
                result[target] = torch.stack(
                    [
                        gram / float(count)
                        for gram, count in zip(grams, counts)
                        if gram is not None
                    ]
                )
        return result

    def aggregate_counts(self) -> Dict[str, int]:
        return {target: sum(counts) for target, counts in self.counts.items()}


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    f1: float
    examples: int
    batches: int
    logits: torch.Tensor
    labels: torch.Tensor
    projected_covariances: Dict[str, torch.Tensor] = field(default_factory=dict)
    projected_counts: Dict[str, int] = field(default_factory=dict)
    projected_layer_covariances: Dict[str, torch.Tensor] = field(default_factory=dict)
    projected_layer_counts: Dict[str, List[int]] = field(default_factory=dict)

    def metrics(self) -> Dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "examples": self.examples,
            "batches": self.batches,
        }


def binary_classification_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    predictions = predictions.detach().cpu().long().reshape(-1)
    labels = labels.detach().cpu().long().reshape(-1)
    if predictions.numel() != labels.numel() or labels.numel() == 0:
        raise ValueError("predictions and labels must have the same positive length")
    accuracy = float((predictions == labels).float().mean())
    true_positive = int(((predictions == 1) & (labels == 1)).sum())
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    false_negative = int(((predictions == 0) & (labels == 1)).sum())
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = 0.0 if denominator == 0 else 2.0 * true_positive / float(denominator)
    return {"accuracy": accuracy, "f1": f1}


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        name: value.to(device) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def evaluate_classifier(
    model: nn.Module,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    activation_collector: Optional[ProjectedActivationCollector] = None,
) -> EvaluationResult:
    """Evaluate one deterministic complete validation pass."""

    was_training = model.training
    model.eval()
    losses = 0.0
    total = 0
    batches = 0
    logits_parts: List[torch.Tensor] = []
    label_parts: List[torch.Tensor] = []
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_batch(raw_batch, device)
            labels = batch.get("labels", batch.get("label"))
            if labels is None:
                raise ValueError("validation batches must contain labels")
            if "label" in batch and "labels" not in batch:
                batch = dict(batch)
                batch["labels"] = batch.pop("label")
            if activation_collector is not None:
                activation_collector.begin_batch(batch.get("attention_mask"))
            outputs = model(**batch)
            if activation_collector is not None:
                activation_collector.end_batch()
            logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
            loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss")
            if loss is None:
                loss = F.cross_entropy(logits, batch["labels"])
            count = int(batch["labels"].shape[0])
            losses += float(loss.detach()) * count
            total += count
            batches += 1
            logits_parts.append(logits.detach().float().cpu())
            label_parts.append(batch["labels"].detach().long().cpu())
    model.train(was_training)
    if total == 0:
        raise ValueError("validation dataloader is empty")
    logits_all = torch.cat(logits_parts, dim=0)
    labels_all = torch.cat(label_parts, dim=0)
    metrics = binary_classification_metrics(logits_all.argmax(dim=-1), labels_all)
    covariances = {} if activation_collector is None else activation_collector.covariances()
    counts = (
        {} if activation_collector is None else activation_collector.aggregate_counts()
    )
    layer_covariances = (
        {} if activation_collector is None else activation_collector.per_layer_covariances()
    )
    layer_counts = (
        {}
        if activation_collector is None
        else {target: list(values) for target, values in activation_collector.counts.items()}
    )
    return EvaluationResult(
        loss=losses / float(total),
        accuracy=metrics["accuracy"],
        f1=metrics["f1"],
        examples=total,
        batches=batches,
        logits=logits_all,
        labels=labels_all,
        projected_covariances=covariances,
        projected_counts=counts,
        projected_layer_covariances=layer_covariances,
        projected_layer_counts=layer_counts,
    )


def evaluation_difference(reference: EvaluationResult, candidate: EvaluationResult) -> Dict[str, Any]:
    if reference.logits.shape != candidate.logits.shape:
        raise ValueError("evaluation logits have different shapes")
    if not torch.equal(reference.labels, candidate.labels):
        raise ValueError("validation order or labels changed between evaluations")
    return {
        "max_abs_logit_error": float((reference.logits - candidate.logits).abs().max()),
        "loss_abs_error": abs(reference.loss - candidate.loss),
        "accuracy_abs_error": abs(reference.accuracy - candidate.accuracy),
        "f1_abs_error": abs(reference.f1 - candidate.f1),
        "prediction_disagreement": float(
            (reference.logits.argmax(dim=-1) != candidate.logits.argmax(dim=-1))
            .float()
            .mean()
        ),
    }


def activation_weighted_pair_distances(
    state: FactorSnapshot,
    projected_covariance: torch.Tensor,
    pairs: Sequence[Pair],
) -> Dict[Pair, torch.Tensor]:
    """Compute a pooled, running-average functional selection proxy.

    This is *not* the disturbance caused by a current lower-to-upper merge: it
    uses ``running_b`` and one covariance pooled across layers.  The exact local
    pre-merge disturbance for that action is implemented by
    :func:`current_lower_uses_upper_activation_rms`.
    """

    covariance = torch.as_tensor(
        projected_covariance,
        dtype=state.running_b.dtype,
        device=state.running_b.device,
    )
    rank = state.running_b.shape[-1]
    if covariance.shape != (rank, rank):
        raise ValueError("projected covariance has the wrong rank")
    result: Dict[Pair, torch.Tensor] = {}
    for pair in pairs:
        lower, upper = pair
        difference = state.running_b[lower] - state.running_b[upper]
        squared = torch.trace(
            difference @ covariance @ difference.transpose(0, 1)
        ).clamp_min(0)
        result[pair] = squared.sqrt()
    return result


def current_effective_update_pair_distances(
    state: FactorSnapshot,
    pairs: Sequence[Pair],
    scaling: float = 1.0,
) -> Dict[Pair, torch.Tensor]:
    """Frobenius change in the current scaled adapter update for each action."""

    scale = float(scaling)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("LoRA scaling must be finite and positive")
    return {
        pair: scale * value
        for pair, value in effective_pair_distances(
            state.shared_a, state.layer_b, pairs=pairs
        ).items()
    }


def current_lower_uses_upper_activation_rms(
    state: FactorSnapshot,
    per_layer_projected_covariances: torch.Tensor,
    pairs: Sequence[Pair],
    scaling: float = 1.0,
) -> Dict[Pair, torch.Tensor]:
    """Exact local RMS adapter-output disturbance on cached pre-merge inputs.

    For action ``lower <- upper`` this computes

    ``(alpha/r) sqrt(tr((B_upper-B_lower) G_lower (B_upper-B_lower)^T))``,

    where ``G_lower=E[(A x_lower)(A x_lower)^T]`` is collected at the lower
    layer in the unmerged model.  It is exact for that layer's immediate LoRA
    output on the complete cached validation distribution; it is not called a
    network-level loss or causal cost.  The latter is measured by actually
    tying the parameter and running a complete validation pass.
    """

    covariances = torch.as_tensor(
        per_layer_projected_covariances,
        dtype=state.layer_b.dtype,
        device=state.layer_b.device,
    )
    num_layers = state.layer_b.shape[0]
    rank = state.layer_b.shape[-1]
    if covariances.shape != (num_layers, rank, rank):
        raise ValueError("per-layer projected covariances have the wrong shape")
    scale = float(scaling)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("LoRA scaling must be finite and positive")
    result: Dict[Pair, torch.Tensor] = {}
    for lower, upper in pairs:
        difference = state.layer_b[upper] - state.layer_b[lower]
        squared = torch.trace(
            difference
            @ covariances[lower]
            @ difference.transpose(0, 1)
        ).clamp_min(0)
        result[(lower, upper)] = scale * squared.sqrt()
    return result


def transform_projected_covariance(covariance: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(q, dtype=covariance.dtype, device=covariance.device)
    left = torch.linalg.solve(q, covariance)
    return torch.linalg.solve(q, left.transpose(0, 1)).transpose(0, 1)


def distance_table(
    snapshot: FirstMergeSnapshot,
    mode: str,
    projected_covariances: Optional[Mapping[str, torch.Tensor]] = None,
    per_layer_projected_covariances: Optional[Mapping[str, torch.Tensor]] = None,
    lora_scalings: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    pairs = candidate_pairs(next(iter(snapshot.targets.values())).layer_b.shape[0], mode)
    per_target: Dict[str, Dict[str, Dict[Pair, torch.Tensor]]] = {}
    for target, state in sorted(snapshot.targets.items()):
        scaling = 1.0 if lora_scalings is None else float(lora_scalings[target])
        metrics: Dict[str, Dict[Pair, torch.Tensor]] = {
            "running_raw_B_selection_proxy": raw_pair_distances(
                state.running_b, pairs=pairs
            ),
            "running_effective_update_selection_proxy": {
                pair: scaling * value
                for pair, value in effective_pair_distances(
                    state.shared_a, state.running_b, pairs=pairs
                ).items()
            },
            "current_effective_update_frobenius": (
                current_effective_update_pair_distances(state, pairs, scaling)
            ),
        }
        if projected_covariances is not None and target in projected_covariances:
            metrics["running_average_activation_selection_proxy"] = {
                pair: scaling * value
                for pair, value in activation_weighted_pair_distances(
                    state, projected_covariances[target], pairs
                ).items()
            }
        if (
            per_layer_projected_covariances is not None
            and target in per_layer_projected_covariances
        ):
            metrics["current_lower_local_activation_rms"] = (
                current_lower_uses_upper_activation_rms(
                    state,
                    per_layer_projected_covariances[target],
                    pairs,
                    scaling,
                )
            )
        per_target[target] = metrics
    joint: Dict[str, Dict[Pair, torch.Tensor]] = {}
    metric_names = set.intersection(
        *(set(metrics) for metrics in per_target.values())
    )
    for metric in sorted(metric_names):
        joint[metric] = {}
        for pair in pairs:
            total = sum(per_target[target][metric][pair].square() for target in per_target)
            joint[metric][pair] = total.sqrt()
    return {"per_projection": per_target, "joint": joint}


def select_action_from_distance_table(
    table: Mapping[str, Any],
    metric: str,
    scope: str,
) -> Dict[str, Pair]:
    if scope == "per_projection":
        return {
            target: select_minimum_pair(metrics[metric])
            for target, metrics in table["per_projection"].items()
        }
    if scope == "joint":
        pair = select_minimum_pair(table["joint"][metric])
        return {target: pair for target in table["per_projection"]}
    raise ValueError("scope must be per_projection or joint")


def distance_table_to_json(table: Mapping[str, Any]) -> Dict[str, Any]:
    """Serialize every distance metric without losing its declared scope."""

    def values_json(values: Mapping[Pair, torch.Tensor]) -> Dict[str, float]:
        return {
            "%d-%d" % pair: float(value.detach().cpu())
            for pair, value in sorted(values.items())
        }

    return {
        "per_projection": {
            target: {metric: values_json(values) for metric, values in metrics.items()}
            for target, metrics in table["per_projection"].items()
        },
        "joint": {
            metric: values_json(values) for metric, values in table["joint"].items()
        },
    }


def snapshot_in_dtype(
    snapshot: FirstMergeSnapshot, dtype: torch.dtype
) -> FirstMergeSnapshot:
    """Copy a factor snapshot into a declared audit precision.

    Counterfactual gauges and distance rankings are algebraic operations on a
    frozen checkpoint.  Computing them in float64 prevents a solve performed
    even with the identity matrix from becoming part of the structural rule.
    The live float32 factorized-network re-execution is audited separately and
    is never used to choose or score a canonical merge action.
    """

    return FirstMergeSnapshot(
        targets={
            target: FactorSnapshot(
                shared_a=state.shared_a.detach().to(dtype=dtype).clone(),
                layer_b=state.layer_b.detach().to(dtype=dtype).clone(),
                running_b=state.running_b.detach().to(dtype=dtype).clone(),
                running_count=state.running_count,
                target_name=state.target_name,
            )
            for target, state in snapshot.targets.items()
        },
        candidate_mode=snapshot.candidate_mode,
        decision_scope=snapshot.decision_scope,
        metadata=dict(snapshot.metadata),
    )


def _metric_error_and_scale(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    metric: str,
) -> Tuple[float, float]:
    errors: List[float] = []
    scales: List[float] = []
    for target in reference["per_projection"]:
        reference_values = reference["per_projection"][target][metric]
        candidate_values = candidate["per_projection"][target][metric]
        for pair, value in reference_values.items():
            errors.append(float((value - candidate_values[pair]).abs().detach().cpu()))
            scales.append(float(value.abs().detach().cpu()))
    for pair, value in reference["joint"][metric].items():
        errors.append(
            float((value - candidate["joint"][metric][pair]).abs().detach().cpu())
        )
        scales.append(float(value.abs().detach().cpu()))
    return max(errors, default=0.0), max(scales, default=0.0)


def gauge_distance_invariance_audit(
    snapshot: FirstMergeSnapshot,
    gauge: NamedGauge,
    modes: Sequence[str],
    projected_covariances: Optional[Mapping[str, torch.Tensor]] = None,
    per_layer_projected_covariances: Optional[Mapping[str, torch.Tensor]] = None,
    lora_scalings: Optional[Mapping[str, float]] = None,
    atol: float = 5e-5,
    rtol: float = 5e-5,
) -> Dict[str, Any]:
    """Check metric invariance after transforming both factors and moments."""

    reference_snapshot = snapshot_in_dtype(snapshot, torch.float64)
    transformed = apply_snapshot_gauges(reference_snapshot, gauge.matrices)
    transformed_pooled: Optional[Dict[str, torch.Tensor]] = None
    if projected_covariances is not None:
        projected_covariances = {
            target: covariance.detach().to(dtype=torch.float64)
            for target, covariance in projected_covariances.items()
        }
        transformed_pooled = {
            target: transform_projected_covariance(
                covariance, gauge.matrices[target]
            )
            for target, covariance in projected_covariances.items()
        }
    transformed_per_layer: Optional[Dict[str, torch.Tensor]] = None
    if per_layer_projected_covariances is not None:
        per_layer_projected_covariances = {
            target: covariances.detach().to(dtype=torch.float64)
            for target, covariances in per_layer_projected_covariances.items()
        }
        transformed_per_layer = {
            target: torch.stack(
                [
                    transform_projected_covariance(covariance, gauge.matrices[target])
                    for covariance in covariances
                ]
            )
            for target, covariances in per_layer_projected_covariances.items()
        }
    orthogonal = all(
        torch.allclose(
            matrix.transpose(0, 1) @ matrix,
            torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device),
            atol=1e-8,
            rtol=1e-8,
        )
        for matrix in gauge.matrices.values()
    )
    records: Dict[str, Any] = {}
    all_expected_pass = True
    for mode in modes:
        reference_table = distance_table(
            reference_snapshot,
            mode,
            projected_covariances,
            per_layer_projected_covariances,
            lora_scalings,
        )
        transformed_table = distance_table(
            transformed,
            mode,
            transformed_pooled,
            transformed_per_layer,
            lora_scalings,
        )
        metric_records: Dict[str, Any] = {}
        for metric in sorted(reference_table["joint"]):
            error, scale = _metric_error_and_scale(
                reference_table, transformed_table, metric
            )
            expected_invariant = (
                metric != "running_raw_B_selection_proxy" or orthogonal
            )
            passed: Optional[bool] = None
            if expected_invariant:
                passed = error <= atol + rtol * scale
                all_expected_pass = all_expected_pass and passed
            metric_records[metric] = {
                "max_abs_error": error,
                "max_abs_reference": scale,
                "expected_invariant": expected_invariant,
                "invariance_pass": passed,
            }
        records[mode] = metric_records
    return {
        "gauge_is_orthogonal_condition_one_sanity": orthogonal,
        "modes": records,
        "all_expected_invariances_pass": all_expected_pass,
    }


def _snapshot_product_errors(
    reference: FirstMergeSnapshot,
    transformed: FirstMergeSnapshot,
) -> Dict[str, float]:
    return {
        target: float(
            (
                effective_updates(state.shared_a, state.layer_b)
                - effective_updates(
                    transformed.targets[target].shared_a,
                    transformed.targets[target].layer_b,
                )
            )
            .abs()
            .max()
            .detach()
            .cpu()
        )
        for target, state in reference.targets.items()
    }


def _snapshot_product_scales(snapshot: FirstMergeSnapshot) -> Dict[str, float]:
    return {
        target: float(
            effective_updates(state.shared_a, state.layer_b)
            .abs()
            .max()
            .detach()
            .cpu()
        )
        for target, state in snapshot.targets.items()
    }


def plan_prevalidation_decisions(
    snapshot: FirstMergeSnapshot,
    gauges: Sequence[NamedGauge],
    modes: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Pair]]]:
    """Select every raw-B action before validation metrics are available."""

    records: List[Dict[str, Any]] = []
    actions: Dict[str, Dict[str, Pair]] = {}
    audit_snapshot = snapshot_in_dtype(snapshot, torch.float64)
    for named in gauges:
        transformed = apply_snapshot_gauges(audit_snapshot, named.matrices)
        gauge_record: Dict[str, Any] = {
            "gauge": named.to_dict(include_matrices=True),
            "product_max_abs_error": _snapshot_product_errors(
                audit_snapshot, transformed
            ),
            "product_max_abs_reference": _snapshot_product_scales(audit_snapshot),
            "modes": {},
            "selection_inputs": "raw_running_average_B_only",
            "selection_audit_dtype": "float64",
            "validation_used_for_selection": False,
        }
        for mode in modes:
            table = distance_table(transformed, mode)
            mode_record: Dict[str, Any] = {
                # This deliberately includes raw and effective distances: only
                # the raw running-B field below is used to choose an action.
                "distance_table": distance_table_to_json(table),
                "decisions": {},
            }
            for scope in ("per_projection", "joint"):
                action = select_action_from_distance_table(
                    table, "running_raw_B_selection_proxy", scope
                )
                key = action_key(action)
                actions[key] = action
                mode_record["decisions"][scope] = {
                    "action": action_to_json(action),
                    "action_key": key,
                }
            gauge_record["modes"][mode] = mode_record
        records.append(gauge_record)
    return records, actions


def candidate_family_actions(
    targets: Sequence[str],
    num_layers: int,
    modes: Sequence[str],
) -> Dict[str, Dict[str, Pair]]:
    """Union of all single-target and common-pair actions requested by modes."""

    pairs: List[Pair] = []
    seen = set()
    for mode in modes:
        for pair in candidate_pairs(num_layers, mode):
            if pair not in seen:
                pairs.append(pair)
                seen.add(pair)
    actions: Dict[str, Dict[str, Pair]] = {}
    for pair in pairs:
        joint = {target: pair for target in targets}
        actions[action_key(joint)] = joint
        for target in targets:
            single = {target: pair}
            actions[action_key(single)] = single
    return actions


def evaluate_action_set(
    handle: RobertaASLoRAHandle,
    dataloader: Iterable[Mapping[str, Any]],
    device: torch.device,
    actions: Mapping[str, Mapping[str, Pair]],
) -> Dict[str, EvaluationResult]:
    results: Dict[str, EvaluationResult] = {}
    for key, action in sorted(actions.items()):
        with temporary_target_action(handle, action):
            results[key] = evaluate_classifier(handle.model, dataloader, device)
    return results


def action_result_json(
    action: Mapping[str, Pair],
    result: EvaluationResult,
    baseline: EvaluationResult,
) -> Dict[str, Any]:
    return {
        "action": action_to_json(action),
        "metrics": result.metrics(),
        "delta_from_unmerged": {
            "loss": result.loss - baseline.loss,
            "accuracy": result.accuracy - baseline.accuracy,
            "f1": result.f1 - baseline.f1,
        },
    }


def validation_loss_oracles(
    action_results: Mapping[str, EvaluationResult],
    actions: Mapping[str, Mapping[str, Pair]],
    modes: Sequence[str],
    num_layers: int,
    targets: Sequence[str],
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for mode in modes:
        pairs = set(candidate_pairs(num_layers, mode))
        families: Dict[str, List[str]] = {"joint_same_pair": []}
        for target in targets:
            families["%s_only" % target] = []
        for key, action in actions.items():
            if len(action) == len(targets):
                unique = set(action.values())
                if len(unique) == 1 and next(iter(unique)) in pairs:
                    families["joint_same_pair"].append(key)
            elif len(action) == 1:
                target, pair = next(iter(action.items()))
                if target in targets and pair in pairs:
                    families["%s_only" % target].append(key)
        output[mode] = {}
        for family, keys in families.items():
            if not keys:
                continue
            best = min(keys, key=lambda key: (action_results[key].loss, key))
            output[mode][family] = {
                "action_key": best,
                "action": action_to_json(actions[best]),
                "validation_metrics": action_results[best].metrics(),
                "oracle_scope": "exact_over_declared_%s_candidate_family" % family,
            }
    return output


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_deterministic_execution(enabled: bool = True) -> Dict[str, Any]:
    """Configure the primary action audit for repeatable inference.

    CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA context is
    created.  The runner calls this function before moving the model to CUDA.
    Disabling TF32 also removes an avoidable source of reassociation drift.
    """

    requested = bool(enabled)
    if requested:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(requested)
    torch.backends.cudnn.deterministic = requested
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    return {
        "requested": requested,
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            getattr(torch.backends.cuda.matmul, "allow_tf32", False)
        ),
        "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
    }


def require_hf() -> Tuple[Any, Any]:
    """Import optional training dependencies with an actionable error."""

    try:
        import datasets
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "ASLoRA MRPC training requires the optional 'aslora' dependencies; "
            "install the project with pip install -e \".[aslora]\""
        ) from error
    return transformers, datasets


def dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    for distribution in ("transformers", "datasets", "tokenizers", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def make_optimizer(model: nn.Module, config: ASLoRAMRPCConfig) -> torch.optim.Optimizer:
    """Use standard AdamW decay grouping and record that convention."""

    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lower = name.lower()
        if name.endswith("bias") or "layernorm.weight" in lower or "layer_norm.weight" in lower:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )


def optimizer_steps_per_epoch(micro_batches: int, gradient_accumulation_steps: int) -> int:
    if micro_batches < 1 or gradient_accumulation_steps < 1:
        raise ValueError("micro_batches and accumulation must be positive")
    return int(math.ceil(micro_batches / float(gradient_accumulation_steps)))


def timeline_provenance(config: ASLoRAMRPCConfig, train_micro_batches: int) -> Dict[str, Any]:
    steps_per_epoch = optimizer_steps_per_epoch(
        train_micro_batches, config.gradient_accumulation_steps
    )
    first = config.first_merge_step
    return {
        "optimizer_step_indexing": "one_based",
        "running_average_observation": "after_each_optimizer_update",
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "train_micro_batches_per_epoch": train_micro_batches,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "planned_total_optimizer_steps": steps_per_epoch * config.epochs,
        "first_merge_optimizer_step": first,
        "first_merge_epoch_one_based": (first - 1) // steps_per_epoch + 1,
        "first_merge_step_within_epoch_one_based": (first - 1) % steps_per_epoch + 1,
        "condition": "t>Ts and (t-Ts)%m==0",
    }


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    optimizer_step: int,
    micro_batches_seen: int,
    config: ASLoRAMRPCConfig,
    metadata: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    payload = {
        "format": "aslora-roberta-mrpc-pre-first-merge-v1",
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "optimizer_step": int(optimizer_step),
        "micro_batches_seen": int(micro_batches_seen),
        "config": config.to_dict(),
        "metadata": dict(metadata),
        "loader_note": (
            "Instantiate the recorded RoBERTa model, call "
            "attach_aslora_to_roberta_classifier with the recorded rank/alpha, "
            "then load model_state_dict strictly."
        ),
    }
    torch.save(payload, destination)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(str(temporary), str(destination))


def runtime_provenance(
    config: ASLoRAMRPCConfig,
    model: nn.Module,
    tokenizer: Any,
    dataset: Mapping[str, Any],
    timeline: Mapping[str, Any],
) -> Dict[str, Any]:
    resolved_revision = getattr(model.config, "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    device_name: Optional[str] = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "format": "aslora-mrpc-provenance-v1",
        "config": config.to_dict(),
        "paper_evidence": dict(PAPER_EVIDENCE),
        "source_ambiguities": dict(ASLORA_SOURCE_NOTES),
        "model": {
            "requested_name": config.model_name,
            "requested_revision": config.model_revision,
            "local_files_only": config.local_files_only,
            "resolved_commit_hash": resolved_revision,
            "tokenizer_resolved_commit_hash": tokenizer_revision,
            "architecture": model.__class__.__name__,
        },
        "dataset": {
            "name": config.dataset_name,
            "config": config.dataset_config,
            "requested_revision": config.dataset_revision,
            "source": (
                "local_parquet"
                if config.train_parquet is not None
                else "huggingface_glue_mrpc"
            ),
            "train_parquet": config.train_parquet,
            "validation_parquet": config.validation_parquet,
            "train_fingerprint": getattr(dataset["train"], "_fingerprint", None),
            "validation_fingerprint": getattr(dataset["validation"], "_fingerprint", None),
            "train_examples": len(dataset["train"]),
            "validation_examples": len(dataset["validation"]),
        },
        "versions": dependency_versions(),
        "hardware": {
            "platform": platform.platform(),
            "requested_device": config.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu": device_name,
        },
        "seed": config.seed,
        "timeline": dict(timeline),
        "optimizer": {
            "name": "torch.optim.AdamW",
            "decay_grouping": "bias_and_layernorm_no_decay",
            "linear_scheduler": True,
            "warmup_ratio": config.warmup_ratio,
            "max_grad_norm": config.max_grad_norm,
        },
        "decision_protocol": {
            "primary_scope": "per_projection",
            "sensitivity_scope": "joint_root_sum_of_squares",
            "candidate_modes": list(config.candidate_modes),
            "gauge_generation_uses_validation": False,
            "candidate_evaluation": "same_complete_MRPC_validation_split",
            "validation_loss_oracle": "post_hoc_baseline_only_not_used_for_gauge_selection",
            "running_B_metrics": "selection_proxies_not_current_action_costs",
            "current_action_local_metric": (
                "scaled_RMS_LoRA_output_disturbance_using_current_B_and_"
                "lower_layer_specific_premerge_activation_moment"
            ),
            "current_action_network_metric": (
                "temporary_parameter_tie_then_complete_validation_loss_accuracy_f1"
            ),
        },
        "undefined_paper_fields": {
            "update_ratio_lambda": config.update_ratio_record_only,
            "handling": "recorded_not_implemented_because_operation_is_unspecified",
        },
    }
