"""Permutation-invariant metrics for planted sharing recovery."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Tuple

import torch


def best_permutation_accuracy(true_ids: torch.Tensor, pred_ids: torch.Tensor, k: int) -> Tuple[float, Tuple[int, ...]]:
    """Clustering accuracy after the best basis-label permutation.

    Brute force is intentional: experiments use K <= 6, and avoiding a SciPy
    dependency keeps the artifact runnable on minimal GPU images.
    """
    true_cpu = true_ids.detach().cpu().long()
    pred_cpu = pred_ids.detach().cpu().long()
    best_acc = -1.0
    best_perm: Tuple[int, ...] = tuple(range(k))
    for perm in itertools.permutations(range(k)):
        mapped = torch.tensor([perm[int(v)] for v in pred_cpu], dtype=torch.long)
        acc = float((mapped == true_cpu).float().mean())
        if acc > best_acc:
            best_acc = acc
            best_perm = tuple(int(v) for v in perm)
    return best_acc, best_perm


def adjusted_rand_index(true_ids: torch.Tensor, pred_ids: torch.Tensor) -> float:
    """Adjusted Rand index without sklearn."""
    a = true_ids.detach().cpu().long()
    b = pred_ids.detach().cpu().long()
    n = int(a.numel())
    if n < 2:
        return 1.0
    ua = torch.unique(a)
    ub = torch.unique(b)
    table = torch.zeros(len(ua), len(ub), dtype=torch.float64)
    for i, va in enumerate(ua):
        for j, vb in enumerate(ub):
            table[i, j] = ((a == va) & (b == vb)).sum()

    def choose2(x: torch.Tensor) -> torch.Tensor:
        return x * (x - 1.0) / 2.0

    sum_cells = choose2(table).sum()
    sum_rows = choose2(table.sum(dim=1)).sum()
    sum_cols = choose2(table.sum(dim=0)).sum()
    total = n * (n - 1.0) / 2.0
    expected = sum_rows * sum_cols / total
    maximum = 0.5 * (sum_rows + sum_cols)
    denom = maximum - expected
    if abs(float(denom)) < 1e-12:
        return 1.0 if torch.equal(a[:, None] == a[None, :], b[:, None] == b[None, :]) else 0.0
    return float((sum_cells - expected) / denom)


def pairwise_graph_accuracy(true_ids: torch.Tensor, pred_ids: torch.Tensor) -> float:
    """Accuracy of the basis-label-invariant same-basis relation."""
    truth = true_ids[:, None] == true_ids[None, :]
    pred = pred_ids[:, None] == pred_ids[None, :]
    mask = ~torch.eye(len(true_ids), dtype=torch.bool, device=true_ids.device)
    return float((truth[mask] == pred[mask]).float().mean())


def normalized_entropy(probabilities: torch.Tensor) -> float:
    k = probabilities.shape[-1]
    if k == 1:
        return 0.0
    p = probabilities.clamp_min(1e-12)
    entropy = -(p * p.log()).sum(dim=-1).mean()
    value = entropy / torch.log(torch.tensor(float(k), device=p.device))
    return float(value.detach())


def recovery_metrics(true_ids: torch.Tensor, probabilities: torch.Tensor) -> Dict[str, float]:
    pred_ids = probabilities.argmax(dim=-1)
    acc, _ = best_permutation_accuracy(true_ids, pred_ids, probabilities.shape[-1])
    return {
        "assignment_accuracy": acc,
        "adjusted_rand": adjusted_rand_index(true_ids, pred_ids),
        "graph_accuracy": pairwise_graph_accuracy(true_ids, pred_ids),
        "normalized_entropy": normalized_entropy(probabilities),
        "num_used": float(torch.unique(pred_ids).numel()),
    }


def coassignment(probabilities: torch.Tensor) -> torch.Tensor:
    """Soft co-assignment matrix, invariant to permutations of basis labels."""
    return probabilities @ probabilities.transpose(0, 1)
