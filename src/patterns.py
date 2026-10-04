"""Planted cross-layer sharing graphs and deterministic utilities."""

from __future__ import annotations

import random
from typing import List

import torch


def make_assignment(pattern: str, depth: int, num_bases: int, seed: int = 0) -> torch.Tensor:
    """Return a length-``depth`` vector of planted basis identifiers.

    The suite deliberately includes schedules with the same number of bases but
    different topology. This prevents a method from looking successful merely
    because its initialization already encodes one particular schedule.
    """
    if depth < num_bases or num_bases < 1:
        raise ValueError("need depth >= num_bases >= 1")

    if pattern == "cycle":
        values = [i % num_bases for i in range(depth)]
    elif pattern == "contiguous":
        values = [min(num_bases - 1, (i * num_bases) // depth) for i in range(depth)]
    elif pattern == "palindrome":
        if num_bases == 1:
            values = [0] * depth
        else:
            period = 2 * num_bases - 2
            values = []
            for i in range(depth):
                t = i % period
                values.append(t if t < num_bases else period - t)
    elif pattern == "random_balanced":
        rng = random.Random(seed)
        values = [i % num_bases for i in range(depth)]
        rng.shuffle(values)
    elif pattern == "random":
        rng = random.Random(seed)
        values = [rng.randrange(num_bases) for _ in range(depth)]
        # Make every basis observable. This is a recoverability benchmark, not
        # a model-selection benchmark with an unknown effective K.
        for j in range(num_bases):
            values[j] = j
        rng.shuffle(values)
    elif pattern == "imbalanced":
        # One common basis plus rare specialists. Each specialist appears once.
        values = [0] * depth
        if num_bases > 1:
            positions = torch.linspace(0, depth - 1, num_bases).round().long().tolist()
            for j, pos in enumerate(positions):
                values[pos] = j
    else:
        raise ValueError("unknown pattern: %s" % pattern)
    return torch.tensor(values, dtype=torch.long)


def assignment_matrix(ids: torch.Tensor, num_bases: int) -> torch.Tensor:
    """Convert integer assignments to a one-hot layer-by-basis matrix."""
    if ids.ndim != 1:
        raise ValueError("ids must be one-dimensional")
    return torch.nn.functional.one_hot(ids, num_classes=num_bases).float()


def available_patterns() -> List[str]:
    return ["cycle", "contiguous", "palindrome", "random_balanced", "random", "imbalanced"]

