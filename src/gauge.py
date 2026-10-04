"""Exact gauge counterfactuals and softmax-saturation experiments."""

from __future__ import annotations

import math
from typing import Dict, List

import torch
from torch.nn import functional as F

from .metrics import coassignment, normalized_entropy, recovery_metrics
from .patterns import assignment_matrix, make_assignment


def _softmax_factor_step(logits: torch.Tensor, bases: torch.Tensor, target: torch.Tensor, lr: float):
    logits_parameter = torch.nn.Parameter(logits.clone())
    bases_parameter = torch.nn.Parameter(bases.clone())
    probabilities = F.softmax(logits_parameter, dim=-1)
    effective = probabilities @ bases_parameter
    loss = 0.5 * ((effective - target) ** 2).sum()
    gradients = torch.autograd.grad(loss, (logits_parameter, bases_parameter))
    with torch.no_grad():
        logits_after = logits_parameter - lr * gradients[0]
        bases_after = bases_parameter - lr * gradients[1]
        effective_after = F.softmax(logits_after, dim=-1) @ bases_after
    return {
        "loss": float(loss.detach()),
        "logit_gradient_norm": float(gradients[0].norm()),
        "basis_gradient_norm": float(gradients[1].norm()),
        "effective_after": effective_after,
    }


def run_gauge_counterfactual(seed: int = 0, depth: int = 8, num_bases: int = 3, dimension: int = 16) -> Dict[str, object]:
    """Construct identical effective weights with different router coordinates."""
    torch.manual_seed(seed)
    raw = torch.rand(depth, num_bases) + 0.2
    alpha = raw / raw.sum(dim=-1, keepdim=True)
    bases = torch.randn(num_bases, dimension)
    effective = alpha @ bases

    # A non-permutation row-stochastic gauge.  Convex mixing preserves router
    # feasibility; the inverse is absorbed by the unconstrained basis vectors.
    random_rows = torch.rand(num_bases, num_bases)
    random_rows = random_rows / random_rows.sum(dim=-1, keepdim=True)
    gauge = 0.65 * torch.eye(num_bases) + 0.35 * random_rows
    transformed_alpha = alpha @ gauge
    transformed_bases = torch.linalg.solve(gauge, bases)
    transformed_effective = transformed_alpha @ transformed_bases

    target = effective + 0.1 * torch.randn_like(effective)
    first_step = _softmax_factor_step(alpha.log(), bases, target, lr=1e-2)
    second_step = _softmax_factor_step(transformed_alpha.log(), transformed_bases, target, lr=1e-2)

    original_graph = alpha.argmax(dim=-1)
    transformed_graph = transformed_alpha.argmax(dim=-1)
    return {
        "seed": seed,
        "depth": depth,
        "num_bases": num_bases,
        "effective_max_abs_error": float((effective - transformed_effective).abs().max()),
        "original_entropy": normalized_entropy(alpha),
        "transformed_entropy": normalized_entropy(transformed_alpha),
        "argmax_agreement": float((original_graph == transformed_graph).float().mean()),
        "coassignment_relative_change": float(
            torch.linalg.norm(coassignment(alpha) - coassignment(transformed_alpha))
            / torch.linalg.norm(coassignment(alpha))
        ),
        "initial_loss_difference": abs(first_step["loss"] - second_step["loss"]),
        "post_step_effective_relative_difference": float(
            torch.linalg.norm(first_step["effective_after"] - second_step["effective_after"])
            / torch.linalg.norm(first_step["effective_after"])
        ),
        "original_logit_gradient_norm": first_step["logit_gradient_norm"],
        "transformed_logit_gradient_norm": second_step["logit_gradient_norm"],
        "alpha": alpha.tolist(),
        "transformed_alpha": transformed_alpha.tolist(),
        "gauge": gauge.tolist(),
    }


def exact_argmax_flip_example() -> Dict[str, object]:
    alpha = torch.tensor([[0.1, 0.9], [0.3, 0.7], [0.4, 0.6], [0.9, 0.1]], dtype=torch.float64)
    gauge = torch.tensor([[0.9, 0.1], [0.4, 0.6]], dtype=torch.float64)
    bases = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.2, 3.0]], dtype=torch.float64)
    transformed_alpha = alpha @ gauge
    transformed_bases = torch.linalg.solve(gauge, bases)
    effective = alpha @ bases
    transformed_effective = transformed_alpha @ transformed_bases
    return {
        "alpha": alpha.tolist(),
        "transformed_alpha": transformed_alpha.tolist(),
        "argmax_before": alpha.argmax(dim=-1).tolist(),
        "argmax_after": transformed_alpha.argmax(dim=-1).tolist(),
        "effective_max_abs_error": float((effective - transformed_effective).abs().max()),
        "entropy_before": normalized_entropy(alpha.float()),
        "entropy_after": normalized_entropy(transformed_alpha.float()),
    }


def run_saturation_sweep(
    seed: int = 0,
    depth: int = 12,
    num_bases: int = 4,
    dimension: int = 64,
    steps: int = 2000,
) -> List[Dict[str, object]]:
    """Fit a wrong planted router with true frozen bases across margins/tau."""
    torch.manual_seed(seed)
    true_ids = make_assignment("cycle", depth, num_bases, seed=seed)
    wrong_ids = make_assignment("contiguous", depth, num_bases, seed=seed)
    true_alpha = assignment_matrix(true_ids, num_bases)
    bases = torch.randn(num_bases, dimension)
    q, _ = torch.linalg.qr(bases.transpose(0, 1), mode="reduced")
    bases = q.transpose(0, 1)
    target = true_alpha @ bases
    records: List[Dict[str, object]] = []
    for temperature in [0.25, 0.5, 1.0, 2.0]:
        for margin in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
            logits = torch.zeros(depth, num_bases)
            if margin == 0:
                logits.add_(0.01 * torch.randn_like(logits))
            else:
                logits.fill_(-margin)
                logits.scatter_(1, wrong_ids[:, None], margin)
            logits = torch.nn.Parameter(logits)
            optimizer = torch.optim.Adam([logits], lr=0.05)
            initial_gradient = None
            step_90 = None
            for step in range(steps):
                probabilities = F.softmax(logits / temperature, dim=-1)
                loss = F.mse_loss(probabilities @ bases, target)
                optimizer.zero_grad()
                loss.backward()
                if initial_gradient is None:
                    initial_gradient = float(logits.grad.norm())
                optimizer.step()
                # Bases are frozen in their planted order, so no label
                # alignment is needed here.  Avoid the factorial Hungarian
                # surrogate in the inner optimization loop.
                raw_accuracy = float((probabilities.detach().argmax(dim=-1) == true_ids).float().mean())
                if step_90 is None and raw_accuracy >= 0.90:
                    step_90 = step
            probabilities = F.softmax(logits / temperature, dim=-1).detach()
            records.append(
                {
                    "seed": seed,
                    "temperature": temperature,
                    "margin": margin,
                    "initial_gradient_norm": initial_gradient,
                    "step_to_90pct": step_90,
                    "final_mse": float(F.mse_loss(probabilities @ bases, target)),
                    **recovery_metrics(true_ids, probabilities),
                }
            )
    return records
