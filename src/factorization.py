"""Planted depth-factorization benchmark and routing objectives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .metrics import recovery_metrics
from .patterns import assignment_matrix, make_assignment


@dataclass
class FactorizationConfig:
    depth: int = 12
    num_bases: int = 4
    dimension: int = 128
    pattern: str = "random_balanced"
    method: str = "soft"
    init: str = "neutral"
    steps: int = 3000
    lr: float = 3e-2
    seed: int = 0
    observation_noise: float = 0.0
    basis_condition: float = 1.0
    tau_start: float = 1.0
    tau_end: float = 1.0
    route_strength: float = 0.0
    balance_strength: float = 0.0
    basis_repulsion: float = 0.0
    log_every: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_planted_problem(config: FactorizationConfig, device: torch.device):
    """Construct affinely independent planted bases and observed layer weights."""
    generator = torch.Generator(device="cpu").manual_seed(10_000 + config.seed)
    raw = torch.randn(config.dimension, config.num_bases, generator=generator)
    q, _ = torch.linalg.qr(raw, mode="reduced")
    scales = torch.logspace(
        0.0,
        math.log10(max(config.basis_condition, 1.0)),
        config.num_bases,
    )
    bases = (q.transpose(0, 1) * scales[:, None]).to(device)
    ids = make_assignment(config.pattern, config.depth, config.num_bases, 20_000 + config.seed).to(device)
    alpha = assignment_matrix(ids, config.num_bases).to(device)
    observed = alpha @ bases
    if config.observation_noise > 0:
        noise = torch.randn(observed.shape, generator=generator).to(device)
        observed = observed + config.observation_noise * observed.std() * noise
    return ids, alpha, bases, observed


class SoftDepthFactorization(nn.Module):
    def __init__(self, config: FactorizationConfig, observed: torch.Tensor, true_bases: torch.Tensor):
        super().__init__()
        self.config = config
        self.register_buffer("observed", observed)
        self.logits = nn.Parameter(torch.empty(config.depth, config.num_bases, device=observed.device))
        self.bases = nn.Parameter(torch.empty(config.num_bases, config.dimension, device=observed.device))
        self._initialize(observed, true_bases)

    def _initialize(self, observed: torch.Tensor, true_bases: torch.Tensor) -> None:
        with torch.no_grad():
            if self.config.init == "neutral":
                self.logits.zero_()
                self.logits.add_(0.01 * torch.randn_like(self.logits))
                self.bases.normal_(0.0, 1.0 / math.sqrt(self.config.dimension))
            elif self.config.init == "random":
                self.logits.normal_(0.0, 1.0)
                self.bases.normal_(0.0, 1.0 / math.sqrt(self.config.dimension))
            elif self.config.init == "observed":
                self.logits.zero_()
                self.logits.add_(0.01 * torch.randn_like(self.logits))
                picks = torch.linspace(0, len(observed) - 1, self.config.num_bases).round().long()
                self.bases.copy_(observed[picks])
            elif self.config.init == "warm":
                self.logits.zero_()
                self.logits.add_(0.01 * torch.randn_like(self.logits))
                perm = torch.randperm(self.config.num_bases, device=observed.device)
                self.bases.copy_(true_bases[perm] + 0.05 * torch.randn_like(true_bases))
            elif self.config.init == "pattern_contiguous":
                ids = make_assignment("contiguous", self.config.depth, self.config.num_bases).to(observed.device)
                self.logits.fill_(-2.0)
                self.logits.scatter_(1, ids[:, None], 2.0)
                self.bases.normal_(0.0, 1.0 / math.sqrt(self.config.dimension))
            elif self.config.init == "pattern_cycle":
                ids = make_assignment("cycle", self.config.depth, self.config.num_bases).to(observed.device)
                self.logits.fill_(-2.0)
                self.logits.scatter_(1, ids[:, None], 2.0)
                self.bases.normal_(0.0, 1.0 / math.sqrt(self.config.dimension))
            else:
                raise ValueError("unknown initialization: %s" % self.config.init)

    def probabilities(self, temperature: float, hard: bool = False) -> torch.Tensor:
        soft = F.softmax(self.logits / temperature, dim=-1)
        if not hard:
            return soft
        indices = soft.argmax(dim=-1)
        one_hot = F.one_hot(indices, num_classes=self.config.num_bases).float()
        return one_hot + soft - soft.detach()

    def forward(self, temperature: float, hard: bool = False) -> torch.Tensor:
        return self.probabilities(temperature, hard=hard) @ self.bases


def router_regularizer(probabilities: torch.Tensor, config: FactorizationConfig) -> torch.Tensor:
    p = probabilities.clamp_min(1e-12)
    conditional_entropy = -(p * p.log()).sum(dim=-1).mean()
    marginal = p.mean(dim=0)
    marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
    # Minimizing conditional entropy produces vertices; maximizing marginal
    # entropy prevents the trivial all-layers-to-one-basis solution.
    return config.route_strength * conditional_entropy - config.balance_strength * marginal_entropy


def basis_repulsion_loss(bases: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(bases, dim=-1)
    gram = normalized @ normalized.transpose(0, 1)
    eye = torch.eye(len(bases), device=bases.device)
    return ((gram - eye) ** 2).sum() / max(1, len(bases) * (len(bases) - 1))


def temperature_at(step: int, config: FactorizationConfig) -> float:
    if config.steps <= 1:
        return config.tau_end
    fraction = step / float(config.steps - 1)
    return config.tau_start * (config.tau_end / config.tau_start) ** fraction


def run_factorization(config: FactorizationConfig, device: Optional[str] = None) -> Dict[str, object]:
    seed_everything(config.seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    true_ids, _, true_bases, observed = make_planted_problem(config, resolved_device)

    if config.method == "oracle":
        probabilities = assignment_matrix(true_ids, config.num_bases).to(resolved_device)
        fitted = torch.linalg.lstsq(probabilities, observed).solution
        reconstruction = probabilities @ fitted
        losses = []
    elif config.method == "kmeans":
        # Deterministic Lloyd iterations with k-means++-like farthest starts.
        centers = [observed[0]]
        for _ in range(1, config.num_bases):
            distances = torch.stack([((observed - c) ** 2).sum(dim=1) for c in centers]).min(dim=0).values
            centers.append(observed[distances.argmax()])
        fitted = torch.stack(centers)
        ids = torch.zeros(config.depth, dtype=torch.long, device=resolved_device)
        for _ in range(100):
            distances = torch.cdist(observed, fitted)
            new_ids = distances.argmin(dim=1)
            if torch.equal(new_ids, ids) and _ > 0:
                break
            ids = new_ids
            for j in range(config.num_bases):
                if (ids == j).any():
                    fitted[j] = observed[ids == j].mean(dim=0)
        probabilities = F.one_hot(ids, num_classes=config.num_bases).float()
        reconstruction = probabilities @ fitted
        losses = []
    else:
        model = SoftDepthFactorization(config, observed, true_bases)
        parameters = list(model.parameters())
        if config.method == "soft_frozen":
            with torch.no_grad():
                model.bases.copy_(true_bases)
            model.bases.requires_grad_(False)
            parameters = [model.logits]
        optimizer = torch.optim.Adam(parameters, lr=config.lr)
        losses = []
        for step in range(config.steps):
            tau = temperature_at(step, config)
            hard = config.method in {"straight_through", "vertex_hard"}
            reconstruction = model(tau, hard=hard)
            probabilities_soft = model.probabilities(tau, hard=False)
            reconstruction_loss = F.mse_loss(reconstruction, observed)
            regularizer = router_regularizer(probabilities_soft, config)
            if config.basis_repulsion:
                regularizer = regularizer + config.basis_repulsion * basis_repulsion_loss(model.bases)
            loss = reconstruction_loss + regularizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if config.log_every and (step % config.log_every == 0 or step == config.steps - 1):
                losses.append({
                    "step": step,
                    "loss": float(loss.detach()),
                    "reconstruction_loss": float(reconstruction_loss.detach()),
                    "temperature": tau,
                })
        final_hard = config.method in {"straight_through", "vertex_hard"}
        probabilities = model.probabilities(config.tau_end, hard=final_hard).detach()
        reconstruction = probabilities @ model.bases.detach()
        fitted = model.bases.detach()

    relative_error = float(torch.linalg.norm(reconstruction - observed) / torch.linalg.norm(observed))
    result: Dict[str, object] = {
        "config": asdict(config),
        "device": str(resolved_device),
        "relative_reconstruction_error": relative_error,
        "mse": float(F.mse_loss(reconstruction, observed)),
        "probabilities": probabilities.detach().cpu().tolist(),
        "true_assignment": true_ids.detach().cpu().tolist(),
        "history": losses,
    }
    result.update(recovery_metrics(true_ids, probabilities))
    return result
