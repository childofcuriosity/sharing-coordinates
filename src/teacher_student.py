"""Functional teacher-student recovery experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn import functional as F

from .factorization import temperature_at
from .metrics import recovery_metrics
from .models import BasisResidualMLP, CausalBasisTransformer
from .patterns import make_assignment


@dataclass
class TeacherStudentConfig:
    architecture: str = "mlp"
    pattern: str = "random_balanced"
    router_init: str = "neutral"
    depth: int = 8
    num_bases: int = 4
    dimension: int = 64
    hidden_dimension: int = 128
    num_heads: int = 4
    vocab_size: int = 64
    sequence_length: int = 24
    train_examples: int = 4096
    test_examples: int = 1024
    batch_size: int = 128
    steps: int = 3000
    lr: float = 3e-3
    seed: int = 0
    tau_start: float = 1.0
    tau_end: float = 1.0
    hard: bool = False
    vertex_strength: float = 0.0
    usage_strength: float = 0.0
    hidden_weight: float = 0.0
    teacher_noise: float = 0.0
    log_every: int = 250


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _temperature(step: int, config: TeacherStudentConfig) -> float:
    if config.steps <= 1:
        return config.tau_end
    fraction = step / float(config.steps - 1)
    return config.tau_start * (config.tau_end / config.tau_start) ** fraction


def _build_model(config: TeacherStudentConfig, teacher: bool, device: torch.device):
    common = dict(
        dimension=config.dimension,
        hidden_dimension=config.hidden_dimension,
        depth=config.depth,
        num_bases=config.num_bases,
        router_init="planted" if teacher else config.router_init,
        router_trainable=not teacher,
        router_pattern=config.pattern if teacher else None,
        seed=(90_000 if teacher else 80_000) + config.seed,
    )
    if config.architecture == "mlp":
        model = BasisResidualMLP(**common)
    elif config.architecture == "transformer":
        model = CausalBasisTransformer(
            vocab_size=config.vocab_size,
            max_length=config.sequence_length,
            num_heads=config.num_heads,
            **common,
        )
    else:
        raise ValueError("unknown architecture: %s" % config.architecture)
    return model.to(device)


def _make_data(config: TeacherStudentConfig, device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(70_000 + config.seed)
    count = config.train_examples + config.test_examples
    if config.architecture == "mlp":
        data = torch.randn(count, config.dimension, generator=generator)
    else:
        # A mixture of local-copy and modular-arithmetic structure exercises
        # causal attention while remaining exactly reproducible.
        data = torch.randint(0, config.vocab_size, (count, config.sequence_length), generator=generator)
        for position in range(2, config.sequence_length):
            choose_copy = torch.rand(count, generator=generator) < 0.55
            derived = (data[:, position - 1] + 2 * data[:, position - 2] + position) % config.vocab_size
            data[choose_copy, position] = derived[choose_copy]
    return data[: config.train_examples].to(device), data[config.train_examples :].to(device)


def _distillation_loss(student_output, teacher_output, architecture: str) -> torch.Tensor:
    if architecture == "mlp":
        return F.mse_loss(student_output, teacher_output)
    teacher_probabilities = F.softmax(teacher_output.detach(), dim=-1)
    return F.kl_div(F.log_softmax(student_output, dim=-1), teacher_probabilities, reduction="batchmean") / student_output.shape[1]


def run_teacher_student(config: TeacherStudentConfig, device: Optional[str] = None) -> Dict[str, object]:
    _seed(config.seed)
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    teacher = _build_model(config, teacher=True, device=resolved).eval()
    student = _build_model(config, teacher=False, device=resolved).train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if config.teacher_noise:
        with torch.no_grad():
            for name, parameter in teacher.named_parameters():
                if "router" not in name:
                    parameter.add_(config.teacher_noise * parameter.std().clamp_min(1e-6) * torch.randn_like(parameter))

    train_data, test_data = _make_data(config, resolved)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(60_000 + config.seed)
    history = []
    for step in range(config.steps):
        indices = torch.randint(0, len(train_data), (config.batch_size,), generator=generator).to(resolved)
        batch = train_data[indices]
        with torch.no_grad():
            teacher_output, teacher_hidden, _ = teacher(batch, hard=True, return_hidden=True)
        temperature = _temperature(step, config)
        student_output, student_hidden, probabilities = student(
            batch,
            temperature=temperature,
            hard=config.hard,
            return_hidden=True,
        )
        task_loss = _distillation_loss(student_output, teacher_output, config.architecture)
        hidden_loss = sum(F.mse_loss(s, t) for s, t in zip(student_hidden, teacher_hidden)) / config.depth
        regularizer = student.router.regularizer(
            temperature,
            vertex_strength=config.vertex_strength,
            usage_strength=config.usage_strength,
        )
        loss = task_loss + config.hidden_weight * hidden_loss + regularizer
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        if config.log_every and (step % config.log_every == 0 or step == config.steps - 1):
            metrics = recovery_metrics(
                make_assignment(config.pattern, config.depth, config.num_bases, 90_000 + config.seed).to(resolved),
                student.router.probabilities(temperature, hard=False),
            )
            history.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "task_loss": float(task_loss.detach()),
                    "hidden_loss": float(hidden_loss.detach()),
                    "temperature": temperature,
                    **metrics,
                }
            )

    student.eval()
    with torch.no_grad():
        teacher_output, teacher_hidden, _ = teacher(test_data, hard=True, return_hidden=True)
        student_output, student_hidden, probabilities = student(
            test_data,
            temperature=config.tau_end,
            hard=config.hard,
            return_hidden=True,
        )
        test_task = _distillation_loss(student_output, teacher_output, config.architecture)
        test_hidden = sum(F.mse_loss(s, t) for s, t in zip(student_hidden, teacher_hidden)) / config.depth
    true_ids = make_assignment(config.pattern, config.depth, config.num_bases, 90_000 + config.seed).to(resolved)
    result: Dict[str, object] = {
        "config": asdict(config),
        "device": str(resolved),
        "test_task_loss": float(test_task),
        "test_hidden_loss": float(test_hidden),
        "probabilities": probabilities.detach().cpu().tolist(),
        "true_assignment": true_ids.detach().cpu().tolist(),
        "history": history,
        "student_parameters": sum(p.numel() for p in student.parameters()),
    }
    result.update(recovery_metrics(true_ids, probabilities))
    return result

