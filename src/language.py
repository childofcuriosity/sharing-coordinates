"""Byte-level WikiText language modeling for router-stability analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from .metrics import normalized_entropy
from .models import CausalBasisTransformer


@dataclass
class LanguageConfig:
    train_path: str = "data/wikitext-2/train.txt"
    validation_path: str = "data/wikitext-2/valid.txt"
    test_path: str = "data/wikitext-2/test.txt"
    router_init: str = "neutral"
    router_trainable: bool = True
    router_pattern: Optional[str] = None
    hard: bool = False
    depth: int = 12
    num_bases: int = 4
    dimension: int = 256
    hidden_dimension: int = 1024
    num_heads: int = 8
    sequence_length: int = 128
    batch_size: int = 32
    steps: int = 1500
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.1
    seed: int = 0
    tau_start: float = 1.0
    tau_end: float = 1.0
    vertex_strength: float = 0.0
    usage_strength: float = 0.0
    eval_batches: int = 100
    log_every: int = 250
    amp: bool = True
    checkpoint_load: Optional[str] = None
    checkpoint_save: Optional[str] = None


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_byte_stream(path: str, device: torch.device) -> torch.Tensor:
    data = Path(path).read_bytes()
    if len(data) < 1024:
        raise ValueError("corpus is unexpectedly small: %s" % path)
    return torch.tensor(list(data), dtype=torch.long, device=device)


def corpus_sha256(config: LanguageConfig) -> Dict[str, str]:
    """Hash each corpus split and their ordered combination.

    Checkpoints record content hashes rather than relying on paths, which may
    legitimately differ across machines.
    """

    hashes: Dict[str, str] = {}
    combined = hashlib.sha256()
    for split, path in (
        ("train", config.train_path),
        ("validation", config.validation_path),
        ("test", config.test_path),
    ):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        hashes[split] = digest
        combined.update(split.encode("utf-8"))
        combined.update(b"\0")
        combined.update(bytes.fromhex(digest))
    hashes["combined"] = combined.hexdigest()
    return hashes


def build_language_model(config: LanguageConfig) -> CausalBasisTransformer:
    """Construct the byte LM architecture described by ``config``."""

    return CausalBasisTransformer(
        vocab_size=256,
        max_length=config.sequence_length,
        dimension=config.dimension,
        hidden_dimension=config.hidden_dimension,
        num_heads=config.num_heads,
        depth=config.depth,
        num_bases=config.num_bases,
        router_init=config.router_init,
        router_trainable=config.router_trainable,
        router_pattern=config.router_pattern,
        seed=100_000 + config.seed,
    )


def save_language_checkpoint(
    path: str,
    model: CausalBasisTransformer,
    config: LanguageConfig,
    corpus_hashes: Mapping[str, str],
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save a portable model checkpoint with provenance metadata."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {
        name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
    }
    payload = {
        "format_version": 1,
        "config": asdict(config),
        "state_dict": state_dict,
        "corpus_sha256": dict(corpus_hashes),
        "extra": dict(extra or {}),
    }
    torch.save(payload, target)


def _torch_load_checkpoint(path: str, map_location: torch.device) -> Dict[str, Any]:
    # PyTorch 2.6 defaults to weights_only=True.  This checkpoint also contains
    # primitive provenance dictionaries, so request the full local payload.
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # torch<2.0 has no weights_only keyword.
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("language checkpoint must contain a dictionary")
    return payload


def load_language_checkpoint(
    path: str,
    device: Optional[str] = None,
    expected_corpus_sha256: Optional[Mapping[str, str]] = None,
) -> Tuple[CausalBasisTransformer, LanguageConfig, Dict[str, Any]]:
    """Load a checkpoint and optionally verify exact corpus contents."""

    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = _torch_load_checkpoint(path, resolved)
    required = {"config", "state_dict", "corpus_sha256"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError("language checkpoint is missing: %s" % ", ".join(missing))
    saved_config = payload["config"]
    if not isinstance(saved_config, dict):
        raise ValueError("checkpoint config must be a dictionary")
    known_fields = LanguageConfig.__dataclass_fields__
    config = LanguageConfig(**{key: value for key, value in saved_config.items() if key in known_fields})
    recorded_hashes = payload["corpus_sha256"]
    if not isinstance(recorded_hashes, dict):
        raise ValueError("checkpoint corpus_sha256 must be a dictionary")
    if expected_corpus_sha256 is not None:
        for split, expected in expected_corpus_sha256.items():
            actual = recorded_hashes.get(split)
            if actual != expected:
                raise ValueError(
                    "corpus SHA256 mismatch for %s: checkpoint=%s current=%s"
                    % (split, actual, expected)
                )
    model = build_language_model(config).to(resolved)
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = {
        "format_version": payload.get("format_version", 0),
        "corpus_sha256": dict(recorded_hashes),
        "extra": dict(payload.get("extra") or {}),
    }
    return model, config, metadata


def sample_batch(
    stream: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
) -> Tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        0,
        len(stream) - sequence_length,
        (batch_size,),
        generator=generator,
    ).to(stream.device)
    offsets = torch.arange(sequence_length, device=stream.device)
    inputs = stream[starts[:, None] + offsets[None]]
    targets = stream[starts[:, None] + offsets[None] + 1]
    return inputs, targets


def _temperature(step: int, config: LanguageConfig) -> float:
    if config.steps <= 1:
        return config.tau_end
    fraction = step / float(config.steps - 1)
    return config.tau_start * (config.tau_end / config.tau_start) ** fraction


def _learning_rate(step: int, config: LanguageConfig) -> float:
    if step < config.warmup_steps:
        return config.lr * (step + 1) / max(1, config.warmup_steps)
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps)
    return config.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(
    model: CausalBasisTransformer,
    stream: torch.Tensor,
    config: LanguageConfig,
    temperature: float,
    seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_loss = 0.0
    total_tokens = 0
    hard = config.hard
    for _ in range(config.eval_batches):
        inputs, targets = sample_batch(stream, config.batch_size, config.sequence_length, generator)
        logits = model(inputs, temperature=temperature, hard=hard)
        loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1), reduction="sum")
        total_loss += float(loss)
        total_tokens += targets.numel()
    model.train()
    return total_loss / total_tokens / math.log(2.0)


def run_language(config: LanguageConfig, device: Optional[str] = None) -> Dict[str, object]:
    _seed(config.seed)
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    corpus_hashes = corpus_sha256(config)
    train_stream = load_byte_stream(config.train_path, resolved)
    validation_stream = load_byte_stream(config.validation_path, resolved)
    test_stream = load_byte_stream(config.test_path, resolved)
    checkpoint_metadata: Optional[Dict[str, Any]] = None
    if config.checkpoint_load:
        model, saved_config, checkpoint_metadata = load_language_checkpoint(
            config.checkpoint_load,
            device=str(resolved),
            expected_corpus_sha256=corpus_hashes,
        )
        architecture_fields = (
            "depth",
            "num_bases",
            "dimension",
            "hidden_dimension",
            "num_heads",
            "sequence_length",
        )
        mismatches = [
            name
            for name in architecture_fields
            if getattr(config, name) != getattr(saved_config, name)
        ]
        if mismatches:
            raise ValueError(
                "checkpoint architecture differs from requested config: %s"
                % ", ".join(mismatches)
            )
        model.router.logits.requires_grad_(config.router_trainable)
    else:
        model = build_language_model(config).to(resolved)
    initial_probabilities = model.router.probabilities(config.tau_start, hard=False).detach().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    use_amp = bool(config.amp and resolved.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    generator = torch.Generator(device="cpu").manual_seed(110_000 + config.seed)
    history = []
    started = time.time()
    best_validation = float("inf")
    last_train_bpb: Optional[float] = None
    for step in range(config.steps):
        lr = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = lr
        inputs, targets = sample_batch(train_stream, config.batch_size, config.sequence_length, generator)
        temperature = _temperature(step, config)
        hard = config.hard
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(inputs, temperature=temperature, hard=hard)
            task_loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
            route_loss = model.router.regularizer(
                temperature,
                config.vertex_strength,
                config.usage_strength,
            )
            loss = task_loss + route_loss
        last_train_bpb = float(task_loss.detach()) / math.log(2.0)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if config.log_every and ((step + 1) % config.log_every == 0 or step == 0):
            validation_bpb = evaluate(
                model,
                validation_stream,
                config,
                temperature,
                120_000 + config.seed,
            )
            best_validation = min(best_validation, validation_bpb)
            probabilities = model.router.probabilities(temperature, hard=False).detach()
            history.append(
                {
                    "step": step + 1,
                    "train_bpb": last_train_bpb,
                    "validation_bpb": validation_bpb,
                    "best_validation_bpb": best_validation,
                    "temperature": temperature,
                    "lr": lr,
                    "router_entropy": normalized_entropy(probabilities),
                    "elapsed_seconds": time.time() - started,
                }
            )
            print(
                "step=%d train_bpb=%.3f val_bpb=%.3f H=%.3f time=%.1fs"
                % (
                    step + 1,
                    last_train_bpb,
                    validation_bpb,
                    normalized_entropy(probabilities),
                    time.time() - started,
                ),
                flush=True,
            )
    final_probabilities = model.router.probabilities(config.tau_end, hard=False).detach().cpu()
    if not history or history[-1]["step"] != config.steps:
        final_validation = evaluate(
            model,
            validation_stream,
            config,
            config.tau_end,
            120_000 + config.seed,
        )
        best_validation = min(best_validation, final_validation)
        history.append(
            {
                "step": config.steps,
                "train_bpb": last_train_bpb,
                "validation_bpb": final_validation,
                "best_validation_bpb": best_validation,
                "temperature": config.tau_end,
                "lr": _learning_rate(max(0, config.steps - 1), config),
                "router_entropy": normalized_entropy(final_probabilities),
                "elapsed_seconds": time.time() - started,
            }
        )
    test_bpb = evaluate(model, test_stream, config, config.tau_end, 130_000 + config.seed)
    if config.checkpoint_save:
        save_language_checkpoint(
            config.checkpoint_save,
            model,
            config,
            corpus_hashes,
            extra={
                "checkpoint_kind": "final_predeclared_training_step",
                "saved_at_step": config.steps,
                "best_validation_bpb": best_validation,
                "test_bpb": test_bpb,
                "elapsed_seconds": time.time() - started,
                "initial_probabilities": initial_probabilities.tolist(),
                "final_probabilities": final_probabilities.tolist(),
                "router_mean_l1_movement": float(
                    (final_probabilities - initial_probabilities)
                    .abs()
                    .sum(dim=-1)
                    .mean()
                ),
            },
        )
    return {
        "config": asdict(config),
        "device": str(resolved),
        "torch_version": torch.__version__,
        "parameter_report": model.parameter_report(),
        "train_bytes": len(train_stream),
        "validation_bytes": len(validation_stream),
        "test_bytes": len(test_stream),
        "corpus_sha256": corpus_hashes,
        "checkpoint_loaded": config.checkpoint_load,
        "checkpoint_saved": config.checkpoint_save,
        "checkpoint_metadata": checkpoint_metadata,
        "best_validation_bpb": best_validation,
        "test_bpb": test_bpb,
        "initial_probabilities": initial_probabilities.tolist(),
        "final_probabilities": final_probabilities.tolist(),
        "router_mean_l1_movement": float((final_probabilities - initial_probabilities).abs().sum(dim=-1).mean()),
        "router_entropy": normalized_entropy(final_probabilities),
        "router_argmax": final_probabilities.argmax(dim=-1).tolist(),
        "elapsed_seconds": time.time() - started,
        "history": history,
    }
