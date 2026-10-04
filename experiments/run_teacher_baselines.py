"""Evaluate task-free reference predictors for the planted distillation tasks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.teacher_student import (
    TeacherStudentConfig,
    _build_model,
    _distillation_loss,
    _make_data,
    _seed,
)


def evaluate_reference(config: TeacherStudentConfig, device: str) -> Dict[str, object]:
    """Compare the teacher with identity (MLP) or uniform logits (Transformer)."""

    _seed(config.seed)
    resolved = torch.device(device)
    teacher = _build_model(config, teacher=True, device=resolved).eval()
    _, test_data = _make_data(config, resolved)
    with torch.no_grad():
        teacher_output = teacher(test_data, hard=True)
        if config.architecture == "mlp":
            reference_output = test_data
            reference_type = "identity"
        else:
            reference_output = torch.zeros_like(teacher_output)
            reference_type = "uniform_logits"
        loss = _distillation_loss(reference_output, teacher_output, config.architecture)
    return {
        "config": asdict(config),
        "reference_type": reference_type,
        "reference_loss": float(loss),
        "device": str(resolved),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path("results/teacher_student_mlp.json"),
            Path("results/teacher_student_transformer.json"),
        ],
    )
    parser.add_argument("--output", type=Path, default=Path("results/teacher_baselines.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    configs: Dict[Tuple[str, str, int], TeacherStudentConfig] = {}
    for path in args.inputs:
        records = json.loads(path.read_text(encoding="utf-8"))
        for record in records:
            config = TeacherStudentConfig(**record["config"])
            key = (config.architecture, config.pattern, config.seed)
            configs[key] = config

    results: List[Dict[str, object]] = []
    for index, key in enumerate(sorted(configs), start=1):
        result = evaluate_reference(configs[key], args.device)
        results.append(result)
        print(
            "[%d/%d] %s %s seed=%d %s=%.6g"
            % (
                index,
                len(configs),
                key[0],
                key[1],
                key[2],
                result["reference_type"],
                result["reference_loss"],
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
