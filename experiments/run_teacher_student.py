"""CLI for functional sharing-graph recovery experiments."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from src.teacher_student import TeacherStudentConfig, run_teacher_student


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/teacher_student.json"))
    parser.add_argument("--architecture", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--pattern", default="random_balanced")
    parser.add_argument("--router-init", default="neutral")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-weight", type=float, default=0.0)
    parser.add_argument("--vertex-strength", type=float, default=0.0)
    parser.add_argument("--usage-strength", type=float, default=0.0)
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-seeds", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    base = TeacherStudentConfig(
        architecture=args.architecture,
        pattern=args.pattern,
        router_init=args.router_init,
        steps=args.steps,
        seed=args.seed,
        hidden_weight=args.hidden_weight,
        vertex_strength=args.vertex_strength,
        usage_strength=args.usage_strength,
        hard=args.hard,
    )
    if args.architecture == "transformer":
        base = replace(base, batch_size=64, train_examples=4096, test_examples=512)
    configs = [base]
    if args.sweep:
        configs = []
        for pattern in ["cycle", "contiguous", "palindrome", "random_balanced"]:
            for seed in range(args.sweep_seeds):
                for router_init in ["neutral", "pattern_cycle", "pattern_contiguous"]:
                    configs.append(replace(base, pattern=pattern, seed=seed, router_init=router_init))
    results = []
    for index, config in enumerate(configs, 1):
        result = run_teacher_student(config, device=args.device)
        results.append(result)
        print(
            "[%d/%d] %s %s %s seed=%d test=%.4g ARI=%.3f acc=%.3f"
            % (
                index,
                len(configs),
                config.architecture,
                config.pattern,
                config.router_init,
                config.seed,
                result["test_task_loss"],
                result["adjusted_rand"],
                result["assignment_accuracy"],
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
