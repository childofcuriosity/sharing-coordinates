"""Run a single planted factorization experiment or a pre-specified sweep."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import List

from src.factorization import FactorizationConfig, run_factorization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/factorization.json"))
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--pattern", default="random_balanced")
    parser.add_argument("--method", default="soft")
    parser.add_argument("--init", default="neutral")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    return parser.parse_args()


def sweep_configs(base: FactorizationConfig) -> List[FactorizationConfig]:
    configs: List[FactorizationConfig] = []
    patterns = ["cycle", "contiguous", "palindrome", "random_balanced", "imbalanced"]
    initializations = ["neutral", "random", "pattern_contiguous", "pattern_cycle"]
    methods = {
        "soft": {},
        "straight_through": {"tau_start": 2.0, "tau_end": 0.15},
        "vertex_hard": {
            "tau_start": 2.0,
            "tau_end": 0.15,
            "route_strength": 0.02,
            "balance_strength": 0.02,
            "basis_repulsion": 0.01,
        },
        "kmeans": {},
        "oracle": {},
    }
    for pattern in patterns:
        for seed in range(10):
            for method, overrides in methods.items():
                inits = initializations if method in {"soft", "straight_through", "vertex_hard"} else ["neutral"]
                for init in inits:
                    cfg = replace(base, pattern=pattern, seed=seed, method=method, init=init, **overrides)
                    configs.append(cfg)
    return configs


def main() -> None:
    args = parse_args()
    base = FactorizationConfig(
        pattern=args.pattern,
        method=args.method,
        init=args.init,
        seed=args.seed,
        steps=args.steps,
    )
    configs = sweep_configs(base) if args.sweep else [base]
    results = []
    for index, config in enumerate(configs, start=1):
        result = run_factorization(config, device=args.device)
        results.append(result)
        print(
            "[%d/%d] %-16s %-18s %-20s seed=%d err=%.3g ARI=%.3f acc=%.3f"
            % (
                index,
                len(configs),
                config.pattern,
                config.method,
                config.init,
                config.seed,
                result["relative_reconstruction_error"],
                result["adjusted_rand"],
                result["assignment_accuracy"],
            ),
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
