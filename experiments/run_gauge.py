"""Run exact observational-equivalence and saturation experiments."""

import argparse
import json
from pathlib import Path

from src.gauge import exact_argmax_flip_example, run_gauge_counterfactual, run_saturation_sweep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/gauge.json"))
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--saturation", action="store_true")
    args = parser.parse_args()
    payload = {
        "exact_argmax_flip": exact_argmax_flip_example(),
        "gauge_trials": [run_gauge_counterfactual(seed=seed) for seed in range(args.seeds)],
    }
    if args.saturation:
        payload["saturation"] = [record for seed in range(min(args.seeds, 5)) for record in run_saturation_sweep(seed)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()

