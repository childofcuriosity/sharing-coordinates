"""Run the paired functional-probe sample-complexity/noise sweep."""

import os

# Must be set before torch is imported through src.probes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from src.probes import (
    PROBE_PROTOCOL_SPEC,
    PROBE_PROTOCOL_VERSION,
    ProbeConfig,
    probe_protocol_fingerprint_sha256,
    run_probe_recovery,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(repository_root: Path):
    relative_paths = (
        "src/probes.py",
        "src/models.py",
        "src/patterns.py",
        "src/metrics.py",
        "experiments/run_probes.py",
    )
    return {
        relative: _sha256(repository_root / relative)
        for relative in relative_paths
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/probes.json"))
    parser.add_argument("--architecture", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument("--device", default=None)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-probes", type=int, default=16)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--projection-dimension", type=int, default=0)
    parser.add_argument("--sweep-seeds", type=int, default=10)
    args = parser.parse_args()
    if args.noise < 0:
        parser.error("--noise must be nonnegative")
    if args.num_probes <= 0 or args.sweep_seeds <= 0:
        parser.error("probe and seed counts must be positive")

    base = ProbeConfig(
        architecture=args.architecture,
        seed=args.seed,
        num_probes=args.num_probes,
        observation_noise=args.noise,
        projection_dimension=args.projection_dimension,
    )
    configs = [base]
    if args.sweep:
        configs = []
        for pattern in ["cycle", "contiguous", "palindrome", "random_balanced"]:
            for seed in range(args.sweep_seeds):
                for num_probes in [1, 2, 4, 8, 16, 32, 64]:
                    for noise in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]:
                        configs.append(
                            replace(
                                base,
                                pattern=pattern,
                                seed=seed,
                                num_probes=num_probes,
                                observation_noise=noise,
                            )
                        )

    repository_root = Path(__file__).resolve().parents[1]
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    results = []
    for index, config in enumerate(configs, 1):
        result = run_probe_recovery(config, device=args.device)
        results.append(result)
        print(
            (
                "[%d/%d] %s %s n=%d sigma=%.2f seed=%d "
                "cf=%.3f native=%.3f parameter=%.3f"
            )
            % (
                index,
                len(configs),
                config.architecture,
                config.pattern,
                config.num_probes,
                config.observation_noise,
                config.seed,
                result["counterfactual"]["adjusted_rand"],
                result["native"]["adjusted_rand"],
                result["effective_parameter"]["adjusted_rand"],
            ),
            flush=True,
        )

    elapsed = time.perf_counter() - started
    payload = {
        "schema_version": "probe-sweep-v2",
        "protocol_version": PROBE_PROTOCOL_VERSION,
        "protocol_fingerprint_payload": PROBE_PROTOCOL_SPEC,
        "protocol_fingerprint_sha256": probe_protocol_fingerprint_sha256(),
        "generation": {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "architecture": args.architecture,
            "device_argument": args.device,
            "sweep": args.sweep,
            "sweep_seeds": args.sweep_seeds,
            "projection_dimension": args.projection_dimension,
            "record_count": len(results),
            "source_sha256": _source_hashes(repository_root),
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(
        "[wrote] %s records=%d elapsed=%.3fs" % (args.output, len(results), elapsed),
        flush=True,
    )


if __name__ == "__main__":
    main()
