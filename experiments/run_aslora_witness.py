"""Audit an ASLoRA first-merge snapshot without downloading or training a model.

For a real RoBERTa run, training code should use
``attach_aslora_to_roberta_classifier`` and call
``handle.update_running_averages()`` after each optimizer step.  Immediately
before the first merge, it should save ``handle.capture_first_merge_snapshot``.
This CLI then audits that saved snapshot.  With no ``--snapshot`` it runs a
small deterministic tensor witness suitable for installation checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

# Permit both ``python -m experiments.run_aslora_witness`` and direct execution
# from a clean source checkout without requiring an editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aslora_witness import (
    ASLORA_SOURCE_NOTES,
    FactorSnapshot,
    FirstMergeSnapshot,
    apply_snapshot_gauges,
    candidate_pairs,
    effective_updates,
    evaluate_candidate_merges,
    gauge_condition_number,
    is_orthogonal_gauge,
    load_first_merge_snapshot,
    make_three_layer_flip_snapshot,
    save_first_merge_snapshot,
    search_ranking_flip_gauge,
    snapshot_distances,
    snapshot_selected_pairs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the first ASLoRA lower-uses-upper merge under a fixed common "
            "GL gauge. No model or dataset is downloaded by this command."
        )
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="versioned first-merge .pt snapshot; omit for the deterministic tensor demo",
    )
    parser.add_argument("--save-snapshot", type=Path, help="save the input/demo snapshot here")
    parser.add_argument("--output", type=Path, help="write JSON audit output instead of stdout only")
    parser.add_argument(
        "--candidate-mode",
        choices=("all", "adjacent"),
        help="override the saved all-vs-adjacent interpretation of the paper",
    )
    parser.add_argument(
        "--decision-scope",
        choices=("per_projection", "joint"),
        help="override independent WQ/WV decisions versus one joint layer-pair decision",
    )
    parser.add_argument(
        "--target",
        default="query",
        help="projection whose gauge is changed and whose immediate merge hooks are evaluated",
    )
    parser.add_argument(
        "--gauge-diagonal",
        type=float,
        nargs="+",
        help="explicit positive diagonal entries for Q; otherwise search for a ranking flip",
    )
    parser.add_argument("--max-condition", type=float, default=100.0)
    parser.add_argument("--search-trials", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.2, help="tensor-demo cheap merge cost")
    parser.add_argument("--delta", type=float, default=0.05, help="tensor-demo gauge contraction")
    return parser


def _distance_json(
    distances: Mapping[str, Mapping[Tuple[int, int], torch.Tensor]]
) -> Dict[str, Dict[str, float]]:
    return {
        name: {
            "%d-%d" % pair: float(value.detach().cpu())
            for pair, value in sorted(values.items())
        }
        for name, values in distances.items()
    }


def _selection_json(selected: Mapping[str, Tuple[int, int]]) -> Dict[str, Sequence[int]]:
    return {name: list(pair) for name, pair in selected.items()}


def _product_errors(
    before: FirstMergeSnapshot,
    after: FirstMergeSnapshot,
) -> Dict[str, float]:
    errors: Dict[str, float] = {}
    for name in before.targets:
        first = effective_updates(before.targets[name].shared_a, before.targets[name].layer_b)
        second = effective_updates(after.targets[name].shared_a, after.targets[name].layer_b)
        errors[name] = float((first - second).abs().max().detach().cpu())
    return errors


def _immediate_tensor_merge_metrics(
    state: FactorSnapshot,
    mode: str,
) -> Sequence[Dict[str, Any]]:
    baseline = effective_updates(state.shared_a, state.layer_b)

    def evaluator(
        pair: Tuple[int, int], shared_a: torch.Tensor, merged_b: torch.Tensor
    ) -> Mapping[str, float]:
        merged = effective_updates(shared_a, merged_b)
        lower, _ = pair
        change = torch.linalg.norm(merged[lower] - baseline[lower])
        relative = change / torch.linalg.norm(baseline[lower]).clamp_min(1e-12)
        return {
            "lower_effective_update_change": float(change.detach().cpu()),
            "lower_relative_effective_update_change": float(relative.detach().cpu()),
        }

    records = evaluate_candidate_merges(state, evaluator=evaluator, mode=mode)
    return [
        {
            **record,
            "pair": list(record["pair"]),
        }
        for record in records
    ]


def _load_or_make_snapshot(args: argparse.Namespace) -> Tuple[FirstMergeSnapshot, Optional[torch.Tensor]]:
    if args.snapshot is not None:
        return load_first_merge_snapshot(args.snapshot), None
    return make_three_layer_flip_snapshot(
        epsilon=args.epsilon,
        delta=args.delta,
        dtype=torch.float64,
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    snapshot, demo_gauge = _load_or_make_snapshot(args)
    if args.candidate_mode is not None:
        snapshot.candidate_mode = args.candidate_mode
    if args.decision_scope is not None:
        snapshot.decision_scope = args.decision_scope
    if args.target not in snapshot.targets:
        raise ValueError(
            "target %r is not in snapshot targets %s"
            % (args.target, sorted(snapshot.targets))
        )
    if args.save_snapshot is not None:
        save_first_merge_snapshot(snapshot, args.save_snapshot)

    target_state = snapshot.targets[args.target]
    gauge: Optional[torch.Tensor]
    gauge_origin: str
    if args.gauge_diagonal is not None:
        if len(args.gauge_diagonal) != target_state.shared_a.shape[0]:
            raise ValueError("--gauge-diagonal must provide exactly rank entries")
        if any(value <= 0 for value in args.gauge_diagonal):
            raise ValueError("--gauge-diagonal entries must be positive")
        gauge = torch.diag(
            torch.tensor(
                args.gauge_diagonal,
                dtype=target_state.shared_a.dtype,
                device=target_state.shared_a.device,
            )
        )
        gauge_origin = "explicit-diagonal"
    elif demo_gauge is not None:
        gauge = demo_gauge
        gauge_origin = "deterministic-demo"
    else:
        gauge = search_ranking_flip_gauge(
            target_state,
            mode=snapshot.candidate_mode,
            max_condition=args.max_condition,
            trials=args.search_trials,
            seed=args.seed,
        )
        gauge_origin = "bounded-random-search"

    before_raw = snapshot_distances(snapshot, invariant=False)
    before_invariant = snapshot_distances(snapshot, invariant=True)
    before_selected = snapshot_selected_pairs(snapshot, invariant=False)

    if gauge is None:
        transformed = snapshot.detached_cpu_clone()
        after_raw = before_raw
        after_invariant = before_invariant
        after_selected = before_selected
        gauge_payload: Dict[str, Any] = {
            "found": False,
            "origin": gauge_origin,
            "search_is_not_a_stability_certificate": True,
        }
    else:
        transformed = apply_snapshot_gauges(snapshot, {args.target: gauge})
        after_raw = snapshot_distances(transformed, invariant=False)
        after_invariant = snapshot_distances(transformed, invariant=True)
        after_selected = snapshot_selected_pairs(transformed, invariant=False)
        gauge_payload = {
            "found": True,
            "origin": gauge_origin,
            "matrix": gauge.detach().cpu().tolist(),
            "condition_number": gauge_condition_number(gauge),
            "orthogonal": is_orthogonal_gauge(gauge),
        }

    payload: Dict[str, Any] = {
        "snapshot_kind": snapshot.metadata.get("kind", "external-roberta-first-merge"),
        "candidate_mode": snapshot.candidate_mode,
        "decision_scope": snapshot.decision_scope,
        "candidate_pairs": [
            list(pair)
            for pair in candidate_pairs(
                target_state.layer_b.shape[0], snapshot.candidate_mode
            )
        ],
        "target_gauged": args.target,
        "gauge": gauge_payload,
        "selected_before": _selection_json(before_selected),
        "selected_after": _selection_json(after_selected),
        "ranking_changed": before_selected != after_selected,
        "raw_distances_before": _distance_json(before_raw),
        "raw_distances_after": _distance_json(after_raw),
        "invariant_distances_before": _distance_json(before_invariant),
        "invariant_distances_after": _distance_json(after_invariant),
        "effective_product_max_abs_errors": _product_errors(snapshot, transformed),
        "immediate_candidate_merge_hooks_before": _immediate_tensor_merge_metrics(
            target_state, snapshot.candidate_mode
        ),
        "immediate_candidate_merge_hooks_after": _immediate_tensor_merge_metrics(
            transformed.targets[args.target], transformed.candidate_mode
        ),
        "running_observations": {
            name: state.running_count for name, state in snapshot.targets.items()
        },
        "snapshot_metadata": snapshot.metadata,
        "paper_scope_notes": dict(ASLORA_SOURCE_NOTES),
    }
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = run(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print("wrote %s" % args.output)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
