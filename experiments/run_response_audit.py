"""Audit soft-router response identifiability at a frozen byte-LM checkpoint.

This runner loads a predeclared step-1500 checkpoint, packs every BasisLinear
weight and bias into a common basis matrix, and compares instantaneous
*Euclidean* response operators through fixed Gaussian probes.  It never claims
to reproduce the checkpoint's AdamW optimizer state or recover historical
sharing truth.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.language import load_language_checkpoint
from src.response_identifiability import (
    audit_packing_reconstruction,
    audit_fixed_gaussian_probes,
    autograd_response_validation,
    cyclic_permutation,
    deterministic_positive_stochastic_gauge,
    file_sha256,
    matrix_diagnostics,
    pack_basis_matrix,
    transform_factors,
    write_json,
)


PRIMARY_TRAINING_STEPS = 1500
PREREGISTERED_GAUGE_STRENGTH = 0.35


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the analytic Euclidean instantaneous-response operator to "
            "fixed Gaussian probes at a frozen step-1500 byte-LM checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/response_audit_seed0.json"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-count", type=int, default=8)
    parser.add_argument("--probe-seed", type=int, default=740_000)
    parser.add_argument("--eta-z", type=float, default=1.0)
    parser.add_argument("--eta-b", type=float, default=1.0)
    parser.add_argument(
        "--gauge-strength", type=float, default=PREREGISTERED_GAUGE_STRENGTH
    )
    parser.add_argument("--minimum-probability", type=float, default=1e-12)
    parser.add_argument(
        "--small-ball-relative-threshold", type=float, default=1e-8
    )
    parser.add_argument("--autograd-rows", type=int, default=4)
    parser.add_argument("--autograd-columns", type=int, default=32)
    parser.add_argument("--autograd-seed", type=int, default=750_000)
    parser.add_argument(
        "--allow-nonprimary-checkpoint",
        action="store_true",
        help="permit a checkpoint whose recorded training protocol is not the frozen 1500-step run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the audit protocol without opening the checkpoint",
    )
    return parser


def _primary_checkpoint_checks(
    config: Any,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    extra = dict(metadata.get("extra") or {})
    checks = {
        "training_steps_is_1500": int(config.steps) == PRIMARY_TRAINING_STEPS,
        "saved_at_step_is_1500": int(extra.get("saved_at_step", -1))
        == PRIMARY_TRAINING_STEPS,
        "checkpoint_kind_is_predeclared_final": extra.get("checkpoint_kind")
        == "final_predeclared_training_step",
        "soft_router_not_hard": not bool(config.hard),
        "router_was_trainable": bool(config.router_trainable),
        "positive_final_temperature": float(config.tau_end) > 0,
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "expected_training_steps": PRIMARY_TRAINING_STEPS,
    }


def _runtime_versions() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    protocol = {
        "checkpoint": str(args.checkpoint.resolve()),
        "output": str(args.output.resolve()),
        "device": args.device,
        "probe_count": args.probe_count,
        "probe_seed": args.probe_seed,
        "eta_z": args.eta_z,
        "eta_b": args.eta_b,
        "gauge_family": "deterministic_positive_stochastic_shrink_to_uniform",
        "gauge_strength": args.gauge_strength,
        "gauge_strength_is_predeclared_default": math.isclose(
            args.gauge_strength,
            PREREGISTERED_GAUGE_STRENGTH,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "minimum_probability": args.minimum_probability,
        "small_ball_relative_threshold": args.small_ball_relative_threshold,
        "operator_materialized": False,
        "estimand": "current_layer_effective_parameter_Euclidean_instantaneous_response",
        "excluded_claims": [
            "AdamW_training_dynamics",
            "optimizer_selection_stability",
            "historical_or_generative_sharing_truth",
            "downstream_router_decision_reliability",
        ],
    }
    if args.dry_run:
        return {
            "format": "router-response-audit-v1",
            "status": "dry_run_checkpoint_not_opened",
            "protocol": protocol,
            "versions": _runtime_versions(),
        }
    if args.probe_count < 1:
        raise ValueError("probe-count must be positive")
    if args.eta_z <= 0 or args.eta_b <= 0:
        raise ValueError("eta-z and eta-b must be positive")
    if args.minimum_probability < 0:
        raise ValueError("minimum-probability must be nonnegative")
    if args.small_ball_relative_threshold < 0:
        raise ValueError("small-ball threshold must be nonnegative")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError("checkpoint does not exist: %s" % checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    model, config, metadata = load_language_checkpoint(
        str(checkpoint), device=str(device)
    )
    model.eval()
    model.requires_grad_(False)
    primary_checks = _primary_checkpoint_checks(config, metadata)
    if not primary_checks["all_pass"] and not args.allow_nonprimary_checkpoint:
        failed = [
            name for name, passed in primary_checks["checks"].items() if not passed
        ]
        raise ValueError(
            "checkpoint is not the frozen primary run: %s"
            % ", ".join(failed)
        )
    temperature = float(config.tau_end)
    packed_native = pack_basis_matrix(model, temperature=temperature)
    packing_reconstruction = audit_packing_reconstruction(model, packed_native)
    packing_tolerance = 5e-6
    if packing_reconstruction["relative_l2_error"] > packing_tolerance:
        raise RuntimeError("packed bases do not reconstruct layer-effective tensors")
    # Float64 makes the permutation control and factor equivalence checks much
    # tighter than checkpoint float32 without mutating the loaded model.
    probabilities = torch.softmax(
        model.router.logits.detach().to(device=device, dtype=torch.float64)
        / temperature,
        dim=-1,
    )
    bases = packed_native.bases.detach().to(device=device, dtype=torch.float64)
    diagnostics = matrix_diagnostics(probabilities, bases)

    permutation = cyclic_permutation(
        diagnostics["num_bases_K"], dtype=torch.float64
    ).to(device)
    permutation_a, permutation_b, permutation_equivalence = transform_factors(
        probabilities,
        bases,
        permutation,
        min_probability=args.minimum_probability,
    )
    nonpermutation = deterministic_positive_stochastic_gauge(
        diagnostics["num_bases_K"],
        strength=args.gauge_strength,
        dtype=torch.float64,
    ).to(device)
    nonpermutation_a, nonpermutation_b, nonpermutation_equivalence = transform_factors(
        probabilities,
        bases,
        nonpermutation,
        min_probability=args.minimum_probability,
    )
    equivalence_tolerance = 5e-11
    if (
        permutation_equivalence["effective_relative_l2_error"]
        > equivalence_tolerance
        or nonpermutation_equivalence["effective_relative_l2_error"]
        > equivalence_tolerance
    ):
        raise RuntimeError("a declared gauge did not preserve effective parameters")

    permutation_probes = audit_fixed_gaussian_probes(
        probabilities,
        bases,
        permutation_a,
        permutation_b,
        probe_count=args.probe_count,
        probe_seed=args.probe_seed,
        eta_z=args.eta_z,
        eta_b=args.eta_b,
        temperature=temperature,
        small_ball_relative_threshold=args.small_ball_relative_threshold,
    )
    nonpermutation_probes = audit_fixed_gaussian_probes(
        probabilities,
        bases,
        nonpermutation_a,
        nonpermutation_b,
        probe_count=args.probe_count,
        probe_seed=args.probe_seed,
        eta_z=args.eta_z,
        eta_b=args.eta_b,
        temperature=temperature,
        small_ball_relative_threshold=args.small_ball_relative_threshold,
    )
    control_tolerance = 5e-11
    permutation_control_pass = (
        permutation_equivalence["effective_relative_l2_error"] <= control_tolerance
        and permutation_probes["maximum_relative_response_difference"]
        <= control_tolerance
    )
    if not permutation_control_pass:
        raise RuntimeError("permutation response control failed")
    nonpermutation_detected = nonpermutation_probes["detected_probe_count"] > 0
    if diagnostics["theorem_rank_assumptions_pass"] and not nonpermutation_detected:
        raise RuntimeError(
            "full-rank checkpoint but fixed probes did not detect the non-permutation response"
        )

    native_autograd = autograd_response_validation(
        probabilities,
        bases,
        rows=args.autograd_rows,
        columns=args.autograd_columns,
        seed=args.autograd_seed,
        eta_z=args.eta_z,
        eta_b=args.eta_b,
        temperature=temperature,
    )
    transformed_autograd = autograd_response_validation(
        nonpermutation_a,
        nonpermutation_b,
        rows=args.autograd_rows,
        columns=args.autograd_columns,
        seed=args.autograd_seed,
        eta_z=args.eta_z,
        eta_b=args.eta_b,
        temperature=temperature,
    )
    autograd_tolerance = 1e-10
    finite_difference_tolerance = 1e-6
    if (
        native_autograd["analytic_vs_autograd_relative_l2_error"]
        > autograd_tolerance
        or transformed_autograd["analytic_vs_autograd_relative_l2_error"]
        > autograd_tolerance
        or native_autograd["best_finite_difference_relative_l2_error"]
        > finite_difference_tolerance
        or transformed_autograd["best_finite_difference_relative_l2_error"]
        > finite_difference_tolerance
    ):
        raise RuntimeError("analytic response failed its autograd check")

    result: Dict[str, Any] = {
        "format": "router-response-audit-v1",
        "status": "completed_current_coordinate_response_audit",
        "protocol": protocol,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "language_config": asdict(config),
            "metadata": metadata,
            "primary_checks": primary_checks,
            "parameters_frozen_for_audit": all(
                not parameter.requires_grad for parameter in model.parameters()
            ),
        },
        "packing": {
            "convention": (
                "model.named_modules BasisLinear registration order; each module "
                "weight then bias; each tensor flattened after its basis axis"
            ),
            "layout": [item.to_dict() for item in packed_native.layout],
            "effective_dimension": packed_native.effective_dimension,
            "reconstruction_audit": packing_reconstruction,
            "reconstruction_relative_tolerance": packing_tolerance,
            "reconstruction_pass": True,
            "excluded_parameters": (
                "embeddings, LayerNorms, and other non-BasisLinear parameters are "
                "not part of the shared-router Theta=A B estimand"
            ),
        },
        "factor_diagnostics": diagnostics,
        "response_definition": {
            "sign": "reported_operator_is_minus_Theta_dot",
            "formula": (
                "eta_b A A^T G + (eta_z/tau^2) row_i["
                "G_i B^T (diag(a_i)-a_i a_i^T)^2 B]"
            ),
            "temperature_tau": temperature,
            "eta_z": args.eta_z,
            "eta_b": args.eta_b,
            "optimizer_warning": (
                "hypothetical Euclidean gradient flow at the checkpoint; the "
                "training checkpoint used AdamW and this audit does not claim "
                "to reproduce AdamW state or its next update"
            ),
        },
        "permutation_control": {
            "gauge": permutation.detach().cpu().tolist(),
            "equivalence": permutation_equivalence,
            "probe_audit": permutation_probes,
            "control_tolerance": control_tolerance,
            "pass": permutation_control_pass,
        },
        "positive_stochastic_nonpermutation": {
            "selection": (
                "predeclared_default_fixed_before_checkpoint_response_inspection"
                if math.isclose(
                    args.gauge_strength,
                    PREREGISTERED_GAUGE_STRENGTH,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                else "explicit_user_override_not_claimed_as_predeclared"
            ),
            "gauge": nonpermutation.detach().cpu().tolist(),
            "is_nonpermutation": True,
            "equivalence": nonpermutation_equivalence,
            "probe_audit": nonpermutation_probes,
            "detected_by_at_least_one_fixed_probe": nonpermutation_detected,
        },
        "formula_validation": {
            "native_extracted_factor_model": native_autograd,
            "nonpermutation_extracted_factor_model": transformed_autograd,
            "autograd_relative_tolerance": autograd_tolerance,
            "finite_difference_relative_tolerance": finite_difference_tolerance,
            "pass": True,
        },
        "interpretation_boundary": {
            "identified_if_exact_full_response_were_available": (
                "current softmax-router factor coordinates up to basis permutation, "
                "under strict positivity, full-rank A/B, and Euclidean-flow assumptions"
            ),
            "not_identified": [
                "historical sharing graph",
                "generative truth",
                "which gauge AdamW would select",
                "reliability of a downstream router-based decision",
            ],
            "finite_probe_warning": (
                "Gaussian probes are a detection audit, not equality certification; "
                "non-permutation gauges can approach permutations arbitrarily closely"
            ),
        },
        "versions": _runtime_versions(),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                PROJECT_ROOT / "src" / "response_identifiability.py",
                PROJECT_ROOT / "src" / "language.py",
                PROJECT_ROOT / "src" / "models.py",
            )
        },
    }
    write_json(args.output.resolve(), result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    else:
        print("wrote response audit: %s" % args.output.resolve())


if __name__ == "__main__":
    main()
