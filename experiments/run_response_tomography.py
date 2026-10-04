"""Run the operator-determining finite response probes at a frozen checkpoint.

This is an additive audit.  It does not replace or mutate the source-attested
Gaussian response experiment in :mod:`experiments.run_response_audit`.
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
    cyclic_permutation,
    deterministic_positive_stochastic_gauge,
    file_sha256,
    matrix_diagnostics,
    pack_basis_matrix,
    transform_factors,
    write_json,
)
from src.response_tomography import audit_structured_tomography


PRIMARY_TRAINING_STEPS = 1500
FIXED_NONPERMUTATION_GAUGE_STRENGTH = 0.35


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply K deterministic, operator-determining Euclidean-response "
            "probes to a frozen full-rank shared-router checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eta-z", type=float, default=1.0)
    parser.add_argument("--eta-b", type=float, default=1.0)
    parser.add_argument(
        "--gauge-strength",
        type=float,
        default=FIXED_NONPERMUTATION_GAUGE_STRENGTH,
    )
    parser.add_argument("--minimum-probability", type=float, default=1e-12)
    parser.add_argument("--equality-relative-tolerance", type=float, default=5e-11)
    parser.add_argument(
        "--allow-nonprimary-checkpoint",
        action="store_true",
        help="permit a checkpoint outside the frozen 1500-step primary protocol",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the protocol without opening the checkpoint",
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
    return {"checks": checks, "all_pass": all(checks.values())}


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
        "probe_design": "effective_rowspace_structured_deterministic_K_probe",
        "eta_z": args.eta_z,
        "eta_b": args.eta_b,
        "gauge_family": "deterministic_positive_stochastic_shrink_to_uniform",
        "gauge_strength": args.gauge_strength,
        "gauge_strength_matches_gaussian_audit": math.isclose(
            args.gauge_strength,
            FIXED_NONPERMUTATION_GAUGE_STRENGTH,
            rel_tol=0.0,
            abs_tol=0.0,
        ),
        "minimum_probability": args.minimum_probability,
        "equality_relative_tolerance": args.equality_relative_tolerance,
        "operator_materialized": False,
        "exact_common_product_holds_by_symbolic_gauge_construction": True,
        "old_source_attested_response_generator_modified": False,
    }
    if args.dry_run:
        return {
            "format": "response-tomography-audit-v2",
            "status": "dry_run_checkpoint_not_opened",
            "protocol": protocol,
            "versions": _runtime_versions(),
        }
    if args.eta_z <= 0 or args.eta_b <= 0:
        raise ValueError("eta-z and eta-b must be positive")
    if not 0 < args.gauge_strength < 1:
        raise ValueError("gauge-strength must lie in (0,1)")
    if args.minimum_probability < 0:
        raise ValueError("minimum-probability must be nonnegative")
    if args.equality_relative_tolerance < 0:
        raise ValueError("equality-relative-tolerance must be nonnegative")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise ValueError(f"checkpoint does not exist: {checkpoint}")
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
            "checkpoint is not the frozen primary run: " + ", ".join(failed)
        )

    temperature = float(config.tau_end)
    packed = pack_basis_matrix(model, temperature=temperature)
    probabilities = torch.softmax(
        model.router.logits.detach().to(device=device, dtype=torch.float64)
        / temperature,
        dim=-1,
    )
    bases = packed.bases.detach().to(device=device, dtype=torch.float64)
    diagnostics = matrix_diagnostics(probabilities, bases)
    if not diagnostics["theorem_rank_assumptions_pass"]:
        raise ValueError("the finite tomography theorem requires full-rank A and B")
    if (
        diagnostics["effective_dimension_d"] - diagnostics["num_bases_K"]
        < diagnostics["depth_n"]
    ):
        raise ValueError("the K-probe packing condition D-K >= L does not hold")

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

    permutation_audit = audit_structured_tomography(
        probabilities,
        bases,
        permutation_a,
        permutation_b,
        eta_z=args.eta_z,
        eta_b=args.eta_b,
        temperature=temperature,
        equality_relative_tolerance=args.equality_relative_tolerance,
        exact_common_product_holds_by_symbolic_gauge_construction=True,
    )
    nonpermutation_audit = audit_structured_tomography(
        probabilities,
        bases,
        nonpermutation_a,
        nonpermutation_b,
        eta_z=args.eta_z,
        eta_b=args.eta_b,
        temperature=temperature,
        equality_relative_tolerance=args.equality_relative_tolerance,
        exact_common_product_holds_by_symbolic_gauge_construction=True,
    )
    reconstruction_tolerance = 5e-10
    reconstruction_errors = []
    for audit in (permutation_audit, nonpermutation_audit):
        reconstruction_errors.extend(
            [
                audit["reconstruction"]["reference_router_gram_relative_error"],
                audit["reconstruction"]["candidate_router_gram_relative_error"],
                audit["reconstruction"]["reference_local_blocks_relative_error"],
                audit["reconstruction"]["candidate_local_blocks_relative_error"],
            ]
        )
    maximum_reconstruction_error = max(reconstruction_errors)
    all_checks_pass = bool(
        permutation_audit[
            "common_design_operator_determining_theorem_applies"
        ]
        and nonpermutation_audit[
            "common_design_operator_determining_theorem_applies"
        ]
        and permutation_audit["common_product_numerical_tolerance_pass"]
        and nonpermutation_audit["common_product_numerical_tolerance_pass"]
        and permutation_audit[
            "response_equal_on_operator_determining_probes_within_tolerance"
        ]
        and nonpermutation_audit["response_operators_distinguished_within_tolerance"]
        and maximum_reconstruction_error <= reconstruction_tolerance
    )
    if not all_checks_pass:
        raise RuntimeError("structured response tomography failed a required check")

    source_paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "response_tomography.py",
        PROJECT_ROOT / "src" / "response_identifiability.py",
        PROJECT_ROOT / "src" / "language.py",
        PROJECT_ROOT / "src" / "models.py",
    )
    result: Dict[str, Any] = {
        "format": "response-tomography-audit-v2",
        "status": "completed_operator_determining_response_tomography",
        "protocol": protocol,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "language_config": asdict(config),
            "metadata": metadata,
            "primary_checks": primary_checks,
        },
        "factor_diagnostics": diagnostics,
        "probe_count_K": diagnostics["num_bases_K"],
        "permutation_control": {
            "gauge": permutation.detach().cpu().tolist(),
            "equivalence": permutation_equivalence,
            "tomography": permutation_audit,
        },
        "positive_stochastic_nonpermutation": {
            "gauge": nonpermutation.detach().cpu().tolist(),
            "equivalence": nonpermutation_equivalence,
            "tomography": nonpermutation_audit,
        },
        "maximum_component_reconstruction_relative_error": (
            maximum_reconstruction_error
        ),
        "component_reconstruction_relative_tolerance": reconstruction_tolerance,
        "all_checks_pass": all_checks_pass,
        "claim_boundary": (
            "The K designed probes uniformly determine the complete structured "
            "instantaneous Euclidean response over exact full-rank factorizations "
            "of this Theta. They identify the current Euclidean chart only after "
            "the exact stabilizer theorem; they do not identify AdamW state, a "
            "finite-step trajectory, functional equivalence, or historical truth."
        ),
        "versions": _runtime_versions(),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): file_sha256(path)
            for path in source_paths
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
        print(f"wrote response tomography: {args.output.resolve()}")


if __name__ == "__main__":
    main()
