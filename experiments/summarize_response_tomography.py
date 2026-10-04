"""Validate and summarize additive finite-response tomography artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from src.response_identifiability import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SEEDS = (0, 1, 2)
REQUIRED_SOURCES = {
    "experiments/run_response_tomography.py",
    "src/response_tomography.py",
    "src/response_identifiability.py",
    "src/language.py",
    "src/models.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _validate(payload: Mapping[str, Any]) -> Dict[str, Any]:
    _require(payload.get("format") == "response-tomography-audit-v2", "wrong format")
    _require(
        payload.get("status")
        == "completed_operator_determining_response_tomography",
        "tomography audit did not complete",
    )
    checkpoint = payload.get("checkpoint")
    _require(isinstance(checkpoint, dict), "missing checkpoint provenance")
    language = checkpoint.get("language_config")
    _require(isinstance(language, dict), "missing language configuration")
    seed = int(language.get("seed", -1))
    _require(seed in PRIMARY_SEEDS, f"unexpected seed {seed}")
    _require(
        checkpoint.get("primary_checks", {}).get("all_pass") is True,
        f"seed {seed} checkpoint checks failed",
    )
    protocol = payload.get("protocol")
    _require(isinstance(protocol, dict), f"seed {seed} lacks protocol")
    expected_protocol = {
        "probe_design": "effective_rowspace_structured_deterministic_K_probe",
        "eta_z": 1.0,
        "eta_b": 1.0,
        "gauge_strength": 0.35,
        "minimum_probability": 1e-12,
        "operator_materialized": False,
        "exact_common_product_holds_by_symbolic_gauge_construction": True,
        "old_source_attested_response_generator_modified": False,
    }
    for key, expected in expected_protocol.items():
        _require(
            protocol.get(key) == expected,
            f"seed {seed} protocol.{key} drifted",
        )
    source = payload.get("source_sha256")
    _require(
        isinstance(source, dict)
        and set(source) == REQUIRED_SOURCES
        and all(len(str(digest)) == 64 for digest in source.values()),
        f"seed {seed} lacks complete source hashes",
    )
    diagnostics = payload.get("factor_diagnostics")
    _require(isinstance(diagnostics, dict), f"seed {seed} lacks factor diagnostics")
    _require(
        diagnostics.get("theorem_rank_assumptions_pass") is True
        and int(diagnostics.get("rank_A", -1)) == 4
        and int(diagnostics.get("rank_B", -1)) == 4
        and int(diagnostics.get("depth_n", -1)) == 12
        and int(diagnostics.get("effective_dimension_d", -1)) == 787_712,
        f"seed {seed} rank or dimension assumptions drifted",
    )
    _require(payload.get("probe_count_K") == 4, f"seed {seed} probe count drifted")
    _require(payload.get("all_checks_pass") is True, f"seed {seed} checks failed")

    permutation = payload.get("permutation_control", {}).get("tomography", {})
    nonpermutation = payload.get("positive_stochastic_nonpermutation", {}).get(
        "tomography", {}
    )
    for name, audit in (("permutation", permutation), ("nonpermutation", nonpermutation)):
        _require(
            audit.get("reference_design_rank_dimension_pass") is True,
            f"seed {seed} {name} reference design failed rank/dimension checks",
        )
        _require(
            audit.get("common_product_numerical_tolerance_pass") is True
            and audit.get(
                "exact_common_product_holds_by_symbolic_gauge_construction"
            )
            is True
            and audit.get(
                "common_design_operator_determining_theorem_applies"
            )
            is True,
            f"seed {seed} {name} lacks an exact common-product construction certificate",
        )
        design = audit.get("design", {})
        _require(
            design.get("dimension_condition_D_minus_K_ge_L") is True
            and int(design.get("theta_rank", -1)) == 4
            and int(design.get("probe_count", -1)) == 4,
            f"seed {seed} {name} design diagnostics failed",
        )
    _require(
        permutation.get(
            "response_equal_on_operator_determining_probes_within_tolerance"
        )
        is True,
        f"seed {seed} permutation control failed",
    )
    _require(
        nonpermutation.get("response_operators_distinguished_within_tolerance")
        is True,
        f"seed {seed} nonpermutation was not distinguished",
    )

    design = permutation["design"]
    return {
        "seed": seed,
        "checkpoint_sha256": str(checkpoint.get("sha256")),
        "rank_A": int(diagnostics["rank_A"]),
        "rank_B": int(diagnostics["rank_B"]),
        "theta_rank": int(design["theta_rank"]),
        "A_condition_number": _finite(
            diagnostics["A_condition_number"], "A condition number"
        ),
        "B_condition_number": _finite(
            diagnostics["B_condition_number"], "B condition number"
        ),
        "row_basis_orthonormal_max_abs_error": _finite(
            design["row_basis_orthonormal_max_abs_error"], "row orthogonality"
        ),
        "complement_codes_orthonormal_max_abs_error": _finite(
            design["complement_codes_orthonormal_max_abs_error"],
            "complement orthogonality",
        ),
        "row_complement_cross_max_abs_error": _finite(
            design["row_complement_cross_max_abs_error"], "cross orthogonality"
        ),
        "theta_rowspace_relative_residual": _finite(
            design["theta_rowspace_relative_residual"], "row-space residual"
        ),
        "maximum_component_reconstruction_relative_error": _finite(
            payload["maximum_component_reconstruction_relative_error"],
            "component reconstruction",
        ),
        "permutation_maximum_probe_relative_response_difference": _finite(
            permutation["reconstruction"][
                "maximum_probe_relative_response_difference"
            ],
            "permutation gap",
        ),
        "nonpermutation_maximum_probe_relative_response_difference": _finite(
            nonpermutation["reconstruction"][
                "maximum_probe_relative_response_difference"
            ],
            "nonpermutation gap",
        ),
    }


def summarize_response_tomography(
    payloads: Sequence[Mapping[str, Any]],
    *,
    raw_paths: Optional[Sequence[Path]] = None,
) -> Dict[str, Any]:
    _require(len(payloads) == 3, "exactly three tomography audits are required")
    rows = sorted((_validate(payload) for payload in payloads), key=lambda row: row["seed"])
    _require(
        tuple(row["seed"] for row in rows) == PRIMARY_SEEDS,
        "tomography audits must cover seeds 0, 1, and 2 exactly once",
    )
    snapshots = [dict(payload["source_sha256"]) for payload in payloads]
    _require(
        all(snapshot == snapshots[0] for snapshot in snapshots[1:]),
        "tomography audits use different source snapshots",
    )
    raw_sha256: Dict[str, str] = {}
    if raw_paths is not None:
        _require(len(raw_paths) == len(payloads), "raw path count does not match payloads")
        for path in raw_paths:
            raw_sha256[str(path).replace("\\", "/")] = file_sha256(path)
    return {
        "format": "response-tomography-summary-v2",
        "status": "validated_three_seed_structured_response_tomography",
        "probe_count_K": 4,
        "dimensions": {"depth_L": 12, "num_bases_K": 4, "effective_D": 787_712},
        "seed_rows": rows,
        "across_seed": {
            "all_rank_and_dimension_checks_pass": True,
            "all_permutation_controls_pass": True,
            "all_fixed_nonpermutations_distinguished": True,
            "maximum_component_reconstruction_relative_error": max(
                row["maximum_component_reconstruction_relative_error"] for row in rows
            ),
            "maximum_permutation_probe_relative_response_difference": max(
                row["permutation_maximum_probe_relative_response_difference"]
                for row in rows
            ),
            "minimum_nonpermutation_probe_relative_response_difference": min(
                row["nonpermutation_maximum_probe_relative_response_difference"]
                for row in rows
            ),
        },
        "source_sha256": snapshots[0],
        "raw_result_sha256": raw_sha256,
        "summary_source_sha256": file_sha256(Path(__file__).resolve()),
        "claim_boundary": (
            "Four deterministic effective-gradient probes reconstruct the complete "
            "instantaneous Euclidean response under the recorded symbolic exact-"
            "common-product construction, exact-rank, and dimension conditions. "
            "The numerical product tolerance alone is not a theorem premise. This "
            "is a structured matrix-probing corollary, not an AdamW, finite-"
            "trajectory, functional, or historical-graph result."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=3, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_json_lf(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a derived JSON artifact with deterministic LF line endings.

    The raw-result writer in ``src.response_identifiability`` remains frozen
    because its source hash is attested by the released raw audits.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    summary = summarize_response_tomography(payloads, raw_paths=args.inputs)
    _write_json_lf(args.output, summary)
    print(f"wrote response tomography summary: {args.output.resolve()}")


if __name__ == "__main__":
    main()
