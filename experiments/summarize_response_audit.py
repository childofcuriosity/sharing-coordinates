"""Validate and summarize the three frozen-checkpoint response audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Dict, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_SEEDS = (0, 1, 2)
REQUIRED_SOURCES = {
    "experiments/run_response_audit.py",
    "src/response_identifiability.py",
    "src/language.py",
    "src/models.py",
}


def _portable_input_path(path: Path) -> str:
    """Serialize release inputs without binding a summary to one checkout."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # External fixtures are content-bound elsewhere; their host-specific
        # absolute prefix is not scientific metadata and must not enter a
        # byte-reproducible summary.
        return path.name


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: object, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("%s must be finite" % name)
    return converted


def _validate(payload: Mapping[str, object]) -> Dict[str, object]:
    _require(payload.get("format") == "router-response-audit-v1", "wrong format")
    _require(
        payload.get("status") == "completed_current_coordinate_response_audit",
        "response audit did not complete",
    )
    checkpoint = payload.get("checkpoint")
    _require(isinstance(checkpoint, dict), "missing checkpoint provenance")
    language = checkpoint.get("language_config")
    metadata = checkpoint.get("metadata", {}).get("extra", {})
    _require(isinstance(language, dict), "missing LanguageConfig")
    seed = int(language.get("seed"))
    _require(seed in PRIMARY_SEEDS, "unexpected seed %d" % seed)
    _require(
        language.get("steps") == 1500
        and language.get("warmup_steps") == 500
        and language.get("hard") is False
        and language.get("router_trainable") is True,
        "seed %d is not the frozen soft-router training protocol" % seed,
    )
    _require(
        metadata.get("saved_at_step") == 1500
        and metadata.get("checkpoint_kind") == "final_predeclared_training_step"
        and checkpoint.get("primary_checks", {}).get("all_pass") is True,
        "seed %d checkpoint checks failed" % seed,
    )
    protocol = payload.get("protocol")
    _require(isinstance(protocol, dict), "seed %d lacks protocol" % seed)
    expected = {
        "probe_count": 8,
        "probe_seed": 740_000 + seed,
        "eta_z": 1.0,
        "eta_b": 1.0,
        "gauge_strength": 0.35,
        "minimum_probability": 1e-12,
        "small_ball_relative_threshold": 1e-8,
        "operator_materialized": False,
    }
    for key, value in expected.items():
        _require(
            protocol.get(key) == value,
            "seed %d protocol.%s=%r, expected %r"
            % (seed, key, protocol.get(key), value),
        )
    _require(
        protocol.get("gauge_strength_is_predeclared_default") is True,
        "seed %d used an exploratory gauge" % seed,
    )

    source = payload.get("source_sha256")
    _require(
        isinstance(source, dict)
        and set(source) == REQUIRED_SOURCES
        and all(len(str(digest)) == 64 for digest in source.values()),
        "seed %d lacks complete source hashes" % seed,
    )
    diagnostics = payload.get("factor_diagnostics", {})
    _require(
        diagnostics.get("theorem_rank_assumptions_pass") is True
        and diagnostics.get("A_full_column_rank") is True
        and diagnostics.get("B_full_row_rank") is True
        and _finite(diagnostics.get("minimum_A_entry"), "minimum A") > 0.0,
        "seed %d does not satisfy the theorem diagnostics" % seed,
    )
    _require(
        payload.get("packing", {}).get("reconstruction_pass") is True,
        "seed %d packing reconstruction failed" % seed,
    )
    formula = payload.get("formula_validation", {})
    _require(
        isinstance(formula, dict) and formula.get("pass") is True,
        "seed %d response formula validation failed" % seed,
    )
    native_formula = formula.get("native_extracted_factor_model", {})
    transformed_formula = formula.get("nonpermutation_extracted_factor_model", {})
    _require(
        isinstance(native_formula, dict)
        and isinstance(transformed_formula, dict)
        and int(native_formula.get("rows", -1)) == 4
        and int(native_formula.get("columns", -1)) == 32
        and int(transformed_formula.get("rows", -1)) == 4
        and int(transformed_formula.get("columns", -1)) == 32,
        "seed %d response formula extraction drifted" % seed,
    )
    native_fd = _finite(
        native_formula.get("best_finite_difference_relative_l2_error"),
        "native finite difference error",
    )
    transformed_fd = _finite(
        transformed_formula.get("best_finite_difference_relative_l2_error"),
        "transformed finite difference error",
    )
    native_autograd = _finite(
        native_formula.get("analytic_vs_autograd_relative_l2_error"),
        "native autograd error",
    )
    transformed_autograd = _finite(
        transformed_formula.get("analytic_vs_autograd_relative_l2_error"),
        "transformed autograd error",
    )
    autograd_tolerance = _finite(
        formula.get("autograd_relative_tolerance"), "autograd tolerance"
    )
    finite_difference_tolerance = _finite(
        formula.get("finite_difference_relative_tolerance"),
        "finite difference tolerance",
    )
    _require(
        max(native_autograd, transformed_autograd) <= autograd_tolerance
        and max(native_fd, transformed_fd) <= finite_difference_tolerance,
        "seed %d response formula errors exceed declared tolerances" % seed,
    )

    permutation = payload.get("permutation_control", {})
    perm_probe = permutation.get("probe_audit", {})
    _require(
        permutation.get("pass") is True
        and int(perm_probe.get("detected_probe_count", -1)) == 0
        and _finite(
            perm_probe.get("maximum_relative_response_difference"),
            "permutation response difference",
        )
        <= _finite(permutation.get("control_tolerance"), "control tolerance"),
        "seed %d permutation control failed" % seed,
    )
    alternative = payload.get("positive_stochastic_nonpermutation", {})
    alt_probe = alternative.get("probe_audit", {})
    detected = int(alt_probe.get("detected_probe_count", -1))
    probe_count = int(alt_probe.get("probe_count", -2))
    _require(
        alternative.get("is_nonpermutation") is True
        and alternative.get("detected_by_at_least_one_fixed_probe") is True
        and 0 < detected <= probe_count == 8,
        "seed %d fixed alternative was not detected" % seed,
    )
    effective_error = _finite(
        alternative.get("equivalence", {}).get("effective_relative_l2_error"),
        "effective relative error",
    )
    _require(effective_error <= 5e-11, "seed %d gauge broke equivalence" % seed)
    return {
        "seed": seed,
        "checkpoint_sha256": checkpoint.get("sha256"),
        "minimum_A_entry": _finite(diagnostics.get("minimum_A_entry"), "minimum A"),
        "A_sigma_min": _finite(diagnostics.get("A_sigma_min"), "A sigma min"),
        "B_sigma_min": _finite(diagnostics.get("B_sigma_min"), "B sigma min"),
        "effective_relative_l2_error": effective_error,
        "permutation_max_relative_response_difference": _finite(
            perm_probe.get("maximum_relative_response_difference"), "permutation max"
        ),
        "nonpermutation_median_relative_response_difference": _finite(
            alt_probe.get("median_relative_response_difference"), "alternative median"
        ),
        "nonpermutation_minimum_relative_response_difference": _finite(
            alt_probe.get("minimum_relative_response_difference"), "alternative minimum"
        ),
        "detected_probes": detected,
        "probe_count": probe_count,
        "formula_extraction_rows": 4,
        "formula_extraction_columns": 32,
        "formula_best_native_fd_relative_error": native_fd,
        "formula_best_transformed_fd_relative_error": transformed_fd,
        "formula_worst_fd_relative_error": max(native_fd, transformed_fd),
        "formula_native_autograd_relative_error": native_autograd,
        "formula_transformed_autograd_relative_error": transformed_autograd,
        "formula_worst_autograd_relative_error": max(
            native_autograd, transformed_autograd
        ),
        "source_sha256": dict(source),
    }


def summarize_response_audits(
    payloads: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    _require(len(payloads) == 3, "response summary requires exactly three audits")
    rows = [_validate(payload) for payload in payloads]
    _require(
        tuple(sorted(row["seed"] for row in rows)) == PRIMARY_SEEDS,
        "response audits must use seeds 0, 1, and 2",
    )
    rows.sort(key=lambda row: int(row["seed"]))
    first_sources = rows[0]["source_sha256"]
    _require(
        all(row["source_sha256"] == first_sources for row in rows[1:]),
        "response audits were generated by different source snapshots",
    )
    medians = [row["nonpermutation_median_relative_response_difference"] for row in rows]
    return {
        "protocol": {
            "seeds": list(PRIMARY_SEEDS),
            "fixed_gauge_strength": 0.35,
            "probes_per_seed": 8,
            "checkpoint_probe_evaluations": 24,
            "probe_seed_starts": [740_000 + seed for seed in PRIMARY_SEEDS],
            "unique_gaussian_draws": 10,
            "probe_windows_overlap": True,
            "source_sha256": first_sources,
        },
        "seed_rows": rows,
        "across_seed": {
            "all_rank_assumptions_pass": True,
            "all_formula_checks_pass": True,
            "all_permutation_controls_pass": True,
            "all_fixed_nonpermutation_probes_detected": all(
                row["detected_probes"] == row["probe_count"] for row in rows
            ),
            "mean_median_relative_response_difference": statistics.fmean(medians),
            "minimum_over_seed_medians": min(medians),
            "maximum_over_seed_medians": max(medians),
            "maximum_effective_relative_l2_error": max(
                row["effective_relative_l2_error"] for row in rows
            ),
            "maximum_formula_fd_relative_error": max(
                row["formula_worst_fd_relative_error"] for row in rows
            ),
            "maximum_formula_autograd_relative_error": max(
                row["formula_worst_autograd_relative_error"] for row in rows
            ),
        },
        "claim_boundary": (
            "This validates a fixed non-permutation alternative at three frozen "
            "checkpoints. The probes are not a universal equality certificate, "
            "and the Euclidean response is not the checkpoint AdamW trajectory "
            "or a historical sharing graph."
        ),
    }


def markdown_table(summary: Mapping[str, object]) -> str:
    lines = [
        "| Seed | min(A) | sigma_min(A) | sigma_min(B) | effective rel. error | permutation max response diff. | non-permutation median response diff. | detected |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["seed_rows"]:
        lines.append(
            "| {seed} | {amin:.3e} | {asig:.3f} | {bsig:.3f} | {eff:.3e} | "
            "{perm:.3e} | {alt:.3f} | {detected}/{count} |".format(
                seed=row["seed"],
                amin=row["minimum_A_entry"],
                asig=row["A_sigma_min"],
                bsig=row["B_sigma_min"],
                eff=row["effective_relative_l2_error"],
                perm=row["permutation_max_relative_response_difference"],
                alt=row["nonpermutation_median_relative_response_difference"],
                detected=row["detected_probes"],
                count=row["probe_count"],
            )
        )
    return "\n".join(lines) + "\n"


def _write_text_lf(path: Path, text: str) -> None:
    """Write canonical derived text with platform-independent LF endings."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=3, type=Path)
    parser.add_argument("--json-output", type=Path, default=Path("results/response_summary.json"))
    parser.add_argument("--csv-output", type=Path, default=Path("results/response_table.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("results/response_table.md"))
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    summary = summarize_response_audits(payloads)
    summary["input_files"] = [_portable_input_path(path) for path in args.inputs]
    for output in (args.json_output, args.csv_output, args.markdown_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    _write_text_lf(
        args.json_output,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    rows = []
    for row in summary["seed_rows"]:
        flat = dict(row)
        flat.pop("source_sha256", None)
        rows.append(flat)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_text_lf(args.markdown_output, markdown_table(summary))
    print("wrote %s, %s, and %s" % (args.json_output, args.csv_output, args.markdown_output))


if __name__ == "__main__":
    main()
