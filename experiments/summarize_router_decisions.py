"""Validate and summarize the predeclared three-seed router-decision audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import statistics
from typing import Dict, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIMARY_TRAINING_STEPS = 1500
PRIMARY_TRAINING_SEEDS = (0, 1, 2)
PRIMARY_EFFECT_THRESHOLD_BPB = 0.01
BOOTSTRAP_RESAMPLES = 10_000
T_CRITICAL_DF2_95 = 4.303


def _portable_input_path(path: Path) -> str:
    """Serialize release inputs without binding a summary to one checkout."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_float(value: object, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("%s must be finite" % name)
    return converted


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("cannot take a quantile of an empty sequence")
    position = probability * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    fraction = position - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def paired_bootstrap_interval(
    paired_differences: Sequence[float],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> Tuple[float, float]:
    """Percentile interval for the mean of fixed paired-batch differences."""

    if not paired_differences:
        raise ValueError("paired bootstrap needs at least one difference")
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    values = [_finite_float(value, "paired difference") for value in paired_differences]
    generator = random.Random(seed)
    count = len(values)
    means = [
        statistics.fmean(generator.choices(values, k=count))
        for _ in range(resamples)
    ]
    means.sort()
    return _linear_quantile(means, 0.025), _linear_quantile(means, 0.975)


def _expect_protocol_config(payload: Mapping[str, object], seed: int) -> None:
    config = payload["config"]
    checkpoint = payload["checkpoint"]
    language = checkpoint["language_config"]
    training = checkpoint["training_metadata"]
    expected = {
        "temperature": 1.0,
        "search_family": "positive_stochastic",
        "gauge_selection": "first_valid",
        "batch_sampling": "nonoverlap",
        "search_trials": 5000,
        "max_condition": 30.0,
        "min_probability": 1e-8,
        "batch_size": 8,
        "sequence_length": 128,
        "eval_batches": 256,
        "max_effective_relative_l2": 1e-5,
        "max_logit_abs_error": 1e-4,
        "max_batch_nll_error": 1e-5,
        "max_bpb_equivalence_error": 1e-5,
        "disable_tf32": True,
        "batch_seed": 610_000 + seed,
        "search_seed": 620_000 + seed,
        "kmeans_seed": 630_000 + seed,
        "kmeans_restarts": 8,
        "kmeans_iterations": 50,
    }
    for name, expected_value in expected.items():
        _require(
            config.get(name) == expected_value,
            "seed %d violates protocol: config.%s=%r, expected %r"
            % (seed, name, config.get(name), expected_value),
        )
    _require(
        checkpoint.get("corpus_split") == "test",
        "seed %d was not evaluated on the test split" % seed,
    )
    _require(
        language.get("steps") == PRIMARY_TRAINING_STEPS,
        "seed %d checkpoint has %r steps, expected %d"
        % (seed, language.get("steps"), PRIMARY_TRAINING_STEPS),
    )
    _require(
        language.get("warmup_steps") == 500,
        "seed %d checkpoint does not use 500 warmup steps" % seed,
    )
    _require(
        training.get("saved_at_step") == PRIMARY_TRAINING_STEPS,
        "seed %d training metadata is not a final step-%d checkpoint"
        % (seed, PRIMARY_TRAINING_STEPS),
    )
    _require(
        training.get("checkpoint_kind") == "final_predeclared_training_step",
        "seed %d checkpoint kind is not final-predeclared" % seed,
    )


def _validate_seed_payload(
    payload: Mapping[str, object],
    strict_protocol: bool,
) -> int:
    checkpoint = payload.get("checkpoint")
    _require(isinstance(checkpoint, dict), "result is missing checkpoint provenance")
    language = checkpoint.get("language_config")
    _require(isinstance(language, dict), "checkpoint is missing LanguageConfig")
    seed = int(language.get("seed"))
    if strict_protocol:
        _expect_protocol_config(payload, seed)

    status = payload.get("status", "completed")
    if status == "no_partition_changing_gauge_found":
        search = payload.get("gauge_search")
        _require(isinstance(search, dict), "seed %d lacks failed-search metadata" % seed)
        counts = search.get("counts")
        _require(isinstance(counts, dict), "seed %d lacks failed-search counts" % seed)
        ordered = [
            int(counts.get(name, -1))
            for name in (
                "attempted",
                "invertible",
                "condition_pass",
                "simplex_pass",
                "group_pass",
                "partition_change",
            )
        ]
        _require(
            all(first >= second for first, second in zip(ordered, ordered[1:])),
            "seed %d failed-search counts are not monotone" % seed,
        )
        _require(
            ordered[-1] == 0,
            "seed %d claims search failure despite a valid partition change" % seed,
        )
        if strict_protocol:
            _require(ordered[0] == 5000, "seed %d did not exhaust 5,000 gauges" % seed)
            _require(
                search.get("search_family") == "positive_stochastic"
                and search.get("selection_rule") == "first_valid"
                and int(search.get("required_groups", -1))
                == int(search.get("original_groups", -2)),
                "seed %d failed search does not match the primary family" % seed,
            )
        numeric = payload.get("numeric_controls")
        _require(
            isinstance(numeric, dict)
            and numeric.get("tf32_disabled_during_audit") is True
            and numeric.get(
                "search_failed_before_model_equivalence_or_decision_evaluation"
            )
            is True,
            "seed %d failed search lacks numeric/stage provenance" % seed,
        )
        return seed

    _require(status == "completed", "seed %d has unknown status %r" % (seed, status))

    equivalence = payload.get("equivalence")
    _require(
        isinstance(equivalence, dict) and equivalence.get("passed") is True,
        "seed %d did not pass exact equivalence" % seed,
    )
    if strict_protocol:
        expected_checks = {
            "effective_relative_l2": 1e-5,
            "max_abs_logit_error": 1e-4,
            "max_abs_batch_nll_error": 1e-5,
            "absolute_bpb_error": 1e-5,
        }
        checks = equivalence.get("checks", {})
        for name, threshold in expected_checks.items():
            check = checks.get(name, {})
            value = _finite_float(check.get("value"), "equivalence check " + name)
            _require(
                check.get("threshold") == threshold and value <= threshold,
                "seed %d equivalence check %s violates the protocol" % (seed, name),
            )
    numeric = payload.get("numeric_controls")
    _require(
        isinstance(numeric, dict)
        and numeric.get("tf32_disabled_during_audit") is True,
        "seed %d did not disable TF32" % seed,
    )
    batches = payload.get("evaluation_batches")
    _require(isinstance(batches, dict), "seed %d lacks batch provenance" % seed)
    _require(
        len(batches.get("starts", [])) == int(batches.get("count", -1)),
        "seed %d has incomplete batch starts" % seed,
    )
    if strict_protocol:
        starts = batches.get("starts", [])
        _require(
            int(batches.get("count", -1)) == 256
            and int(batches.get("batch_size", -1)) == 8
            and int(batches.get("sequence_length", -1)) == 128
            and batches.get("sampling") == "nonoverlap"
            and all(len(row) == 8 for row in starts),
            "seed %d batch metadata violates the protocol" % seed,
        )
        flattened_starts = [int(start) for row in starts for start in row]
        _require(
            len(flattened_starts) == len(set(flattened_starts))
            and all(start % 129 == 0 for start in flattened_starts),
            "seed %d evaluation blocks are not disjoint protocol blocks" % seed,
        )

    search = payload.get("gauge_search")
    _require(isinstance(search, dict), "seed %d lacks gauge search metadata" % seed)
    counts = search.get("counts")
    _require(isinstance(counts, dict), "seed %d lacks search stage counts" % seed)
    ordered = [
        int(counts.get(name, -1))
        for name in (
            "attempted",
            "invertible",
            "condition_pass",
            "simplex_pass",
            "group_pass",
            "partition_change",
        )
    ]
    _require(
        all(first >= second for first, second in zip(ordered, ordered[1:])),
        "seed %d search counts are not monotone" % seed,
    )
    _require(ordered[-1] >= 1, "seed %d has no partition-changing gauge" % seed)
    if strict_protocol:
        _require(ordered[0] == 5000, "seed %d did not attempt 5,000 gauges" % seed)
        _require(
            search.get("search_family") == "positive_stochastic"
            and search.get("selection_rule") == "first_valid"
            and float(search.get("minimum_gauge_entry")) > 0.0,
            "seed %d selected a non-primary gauge family" % seed,
        )
    _require(
        search.get("selected_trial") == search.get("first_valid_trial"),
        "seed %d did not select the first valid gauge" % seed,
    )
    condition = _finite_float(search.get("condition_number"), "condition number")
    _require(condition <= 30.0, "seed %d exceeds the condition bound" % seed)
    gauge = payload.get("gauge")
    _require(isinstance(gauge, dict), "seed %d lacks applied-gauge metadata" % seed)
    _require(
        _finite_float(gauge.get("row_sum_error"), "gauge row-sum error") <= 5e-6
        and _finite_float(
            gauge.get("minimum_transformed_probability"),
            "minimum transformed probability",
        )
        > 1e-8,
        "seed %d applied gauge is not legal" % seed,
    )

    primary = payload.get("primary_causal_contrast")
    _require(
        isinstance(primary, dict)
        and primary.get("name") == "common_effective_partition_folds"
        and primary.get("common_effective_tensors") is True,
        "seed %d lacks the partition-only primary contrast" % seed,
    )
    decisions = payload.get("decisions")
    _require(isinstance(decisions, dict), "seed %d lacks decisions" % seed)
    required_decisions = {
        "literal_argmax_original",
        "literal_argmax_gauged",
        "partition_fold_original",
        "partition_fold_gauged",
        "effective_theta_kmeans",
        "effective_theta_ward",
    }
    _require(
        required_decisions.issubset(decisions),
        "seed %d lacks one or more required decisions" % seed,
    )
    for literal in ("literal_argmax_original", "literal_argmax_gauged"):
        _require(
            decisions[literal].get("causal_role")
            == "secondary_coordinate_and_partition_confounding",
            "seed %d literal hardening is not marked confounded" % seed,
        )
    for folding in ("partition_fold_original", "partition_fold_gauged"):
        _require(
            decisions[folding].get("common_effective_partition_refit") is True,
            "seed %d primary fold is not common-effective" % seed,
        )

    reference_name = primary.get("reference_decision")
    candidate_name = primary.get("candidate_decision")
    reference_batches = decisions[reference_name].get("batch_bpb", [])
    candidate_batches = decisions[candidate_name].get("batch_bpb", [])
    _require(
        len(reference_batches) == len(candidate_batches) > 0,
        "seed %d primary paired batches are incomplete" % seed,
    )
    recomputed = [
        _finite_float(candidate, "candidate batch BPB")
        - _finite_float(reference, "reference batch BPB")
        for reference, candidate in zip(reference_batches, candidate_batches)
    ]
    recorded = primary.get("batch_bpb_delta_candidate_minus_reference", [])
    _require(
        len(recorded) == len(recomputed)
        and all(
            math.isclose(float(first), second, rel_tol=1e-10, abs_tol=1e-10)
            for first, second in zip(recorded, recomputed)
        ),
        "seed %d recorded paired deltas do not match decision outputs" % seed,
    )
    aggregate = (
        _finite_float(decisions[candidate_name]["instantaneous_bpb"], "candidate BPB")
        - _finite_float(decisions[reference_name]["instantaneous_bpb"], "reference BPB")
    )
    _require(
        math.isclose(
            aggregate,
            _finite_float(
                primary["instantaneous_bpb_delta_candidate_minus_reference"],
                "primary BPB delta",
            ),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ),
        "seed %d aggregate delta is inconsistent" % seed,
    )
    _require(
        math.isclose(
            aggregate,
            statistics.fmean(recomputed),
            rel_tol=1e-8,
            abs_tol=1e-8,
        ),
        "seed %d paired-batch and aggregate deltas disagree" % seed,
    )

    group_budget = int(payload.get("group_budget"))
    for name in (
        "partition_fold_original",
        "partition_fold_gauged",
        "effective_theta_kmeans",
        "effective_theta_ward",
    ):
        _require(
            int(decisions[name].get("num_groups", -1)) == group_budget,
            "seed %d decision %s violates the group budget" % (seed, name),
        )
    kmeans = decisions["effective_theta_kmeans"]
    ward = decisions["effective_theta_ward"]
    _require(
        math.isclose(
            float(kmeans["effective_fit"]["squared_error"]),
            float(kmeans["kmeans"]["inertia"]),
            rel_tol=1e-5,
            abs_tol=1e-8,
        ),
        "seed %d k-means inertia is stale" % seed,
    )
    _require(
        math.isclose(
            float(ward["effective_fit"]["squared_error"]),
            float(ward["ward"]["inertia"]),
            rel_tol=1e-5,
            abs_tol=1e-8,
        ),
        "seed %d Ward inertia is inconsistent" % seed,
    )
    return seed


def _seed_row(
    payload: Mapping[str, object],
    seed: int,
    bootstrap_resamples: int,
) -> Dict[str, object]:
    checkpoint = payload["checkpoint"]
    training = checkpoint["training_metadata"]
    search = payload["gauge_search"]
    counts = search["counts"]
    primary = payload["primary_causal_contrast"]
    decisions = payload["decisions"]
    paired = [
        _finite_float(value, "paired BPB delta")
        for value in primary["batch_bpb_delta_candidate_minus_reference"]
    ]
    lower, upper = paired_bootstrap_interval(
        paired,
        seed=640_000 + seed,
        resamples=bootstrap_resamples,
    )
    delta = _finite_float(
        primary["instantaneous_bpb_delta_candidate_minus_reference"],
        "primary BPB delta",
    )
    excludes_zero = lower > 0.0 or upper < 0.0
    unigram = _finite_float(checkpoint["add_one_byte_unigram_bpb"], "unigram BPB")
    test_bpb = _finite_float(training["test_bpb"], "soft-model test BPB")
    movement = _finite_float(training["router_mean_l1_movement"], "router movement")
    weak_model = unigram - test_bpb < 0.10
    initialization_dominated = movement < 0.01
    original_fold = decisions["partition_fold_original"]
    gauged_fold = decisions["partition_fold_gauged"]
    kmeans = decisions["effective_theta_kmeans"]
    ward = decisions["effective_theta_ward"]
    literal_original = decisions["literal_argmax_original"]
    literal_gauged = decisions["literal_argmax_gauged"]
    group_pass = int(counts["group_pass"])
    return {
        "status": "completed",
        "decision_evaluations_stored": True,
        "gauge_invariant_baselines_stored": True,
        "seed": seed,
        "checkpoint_sha256": checkpoint["file_sha256"],
        "soft_test_bpb": test_bpb,
        "unigram_bpb": unigram,
        "unigram_improvement_bpb": unigram - test_bpb,
        "router_mean_l1_movement": movement,
        "weak_model_flag": weak_model,
        "initialization_dominated_flag": initialization_dominated,
        "condition_number": _finite_float(search["condition_number"], "condition"),
        "selected_trial": int(search["selected_trial"]),
        "search_attempted": int(counts["attempted"]),
        "search_invertible": int(counts["invertible"]),
        "search_condition_pass": int(counts["condition_pass"]),
        "search_simplex_pass": int(counts["simplex_pass"]),
        "search_group_pass": group_pass,
        "search_partition_change": int(counts["partition_change"]),
        "partition_change_rate_given_group_pass": (
            int(counts["partition_change"]) / group_pass if group_pass else 0.0
        ),
        "group_budget": int(payload["group_budget"]),
        "reference_group_sizes": primary["reference_group_sizes"],
        "candidate_group_sizes": primary["candidate_group_sizes"],
        "normalized_partition_disagreement": _finite_float(
            primary["normalized_pairwise_partition_disagreement"],
            "normalized partition disagreement",
        ),
        "reference_fold_bpb": _finite_float(original_fold["instantaneous_bpb"], "reference fold BPB"),
        "candidate_fold_bpb": _finite_float(gauged_fold["instantaneous_bpb"], "candidate fold BPB"),
        "delta_bpb": delta,
        "absolute_delta_bpb": abs(delta),
        "paired_bootstrap_ci_lower": lower,
        "paired_bootstrap_ci_upper": upper,
        "paired_ci_excludes_zero": excludes_zero,
        "seed_measurable_effect": (
            abs(delta) >= PRIMARY_EFFECT_THRESHOLD_BPB and excludes_zero
        ),
        "reference_fold_sse": _finite_float(original_fold["effective_fit"]["squared_error"], "reference SSE"),
        "candidate_fold_sse": _finite_float(gauged_fold["effective_fit"]["squared_error"], "candidate SSE"),
        "kmeans_bpb": _finite_float(kmeans["instantaneous_bpb"], "k-means BPB"),
        "kmeans_sse": _finite_float(kmeans["effective_fit"]["squared_error"], "k-means SSE"),
        "ward_bpb": _finite_float(ward["instantaneous_bpb"], "Ward BPB"),
        "ward_sse": _finite_float(ward["effective_fit"]["squared_error"], "Ward SSE"),
        "literal_original_bpb_secondary": _finite_float(literal_original["instantaneous_bpb"], "literal original BPB"),
        "literal_gauged_bpb_secondary": _finite_float(literal_gauged["instantaneous_bpb"], "literal gauged BPB"),
    }


def _failed_seed_row(payload: Mapping[str, object], seed: int) -> Dict[str, object]:
    checkpoint = payload["checkpoint"]
    training = checkpoint["training_metadata"]
    search = payload["gauge_search"]
    counts = search["counts"]
    unigram = _finite_float(checkpoint["add_one_byte_unigram_bpb"], "unigram BPB")
    test_bpb = _finite_float(training["test_bpb"], "soft-model test BPB")
    movement = _finite_float(training["router_mean_l1_movement"], "router movement")
    group_pass = int(counts["group_pass"])
    return {
        "status": "no_partition_changing_gauge_found",
        # The released runner exits after the finite search fails.  Do not
        # imply that missing fold or invariant-baseline numbers were evaluated
        # and merely omitted.
        "decision_evaluations_stored": False,
        "gauge_invariant_baselines_stored": False,
        "seed": seed,
        "checkpoint_sha256": checkpoint["file_sha256"],
        "soft_test_bpb": test_bpb,
        "unigram_bpb": unigram,
        "unigram_improvement_bpb": unigram - test_bpb,
        "router_mean_l1_movement": movement,
        "weak_model_flag": unigram - test_bpb < 0.10,
        "initialization_dominated_flag": movement < 0.01,
        "search_attempted": int(counts["attempted"]),
        "search_invertible": int(counts["invertible"]),
        "search_condition_pass": int(counts["condition_pass"]),
        "search_simplex_pass": int(counts["simplex_pass"]),
        "search_group_pass": group_pass,
        "search_partition_change": 0,
        "partition_change_rate_given_group_pass": 0.0,
        "group_budget": int(search["original_groups"]),
        "original_assignment": search["original_assignment"],
        "failure_reason": search["message"],
    }


def summarize_router_decision_results(
    payloads: Sequence[Mapping[str, object]],
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    strict_protocol: bool = True,
) -> Dict[str, object]:
    """Validate raw JSON payloads and compute predeclared seed-level stats."""

    _require(len(payloads) == 3, "the primary audit requires exactly three payloads")
    indexed = {}
    for payload in payloads:
        seed = _validate_seed_payload(payload, strict_protocol=strict_protocol)
        _require(seed not in indexed, "duplicate training seed %d" % seed)
        indexed[seed] = payload
    if strict_protocol:
        _require(
            tuple(sorted(indexed)) == PRIMARY_TRAINING_SEEDS,
            "primary payloads must use training seeds 0, 1, and 2",
        )
        _require(
            bootstrap_resamples == BOOTSTRAP_RESAMPLES,
            "the primary audit requires exactly 10,000 bootstrap resamples",
        )
    rows = [
        (
            _failed_seed_row(indexed[seed], seed)
            if indexed[seed].get("status") == "no_partition_changing_gauge_found"
            else _seed_row(indexed[seed], seed, bootstrap_resamples)
        )
        for seed in sorted(indexed)
    ]
    completed = [row for row in rows if row["status"] == "completed"]
    deltas = [float(row["delta_bpb"]) for row in completed]
    mean_delta = statistics.fmean(deltas) if deltas else None
    standard_deviation = statistics.stdev(deltas) if len(deltas) >= 2 else None
    if len(deltas) == 3:
        half_width = T_CRITICAL_DF2_95 * standard_deviation / math.sqrt(3.0)
        interval_lower = mean_delta - half_width
        interval_upper = mean_delta + half_width
    else:
        interval_lower = None
        interval_upper = None
    measurable = sum(
        bool(row["seed_measurable_effect"]) for row in completed
    )
    weak = sum(bool(row["weak_model_flag"]) for row in rows)
    initialization_dominated = sum(
        bool(row["initialization_dominated_flag"]) for row in rows
    )
    replicated = measurable >= 2
    learned_system_eligible = weak < 2 and initialization_dominated < 2
    executed_search_families = sorted(
        {
            str(payload["gauge_search"]["search_family"])
            for payload in payloads
        }
    )
    initialization_boundary = (
        "All three routers meet the protocol's initialization-dominated flag, "
        "so this is a controlled trained-model witness rather than learned-router "
        "evidence. "
        if initialization_dominated == len(rows)
        else (
            "%d/%d routers meet the protocol's initialization-dominated flag. "
            % (initialization_dominated, len(rows))
        )
    )
    return {
        "protocol": {
            "training_steps": PRIMARY_TRAINING_STEPS,
            "training_seeds": list(PRIMARY_TRAINING_SEEDS),
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_seed_rule": "640000 + training_seed",
            "effect_threshold_bpb": PRIMARY_EFFECT_THRESHOLD_BPB,
            "strict_validation": strict_protocol,
            "executed_search_families": executed_search_families,
            "executed_sensitivity_analyses": [],
            "unexecuted_nonprimary_protocol_amendments": [
                "signed_local_search",
                "same_sorted_group_size_multiset_search",
            ],
        },
        "seed_rows": rows,
        "across_seed": {
            "mean_delta_bpb": mean_delta,
            "sample_standard_deviation_bpb": standard_deviation,
            "t_interval_lower": interval_lower,
            "t_interval_upper": interval_upper,
            "t_critical_df2": T_CRITICAL_DF2_95,
            "preregistered_search_seed_count": len(rows),
            "partition_changing_gauge_found_count": len(completed),
            "partition_changing_gauge_not_found_count": len(rows) - len(completed),
            "finite_search_success_fraction": len(completed) / len(rows),
            "measurable_seed_count": measurable,
            "weak_model_seed_count": weak,
            "initialization_dominated_seed_count": initialization_dominated,
            "replicated_measurable_consequence": replicated,
            "learned_system_eligibility": learned_system_eligible,
            "primary_claim_supported": replicated and learned_system_eligible,
            "existence_chain_observed": bool(completed),
        },
        "claim_boundary": (
            "The finite positive-stochastic search found a same-budget partition "
            "change for only the recorded successful seeds. It supports existence, "
            "not prevalence or reliable replication. "
            + initialization_boundary
            + "Literal "
            "hardening is secondary and confounded; the primary completed contrast "
            "refits both partitions from common effective tensors. The signed-local "
            "and same-sorted-group-size searches are unexecuted non-primary protocol "
            "amendments, not completed sensitivity analyses."
        ),
    }


def markdown_table(summary: Mapping[str, object]) -> str:
    lines = [
        "| Seed | Soft BPB | Router L1 | cond(M) | Group sizes (ref→cand) | Edge disagreement | Fold BPB (ref→cand) | Δ BPB [paired 95% CI] | k-means BPB | Ward BPB | Flags |",
        "|---:|---:|---:|---:|:---|---:|:---|:---|---:|---:|:---|",
    ]
    # Replace a legacy wide header (kept above only to minimize artifact churn)
    # with a failure-aware table used by the revised audit.
    lines = [
        "| Seed | Search status | Soft BPB | Router L1 | Valid changes / group-pass | Partition-only consequence | Gauge-invariant baselines |",
        "|---:|:---|---:|---:|---:|:---|:---|",
    ]
    for row in summary["seed_rows"]:
        if row["status"] != "completed":
            lines.append(
                "| {seed} | no same-budget positive gauge found | {soft:.4f} | "
                "{movement:.4f} | 0 / {group_pass} | not evaluated | "
                "not stored/evaluated (search exited first) |".format(
                    seed=row["seed"],
                    soft=row["soft_test_bpb"],
                    movement=row["router_mean_l1_movement"],
                    group_pass=row["search_group_pass"],
                )
            )
            continue
        flags = []
        if row["weak_model_flag"]:
            flags.append("weak-model")
        if row["initialization_dominated_flag"]:
            flags.append("init-dominated")
        if not row["seed_measurable_effect"]:
            flags.append("below effect rule")
        lines.append(
            "| {seed} | found at trial {trial} | {soft:.4f} | {movement:.4f} | "
            "{changes} / {group_pass} | fold BPB {reference:.4f} -> "
            "{candidate:.4f}; delta {delta:+.4f} [{lower:+.4f}, {upper:+.4f}] "
            "({flags}) | k-means {kmeans:.4f}; Ward {ward:.4f} |".format(
                seed=row["seed"],
                trial=row["selected_trial"],
                soft=row["soft_test_bpb"],
                movement=row["router_mean_l1_movement"],
                changes=row["search_partition_change"],
                group_pass=row["search_group_pass"],
                reference=row["reference_fold_bpb"],
                candidate=row["candidate_fold_bpb"],
                delta=row["delta_bpb"],
                lower=row["paired_bootstrap_ci_lower"],
                upper=row["paired_bootstrap_ci_upper"],
                kmeans=row["kmeans_bpb"],
                ward=row["ward_bpb"],
                flags=", ".join(flags) if flags else "no warning",
            )
        )
        continue
        flags = []
        if row["weak_model_flag"]:
            flags.append("weak-model")
        if row["initialization_dominated_flag"]:
            flags.append("init-dominated")
        if not row["seed_measurable_effect"]:
            flags.append("no-replicated-seed-effect")
        lines.append(
            "| {seed} | {soft:.4f} | {movement:.4f} | {condition:.3f} | {ref}→{cand} | {disagreement:.3f} | {ref_bpb:.4f}→{cand_bpb:.4f} | {delta:+.4f} [{lower:+.4f}, {upper:+.4f}] | {kmeans:.4f} | {ward:.4f} | {flags} |".format(
                seed=row["seed"],
                soft=row["soft_test_bpb"],
                movement=row["router_mean_l1_movement"],
                condition=row["condition_number"],
                ref="/".join(str(value) for value in row["reference_group_sizes"]),
                cand="/".join(str(value) for value in row["candidate_group_sizes"]),
                disagreement=row["normalized_partition_disagreement"],
                ref_bpb=row["reference_fold_bpb"],
                cand_bpb=row["candidate_fold_bpb"],
                delta=row["delta_bpb"],
                lower=row["paired_bootstrap_ci_lower"],
                upper=row["paired_bootstrap_ci_upper"],
                kmeans=row["kmeans_bpb"],
                ward=row["ward_bpb"],
                flags=", ".join(flags) if flags else "none",
            )
        )
    aggregate = summary["across_seed"]
    if aggregate["partition_changing_gauge_found_count"] < 3:
        lines.extend(
            [
                "",
                "The predeclared finite search succeeded for {found}/3 seeds; "
                "measurable completed consequences = {count}/3; replicated "
                "primary claim supported = **{supported}**.".format(
                    found=aggregate["partition_changing_gauge_found_count"],
                    count=aggregate["measurable_seed_count"],
                    supported=str(aggregate["primary_claim_supported"]).lower(),
                ),
                "",
                "Literal-hardening BPB is omitted because it changes basis "
                "coordinates and partition jointly.",
                "",
                "All three routers are initialization-dominated under the "
                "protocol threshold. The signed-local and same-sorted-group-size "
                "searches were not executed; they are non-primary protocol "
                "amendments and have no released outcomes.",
            ]
        )
        return "\n".join(lines) + "\n"
    lines.extend(
        [
            "",
            "Across seeds: mean Δ BPB = {mean:+.4f}, 95% t interval [{lower:+.4f}, {upper:+.4f}]; measurable seeds = {count}/3; primary claim supported = **{supported}**.".format(
                mean=aggregate["mean_delta_bpb"],
                lower=aggregate["t_interval_lower"],
                upper=aggregate["t_interval_upper"],
                count=aggregate["measurable_seed_count"],
                supported=str(aggregate["primary_claim_supported"]).lower(),
            ),
            "",
            "Literal-hardening BPB is intentionally omitted from the primary table because it changes basis coordinates and partition jointly.",
        ]
    )
    return "\n".join(lines) + "\n"


def _csv_rows(rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    keys = sorted({key for row in rows for key in row})
    flattened = []
    for row in rows:
        converted = {key: row.get(key) for key in keys}
        for field in (
            "reference_group_sizes",
            "candidate_group_sizes",
            "original_assignment",
        ):
            value = converted.get(field)
            if isinstance(value, list):
                converted[field] = "/".join(str(item) for item in value)
        flattened.append(converted)
    return flattened


def _write_text_lf(path: Path, text: str) -> None:
    """Write canonical derived text with platform-independent LF endings."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs=3, type=Path)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/router_decisions_summary.json"),
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("results/router_decisions_table.csv"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("results/router_decisions_table.md"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument(
        "--allow-nonprotocol",
        action="store_true",
        help="summarize pilots without enforcing the predeclared settings",
    )
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    summary = summarize_router_decision_results(
        payloads,
        bootstrap_resamples=args.bootstrap_resamples,
        strict_protocol=not args.allow_nonprotocol,
    )
    summary["input_files"] = [_portable_input_path(path) for path in args.inputs]

    for path in (args.json_output, args.csv_output, args.markdown_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_text_lf(
        args.json_output,
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    rows = _csv_rows(summary["seed_rows"])
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_text_lf(args.markdown_output, markdown_table(summary))
    print("wrote %s, %s, and %s" % (args.json_output, args.csv_output, args.markdown_output))


if __name__ == "__main__":
    main()
