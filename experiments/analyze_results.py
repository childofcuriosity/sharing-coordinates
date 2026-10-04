"""Aggregate experiment JSON into paper figures, LaTeX tables, and a summary.

The analysis intentionally depends only on the Python standard library,
NumPy, and Matplotlib.  Every reported experiment value is either copied from
the JSON records or computed as an explicit aggregate of those values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import copy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, TextIO, Union

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np


INPUT_FILES = (
    "gauge_full.json",
    "factorization_full.json",
    "teacher_student_mlp.json",
    "teacher_student_transformer.json",
    "teacher_baselines.json",
    "probes_mlp.json",
    "probes_transformer.json",
    "language_full.json",
)

FACTOR_METRICS = (
    "relative_reconstruction_error",
    "adjusted_rand",
    "assignment_accuracy",
    "graph_accuracy",
    "normalized_entropy",
)
DISTILLATION_METRICS = (
    "test_task_loss",
    "test_hidden_loss",
    "adjusted_rand",
    "assignment_accuracy",
    "graph_accuracy",
    "normalized_entropy",
)
PROBE_METRICS = (
    "adjusted_rand",
    "assignment_accuracy",
    "graph_accuracy",
    "empirical_gap",
)
PROBE_MODES = ("counterfactual", "native", "effective_parameter")
PROBE_PROTOCOL_VERSION = "paired-crn-fixed-gaussian-v1"
LANGUAGE_METRICS = (
    "test_bpb",
    "best_validation_bpb",
    "router_mean_l1_movement",
    "router_entropy",
    "elapsed_seconds",
)


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _value(record: Mapping[str, Any], path: Union[str, Sequence[str]]) -> Any:
    keys = path.split(".") if isinstance(path, str) else path
    current: Any = record
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _config(record: Mapping[str, Any]) -> Mapping[str, Any]:
    config = record.get("config", {})
    return config if isinstance(config, Mapping) else {}


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        nested = payload.get("results")
        if isinstance(nested, list):
            return [item for item in nested if isinstance(item, dict)]
        return [payload]
    return []


def _probe_payload_metadata(payload: Any) -> Optional[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return None
    generation = payload.get("generation")
    return {
        "schema_version": payload.get("schema_version"),
        "protocol_version": payload.get("protocol_version"),
        "protocol_fingerprint_sha256": payload.get(
            "protocol_fingerprint_sha256"
        ),
        "generation": dict(generation) if isinstance(generation, Mapping) else None,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_bound_probe_artifact(label: str, payload: Any) -> None:
    """Verify fingerprint, record count, and generation source bindings."""

    # Unit-test fixtures and historical one-record calls may be plain lists;
    # full sweep artifacts use the bound envelope and are checked strictly.
    if not isinstance(payload, Mapping) or "results" not in payload:
        return
    records = payload.get("results")
    protocol_spec = payload.get("protocol_fingerprint_payload")
    fingerprint = payload.get("protocol_fingerprint_sha256")
    generation = payload.get("generation")
    if not isinstance(records, list):
        raise ValueError(f"{label} has no result list")
    if not isinstance(protocol_spec, Mapping) or not isinstance(fingerprint, str):
        raise ValueError(f"{label} has no protocol fingerprint binding")
    encoded = json.dumps(
        protocol_spec, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != fingerprint:
        raise ValueError(f"{label} protocol fingerprint does not verify")
    if not isinstance(generation, Mapping):
        raise ValueError(f"{label} has no generation mapping")
    if generation.get("record_count") != len(records):
        raise ValueError(f"{label} generation record count does not verify")
    source_hashes = generation.get("source_sha256")
    if not isinstance(source_hashes, Mapping):
        raise ValueError(f"{label} has no generation source binding")
    repository_root = Path(__file__).resolve().parents[1]
    for relative in ("src/probes.py", "experiments/run_probes.py"):
        expected = source_hashes.get(relative)
        if expected != _sha256_file(repository_root / relative):
            raise ValueError(f"{label} source binding failed for {relative}")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _stat(values: Iterable[Any]) -> dict[str, Optional[Union[float, int]]]:
    cleaned = [_finite_float(value) for value in values]
    array = np.asarray([value for value in cleaned if value is not None], dtype=float)
    if array.size == 0:
        return {"mean": None, "std": None, "n": 0}
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(np.mean(array)), "std": std, "n": int(array.size)}


def _metric_stat(
    records: Sequence[Mapping[str, Any]], path: str
) -> dict[str, Optional[Union[float, int]]]:
    return _stat(_value(record, path) for record in records)


def _require_finite_metrics(
    family: str,
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[str],
) -> None:
    """Reject partial canonical sweeps instead of silently shrinking ``n``."""

    for index, record in enumerate(records):
        for path in paths:
            if _finite_float(_value(record, path)) is None:
                raise ValueError(
                    f"{family} record {index} has missing or non-finite metric {path!r}"
                )


def _require_paired_probe_protocol(records: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed on legacy or internally inconsistent probe records."""

    required_true = (
        "observation_noise_scale_is_absolute",
        "same_initial_probes_across_functional_modes",
        "same_observation_noise_across_functional_modes",
        "same_projection_across_functional_modes",
    )
    for index, record in enumerate(records):
        config = _config(record)
        protocol = record.get("protocol")
        if not isinstance(protocol, Mapping):
            raise ValueError(f"probe record {index} has no protocol mapping")
        if protocol.get("version") != PROBE_PROTOCOL_VERSION:
            raise ValueError(
                f"probe record {index} uses unsupported protocol "
                f"{protocol.get('version')!r}"
            )
        if protocol.get("comparison_design") != "paired_common_random_numbers":
            raise ValueError(f"probe record {index} is not a paired CRN comparison")
        for field in required_true:
            if protocol.get(field) is not True:
                raise ValueError(f"probe record {index} does not attest {field}")
        if protocol.get("native_state_evolution") != "clean_update_only":
            raise ValueError(f"probe record {index} has confounded native evolution")
        if protocol.get("empirical_gap_is_population_certificate") is not False:
            raise ValueError(f"probe record {index} mislabels the empirical gap")
        if (
            protocol.get("effective_parameter_baseline_is_functional_equivalence_test")
            is not False
        ):
            raise ValueError(f"probe record {index} mislabels the parameter baseline")
        sigma = _finite_float(config.get("observation_noise"))
        recorded_sigma = _finite_float(protocol.get("observation_noise_scale"))
        if sigma is None or recorded_sigma != sigma:
            raise ValueError(f"probe record {index} has inconsistent noise scale")
        seed_value = config.get("seed")
        try:
            seed = int(seed_value)
        except (TypeError, ValueError):
            raise ValueError(f"probe record {index} has invalid seed")
        expected_seeds = {
            "initial_probe_seed": 40_000 + seed,
            "projection_seed": 50_000 + seed,
            "observation_noise_seed": 60_000 + seed,
        }
        for field, expected in expected_seeds.items():
            if protocol.get(field) != expected:
                raise ValueError(
                    f"probe record {index} has inconsistent {field}: "
                    f"{protocol.get(field)!r} != {expected}"
                )


def _group_by(
    records: Sequence[dict[str, Any]],
    key: Callable[[dict[str, Any]], tuple[Any, ...]],
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return sorted(groups.items(), key=lambda item: tuple(str(part) for part in item[0]))


def _augment_architecture(records: Sequence[dict[str, Any]], architecture: str) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for record in records:
        copied = dict(record)
        config = dict(_config(record))
        config.setdefault("architecture", architecture)
        copied["config"] = config
        augmented.append(copied)
    return augmented


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _architecture_label(value: Any) -> str:
    name = str(value)
    return "MLP" if name == "mlp" else name.capitalize()


def _tex_number(value: Optional[Union[float, int]]) -> str:
    if value is None:
        return "--"
    numeric = float(value)
    if numeric == 0.0:
        return "0"
    magnitude = abs(numeric)
    if 1e-3 <= magnitude < 1e3:
        return f"{numeric:.3f}"
    mantissa, exponent = f"{numeric:.2e}".split("e")
    return rf"{mantissa}\!\times\!10^{{{int(exponent)}}}"


def _tex_stat(stat: Mapping[str, Any]) -> str:
    if not stat.get("n"):
        return "--"
    return rf"${_tex_number(stat.get('mean'))} \pm {_tex_number(stat.get('std'))}$"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text.rstrip() + "\n")


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _factor_key(record: dict[str, Any]) -> tuple[Any, ...]:
    config = _config(record)
    return (
        config.get("method", "unknown"),
        config.get("pattern", "unknown"),
        init_relation(record, "init"),
    )


def _factor_full_key(record: dict[str, Any]) -> tuple[Any, ...]:
    config = _config(record)
    return (
        config.get("method", "unknown"),
        config.get("pattern", "unknown"),
        init_relation(record, "init"),
        config.get("init", "unknown"),
    )


def init_relation(record: Mapping[str, Any], init_field: str) -> str:
    """Classify an initialization relative to that record's teacher graph.

    A raw structured initialization is ``matched`` only for the corresponding
    teacher pattern.  For example, ``pattern_cycle`` is matched for a cycle
    teacher and ``mismatched`` for a contiguous teacher.  This
    record-local classification prevents an aggregation from conflating the
    two experimental conditions.
    """

    config = _config(record)
    raw_init = str(config.get(init_field, "unknown"))
    teacher_pattern = config.get("pattern")
    if raw_init == "neutral":
        return "neutral"
    if raw_init == "random":
        return "random"
    if raw_init.startswith("pattern_"):
        init_pattern = raw_init[len("pattern_") :]
        if teacher_pattern is not None and init_pattern == str(teacher_pattern):
            return "matched"
        paired_patterns = {"cycle", "contiguous"}
        if str(teacher_pattern) in paired_patterns and init_pattern in paired_patterns:
            return "mismatched"
    return "structured_other"


def _distillation_key(record: dict[str, Any]) -> tuple[Any, ...]:
    config = _config(record)
    return (
        config.get("architecture", "unknown"),
        config.get("pattern", "unknown"),
        init_relation(record, "router_init"),
        bool(config.get("hard", False)),
        float(config.get("hidden_weight", 0.0) or 0.0),
        float(config.get("vertex_strength", 0.0) or 0.0),
        float(config.get("usage_strength", 0.0) or 0.0),
    )


def _distillation_full_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return _distillation_key(record) + (_config(record).get("router_init", "unknown"),)


def _probe_key(record: dict[str, Any]) -> tuple[Any, ...]:
    config = _config(record)
    return (config.get("architecture", "unknown"),)


def _language_key(record: dict[str, Any]) -> tuple[Any, ...]:
    config = _config(record)
    return (
        config.get("router_init", "unknown"),
        bool(config.get("router_trainable", True)),
        config.get("router_pattern") or "--",
        bool(config.get("hard", False)),
    )


def _summarize_groups(
    groups: Sequence[tuple[tuple[Any, ...], list[dict[str, Any]]]],
    names: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    result = []
    for key, records in groups:
        item: dict[str, Any] = {name: value for name, value in zip(names, key)}
        item["n_records"] = len(records)
        item["metrics"] = {metric: _metric_stat(records, metric) for metric in metrics}
        result.append(item)
    return result


def _attach_init_metadata(
    summaries: list[dict[str, Any]],
    groups: Sequence[tuple[tuple[Any, ...], list[dict[str, Any]]]],
    init_field: str,
) -> None:
    """Retain raw names for auditability without using them as the comparison axis."""

    for summary, (_, records) in zip(summaries, groups):
        summary["raw_initializations"] = sorted(
            {str(_config(record).get(init_field, "unknown")) for record in records}
        )
        summary["teacher_patterns"] = sorted(
            {str(_config(record).get("pattern", "unknown")) for record in records}
        )


def _factorization_figure(records: list[dict[str, Any]], path: Path) -> bool:
    controlled = [
        record
        for record in records
        if _config(record).get("method") == "soft"
        and _config(record).get("pattern") in {"cycle", "contiguous"}
        and init_relation(record, "init") in {"matched", "mismatched", "neutral", "random"}
    ]
    groups = _group_by(
        controlled,
        lambda record: (_config(record).get("pattern"), init_relation(record, "init")),
    )
    points = []
    for key, members in groups:
        loss = _metric_stat(members, "relative_reconstruction_error")
        ari = _metric_stat(members, "adjusted_rand")
        if loss["mean"] is not None and ari["mean"] is not None:
            points.append((key, loss, ari))
    if not points:
        return False

    pattern_order = [name for name in ("cycle", "contiguous") if any(str(key[0]) == name for key, _, _ in points)]
    relation_order = [
        name
        for name in ("matched", "mismatched", "neutral", "random")
        if any(str(key[1]) == name for key, _, _ in points)
    ]
    colors = {
        relation: plt.get_cmap("tab10")(index % 10)
        for index, relation in enumerate(relation_order)
    }
    marker_choices = ("o", "s", "^", "D")
    markers = {
        relation: marker_choices[index % len(marker_choices)]
        for index, relation in enumerate(relation_order)
    }

    fig, axes = plt.subplots(1, len(pattern_order), figsize=(7.2, 3.25), sharey=True, squeeze=False)
    for column, pattern in enumerate(pattern_order):
        ax = axes[0, column]
        pattern_points = [(key, loss, ari) for key, loss, ari in points if str(key[0]) == pattern]
        x_values = []
        for key, loss, ari in pattern_points:
            relation = str(key[1])
            x = float(loss["mean"])
            y = float(ari["mean"])
            x_values.append(x)
            xerr = float(loss["std"] or 0.0)
            if x > 0:
                xerr = min(xerr, 0.99 * x)
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=float(ari["std"] or 0.0),
                color=colors[relation],
                marker=markers[relation],
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.35,
                capsize=2,
                linestyle="none",
                alpha=0.9,
            )
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.locator_params(axis="x", nbins=4)
        ax.margins(x=0.12)
        ax.set_title(pattern.capitalize())
        ax.set_xlabel("Relative reconstruction error")
        ax.grid(True, which="both", alpha=0.22)
    axes[0, 0].set_ylabel("Adjusted Rand index")
    relation_handles = [
        Line2D(
            [0],
            [0],
            color=colors[name],
            marker=markers[name],
            linestyle="none",
            label=name,
        )
        for name in relation_order
    ]
    fig.legend(
        handles=relation_handles,
        title="Initialization relation",
        fontsize=7,
        title_fontsize=8,
        loc="lower center",
        ncol=max(1, len(relation_handles)),
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.88, wspace=0.18)
    _save_figure(fig, path)
    return True


def _distillation_label(key: tuple[Any, ...]) -> str:
    architecture, teacher_pattern, relation, hard, hidden, vertex, usage = key
    modifiers = []
    if hard:
        modifiers.append("hard")
    if hidden:
        modifiers.append(f"hidden={hidden:g}")
    if vertex:
        modifiers.append(f"vertex={vertex:g}")
    if usage:
        modifiers.append(f"usage={usage:g}")
    suffix = "; " + ", ".join(modifiers) if modifiers else ""
    return f"{_architecture_label(architecture)}: {teacher_pattern} / {relation}{suffix}"


def _distillation_figure(records: list[dict[str, Any]], path: Path) -> bool:
    controlled = [
        record
        for record in records
        if _config(record).get("pattern") in {"cycle", "contiguous"}
        and init_relation(record, "router_init") in {"matched", "mismatched", "neutral"}
    ]
    groups = _group_by(
        controlled,
        lambda record: (
            str(_config(record).get("architecture", "unknown")),
            init_relation(record, "router_init"),
        ),
    )
    points = []
    for key, members in groups:
        loss = _metric_stat(members, "test_task_loss")
        ari = _metric_stat(members, "adjusted_rand")
        if loss["mean"] is not None and ari["mean"] is not None:
            points.append((key, loss, ari))
    if not points:
        return False

    architecture_priority = {"mlp": 0, "transformer": 1}
    architectures = sorted(
        {str(key[0]) for key, _, _ in points},
        key=lambda name: (architecture_priority.get(name, 99), name),
    )
    relations = [
        name
        for name in ("matched", "mismatched", "neutral")
        if any(str(key[1]) == name for key, _, _ in points)
    ]
    colors = {name: plt.get_cmap("tab10")(index) for index, name in enumerate(relations)}
    marker_choices = ("o", "s", "^")
    markers = {name: marker_choices[index] for index, name in enumerate(relations)}

    width = 4.3 if len(architectures) == 1 else 7.2
    fig, axes = plt.subplots(1, len(architectures), figsize=(width, 3.25), sharey=True, squeeze=False)
    for column, architecture in enumerate(architectures):
        ax = axes[0, column]
        architecture_points = [
            (key, loss, ari) for key, loss, ari in points if str(key[0]) == architecture
        ]
        x_values = []
        for key, loss, ari in architecture_points:
            relation = str(key[1])
            x = float(loss["mean"])
            x_values.append(x)
            xerr = float(loss["std"] or 0.0)
            if x > 0:
                xerr = min(xerr, 0.99 * x)
            ax.errorbar(
                x,
                float(ari["mean"]),
                xerr=xerr,
                yerr=float(ari["std"] or 0.0),
                color=colors[relation],
                marker=markers[relation],
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.35,
                capsize=2,
                linestyle="none",
                alpha=0.9,
            )
        # Each architecture has a narrow but different loss scale.  A linear
        # axis with a scientific offset is more legible here than dense log
        # minor ticks, and the panels must not imply cross-architecture scale.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.margins(x=0.15)
        ax.set_title(_architecture_label(architecture))
        ax.set_xlabel("Test distillation loss")
        ax.grid(True, which="both", alpha=0.22)
    axes[0, 0].set_ylabel("Adjusted Rand index")
    relation_handles = [
        Line2D(
            [0],
            [0],
            color=colors[name],
            marker=markers[name],
            linestyle="none",
            label=name,
        )
        for name in relations
    ]
    fig.legend(
        handles=relation_handles,
        title="Initialization relation",
        fontsize=7,
        title_fontsize=8,
        loc="lower center",
        ncol=max(1, len(relation_handles)),
    )
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.88, wspace=0.18)
    _save_figure(fig, path)
    return True


def _probe_phase_figure(records: list[dict[str, Any]], path: Path) -> bool:
    architectures = sorted({str(_config(record).get("architecture", "unknown")) for record in records})
    if not architectures:
        return False
    panels: dict[tuple[str, str], tuple[list[float], list[float], np.ndarray]] = {}
    all_values = []
    for architecture in architectures:
        subset = [record for record in records if str(_config(record).get("architecture", "unknown")) == architecture]
        probe_counts = sorted(
            {float(value) for record in subset if (value := _finite_float(_config(record).get("num_probes"))) is not None}
        )
        noise_levels = sorted(
            {
                float(value)
                for record in subset
                if (value := _finite_float(_config(record).get("observation_noise"))) is not None
            }
        )
        if not probe_counts or not noise_levels:
            continue
        for mode in PROBE_MODES:
            matrix = np.full((len(noise_levels), len(probe_counts)), np.nan)
            for row, noise in enumerate(noise_levels):
                for column, count in enumerate(probe_counts):
                    cell = [
                        record
                        for record in subset
                        if _finite_float(_config(record).get("num_probes")) == count
                        and _finite_float(_config(record).get("observation_noise")) == noise
                    ]
                    stat = _metric_stat(cell, f"{mode}.adjusted_rand")
                    if stat["mean"] is not None:
                        matrix[row, column] = float(stat["mean"])
                        all_values.append(float(stat["mean"]))
            panels[(architecture, mode)] = (probe_counts, noise_levels, matrix)
    if not panels:
        return False

    fig, axes = plt.subplots(
        len(architectures),
        len(PROBE_MODES),
        figsize=(12.0, 3.2 * len(architectures)),
        squeeze=False,
    )
    image = None
    for row, architecture in enumerate(architectures):
        for column, mode in enumerate(PROBE_MODES):
            ax = axes[row, column]
            panel = panels.get((architecture, mode))
            if panel is None:
                ax.axis("off")
                continue
            probe_counts, noise_levels, matrix = panel
            masked = np.ma.masked_invalid(matrix)
            cmap = copy(plt.get_cmap("viridis"))
            cmap.set_bad("0.9")
            image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, vmin=-0.2, vmax=1.0)
            ax.set_xticks(range(len(probe_counts)), [f"{count:g}" for count in probe_counts])
            ax.set_yticks(range(len(noise_levels)), [f"{noise:g}" for noise in noise_levels])
            ax.set_xlabel("Configured probe count")
            ax.set_ylabel("Observation noise")
            mode_label = {
                "counterfactual": "Shared input",
                "native": "Native",
                "effective_parameter": "Effective parameter",
            }[mode]
            ax.set_title(f"{_architecture_label(architecture)} - {mode_label}")
            if matrix.size <= 50:
                for cell_row in range(matrix.shape[0]):
                    for cell_column in range(matrix.shape[1]):
                        value = matrix[cell_row, cell_column]
                        if math.isfinite(float(value)):
                            color = "white" if value < 0.35 else "black"
                            ax.text(cell_column, cell_row, f"{value:.2f}", ha="center", va="center", fontsize=6, color=color)
    if image is not None:
        color_axis = fig.add_axes([0.925, 0.15, 0.014, 0.70])
        fig.colorbar(image, cax=color_axis, label="Mean adjusted Rand index")
    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.10, top=0.92, hspace=0.38, wspace=0.28)
    _save_figure(fig, path)
    return True


def _gauge_figure(payload: Mapping[str, Any], path: Path) -> bool:
    trials = payload.get("gauge_trials", [])
    trials = [trial for trial in trials if isinstance(trial, dict)] if isinstance(trials, list) else []
    x = [_finite_float(trial.get("coassignment_relative_change")) for trial in trials]
    y = [_finite_float(trial.get("effective_max_abs_error")) for trial in trials]
    color = [_gauge_partition_ari(trial) for trial in trials]
    valid = [(a, b, c) for a, b, c in zip(x, y, color) if a is not None and b is not None and c is not None]
    if not valid:
        return False
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    scatter = ax.scatter(
        [item[0] for item in valid],
        [item[1] for item in valid],
        c=[item[2] for item in valid],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        edgecolor="black",
        linewidth=0.3,
    )
    if all(item[1] > 0 for item in valid):
        ax.set_yscale("log")
    ax.set_xlabel("Relative change in router co-assignment")
    ax.set_ylabel("Maximum effective-weight error")
    ax.grid(True, which="both", alpha=0.22)
    fig.colorbar(scatter, ax=ax, label="Partition adjusted Rand index")
    _save_figure(fig, path)
    return True


def _saturation_figure(payload: Mapping[str, Any], path: Path) -> bool:
    raw = payload.get("saturation", [])
    records = [record for record in raw if isinstance(record, dict)] if isinstance(raw, list) else []
    temperatures = sorted(
        {float(value) for record in records if (value := _finite_float(record.get("temperature"))) is not None}
    )
    margins = sorted({float(value) for record in records if (value := _finite_float(record.get("margin"))) is not None})
    if not temperatures or not margins:
        return False
    matrix = np.full((len(temperatures), len(margins)), np.nan)
    for row, temperature in enumerate(temperatures):
        for column, margin in enumerate(margins):
            cell = [
                record
                for record in records
                if _finite_float(record.get("temperature")) == temperature
                and _finite_float(record.get("margin")) == margin
            ]
            stat = _metric_stat(cell, "adjusted_rand")
            if stat["mean"] is not None:
                matrix[row, column] = float(stat["mean"])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    masked = np.ma.masked_invalid(matrix)
    cmap = copy(plt.get_cmap("viridis"))
    cmap.set_bad("0.9")
    image = ax.imshow(masked, origin="lower", aspect="auto", cmap=cmap, vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(margins)), [f"{value:g}" for value in margins])
    ax.set_yticks(range(len(temperatures)), [f"{value:g}" for value in temperatures])
    ax.set_xlabel("Wrong-initialization logit margin")
    ax.set_ylabel("Softmax temperature")
    fig.colorbar(image, ax=ax, label="Mean adjusted Rand index")
    _save_figure(fig, path)
    return True


def _language_figure(records: list[dict[str, Any]], path: Path) -> bool:
    groups = _group_by(records, _language_key)
    points = []
    for key, members in groups:
        test = _metric_stat(members, "test_bpb")
        movement = _metric_stat(members, "router_mean_l1_movement")
        if test["mean"] is not None and movement["mean"] is not None:
            points.append((key, test, movement))
    if not points:
        return False
    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    colors = plt.get_cmap("tab10")
    for index, (key, test, movement) in enumerate(points):
        label = f"{key[0]} / {'learned' if key[1] else 'fixed'} / {key[2]}"
        ax.errorbar(
            float(movement["mean"]),
            float(test["mean"]),
            xerr=float(movement["std"] or 0.0),
            yerr=float(test["std"] or 0.0),
            marker="o",
            linestyle="none",
            color=colors(index % 10),
            capsize=2,
            label=label,
        )
    ax.set_xlabel("Mean final–initial router L1 movement")
    ax.set_ylabel("Test bits per byte (lower is better)")
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=6, loc="best")
    _save_figure(fig, path)
    return True


def _factorization_main_groups(
    records: Sequence[dict[str, Any]],
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        config = _config(record)
        if config.get("pattern") not in {"cycle", "contiguous"}:
            continue
        method = str(config.get("method", "unknown"))
        relation = init_relation(record, "init")
        if method == "soft" and relation in {"matched", "mismatched", "neutral", "random"}:
            groups[(method, relation)].append(record)
        elif method in {"kmeans", "oracle"}:
            groups[(method, "one-hot")].append(record)
    order = {
        ("soft", "matched"): 0,
        ("soft", "mismatched"): 1,
        ("soft", "neutral"): 2,
        ("soft", "random"): 3,
        ("kmeans", "one-hot"): 4,
        ("oracle", "one-hot"): 5,
    }
    return sorted(groups.items(), key=lambda item: order.get(item[0], 99))


def _factorization_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Controlled exact-factorization comparison on cycle and contiguous teachers. Values are mean $\pm$ sample standard deviation across both patterns and seeds.}",
        r"\label{tab:factorization}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Method & Init relation & Rel.\ error & ARI & Assignment acc. & $n$ \\",
        r"\midrule",
    ]
    for (method, relation), members in _factorization_main_groups(records):
        lines.append(
            f"{_tex_escape(method)} & {_tex_escape(relation)} & "
            f"{_tex_stat(_metric_stat(members, 'relative_reconstruction_error'))} & "
            f"{_tex_stat(_metric_stat(members, 'adjusted_rand'))} & "
            f"{_tex_stat(_metric_stat(members, 'assignment_accuracy'))} & {len(members)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines)


def _factorization_full_table(records: list[dict[str, Any]]) -> str:
    lines = ["% Auto-generated by experiments/analyze_results.py; do not edit by hand."]
    preferred_patterns = ("cycle", "contiguous", "palindrome", "random_balanced", "imbalanced")
    observed_patterns = {str(_config(record).get("pattern", "unknown")) for record in records}
    patterns = [pattern for pattern in preferred_patterns if pattern in observed_patterns]
    patterns.extend(sorted(observed_patterns.difference(patterns)))
    method_names = {
        "kmeans": "k-means",
        "straight_through": "straight-through",
        "vertex_hard": "vertex-hard",
    }
    init_names = {
        "pattern_contiguous": "contiguous",
        "pattern_cycle": "cycle",
        "structured_other": "structured-other",
    }
    for index, teacher_pattern in enumerate(patterns):
        members_for_pattern = [
            record for record in records if str(_config(record).get("pattern", "unknown")) == teacher_pattern
        ]
        teacher_name = teacher_pattern.replace("_", " ")
        label = "tab:factorization-full" if index == 0 else f"tab:factorization-full-{teacher_pattern.replace('_', '-')}"
        lines.extend(
            [
                r"\begin{table}[H]",
                r"\centering",
                r"\scriptsize",
                rf"\caption{{Exact-factorization results for { _tex_escape(teacher_name) } teachers. Values are mean $\pm$ sample standard deviation across seeds.}}",
                rf"\label{{{label}}}",
                r"\resizebox{\linewidth}{!}{%",
                r"\begin{tabular}{lllrrrr}",
                r"\toprule",
                r"Method & Init relation & Raw init & Rel.\ error & ARI & Assignment acc. & $n$ \\",
                r"\midrule",
            ]
        )
        for (method, _pattern, relation, raw_init), group in _group_by(members_for_pattern, _factor_full_key):
            display_method = method_names.get(str(method), str(method))
            display_relation = init_names.get(str(relation), str(relation))
            display_init = init_names.get(str(raw_init), str(raw_init))
            lines.append(
                f"{_tex_escape(display_method)} & {_tex_escape(display_relation)} & {_tex_escape(display_init)} & "
                f"{_tex_stat(_metric_stat(group, 'relative_reconstruction_error'))} & "
                f"{_tex_stat(_metric_stat(group, 'adjusted_rand'))} & "
                f"{_tex_stat(_metric_stat(group, 'assignment_accuracy'))} & {len(group)} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines)


def _distillation_main_groups(
    records: Sequence[dict[str, Any]],
) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        config = _config(record)
        relation = init_relation(record, "router_init")
        is_baseline = (
            not bool(config.get("hard", False))
            and float(config.get("hidden_weight", 0.0) or 0.0) == 0.0
            and float(config.get("vertex_strength", 0.0) or 0.0) == 0.0
            and float(config.get("usage_strength", 0.0) or 0.0) == 0.0
        )
        if (
            config.get("pattern") in {"cycle", "contiguous"}
            and relation in {"matched", "mismatched", "neutral"}
            and is_baseline
        ):
            groups[(str(config.get("architecture", "unknown")), relation)].append(record)
    architecture_order = {"mlp": 0, "transformer": 1}
    relation_order = {"matched": 0, "mismatched": 1, "neutral": 2}
    return sorted(
        groups.items(),
        key=lambda item: (
            architecture_order.get(item[0][0], 99),
            relation_order.get(item[0][1], 99),
        ),
    )


def _distillation_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Controlled functional-distillation comparison on cycle and contiguous teachers. Values are mean $\pm$ sample standard deviation across both patterns and seeds for soft, task-only students.}",
        r"\label{tab:distillation}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Architecture & Init relation & Test loss & ARI & $n$ \\",
        r"\midrule",
    ]
    for (architecture, relation), members in _distillation_main_groups(records):
        lines.append(
            f"{_tex_escape(_architecture_label(architecture))} & {_tex_escape(relation)} & "
            f"{_tex_stat(_metric_stat(members, 'test_task_loss'))} & "
            f"{_tex_stat(_metric_stat(members, 'adjusted_rand'))} & {len(members)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _distillation_full_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Full functional-distillation results by architecture, teacher pattern, and raw initialization. Values are mean $\pm$ sample standard deviation.}",
        r"\label{tab:distillation-full}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Student / init relation & Raw init & Test loss & ARI & $n$ \\",
        r"\midrule",
    ]
    for key, members in _group_by(records, _distillation_full_key):
        base_key, raw_init = key[:-1], key[-1]
        lines.append(
            f"{_tex_escape(_distillation_label(base_key))} & {_tex_escape(raw_init)} & "
            f"{_tex_stat(_metric_stat(members, 'test_task_loss'))} & "
            f"{_tex_stat(_metric_stat(members, 'adjusted_rand'))} & {len(members)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"])
    return "\n".join(lines)


def _probe_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Paired probe recovery aggregated over probe counts, fixed additive-noise levels, graphs, and seeds. Functional modes share inputs, noise, and projection. The visible-effective-parameter baseline uses neither observation noise nor projection and is not a test of functional equivalence. Values are mean $\pm$ sample standard deviation.}",
        r"\label{tab:probes}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Architecture & Signature & ARI & Empirical gap & $n$ \\",
        r"\midrule",
    ]
    mode_labels = {
        "counterfactual": "shared input",
        "native": "native",
        "effective_parameter": "effective parameter",
    }
    for (architecture,), members in _group_by(records, _probe_key):
        for mode in PROBE_MODES:
            lines.append(
                f"{_tex_escape(_architecture_label(architecture))} & {_tex_escape(mode_labels[mode])} & "
                f"{_tex_stat(_metric_stat(members, mode + '.adjusted_rand'))} & "
                f"{_tex_stat(_metric_stat(members, mode + '.empirical_gap'))} & {len(members)} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _paired_probe_comparison(
    records: Sequence[Mapping[str, Any]], first: str, second: str
) -> dict[str, Any]:
    differences = []
    differences_by_seed: dict[int, list[float]] = defaultdict(list)
    wins = 0
    ties = 0
    losses = 0
    for record in records:
        first_value = _finite_float(_value(record, f"{first}.adjusted_rand"))
        second_value = _finite_float(_value(record, f"{second}.adjusted_rand"))
        if first_value is None or second_value is None:
            continue
        difference = first_value - second_value
        differences.append(difference)
        seed_value = _config(record).get("seed")
        try:
            differences_by_seed[int(seed_value)].append(difference)
        except (TypeError, ValueError):
            pass
        if difference > 1e-12:
            wins += 1
        elif difference < -1e-12:
            losses += 1
        else:
            ties += 1
    per_seed_means = [
        float(np.mean(values))
        for _, values in sorted(differences_by_seed.items())
        if values
    ]
    seed_wins = sum(value > 1e-12 for value in per_seed_means)
    seed_losses = sum(value < -1e-12 for value in per_seed_means)
    seed_ties = len(per_seed_means) - seed_wins - seed_losses
    return {
        "first": first,
        "second": second,
        "adjusted_rand_difference": _stat(differences),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "per_seed_mean_difference": {
            **_stat(per_seed_means),
            "min": min(per_seed_means) if per_seed_means else None,
            "max": max(per_seed_means) if per_seed_means else None,
            "positive_seeds": seed_wins,
            "zero_seeds": seed_ties,
            "negative_seeds": seed_losses,
        },
    }


def _comb2(value: int) -> float:
    return value * (value - 1) / 2.0


def _adjusted_rand(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> Optional[float]:
    if len(labels_a) != len(labels_b) or len(labels_a) < 2:
        return None
    contingency: dict[tuple[Any, Any], int] = defaultdict(int)
    rows: dict[Any, int] = defaultdict(int)
    columns: dict[Any, int] = defaultdict(int)
    for first, second in zip(labels_a, labels_b):
        contingency[(first, second)] += 1
        rows[first] += 1
        columns[second] += 1
    total_pairs = _comb2(len(labels_a))
    sum_cells = sum(_comb2(value) for value in contingency.values())
    sum_rows = sum(_comb2(value) for value in rows.values())
    sum_columns = sum(_comb2(value) for value in columns.values())
    expected = sum_rows * sum_columns / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (sum_rows + sum_columns)
    denominator = maximum - expected
    if abs(denominator) < 1e-15:
        return 1.0 if sum_cells == maximum else 0.0
    return float((sum_cells - expected) / denominator)


def _gauge_partition_ari(record: Mapping[str, Any]) -> Optional[float]:
    before = record.get("alpha")
    after = record.get("transformed_alpha")
    if not isinstance(before, list) or not isinstance(after, list) or len(before) != len(after):
        return None
    try:
        before_labels = [max(range(len(row)), key=lambda index: row[index]) for row in before]
        after_labels = [max(range(len(row)), key=lambda index: row[index]) for row in after]
    except (TypeError, ValueError):
        return None
    return _adjusted_rand(before_labels, after_labels)


def _gauge_partition_changed(record: Mapping[str, Any]) -> Optional[bool]:
    score = _gauge_partition_ari(record)
    return None if score is None else abs(score - 1.0) > 1e-12


def _pairwise_router_stability(
    records: Sequence[dict[str, Any]],
) -> dict[str, Optional[Union[float, int]]]:
    assignments = [record.get("router_argmax") for record in records]
    valid = [assignment for assignment in assignments if isinstance(assignment, list)]
    scores = []
    for first, second in combinations(valid, 2):
        score = _adjusted_rand(first, second)
        if score is not None:
            scores.append(score)
    return _stat(scores)


def _language_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        r"\caption{Language-model stress test. BPB is estimated from 100 seeded, sampled test batches. Values are mean $\pm$ sample standard deviation across seeds; pairwise ARI descriptively summarizes the three dependent run pairs.}",
        r"\label{tab:language}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Initialization & Sampled test BPB & L1 movement & Pairwise ARI & $n$ \\",
        r"\midrule",
    ]
    for key, members in _group_by(records, _language_key):
        init, _trainable, _pattern, _hard = key
        lines.append(
            f"{_tex_escape(init)} & {_tex_stat(_metric_stat(members, 'test_bpb'))} & "
            f"{_tex_stat(_metric_stat(members, 'router_mean_l1_movement'))} & "
            f"{_tex_stat(_pairwise_router_stability(members))} & {len(members)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def _gauge_tables(payload: Mapping[str, Any]) -> dict[str, str]:
    outputs: dict[str, str] = {}
    raw_trials = payload.get("gauge_trials", [])
    trials = [record for record in raw_trials if isinstance(record, dict)] if isinstance(raw_trials, list) else []
    if trials:
        trials = [dict(record, partition_ari=_gauge_partition_ari(record)) for record in trials]
        metrics = (
            ("Effective max error", "effective_max_abs_error"),
            ("Argmax-partition ARI", "partition_ari"),
            ("Co-assignment relative change", "coassignment_relative_change"),
            ("Post-step effective relative difference", "post_step_effective_relative_difference"),
        )
        lines = [
            "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
            r"\begin{table}[H]",
            r"\centering",
            r"\small",
            r"\caption{Gauge counterfactuals. Values are mean $\pm$ sample standard deviation across trials.}",
            r"\label{tab:gauge}",
            r"\begin{tabular}{lrr}",
            r"\toprule",
            r"Metric & Value & $n$ \\",
            r"\midrule",
        ]
        for label, metric in metrics:
            stat = _metric_stat(trials, metric)
            lines.append(f"{label} & {_tex_stat(stat)} & {stat['n']} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        outputs["gauge_table.tex"] = "\n".join(lines)

    raw_saturation = payload.get("saturation", [])
    saturation = [record for record in raw_saturation if isinstance(record, dict)] if isinstance(raw_saturation, list) else []
    if saturation:
        groups = _group_by(saturation, lambda record: (record.get("temperature"), record.get("margin")))
        lines = [
            "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
            r"\begin{table}[H]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Softmax-saturation sweep. Values are aggregated across seeds for each temperature and wrong-initialization margin.}",
            r"\label{tab:saturation}",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"Temperature & Margin & Initial grad. & Final MSE & ARI & $n$ \\",
            r"\midrule",
        ]
        for (temperature, margin), members in groups:
            lines.append(
                f"{_tex_number(_finite_float(temperature))} & {_tex_number(_finite_float(margin))} & "
                f"{_tex_stat(_metric_stat(members, 'initial_gradient_norm'))} & "
                f"{_tex_stat(_metric_stat(members, 'final_mse'))} & "
                f"{_tex_stat(_metric_stat(members, 'adjusted_rand'))} & {len(members)} \\\\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        outputs["saturation_table.tex"] = "\n".join(lines)
    return outputs


def _gauge_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    exact = payload.get("exact_argmax_flip")
    exact_summary = exact if isinstance(exact, dict) else None
    raw_trials = payload.get("gauge_trials", [])
    trials = [record for record in raw_trials if isinstance(record, dict)] if isinstance(raw_trials, list) else []
    trial_metrics = (
        "effective_max_abs_error",
        "original_entropy",
        "transformed_entropy",
        "argmax_agreement",
        "coassignment_relative_change",
        "initial_loss_difference",
        "post_step_effective_relative_difference",
        "original_logit_gradient_norm",
        "transformed_logit_gradient_norm",
    )
    raw_saturation = payload.get("saturation", [])
    saturation = [record for record in raw_saturation if isinstance(record, dict)] if isinstance(raw_saturation, list) else []
    saturation_groups = _group_by(saturation, lambda record: (record.get("temperature"), record.get("margin")))
    return {
        "exact_argmax_flip": exact_summary,
        "gauge_trials": {
            "n_records": len(trials),
            "metrics": {metric: _metric_stat(trials, metric) for metric in trial_metrics},
        },
        "saturation": {
            "n_records": len(saturation),
            "groups": _summarize_groups(
                saturation_groups,
                ("temperature", "margin"),
                ("initial_gradient_norm", "final_mse", "adjusted_rand", "assignment_accuracy", "step_to_90pct"),
            ),
        },
    }


def _key_numbers_tex(payloads: Mapping[str, Any], inputs: Mapping[str, Mapping[str, Any]]) -> str:
    """Create manuscript macros only when their canonical source was loaded."""

    lines = [
        "% Auto-generated by experiments/analyze_results.py; do not edit by hand.",
        "% A macro is omitted when its canonical JSON input or metric is unavailable.",
    ]

    def add_count(name: str, value: int) -> None:
        lines.append(r"\newcommand{\%s}{%d}" % (name, value))

    def add_number(name: str, value: Any) -> None:
        numeric = _finite_float(value)
        if numeric is not None:
            lines.append(r"\newcommand{\%s}{\ensuremath{%s}}" % (name, _tex_number(numeric)))

    count_specs = (
        ("factorization_full.json", "FactorizationRunCount"),
        ("teacher_student_mlp.json", "TeacherStudentMLPRunCount"),
        ("teacher_student_transformer.json", "TeacherStudentTransformerRunCount"),
        ("teacher_baselines.json", "TeacherBaselineRunCount"),
        ("probes_mlp.json", "ProbeMLPRunCount"),
        ("probes_transformer.json", "ProbeTransformerRunCount"),
        ("language_full.json", "LanguageRunCount"),
    )
    for filename, macro in count_specs:
        if inputs.get(filename, {}).get("status") == "loaded":
            add_count(macro, len(_records(payloads.get(filename))))

    if all(inputs.get(name, {}).get("status") == "loaded" for name in (
        "teacher_student_mlp.json",
        "teacher_student_transformer.json",
    )):
        add_count(
            "DistillationRunCount",
            len(_records(payloads.get("teacher_student_mlp.json")))
            + len(_records(payloads.get("teacher_student_transformer.json"))),
        )

    baseline_payload = payloads.get("teacher_baselines.json")
    if inputs.get("teacher_baselines.json", {}).get("status") == "loaded":
        baseline_records = _records(baseline_payload)
        for architecture, macro in (
            ("mlp", "MLPReferenceLoss"),
            ("transformer", "TransformerReferenceLoss"),
        ):
            controlled = [
                record
                for record in baseline_records
                if _config(record).get("architecture") == architecture
                and _config(record).get("pattern") in {"cycle", "contiguous"}
            ]
            add_number(macro, _metric_stat(controlled, "reference_loss").get("mean"))
    if all(inputs.get(name, {}).get("status") == "loaded" for name in (
        "probes_mlp.json",
        "probes_transformer.json",
    )):
        add_count(
            "ProbeRunCount",
            len(_records(payloads.get("probes_mlp.json")))
            + len(_records(payloads.get("probes_transformer.json"))),
        )

    gauge_payload = payloads.get("gauge_full.json")
    if inputs.get("gauge_full.json", {}).get("status") == "loaded" and isinstance(gauge_payload, dict):
        raw_trials = gauge_payload.get("gauge_trials")
        if isinstance(raw_trials, list):
            trials = [trial for trial in raw_trials if isinstance(trial, dict)]
            add_count("GaugeTrialRunCount", len(trials))
            mean_specs = (
                ("effective_max_abs_error", "GaugeTrialMeanEffectiveError"),
                ("coassignment_relative_change", "GaugeTrialMeanCoassignmentChange"),
            )
            for metric, macro in mean_specs:
                add_number(macro, _metric_stat(trials, metric).get("mean"))
            partition_scores = [
                score for trial in trials if (score := _gauge_partition_ari(trial)) is not None
            ]
            add_number("GaugeTrialMeanPartitionARI", _stat(partition_scores).get("mean"))
            add_count(
                "GaugeTrialChangedPartitions",
                sum(_gauge_partition_changed(trial) is True for trial in trials),
            )
        raw_saturation = gauge_payload.get("saturation")
        if isinstance(raw_saturation, list):
            add_count(
                "GaugeSaturationRunCount",
                len([record for record in raw_saturation if isinstance(record, dict)]),
            )

        exact = gauge_payload.get("exact_argmax_flip")
        if isinstance(exact, dict):
            add_number("GaugeExactError", exact.get("effective_max_abs_error"))
            before = exact.get("argmax_before")
            after = exact.get("argmax_after")
            if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
                add_count("GaugeExactArgmaxFlips", sum(first != second for first, second in zip(before, after)))
                add_count("GaugeExactArgmaxTotal", len(before))

    missing = [filename for filename in INPUT_FILES if inputs.get(filename, {}).get("status") != "loaded"]
    if missing:
        lines.append("% Missing canonical inputs (numeric macros omitted): " + ", ".join(missing))
    return "\n".join(lines)


def analyze_results(
    results_dir: Path,
    figures_dir: Path,
    generated_dir: Path,
    summary_path: Path,
    stream: Optional[TextIO] = None,
) -> dict[str, Any]:
    """Read available result files and create all applicable artifacts."""

    stream = stream or sys.stdout
    payloads: dict[str, Any] = {}
    inputs: dict[str, dict[str, Any]] = {}
    for filename in INPUT_FILES:
        path = results_dir / filename
        if not path.is_file():
            inputs[filename] = {"status": "missing", "records": 0}
            print(f"[missing] {filename}; dependent artifacts will be skipped", file=stream)
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError) as error:
            inputs[filename] = {"status": "error", "records": 0, "message": str(error)}
            print(f"[error] {filename}: {error}", file=stream)
            continue
        payloads[filename] = payload
        if filename == "gauge_full.json" and isinstance(payload, dict):
            count = len(payload.get("gauge_trials", [])) + len(payload.get("saturation", []))
        else:
            count = len(_records(payload))
        inputs[filename] = {"status": "loaded", "records": count}
        print(f"[loaded] {filename}: {count} records", file=stream)

    for filename in ("probes_mlp.json", "probes_transformer.json"):
        if filename in payloads:
            _require_bound_probe_artifact(filename, payloads[filename])

    factorization = _records(payloads.get("factorization_full.json", []))
    teacher_mlp = _augment_architecture(_records(payloads.get("teacher_student_mlp.json", [])), "mlp")
    teacher_transformer = _augment_architecture(
        _records(payloads.get("teacher_student_transformer.json", [])), "transformer"
    )
    distillation = teacher_mlp + teacher_transformer
    teacher_baselines = _records(payloads.get("teacher_baselines.json", []))
    probes_mlp = _augment_architecture(_records(payloads.get("probes_mlp.json", [])), "mlp")
    probes_transformer = _augment_architecture(_records(payloads.get("probes_transformer.json", [])), "transformer")
    probes = probes_mlp + probes_transformer
    language = _records(payloads.get("language_full.json", []))
    gauge_payload = payloads.get("gauge_full.json")
    gauge = gauge_payload if isinstance(gauge_payload, dict) else {}

    _require_finite_metrics(
        "factorization",
        factorization,
        ("relative_reconstruction_error", "adjusted_rand", "assignment_accuracy"),
    )
    _require_finite_metrics(
        "distillation", distillation, ("test_task_loss", "adjusted_rand")
    )
    _require_finite_metrics("teacher baselines", teacher_baselines, ("reference_loss",))
    _require_finite_metrics(
        "probes",
        probes,
        (
            "counterfactual.adjusted_rand",
            "counterfactual.empirical_gap",
            "native.adjusted_rand",
            "native.empirical_gap",
            "effective_parameter.adjusted_rand",
            "effective_parameter.empirical_gap",
        ),
    )
    _require_paired_probe_protocol(probes)
    _require_finite_metrics(
        "language", language, ("test_bpb", "router_mean_l1_movement")
    )
    if gauge:
        _require_finite_metrics(
            "gauge trials",
            _records(gauge.get("gauge_trials", [])),
            ("effective_max_abs_error", "coassignment_relative_change"),
        )
        _require_finite_metrics(
            "saturation",
            _records(gauge.get("saturation", [])),
            ("final_mse", "adjusted_rand"),
        )

    generated_figures: list[str] = []
    generated_tables: list[str] = []

    if factorization:
        target = figures_dir / "factorization_loss_vs_ari.pdf"
        if _factorization_figure(factorization, target):
            generated_figures.append(target.name)
        table = generated_dir / "factorization_table.tex"
        _write_text(table, _factorization_table(factorization))
        generated_tables.append(table.name)
        full_table = generated_dir / "factorization_full_table.tex"
        _write_text(full_table, _factorization_full_table(factorization))
        generated_tables.append(full_table.name)
    if distillation:
        target = figures_dir / "distillation_loss_vs_ari.pdf"
        if _distillation_figure(distillation, target):
            generated_figures.append(target.name)
        table = generated_dir / "distillation_table.tex"
        _write_text(table, _distillation_table(distillation))
        generated_tables.append(table.name)
        full_table = generated_dir / "distillation_full_table.tex"
        _write_text(full_table, _distillation_full_table(distillation))
        generated_tables.append(full_table.name)
    if probes:
        target = figures_dir / "probe_phase_diagram.pdf"
        if _probe_phase_figure(probes, target):
            generated_figures.append(target.name)
        table = generated_dir / "probe_table.tex"
        _write_text(table, _probe_table(probes))
        generated_tables.append(table.name)
    if language:
        target = figures_dir / "language_router_tradeoff.pdf"
        if _language_figure(language, target):
            generated_figures.append(target.name)
        table = generated_dir / "language_table.tex"
        _write_text(table, _language_table(language))
        generated_tables.append(table.name)
    if gauge:
        target = figures_dir / "gauge_equivalence.pdf"
        if _gauge_figure(gauge, target):
            generated_figures.append(target.name)
        saturation_target = figures_dir / "saturation_recovery.pdf"
        if _saturation_figure(gauge, saturation_target):
            generated_figures.append(saturation_target.name)
        for filename, content in _gauge_tables(gauge).items():
            table = generated_dir / filename
            _write_text(table, content)
            generated_tables.append(table.name)

    key_numbers = generated_dir / "key_numbers.tex"
    _write_text(key_numbers, _key_numbers_tex(payloads, inputs))
    generated_tables.append(key_numbers.name)

    factor_groups = _group_by(factorization, _factor_key)
    distillation_groups = _group_by(distillation, _distillation_key)
    probe_groups = _group_by(probes, _probe_key)
    language_groups = _group_by(language, _language_key)
    language_summary = _summarize_groups(
        language_groups,
        ("router_init", "router_trainable", "router_pattern", "hard"),
        LANGUAGE_METRICS,
    )
    for item, (_, members) in zip(language_summary, language_groups):
        item["pairwise_router_ari"] = _pairwise_router_stability(members)

    factor_summary = _summarize_groups(
        factor_groups,
        ("method", "teacher_pattern", "init_relation"),
        FACTOR_METRICS,
    )
    _attach_init_metadata(factor_summary, factor_groups, "init")
    distillation_summary = _summarize_groups(
        distillation_groups,
        (
            "architecture",
            "teacher_pattern",
            "init_relation",
            "hard",
            "hidden_weight",
            "vertex_strength",
            "usage_strength",
        ),
        DISTILLATION_METRICS,
    )
    _attach_init_metadata(distillation_summary, distillation_groups, "router_init")

    probe_summary = []
    for (architecture,), members in probe_groups:
        item: dict[str, Any] = {"architecture": architecture, "n_records": len(members), "modes": {}}
        for mode in PROBE_MODES:
            item["modes"][mode] = {
                metric: _metric_stat(members, f"{mode}.{metric}") for metric in PROBE_METRICS
            }
        item["paired_adjusted_rand_comparisons"] = [
            _paired_probe_comparison(members, "counterfactual", "native"),
            _paired_probe_comparison(members, "counterfactual", "effective_parameter"),
            _paired_probe_comparison(members, "native", "effective_parameter"),
        ]
        probe_summary.append(item)

    baseline_groups = _group_by(
        teacher_baselines,
        lambda record: (
            _config(record).get("architecture", "unknown"),
            record.get("reference_type", "unknown"),
        ),
    )
    baseline_summary = _summarize_groups(
        baseline_groups,
        ("architecture", "reference_type"),
        ("reference_loss",),
    )

    summary: dict[str, Any] = {
        "inputs": inputs,
        "factorization": {
            "n_records": len(factorization),
            "groups": factor_summary,
        },
        "distillation": {
            "n_records": len(distillation),
            "groups": distillation_summary,
        },
        "teacher_baselines": {
            "n_records": len(teacher_baselines),
            "groups": baseline_summary,
        },
        "probes": {
            "n_records": len(probes),
            "artifacts": {
                "mlp": _probe_payload_metadata(payloads.get("probes_mlp.json")),
                "transformer": _probe_payload_metadata(
                    payloads.get("probes_transformer.json")
                ),
            },
            "groups": probe_summary,
        },
        "language": {"n_records": len(language), "groups": language_summary},
        "gauge": _gauge_summary(gauge) if gauge else None,
        "generated": {
            "figures": sorted(generated_figures),
            "tables": sorted(generated_tables),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(f"[wrote] {summary_path.name}", file=stream)
    if generated_figures:
        print("[figures] " + ", ".join(sorted(generated_figures)), file=stream)
    if generated_tables:
        print("[tables] " + ", ".join(sorted(generated_tables)), file=stream)
    return summary


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument("--figures-dir", type=Path, default=root / "paper" / "figures")
    parser.add_argument("--generated-dir", type=Path, default=root / "paper" / "generated")
    parser.add_argument("--summary", type=Path, default=root / "results" / "summary.json")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    analyze_results(args.results_dir, args.figures_dir, args.generated_dir, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
