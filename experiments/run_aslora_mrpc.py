"""Train paper-configured ASLoRA on MRPC and audit the first merge decision.

The module imports Hugging Face dependencies only after argument parsing.  Thus
``--help`` and ``--dry-run`` work in the base test environment and never start
a download.  A real run stops at optimizer step 560 immediately before any
hard merge, saves the model/snapshot, and evaluates all temporary candidate
actions on the same complete MRPC validation split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.aslora_training import (
    ASLoRAMRPCConfig,
    PAPER_EVIDENCE,
    ProjectedActivationCollector,
    action_key,
    action_result_json,
    action_to_json,
    build_deterministic_gauge_bank,
    candidate_family_actions,
    configure_deterministic_execution,
    dependency_versions,
    distance_table,
    distance_table_to_json,
    evaluate_action_set,
    evaluate_classifier,
    evaluation_difference,
    gauge_distance_invariance_audit,
    is_scheduled_merge_step,
    load_external_gauge_bank,
    make_optimizer,
    plan_prevalidation_decisions,
    require_hf,
    runtime_provenance,
    save_training_checkpoint,
    select_action_from_distance_table,
    set_global_seed,
    structural_identity_signature,
    temporary_live_gauges,
    temporary_target_action,
    timeline_provenance,
    trainable_tensor_digest,
    validation_loss_oracles,
    write_json,
)
from src.aslora_witness import (
    FirstMergeSnapshot,
    attach_aslora_to_roberta_classifier,
    save_first_merge_snapshot,
)


def _source_sha256() -> Dict[str, str]:
    """Hash the in-repository files that define the first-merge audit."""

    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src" / "aslora_training.py",
        PROJECT_ROOT / "src" / "aslora_witness.py",
    )
    output: Dict[str, str] = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        key = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        output[key] = digest.hexdigest()
    return output


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_inventory(path: Path) -> Dict[str, Any]:
    """Byte-bind a local input file or model directory in provenance."""

    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError("input artifact does not exist: %s" % resolved)
    files = [resolved] if resolved.is_file() else sorted(
        item for item in resolved.rglob("*") if item.is_file()
    )
    if not files:
        raise ValueError("input artifact contains no files: %s" % resolved)
    entries = []
    for item in files:
        relative = item.name if resolved.is_file() else str(
            item.relative_to(resolved)
        ).replace("\\", "/")
        entries.append(
            {
                "relative_path": relative,
                "size_bytes": item.stat().st_size,
                "sha256": _file_sha256(item),
            }
        )
    canonical = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "resolved_path": str(resolved),
        "kind": "file" if resolved.is_file() else "directory",
        "file_count": len(entries),
        "files": entries,
        "inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _model_state_sha256(model: torch.nn.Module) -> str:
    """Hash every initial tensor and buffer after ASLoRA attachment."""

    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            raise TypeError("model state contains a non-tensor entry: %s" % name)
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train RoBERTa-base/ASLoRA on MRPC to the first one-indexed merge "
            "step t=560, then run non-destructive full-validation candidate and "
            "bounded-GL audits. --help and --dry-run never download anything."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/aslora_mrpc_seed0"),
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--model-name", default="FacebookAI/roberta-base")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--train-parquet",
        type=Path,
        default=None,
        help="optional local MRPC train parquet (requires --validation-parquet)",
    )
    parser.add_argument(
        "--validation-parquet",
        type=Path,
        default=None,
        help="optional local MRPC validation parquet (requires --train-parquet)",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="forbid model/tokenizer network access; pair with local model files and parquets",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--candidate-modes",
        nargs="+",
        choices=("adjacent", "all"),
        default=("adjacent", "all"),
        help="both interpretations are reported from one trained snapshot by default",
    )
    parser.add_argument(
        "--gauge-condition-limits",
        nargs="+",
        type=float,
        default=(2.0, 4.0, 8.0, 30.0),
    )
    parser.add_argument(
        "--gauge-witness",
        type=Path,
        action="append",
        default=[],
        help="optional predeclared decision_audit JSON with Q, R, or metric_C",
    )
    parser.add_argument("--max-imported-gauge-condition", type=float, default=30.0)

    # Exact Appendix Table 5 defaults.  Changes require explicit opt-in so a
    # remote launch cannot silently cease to be the paper-configured audit.
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--start-merge-step", type=int, default=320)
    parser.add_argument("--merge-interval", type=int, default=240)
    parser.add_argument("--merge-count", type=int, default=7)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--allow-nondeterministic-algorithms",
        action="store_true",
        help="disable the primary deterministic-algorithm gate (not used for released runs)",
    )
    parser.add_argument(
        "--primary-action-evaluation-repeats",
        type=int,
        default=2,
        help="complete validation passes for each raw-B-selected canonical action",
    )
    parser.add_argument(
        "--allow-paper-config-deviation",
        action="store_true",
        help="permit explicit hyperparameter deviations while preserving provenance",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the schedule without importing Transformers or downloading",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ASLoRAMRPCConfig:
    return ASLoRAMRPCConfig(
        model_name=args.model_name,
        model_revision=args.model_revision,
        dataset_revision=args.dataset_revision,
        train_parquet=(
            None if args.train_parquet is None else str(args.train_parquet.resolve())
        ),
        validation_parquet=(
            None
            if args.validation_parquet is None
            else str(args.validation_parquet.resolve())
        ),
        local_files_only=args.local_files_only,
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        start_merge_step=args.start_merge_step,
        merge_interval=args.merge_interval,
        merge_count=args.merge_count,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        candidate_modes=tuple(args.candidate_modes),
        gauge_condition_limits=tuple(args.gauge_condition_limits),
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        deterministic_algorithms=not args.allow_nondeterministic_algorithms,
        primary_action_evaluation_repeats=args.primary_action_evaluation_repeats,
    )


def _prepare_data_and_model(
    config: ASLoRAMRPCConfig,
    cache_dir: Path,
) -> Tuple[Any, Any, Any, DataLoader, DataLoader, Any, Any]:
    transformers, datasets = require_hf()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if config.train_parquet is not None:
        raw = datasets.load_dataset(
            "parquet",
            data_files={
                "train": config.train_parquet,
                "validation": config.validation_parquet,
            },
            cache_dir=str(cache_dir),
        )
    else:
        raw = datasets.load_dataset(
            config.dataset_name,
            config.dataset_config,
            revision=config.dataset_revision,
            cache_dir=str(cache_dir),
        )
    required_columns = {"sentence1", "sentence2", "label"}
    for split in ("train", "validation"):
        missing = required_columns - set(raw[split].column_names)
        if missing:
            raise ValueError(
                "%s data are missing MRPC columns: %s"
                % (split, ", ".join(sorted(missing)))
            )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        cache_dir=str(cache_dir),
        use_fast=True,
        local_files_only=config.local_files_only,
    )

    def tokenize(batch: Mapping[str, Sequence[Any]]) -> Dict[str, Any]:
        encoded = tokenizer(
            batch["sentence1"],
            batch["sentence2"],
            truncation=True,
            max_length=config.max_length,
        )
        encoded["labels"] = list(batch["label"])
        return encoded

    tokenized = raw.map(
        tokenize,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing GLUE/MRPC",
    )
    collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=config.train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    validation_loader = DataLoader(
        tokenized["validation"],
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        cache_dir=str(cache_dir),
        num_labels=2,
        local_files_only=config.local_files_only,
    )
    handle = attach_aslora_to_roberta_classifier(
        model,
        rank=config.rank,
        alpha=config.alpha,
        target_names=config.target_names,
        a_scope=config.a_scope,
        a_init_std=0.02,
        freeze_backbone=True,
        train_classifier=True,
        seed=config.seed,
    )
    return transformers, tokenized, tokenizer, train_loader, validation_loader, model, handle


def _train_to_first_merge(
    config: ASLoRAMRPCConfig,
    transformers: Any,
    model: torch.nn.Module,
    handle: Any,
    train_loader: DataLoader,
    device: torch.device,
) -> Tuple[FirstMergeSnapshot, torch.optim.Optimizer, Any, Dict[str, Any]]:
    # Moving a module after constructing its optimizer can replace tensor
    # storage on some backends.  Establish the final device first so optimizer
    # state and parameter references are unambiguous.
    model.to(device)
    handle.align_running_state_to_parameters()
    optimizer = make_optimizer(model, config)
    steps_per_epoch = int(
        math.ceil(len(train_loader) / float(config.gradient_accumulation_steps))
    )
    total_steps = steps_per_epoch * config.epochs
    warmup_steps = int(math.ceil(config.warmup_ratio * total_steps))
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if total_steps < config.first_merge_step:
        raise RuntimeError(
            "the configured training run has %d optimizer steps, before first merge t=%d"
            % (total_steps, config.first_merge_step)
        )
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_batches_seen = 0
    step_trace: List[Dict[str, Any]] = []
    accumulated_loss = 0.0
    accumulated_micro_batches = 0
    started = time.time()
    for epoch_index in range(config.epochs):
        model.train()
        for batch_index, raw_batch in enumerate(train_loader):
            micro_batches_seen += 1
            batch = {
                name: value.to(device) if torch.is_tensor(value) else value
                for name, value in raw_batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss
            accumulated_loss += float(loss.detach())
            accumulated_micro_batches += 1
            (loss / float(config.gradient_accumulation_steps)).backward()
            at_epoch_end = batch_index + 1 == len(train_loader)
            update_now = (
                accumulated_micro_batches == config.gradient_accumulation_steps
                or at_epoch_end
            )
            if not update_now:
                continue
            if accumulated_micro_batches < config.gradient_accumulation_steps:
                # Each micro-loss was divided by the configured accumulation
                # count.  Restore the mean-gradient scale for a final partial
                # update when an explicitly deviating configuration does not
                # divide the epoch length.  The paper default is accumulation=1.
                correction = (
                    float(config.gradient_accumulation_steps)
                    / float(accumulated_micro_batches)
                )
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.max_grad_norm,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            handle.update_running_averages()
            step_trace.append(
                {
                    "optimizer_step": optimizer_step,
                    "epoch_one_based": epoch_index + 1,
                    "micro_batch_within_epoch_one_based": batch_index + 1,
                    "micro_batches_in_update": accumulated_micro_batches,
                    "mean_micro_batch_loss": accumulated_loss
                    / float(accumulated_micro_batches),
                    "learning_rate_after_step": float(scheduler.get_last_lr()[0]),
                }
            )
            accumulated_loss = 0.0
            accumulated_micro_batches = 0
            if optimizer_step % 20 == 0 or optimizer_step == config.first_merge_step:
                print(
                    "optimizer_step=%d/%d epoch=%d first_merge=%d"
                    % (optimizer_step, total_steps, epoch_index + 1, config.first_merge_step),
                    flush=True,
                )
            if is_scheduled_merge_step(
                optimizer_step,
                config.start_merge_step,
                config.merge_interval,
                merges_completed=0,
                merge_count=config.merge_count,
            ):
                if optimizer_step != config.first_merge_step:
                    raise AssertionError("first merge schedule drifted from the declared step")
                snapshot = handle.capture_first_merge_snapshot(
                    candidate_mode="all",
                    decision_scope="per_projection",
                    metadata={
                        "kind": "roberta-base-mrpc-pre-first-merge",
                        "optimizer_step": optimizer_step,
                        "optimizer_step_indexing": "one_based",
                        "running_average_sampling": "after_optimizer_update",
                        "running_average_observations": optimizer_step,
                        "epoch_one_based": epoch_index + 1,
                        "micro_batch_within_epoch_one_based": batch_index + 1,
                        "micro_batches_seen": micro_batches_seen,
                        "candidate_modes_audited": list(config.candidate_modes),
                        "decision_scopes_audited": list(config.decision_scopes),
                    },
                )
                for state in snapshot.targets.values():
                    if state.running_count != optimizer_step:
                        raise AssertionError("running-average count must equal optimizer steps")
                execution = {
                    "optimizer_step": optimizer_step,
                    "micro_batches_seen": micro_batches_seen,
                    "epoch_one_based": epoch_index + 1,
                    "micro_batch_within_epoch_one_based": batch_index + 1,
                    "elapsed_training_seconds": time.time() - started,
                    "warmup_steps": warmup_steps,
                    "planned_total_optimizer_steps": total_steps,
                    "step_trace": step_trace,
                }
                return snapshot, optimizer, scheduler, execution
    raise RuntimeError("training ended without reaching the first merge event")


def _candidate_metric_actions(
    snapshot: FirstMergeSnapshot,
    pooled_covariances: Mapping[str, torch.Tensor],
    per_layer_covariances: Mapping[str, torch.Tensor],
    lora_scalings: Mapping[str, float],
    modes: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Tuple[int, int]]], Dict[str, Any]]:
    actions: Dict[str, Dict[str, Tuple[int, int]]] = {}
    records: Dict[str, Any] = {}
    for mode in modes:
        table = distance_table(
            snapshot,
            mode,
            projected_covariances=pooled_covariances,
            per_layer_projected_covariances=per_layer_covariances,
            lora_scalings=lora_scalings,
        )
        records[mode] = {
            "distances": distance_table_to_json(table),
            "selections": {},
        }
        for metric in (
            "running_raw_B_selection_proxy",
            "running_effective_update_selection_proxy",
            "running_average_activation_selection_proxy",
            "current_effective_update_frobenius",
            "current_lower_local_activation_rms",
        ):
            if metric not in table["joint"]:
                continue
            records[mode]["selections"][metric] = {}
            for scope in ("per_projection", "joint"):
                action = select_action_from_distance_table(table, metric, scope)
                key = action_key(action)
                actions[key] = action
                records[mode]["selections"][metric][scope] = {
                    "action_key": key,
                    "action": action_to_json(action),
                }
    return actions, records


def _gauge_model_audits(
    config: ASLoRAMRPCConfig,
    handle: Any,
    validation_loader: DataLoader,
    device: torch.device,
    baseline: Any,
    gauges: Sequence[Any],
    planned_records: Sequence[Mapping[str, Any]],
    actions: Mapping[str, Mapping[str, Tuple[int, int]]],
    action_results: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for named, planned in zip(gauges, planned_records):
        selected_keys = sorted(
            {
                decision["action_key"]
                for mode_record in planned["modes"].values()
                for decision in mode_record["decisions"].values()
            }
        )
        with temporary_live_gauges(handle, named.matrices) as restoration:
            gauged_baseline = evaluate_classifier(handle.model, validation_loader, device)
            baseline_difference = evaluation_difference(baseline, gauged_baseline)
            fixed_actions: Dict[str, Any] = {}
            for key in selected_keys:
                with temporary_target_action(handle, actions[key]):
                    gauged_result = evaluate_classifier(
                        handle.model, validation_loader, device
                    )
                difference = evaluation_difference(action_results[key], gauged_result)
                logit_scale = float(action_results[key].logits.abs().max())
                loss_scale = abs(float(action_results[key].loss))
                fixed_actions[key] = {
                    "action": action_to_json(actions[key]),
                    "gauged_metrics": gauged_result.metrics(),
                    "difference_from_same_action_in_original_gauge": difference,
                    "fixed_action_invariance_pass": (
                        difference["max_abs_logit_error"]
                        <= config.gauge_atol + config.gauge_rtol * logit_scale
                        and difference["loss_abs_error"]
                        <= config.gauge_atol + config.gauge_rtol * loss_scale
                        and difference["prediction_disagreement"] == 0.0
                    ),
                }
        product_pass = all(
            value
            <= config.gauge_atol
            + config.gauge_rtol * planned["product_max_abs_reference"][target]
            for target, value in planned["product_max_abs_error"].items()
        )
        baseline_logit_scale = float(baseline.logits.abs().max())
        baseline_loss_scale = abs(float(baseline.loss))
        baseline_pass = (
            baseline_difference["max_abs_logit_error"]
            <= config.gauge_atol + config.gauge_rtol * baseline_logit_scale
            and baseline_difference["loss_abs_error"]
            <= config.gauge_atol + config.gauge_rtol * baseline_loss_scale
            and baseline_difference["prediction_disagreement"] == 0.0
        )
        record = dict(planned)
        record.update(
            {
                "unmerged_gauged_metrics": gauged_baseline.metrics(),
                "unmerged_function_invariance": baseline_difference,
                "product_invariance_pass": product_pass,
                "unmerged_function_invariance_pass": baseline_pass,
                "selected_fixed_action_audits": fixed_actions,
                "all_selected_fixed_actions_invariant": all(
                    item["fixed_action_invariance_pass"]
                    for item in fixed_actions.values()
                ),
                "restoration_exact": restoration.restoration_exact,
                "parameter_identity_restored": restoration.identity_restored,
            }
        )
        # The causal comparison evaluates every selected action in the one
        # unchanged canonical model.  Re-executing a mathematically equivalent
        # factorization in float32 is a numerical stress test, not an input to
        # that comparison: even an identity torch.linalg.solve can perturb a
        # deep network enough to move a borderline prediction.  Keep those
        # diagnostics visible, but gate the main audit on float64 product and
        # distance algebra plus exact restoration of the canonical model.
        record["factorized_float32_reexecution_pass"] = (
            record["unmerged_function_invariance_pass"]
            and record["all_selected_fixed_actions_invariant"]
        )
        record["canonical_action_audit_pass"] = (
            record["product_invariance_pass"]
            and record["restoration_exact"]
            and record["parameter_identity_restored"]
            and record["distance_gauge_audit"]["all_expected_invariances_pass"]
        )
        # Backward-compatible name, now explicitly equal to the documented
        # canonical-action gate rather than the optional float32 stress test.
        record["all_required_invariances_pass"] = record[
            "canonical_action_audit_pass"
        ]
        output.append(record)
    return output


def _repeat_primary_canonical_actions(
    config: ASLoRAMRPCConfig,
    handle: Any,
    validation_loader: DataLoader,
    device: torch.device,
    baseline: Any,
    actions: Mapping[str, Mapping[str, Tuple[int, int]]],
    reference_results: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fail closed unless every raw-B-selected canonical action repeats exactly."""

    if not actions:
        raise ValueError("at least one primary selected action is required")
    records: List[Dict[str, Any]] = []
    all_pass = True
    for repeat_index in range(1, config.primary_action_evaluation_repeats):
        repeated_baseline = evaluate_classifier(
            handle.model, validation_loader, device
        )
        baseline_difference = evaluation_difference(baseline, repeated_baseline)
        repeated_results = evaluate_action_set(
            handle, validation_loader, device, actions
        )
        action_differences = {
            key: evaluation_difference(reference_results[key], repeated_results[key])
            for key in sorted(actions)
        }
        differences = [baseline_difference] + list(action_differences.values())
        repeat_pass = all(
            difference["max_abs_logit_error"] == 0.0
            and difference["loss_abs_error"] == 0.0
            and difference["accuracy_abs_error"] == 0.0
            and difference["f1_abs_error"] == 0.0
            and difference["prediction_disagreement"] == 0.0
            for difference in differences
        )
        all_pass = all_pass and repeat_pass
        records.append(
            {
                "repeat_index_zero_based": repeat_index,
                "baseline_difference": baseline_difference,
                "selected_action_differences": action_differences,
                "exact_repeat_pass": repeat_pass,
            }
        )
    return {
        "required_complete_passes": config.primary_action_evaluation_repeats,
        "selected_action_count": len(actions),
        "selected_action_keys": sorted(actions),
        "deterministic_algorithms_required": config.deterministic_algorithms,
        "exact_repeat_gate": True,
        "all_pass": all_pass,
        "records": records,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    config = config_from_args(args)
    config.validate(allow_paper_config_deviation=args.allow_paper_config_deviation)
    determinism = configure_deterministic_execution(
        enabled=config.deterministic_algorithms
    )
    if args.dry_run:
        return {
            "status": "dry_run_no_dependencies_imported_no_download",
            "config": config.to_dict(),
            "deterministic_execution": determinism,
            "paper_evidence": dict(PAPER_EVIDENCE),
            "versions": dependency_versions(),
        }
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if config.local_files_only:
        # Set these before importing the optional HF libraries so an offline
        # audit cannot silently reach the hub.  The model path and both parquet
        # paths must already exist on the remote host.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    set_global_seed(config.seed)
    output_dir = args.output_dir.resolve()
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else (output_dir / "hf_cache")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        transformers,
        dataset,
        tokenizer,
        train_loader,
        validation_loader,
        model,
        handle,
    ) = _prepare_data_and_model(config, cache_dir)
    device = torch.device(config.device)
    timeline = timeline_provenance(config, len(train_loader))
    provenance = runtime_provenance(config, model, tokenizer, dataset, timeline)
    provenance["source_sha256"] = _source_sha256()
    input_artifacts: Dict[str, Any] = {
        "initial_attached_model_state_sha256": _model_state_sha256(model),
        "binding_note": (
            "the local model directory and parquet inputs are byte-bound before "
            "training; the state digest additionally binds all tensors and buffers "
            "after deterministic ASLoRA attachment"
        ),
    }
    model_path = Path(config.model_name)
    if model_path.exists():
        input_artifacts["local_model"] = _artifact_inventory(model_path)
    elif config.local_files_only:
        raise FileNotFoundError(
            "local-files-only model path does not exist: %s" % model_path
        )
    if config.train_parquet is not None:
        input_artifacts["train_parquet"] = _artifact_inventory(
            Path(config.train_parquet)
        )
        input_artifacts["validation_parquet"] = _artifact_inventory(
            Path(config.validation_parquet)
        )
    provenance["input_artifacts"] = input_artifacts
    provenance["cache_dir"] = str(cache_dir)
    provenance["implementation_choices"] = {
        "shared_A_initialization": "normal_std_0.02_B_zero",
        "lora_dropout": 0.0,
        "padding": "dynamic_to_longest_in_batch_with_max_length_512_truncation",
        "evaluation": "float32_no_autocast_model_eval",
        "run_boundary": "stop_at_first_merge_without_committing_any_merge",
        "deterministic_execution": determinism,
        "primary_action_evaluation_repeats": (
            config.primary_action_evaluation_repeats
        ),
    }
    write_json(output_dir / "provenance.json", provenance)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    model.config.to_json_file(str(output_dir / "model_config.json"))

    snapshot, optimizer, scheduler, execution = _train_to_first_merge(
        config, transformers, model, handle, train_loader, device
    )
    snapshot_path = output_dir / "pre_first_merge_snapshot.pt"
    checkpoint_path = output_dir / "pre_first_merge_checkpoint.pt"
    save_first_merge_snapshot(snapshot, snapshot_path)
    save_training_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        scheduler,
        execution["optimizer_step"],
        execution["micro_batches_seen"],
        config,
        metadata=snapshot.metadata,
    )

    # Every raw-B action is frozen before the first validation forward pass.
    gauges = build_deterministic_gauge_bank(
        snapshot, config.gauge_condition_limits, config.seed
    )
    for witness_path in args.gauge_witness:
        gauges.extend(
            load_external_gauge_bank(
                witness_path,
                snapshot,
                max_condition_number=args.max_imported_gauge_condition,
            )
        )
    planned_gauge_records, planned_actions = plan_prevalidation_decisions(
        snapshot, gauges, config.candidate_modes
    )

    identity_before = structural_identity_signature(handle)
    digest_before = trainable_tensor_digest(model)
    with ProjectedActivationCollector(handle) as collector:
        baseline = evaluate_classifier(
            model, validation_loader, device, activation_collector=collector
        )
    lora_scalings = {
        target: config.alpha / float(config.rank) for target in snapshot.targets
    }
    metric_actions, metric_records = _candidate_metric_actions(
        snapshot,
        baseline.projected_covariances,
        baseline.projected_layer_covariances,
        lora_scalings,
        config.candidate_modes,
    )
    for named, planned in zip(gauges, planned_gauge_records):
        planned["distance_gauge_audit"] = gauge_distance_invariance_audit(
            snapshot,
            named,
            config.candidate_modes,
            projected_covariances=baseline.projected_covariances,
            per_layer_projected_covariances=baseline.projected_layer_covariances,
            lora_scalings=lora_scalings,
            atol=config.gauge_atol,
            rtol=config.gauge_rtol,
        )
    num_layers = next(iter(snapshot.targets.values())).layer_b.shape[0]
    actions = candidate_family_actions(
        sorted(snapshot.targets), num_layers, config.candidate_modes
    )
    actions.update(planned_actions)
    actions.update(metric_actions)
    action_results = evaluate_action_set(
        handle, validation_loader, device, actions
    )
    primary_repeatability = _repeat_primary_canonical_actions(
        config,
        handle,
        validation_loader,
        device,
        baseline,
        planned_actions,
        action_results,
    )
    oracle_records = validation_loss_oracles(
        action_results,
        actions,
        config.candidate_modes,
        num_layers,
        sorted(snapshot.targets),
    )
    gauge_audits = _gauge_model_audits(
        config,
        handle,
        validation_loader,
        device,
        baseline,
        gauges,
        planned_gauge_records,
        actions,
        action_results,
    )
    # Persist every numerical check before enforcing the fail-closed gate.  A
    # rejected run must remain diagnosable: otherwise one cannot distinguish a
    # mathematical product failure from a floating-point failure confined to
    # one high-condition witness.  This file contains no validation-selected
    # replacement gauge and is written for successful runs as well.
    invariance_diagnostics_path = output_dir / "invariance_diagnostics.json"
    write_json(
        invariance_diagnostics_path,
        {
            "format": "aslora-gauge-invariance-diagnostics-v1",
            "all_required_invariances_pass": all(
                record["canonical_action_audit_pass"] for record in gauge_audits
            ),
            "factorized_float32_reexecution_all_pass": all(
                record["factorized_float32_reexecution_pass"]
                for record in gauge_audits
            ),
            "primary_canonical_action_repeatability": primary_repeatability,
            "gate_definition": (
                "float64 frozen-snapshot product/distance algebra and exact "
                "canonical model restoration; live float32 factorized "
                "re-execution is retained as a non-gating numerical stress test"
            ),
            "gauge_audits": gauge_audits,
        },
    )
    if not all(record["canonical_action_audit_pass"] for record in gauge_audits):
        raise RuntimeError(
            "one or more required gauge invariance checks failed; see %s"
            % invariance_diagnostics_path
        )
    if not primary_repeatability["all_pass"]:
        raise RuntimeError(
            "canonical primary actions were not exactly repeatable; see %s"
            % invariance_diagnostics_path
        )
    digest_after = trainable_tensor_digest(model)
    identity_after = structural_identity_signature(handle)
    original_model_unchanged = digest_before == digest_after and identity_before == identity_after
    if not original_model_unchanged:
        raise RuntimeError("candidate or gauge audit failed to restore the pre-merge model")

    candidate_json = {
        key: action_result_json(actions[key], action_results[key], baseline)
        for key in sorted(actions)
    }
    # Attach the cached original-gauge consequence to every raw selection.
    for gauge_record in gauge_audits:
        for mode_record in gauge_record["modes"].values():
            for decision in mode_record["decisions"].values():
                key = decision["action_key"]
                decision["original_gauge_candidate_evaluation"] = candidate_json[key]

    provenance["execution"] = {
        key: value for key, value in execution.items() if key != "step_trace"
    }
    provenance["artifacts"] = {
        "checkpoint": str(checkpoint_path),
        "snapshot": str(snapshot_path),
        "results": str(output_dir / "audit.json"),
        "provenance": str(output_dir / "provenance.json"),
    }
    write_json(output_dir / "provenance.json", provenance)
    result: Dict[str, Any] = {
        "format": "aslora-roberta-mrpc-first-merge-audit-v1",
        "status": "completed_pre_first_merge_audit_no_merge_committed",
        "config": config.to_dict(),
        "timeline": timeline,
        "execution": execution,
        "deterministic_execution": determinism,
        "baseline_validation": baseline.metrics(),
        "activation_weighting": {
            "current_action_estimand": (
                "exact RMS change in the lower layer's scaled LoRA output on "
                "unmerged validation inputs: (alpha/r)*sqrt(tr((B_upper-B_lower) "
                "G_lower (B_upper-B_lower)^T)); this is a local disturbance, "
                "not network validation loss"
            ),
            "network_consequence": (
                "candidate_evaluations temporarily tie current lower B to current "
                "upper B and rerun the same complete MRPC validation split"
            ),
            "running_average_proxy_warning": (
                "pooled moments combined with running-average B are reported only "
                "as a selection proxy and are not called current merge cost"
            ),
            "lora_scalings_alpha_over_rank": lora_scalings,
            "pooled_counts_across_layers": baseline.projected_counts,
            "pooled_covariances_for_running_proxy": {
                target: covariance.tolist()
                for target, covariance in baseline.projected_covariances.items()
            },
            "per_layer_counts": baseline.projected_layer_counts,
            "per_layer_covariances_for_current_action": {
                target: covariance.tolist()
                for target, covariance in baseline.projected_layer_covariances.items()
            },
        },
        "distance_baselines": metric_records,
        "distance_baseline_data_usage": {
            "running_raw_B_selection_proxy": "prevalidation_snapshot_only_ASLoRA_rule",
            "running_effective_update_selection_proxy": "prevalidation_snapshot_only",
            "current_effective_update_frobenius": "prevalidation_snapshot_only",
            "running_average_activation_selection_proxy": (
                "complete_validation_inputs_no_labels_posthoc_diagnostic"
            ),
            "current_lower_local_activation_rms": (
                "complete_validation_inputs_no_labels_posthoc_diagnostic"
            ),
            "validation_loss_oracles": (
                "complete_validation_inputs_and_labels_posthoc_oracle_never_used_"
                "for_raw_gauge_selection"
            ),
        },
        "validation_loss_oracles": oracle_records,
        "validation_loss_oracle_coverage": {
            "enumerated_exactly": [
                "every_query_only_pair",
                "every_value_only_pair",
                "every_joint_same_pair",
            ],
            "also_evaluated": (
                "every distinct per_projection composite selected by a declared "
                "distance baseline or gauge"
            ),
            "not_claimed": (
                "a full query_pair_by_value_pair Cartesian oracle; that would be "
                "4356 actions for all 12-layer pairs and is not needed to evaluate "
                "the actions actually selected in this first-merge audit"
            ),
        },
        "candidate_evaluations": candidate_json,
        "candidate_evaluation_count": len(candidate_json),
        "primary_canonical_action_repeatability": primary_repeatability,
        "gauge_audits": gauge_audits,
        "equivalence_gate": {
            "canonical_action_audit_all_pass": all(
                record["canonical_action_audit_pass"] for record in gauge_audits
            ),
            "primary_canonical_action_repeatability_pass": primary_repeatability[
                "all_pass"
            ],
            "factorized_float32_reexecution_all_pass": all(
                record["factorized_float32_reexecution_pass"]
                for record in gauge_audits
            ),
            "main_causal_path": (
                "transform the frozen running factors in float64 to choose an "
                "action, map that layer-index action back to the unchanged "
                "canonical model, and evaluate the canonical temporary tie"
            ),
            "float32_warning": (
                "live factorized re-execution is diagnostic only because "
                "finite-precision associativity can change logits despite the "
                "same algebraic effective updates"
            ),
            "diagnostics_path": str(invariance_diagnostics_path),
        },
        "gauge_generation": {
            "performed_before_validation": True,
            "deterministic_bank_uses_only_snapshot_shapes_and_seed": True,
            "external_witness_count": len(args.gauge_witness),
            "external_witness_validation_independence": (
                "caller_declaration_recorded_by_path_and_sha256_not_machine_verifiable"
                if args.gauge_witness
                else "not_applicable"
            ),
            "validation_used_for_raw_action_selection": False,
            "families": [gauge.to_dict(include_matrices=False) for gauge in gauges],
        },
        "restoration": {
            "trainable_digest_before": digest_before,
            "trainable_digest_after": digest_after,
            "structural_parameter_identity_before": identity_before,
            "structural_parameter_identity_after": identity_after,
            "original_premerge_model_unchanged": original_model_unchanged,
            "no_merge_committed": True,
        },
        "artifacts": provenance["artifacts"],
        "source_sha256": provenance["source_sha256"],
        "paper_evidence": dict(PAPER_EVIDENCE),
    }
    write_json(output_dir / "audit.json", result)
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
        print(
            "completed first-merge audit: %s"
            % (args.output_dir.resolve() / "audit.json")
        )


if __name__ == "__main__":
    main()
