"""Audit gauge-equivalent router decisions on a byte-LM checkpoint."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from src.language import load_byte_stream, load_language_checkpoint
from src.router_decisions import (
    GaugeSearchFailure,
    RouterDecisionConfig,
    run_router_decision_audit,
)


def _byte_unigram_bpb(train_path: Path, evaluation_path: Path) -> float:
    """Add-one-smoothed byte unigram baseline fitted on the training split."""

    train = torch.tensor(list(train_path.read_bytes()), dtype=torch.long)
    evaluation = torch.tensor(list(evaluation_path.read_bytes()), dtype=torch.long)
    counts = torch.bincount(train, minlength=256).double() + 1.0
    log_probabilities = (counts / counts.sum()).log()
    return float(-log_probabilities[evaluation].mean() / torch.log(torch.tensor(2.0)))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/router_decisions.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument(
        "--train-corpus",
        type=Path,
        default=None,
        help="override the checkpoint's recorded train path while retaining its SHA256 gate",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--batch-seed", type=int, default=610_000)
    parser.add_argument("--search-seed", type=int, default=620_000)
    parser.add_argument("--search-trials", type=int, default=5000)
    parser.add_argument("--max-condition", type=float, default=30.0)
    parser.add_argument(
        "--search-family",
        choices=("positive_stochastic", "signed_local", "mixed"),
        default="positive_stochastic",
    )
    parser.add_argument(
        "--gauge-selection",
        choices=("first_valid", "max_disagreement"),
        default="first_valid",
    )
    parser.add_argument(
        "--batch-sampling",
        choices=("nonoverlap", "with_replacement"),
        default="nonoverlap",
    )
    parser.add_argument("--max-effective-relative-l2", type=float, default=1e-5)
    parser.add_argument("--max-logit-abs-error", type=float, default=1e-4)
    parser.add_argument("--max-batch-nll-error", type=float, default=1e-5)
    parser.add_argument("--max-bpb-equivalence-error", type=float, default=1e-5)
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="allow backend TF32 during the exact-equivalence audit (not preregistered)",
    )
    parser.add_argument("--kmeans-seed", type=int, default=630_000)
    parser.add_argument("--kmeans-restarts", type=int, default=8)
    parser.add_argument("--kmeans-iterations", type=int, default=50)
    return parser.parse_args()


def main():
    args = _parse_args()
    resolved = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, language_config, checkpoint_metadata = load_language_checkpoint(
        str(args.checkpoint),
        device=str(resolved),
    )
    path_field = {
        "train": "train_path",
        "validation": "validation_path",
        "test": "test_path",
    }[args.split]
    corpus_path = args.corpus or Path(getattr(language_config, path_field))
    corpus_digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    recorded_digest = checkpoint_metadata["corpus_sha256"].get(args.split)
    if corpus_digest != recorded_digest:
        raise ValueError(
            "corpus SHA256 mismatch for %s: checkpoint=%s current=%s"
            % (args.split, recorded_digest, corpus_digest)
        )
    train_path = args.train_corpus or Path(language_config.train_path)
    train_digest = hashlib.sha256(train_path.read_bytes()).hexdigest()
    if train_digest != checkpoint_metadata["corpus_sha256"].get("train"):
        raise ValueError(
            "training corpus SHA256 mismatch: checkpoint=%s current=%s"
            % (checkpoint_metadata["corpus_sha256"].get("train"), train_digest)
        )
    stream = load_byte_stream(str(corpus_path), resolved)
    config = RouterDecisionConfig(
        temperature=args.temperature,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        eval_batches=args.eval_batches,
        batch_seed=args.batch_seed,
        search_seed=args.search_seed,
        search_trials=args.search_trials,
        max_condition=args.max_condition,
        search_family=args.search_family,
        gauge_selection=args.gauge_selection,
        batch_sampling=args.batch_sampling,
        max_effective_relative_l2=args.max_effective_relative_l2,
        max_logit_abs_error=args.max_logit_abs_error,
        max_batch_nll_error=args.max_batch_nll_error,
        max_bpb_equivalence_error=args.max_bpb_equivalence_error,
        disable_tf32=not args.allow_tf32,
        kmeans_seed=args.kmeans_seed,
        kmeans_restarts=args.kmeans_restarts,
        kmeans_iterations=args.kmeans_iterations,
    )
    try:
        result = run_router_decision_audit(model, stream, config)
        result["status"] = "completed"
    except GaugeSearchFailure as error:
        # Exhausting the preregistered search is a scientific negative result,
        # not a reason to silently enlarge the budget or change gauge families.
        result = {
            "status": "no_partition_changing_gauge_found",
            "config": asdict(config),
            "gauge_search": error.metadata,
            "numeric_controls": {
                "tf32_disabled_during_audit": config.disable_tf32,
                "search_failed_before_model_equivalence_or_decision_evaluation": True,
            },
        }
    source_root = Path(__file__).resolve().parents[1]
    source_files = {
        "experiments/run_router_decisions.py": Path(__file__).resolve(),
        "src/router_decisions.py": source_root / "src" / "router_decisions.py",
        "src/language.py": source_root / "src" / "language.py",
        "src/models.py": source_root / "src" / "models.py",
    }
    result["checkpoint"] = {
        "path": str(args.checkpoint.resolve()),
        "file_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "format_version": checkpoint_metadata["format_version"],
        "language_config": asdict(language_config),
        "training_metadata": checkpoint_metadata.get("extra", {}),
        "corpus_split": args.split,
        "corpus_path": str(corpus_path.resolve()),
        "corpus_sha256": corpus_digest,
        "train_corpus_path": str(train_path.resolve()),
        "train_corpus_sha256": train_digest,
        "add_one_byte_unigram_bpb": _byte_unigram_bpb(train_path, corpus_path),
        "device": str(resolved),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "source_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
