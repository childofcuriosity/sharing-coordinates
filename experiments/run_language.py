"""Run one WikiText-2 byte-LM experiment or the router-stability sweep."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from src.language import LanguageConfig, run_language


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/language.json"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--router-init", default="neutral")
    parser.add_argument("--pattern", default=None)
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--hard", action="store_true")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-path", default="data/wikitext-2/train.txt")
    parser.add_argument("--validation-path", default="data/wikitext-2/valid.txt")
    parser.add_argument("--test-path", default="data/wikitext-2/test.txt")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-bases", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--hidden-dimension", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-seeds", type=int, default=3)
    parser.add_argument("--checkpoint-load", type=Path, default=None)
    parser.add_argument("--checkpoint-save", type=Path, default=None)
    args = parser.parse_args()
    base = LanguageConfig(
        train_path=args.train_path,
        validation_path=args.validation_path,
        test_path=args.test_path,
        router_init=args.router_init,
        router_trainable=not args.fixed,
        router_pattern=args.pattern,
        hard=args.hard,
        depth=args.depth,
        num_bases=args.num_bases,
        dimension=args.dimension,
        hidden_dimension=args.hidden_dimension,
        num_heads=args.num_heads,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        eval_batches=args.eval_batches,
        log_every=args.log_every,
        amp=not args.no_amp,
        checkpoint_load=None if args.checkpoint_load is None else str(args.checkpoint_load),
        checkpoint_save=None if args.checkpoint_save is None else str(args.checkpoint_save),
    )
    configs = [base]
    if args.sweep:
        if args.checkpoint_load is not None or args.checkpoint_save is not None:
            parser.error("checkpoint load/save is only supported for a single run")
        configs = []
        settings = [
            ("neutral", True, None, False),
            ("random", True, None, False),
            ("pattern_cycle", True, None, False),
            ("pattern_contiguous", True, None, False),
        ]
        for seed in range(args.sweep_seeds):
            for init, trainable, pattern, hard in settings:
                configs.append(
                    replace(
                        base,
                        seed=seed,
                        router_init=init,
                        router_trainable=trainable,
                        router_pattern=pattern,
                        hard=hard,
                    )
                )
    results = []
    for index, config in enumerate(configs, 1):
        print("run %d/%d: init=%s trainable=%s seed=%d" % (index, len(configs), config.router_init, config.router_trainable, config.seed))
        result = run_language(config, device=args.device)
        results.append(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
