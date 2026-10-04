from pathlib import Path

import torch
import pytest

from src.language import (
    LanguageConfig,
    build_language_model,
    corpus_sha256,
    evaluate,
    load_byte_stream,
    load_language_checkpoint,
    sample_batch,
    save_language_checkpoint,
)


def test_byte_batches_shift_targets(tmp_path: Path):
    path = tmp_path / "text.txt"
    path.write_bytes(bytes(range(256)) * 8)
    stream = load_byte_stream(str(path), torch.device("cpu"))
    generator = torch.Generator().manual_seed(0)
    inputs, targets = sample_batch(stream, 4, 16, generator)
    assert inputs.shape == targets.shape == (4, 16)
    assert torch.equal((inputs + 1) % 256, targets)


def test_frozen_router_is_not_silently_hardened():
    class RecordingModel:
        def __init__(self):
            self.hard_values = []

        def eval(self):
            return self

        def train(self):
            return self

        def __call__(self, inputs, temperature, hard):
            self.hard_values.append(hard)
            return torch.zeros(*inputs.shape, 256)

    model = RecordingModel()
    stream = torch.arange(2048, dtype=torch.long) % 256
    config = LanguageConfig(
        router_trainable=False,
        hard=False,
        batch_size=2,
        sequence_length=8,
        eval_batches=1,
    )
    evaluate(model, stream, config, temperature=1.0, seed=0)
    assert model.hard_values == [False]


def test_language_checkpoint_round_trip_and_corpus_hash(tmp_path: Path):
    paths = {}
    for index, split in enumerate(("train", "validation", "test")):
        path = tmp_path / (split + ".txt")
        path.write_bytes(bytes(range(256)) * (5 + index))
        paths[split] = path
    config = LanguageConfig(
        train_path=str(paths["train"]),
        validation_path=str(paths["validation"]),
        test_path=str(paths["test"]),
        depth=3,
        num_bases=2,
        dimension=8,
        hidden_dimension=16,
        num_heads=2,
        sequence_length=8,
        batch_size=2,
        steps=0,
        amp=False,
    )
    model = build_language_model(config)
    hashes = corpus_sha256(config)
    checkpoint = tmp_path / "language.pt"
    save_language_checkpoint(str(checkpoint), model, config, hashes, extra={"tag": "test"})
    loaded, loaded_config, metadata = load_language_checkpoint(
        str(checkpoint),
        device="cpu",
        expected_corpus_sha256=hashes,
    )
    assert loaded_config.dimension == config.dimension
    assert metadata["corpus_sha256"] == hashes
    assert metadata["extra"] == {"tag": "test"}
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name])

    wrong = dict(hashes)
    wrong["validation"] = "0" * 64
    with pytest.raises(ValueError, match="corpus SHA256 mismatch"):
        load_language_checkpoint(
            str(checkpoint),
            device="cpu",
            expected_corpus_sha256=wrong,
        )
