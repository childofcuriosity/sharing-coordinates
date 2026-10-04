import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.run_aslora_mrpc import (
    _repeat_primary_canonical_actions,
    _source_sha256,
    _train_to_first_merge,
    build_parser,
    run,
)
from src.aslora_training import (
    ASLoRAMRPCConfig,
    ProjectedActivationCollector,
    binary_classification_metrics,
    build_deterministic_gauge_bank,
    distance_table,
    evaluate_action_set,
    evaluate_classifier,
    first_scheduled_merge_step,
    gauge_distance_invariance_audit,
    is_scheduled_merge_step,
    make_bounded_gauge,
    plan_prevalidation_decisions,
    structural_identity_signature,
    temporary_live_gauges,
    temporary_target_action,
    timeline_provenance,
    trainable_tensor_digest,
    transform_projected_covariance,
)
from src.aslora_witness import (
    apply_snapshot_gauges,
    attach_aslora_to_roberta_classifier,
    effective_updates,
    gauge_condition_number,
)


class _TinyAttentionSelf(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.value = nn.Linear(width, width)


class _TinyLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = nn.Module()
        setattr(self.attention, "self", _TinyAttentionSelf(width))


class _TinyRobertaClassifier(nn.Module):
    """Just enough RoBERTa-shaped structure to test without Transformers."""

    def __init__(self, width=2, layers=3):
        super().__init__()
        self.roberta = nn.Module()
        self.roberta.encoder = nn.Module()
        self.roberta.encoder.layer = nn.ModuleList(
            [_TinyLayer(width) for _ in range(layers)]
        )
        self.classifier = nn.Linear(width, 2)

    def forward(self, features, labels=None, attention_mask=None):
        hidden = features.float()
        for layer in self.roberta.encoder.layer:
            hidden = torch.tanh(
                layer.attention.self.query(hidden)
                + layer.attention.self.value(hidden)
            )
        logits = self.classifier(hidden)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return SimpleNamespace(logits=logits, loss=loss)


class _RepeatedLoader:
    def __init__(self, batch, length):
        self.batch = batch
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            yield {name: value.clone() for name, value in self.batch.items()}


class _SchedulerFactory:
    @staticmethod
    def get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    ):
        def multiplier(step):
            if num_warmup_steps and step < num_warmup_steps:
                return float(step) / float(max(1, num_warmup_steps))
            remaining = max(0, num_training_steps - step)
            denominator = max(1, num_training_steps - num_warmup_steps)
            return float(remaining) / float(denominator)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _make_handle(seed=3):
    model = _TinyRobertaClassifier()
    handle = attach_aslora_to_roberta_classifier(
        model,
        rank=2,
        alpha=2.0,
        a_scope="per_projection",
        seed=seed,
    )
    with torch.no_grad():
        for target_index, target in enumerate(("query", "value")):
            handle.modules[target][0].shared_a.copy_(
                torch.tensor([[1.0, 0.2], [-0.1, 0.8]])
                + 0.1 * target_index
            )
            for layer_index, module in enumerate(handle.modules[target]):
                module.b.copy_(
                    torch.tensor(
                        [
                            [0.2 + 0.1 * layer_index, -0.3 + 0.05 * target_index],
                            [0.1 * target_index, 0.4 - 0.08 * layer_index],
                        ]
                    )
                )
    handle.update_running_averages()
    return model, handle


def test_paper_schedule_is_one_indexed_and_first_event_is_560():
    assert first_scheduled_merge_step(320, 240) == 560
    assert not is_scheduled_merge_step(320, 320, 240, merge_count=7)
    assert not is_scheduled_merge_step(559, 320, 240, merge_count=7)
    assert is_scheduled_merge_step(560, 320, 240, merge_count=7)
    assert is_scheduled_merge_step(800, 320, 240, merges_completed=1, merge_count=7)
    assert not is_scheduled_merge_step(800, 320, 240, merges_completed=7, merge_count=7)

    config = ASLoRAMRPCConfig(device="cpu")
    config.validate()
    timeline = timeline_provenance(config, train_micro_batches=230)
    assert timeline["optimizer_steps_per_epoch"] == 230
    assert timeline["first_merge_optimizer_step"] == 560
    assert timeline["first_merge_epoch_one_based"] == 3


def test_config_deviations_require_explicit_opt_in():
    config = ASLoRAMRPCConfig(device="cpu", rank=4)
    with pytest.raises(ValueError, match="explicit opt-in"):
        config.validate()
    config.validate(allow_paper_config_deviation=True)
    assert config.paper_deviations()["rank"] == {"paper": 8, "actual": 4}


def test_repeat_count_below_two_is_rejected():
    config = ASLoRAMRPCConfig(
        device="cpu", primary_action_evaluation_repeats=1
    )
    with pytest.raises(ValueError, match="at least two"):
        config.validate()


def test_bounded_gauges_preserve_products_and_condition_one_is_sanity():
    model, handle = _make_handle()
    snapshot = handle.capture_first_merge_snapshot(candidate_mode="all")
    gauges = build_deterministic_gauge_bank(
        snapshot, condition_limits=(2.0, 4.0), seed=19
    )
    assert [gauge.name for gauge in gauges[:2]] == ["identity", "orthogonal_k1"]
    assert math.isclose(
        max(gauges[1].condition_numbers().values()), 1.0, abs_tol=1e-10
    )
    assert all(
        condition <= gauge.requested_condition_limit + 1e-8
        for gauge in gauges
        for condition in gauge.condition_numbers().values()
    )

    records, actions = plan_prevalidation_decisions(
        snapshot, gauges, modes=("adjacent", "all")
    )
    assert actions
    assert all(record["validation_used_for_selection"] is False for record in records)
    assert all(record["selection_inputs"] == "raw_running_average_B_only" for record in records)
    assert all(
        "running_raw_B_selection_proxy" in mode["distance_table"]["joint"]
        and "running_effective_update_selection_proxy"
        in mode["distance_table"]["joint"]
        and "current_effective_update_frobenius"
        in mode["distance_table"]["joint"]
        for record in records
        for mode in record["modes"].values()
    )
    assert max(
        error
        for record in records
        for error in record["product_max_abs_error"].values()
    ) < 1e-6

    orthogonal_audit = gauge_distance_invariance_audit(
        snapshot, gauges[1], modes=("adjacent", "all")
    )
    assert orthogonal_audit["gauge_is_orthogonal_condition_one_sanity"]
    assert orthogonal_audit["all_expected_invariances_pass"]
    assert all(
        mode["running_raw_B_selection_proxy"]["invariance_pass"]
        for mode in orthogonal_audit["modes"].values()
    )
    nonorthogonal_audit = gauge_distance_invariance_audit(
        snapshot, gauges[2], modes=("all",)
    )
    assert not nonorthogonal_audit["gauge_is_orthogonal_condition_one_sanity"]
    assert (
        nonorthogonal_audit["modes"]["all"]["running_raw_B_selection_proxy"][
            "invariance_pass"
        ]
        is None
    )
    assert nonorthogonal_audit["all_expected_invariances_pass"]

    identity_table = records[0]["modes"]["all"]["distance_table"]
    orthogonal_table = records[1]["modes"]["all"]["distance_table"]
    for target in ("query", "value"):
        assert identity_table["per_projection"][target][
            "running_raw_B_selection_proxy"
        ] == pytest.approx(
            orthogonal_table["per_projection"][target][
                "running_raw_B_selection_proxy"
            ],
            abs=1e-6,
        )

    # The activation-weighted distance is invariant only if its low-rank
    # covariance is transformed along with the factors.
    covariance = torch.tensor([[1.2, 0.1], [0.1, 0.7]])
    q = make_bounded_gauge(2, "dense", 4.0, seed=8)
    transformed = apply_snapshot_gauges(
        snapshot, {target: q for target in snapshot.targets}
    )
    layer_covariances = {
        target: torch.stack(
            [covariance * float(layer + 1) for layer in range(state.layer_b.shape[0])]
        )
        for target, state in snapshot.targets.items()
    }
    scalings = {target: 2.0 for target in snapshot.targets}
    before = distance_table(
        snapshot,
        "all",
        projected_covariances={target: covariance for target in snapshot.targets},
        per_layer_projected_covariances=layer_covariances,
        lora_scalings=scalings,
    )
    transformed_covariance = transform_projected_covariance(covariance, q)
    transformed_layer_covariances = {
        target: torch.stack(
            [transform_projected_covariance(matrix, q) for matrix in matrices]
        )
        for target, matrices in layer_covariances.items()
    }
    after = distance_table(
        transformed,
        "all",
        projected_covariances={
            target: transformed_covariance for target in transformed.targets
        },
        per_layer_projected_covariances=transformed_layer_covariances,
        lora_scalings=scalings,
    )
    for target in snapshot.targets:
        for metric in (
            "running_average_activation_selection_proxy",
            "current_lower_local_activation_rms",
            "current_effective_update_frobenius",
        ):
            for pair, value in before["per_projection"][target][metric].items():
                assert torch.allclose(
                    value,
                    after["per_projection"][target][metric][pair],
                    atol=1e-5,
                    rtol=1e-5,
                )

    state = snapshot.targets["query"]
    pair = (0, 2)
    difference = state.layer_b[pair[1]] - state.layer_b[pair[0]]
    expected_local = 2.0 * torch.trace(
        difference @ layer_covariances["query"][pair[0]] @ difference.T
    ).sqrt()
    assert torch.allclose(
        before["per_projection"]["query"]["current_lower_local_activation_rms"][pair],
        expected_local,
    )

    del model


def test_live_gauge_and_candidate_ties_are_exactly_restored():
    model, handle = _make_handle(seed=7)
    features = torch.tensor([[0.2, -0.4], [1.0, 0.3], [-0.2, 0.8]])
    labels = torch.tensor([0, 1, 0])
    loader = [{"features": features, "labels": labels}]
    baseline = evaluate_classifier(model, loader, torch.device("cpu"))
    digest_before = trainable_tensor_digest(model)
    identities_before = structural_identity_signature(handle)
    products_before = {
        target: effective_updates(
            modules[0].shared_a,
            torch.stack([module.b for module in modules]),
        )
        for target, modules in handle.modules.items()
    }

    matrices = {
        "query": make_bounded_gauge(2, "diagonal", 2.0, seed=1),
        "value": make_bounded_gauge(2, "dense", 4.0, seed=2),
    }
    assert gauge_condition_number(matrices["query"]) <= 2.0 + 1e-8
    with temporary_live_gauges(handle, matrices) as restoration:
        gauged = evaluate_classifier(model, loader, torch.device("cpu"))
        assert torch.allclose(baseline.logits, gauged.logits, atol=1e-5, rtol=1e-5)
        for target, modules in handle.modules.items():
            products = effective_updates(
                modules[0].shared_a,
                torch.stack([module.b for module in modules]),
            )
            assert torch.allclose(products_before[target], products, atol=1e-5, rtol=1e-5)
        original_lower = handle.modules["query"][0].b
        upper = handle.modules["query"][2].b
        with temporary_target_action(handle, {"query": (0, 2)}):
            assert handle.modules["query"][0].b is upper
        assert handle.modules["query"][0].b is original_lower

    assert restoration.restoration_exact
    assert restoration.identity_restored
    assert structural_identity_signature(handle) == identities_before
    assert trainable_tensor_digest(model) == digest_before


def test_complete_evaluation_metrics_and_activation_collection():
    model, handle = _make_handle(seed=11)
    batches = [
        {
            "features": torch.tensor([[0.2, 0.5], [-0.4, 0.3]]),
            "labels": torch.tensor([1, 0]),
        },
        {
            "features": torch.tensor([[0.8, -0.1]]),
            "labels": torch.tensor([1]),
        },
    ]
    with ProjectedActivationCollector(handle) as collector:
        result = evaluate_classifier(
            model, batches, torch.device("cpu"), activation_collector=collector
        )
    assert result.examples == 3
    assert result.batches == 2
    assert set(result.projected_covariances) == {"query", "value"}
    assert result.projected_counts == {"query": 9, "value": 9}
    assert all(matrix.shape == (2, 2) for matrix in result.projected_covariances.values())
    assert result.projected_layer_counts == {
        "query": [3, 3, 3],
        "value": [3, 3, 3],
    }
    assert all(
        matrix.shape == (3, 2, 2)
        for matrix in result.projected_layer_covariances.values()
    )
    direct = binary_classification_metrics(result.logits.argmax(-1), result.labels)
    assert result.accuracy == direct["accuracy"]
    assert result.f1 == direct["f1"]


def test_primary_canonical_actions_repeat_exactly_on_cpu():
    model, handle = _make_handle(seed=17)
    batches = [
        {
            "features": torch.tensor([[0.2, 0.5], [-0.4, 0.3]]),
            "labels": torch.tensor([1, 0]),
        },
        {
            "features": torch.tensor([[0.8, -0.1]]),
            "labels": torch.tensor([1]),
        },
    ]
    actions = {
        "query:0-1": {"query": (0, 1)},
        "value:1-2": {"value": (1, 2)},
    }
    device = torch.device("cpu")
    baseline = evaluate_classifier(model, batches, device)
    references = evaluate_action_set(handle, batches, device, actions)
    audit = _repeat_primary_canonical_actions(
        ASLoRAMRPCConfig(device="cpu"),
        handle,
        batches,
        device,
        baseline,
        actions,
        references,
    )
    assert audit["all_pass"] is True
    assert audit["required_complete_passes"] == 2
    assert audit["selected_action_count"] == 2
    assert all(record["exact_repeat_pass"] for record in audit["records"])


def test_mock_training_updates_average_after_optimizer_update_and_stops_premerge():
    model = _TinyRobertaClassifier(layers=2)
    handle = attach_aslora_to_roberta_classifier(
        model, rank=2, alpha=2.0, a_scope="per_projection", seed=13
    )
    config = ASLoRAMRPCConfig(
        device="cpu",
        rank=2,
        alpha=2.0,
        epochs=1,
        train_batch_size=1,
        start_merge_step=1,
        merge_interval=2,
        gradient_accumulation_steps=2,
    )
    loader = _RepeatedLoader(
        {
            "features": torch.tensor([[0.3, -0.2]]),
            "labels": torch.tensor([1]),
        },
        length=6,
    )
    snapshot, optimizer, scheduler, execution = _train_to_first_merge(
        config,
        _SchedulerFactory,
        model,
        handle,
        loader,
        torch.device("cpu"),
    )
    assert execution["optimizer_step"] == 3
    assert execution["micro_batches_seen"] == 6
    assert execution["step_trace"][-1]["micro_batches_in_update"] == 2
    assert all(state.running_count == 3 for state in snapshot.targets.values())
    assert snapshot.metadata["running_average_sampling"] == "after_optimizer_update"
    assert snapshot.metadata["optimizer_step"] == 3
    assert len(optimizer.param_groups) == 2
    assert scheduler.last_epoch == 3
    assert all(
        len({id(module.b) for module in modules}) == 2
        for modules in handle.modules.values()
    )


def test_running_average_state_tracks_model_dtype_after_model_to():
    model = _TinyRobertaClassifier().double()
    handle = attach_aslora_to_roberta_classifier(
        model, rank=2, alpha=2.0, a_scope="per_projection", seed=5
    )
    assert all(value.dtype == torch.float64 for value in handle.running_b.values())
    model.float()
    assert all(value.dtype == torch.float64 for value in handle.running_b.values())
    handle.align_running_state_to_parameters()
    assert all(value.dtype == torch.float32 for value in handle.running_b.values())
    handle.update_running_averages()
    assert handle.running_count == {"query": 1, "value": 1}


def test_cli_dry_run_needs_no_hugging_face_import_or_download(tmp_path):
    args = build_parser().parse_args(
        ["--dry-run", "--device", "cpu", "--output-dir", str(tmp_path / "unused")]
    )
    result = run(args)
    assert result["status"] == "dry_run_no_dependencies_imported_no_download"
    assert result["config"]["first_merge_step"] == 560
    assert result["config"]["deterministic_algorithms"] is True
    assert result["config"]["primary_action_evaluation_repeats"] == 2
    assert result["deterministic_execution"][
        "torch_deterministic_algorithms_enabled"
    ] is True
    assert not (tmp_path / "unused").exists()


def test_cli_dry_run_records_offline_local_model_and_parquets(tmp_path):
    model_dir = tmp_path / "roberta-base"
    train_parquet = tmp_path / "mrpc-train.parquet"
    validation_parquet = tmp_path / "mrpc-validation.parquet"
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--device",
            "cpu",
            "--model-name",
            str(model_dir),
            "--local-files-only",
            "--train-parquet",
            str(train_parquet),
            "--validation-parquet",
            str(validation_parquet),
        ]
    )
    result = run(args)
    assert result["config"]["model_name"] == str(model_dir)
    assert result["config"]["local_files_only"] is True
    assert result["config"]["train_parquet"] == str(train_parquet)
    assert result["config"]["validation_parquet"] == str(validation_parquet)


def test_aslora_source_attestation_hashes_all_audit_implementation_files():
    hashes = _source_sha256()
    assert set(hashes) == {
        "experiments/run_aslora_mrpc.py",
        "src/aslora_training.py",
        "src/aslora_witness.py",
    }
    assert all(len(digest) == 64 for digest in hashes.values())
