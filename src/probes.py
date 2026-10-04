"""Paired functional probes and an effective-parameter baseline.

The functional comparison deliberately uses common random numbers: the
counterfactual and native signatures receive the same initial probes, the
same optional random projection, and the same realization of additive
observation noise. Native hidden states are advanced with the *clean*
residual update, so measurement noise cannot perturb later inputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .metrics import recovery_metrics
from .models import BasisLinear, BasisResidualMLP, CausalBasisTransformer
from .patterns import make_assignment


PROBE_PROTOCOL_VERSION = "paired-crn-fixed-gaussian-v1"
_INPUT_SEED_OFFSET = 40_000
_PROJECTION_SEED_OFFSET = 50_000
_NOISE_SEED_OFFSET = 60_000

PROBE_PROTOCOL_SPEC = {
    "version": PROBE_PROTOCOL_VERSION,
    "comparison_design": "paired_common_random_numbers",
    "initial_probe_distribution": "iid_standard_normal",
    "random_seed_scheme": {
        "initial_probe": "40000 + config.seed",
        "projection": "50000 + config.seed",
        "observation_noise": "60000 + config.seed",
    },
    "observation_model": "Y_i(H_s)=r_i(H_s)+sigma*epsilon_i_s",
    "observation_noise_distribution": (
        "iid_standard_normal_across_layer_probe_output_coordinates"
    ),
    "observation_noise_scale_is_absolute": True,
    "same_initial_probes_across_functional_modes": True,
    "same_observation_noise_across_functional_modes": True,
    "same_projection_across_functional_modes": True,
    "native_state_evolution": "clean_update_only",
    "clusterer": "average_linkage_with_supplied_num_bases",
    "empirical_gap_partition": "planted_teacher_assignment",
    "empirical_gap_is_population_certificate": False,
    "effective_parameter_baseline": (
        "euclidean_distance_between_concatenated_effective_"
        "BasisLinear_weights_and_biases"
    ),
    "effective_parameter_baseline_uses_noise_or_projection": False,
    "effective_parameter_baseline_is_functional_equivalence_test": False,
}


def probe_protocol_fingerprint_sha256() -> str:
    """Hash the invariant protocol semantics, excluding run-specific values."""

    encoded = json.dumps(
        PROBE_PROTOCOL_SPEC, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class ProbeConfig:
    architecture: str = "mlp"
    pattern: str = "random_balanced"
    depth: int = 12
    num_bases: int = 4
    dimension: int = 64
    hidden_dimension: int = 128
    num_heads: int = 4
    probe_sequence_length: int = 8
    num_probes: int = 16
    projection_dimension: int = 32
    # Absolute standard deviation in Y = r(H) + sigma * epsilon. It is not
    # rescaled by a batch statistic or by the magnitude of r(H).
    observation_noise: float = 0.05
    seed: int = 0


@dataclass
class PairedProbeRandomness:
    """Random objects shared by the two functional-signature estimators."""

    initial_probes: torch.Tensor
    standard_observation_noise: torch.Tensor
    projection: Optional[torch.Tensor]


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _teacher(config: ProbeConfig, device: torch.device):
    common = dict(
        dimension=config.dimension,
        hidden_dimension=config.hidden_dimension,
        depth=config.depth,
        num_bases=config.num_bases,
        router_init="planted",
        router_trainable=False,
        router_pattern=config.pattern,
        seed=30_000 + config.seed,
    )
    if config.architecture == "mlp":
        model = BasisResidualMLP(**common)
    elif config.architecture == "transformer":
        model = CausalBasisTransformer(
            vocab_size=32,
            max_length=config.probe_sequence_length,
            num_heads=config.num_heads,
            **common,
        )
    else:
        raise ValueError(config.architecture)
    return model.to(device).eval()


def _random_projection(
    input_dimension: int,
    output_dimension: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(input_dimension, output_dimension, generator=generator)
    matrix = matrix / math.sqrt(output_dimension)
    return matrix.to(device)


def _probe_shape(config: ProbeConfig) -> Tuple[int, ...]:
    if config.architecture == "mlp":
        return (config.num_probes, config.dimension)
    if config.architecture == "transformer":
        return (
            config.num_probes,
            config.probe_sequence_length,
            config.dimension,
        )
    raise ValueError(config.architecture)


def paired_probe_randomness(
    config: ProbeConfig, device: torch.device
) -> PairedProbeRandomness:
    """Create the common-random-number coupling used by both estimators."""

    shape = _probe_shape(config)
    input_generator = torch.Generator(device="cpu").manual_seed(
        _INPUT_SEED_OFFSET + config.seed
    )
    initial = torch.randn(shape, generator=input_generator).to(device)

    noise_generator = torch.Generator(device="cpu").manual_seed(
        _NOISE_SEED_OFFSET + config.seed
    )
    standard_noise = torch.randn(
        (config.depth,) + shape, generator=noise_generator
    ).to(device)

    flattened_dimension = math.prod(shape)
    projection = None
    if 0 < config.projection_dimension < flattened_dimension:
        projection = _random_projection(
            flattened_dimension,
            config.projection_dimension,
            _PROJECTION_SEED_OFFSET + config.seed,
            device,
        )
    return PairedProbeRandomness(initial, standard_noise, projection)


def _observe(
    clean_update: torch.Tensor,
    standard_noise: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    if sigma < 0:
        raise ValueError("observation_noise must be nonnegative")
    if sigma == 0:
        return clean_update
    # Fixed-scale iid Gaussian observation model. In particular, sigma does
    # not depend on clean_update.std() or any other realized batch statistic.
    return clean_update + sigma * standard_noise.to(
        device=clean_update.device, dtype=clean_update.dtype
    )


def _finish_signatures(
    outputs: List[torch.Tensor],
    config: ProbeConfig,
    randomness: PairedProbeRandomness,
) -> torch.Tensor:
    # This makes squared Euclidean distance the empirical mean, over the n
    # independently drawn probes, of the squared vector-output discrepancy.
    signatures = torch.stack([output.flatten() for output in outputs])
    signatures = signatures / math.sqrt(config.num_probes)
    if randomness.projection is not None:
        signatures = signatures @ randomness.projection.to(
            device=signatures.device, dtype=signatures.dtype
        )
    return signatures


def counterfactual_signatures(
    model,
    config: ProbeConfig,
    device: torch.device,
    randomness: Optional[PairedProbeRandomness] = None,
) -> torch.Tensor:
    """Evaluate every effective layer on the same hidden-state probes."""

    shared = randomness or paired_probe_randomness(config, device)
    probabilities = model.router.probabilities(temperature=1.0, hard=True)
    outputs = []
    with torch.no_grad():
        for layer in range(config.depth):
            clean_update = model.residual_update(
                shared.initial_probes, probabilities[layer]
            )
            outputs.append(
                _observe(
                    clean_update,
                    shared.standard_observation_noise[layer],
                    config.observation_noise,
                )
            )
    return _finish_signatures(outputs, config, shared)


def native_trajectory_signatures(
    model,
    config: ProbeConfig,
    device: torch.device,
    randomness: Optional[PairedProbeRandomness] = None,
) -> torch.Tensor:
    """Evaluate layers at native inputs under the paired observation model."""

    shared = randomness or paired_probe_randomness(config, device)
    probabilities = model.router.probabilities(temperature=1.0, hard=True)
    x = shared.initial_probes.clone()
    outputs = []
    with torch.no_grad():
        for layer in range(config.depth):
            clean_update = model.residual_update(x, probabilities[layer])
            outputs.append(
                _observe(
                    clean_update,
                    shared.standard_observation_noise[layer],
                    config.observation_noise,
                )
            )
            if config.architecture == "mlp":
                # MLP residual_update returns the unscaled branch.
                x = x + model.residual_scale * clean_update
            else:
                # Transformer residual_update is block_transition(x) - x and
                # already contains its attention/MLP residual scales.
                x = x + clean_update
    return _finish_signatures(outputs, config, shared)


def effective_parameter_signatures(model) -> torch.Tensor:
    """Concatenate each layer's visible effective ``BasisLinear`` tensors.

    This is a simple coordinate-invariant baseline for the present linear
    weight-mixing implementation. Equality of these vectors is sufficient
    for equality of the instantiated block parameters, but parameter distance
    is not a general test of functional equivalence: neural parameter
    symmetries and behavior restricted to an input distribution remain.
    """

    basis_linears = [
        module for module in model.modules() if isinstance(module, BasisLinear)
    ]
    if not basis_linears:
        raise ValueError("model exposes no BasisLinear effective parameters")
    probabilities = model.router.probabilities(temperature=1.0, hard=True)
    layer_vectors = []
    with torch.no_grad():
        for layer in range(model.depth):
            parts = []
            for module in basis_linears:
                weight, bias = module.mixed_parameters(probabilities[layer])
                parts.append(weight.flatten())
                if bias is not None:
                    parts.append(bias.flatten())
            vector = torch.cat(parts)
            layer_vectors.append(vector / math.sqrt(vector.numel()))
    return torch.stack(layer_vectors)


def agglomerative_partition(
    signatures: torch.Tensor, num_clusters: int
) -> torch.Tensor:
    """Average-linkage clustering for the small layer counts used here."""

    distances = torch.cdist(signatures, signatures).detach().cpu()
    clusters: List[List[int]] = [[i] for i in range(len(signatures))]
    while len(clusters) > num_clusters:
        best = None
        best_distance = float("inf")
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                block = distances[clusters[i]][:, clusters[j]]
                value = float(block.mean())
                if value < best_distance:
                    best_distance = value
                    best = (i, j)
        assert best is not None
        i, j = best
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]
    assignment = torch.empty(len(signatures), dtype=torch.long)
    for index, cluster in enumerate(clusters):
        assignment[cluster] = index
    return assignment.to(signatures.device)


def separation_certificate(
    signatures: torch.Tensor, target_assignment: torch.Tensor
) -> Dict[str, float]:
    """Finite-sample gap relative to a supplied target partition.

    Despite the historical function name, this descriptive empirical gap is
    not by itself a population certificate. It instantiates the within- and
    between-class quantities for the planted partition in the saved sweep.
    """

    distances = torch.cdist(signatures, signatures).square()
    same = target_assignment[:, None] == target_assignment[None, :]
    diagonal = torch.eye(
        len(target_assignment), dtype=torch.bool, device=target_assignment.device
    )
    within_values = distances[same & ~diagonal]
    between_values = distances[~same]
    within = float(within_values.max()) if within_values.numel() else 0.0
    between = (
        float(between_values.min()) if between_values.numel() else float("inf")
    )
    return {
        "max_within_distance": within,
        "min_between_distance": between,
        "empirical_gap": between - within,
    }


def _mode_result(
    truth: torch.Tensor, signatures: torch.Tensor, num_clusters: int
) -> Tuple[Dict[str, float], torch.Tensor]:
    assignment = agglomerative_partition(signatures, num_clusters)
    probabilities = torch.nn.functional.one_hot(
        assignment, num_clusters
    ).float()
    metrics = {
        **recovery_metrics(truth, probabilities),
        **separation_certificate(signatures, truth),
    }
    return metrics, assignment


def probe_protocol(config: ProbeConfig) -> Dict[str, object]:
    protocol = dict(PROBE_PROTOCOL_SPEC)
    protocol.update(
        {
        "initial_probe_seed": _INPUT_SEED_OFFSET + config.seed,
        "observation_noise_seed": _NOISE_SEED_OFFSET + config.seed,
        "observation_noise_scale": config.observation_noise,
        "projection_seed": _PROJECTION_SEED_OFFSET + config.seed,
        "projection_dimension": config.projection_dimension,
        }
    )
    return protocol


def run_probe_recovery(
    config: ProbeConfig, device: Optional[str] = None
) -> Dict[str, object]:
    _seed(config.seed)
    resolved = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _teacher(config, resolved)
    truth = make_assignment(
        config.pattern, config.depth, config.num_bases, 30_000 + config.seed
    ).to(resolved)

    # The same object is intentionally passed to both estimators. Thus each
    # estimator has the correct marginal Gaussian experiment, while their
    # difference has lower Monte Carlo variance.
    randomness = paired_probe_randomness(config, resolved)
    counterfactual = counterfactual_signatures(
        model, config, resolved, randomness
    )
    native = native_trajectory_signatures(model, config, resolved, randomness)
    effective_parameter = effective_parameter_signatures(model)

    counterfactual_metrics, counterfactual_ids = _mode_result(
        truth, counterfactual, config.num_bases
    )
    native_metrics, native_ids = _mode_result(
        truth, native, config.num_bases
    )
    parameter_metrics, parameter_ids = _mode_result(
        truth, effective_parameter, config.num_bases
    )
    return {
        "config": asdict(config),
        "protocol": probe_protocol(config),
        "device": str(resolved),
        "counterfactual": counterfactual_metrics,
        "native": native_metrics,
        "effective_parameter": parameter_metrics,
        "true_assignment": truth.detach().cpu().tolist(),
        "counterfactual_assignment": counterfactual_ids.detach().cpu().tolist(),
        "native_assignment": native_ids.detach().cpu().tolist(),
        "effective_parameter_assignment": parameter_ids.detach().cpu().tolist(),
    }
