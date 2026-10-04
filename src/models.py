"""Weight-mixed residual networks with an explicit depth router.

The same layer-by-basis probabilities are used to mix every weight tensor in a
residual block.  A layer therefore executes once, unlike an output-space
mixture that evaluates every basis function.  This makes the experimental
model match the factorization studied in the theory: ``Theta = A @ B``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .patterns import make_assignment


class DepthRouter(nn.Module):
    def __init__(
        self,
        depth: int,
        num_bases: int,
        init: str = "neutral",
        init_margin: float = 4.0,
        seed: int = 0,
        trainable: bool = True,
        planted_pattern: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.num_bases = num_bases
        logits = torch.zeros(depth, num_bases)
        generator = torch.Generator().manual_seed(seed)
        if init == "neutral":
            logits.normal_(0.0, 0.01, generator=generator)
        elif init == "random":
            logits.normal_(0.0, 1.0, generator=generator)
        elif init.startswith("pattern_") or init == "planted":
            pattern = planted_pattern if init == "planted" else init[len("pattern_") :]
            if pattern is None:
                raise ValueError("planted router initialization needs a pattern")
            ids = make_assignment(pattern, depth, num_bases, seed=seed)
            logits.fill_(-init_margin)
            logits.scatter_(1, ids[:, None], init_margin)
        else:
            raise ValueError("unknown router initialization: %s" % init)
        self.logits = nn.Parameter(logits, requires_grad=trainable)

    def probabilities(self, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
        soft = F.softmax(self.logits / temperature, dim=-1)
        if not hard:
            return soft
        ids = soft.argmax(dim=-1)
        one_hot = F.one_hot(ids, num_classes=self.num_bases).to(soft.dtype)
        if self.logits.requires_grad:
            return one_hot + soft - soft.detach()
        return one_hot

    def regularizer(self, temperature: float, vertex_strength: float, usage_strength: float) -> torch.Tensor:
        probabilities = self.probabilities(temperature, hard=False).clamp_min(1e-12)
        conditional_entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean()
        marginal = probabilities.mean(dim=0)
        marginal_entropy = -(marginal * marginal.clamp_min(1e-12).log()).sum()
        return vertex_strength * conditional_entropy - usage_strength * marginal_entropy


class BasisLinear(nn.Module):
    """Linear map whose weight is a convex combination of K basis tensors."""

    def __init__(self, num_bases: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.num_bases = num_bases
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(num_bases, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_bases, out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for basis in self.weight:
            nn.init.kaiming_uniform_(basis, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def mixed_parameters(self, alpha: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = torch.einsum("k,koi->oi", alpha, self.weight)
        bias = None if self.bias is None else torch.einsum("k,ko->o", alpha, self.bias)
        return weight, bias

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        weight, bias = self.mixed_parameters(alpha)
        return F.linear(x, weight, bias)


class BasisResidualMLP(nn.Module):
    """Residual MLP stack used for controlled functional recovery."""

    def __init__(
        self,
        dimension: int,
        hidden_dimension: int,
        depth: int,
        num_bases: int,
        router_init: str = "neutral",
        router_trainable: bool = True,
        router_pattern: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.depth = depth
        self.num_bases = num_bases
        self.router = DepthRouter(
            depth,
            num_bases,
            init=router_init,
            seed=seed,
            trainable=router_trainable,
            planted_pattern=router_pattern,
        )
        self.up = BasisLinear(num_bases, dimension, hidden_dimension)
        self.down = BasisLinear(num_bases, hidden_dimension, dimension)
        self.residual_scale = 1.0 / math.sqrt(depth)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        return_hidden: bool = False,
    ):
        probabilities = self.router.probabilities(temperature, hard=hard)
        hidden: List[torch.Tensor] = []
        for layer in range(self.depth):
            update = self.residual_update(x, probabilities[layer])
            x = x + self.residual_scale * update
            if return_hidden:
                hidden.append(x)
        if return_hidden:
            return x, hidden, probabilities
        return x

    def residual_update(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Evaluate one effective residual operator on arbitrary shared probes."""
        normalized = F.layer_norm(x, (self.dimension,))
        return self.down(F.gelu(self.up(normalized, alpha)), alpha)

    def effective_parameter_vectors(self, temperature: float = 1.0) -> torch.Tensor:
        probabilities = self.router.probabilities(temperature, hard=False)
        basis_vectors = []
        for k in range(self.num_bases):
            parts = [self.up.weight[k].flatten(), self.down.weight[k].flatten()]
            if self.up.bias is not None:
                parts.extend([self.up.bias[k].flatten(), self.down.bias[k].flatten()])
            basis_vectors.append(torch.cat(parts))
        return probabilities @ torch.stack(basis_vectors)


class BasisSelfAttention(nn.Module):
    def __init__(self, num_bases: int, dimension: int, num_heads: int) -> None:
        super().__init__()
        if dimension % num_heads:
            raise ValueError("dimension must be divisible by num_heads")
        self.dimension = dimension
        self.num_heads = num_heads
        self.head_dimension = dimension // num_heads
        self.qkv = BasisLinear(num_bases, dimension, 3 * dimension, bias=False)
        self.output = BasisLinear(num_bases, dimension, dimension, bias=False)

    def forward(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = x.shape
        qkv = self.qkv(x, alpha)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dimension).transpose(1, 2)
        k = k.view(batch, length, self.num_heads, self.head_dimension).transpose(1, 2)
        v = v.view(batch, length, self.num_heads, self.head_dimension).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal[None, None], torch.finfo(scores.dtype).min)
        attention = F.softmax(scores, dim=-1)
        values = torch.matmul(attention, v)
        values = values.transpose(1, 2).contiguous().view(batch, length, dimension)
        return self.output(values, alpha)


class CausalBasisTransformer(nn.Module):
    """Small decoder-only Transformer with parameter-space layer mixing."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        dimension: int,
        hidden_dimension: int,
        num_heads: int,
        depth: int,
        num_bases: int,
        router_init: str = "neutral",
        router_trainable: bool = True,
        router_pattern: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.dimension = dimension
        self.depth = depth
        self.num_bases = num_bases
        self.router = DepthRouter(
            depth,
            num_bases,
            init=router_init,
            seed=seed,
            trainable=router_trainable,
            planted_pattern=router_pattern,
        )
        self.token_embedding = nn.Embedding(vocab_size, dimension)
        self.position_embedding = nn.Embedding(max_length, dimension)
        # Tied embeddings also serve as the LM output matrix.  PyTorch's
        # default unit-scale embedding initialization would yield logits with
        # enormous variance after the final LayerNorm (hundreds of bits per
        # byte at step zero).  Use the standard small Transformer scale.
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        self.attention = BasisSelfAttention(num_bases, dimension, num_heads)
        self.up = BasisLinear(num_bases, dimension, hidden_dimension)
        self.down = BasisLinear(num_bases, hidden_dimension, dimension)
        self.final_norm = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.residual_scale = 1.0 / math.sqrt(2.0 * depth)

    def forward(
        self,
        tokens: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        return_hidden: bool = False,
    ):
        batch, length = tokens.shape
        if length > self.max_length:
            raise ValueError("sequence longer than max_length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        probabilities = self.router.probabilities(temperature, hard=hard)
        hidden: List[torch.Tensor] = []
        for layer in range(self.depth):
            alpha = probabilities[layer]
            x = self.block_transition(x, alpha)
            if return_hidden:
                hidden.append(x)
        logits = self.output(self.final_norm(x))
        if return_hidden:
            return logits, hidden, probabilities
        return logits

    def block_transition(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Apply one effective block to arbitrary hidden-state probes."""
        x = x + self.residual_scale * self.attention(F.layer_norm(x, (self.dimension,)), alpha)
        update = self.down(F.gelu(self.up(F.layer_norm(x, (self.dimension,)), alpha)), alpha)
        return x + self.residual_scale * update

    def residual_update(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        return self.block_transition(x, alpha) - x

    def parameter_report(self) -> Dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        router = self.router.logits.numel()
        return {"total": total, "router": router, "basis": total - router}
