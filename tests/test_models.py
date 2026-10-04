import torch

from src.models import BasisLinear, CausalBasisTransformer


def test_basis_linear_one_hot_selects_exact_basis():
    layer = BasisLinear(3, 5, 7)
    x = torch.randn(2, 5)
    for k in range(3):
        alpha = torch.nn.functional.one_hot(torch.tensor(k), num_classes=3).float()
        expected = torch.nn.functional.linear(x, layer.weight[k], layer.bias[k])
        assert torch.allclose(layer(x, alpha), expected, atol=1e-6)


def test_transformer_is_causal():
    torch.manual_seed(0)
    model = CausalBasisTransformer(
        vocab_size=31,
        max_length=12,
        dimension=16,
        hidden_dimension=32,
        num_heads=4,
        depth=4,
        num_bases=2,
        router_init="pattern_cycle",
        router_trainable=False,
    ).eval()
    first = torch.randint(0, 31, (2, 12))
    second = first.clone()
    second[:, 7:] = torch.randint(0, 31, second[:, 7:].shape)
    with torch.no_grad():
        logits_first = model(first, hard=True)
        logits_second = model(second, hard=True)
    assert torch.allclose(logits_first[:, :7], logits_second[:, :7], atol=1e-5)


def test_transformer_initial_logits_have_language_model_scale():
    torch.manual_seed(1)
    model = CausalBasisTransformer(
        vocab_size=64,
        max_length=16,
        dimension=32,
        hidden_dimension=64,
        num_heads=4,
        depth=4,
        num_bases=2,
    ).eval()
    tokens = torch.randint(0, 64, (8, 16))
    with torch.no_grad():
        logits = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), tokens.reshape(-1))
    assert torch.isfinite(loss)
    assert float(loss) < 6.0
