import torch

from src.metrics import adjusted_rand_index, best_permutation_accuracy, pairwise_graph_accuracy
from src.patterns import assignment_matrix, available_patterns, make_assignment


def test_all_patterns_use_every_basis():
    for pattern in available_patterns():
        ids = make_assignment(pattern, depth=12, num_bases=4, seed=3)
        assert ids.shape == (12,)
        assert set(ids.tolist()) == {0, 1, 2, 3}
        matrix = assignment_matrix(ids, 4)
        assert torch.allclose(matrix.sum(dim=1), torch.ones(12))


def test_recovery_metrics_are_permutation_invariant():
    truth = torch.tensor([0, 0, 1, 2, 1, 2])
    prediction = torch.tensor([2, 2, 0, 1, 0, 1])
    accuracy, _ = best_permutation_accuracy(truth, prediction, 3)
    assert accuracy == 1.0
    assert adjusted_rand_index(truth, prediction) == 1.0
    assert pairwise_graph_accuracy(truth, prediction) == 1.0

