import torch
from src.llm_decisions import balanced_assignment,balanced_effective_partition,partition_difference,refit_bases
from experiments.calibrate_llm_recovery import choose_threshold,accepted


def test_balanced_partitions_preserve_labels_and_budgets():
    torch.manual_seed(2)
    a=torch.randn(12,4,dtype=torch.double).softmax(-1)
    labels=balanced_assignment(a)
    permuted=balanced_assignment(a[:,[3,0,1,2]])
    assert partition_difference(labels,permuted)==0
    assert torch.bincount(labels).tolist()==[3,3,3,3]
    b=torch.randn(4,35,dtype=torch.double)
    theta=a@b
    reduced=a@torch.linalg.cholesky(b@b.T)
    direct=balanced_effective_partition(theta,4)
    compact=balanced_effective_partition(reduced,4)
    assert partition_difference(direct,compact)==0
    fitted=refit_bases(theta,labels,4)
    for group in range(4):
        torch.testing.assert_close(fitted[group],theta[labels==group].mean(0))


def test_calibration_keeps_ties_together_and_failed_cases_in_denominator():
    rows=[]
    for i in range(40):
        rows.append(dict(status='ok',diagnostics=dict(admissible=True,empirical_score=1 if i<20 else 2),
                         evaluation=dict(quality='good' if i<20 else 'bad')))
    rows.append(dict(status='failed'))
    policy=choose_threshold(rows,'empirical_score')
    assert policy['threshold']==1 and policy['accepted']==20 and policy['total']==41
    assert accepted(rows[0],'empirical_score',policy)
    assert not accepted(rows[20],'empirical_score',policy)
    assert not accepted(rows[-1],'empirical_score',policy)
    policy=choose_threshold(rows[20:],'empirical_score')
    assert policy['threshold'] is None
