import json
from pathlib import Path

import pytest

from experiments.summarize_autograd_tomography import ROOT, summarize


def test_released_autodiff_records_are_source_bound_and_complete():
    summary = summarize(ROOT / 'results/autograd_tomography')
    assert [row['seed'] for row in summary['rows']] == [0, 1, 2]
    assert len(summary['raw_sha256']) == 3


@pytest.mark.parametrize('corruption', ['formula_error', 'missing_probe', 'checkpoint', 'noise',
                                       'inflated_noise_bound', 'invalid_noise_norm'])
def test_summary_rejects_corrupt_evidence_even_when_pass_flag_is_true(tmp_path, corruption):
    record = json.loads((ROOT / 'results/autograd_tomography/seed0.json').read_text())
    assert record['all_pass'] is True
    if corruption == 'formula_error':
        record['charts']['native']['probes'][0]['formula_relative_error'] = .1
    elif corruption == 'missing_probe':
        record['charts']['native']['heldout'].pop()
    elif corruption == 'checkpoint':
        record['checkpoint_sha256'] = '0' * 64
    elif corruption == 'noise':
        record['noise'][0]['local_absolute_error'] = 1e10
    elif corruption == 'inflated_noise_bound':
        record['noise'][-1]['local_bound'] = 1e10
    else:
        record['noise'][-1]['absolute_noise_norms'][0] = float('nan')
    (tmp_path / 'seed0.json').write_text(json.dumps(record))
    with pytest.raises(ValueError):
        summarize(tmp_path)
