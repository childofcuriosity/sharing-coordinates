import json
from pathlib import Path
import pytest
from experiments.summarize_enhancement import read_bound,paired_interval

def test_source_corruption_is_rejected(tmp_path):
    p=tmp_path/'bad.json'
    p.write_text(json.dumps({'source_sha256':{'src/enhancement.py':'0'*64}}))
    with pytest.raises(ValueError,match='Source hash mismatch'):
        read_bound(p)

def test_paired_constant_effect_has_exact_interval():
    result=paired_interval([3.,4.,5.],[1.,2.,3.],19)
    assert result['delta_bpb']==2
    assert result['conditional_bootstrap_95']==[2,2]
