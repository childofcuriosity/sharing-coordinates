import copy

import pytest

from experiments.generate_paper_audit_tables import (
    DEFAULT_ASLORA,
    DEFAULT_RESPONSE,
    DEFAULT_ROUTER,
    aslora_baseline_table,
    aslora_metric_detail_table,
    aslora_table,
    generate_tables,
    response_table,
    router_decision_detail_table,
    router_decision_table,
    _load,
)


def test_primary_tables_are_generated_only_from_validated_summaries():
    outputs = generate_tables(DEFAULT_RESPONSE, DEFAULT_ASLORA, DEFAULT_ROUTER)
    assert set(outputs) == {
        "response_table.tex",
        "aslora_table.tex",
        "aslora_metric_detail_table.tex",
        "aslora_baseline_table.tex",
        "router_decision_table.tex",
        "router_decision_detail_table.tex",
    }

    response = outputs["response_table.tex"]
    assert "\\ResponseProbeDetected}{24}" in response
    assert "\\ResponseFormulaMax}{\\ensuremath{9.57\\!\\times\\!10^{-13}}}" in response
    assert "0.5045" in response and "0.5058" in response
    assert "8/8" in response

    aslora = outputs["aslora_table.tex"]
    assert "\\ASLORAAdjacentPrimaryChanges}{11}" in aslora
    assert "\\ASLORAAllPrimaryChanges}{14}" in aslora
    assert "$(2,6,3)$ & 11/30 & 3/3" in aslora
    assert "$(4,7,3)$ & 14/30 & 3/3" in aslora
    assert "not a full Cartesian per-projection oracle" not in aslora
    assert "\\resizebox" not in aslora
    assert "$\\Delta$ accuracy" not in aslora
    assert "$\\Delta$ F1" not in aslora
    assert "$[-0.0531,+0.0089]$" in aslora

    aslora_metrics = outputs["aslora_metric_detail_table.tex"]
    assert "$\\Delta$ accuracy" in aslora_metrics
    assert "$\\Delta$ F1" in aslora_metrics
    assert "$[-0.0025,+0.0098]$" in aslora_metrics
    assert "$[-0.0022,+0.0057]$" in aslora_metrics
    assert "\\resizebox" not in aslora_metrics

    aslora_baselines = outputs["aslora_baseline_table.tex"]
    assert "not a full Cartesian per-projection oracle" in aslora_baselines
    assert "effective has 2 negative/2 positive/2 zero" in aslora_baselines
    assert "local activation has 2 negative/1 positive/3 zero" in aslora_baselines
    assert "$\\Delta L=-0.0336$" in aslora_baselines
    assert "$\\Delta L=+0.0028$" in aslora_baselines
    assert "$\\Delta L=-0.0008$" in aslora_baselines
    assert "$\\Delta L=+0.0001$" in aslora_baselines
    assert "$\\Delta L=+0.0000$" in aslora_baselines

    router = outputs["router_decision_table.tex"]
    assert "\\RouterSearchSuccesses}{1}" in router
    assert "2.175$\\to$4.815" in router
    assert "$+2.640\\;[+2.629,+2.651]$" in router
    assert "\\RouterWitnessKMeansBPB}{2.1753}" in router
    assert "\\RouterWitnessWardBPB}{2.1753}" in router
    assert "trained, initialization-dominated byte-LM checkpoints" in router
    assert "one witness in three seeds" in router
    assert "no (1/3)" in router
    assert "\\resizebox" not in router
    assert "Soft BPB" not in router
    assert "Router $L_1$ move" not in router

    router_details = outputs["router_decision_detail_table.tex"]
    assert "5000/5000/4586/4586/3762/1" in router_details
    assert "5000/5000/4601/4601/3790/0" in router_details
    assert "5000/5000/4581/4581/3769/0" in router_details
    assert "3/3/3/3$\\to$5/3/3/1" in router_details
    assert "2.175/2.175" in router_details
    assert "4.624" in router_details
    assert "\\resizebox" not in router_details


def test_generated_header_uses_logical_release_path_for_copied_summary(tmp_path):
    copied = tmp_path / "response_summary.json"
    copied.write_bytes(DEFAULT_RESPONSE.read_bytes())
    rendered = response_table(_load(copied), copied)
    assert "% Source: results/response_summary.json" in rendered
    assert str(tmp_path) not in rendered


def test_table_generation_rejects_failed_scientific_gates():
    response = copy.deepcopy(_load(DEFAULT_RESPONSE))
    response["across_seed"]["all_formula_checks_pass"] = False
    with pytest.raises(ValueError, match="required control"):
        response_table(response, DEFAULT_RESPONSE)

    aslora = copy.deepcopy(_load(DEFAULT_ASLORA))
    aslora["validation"]["fully_attested_reproducibility_gate"] = False
    with pytest.raises(ValueError, match="source gates"):
        aslora_table(aslora, DEFAULT_ASLORA)
    with pytest.raises(ValueError, match="source gates"):
        aslora_baseline_table(aslora, DEFAULT_ASLORA)
    with pytest.raises(ValueError, match="source gates"):
        aslora_metric_detail_table(aslora, DEFAULT_ASLORA)

    router = copy.deepcopy(_load(DEFAULT_ROUTER))
    router["across_seed"]["partition_changing_gauge_found_count"] = 3
    with pytest.raises(ValueError, match="search counts"):
        router_decision_table(router, DEFAULT_ROUTER)
    with pytest.raises(ValueError, match="search counts"):
        router_decision_detail_table(router, DEFAULT_ROUTER)

    router = copy.deepcopy(_load(DEFAULT_ROUTER))
    router["protocol"]["executed_sensitivity_analyses"] = ["signed_local"]
    with pytest.raises(ValueError, match="execution/amendment status"):
        router_decision_table(router, DEFAULT_ROUTER)
