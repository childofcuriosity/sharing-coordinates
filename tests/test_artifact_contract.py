import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_released_inputs_match_documented_hashes():
    expected = {
        "data/wikitext-2/train.txt": (
            "9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f"
        ),
        "data/wikitext-2/valid.txt": (
            "f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8"
        ),
        "data/wikitext-2/test.txt": (
            "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0"
        ),
        "results/checkpoints/language_seed0.pt": (
            "6f9fce092e9a7dba6a24de9178736425cbf27d62b2e385ecce0b2210a217324d"
        ),
        "results/checkpoints/language_seed1.pt": (
            "ae09aaf044be8bbcdc7acf08827d96a47423a4422bff361e2d0b82513ab7a4d5"
        ),
        "results/checkpoints/language_seed2.pt": (
            "77006e2d6d5c024ddca274f3cd029c9d4672a0c53ed1d99ac7528453521c2fe4"
        ),
    }
    for relative, expected_digest in expected.items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == expected_digest


def test_documented_primary_numbers_are_bound_to_validated_summaries():
    response = _json("response_summary.json")
    assert response["across_seed"]["maximum_effective_relative_l2_error"] == pytest.approx(
        1.697808166886258e-16
    )
    assert [
        row["nonpermutation_median_relative_response_difference"]
        for row in response["seed_rows"]
    ] == pytest.approx([0.5044997862109962, 0.5051310351033897, 0.50579142960789])
    assert sum(row["detected_probes"] for row in response["seed_rows"]) == 24

    tomography = _json("response_tomography_summary.json")
    assert tomography["format"] == "response-tomography-summary-v2"
    assert tomography["probe_count_K"] == 4
    assert tomography["dimensions"] == {
        "depth_L": 12,
        "num_bases_K": 4,
        "effective_D": 787_712,
    }
    tomography_across = tomography["across_seed"]
    assert tomography_across[
        "maximum_component_reconstruction_relative_error"
    ] == pytest.approx(6.853653630597927e-11)
    assert tomography_across[
        "maximum_permutation_probe_relative_response_difference"
    ] == pytest.approx(9.43560526260365e-17)
    assert tomography_across[
        "minimum_nonpermutation_probe_relative_response_difference"
    ] == pytest.approx(0.9730757594220084)

    aslora = _json("aslora_summary.json")
    primary = {
        row["candidate_mode"]: row
        for row in aslora["across_seed_mode_scope"]
        if row["decision_scope"] == "per_projection"
    }
    assert primary["adjacent"]["action_change_counts_by_seed"] == {
        "0": 2,
        "1": 6,
        "2": 3,
    }
    assert primary["adjacent"]["total_action_change_count"] == 11
    assert primary["all"]["action_change_counts_by_seed"] == {
        "0": 4,
        "1": 7,
        "2": 3,
    }
    assert primary["all"]["total_action_change_count"] == 14
    coverage = aslora["validation"]["candidate_coverage"]
    assert [
        coverage[str(seed)]["actual_candidate_evaluation_count"]
        for seed in range(3)
    ] == [204, 208, 200]
    assert aslora["validation"]["fully_attested_reproducibility_gate"] is True
    for seed in range(3):
        gates = aslora["validation"]["equivalence_and_restoration"][str(seed)]
        assert gates["deterministic_primary_action_repeatability_all_pass"] is True
        assert gates["factorized_float32_reexecution_all_pass"] is True
        assert gates["factorized_float32_role"] == (
            "diagnostic_only_not_scientific_gate"
        )

    router = _json("router_decisions_summary.json")
    assert router["across_seed"]["partition_changing_gauge_found_count"] == 1
    assert router["across_seed"]["primary_claim_supported"] is False
    witness = next(row for row in router["seed_rows"] if row["status"] == "completed")
    assert witness["reference_fold_bpb"] == pytest.approx(2.1753, abs=5e-5)
    assert witness["candidate_fold_bpb"] == pytest.approx(4.8149, abs=5e-5)
    assert witness["delta_bpb"] == pytest.approx(2.6396, abs=5e-5)
    assert witness["paired_bootstrap_ci_lower"] == pytest.approx(2.6287, abs=5e-5)
    assert witness["paired_bootstrap_ci_upper"] == pytest.approx(2.6510, abs=5e-5)
    assert witness["gauge_invariant_baselines_stored"] is True
    assert witness["kmeans_bpb"] == pytest.approx(2.1753, abs=5e-5)
    assert witness["ward_bpb"] == pytest.approx(2.1753, abs=5e-5)
    failed = [row for row in router["seed_rows"] if row["status"] != "completed"]
    assert len(failed) == 2
    assert all(row["decision_evaluations_stored"] is False for row in failed)
    assert all(row["gauge_invariant_baselines_stored"] is False for row in failed)
    assert router["across_seed"]["initialization_dominated_seed_count"] == 3
    assert router["across_seed"]["learned_system_eligibility"] is False
    assert router["protocol"]["executed_search_families"] == [
        "positive_stochastic"
    ]
    assert router["protocol"]["executed_sensitivity_analyses"] == []


def test_primary_summaries_use_checkout_portable_input_paths():
    response = _json("response_summary.json")
    assert response["input_files"] == [
        f"results/primary_remote/response_audit_seed{seed}.json"
        for seed in range(3)
    ]

    router = _json("router_decisions_summary.json")
    assert router["input_files"] == [
        f"results/primary_remote/router_decisions_seed{seed}.json"
        for seed in range(3)
    ]

    aslora = _json("aslora_summary.json")
    for seed, record in enumerate(aslora["inputs"]):
        prefix = f"results/aslora_seed{seed}"
        assert record["audit_path"] == f"{prefix}/audit.json"
        assert record["provenance_path"] == f"{prefix}/provenance.json"
        assert record["snapshot_path"] == f"{prefix}/pre_first_merge_snapshot.pt"


def test_documented_release_entry_points_and_outputs_exist():
    required = (
        "scripts/run_all.sh",
        "scripts/write_manifest.py",
        "scripts/download_wikitext2.sh",
        "src/response_tomography.py",
        "experiments/run_response_tomography.py",
        "experiments/summarize_response_tomography.py",
        "experiments/generate_response_tomography_latex.py",
        "paper/main.tex",
        "paper/main.pdf",
        "paper/generated/response_table.tex",
        "paper/generated/response_tomography_numbers.tex",
        "paper/generated/aslora_table.tex",
        "paper/generated/aslora_metric_detail_table.tex",
        "paper/generated/aslora_baseline_table.tex",
        "paper/generated/router_decision_table.tex",
        "paper/generated/router_decision_detail_table.tex",
        "research/PRIOR_ART_REGISTRY.md",
        "research/LITERATURE_EVIDENCE.md",
        "research/RESPONSE_IDENTIFIABILITY.md",
        "research/RESPONSE_TOMOGRAPHY.md",
        "research/RESPONSE_CANONICAL_INDEPENDENT_AUDIT.md",
        "research/ROUTER_DECISION_CANONICAL_INDEPENDENT_AUDIT.md",
        "research/EMPIRICAL_PROTOCOL.md",
        "research/DECISION_THEORY.md",
        "research/ASLORA_EMPIRICAL_PROTOCOL.md",
        "research/ASLORA_CANONICAL_INDEPENDENT_AUDIT.md",
        "results/response_summary.json",
        "results/response_tomography_seed0.json",
        "results/response_tomography_seed1.json",
        "results/response_tomography_seed2.json",
        "results/response_tomography_summary.json",
        "results/aslora_summary.json",
        "results/router_decisions_summary.json",
        "README.md",
        "RESULTS.md",
        "REPRODUCIBILITY.md",
        "MANIFEST.sha256",
    )
    for relative in required:
        assert (ROOT / relative).is_file(), relative

    runner = (ROOT / "scripts/run_all.sh").read_text(encoding="utf-8")
    for mode in ("verify", "derive", "byte-lm", "aslora", "legacy", "release"):
        assert f"  {mode})" in runner
    assert '--probe-seed "$((740000 + seed))"' in runner
    assert "--autograd-seed 750000" in runner
    assert "experiments.run_response_tomography" in runner
    assert "experiments.summarize_response_tomography" in runner
    assert "experiments.generate_response_tomography_latex" in runner
    assert 'if [[ ! -d "$ASLORA_MODEL" ]]' in runner
    assert "ASLORA_MODEL must be a local roberta-base directory" in runner
    assert '"${ASLORA_LOCAL_FILES_ONLY:-1}" == "1"' in runner
    assert "cmp -s" in runner
    assert "cmp --silent" not in runner


def test_docs_disclose_aslora_undefined_update_ratio_and_scope():
    for relative in (
        "README.md",
        "RESULTS.md",
        "REPRODUCIBILITY.md",
        "research/PRIOR_ART_REGISTRY.md",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "lambda=0.5" in text
    results_text = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    assert "documented reimplementation" in results_text
    assert "not an official reproduction" in results_text
    assert "not estimates" in results_text or "not a prevalence" in results_text


def test_docs_disclose_tomography_analytic_evaluation_boundary():
    for relative in ("README.md", "RESULTS.md", "REPRODUCIBILITY.md"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        normalized = " ".join(text.split())
        assert "4 x 32" in text
        assert "full-model JVP" in normalized
        assert "analytic" in text


def test_manuscript_defines_initialization_dominated_flag():
    appendix = (ROOT / "paper/sections/appendix.tex").read_text(encoding="utf-8")
    assert "strictly below $0.01$" in appendix
    assert "round to $0.000313$--$0.000401$" in appendix
    assert "not recovery of a" in appendix
    assert "historical graph" in appendix


def test_results_report_aslora_invariant_baseline_outcome():
    text = (ROOT / "RESULTS.md").read_text(encoding="utf-8")
    assert "`2/2/2`" in text
    assert "`2/1/3`" in text
    assert "neither invariant rule uniformly improves" in text
    assert "all-pairs reading for seeds 0 and 1" in text
