from pathlib import Path

from experiments.analyze_results import _write_text as write_legacy_text
from experiments.summarize_response_audit import _write_text_lf as write_response_text
from experiments.summarize_response_tomography import _write_json_lf
from experiments.summarize_router_decisions import _write_text_lf as write_router_text


ROOT = Path(__file__).resolve().parents[1]


def _canonical_non_csv_derived_files():
    result_names = (
        "response_summary.json",
        "response_table.md",
        "response_tomography_summary.json",
        "aslora_summary.json",
        "aslora_summary.md",
        "router_decisions_summary.json",
        "router_decisions_table.md",
        "summary.json",
    )
    yield from (ROOT / "results" / name for name in result_names)
    yield from sorted((ROOT / "paper" / "generated").glob("*.tex"))


def test_canonical_non_csv_derived_text_is_lf_only():
    paths = tuple(_canonical_non_csv_derived_files())
    assert paths
    for path in paths:
        assert path.is_file(), path
        assert b"\r" not in path.read_bytes(), path


def test_verify_path_text_writers_emit_lf_on_windows(tmp_path):
    plain_writers = (write_response_text, write_router_text, write_legacy_text)
    for index, writer in enumerate(plain_writers):
        output = tmp_path / f"plain-{index}.txt"
        writer(output, "alpha\nbeta\n")
        assert output.read_bytes() == b"alpha\nbeta\n"

    json_output = tmp_path / "summary.json"
    _write_json_lf(json_output, {"alpha": 1, "beta": [2, 3]})
    assert b"\r" not in json_output.read_bytes()
    assert json_output.read_bytes().endswith(b"\n")


def test_verify_generators_do_not_use_platform_default_write_text():
    generator_paths = (
        "experiments/summarize_response_audit.py",
        "experiments/summarize_response_tomography.py",
        "experiments/summarize_aslora_audit.py",
        "experiments/summarize_router_decisions.py",
        "experiments/generate_paper_audit_tables.py",
        "experiments/generate_response_tomography_latex.py",
        "experiments/analyze_results.py",
    )
    for relative in generator_paths:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert ".write_text(" not in source, relative


def test_verify_mode_confines_python_and_matplotlib_side_effects():
    script = (ROOT / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    verify_body = script.split("verify_release() (", 1)[1].split("\n)\n", 1)[0]
    assert "export PYTHONDONTWRITEBYTECODE=1" in verify_body
    assert 'MPLCONFIGDIR="$verify_dir/matplotlib"' in verify_body
    assert 'run_tests "$verify_dir/pytest"' in verify_body
    assert 'MPLCONFIGDIR="$pytest_dir/matplotlib"' in script
    assert "-p no:cacheprovider" in script
