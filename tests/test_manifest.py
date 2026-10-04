from pathlib import Path

from scripts.write_manifest import _excluded_relative, _included, _manifest_text


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_includes_released_corpora_and_frozen_checkpoints():
    required = (
        ROOT / "data" / "wikitext-2" / "train.txt",
        ROOT / "data" / "wikitext-2" / "valid.txt",
        ROOT / "data" / "wikitext-2" / "test.txt",
        ROOT / "results" / "checkpoints" / "language_seed0.pt",
        ROOT / "results" / "checkpoints" / "language_seed1.pt",
        ROOT / "results" / "checkpoints" / "language_seed2.pt",
    )
    for path in required:
        assert path.is_file()
        assert _included(path, ROOT)

    manifest = _manifest_text(ROOT)
    for path in required:
        assert f"  {path.relative_to(ROOT).as_posix()}\n" in manifest
    for ephemeral in (
        ".audit_remote_snapshot/",
        ".pytest_tmp_aslora_existing/",
        ".tmp_pytest_tomography_semantics/",
        "test-tmp-review/",
        "_pytest_aslora_final/",
        "_audit_response_recompute/",
        "tests/_tmp_aslora_audit_all/",
        "paper/.audit_pages/page-01.png",
        "paper/.final-review-01.png",
        "paper/prior_art_aslora.pdf",
        "router_sharing_graph.egg-info/PKG-INFO",
        "build/lib/package.py",
        "dist/release.whl",
    ):
        assert ephemeral not in manifest


def test_manifest_excludes_ephemeral_build_and_cache_paths():
    assert _excluded_relative(Path("src/__pycache__/module.pyc"))
    assert _excluded_relative(Path("paper/main.aux"))
    assert _excluded_relative(Path("paper/main.synctex.gz"))
    assert _excluded_relative(Path("paper/main.run.xml"))
    assert _excluded_relative(Path(".cache/aslora/model.bin"))
    assert _excluded_relative(Path(".audit_remote_snapshot/src/module.py"))
    assert _excluded_relative(Path(".pytest_tmp_aslora_existing/session"))
    assert _excluded_relative(Path(".tmp_pytest_tomography_semantics/session"))
    assert _excluded_relative(Path("test-tmp-review/session/file"))
    assert _excluded_relative(Path("_pytest_aslora_final/session/file"))
    assert _excluded_relative(Path("_audit_response_recompute/seed0.json"))
    assert _excluded_relative(Path("tests/_tmp_aslora_audit_all/session"))
    assert _excluded_relative(Path("paper/.audit-main-01.png"))
    assert _excluded_relative(Path("paper/.final_pdf_review/page-01.png"))
    assert _excluded_relative(Path("paper/prior_art_2209.12638.pdf"))
    assert _excluded_relative(Path("router_sharing_graph.egg-info/PKG-INFO"))
    assert _excluded_relative(Path("build/lib/package.py"))
    assert _excluded_relative(Path("dist/release.whl"))
    assert not _excluded_relative(Path("paper/main.tex"))
    assert not _excluded_relative(Path("data/wikitext-2/train.txt"))
    assert not _excluded_relative(
        Path("results/checkpoints/language_seed0.pt")
    )
