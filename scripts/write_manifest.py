"""Write the SHA-256 inventory for files included in the released artifact."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


EXCLUDED_DIRECTORIES = {
    ".audit_remote_snapshot",
    ".cache",
    ".git",
    ".pytest_cache",
    ".tmp",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
EXCLUDED_PART_PREFIXES = (
    ".audit",
    ".artifact-",
    ".final",
    ".metadata-",
    ".pip-",
    ".pytest_",
    ".pytest-",
    ".test-tmp",
    ".testtmp_",
    ".tmp_",
    ".wheel-",
    "_audit_",
    "_pytest_",
    "_tmp_",
    "pytest_tmp",
    "prior_art_",
    "test-tmp-",
)
EXCLUDED_NAMES = {
    "MANIFEST.sha256",
    "main.aux",
    "main.bbl",
    "main.bcf",
    "main.blg",
    "main.fdb_latexmk",
    "main.fls",
    "main.lof",
    "main.log",
    "main.lot",
    "main.out",
    "main.run.xml",
    "main.synctex.gz",
    "main.toc",
    "texput.log",
}


def _excluded_relative(relative: Path) -> bool:
    """Return whether a repository-relative path is an ephemeral artifact."""

    return (
        any(
            part in EXCLUDED_DIRECTORIES
            or part.endswith(".egg-info")
            or part.startswith(EXCLUDED_PART_PREFIXES)
            for part in relative.parts
        )
        or relative.name in EXCLUDED_NAMES
        or relative.suffix in {".pyc", ".pyo"}
    )


def _included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        path.is_file()
        and not path.is_symlink()
        and not _excluded_relative(relative)
    )


def _raise_walk_error(error: OSError) -> None:
    """Fail closed instead of silently omitting an unreadable release path."""

    raise error


def _manifest_text(root: Path) -> str:
    paths = []
    for directory, directory_names, file_names in os.walk(
        root, topdown=True, onerror=_raise_walk_error
    ):
        directory_path = Path(directory)
        directory_names[:] = [
            name
            for name in directory_names
            if not _excluded_relative(
                (directory_path / name).relative_to(root)
            )
        ]
        paths.extend(
            path
            for name in file_names
            if _included((path := directory_path / name), root)
        )
    paths.sort(key=lambda path: path.relative_to(root).as_posix())
    lines = []
    for path in paths:
        digest_state = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest_state.update(block)
        digest = digest_state.hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = root / "MANIFEST.sha256"
    expected = _manifest_text(root)
    if args.check:
        observed = target.read_text(encoding="utf-8")
        if observed != expected:
            raise SystemExit("MANIFEST.sha256 does not match the current artifact")
        print(f"verified {target.name}")
        return
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(expected)
    print(f"wrote {target.name} with {expected.count(chr(10))} entries")


if __name__ == "__main__":
    main()
