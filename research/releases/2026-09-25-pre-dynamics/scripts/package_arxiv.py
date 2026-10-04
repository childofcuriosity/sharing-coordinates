"""Build and verify a minimal local arXiv source draft; does not upload."""

import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper'


def command(args, cwd, logfile):
    with logfile.open('w') as handle:
        subprocess.run(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main():
    output = ROOT/'dist'
    output.mkdir(exist_ok=True)
    command(['latexmk', '-g', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
             'main.tex'], PAPER, output/'arxiv-original-build.log')
    # The TeX recorder identifies files actually used, including conditional
    # inputs. Standard distribution packages are resolved by TeX, not bundled.
    files = {Path('references.bib'), Path('style/iclr2027/iclr2027_conference.bst')}
    allowed = {'.tex', '.sty', '.pdf', '.bbl'}
    for line in (PAPER/'main.fls').read_text().splitlines():
        if not line.startswith('INPUT '):
            continue
        path = (PAPER/line[6:]).resolve()
        if not path.is_relative_to(PAPER) or not path.is_file():
            continue
        relative = path.relative_to(PAPER)
        if path.suffix in allowed and relative != Path('main.pdf'):
            files.add(relative)
    if Path('main.tex') not in files or Path('main.bbl') not in files:
        raise RuntimeError('Incomplete recorded main document or bibliography')
    payloads = {str(name): (PAPER/name).read_bytes() for name in sorted(files)}
    archive = output/'sharing-coordinates-arxiv-draft.tar.gz'
    with archive.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as tar:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(payload), 0o644, 0
                    tar.addfile(info, io.BytesIO(payload))
    with tempfile.TemporaryDirectory(prefix='sharing-arxiv-') as directory:
        fresh = Path(directory)
        # Verify the archived bytes themselves, not a separate staging tree.
        with tarfile.open(archive) as tar:
            tar.extractall(fresh, filter='data')
        for iteration in range(4):
            command(['pdflatex', '-no-shell-escape', '-interaction=nonstopmode',
                     '-halt-on-error', 'main.tex'], fresh,
                    output/f'arxiv-clean-build-{iteration+1}.log')
            log = (fresh/'main.log').read_text()
            if iteration >= 1 and not any(message in log for message in
                    ('Rerun to get cross-references right', 'Rerun to get outlines right')):
                break
        log = (fresh/'main.log').read_text()
        problems = [line for line in log.splitlines()
                    if ('undefined' in line.lower() or 'Overfull' in line
                        or 'Rerun to get cross-references right' in line)]
        if problems:
            raise RuntimeError('Clean build diagnostics: '+repr(problems))
        def extracted_text(path):
            return subprocess.check_output(['pdftotext', '-layout', str(path), '-'])
        if extracted_text(fresh/'main.pdf') != extracted_text(PAPER/'main.pdf'):
            raise RuntimeError('Archive PDF text differs from the working manuscript')
        shutil.copyfile(fresh/'main.pdf', output/'sharing-coordinates-preview.pdf')
    report = {
        'format': 'arxiv-draft-package-v1',
        'status': 'local source package; not submitted; scientific assessment is separate',
        'archive': archive.name,
        'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'files': {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
        'clean_pdflatex_builds': iteration+1,
        'pdf_text_matches_worktree': True,
        'author_placeholder_present': 'Author information pending' in (PAPER/'main.tex').read_text(),
        'remaining': ['Author metadata is deferred by the user',
                      'Actual arXiv processing has not been tested'],
        'scientific_assessment': 'research/LLM_REVIEW.md; packaging does not certify science',
    }
    (output/'arxiv-package-report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'archive': str(archive), 'files': len(files),
                      'clean_compile': True, 'text_matches': True}))


if __name__ == '__main__':
    main()
