# Local arXiv draft packaging

Status: local source package, not uploaded. The scientific recommendation
is recorded separately in PUBLICATION_REVIEW.md; packaging success itself is
not scientific validation or evidence of acceptance by arXiv's processing
system. Author metadata is deferred by request.
The user explicitly requested retaining the ICLR 2027 template and deferring
author details. The current author placeholder is intentional for this draft.

## Rebuild

From the repository root, using the tested Python 3.12 environment:

```bash
.venv/bin/python scripts/package_arxiv.py
```

The script rebuilds the manuscript, follows its TeX recorder dependencies,
includes the bibliography source, generated bibliography and local style,
and writes `dist/sharing-coordinates-arxiv-draft.tar.gz`. It excludes unused
figures, template examples, auxiliary logs, checkpoints and the manuscript PDF.
It then extracts that archive into a fresh temporary directory, runs pdflatex
without shell escape until references stabilize (at most four passes), and
compares extracted PDF text against the working manuscript. The clean-build
PDF is `dist/sharing-coordinates-preview.pdf`; file hashes and build checks are
in `dist/arxiv-package-report.json`. `dist/` is generated and ignored by Git
and the repository manifest; reproduce it from the packaging script.

Historical first successful run: 39 source/figure/bibliography files, three clean TeX
passes, identical extracted manuscript text, no undefined references or
overfull-box diagnostics. Local TeX package versions may differ from arXiv.

The package follows the source-dependency and bibliography guidance in
[arXiv's TeX submission documentation](https://info.arxiv.org/help/submit_tex.html),
checked 2026-09-24. It retains `main.tex` at the archive root and includes only
the style files actually required. Actual submission requires checking the
PDF produced by arXiv as well.

## Actual submission and author responsibilities

- Author names, order, affiliations and contact information remain placeholders.
- Human authors retain responsibility for the findings, author information,
  and AI-assistance disclosure. The automated reviews do not constitute
  external human peer review; no such review is claimed.
- The scoped scientific recommendation and remaining research limitations
  are documented in PUBLICATION_REVIEW.md.
- Regenerate the archive after any manuscript edit. The generated report is
  about those exact archived bytes, not future versions.

## Enhancement package — 2026-09-25

The historical enhancement draft has 47 PDF pages and 44 archived dependencies.
The archive compiles in a clean directory with identical extracted text, no
undefined references and no overfull boxes. See ENHANCEMENT_VALIDATION.json
for exact file hashes and the scope of empirical and CPU replay checks.

## Theory-focused release package — 2026-09-25

The preserved theory snapshot has a seven-page main text and 54 pages in total.
The 51-file source archive passes three clean-directory pdflatex builds
without shell escape; the extracted PDF text matches the working manuscript.
No undefined references or overfull boxes are reported. Exact hashes are in
RELEASE_VALIDATION.json and dist/arxiv-package-report.json. The older package
counts above are historical. The template and deferred author information
remain unchanged by user instruction. The package has not been sent to arXiv.

## Current pretrained-model extension — 2026-09-25

The manuscript now has 61 pages, with the main text ending on page 8.
Its 60-file manuscript-only archive passes three clean pdflatex builds and
matches the working PDF's extracted text. The added material comprises
pretrained-model experiments, a fixed-isometry response derivation, complete
protocol/accounting and retained negative decision results. Scientific scope
and exact validation are recorded in PUBLICATION_REVIEW.md and LLM_VALIDATION.json.
The earlier 54-page PDF/archive remains under releases/2026-09-25-theory/.
