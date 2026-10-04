# Prose revision, 2026-10-04

This revision polishes the README, abstract, main-text sections, and explanatory prose in the experimental and literature appendices. The baseline is commit `b3efd67`. It replaces audit-style wording, repeated defensive framing, and awkward compounds with direct descriptions of the questions, results, and limitations. It also repairs missing spaces in model, software, and experimental settings.

Displayed equations, citations, labels, and cross-references were compared with the baseline and are unchanged. Numerical outcomes and scientific qualifications were retained. The frozen proof appendices, AI assistance disclosure, bibliography, generated tables, figures, experiment code, and raw results were not edited. The README now distinguishes local file verification from full experimental replay, consistent with `INTEGRATION.md`.

`paper/main.pdf` was rebuilt from the revised source with latexmk, pdfLaTeX, and BibTeX. The 73-page output has no undefined references, missing-character warnings, or overfull boxes. All pages were rendered for a layout overview; the first page and selected experimental, discussion, and literature pages were also inspected at higher resolution.

Earlier publication and integration records describe their original versions. This prose revision does not retroactively extend their hash attestations to the new manuscript. Git retains the preceding source and PDF; the regenerated root `MANIFEST.sha256` covers the current files. No experiments were rerun for this language edit.
