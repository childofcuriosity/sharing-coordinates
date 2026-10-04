# Rendered manuscript review

The existing ICLR 2027 template is retained, as requested. Visual inspection
is separate from successful TeX compilation.

## Inspected draft

On 2026-09-24, inspected all 43 pages of `paper/main.pdf` as rendered PNGs
at 900 pixels per page. The inspected PDF SHA-256 is
`cc7ce4232918488f19a56ad6364d175743d93d0668d78c36a834ca5e6a3b04b2`.
Pages 35--43 were also inspected at 1,000 pixels during the preceding revision.

- Pages 1--10: title, abstract, introduction, main theory and empirical tables.
  No clipping or overlaps found. The response summary table is dense and uses
  smaller text; this is a remaining readability limitation, not missing data.
- Pages 10--14: reference list, URLs, disclosure and appendix transition. No
  visible margin overflow or overlapping entries found. This check does not
  verify the bibliographic facts or the content of cited papers.
- Pages 14--30: secondary statements, exact and quantitative proofs, tomography,
  factor recovery, edge cases and claim boundaries. Displayed equations and
  numbering fit within the text area. Proof continuation across pages remains
  readable; mathematical validity is tracked in the separate proof review.
- Pages 30--34: detailed prior-art discussion and experimental appendix opening.
  Long paths and identifiers wrap within the text area.
- Pages 35--43: experimental inventory, baselines, secondary results, all four
  figures and final checklist. Axes, legends and color bars remain visible.
  Dense supplemental tables have smaller text but no visible clipping.

## Corrections retained

The inventory and ASLoRA baseline tables previously used excessive whole-table
scaling. The inventory now wraps prose in fixed-width columns; the baseline
formatter splits query/value actions and loss differences into cell lines.
Scientific values and source-summary hashes are unchanged. Pages 35--43 were
rendered again after these corrections and inspected in the final 43-page layout.

## Scope

This is a screen-scale visual pass, not a print accessibility certification.
Some supplemental tables and heatmap annotations require zooming. No further
template redesign is planned. Reinspect affected pages after manuscript edits;
this record applies to the PDF hash above, not to future versions.

## Follow-up after explicit local-response attribution

After adding the Varre Lemma D.2 comparison, the 43-page PDF has SHA-256
`2289b54db13af1f6dc3d4d821ddca4c876eb22bf86e6a31252d066b22a3cb92a`.
Reinspected pages 32--35 at 900 pixels and pages 36--43 at 700 pixels for
changed wrapping/page flow. No overlaps or clipping found. Earlier sections
were not edited. The rebuilt 39-file archive again compiled cleanly and its
extracted PDF text matched the manuscript. Dense-table readability limitations
recorded above remain.

## Follow-up after proceedings metadata update

The PDF SHA-256 after updating the two Wang/Wang references and adding the
Fisher-regularity qualification is
`f902244fcd46fa0f12b4570a2b7ca5fa52fa114a8c8294be52aa0b944b965cce`.
The document remains 43 pages. Inspected pages 13--15 and 32--35 at 900 pixels:
reference wrapping, appendix transition and the qualified comparison fit
without clipping. The archive clean-compiles and its extracted text matches.
This targeted follow-up supplements the earlier full visual pass.

The Khilar version-note correction was rebuilt and its reference on page 12
inspected at 900 pixels; it fits within the same entry layout. Latest PDF
SHA-256: `7840af89d8f61aeb06ed0a3d2c29499ab26e13bf02f4b1c41ce60515b61a7d3f`.
Clean archive compilation and extracted-text equality passed again.

### LoRA-S qualification follow-up

Manuscript SHA-256: `70e320c1e22f348ec94ee782bc1d40309f71fcdbcc5efbfed8f0870d1e7710e7`. Rebuilt the local source archive
and clean-directory preview after clarifying the common product-space
direction used in the LoRA-S invariance comparison. Inspected rendered page
31 at 1500-pixel height: the revised paragraph and surrounding text remain
legible within margins, with no overlap or clipping. This is a targeted
check; earlier full-document visual checks retain their stated scope.
The packaging check passed with identical extracted manuscript/preview text.

## Enhancement review — 2026-09-25

Inspected rendered title, new empirical tables, joint-stability corollary and
proof, natural-probe proposition, and precision/conditioning figure. Final
edits moved the historical fold witness into the appendix, removed duplicated
prose and eliminated an almost-empty trailing page. The final draft has 47
pages. Newly affected table pages 9–10 and final page 47 were checked after
reflow; the proof pages 31–32 were checked at readable resolution. No clipping,
unreadable labels or overlapping equations were observed. The new figure
regenerates byte-identically; final clean TeX compilation has no undefined
references or overfull boxes. This is a targeted follow-up, not a claim of a
new independent scientific review of every page.

## Focused noisy-recovery revision — 2026-09-25

Inspected the 52-page manuscript with PDF SHA-256 6ae04eb330984d1a83f31bd1194e93b7898f3aafbc397cf0fbba3598b0a5274b.
All pages were inspected in rendered contact sheets at 390-pixel page height;
new recovery/table/figure page 11 and page 51 were additionally inspected at
1400 pixels, and local-theory pages 33–34 at 1600 pixels. No clipping or
overlapping text was seen. Tables and the new noise plot are legible at the
larger inspection size. Dense historical appendix tables still require zoom.
The last page has a short continuation paragraph; this is retained without
changing the requested ICLR 2027 template. This is a visual layout check,
not an independent scientific review. Clean archive compilation reports no
overfull boxes or undefined references and matches extracted manuscript text.

## Finite-noisy global theory addition — 2026-09-25

The 55-page draft has PDF SHA-256 49f03f899d7529cfcaffed6584f60f0116ffb5863a775ff74fedd43448953cea.
Inspected the main corollary on page 6, the explicit constants/proof on
pages 35–36, and the transition to the older local theorem on page 37 at
1200-pixel rendering; the domain/probe/feasible-set definitions on page 34
were inspected at 1400 pixels. The abstract was then clarified to state the
dimension condition and product-noise threshold; the archive was rebuilt.
A wide constants display was split across rows. The final clean build has
no overfull boxes or undefined references and matches the archive PDF text.
This checks layout only; proof audits are separately version-bound.
