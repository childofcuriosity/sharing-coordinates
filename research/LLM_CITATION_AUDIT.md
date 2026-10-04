# LLM study citation audit

Six BibTeX entries were fetched from the official arXiv BibTeX endpoints; raw responses and HTTP statuses are retained in llm-citation-sources/. The bibliography only changes citation keys and protects title capitalization. Titles, authors and years are retained from the responses.

- biderman2023pythia (2304.01373): Pythia model suite and its controlled data/order design.
- yang2025qwen3 (2505.09388): Qwen3 family; our exact non-Base dense model IDs/revisions are in LLM_ACCESS_INVENTORY.json. This is not an architecture-only comparison.
- allal2025smollm2 (2502.02737): SmolLM2 family; the two exact base-model sizes are independently verified in the pinned model cards/configs.
- merity2016wikitext (1609.07843): provenance of WikiText corpus, not an assertion that our block evaluation matches its official benchmark protocol.
- cobbe2021gsm8k (2110.14168): provenance of math word-problem text. We measure token NLL and gradients, not answer accuracy.
- austin2021mbpp (2108.07732): provenance of Python-programming text. We measure token NLL and gradients, not executed-program correctness.

Official abstract pages were checked for model-suite or dataset-introduction claims; exact downloaded data commits/hashes and model revisions are separately recorded. No new state-of-the-art, priority, contamination-free or unrestricted LLM-generalization claim is made.
