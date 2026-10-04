# Sharing Coordinates

**Identifiability and Decisions from Optimizer Response**

[Paper](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Reproduction guide](REPRODUCIBILITY.md) · [Task-weighted follow-up](studies/task-weighted-response/REPORT.md)

When layers share a bank of learned parameters, different routers and bases can
produce exactly the same effective weights. Those representations can still
respond differently to a training update and suggest different layers to tie
together.

We study what optimizer response tells us about a model's current sharing
coordinates, and how much of that information is actually useful for choosing a
sharing partition. This repository contains the paper, implementations, and
experimental records.

## The idea

Our model writes the effective layer weights as

$$
\Theta = AB,
$$

where each row of the router $A$ mixes the learned bases in $B$. Observing
$\Theta$ alone generally leaves several possible factorizations. We ask what
changes if we can also measure how $\Theta$ responds to controlled gradients
under a specified optimizer.

For strictly positive, full-rank softmax routers and full-rank bases with the
same product, the complete Euclidean response identifies the factors up to a
common permutation, provided both representations use the same known positive
learning rates and temperature. The paper also gives stability bounds and a
finite-query construction for noisy observations under additional dimension,
conditioning, and noise assumptions.

Choosing a partition is a different question. **Balanced response-optimal tying
needs only off-diagonal entries of the router Gram matrix.** Recovering every
factor is unnecessary for that objective. Its preferred partition can differ
from weight clustering, and training can move the two decisions apart.

## What we found

| Experiment | Main observation |
| --- | --- |
| Factor recovery on nine pretrained models, from 160M to 8B parameters | Across 27 adapter checkpoints, all 1,080 cases meeting the observed rank conditions had joint-fit factor error below `0.034`. |
| Sharing decisions over 18 language-training runs | Four of nine AdamW runs entered coefficient/weight disagreement; all nine entered router-clustering/weight disagreement. None of nine SGD runs crossed under the unretuned schedule. |
| Matched decisions at nine checkpoints | Recovered routers reproduced the native partitions, but effective-weight clustering found the same partitions. Full factor recovery gave no additional benefit in this test. |

These experiments train shared modules on frozen language-model backbones. The
recovery measurements use controlled Euclidean responses; the AdamW runs study
how decisions change during training. The crossing counts describe this finite,
dependent set of runs, rather than a frequency estimate for training in general.

The repository also includes an [ASLoRA first-merge study](research/ASLORA_EMPIRICAL_PROTOCOL.md),
byte-level language-model experiments, and synthetic checks. The ASLoRA study
uses our reimplementation of the paper's rule and locally trained checkpoints.
Detailed protocols and results are in the paper's appendices and `research/`.

### Does a better response match improve task performance?

Our [task-weighted follow-up](studies/task-weighted-response/REPORT.md) tests this
directly. It derives a grouping objective from task gradients and compares seven
methods at nine checkpoints, with both SGD and AdamW recovery.

The corrected experiment found no consistent advantage over weight clustering:
after SGD recovery, task-weighted grouping was better at three checkpoints,
equal at two, and worse at four. Its mean NLL difference was `+0.005224`
nats/token. Matching the current response more closely did not reliably improve
the task loss. Both the original experiment and its numerical correction are
available in [the study directory](studies/task-weighted-response/).

## Getting started

Install the project and test dependencies from the repository root:

```bash
python -m pip install -e ".[test]"
```

For a small CPU check of the core factorization and decision results:

```bash
python -m pytest -q tests/test_factorization.py tests/test_gauge.py tests/test_routing_geometry.py
python -m scripts.check_decision_certificates
```

The [reproduction guide](REPRODUCIBILITY.md) covers training, result generation,
and full verification. The optional `aslora` and `llm` dependency groups support
the corresponding experiments.

Some large inputs for the newer experiments, including checkpoints, raw gradients,
and model caches, are kept outside this repository. See [INTEGRATION.md](INTEGRATION.md)
for what is included and what a full replay needs. To verify the files shipped
here:

```bash
python -X utf8 scripts/write_manifest.py --check
```

## Where to look

| Path | Contents |
| --- | --- |
| [`paper/`](paper/) | Manuscript, proofs, figures, and detailed experimental results |
| [`src/`](src/) | Factorization, response recovery, and sharing-decision implementations |
| [`experiments/`](experiments/) | Training, measurement, analysis, and table-generation scripts |
| [`tests/`](tests/) | Mathematical examples, implementation checks, and result-validation tests |
| [`results/`](results/) | Saved experiment records and summaries |
| [`research/`](research/) | Derivations, protocols, prior-work comparisons, and validation records |
| [`studies/task-weighted-response/`](studies/task-weighted-response/) | Follow-up on task gradients and downstream decision quality |

The paper is a preprint draft with author information still pending. Its
[AI assistance disclosure](paper/sections/ai_disclosure.tex) is included in the
manuscript. Earlier release and validation records remain in the repository;
each applies to the version it names.
