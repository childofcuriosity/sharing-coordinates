# Real pretrained LM training trajectories: exploratory extension

2026-09-25, fixed before running this extension. Prior endpoint geometry and the constructed toy crossing are already known. This is an exploratory, outcome-independent grid, not a blinded confirmatory study or a population prevalence estimate.

Models available locally: Pythia-160M, Qwen3-0.6B, SmolLM2-360M. Seeds 0,1,2. Optimizers: AdamW (zero decay, no momentum customization) and plain SGD (no momentum/decay), 18 runs total. Each run starts with the repository's ordinary random shared-adapter initialization, not a constructed boundary or selected checkpoint. Frozen pretrained backbone; 12 attention-output projections selected uniformly across depth, 64 output channels each, four shared bases. Pythia uses every layer. Selecting 12 layers on the other models keeps exact balanced partition enumeration feasible; this differs from their earlier all-layer study and will be disclosed.

Task: WikiText-2 training split, ordinary next-token cross entropy, original 128-token length and batch size 4. Same seed-defined batches and original 256-step warmup/cosine learning-rate schedule (basis .001, logits .01), gradient clipping at norm 1. Backbone BF16/autocast, factors FP32. Both optimizers use the same schedule, without retuning for SGD. Loss outcomes are not an optimizer-quality comparison.

At initialization and after EVERY update, record A and BB^T in FP64, all decision gaps and winners. Coefficient assignment: exact capacity-constrained Hungarian maximum. Weight and router-vector clustering: exhaustive enumeration of all 15,400 partitions into four groups of three. No heuristic k-means result enters the principal decision criteria.

Distinct events, all reported:
1. strict coefficient/weight agreement at step s-1, strict disagreement at s;
2. the stronger router-boundary event: event 1 with unchanged global weight partition (coefficient partition changed);
3. strict router-k-means/weight-k-means agreement at s-1, strict disagreement at s (the response-optimal comparison from the theorem).
A strict gap is >1e-8*max(1,absolute winning objective). Save all runs, including initial disagreements and no-event runs. Also report whether disagreement persists for five consecutive states after each event. Multiple events in one run are not independent replicates.

Save first event factors before/after, the preceding optimizer state, gradient arrays, exact batch IDs, clipping scale and actual updates. This permits local replay and comparison of the observed router boundary-score change with its differential prediction. Use ordinary validation-split language-model loss at initialization and completion to establish that this is a real task adaptation, not optimization of a grouping objective. Do not select runs or step sizes using test loss.

Control interpretation: the separately fixed squared-regression experiment checks chain-rule gradients, discrete-step convergence and a stationary target. The actual-language experiment checks whether analogous decisions change under natural task gradients. AdamW/stochastic minibatches are not asserted to satisfy the continuous Euclidean theorem. An observed event establishes occurrence in this adapter setup, not broad LLM prevalence or downstream benefit of recovery.

All outputs go under results/llm_routing_trajectories; original frozen study remains untouched. Record source/protocol hashes, model revision, tokens, package versions, initialization/final states and full trajectory. No optional stopping or replacement of failed/no-event seeds.
