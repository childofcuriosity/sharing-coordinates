# Prospective cross-model LLM study — initial pilot contract

Date: 2026-09-25. Authorized unattended work; this file precedes any pilot
training or recovery measurement. Final budgets and code will be frozen after
the explicitly labelled pilot and before confirmatory results are inspected.
Existing theory release is preserved in releases/2026-09-25-theory/.

## Target and sampling frame

Study current softmax/shared-basis additive modules on frozen pretrained
causal LMs. This is module-level identification in an actual LLM loss, not
identification of an entire LLM or of its AdamW trajectory. A pretrained
checkpoint is one unit; adaptation seeds are nested repeats, not independent
pretraining. Family and size coverage is purposive, not a random model sample.

Planned models: Pythia 160M/1B/2.8B; Qwen3 dense 0.6B/1.7B/4B/8B;
Llama3.2 1B/3B. Public access checks returned HTTP 401 for both Llama configs.
Pre-outcome fallback is SmolLM2 360M/1.7B, giving nine accessible models across
three families. Record the absent Llama coverage; no bypass or claim of Llama
results. Model revisions/configs are in LLM_ACCESS_INVENTORY.json.
Qwen3 named models include post-training, whereas Pythia/SmolLM2 are base
models; do not interpret family differences as architecture-only effects.

Pilot models: Pythia160M, Qwen3-0.6B, Pythia1B. SmolLM2 is a whole-family
held-out target; no recovery/error/threshold tuning on it. Loading/tokenizer
compatibility checks without outcome inspection are allowed.

## Module and observation

Attach a shared additive map to each attention output projection of matching
shape. All pretrained parameters remain frozen. In the primary feasible
configuration, select 64 fixed, evenly spaced output channels and retain all
input channels. The learned K=4 basis matrices have shape 64 by hidden width;
their embedding into the full output is a fixed isometry, so the Euclidean
AB law applies exactly to this explicitly restricted module. It does not
establish a full-matrix or all-layer-type claim. All matching layers are
included (a multiple-of-four subset is used only if needed for matched hard
partition sizes, selected deterministically before losses).

Router logits and direct basis entries are learned; bases are not themselves
LoRA products. Temperature is one. Bases start small and nonzero; routers
start independently nonuniform. Store initial/final rank margins, basis and
router movement, parameter counts, frozen-baseline loss, and training losses.
No assumption that random initialization alone is a meaningful trained chart.

At fixed checkpoints, obtain actual next-token-loss gradients with respect
to independent effective module parameters. Measure response by reverse-mode
factor gradients and forward-mode JVP of the factor map, independently of
the analytic comparator. Audit FP64 factors using the measured directions;
this isolates recovery arithmetic from BF16 backbone gradients. Finite-step
FP32/BF16 comparisons are reported separately and may fail.

## Tasks, queries, and blind evaluation

General-text adaptation uses existing WikiText-2 train. General probes and
loss evaluation use disjoint validation/test documents. Code (MBPP) and math
(GSM8K) are held-out gradient domains, never used to train the module or tune
recovery. This measures token-loss gradients and losses, not benchmark
generation accuracy; public pretraining contamination is not ruled out.
Hash every raw corpus and record document IDs and tokenizer versions.

Compare (1) the theorem's repeated-complement unit queries designed from the
observed product, (2) seeded Gaussian unit queries, and (3) normalized real
task gradients. Query budgets include insufficient q<K, q=K, and q>K, with
independent held-out responses. Perturb both product and response. Record
noise-free, moderate-noise, and precision/conditioning failures; observed
product subspace is re-estimated under noise. No use of true factors by the
estimator. Truth is supplied to a separate scorer only.

Primary outcomes: permutation-aligned factor errors, unseen response error,
complement/local design ranks and singular values, failed returns, and wall
time/peak VRAM. Robustness includes fixed non-permutation and permutation
controls, sensitivity to noise, and representative rank/interior stress.
Use spectral and constrained joint-fit comparators. Candidate diagnostics
and acceptance thresholds must be fixed using development/calibration data
only, then transferred unchanged to larger sizes, new domains and held-out
family. Preserve all-reject and falsely accepted outputs; report error versus
coverage, not acceptance accuracy alone. A finite sweep cannot validate the
whole-feasible-set universal quantifier or prove practical theorem constants.

## Structural decision loop

On declared representative checkpoints compare hard partitions selected from
native/recovered routers with a simple effective-parameter clustering control.
Use equal nonempty group sizes and K bases, refit from the same effective
parameters, and evaluate both immediate loss and matched recovery training.
Freeze data, optimizer initialization, parameter and update budgets across
actions. Include failed recovery/partition cases in the accounting. Recovery
error does not logically imply a harmful decision or a better compression
method; mixed signs and no benefit are valid outcomes.

## Pilot and freeze rules

Pilot begins with 20 adaptation steps, batch 2, sequence length 128, BF16
frozen backbone and FP32 trainable parameters, AdamW basis lr 1e-3/router lr
1e-2, zero weight decay, fixed K=4/channel width=64. Inspect correctness,
memory, throughput, loss/movement and decoder conditioning. Any change is
documented before confirmation; do not silently replace a failing pilot.
The final adaptation, recovery-training, noise and query budgets are fixed
after this pilot, then code/protocol/data hashes are frozen. Main runs use
three adaptation seeds unless a prospectively documented feasibility change
is necessary. Retain all results, interruptions and implementation corrections.

No upload, leaderboard superiority, all-LLM generality, or efficient global
solver claim is authorized by numerical success. Completion requires a
coherent applicability/failure account and a reviewed manuscript update.

## Final pilot disposition and confirmatory freeze

The three article-split-corrected pilot runs are pythia160m-v3, qwen06-v2,
and pythia1b-v2, all seed90 and 20 steps. They pass the response/permutation
law and same-forward-graph chain checks. Four real directions recover factors
at roughly 1e-7 in exact FP64 observations; one direction does not. This is
pilot evidence only. Pythia160M validation NLL worsens slightly; the other
pilots improve. No model is removed or given a favorable replacement seed.
A separate BF16 backward replay differs by about 1% in one Pythia1B gradient;
FP64 factor-response auditing treats the measured directions as fixed inputs,
and this arithmetic discrepancy is retained rather than called exact
end-to-end gradient agreement.

Frozen main budget: 256 updates, batch4, length128 (131,072 training tokens),
16 warmup steps, cosine decay to 0.1 times initial lr, basis lr1e-3,
router lr1e-2, no weight decay, gradient norm clipping1. Three adaptation
seeds 0/1/2 per model. Last checkpoint only, no validation checkpoint choice.
This is a fixed-budget adaptation study, not convergence-certified fine-tuning.

Calibration uses four separate checkpoints: Pythia160M and Pythia1B, seeds
101/102, general-text validation only. Confirmatory recovery uses the nine
models with seeds0/1/2; WikiText test and code/math article-disjoint role
partitions. General/code/math losses are token NLL, not generation accuracy.
Raw-gradient collection on held-out models can proceed while calibration is
running, but recovery quality and acceptance summaries must not be used to
alter the frozen algorithms or calibration policy.

Query budgets1/4/8; twelve directions comprise eight fitting candidates and
four held-out directions. Designed queries repeat W in all four unit probes
and repeat that cycle for q8; their held-out directions are independent
Gaussian queries. Gaussian and natural families use their independent last
four directions. Noise levels0/1e-6/1e-4/1e-2 affect both product and each
response at fixed relative Frobenius norm. Reuse noise directions across
levels; design probes from the observed product. Estimated-subspace fits
retain orthogonal product and response residuals and score error in the
original basis space.

Compare spectral, simplex-projected spectral, and constrained joint fitting
with3 starts and80 function evaluations per start. Joint fitting is explicitly
not run if the observed per-layer query rank is insufficient; that case is
rejected and retained, not reported as successful recovery. The diagnostics
are residual-only, condition-only, and combined residual/conditioning/subspace/
competitive-start disagreement, including a1e-10 numerical floor. Thresholds
maximize accepted calibration count subject to at most5% bad outputs
(error>0.10) and at least20 accepted cases, keeping all score ties together.
No feasible threshold means all reject. Good error<=0.05; intermediate cases
remain in the denominators. These are empirical thresholds, not global
certificates. Sufficient query rank and candidate simplex feasibility are
common acceptance gates. Calibration cases are correlated within checkpoint.

Decision units: Pythia1B, Qwen3-4B and SmolLM2-1.7B, each seeds0/1/2. Use K4
hard groups with identical equal sizes selected by native router, joint-fit
router (q4, noise1e-4 and1e-2), effective-parameter balanced clustering, and
fixed-seed random balanced routing. Refit every action from the same native
effective tensors. Train only the K bases for64 matched recovery steps,
batch4/length128, lr1e-4, no weight decay, fresh AdamW states and identical
batches. Compare immediate and post-recovery NLL on32 fixed blocks per domain.
A rejection policy retains the native action; its result can be read from the
already executed native/estimated actions without extra tuning. No claimed
wall-clock compression benefit: frozen-backbone storage is unchanged.

An explicit source/config/data freeze is written to LLM_STUDY_FREEZE.json
before any calibration/confirmatory run. Corrections require an amendment,
retained old evidence and renewed checks; they may not selectively discard
unfavorable outcomes. Nine dedicated tests check embedding, chain rule,
response law, article separation, noisy compression, full-space error,
query insufficiency, permutation-invariant balanced actions, and calibration
tie/failure accounting.
