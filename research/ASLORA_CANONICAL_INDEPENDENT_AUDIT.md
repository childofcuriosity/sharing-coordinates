# Independent audit of the canonical ASLoRA records

Date: 2026-08-23 (Asia/Shanghai)

## Scope and independence

This audit treats `results/aslora_seed{0,1,2}/audit.json`, the adjacent
`provenance.json` files, and the three frozen factor snapshots as the primary
records.  The independent test
`tests/test_aslora_canonical_results_independent.py` does **not** import the
ASLoRA runner, training module, witness module, or production summarizer.  It
reconstructs the raw-factor distances, decisions, candidate families,
identity-relative consequences, and summary aggregates directly from JSON and
tensor bytes.  The locked generation sources were additionally read line by
line to audit action semantics; they were not modified.

## Recomputed headline results

For each seed and candidate mode, the identity action is the raw-running-`B`
action under the identity gauge.  Every other gauge's canonical action metrics
are differenced from that action in the same seed.

| Candidate mode | Changes by seed | Changes / 30 | Consequential seeds | Loss-difference range |
|---|---:|---:|---:|---:|
| adjacent, per projection | 2, 6, 3 | 11/30 | 3/3 | [-0.0530944957452662, +0.00886300556799946] |
| all pairs, per projection | 4, 7, 3 | 14/30 | 3/3 | [-0.0530944957452662, +0.002723619049670667] |
| adjacent, joint sensitivity | 0, 5, 0 | 5/30 | 1/3 | [0, +0.007709831583733673] |
| all pairs, joint sensitivity | 0, 0, 0 | 0/30 | 0/3 | [0, 0] |

The primary adjacent chain has explicit witnesses in every seed:

- seed 0, `dense_k30`: identity `query:0-1|value:0-1` changes to
  `query:2-3|value:0-1`; loss difference -0.019626289898273974;
- seed 1, `dense_k30`: identity `query:0-1|value:2-3` changes to
  `query:2-3|value:10-11`; loss difference +0.00886300556799946;
- seed 2, `dense_k8`: identity `query:0-1|value:0-1` changes to
  `query:0-1|value:9-10`; loss difference -0.0530944957452662.

For all-pairs candidates, seed 0 and seed 2 have the same displayed witnesses;
seed 1 has several, including `diagonal_k2`, which changes
`query:0-2|value:0-2` to `query:0-2|value:10-11` with loss difference
+0.002723619049670667.  Thus “3/3” means at least one changed action with a
nonzero canonical validation consequence in each seed.  It is not a frequency
estimate, a claim that every changed action is worse, or a test-regret result.

The independently recomputed global accuracy range for both primary modes is
[-0.002451002597808838, +0.0098038911819458], and the F1 range is
[-0.002246155282257134, +0.005659439629292384].  These values agree with
`results/aslora_summary.json` before display rounding.

## Ranking and equivalence checks

The test reconstructs every raw Euclidean distance from the float32 snapshot
bytes after promotion to float64 and multiplication by each serialized gauge
matrix: 3 seeds x 10 gauges x 2 target projections x 66 all-layer pairs =
3,960 target-specific distances.  It also reconstructs the joint
root-sum-of-squares tables.  Every serialized value and every lexicographic
argmin agrees.  The smallest first-versus-second distance margin over every
reported target/joint ranking and candidate mode is
0.0004627883962863566 (seed 1, `dense_k4`, all-pairs value projection), so none
of the released decisions is a float64 tie or a last-bit rounding switch.

All serialized gauges are 8 x 8 and invertible.  Their directly recomputed
condition numbers agree with the records; identity is exact, the condition-one
control is orthogonal to numerical precision, and each nonorthogonal gauge is
within its declared 2, 4, 8, or 30 limit.  Directly evaluating
`Q @ solve(Q, A)` for all 60 target-specific matrices recovers every entry of
`A` within the test tolerance.  This independently checks the rank-space
identity behind `(B Q)(Q^{-1} A) = B A`; the artifacts' fuller product and
declared invariant-distance gates also report pass for every gauge.

## Candidate coverage, identity control, and action execution

The exhaustive declared base family has 198 actions: all 66 query-only, 66
value-only, and 66 joint-same-pair actions.  Adding every distinct composite
selected by a gauge or declared baseline yields exactly 204, 208, and 200
actions for seeds 0, 1, and 2.  Reconstructing that union produces no missing
or unexpected key.  Each candidate's action dictionary, metrics, and
delta-from-unmerged arithmetic is internally exact.  The labeled post-hoc
oracle is the true loss minimum within each of its three declared families;
there is no 66 x 66 per-projection Cartesian oracle, and the paper now says so.

For both candidate modes and scopes, the standalone raw-`B` baseline record is
exactly the identity-gauge decision.  The union of every raw-`B` action selected
across gauges/modes/scopes is exactly the union covered by the deterministic
repeat gate.  Each seed contains two complete passes in total (the cached first
evaluation and one required repeat); every recorded baseline, loss, logit,
prediction, accuracy, and F1 repeat difference is exactly zero.  Before/after
trainable digests and structural parameter identities agree, and the records
state that no merge was committed.

The attested source implements the claimed directed action as follows:

- `src/aslora_witness.py` defines lower-uses-upper by replacing the lower
  layer's current `B` with the upper layer's current `B`;
- `src/aslora_training.py` applies that tie in a context manager separately to
  the selected query/value targets;
- `experiments/run_aslora_mrpc.py` evaluates the selected layer-index action in
  the unchanged canonical model on the complete 408-example validation split,
  then checks restoration.

Accordingly the measured network estimand is the immediate complete-validation
consequence of a temporary first tie.  It is not the raw running-average
distance, the local activation RMS diagnostic, a committed merge tree, or
full-training regret.  The paper makes these distinctions explicitly.

## Source and input byte binding

The three released audits and provenance records agree on, and the current
workspace bytes independently reproduce, these generation hashes:

- runner: `ab09259c092a9a3fab49e5aceb06aad7f56eb20b923b6edb3965bd8c940576d9`;
- training: `69818d7c8eb7d20ec3c123f49b0c42ec3ffeb66b835bb79429083bcb4a646664`;
- witness: `d57734071e18ab0f4384772a048faa0264813411069ceabe4fca82906d66bc64`.

For every provenance inventory, the test canonicalizes the complete entry list
and independently recovers the recorded inventory SHA256.  This is stronger
than merely checking that the stored digest has 64 hexadecimal characters.
All three seeds bind the same external inputs:

- RoBERTa directory: 40 files, 2,815,193,435 total bytes, inventory
  `8eba55c3adf87b2a677b3a55677a21648d9cc6548ae8b83f70e36a32aeb2cb9f`;
- MRPC train parquet: 649,281 bytes, file SHA256
  `61fd41301e0e244b0420c4350a170c8e7cf64740335fc875a4af2d79af0df0af`;
- MRPC validation parquet: 75,678 bytes, file SHA256
  `33c007dbf5bfa8463d87a13e6226df8c0fcf2596c2cd39d0f3bb79754e00f50f`.

A read-only `ssh Seeta` audit independently hashed the two actual parquet files
and all 40 actual files in the recorded remote RoBERTa directory; every size
and file digest matches the released manifests.  The three complete initial
attached-model state hashes are valid and distinct across seeds.  They bind the
generation state but cannot be recomputed from the compact released factor
snapshots alone.

## Protocol and selective-reporting audit

The raw records agree on rank 8, alpha 16, batch 16, 30 planned epochs,
deterministic CUDA algorithms, disabled TF32, first decision at one-indexed
optimizer step 560, 560 after-update running-average observations, 3,668 train
examples, and a 408-example/26-batch validation pass.  The bank is exactly the
ten declared gauges, has no external witness, and is recorded as constructed
before validation.  Raw action selection uses only the frozen running factors.

The negative joint result is retained, as are all ten gauges in all three
seeds.  The canonical changes (11 and 14) are lower than the superseded changes
(12 and 16), which argues against choosing the rerun to enlarge the headline.
The amendment chronology in `research/ASLORA_EMPIRICAL_PROTOCOL.md` now states
that deterministic/repeat/input-binding source was locked before the
replacement outputs.  On Seeta, the three locked source mtimes precede the seed
0 audit, and seed audits were written sequentially at 01:49, 01:53, and 01:55
local time.  Timestamps and embedded hashes support the chronology but cannot,
by themselves, prove what investigators knew.

The remote archive still contains the superseded attested audits.  A read-only
recalculation gives changes (3,5,4) = 12/30 and (5,7,4) = 16/30, candidate
counts 202/207/206, and confirms that the old records lack deterministic and
repeat fields.  Their audit-file SHA256 values are, by seed,
`3062ac616447bec9b0f51747e5b6015c824497d0b807993549dd41a49a663049`,
`e7938d9f0bd08bf8edf0555a515682a6eb7d10bfd6f932ac69d7017d87832877`,
and `d395e157752e2f60613e2323bdea3e466110e207ebb48fccafaa99ae24fa03c7`.
They are not canonical and are not pooled with the deterministic results.

## Remaining evidentiary boundary

The released 1.23 MB factor snapshot suffices to replay gauges, distances, and
actions, but it omits the frozen backbone/classifier and optimizer checkpoint.
Therefore a reader cannot re-execute the 408-example network consequences from
that compact snapshot alone.  The action results are supported by source/input
attestation, complete candidate records, exact in-run repeats, and restoration
checks; independent end-to-end metric regeneration still requires the external
byte-matched RoBERTa/MRPC inputs and retraining, or retention of the roughly
full-model pre-first-merge checkpoint.  The reproducibility document discloses
this boundary and does not promise cross-hardware bit identity.

## Verdict

No counterexample was found to the canonical numerical claims.  The exact raw
records support 11/30, 14/30, both primary 3/3 chains, all printed metric
ranges, candidate coverage, identity controls, and the stated first-action
estimand.  The correct claim is the scoped existence of coordinate-dependent
first decisions with different canonical validation consequences in this
reimplementation.  The records do not support prevalence, official-ASLoRA
reproduction, identity-action optimality, full-training regret, or a generally
superior invariant replacement rule.

