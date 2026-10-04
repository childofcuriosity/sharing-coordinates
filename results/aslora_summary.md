# ASLoRA three-seed first-merge audit

Scientific gate: canonical action audit, candidate coverage, and restoration all pass: **true**.
Generation-time source hashes embedded in all three artifacts: **true**.
Factorized float32 re-execution is diagnostic only and is not used as the equivalence gate.

| Seed | Candidates | Scope | Changed / 10 gauges | Unique actions | Identity raw-B action | Loss delta range | Accuracy delta range | F1 delta range |
|---:|:---|:---|---:|---:|:---|:---|:---|:---|
| 0 | adjacent | per_projection | 2/10 | 3 | `query:0-1|value:0-1` | [-0.019626, +0.000000] | [+0.000000, +0.009804] | [+0.000000, +0.005659] |
| 0 | adjacent | joint | 0/10 | 1 | `query:0-1|value:0-1` | [+0.000000, +0.000000] | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| 0 | all | per_projection | 4/10 | 4 | `query:0-1|value:0-1` | [-0.019626, +0.000000] | [+0.000000, +0.009804] | [-0.000767, +0.005659] |
| 0 | all | joint | 0/10 | 1 | `query:0-1|value:0-1` | [+0.000000, +0.000000] | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| 1 | adjacent | per_projection | 6/10 | 4 | `query:0-1|value:2-3` | [+0.000000, +0.008863] | [+0.000000, +0.004902] | [-0.000472, +0.003260] |
| 1 | adjacent | joint | 5/10 | 2 | `query:0-1|value:0-1` | [+0.000000, +0.007710] | [-0.004902, +0.000000] | [-0.005119, +0.000000] |
| 1 | all | per_projection | 7/10 | 5 | `query:0-2|value:0-2` | [-0.000568, +0.002724] | [-0.002451, +0.000000] | [-0.002082, +0.000000] |
| 1 | all | joint | 0/10 | 1 | `query:0-2|value:0-2` | [+0.000000, +0.000000] | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| 2 | adjacent | per_projection | 3/10 | 3 | `query:0-1|value:0-1` | [-0.053094, +0.000000] | [-0.002451, +0.002451] | [-0.002246, +0.000825] |
| 2 | adjacent | joint | 0/10 | 1 | `query:0-1|value:0-1` | [+0.000000, +0.000000] | [+0.000000, +0.000000] | [+0.000000, +0.000000] |
| 2 | all | per_projection | 3/10 | 3 | `query:0-1|value:0-1` | [-0.053094, +0.000000] | [-0.002451, +0.002451] | [-0.002246, +0.000825] |
| 2 | all | joint | 0/10 | 1 | `query:0-1|value:0-1` | [+0.000000, +0.000000] | [+0.000000, +0.000000] | [+0.000000, +0.000000] |

## Across seeds

| Candidates | Scope | Changed / 30 gauge instances | Seeds with change | Global loss delta | Global accuracy delta | Global F1 delta | Chain in all seeds |
|:---|:---|---:|---:|:---|:---|:---|:---|
| adjacent | per_projection | 11/30 | 3/3 | [-0.053094, +0.008863] | [-0.002451, +0.009804] | [-0.002246, +0.005659] | true |
| adjacent | joint | 5/30 | 1/3 | [+0.000000, +0.007710] | [-0.004902, +0.000000] | [-0.005119, +0.000000] | false |
| all | per_projection | 14/30 | 3/3 | [-0.053094, +0.002724] | [-0.002451, +0.009804] | [-0.002246, +0.005659] | true |
| all | joint | 0/30 | 0/3 | [+0.000000, +0.000000] | [+0.000000, +0.000000] | [+0.000000, +0.000000] | false |

## Baseline meanings

- **Raw-B rule:** ASLoRA's cumulative running-average `B` Euclidean rule; prevalidation and gauge dependent.
- **Current effective baseline:** current `B_i A` Frobenius distance; gauge invariant and prevalidation.
- **Current local activation baseline:** lower-layer RMS output disturbance on unlabeled validation inputs; gauge invariant, post hoc, and not network loss.
- **Validation oracle:** labeled post-hoc enumeration of query-only, value-only, and joint-same-pair families. Per-projection Cartesian composites are not exhaustively enumerated.

## Conclusion

Across all three seeds, in the primary per-projection scope and under both adjacent and all-pairs candidate interpretations, gauge-equivalent running factors select different raw-B actions and at least one changed action has a different canonical validation outcome from the identity-gauge action. The joint sensitivity result is not uniform across both modes. The generation-time source hashes embedded in all three artifacts match one another and the released source snapshot.

Generation-time source bundle SHA256: `d92bcfab481c6fa865ad0a7ef9a3c7543a18e1f4800be26e3e8cad10f7c510cd`; it matches the current released source snapshot.
