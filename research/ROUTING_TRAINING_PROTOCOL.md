# Constructed crossing: actual autograd training protocol

Fixed before this training experiment, 2026-09-25. The existing analytical example and ODE observations were already known; this is a targeted reproduction, not a discovery or a natural-training prevalence study.

Question: Does forward discrete SGD, using automatic differentiation of a fixed squared regression loss, enter strict disagreement from strict agreement in the constructed interior full-rank example?

Model: six tasks, nine input coordinates, three shared bases. On the nine canonical input vectors, task i predicts the entries of softmax(Z_i) B. The fixed target matrix is the target in `boundary_example`; loss is one half of the SUM of the 54 squared errors, not their mean. Both logits and bases are trained. All computation is local CPU, appropriate for this small model.

Initialization: compute the analytical gradient-flow velocity at the boundary, then Z_init=log(A_boundary)-0.1*Zdot_boundary and B_init=B_boundary-0.1*Bdot_boundary. This is an explicit tangent displacement, not backward ODE integration. Target remains fixed. Save exact initial arrays and targets for replay.

Primary runs: vanilla PyTorch SGD, no momentum or weight decay, float64 and float32; step sizes .02, .01, .005, .0025; duration .4 (20,40,80,160 updates). No selection or adjustment of step sizes after seeing results.

Every step: record loss; coefficient-assignment winner and runner-up among all 90 labeled balanced assignments; global effective-weight clustering winner and runner-up among all 15 unlabeled balanced partitions; p-minus-q score; minimum route coefficient and smallest singular values of A and B. Thus no Lloyd local optimum confound. Save initial/final factor arrays and factors at first qualifying disagreement.

Strict decision margin threshold: 1e-8 for both coefficient score and clustering SSE. A qualifying crossing must start with strict agreeing decisions and reach strict disagreeing decisions with the weight partition still equal to its initial value. Sustained crossing additionally requires five consecutive recorded post-update states satisfying this condition. Report both outcomes; retain failures and initially ineligible runs. Loss monotonicity tolerance: 1e-12 for float64 and 1e-7 for float32. Positivity/rank are recorded, not inferred from final state alone.

Robustness: float64 SGD, step .005, duration .4; independent Gaussian perturbations to each initial logit and basis entry, standard deviations .0001, .001, .01; seeds 0 through 19 at EACH scale. Fixed target unchanged. All 60 runs retained, including initial disagreements, and denominator reported explicitly. These scales describe only this chosen neighborhood, not a population of natural models.

Control: identical initial model, target replaced by its own initial effective weights, float64 SGD .005 for 80 steps. Expected stationary zero-gradient training and unchanged decisions. This checks that the measurement procedure itself does not produce a crossing.

Continuous reference: high-accuracy solve_ivp FORWARD from the same explicitly displaced initial state for duration .4, with rtol=1e-11 and atol=1e-13. Compare all float64 SGD endpoints to this reference; report convergence errors at all four step sizes. This ODE is a reference, not the SGD update implementation.

Validation: compare autograd parameter gradients against the separately coded chain rule at initialization; verify actual movement of both factors; record package versions, protocol/source hashes and all run traces. No LLM fine-tuning or downstream utility claim is part of this experiment.

The scientific-critical-thinking skill informed explicit controls, retention of unfavorable runs, and separation of construction from typicality. Local instruction source: `/root/.codex/skills/scientific-critical-thinking/SKILL.md`.
