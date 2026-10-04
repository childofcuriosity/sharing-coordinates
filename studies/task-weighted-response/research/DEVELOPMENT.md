# Development record

Before any real-model selection or evaluation, five synthetic tests checked the empirical objective, isotropic limit, batch correction, activation distortion and a task-specific partition change. The first attempt assumed a generic random draw would change the winner; it did not. The first log is retained in `logs/tests-development-first.log`. Replaced that unjustified expectation with an explicit low-contrast two-basis router and opposite task gradients. This is a deterministic constructed identity test, not an observed natural-model result. All five tests then passed. No hyperparameter was tuned on model outcomes.

The design review uses scientific-critical-thinking and cites its verified Scientific Agent Skills source in `research/DESIGN_REVIEW.md`. Current paper and raw results are not modified.
