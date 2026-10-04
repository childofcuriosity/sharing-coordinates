#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_cmd="${PYTHON:-python}"
device="${DEVICE:-cuda:0}"
output="${1:?Usage: reproduce_enhancement.sh NEW_OUTPUT_DIRECTORY}"
if [[ -e "$output" ]]; then
  printf 'Refusing existing output directory: %s\n' "$output" >&2
  exit 1
fi
mkdir -p "$output"
for seed in 10 11 12; do
  "$python_cmd" -m experiments.train_enhancement --seed "$seed" --steps 6000 \
    --device "$device" --output "$output/seed$seed"
  "$python_cmd" -m experiments.run_enhanced_decisions \
    --checkpoint "$output/seed$seed/checkpoint.pt" --device "$device" \
    --output "$output/seed$seed/decisions.json"
  "$python_cmd" -m experiments.run_task_gradients \
    --checkpoint "$output/seed$seed/checkpoint.pt" --device "$device" \
    --output "$output/task_seed$seed.json"
done
for seed in 0 1 2; do
  "$python_cmd" -m experiments.run_task_gradients \
    --checkpoint "results/checkpoints/language_seed$seed.pt" --device "$device" \
    --output "$output/task_seed$seed.json"
done
"$python_cmd" -m experiments.run_joint_noise --output "$output/joint_noise.json"
printf 'Fresh runs written to %s. Compare scientific metrics; timing and hashes may differ.\n' "$output"
