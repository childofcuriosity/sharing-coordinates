#!/usr/bin/env bash
set -euo pipefail

# Staged entry point for the released artifact.  The default mode is the
# non-training verification path; the expensive GPU modes are explicit.
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root_dir"

device="${DEVICE:-cuda}"
python_cmd="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_all.sh MODE

Modes:
  verify   Rebuild all summaries/tables in a temporary directory, compare
           them with the release, run tests, and check MANIFEST.sha256.
  derive   Rewrite summaries, generated tables, and legacy figures from the
           checked-in raw records.  Does not train models.
  byte-lm  Retrain the three byte-LM checkpoints and rerun the Gaussian
           response, structured-tomography, and partition audits, then derive
           their summaries and tables.  Uses DEVICE.
  aslora   Rerun the three ASLoRA/MRPC first-merge audits, then derive their
           summary and table.  Requires ASLORA_TRAIN_PARQUET,
           ASLORA_VALIDATION_PARQUET, and ASLORA_MODEL.  Uses DEVICE.
  legacy   Regenerate all older synthetic/diagnostic raw records and their
           aggregate, figures, and tables.  Uses DEVICE.
  release  Run derive and tests, compile paper/main.pdf, then rewrite and
           verify MANIFEST.sha256.  Does not retrain raw GPU records.

Optional ASLoRA variables:
  ASLORA_CACHE_DIR         cache directory (default: .cache/aslora)
  ASLORA_MODEL_REVISION    recorded model revision (default: main)
  ASLORA_LOCAL_FILES_ONLY  prohibit model/tokenizer downloads (default: 1)
  ASLORA_KEEP_CHECKPOINT   set to 1 to retain the large optimizer checkpoint
  PYTHON                   Python executable (default: python)
EOF
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf 'required file is missing: %s\n' "$1" >&2
    exit 1
  fi
}

compare_file() {
  local released="$1"
  local rebuilt="$2"
  if ! cmp -s "$released" "$rebuilt"; then
    printf 'derived artifact differs from release: %s\n' "$released" >&2
    exit 1
  fi
}

summarize_primary() {
  local output_dir="$1"
  mkdir -p "$output_dir"
  "$python_cmd" -m experiments.summarize_response_audit \
    results/primary_remote/response_audit_seed0.json \
    results/primary_remote/response_audit_seed1.json \
    results/primary_remote/response_audit_seed2.json \
    --json-output "$output_dir/response_summary.json" \
    --csv-output "$output_dir/response_table.csv" \
    --markdown-output "$output_dir/response_table.md"
  "$python_cmd" -m experiments.summarize_response_tomography \
    results/response_tomography_seed0.json \
    results/response_tomography_seed1.json \
    results/response_tomography_seed2.json \
    --output "$output_dir/response_tomography_summary.json"
  "$python_cmd" -m experiments.summarize_aslora_audit \
    results/aslora_seed0/audit.json \
    results/aslora_seed1/audit.json \
    results/aslora_seed2/audit.json \
    --json-output "$output_dir/aslora_summary.json" \
    --csv-output "$output_dir/aslora_summary.csv" \
    --markdown-output "$output_dir/aslora_summary.md"
  "$python_cmd" -m experiments.summarize_router_decisions \
    results/primary_remote/router_decisions_seed0.json \
    results/primary_remote/router_decisions_seed1.json \
    results/primary_remote/router_decisions_seed2.json \
    --json-output "$output_dir/router_decisions_summary.json" \
    --csv-output "$output_dir/router_decisions_table.csv" \
    --markdown-output "$output_dir/router_decisions_table.md"
}

generate_primary_tables() {
  local summary_dir="$1"
  local output_dir="$2"
  "$python_cmd" -m experiments.generate_paper_audit_tables \
    --response "$summary_dir/response_summary.json" \
    --aslora "$summary_dir/aslora_summary.json" \
    --router-decisions "$summary_dir/router_decisions_summary.json" \
    --output-dir "$output_dir"
  "$python_cmd" -m experiments.generate_response_tomography_latex \
    --input "$summary_dir/response_tomography_summary.json" \
    --output "$output_dir/response_tomography_numbers.tex"
}

derive_all() {
  "$python_cmd" scripts/verify_source_attestation.py
  "$python_cmd" -m experiments.summarize_autograd_tomography
  "$python_cmd" -m experiments.summarize_factor_recovery
  "$python_cmd" -m experiments.summarize_enhancement
  "$python_cmd" -m experiments.summarize_trusted_recovery
  "$python_cmd" -m experiments.generate_dynamics_paper
  summarize_primary results
  generate_primary_tables results paper/generated
  MPLCONFIGDIR="${MPLCONFIGDIR:-$root_dir/.cache/matplotlib}" \
    "$python_cmd" -m experiments.analyze_results
}

run_tests() (
  local pytest_dir="${1:-}"
  if [[ -z "$pytest_dir" ]]; then
    pytest_dir="$(mktemp -d)"
    if [[ -z "$pytest_dir" || ! -d "$pytest_dir" ]]; then
      printf 'could not create pytest directory\n' >&2
      exit 1
    fi
    trap 'rm -rf -- "$pytest_dir"' EXIT
  fi
  MPLCONFIGDIR="$pytest_dir/matplotlib" PYTHONDONTWRITEBYTECODE=1 \
    "$python_cmd" -m pytest -p no:cacheprovider --basetemp "$pytest_dir"
)

verify_release() (
  local verify_dir
  export PYTHONDONTWRITEBYTECODE=1
  verify_dir="$(mktemp -d)"
  if [[ -z "$verify_dir" || ! -d "$verify_dir" ]]; then
    printf 'could not create verification directory\n' >&2
    exit 1
  fi
  trap 'rm -rf -- "$verify_dir"' EXIT

  "$python_cmd" scripts/verify_source_attestation.py
  run_tests "$verify_dir/pytest"
  "$python_cmd" -m experiments.summarize_autograd_tomography --check
  "$python_cmd" -m experiments.summarize_factor_recovery --check
  "$python_cmd" -m experiments.summarize_enhancement --check
  "$python_cmd" -m experiments.summarize_trusted_recovery --check
  "$python_cmd" -m experiments.compare_trusted_replay --check
  "$python_cmd" -m scripts.verify_trusted_records
  if [[ -f results/llm/summary.json ]]; then
    "$python_cmd" -m experiments.summarize_llm_study --check
    MPLCONFIGDIR="$verify_dir/matplotlib" "$python_cmd" -m experiments.generate_llm_paper --check
  fi
  "$python_cmd" -m experiments.generate_dynamics_paper --check
  "$python_cmd" -m scripts.check_decision_certificates
  "$python_cmd" -m scripts.verify_publication_records
  summarize_primary "$verify_dir/results"
  generate_primary_tables "$verify_dir/results" "$verify_dir/generated"
  MPLCONFIGDIR="$verify_dir/matplotlib" "$python_cmd" -m experiments.analyze_results \
    --results-dir results \
    --figures-dir "$verify_dir/figures" \
    --generated-dir "$verify_dir/legacy-generated" \
    --summary "$verify_dir/results/summary.json"

  local name
  for name in \
    response_summary.json response_table.csv response_table.md \
    response_tomography_summary.json \
    aslora_summary.json aslora_summary.csv aslora_summary.md \
    router_decisions_summary.json router_decisions_table.csv \
    router_decisions_table.md summary.json; do
    compare_file "results/$name" "$verify_dir/results/$name"
  done
  for name in \
    response_table.tex response_tomography_numbers.tex \
    aslora_table.tex aslora_metric_detail_table.tex \
    aslora_baseline_table.tex router_decision_table.tex \
    router_decision_detail_table.tex; do
    compare_file "paper/generated/$name" "$verify_dir/generated/$name"
  done
  "$python_cmd" scripts/write_manifest.py --check
  printf 'release verification passed\n'
)

run_byte_lm() {
  require_file data/wikitext-2/train.txt
  require_file data/wikitext-2/valid.txt
  require_file data/wikitext-2/test.txt
  mkdir -p results/checkpoints results/primary_remote

  local seed checkpoint
  for seed in 0 1 2; do
    checkpoint="results/checkpoints/language_seed${seed}.pt"
    "$python_cmd" -m experiments.run_language \
      --output "results/primary_remote/language_seed${seed}.json" \
      --checkpoint-save "$checkpoint" \
      --router-init pattern_cycle \
      --steps 1500 \
      --seed "$seed" \
      --device "$device"
    "$python_cmd" -m experiments.run_response_audit \
      --checkpoint "$checkpoint" \
      --output "results/primary_remote/response_audit_seed${seed}.json" \
      --probe-seed "$((740000 + seed))" \
      --autograd-seed 750000 \
      --device "$device"
    "$python_cmd" -m experiments.run_response_tomography \
      --checkpoint "$checkpoint" \
      --output "results/response_tomography_seed${seed}.json" \
      --device "$device"
    "$python_cmd" -m experiments.run_router_decisions \
      --checkpoint "$checkpoint" \
      --output "results/primary_remote/router_decisions_seed${seed}.json" \
      --corpus data/wikitext-2/test.txt \
      --train-corpus data/wikitext-2/train.txt \
      --batch-size 8 \
      --sequence-length 128 \
      --eval-batches 256 \
      --batch-seed "$((610000 + seed))" \
      --search-seed "$((620000 + seed))" \
      --kmeans-seed "$((630000 + seed))" \
      --device "$device"
  done
  "$python_cmd" scripts/verify_source_attestation.py
  summarize_primary results
  generate_primary_tables results paper/generated
}

run_aslora() {
  : "${ASLORA_TRAIN_PARQUET:?set ASLORA_TRAIN_PARQUET to the MRPC training parquet}"
  : "${ASLORA_VALIDATION_PARQUET:?set ASLORA_VALIDATION_PARQUET to the MRPC validation parquet}"
  : "${ASLORA_MODEL:?set ASLORA_MODEL to a local roberta-base directory}"
  require_file "$ASLORA_TRAIN_PARQUET"
  require_file "$ASLORA_VALIDATION_PARQUET"
  if [[ ! -d "$ASLORA_MODEL" ]]; then
    printf 'ASLORA_MODEL must be a local roberta-base directory so its files can be byte-inventoried: %s\n' "$ASLORA_MODEL" >&2
    exit 1
  fi
  case "${ASLORA_MODEL%/}" in
    *roberta-base) ;;
    *)
      printf 'ASLORA_MODEL must end in roberta-base for the strict protocol: %s\n' "$ASLORA_MODEL" >&2
      exit 1
      ;;
  esac
  case "${ASLORA_TRAIN_PARQUET//\\//}" in
    */glue/mrpc/train-00000-of-00001.parquet) ;;
    *)
      printf 'ASLORA_TRAIN_PARQUET has the wrong strict-protocol path: %s\n' "$ASLORA_TRAIN_PARQUET" >&2
      exit 1
      ;;
  esac
  case "${ASLORA_VALIDATION_PARQUET//\\//}" in
    */glue/mrpc/validation-00000-of-00001.parquet) ;;
    *)
      printf 'ASLORA_VALIDATION_PARQUET has the wrong strict-protocol path: %s\n' "$ASLORA_VALIDATION_PARQUET" >&2
      exit 1
      ;;
  esac

  local cache_dir="${ASLORA_CACHE_DIR:-.cache/aslora}"
  local local_only=()
  if [[ "${ASLORA_LOCAL_FILES_ONLY:-1}" == "1" ]]; then
    local_only=(--local-files-only)
  fi
  local seed output_dir
  for seed in 0 1 2; do
    output_dir="results/aslora_seed${seed}"
    "$python_cmd" -m experiments.run_aslora_mrpc \
      --output-dir "$output_dir" \
      --cache-dir "$cache_dir" \
      --model-name "$ASLORA_MODEL" \
      --model-revision "${ASLORA_MODEL_REVISION:-main}" \
      --train-parquet "$ASLORA_TRAIN_PARQUET" \
      --validation-parquet "$ASLORA_VALIDATION_PARQUET" \
      --seed "$seed" \
      --device "$device" \
      "${local_only[@]}"
    if [[ "${ASLORA_KEEP_CHECKPOINT:-0}" != "1" ]]; then
      rm -f -- "$output_dir/pre_first_merge_checkpoint.pt"
    fi
  done
  "$python_cmd" scripts/verify_source_attestation.py
  summarize_primary results
  generate_primary_tables results paper/generated
}

run_legacy() {
  require_file data/wikitext-2/train.txt
  require_file data/wikitext-2/valid.txt
  require_file data/wikitext-2/test.txt
  "$python_cmd" -m experiments.run_factorization \
    --sweep --output results/factorization_full.json --device "$device"
  "$python_cmd" -m experiments.run_teacher_student \
    --architecture mlp --sweep --sweep-seeds 5 \
    --output results/teacher_student_mlp.json --device "$device"
  "$python_cmd" -m experiments.run_teacher_student \
    --architecture transformer --sweep --sweep-seeds 5 \
    --output results/teacher_student_transformer.json --device "$device"
  "$python_cmd" -m experiments.run_teacher_baselines \
    --inputs results/teacher_student_mlp.json \
      results/teacher_student_transformer.json \
    --output results/teacher_baselines.json --device "$device"
  "$python_cmd" -m experiments.run_probes \
    --architecture mlp --sweep --sweep-seeds 10 \
    --output results/probes_mlp.json --device "$device"
  "$python_cmd" -m experiments.run_probes \
    --architecture transformer --sweep --sweep-seeds 10 \
    --output results/probes_transformer.json --device "$device"
  "$python_cmd" -m experiments.run_gauge \
    --seeds 20 --saturation --output results/gauge_full.json
  "$python_cmd" -m experiments.run_language \
    --sweep --sweep-seeds 3 --steps 1500 \
    --output results/language_full.json --device "$device"
  MPLCONFIGDIR="${MPLCONFIGDIR:-$root_dir/.cache/matplotlib}" \
    "$python_cmd" -m experiments.analyze_results
}

build_paper() {
  if command -v latexmk >/dev/null 2>&1; then
    (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex)
    return
  fi
  if ! command -v pdflatex >/dev/null 2>&1 || ! command -v bibtex >/dev/null 2>&1; then
    printf 'release mode requires latexmk, or both pdflatex and bibtex\n' >&2
    exit 1
  fi
  (
    cd paper
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    bibtex main
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
    pdflatex -interaction=nonstopmode -halt-on-error main.tex
  )
}

mode="${1:-verify}"
case "$mode" in
  verify)
    verify_release
    ;;
  derive)
    derive_all
    ;;
  byte-lm)
    run_byte_lm
    ;;
  aslora)
    run_aslora
    ;;
  legacy)
    run_legacy
    ;;
  release)
    derive_all
    run_tests
    build_paper
    "$python_cmd" scripts/write_manifest.py
    "$python_cmd" scripts/write_manifest.py --check
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    printf 'unknown mode: %s\n' "$mode" >&2
    usage >&2
    exit 2
    ;;
esac
