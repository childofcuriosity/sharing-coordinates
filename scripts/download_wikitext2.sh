#!/usr/bin/env bash
set -euo pipefail

target_dir="${1:-data/wikitext-2}"
mkdir -p "$target_dir"
revision="acc295dc7b90714f1bf47f06004fc19a7fe235c4"
base="https://raw.githubusercontent.com/pytorch/examples/${revision}/word_language_model/data/wikitext-2"
curl -fL "$base/train.txt" -o "$target_dir/train.txt"
curl -fL "$base/valid.txt" -o "$target_dir/valid.txt"
curl -fL "$base/test.txt" -o "$target_dir/test.txt"
(
  cd "$target_dir"
  printf '%s  %s\n' \
    '9e9fa1ad55b1c2c95b08e37dd8e653f638fac2c6de904b79e813611eefbc985f' 'train.txt' \
    'f0737ed31fc1329026e95cb8b98e19c2a182c39c240ab909dc31abf2f8af58e8' 'valid.txt' \
    'd790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0' 'test.txt' \
    | sha256sum --check --strict
)
wc -c "$target_dir"/*.txt
