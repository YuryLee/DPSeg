#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
mkdir -p reports/v2/legacy
for mode in E0 E2 E3; do
  for split in val test; do
    timeout 600 .venv/bin/python tools/run.py evaluate \
      --data data/graspnet_expanded_v2 --checkpoint "outputs/graspnet/${mode}/model.pt" \
      --split "$split" --device cuda:0 --threads 4 \
      --output "reports/v2/legacy/${mode}_${split}.json" \
      --extra-output "reports/v2/legacy/${mode}_${split}_extra.json" \
      > "reports/v2/legacy/${mode}_${split}.log" 2>&1
  done
done
