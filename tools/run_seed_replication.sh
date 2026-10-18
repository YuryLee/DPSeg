#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
DATA=data/graspnet_expanded_v2
mkdir -p reports/v3
for seed in 23 41; do
  for mode in E0 E2; do
    output="outputs/graspnet_seed${seed}/${mode}"
    timeout 1800 .venv/bin/python tools/run.py train \
      --data "$DATA" --config "configs/seed_replication/seed${seed}_${mode}.json" \
      --output "$output" --steps 800 --batch-size 2 \
      --lr 0.0002 --backbone-lr-factor 0.1 --warmup-steps 100 --cosine-decay \
      --augment --crop-probability 0.35 \
      --init-rgb pretrained/maskrcnn_resnet50_fpn_v2_coco_mirror_state_dict.pt \
      --device cuda:0 --threads 4 --cache-data --cuda-memory-fraction 0.30 \
      > "reports/v3/train_seed${seed}_${mode}.log" 2>&1
    for split in val test; do
      suffix=""
      [[ "$split" == test ]] && suffix="_test"
      timeout 600 .venv/bin/python tools/run.py evaluate --data "$DATA" \
        --checkpoint "$output/model.pt" --split "$split" --device cuda:0 --threads 4 \
        --output "$output/evaluation${suffix}.json" --extra-output "$output/extra_${split}.json" \
        > "reports/v3/evaluate_seed${seed}_${mode}_${split}.log" 2>&1
    done
  done
done
