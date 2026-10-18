#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0
DATA=data/graspnet_expanded_v2
[[ -f "$DATA/manifest.json" ]]
mkdir -p outputs/graspnet_v2 reports/v2
for mode in E0 E2 E3; do
  timeout 1800 .venv/bin/python tools/run.py train \
    --data "$DATA" --config "configs/GraspNetV2_${mode}.json" \
    --output "outputs/graspnet_v2/${mode}" --steps 800 --batch-size 2 \
    --lr 0.0002 --backbone-lr-factor 0.1 --warmup-steps 100 --cosine-decay \
    --augment --crop-probability 0.35 \
    --init-rgb pretrained/maskrcnn_resnet50_fpn_v2_coco_mirror_state_dict.pt \
    --device cuda:0 --threads 4 --cache-data --cuda-memory-fraction 0.30 \
    > "reports/v2/train_${mode}.log" 2>&1
done
for mode in E0 E2 E3; do
  for split in val test; do
    suffix=""
    [[ "$split" == test ]] && suffix="_test"
    timeout 600 .venv/bin/python tools/run.py evaluate --data "$DATA" \
      --checkpoint "outputs/graspnet_v2/${mode}/model.pt" --split "$split" \
      --device cuda:0 --threads 4 \
      --output "outputs/graspnet_v2/${mode}/evaluation${suffix}.json" \
      --extra-output "outputs/graspnet_v2/${mode}/extra_${split}.json" \
      > "reports/v2/evaluate_${mode}_${split}.log" 2>&1
  done
done
