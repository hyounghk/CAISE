#!/bin/
model_dir=best_dir

dataset=multidial
gpu=$1

log_dir=$dataset
mkdir -p $model_dir/$log_dir

CUDA_VISIBLE_DEVICES=$gpu stdbuf -i0 -o0 -e0 python src/main.py --output $model_dir/$log_dir \
    --maxInput 40  --worker 4 --train speaker --dataset $dataset \
    --batchSize 25 --hidDim 512 --dropout 0.5 \
    --seed 2020 \
    --optim adam --lr 1e-4 --epochs 500
