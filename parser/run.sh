#!/bin/bash
# 用法: [EVALS="sft/G_dev.jsonl sft/RD_dev.jsonl"] bash run.sh <gpu> <name> <fmt G|P> <train 参数...>
# 训练 → 评测 (默认 sft/<fmt>_dev.jsonl; 名字含 R 的再做换根投票)
cd "$(dirname "$0")"; GPU=$1; NAME=$2; FMT=$3; shift 3
export CUDA_VISIBLE_DEVICES=$GPU PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=~/t2g/env/bin/python; mkdir -p runs/$NAME; : > runs/$NAME/stdout.log
$PY train_lora.py --name $NAME "$@" >> runs/$NAME/stdout.log 2>&1 || { echo FAILED_TRAIN >> runs/$NAME/stdout.log; exit 1; }
for E in ${EVALS:-sft/${FMT}_dev.jsonl}; do $PY infer_eval.py --name $NAME --fmt $FMT --data $E >> runs/$NAME/stdout.log 2>&1; done
case $NAME in *GR*) $PY infer_eval.py --name $NAME --fmt $FMT --data sft/${FMT}_dev.jsonl --vote >> runs/$NAME/stdout.log 2>&1;; esac
echo RUN_DONE >> runs/$NAME/stdout.log
