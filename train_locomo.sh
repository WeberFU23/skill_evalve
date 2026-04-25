#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# --disable-flash-attn \
# --reward-metric llm_judge \
# --locomo-train-query-sampling-ratio 0.2 \
# --resume-new-wandb-run
python main.py \
    --dataset locomo \
    --data-file "./data/locomo10.json" \
    --model "glm-5.1" \
    --designer-model "glm-5.1" \
    --api \
    --api-base "https://open.bigmodel.cn/api/paas/v4/" \
    --api-key "847d6fee3e8b451ab818be188da47445.FGVwGlyUhpq5OFMe" "3e11a733dc044dc58910ed6fdeb740d3.aT7i5zp2LSOIVOFu" \
    --retriever contriever \
    --state-encoder sentence-transformers/all-MiniLM-L6-v2 \
    --op-encoder sentence-transformers/all-MiniLM-L6-v2 \
    --disable-flash-attn \
    --designer-freq 1 \
    --inner-epochs 1 \
    --outer-epochs 1 \
    --batch-size 1 \
    --encode-batch-size 2 \
    --session-mode full-session \
    --ppo-epochs 2 \
    --action-top-k 1 \
    --mem-top-k 3 \
    --mem-top-k-eval 3 \
    --reward-metric f1 \
    --device cuda \
    --enable-designer \
    --wandb-run-name locomo-train-lite \
    --save-dir ./checkpoints/locomo_with_designer_lite \
    --out-file ./results/locomo_with_designer_lite.json
