#!/usr/bin/env bash
# =============================================================================
# LOGO Training Launch Script — Qwen3.5-0.8B + DS Clinical Data
# =============================================================================
#
# Designed for 2×GPU (IDs 0,1) with DeepSpeed ZeRO-2.
# The 0.8B model is small enough to fit without CPU offload.
#
# Usage:
#   bash scripts/train_logo_qwen_ds.sh
#
# To override paths:
#   MODEL_PATH=... DATASET_PATH=... OUTPUT_DIR=... bash scripts/train_logo_qwen_ds.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ==========================  paths  ===========================================

MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-0.8B}"
MODEL_TYPE="${MODEL_TYPE:-qwen3.5}"

# Must point to a DatasetDict directory (saved via datasets.save_to_disk).
DATASET_PATH="${DATASET_PATH:-./data/DS_test/ds_logo_tokenized}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/qwen_ds_logo_$(date +%Y%m%d_%H%M%S)}"

# ==========================  GPU  =============================================

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ==========================  model config  ====================================

# FlashAttention-2 (REQUIRED)
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"

# Position / RoPE — Qwen3.5 natively supports 32K, we set 8K for DS data
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-8192}"
ROPE_TYPE="${ROPE_TYPE:-}"
ROPE_FACTOR="${ROPE_FACTOR:-}"
ROPE_THETA="${ROPE_THETA:-}"

# LoRA — smaller rank for 0.8B model
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-8}"

# ==========================  training hyperparams  ============================

NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:--1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
OPTIM="${OPTIM:-adamw_torch}"

# Batch — LOGO requires per_device_train_batch_size=1
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"

# Sequence lengths
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-4096}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-1024}"

# LOGO / SimPO loss
BETA="${BETA:-3.0}"
GAMMA_BETA_RATIO="${GAMMA_BETA_RATIO:-0.2}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
SFT_WEIGHT="${SFT_WEIGHT:-0.3}"

# ==========================  DeepSpeed  =======================================

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-training/config/zero2-minimal.json}"

# ==========================  logging / checkpointing  =========================

SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-50}"
EVAL_STEPS="${EVAL_STEPS:-50}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
REPORT_TO="${REPORT_TO:-none}"

# ==========================  precision  =======================================

BF16="${BF16:-True}"

# ==========================  misc  ============================================

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

# ==========================  build optional args  =============================

ROPE_TYPE_ARG=""
if [ -n "$ROPE_TYPE" ]; then
    ROPE_TYPE_ARG="--rope_type $ROPE_TYPE"
fi

ROPE_FACTOR_ARG=""
if [ -n "$ROPE_FACTOR" ]; then
    ROPE_FACTOR_ARG="--factor $ROPE_FACTOR"
fi

ROPE_THETA_ARG=""
if [ -n "$ROPE_THETA" ]; then
    ROPE_THETA_ARG="--rope_theta $ROPE_THETA"
fi

MAX_STEPS_ARG=""
if [ "$MAX_STEPS" -gt 0 ]; then
    MAX_STEPS_ARG="--max_steps $MAX_STEPS"
fi

echo "============================================================"
echo " LOGO Training — Qwen3.5-0.8B + DS Clinical Data"
echo "============================================================"
echo " Model:               $MODEL_PATH"
echo " Model type:          $MODEL_TYPE"
echo " Dataset:             $DATASET_PATH"
echo " Output:              $OUTPUT_DIR"
echo " GPUs:                $CUDA_VISIBLE_DEVICES"
echo " Attention:           $ATTN_IMPLEMENTATION"
echo " Max position embeds: $MAX_POSITION_EMBEDDINGS"
echo " Max seq length:      $MAX_SEQ_LENGTH"
echo " Max target length:   $MAX_TARGET_LENGTH"
echo " Epochs:              $NUM_EPOCHS"
echo " LR:                  $LEARNING_RATE"
echo " LR scheduler:        $LR_SCHEDULER_TYPE"
echo " Warmup steps:        $WARMUP_STEPS"
echo " Batch size / GPU:    $PER_DEVICE_BATCH_SIZE"
echo " Grad accum steps:    $GRADIENT_ACCUMULATION_STEPS"
echo " Effective batch:     $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2))"
echo " Beta:                $BETA"
echo " Gamma/beta ratio:    $GAMMA_BETA_RATIO"
echo " SFT weight:          $SFT_WEIGHT"
echo " LoRA r/alpha:        $LORA_R / $LORA_ALPHA"
echo " DeepSpeed config:    $DEEPSPEED_CONFIG"
echo " Seed:                $SEED"
echo "============================================================"

deepspeed --include "localhost:${CUDA_VISIBLE_DEVICES//,/,}" training/logo_train.py \
    --model_name_or_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --attn_implementation "$ATTN_IMPLEMENTATION" \
    --max_position_embeddings "$MAX_POSITION_EMBEDDINGS" \
    $ROPE_TYPE_ARG \
    $ROPE_FACTOR_ARG \
    $ROPE_THETA_ARG \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    $MAX_STEPS_ARG \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "$LR_SCHEDULER_TYPE" \
    --warmup_steps "$WARMUP_STEPS" \
    --weight_decay "$WEIGHT_DECAY" \
    --optim "$OPTIM" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --max_target_length "$MAX_TARGET_LENGTH" \
    --beta "$BETA" \
    --gamma_beta_ratio "$GAMMA_BETA_RATIO" \
    --loss_type "$LOSS_TYPE" \
    --label_smoothing "$LABEL_SMOOTHING" \
    --sft_weight "$SFT_WEIGHT" \
    --low_rank_training True \
    --disable_dropout True \
    --label_pad_token_id -100 \
    --seed "$SEED" \
    --save_steps "$SAVE_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy steps \
    --eval_strategy steps \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --load_best_model_at_end False \
    --gradient_checkpointing True \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --remove_unused_columns False \
    --report_to "$REPORT_TO" \
    --bf16 "$BF16" \
    --deepspeed "$DEEPSPEED_CONFIG"

echo ""
echo "============================================================"
echo " Training finished."
echo " Model saved to: $OUTPUT_DIR"
echo "============================================================"
