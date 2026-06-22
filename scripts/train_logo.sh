#!/usr/bin/env bash
# =============================================================================
# LOGO Training Launch Script (complete, self-contained, production-ready)
# =============================================================================
#
# This script pins every hyperparameter required to reproduce LOGO training.
# It is designed for 8×A800-80G with DeepSpeed ZeRO-3 and FlashAttention-2.
#
# Usage:
#   bash scripts/train_logo.sh
#
# To override paths without editing the script:
#   MODEL_PATH=... DATASET_PATH=... OUTPUT_DIR=... bash scripts/train_logo.sh
#
# Prerequisites:
#   1. A DatasetDict saved via datasets.DatasetDict.save_to_disk() with
#      "train" / "test" splits and the 12 standard LOGO fields.
#   2. A HuggingFace-compatible instruct model (Llama-3.1-8B-Instruct or similar).
#   3. 8 GPUs with CUDA + FlashAttention-2 installed.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ==========================  paths  ===========================================

# --- model (Llama-3.1-8B-Instruct or your base model) ---
MODEL_PATH="${MODEL_PATH:-meta-llama/Llama-3.1-8B-Instruct}"
MODEL_TYPE="${MODEL_TYPE:-llama-3}"          # llama-2 / llama-3 / mistral

# --- dataset ---
# Must point to a DatasetDict directory (saved via datasets.save_to_disk).
DATASET_PATH="${DATASET_PATH:-./data/train_data}"

# --- output ---
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/logo_run_$(date +%Y%m%d_%H%M%S)}"

# ==========================  model config  ====================================

# --- FlashAttention-2 (REQUIRED for long-context efficiency) ---
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

# --- position / RoPE (64K target; use 81920 for 80K) ---
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-65536}"
ROPE_TYPE="${ROPE_TYPE:-}"                    # "yarn" / "dynamic" (set if needed)
ROPE_FACTOR="${ROPE_FACTOR:-}"                # required when rope_type is set
ROPE_THETA="${ROPE_THETA:-}"                  # optional, e.g. 500000.0

# --- LoRA ---
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-16}"

# ==========================  training hyperparams  ============================

# --- schedule ---
NUM_EPOCHS="${NUM_EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:--1}"                  # -1 = use epochs
LEARNING_RATE="${LEARNING_RATE:-5e-7}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WARMUP_STEPS="${WARMUP_STEPS:-120}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
OPTIM="${OPTIM:-paged_adamw_32bit}"

# --- batch ---
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"   # LOGO requires 1
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"

# --- sequence lengths ---
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-10000}"
MAX_TARGET_LENGTH="${MAX_TARGET_LENGTH:-2000}"

# --- LOGO / SimPO loss ---
BETA="${BETA:-2.0}"
GAMMA_BETA_RATIO="${GAMMA_BETA_RATIO:-0.25}"
LOSS_TYPE="${LOSS_TYPE:-sigmoid}"             # sigmoid or hinge
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.0}"
SFT_WEIGHT="${SFT_WEIGHT:-0.1}"

# ==========================  DeepSpeed  =======================================

# ZeRO-3 with CPU offload (for 8×80G).  Use zero3-fast.json for no offload.
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-training/config/zero3.json}"

# ==========================  logging / checkpointing  =========================

SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-10}"
REPORT_TO="${REPORT_TO:-tensorboard}"

# ==========================  precision  =======================================

# bf16 for A800/A100/H100; use --fp16 if bf16 not supported.
BF16="${BF16:-True}"

# ==========================  misc  ============================================

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-24}"

# ==========================  launch  ==========================================

# Build optional arguments
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
echo " LOGO Training Launch"
echo "============================================================"
echo " Model:               $MODEL_PATH"
echo " Model type:          $MODEL_TYPE"
echo " Dataset:             $DATASET_PATH"
echo " Output:              $OUTPUT_DIR"
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
echo " Beta:                $BETA"
echo " Gamma/beta ratio:    $GAMMA_BETA_RATIO"
echo " SFT weight:          $SFT_WEIGHT"
echo " Loss type:           $LOSS_TYPE"
echo " DeepSpeed:           $DEEPSPEED_CONFIG"
echo " Seed:                $SEED"
echo "============================================================"

deepspeed training/logo_train.py \
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
    --evaluation_strategy steps \
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
