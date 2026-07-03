#!/usr/bin/env bash
# =============================================================================
# End-to-End Pipeline: DS Data → LOGO Training
# =============================================================================
#
# 1. Convert DS raw data → LOGO-compatible HuggingFace Dataset
# 2. Build tokenized LOGO dataset (position synthesis + 12-field output)
# 3. Launch training with Qwen3.5-0.8B on 2 GPUs
#
# Usage:
#   bash scripts/run_ds_pipeline.sh
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ==========================  config  ==========================================

INPUT_DIR="${INPUT_DIR:-./data/DS_test}"
CONVERTED_PATH="${CONVERTED_PATH:-./data/DS_test/ds_logo_dataset}"
TOKENIZED_PATH="${TOKENIZED_PATH:-./data/DS_test/ds_logo_tokenized}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen3.5-0.8B}"
MODEL_TYPE="${MODEL_TYPE:-qwen3.5}"
SEED="${SEED:-42}"

echo "============================================================"
echo " Pipeline: DS Clinical Data → LOGO Training"
echo "============================================================"
echo " Input:         $INPUT_DIR"
echo " Converted:     $CONVERTED_PATH"
echo " Tokenized:     $TOKENIZED_PATH"
echo " Model:         $MODEL_PATH"
echo " Model type:    $MODEL_TYPE"
echo " Seed:          $SEED"
echo "============================================================"

# ==========================  Step 1: Convert  =================================

echo ""
echo "--- Step 1/3: Convert DS data to LOGO format ---"
python data/convert_ds_to_logo.py \
    --input_dir "$INPUT_DIR" \
    --output_path "$CONVERTED_PATH" \
    --chunk_size 300 \
    --seed "$SEED" \
    --overwrite

# ==========================  Step 2: Build dataset  ===========================

echo ""
echo "--- Step 2/3: Build tokenized LOGO dataset ---"
python data/build_logo_dataset.py \
    --input_path "$CONVERTED_PATH" \
    --output_path "$TOKENIZED_PATH" \
    --tokenizer_path "$MODEL_PATH" \
    --model_type "$MODEL_TYPE" \
    --context_mode paper \
    --max_seq_length 4096 \
    --target_position_length 8192 \
    --num_chunks 16 \
    --chunk_token_size 256 \
    --max_answer_tokens 512 \
    --position_variants_per_sample 2 \
    --continuous_ratio 0.8 \
    --seed "$SEED" \
    --overwrite

# ==========================  Step 3: Train  ===================================

echo ""
echo "--- Step 3/3: Launch LOGO training ---"
DATASET_PATH="$TOKENIZED_PATH" \
MODEL_PATH="$MODEL_PATH" \
MODEL_TYPE="$MODEL_TYPE" \
    bash scripts/train_logo_qwen_ds.sh
