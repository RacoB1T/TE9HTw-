# LOGO Training Data Pipeline

End-to-end pipeline from post-processed QA data to training-ready `DatasetDict`.

## Overview

```
post_process_critical_paths.py          (field normalization, typo-fixed)
        ↓
build_logo_dataset.py                   (tokenize, position synthesis, save)
        ↓
inspect_logo_dataset.py                 (validate & inspect)
        ↓
training/logo_train.py                  (load_from_disk → train)
```

## Quick Start

### Step 1: Build training dataset

```bash
python data/build_logo_dataset.py \
    --input_path /data/pre-process \
    --output_path ./logo_train_data \
    --tokenizer_path meta-llama/Meta-Llama-3-8B-Instruct \
    --context_mode existing \
    --max_seq_length 10000 \
    --target_position_length 65536 \
    --num_chunks 16 \
    --chunk_token_size 512 \
    --max_answer_tokens 512 \
    --position_variants_per_sample 2 \
    --continuous_ratio 0.9 \
    --test_ratio 0.02 \
    --seed 42 \
    --overwrite
```

### Step 2: Inspect the output

```bash
python data/inspect_logo_dataset.py \
    --dataset_path ./logo_train_data \
    --K_target 65536
```

### Step 3: Train

```bash
python training/logo_train.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset_path ./logo_train_data \
    --output_dir ./ckpt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    ...
```

## Input Format

The builder expects post-processed data (from `post_process_critical_paths.py`) with these fields per row:

| Field | Type | Description |
|---|---|---|
| `all_ref_text` | `List[str]` | Context chunks (shared across branches) |
| `combined_question` | `str` | Question |
| `final_answer` | `str` | Chosen answer (full paths) |
| `prefix_a` | `str` | Rejected answer 1 (half paths) |
| `suffix_a` | `str` | Rejected answer 2 (no critical paths) |
| `label` | `str` | Ground-truth answer (optional, metadata only) |

**Backward compatibility**: `siffix_a` (typo) is accepted as fallback for `suffix_a`.

## Output Format

The builder produces a HuggingFace `DatasetDict` with `train` and `test` splits.
Each row contains exactly 12 fields (all `List[int]`):

| Field | Description |
|---|---|
| `chosen_input_ids` | Full sequence: system + context + question + assistant_header + chosen_answer |
| `chosen_attention_mask` | All-ones mask |
| `chosen_position_ids` | Synthetic position IDs for chosen branch |
| `chosen_labels` | -100 for prompt, token IDs for answer |
| `reject_1_input_ids` | Same prefix + rejected_1_answer |
| `reject_1_attention_mask` | All-ones mask |
| `reject_1_position_ids` | Synthetic position IDs for rejected_1 branch |
| `reject_1_labels` | -100 for prompt, token IDs for answer |
| `reject_2_input_ids` | Same prefix + rejected_2_answer |
| `reject_2_attention_mask` | All-ones mask |
| `reject_2_position_ids` | Synthetic position IDs for rejected_2 branch |
| `reject_2_labels` | -100 for prompt, token IDs for answer |

**Key invariants**:
- All three branches share identical prompt prefix tokens.
- Labels are -100 on all prompt tokens.
- Labels equal input_ids on all answer tokens.
- Position IDs are strictly increasing and < `target_position_length`.
- No sample_id appears in both train and test.

## Auxiliary Files

| File | Content |
|---|---|
| `build_config.json` | Tokenizer, seed, dimensions, ratios, timestamp |
| `build_report.json` | Input/valid/dropped counts, split sizes, statistics |
| `metadata.jsonl` | Per-sample: sample_id, source, split, variant, strategy, lengths |

## Position Synthesis

Two strategies are supported:

- **Continuous** (~90%): Each chunk group gets contiguous position IDs within its slot. Adjacent chunk tokens have position delta = 1.
- **Sparse** (~10%): Chunk tokens get randomly sampled unique positions within the slot, sorted ascending. Adjacent tokens typically have delta > 1.

The position layout:

```
[0 ... system_len-1]                          ← system region (continuous)
[system_len ... tail_start-1]                 ← context region (N slots, synthetic)
[tail_start ... K_target-1]                   ← QA tail region (continuous)
```

All randomness is controlled by a single seed for full reproducibility.

## Command-Line Reference

### `build_logo_dataset.py`

```
--input_path                  Path to post-processed dataset(s)  [required]
--output_path                 Output directory for DatasetDict   [required]
--tokenizer_path              HuggingFace tokenizer path         [required]
--context_mode                existing | paper                   [default: existing]
--max_seq_length              Max real token length              [default: 10000]
--real_reference_tokens       Budget for reference tokens        [default: 8192]
--target_position_length      K_target position window           [default: 65536]
--num_chunks                  Max context chunks                 [default: 16]
--chunk_token_size            Max tokens per chunk               [default: 512]
--max_answer_tokens           Max answer tokens                  [default: 512]
--position_variants_per_sample Variants per sample               [default: 2]
--continuous_ratio            Fraction of continuous variants    [default: 0.9]
--test_ratio                  Test split fraction                [default: 0.02]
--seed                        Random seed                        [default: 42]
--strict                      Strict mode (fail on short)        [flag]
--overwrite                   Overwrite existing output          [flag]
--no_deduplicate              Disable answer dedup               [flag]
```

### `inspect_logo_dataset.py`

```
--dataset_path                Path to built DatasetDict          [required]
--K_target                    Position window to check against   [default: 65536]
--sample_idx                  Sample index to display            [default: 0]
--split                       Split to display from              [default: train]
```

## Tests

```bash
# Position synthesis tests
python -m pytest tests/test_position_synthesis.py -v

# Builder end-to-end tests
python -m pytest tests/test_build_logo_dataset.py -v

# All tests
python -m pytest tests/ -v
```
