# LOGO 训练流程文档

## 概述

LOGO (Long-context Gap-filling with Odds-ratio Optimization) 是一个长上下文偏好对齐训练框架。本实现针对**出院小结生成**场景：输入患者的临床时序事件（MIMIC-III），输出 Diagnosis + Brief Hospital Course + Discharge Instructions。

### 核心技术

- **Position 合成**：将 context chunks 映射到高维合成 position space
- **SimPO Loss**：reference-free 偏好优化，无需 ref model
- **时间驱动的 Position 编码**：保留临床事件的时序结构
- **Cross-patient + Entity-swap 负样本**：构造需要内容推理的 rejected answers

---

## 数据流转

```
DS_long/input/*.csv (3807 个患者的临床时序事件)
DS_long/gold_process/*.txt (3802 个患者的出院小结)
        │
        ▼
Step 1: data/convert_ds_to_logo.py
        │
        ├── 读取 CSV → 按 TIME 排序 → chunk 分组 (~300 chars)
        ├── 解析时间戳 (绝对 YYYY-MM-DD + 相对 "14 hours")
        ├── 临床显著性评分 (正则匹配 手术/诊断/危急/会诊/用药等)
        ├── ROUGE-1 对齐得分 (快速 unigram overlap)
        ├── 选出 critical_chunks (top-50%) + irrelevant_chunks
        ├── 生成 rejected answers:
        │   ├── reject_1: 删除临床实体密集的句子 + 剩余句子中替换实体
        │   └── reject_2: 段落重排 + 跨患者注入 + 删句子
        └── 输出 → DatasetDict {train: ~3726, test: ~76}
             字段: all_ref_text, chunk_timestamps, combined_question,
                   final_answer, prefix_a, suffix_a, label,
                   critical_chunks, irrelevant_chunks
        │
        ▼
Step 2: data/build_logo_dataset.py
        │
        ├── Tokenize (Llama-3 ChatML 或 Qwen3.5 格式)
        ├── Context 构建 (paper mode: critical + sampled irrelevant)
        ├── 长度控制 (优先级: 答案 > chunks > question)
        ├── Position ID 合成:
        │   ├── 有 timestamp → 时间驱动 (TimeLayout)
        │   │   └── 按时间间隔比例映射到 position space
        │   │   └── continuous: ±2% jitter | sparse: ±15% jitter
        │   ├── 无 timestamp → 等宽 slot (向后兼容)
        │   └── QA tail: continuous 连续 position
        ├── 三路共享 prefix, labels 中 prompt 区 = -100
        └── 输出 → 12 字段 DatasetDict {train: ~7452, test: ~152}
             chosen/reject_1/reject_2 × (input_ids, attention_mask,
                                          position_ids, labels)
        │
        ▼
Step 3: training/logo_train.py (LOGOTrainer)
        │
        ├── 模型加载 + LoRA (q/k/v/o_proj, r=8, alpha=4)
        ├── modules_to_save: 65 norm layers
        ├── concatenated_forward (3 分支, sequential 或 batched)
        ├── SimPO Loss:
        │   ├── pi_logratios = chosen_logps - avg(rejected_logps)
        │   ├── logits = pi_logratios - γ/β
        │   ├── loss = -log(σ(β*logits))
        │   └── + sft_weight × CE_loss (仅 answer token)
        ├── 2×GPU + DeepSpeed ZeRO-2 + flash_attn_2 + bf16
        └── 输出 → checkpoints + merged model
```

---

## 关键参数

### 推荐模型
| 模型 | Vocab | 推荐 |
|---|---|---|
| Llama3.1-8B-Instruct | 128K | ✅ 稳定，~10s/step |
| Qwen3.5-4B | 248K | ⚠️ 需 sequential forward (~42s/step) |

### 训练参数 (当前)
| 参数 | 值 | 说明 |
|---|---|---|
| max_seq_length | 4096 | 2×32GB GPU 上限 |
| target_position_length | 8192 | Position 空间大小 |
| num_chunks | 16 | Context chunk 数 |
| chunk_token_size | 200 | 每个 chunk 最大 token 数 |
| max_answer_tokens | 1024 | 答案最大 token 数 |
| lora_r / alpha | 8 / 4 | LoRA 秩 |
| trainable_params | norm | 65 层 norm，7.3M params |
| max_steps | 1200 | ~1.3 epochs |
| learning_rate | 2e-5 | cosine schedule |
| beta | 3.0 | SimPO 温度 |
| gamma_beta_ratio | 0.2 | 偏好 margin |
| sft_weight | 0.3 | SFT 辅助 loss 权重 |
| label_smoothing | 0.05 | SimPO label smoothing |
| batch | 1×4grad×2GPU=8 | 有效 batch size |

---

## Rejected Answer 设计 (v5)

### reject_1: 临床实体替换 + 关键句删除
1. 按临床实体密度对句子排序（用 CLINICAL_ENTITY_TABLE 匹配）
2. 删除 35% 实体最密集的句子
3. 在剩余句子中随机替换 4 个临床实体（诊断/手术/药物/结局）
4. 结果: ~60% token 重叠，需要检查事实准确性

**替换表示例** (120+ 对):
```
STEMI → NSTEMI, PCI → CABG, metoprolol → atenolol,
sepsis → SIRS, improved → worsened, discharged → transferred to ICU
```

### reject_2: 结构性破坏 + 跨患者污染
1. 随机重排所有句子
2. 删除 30-40% 句子
3. 随机插入 5-8 句来自另一个患者的 gold
4. 结果: ~70% token 重叠，需要识别结构错误和外来内容

---

## 时间驱动的 Position 编码

### 时间戳解析
```python
# 绝对时间: "2119-01-30 00:00:00" → Unix timestamp
# 相对时间: "14 hours", "38 minutes later" → 累积偏移
# 锚点: 第一个绝对时间戳
```

### Position 映射
```
Context region: [system_len, tail_start)
Chunk i position: system_len + (t_i - t_min) / (t_max - t_min) × context_size
```
时间间隔大的 chunk 之间 → position gap 大
ICU 密集事件的 chunk → position 紧凑

---

## 文件清单

| 文件 | 角色 |
|---|---|
| `data/convert_ds_to_logo.py` | DS 原始数据 → LOGO 格式 (临床评分 + 时间戳 + rejected 生成) |
| `data/build_logo_dataset.py` | Tokenize + Position 合成 → 12 字段训练数据 |
| `data/position_synthesis.py` | TimeLayout + 时序/sparse/continuous position 合成 |
| `training/logo_train.py` | LOGOTrainer (SimPO loss + concatenated_forward) |
| `training/simpo_trainer.py` | SimPOTrainer 基类 |
| `training/custom_dataset.py` | SimPODataCollator (12 字段 batch 化) |
| `utils/utils.py` | 模型加载 + LoRA 配置 + Position 编码配置 |
| `training/config/zero2-minimal.json` | DeepSpeed ZeRO-2 配置 |
| `evaluate_ds.py` | 推理 + ROUGE-L 评测 |

---

## 完整训练命令

```bash
cd /home/qluai/lzy/TE9HTw-/

# Step 1: 转换 DS 数据
python3 data/convert_ds_to_logo.py \
    --input_dir data/DS_long \
    --output_path data/DS_long/ds_logo_dataset_v5 \
    --chunk_size 300 --test_ratio 0.02 --seed 42 --overwrite

# Step 2: 构建 tokenized 数据集 (Llama-3)
python3 data/build_logo_dataset.py \
    --input_path data/DS_long/ds_logo_dataset_v5 \
    --output_path data/DS_long/ds_logo_tokenized_v5_llama \
    --tokenizer_path ~/models/Llama3.1-8B-Instruct \
    --model_type llama-3 --context_mode paper \
    --max_seq_length 4096 --target_position_length 8192 \
    --num_chunks 16 --chunk_token_size 200 --max_answer_tokens 1024 \
    --position_variants_per_sample 2 --continuous_ratio 0.8 \
    --seed 42 --overwrite

# Step 3: 训练
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
deepspeed --include localhost:0,1 --master_port 33600 \
  training/logo_train.py \
  --model_name_or_path ~/models/Llama3.1-8B-Instruct \
  --model_type llama-3 --attn_implementation flash_attention_2 \
  --max_position_embeddings 8192 --lora_r 8 --lora_alpha 4 \
  --dataset_path data/DS_long/ds_logo_tokenized_v5_llama \
  --output_dir outputs/llama_ds_long_v5 --max_steps 1200 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_steps 60 \
  --optim adamw_torch --max_seq_length 4096 --max_target_length 1024 \
  --beta 3.0 --gamma_beta_ratio 0.2 --sft_weight 0.3 \
  --low_rank_training True --trainable_params norm \
  --gradient_checkpointing True --bf16 True \
  --deepspeed training/config/zero2-minimal.json \
  --save_steps 200 --eval_steps 200 --logging_steps 20 \
  --load_best_model_at_end True --metric_for_best_model eval_loss \
  --save_total_limit 5 --seed 42 --report_to none
```

---

## 已知局限

1. **ROUGE-1 替代 ROUGE-L** — chunk 评分用快速 unigram overlap，精度略低于 LCS
2. **实体替换表覆盖不全** — 120+ 对，但某些罕见诊断可能没覆盖
3. **评估只有 ROUGE-L** — 缺 CUI F-score (QuickUMLS) 和 SapBERT 语义相似度
4. **norm only 训练** — 0.09% trainable params，embed 层因显存限制未参与训练
5. **Qwen3.5 慢** — GatedDeltaNet 无 FLA 加速，仅 torch fallback
