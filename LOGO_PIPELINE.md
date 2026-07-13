# LOGO 训练流程文档

## 概述

LOGO (Long-context Gap-filling with Odds-ratio Optimization) 是一个长上下文偏好对齐训练框架。本实现针对**出院小结生成**场景：输入患者的临床时序事件（MIMIC-III），输出 Diagnosis + Brief Hospital Course + Discharge Instructions。

### 核心技术

- **Position 合成**：将 context chunks 映射到高维合成 position space（时间驱动 / 等宽 slot）
- **SimPO Loss**：reference-free 偏好优化，无需 ref model
- **LoRA 微调**：只训 attention projection + norm 层（~7.3M 参数）
- **三路 Rejected Answer**：数值/时间篡改 + 事件替换 + 段落污染

---

## 数据流转

```
data/DS_long/input/*.csv (3807 个患者的临床时序事件)
data/DS_long/gold_process/*.txt (3802 个患者的出院小结)
        │
        ▼
Step 1: data/convert_ds_to_logo.py
        │
        ├── 读取 CSV → 按 TIME 排序 → 切句 → chunk 分组 (~80 tokens)
        ├── 解析时间戳 (绝对 YYYY-MM-DD + 相对 "14 hours later")
        ├── 五层临床重要性评分 (score_chunk_clinical):
        │   ├── Layer 1: 事件类型基础分 (clinical_note=28, procedure=12, med_change=8...)
        │   ├── Layer 2: 检验值危急加分 (critical=+5, abnormal=+3, 30+ lab tests)
        │   ├── Layer 3: 时间上下文乘数 (admission ×1.5, pre-discharge ×1.0-1.5)
        │   ├── Layer 4: ROUGE-1 对齐 gold 乘数 (1 + ROUGE-1)
        │   ├── Layer 5: 面板效应加分 (5+ 异常值 → +5)
        │   └── 长度归一化: 除以 ln(estimated_tokens + 1)
        ├── 区分 critical_chunks (top-50%) 和 irrelevant_chunks (bottom-50%)
        ├── 生成 3 种 rejected answers:
        │   ├── reject_1 (数值/时间篡改): 改 8-15 个检验值/药物剂量 + 偏移 1-3 个时间标记
        │   │    期望 token 重叠 ~92-96%
        │   ├── reject_2 (事件替换): 替换手术/操作 + 插入伪造并发症 + 增删诊断
        │   │    期望 token 重叠 ~75-85%
        │   └── reject_3 (段落污染): 句子跨段迁移 + 跨患者注入 + 药物矛盾
        │        期望 token 重叠 ~75-85%
        └── 输出 → DatasetDict {train: ~3726, test: ~76}
             字段: all_ref_text, chunk_timestamps, combined_question,
                   final_answer, prefix_a (reject_1), suffix_a (reject_2),
                   tertiary_a (reject_3), label,
                   critical_chunks, irrelevant_chunks
        │
        ▼
Step 2: data/build_logo_dataset.py
        │
        ├── Tokenize (Llama-3 ChatML / Qwen3.5 / Llama2 三种模板)
        ├── Context 构建:
        │   ├── "existing" mode: 使用 all_ref_text 原样
        │   └── "paper" mode: critical_chunks + 随机采样 irrelevant_chunks
        ├── 长度控制 (截断优先级: answers > chunks > question framing)
        ├── Position ID 合成 (data/position_synthesis.py):
        │   ├── 有 timestamp → 时间驱动 (TimeLayout)
        │   │   └── 按时间间隔比例映射到 position space
        │   │   └── continuous (±2% jitter, 80-90%) | sparse (±15% jitter, 10-20%)
        │   ├── 无 timestamp → 等宽 slot (Layout, 向后兼容)
        │   └── QA tail: 始终 continuous 连续 position
        ├── 四路共享 prefix, labels 中 prompt 区 = -100
        └── 输出 → 12 字段 DatasetDict {train: ~7452, test: ~152}
             chosen / reject_1 / reject_2 / reject_3 ×
               (input_ids, attention_mask, position_ids, labels)
        │
        ▼
Step 3a: training/logo_train.py (LOGOTrainer, 主训练方式)
        │
        ├── 模型加载 (utils/utils.py): FlashAttention-2 + bf16 + LoRA
        │   ├── LoRA targets: q_proj, k_proj, v_proj, o_proj
        │   ├── modules_to_save: embed_tokens + 全部 norm 层 (默认 trainable_params="embed, norm")
        │   └── gradient checkpointing
        ├── LOGOTrainer.concatenated_forward: 四分支顺序前向 (节省显存)
        ├── SimPO Loss:
        │   ├── pi_logratios = chosen_logps - avg(reject_1/2/3_logps)
        │   ├── logits = pi_logratios - γ/β
        │   ├── loss = -log(σ(β*logits))
        │   └── + sft_weight × CE_loss (仅 answer tokens)
        ├── SimPODataCollator: 12 字段 stack + pad/truncate
        └── 输出 → LoRA checkpoints + merged model
        │
        ▼
Step 3b: training/sft_train.py (SFT 基线, 用于与 LOGO 对比)
        │
        ├── 仅用 chosen 分支做标准监督微调
        ├── ChosenOnlyDataset 包装器
        └── 标准 HuggingFace Trainer (无 SimPO loss)，作为性能对比基线

        │
        ▼
Step 4: evaluate_ds.py (评测)
        │
        ├── 加载合并后模型 (FlashAttention-2, bf16, device_map="auto")
        ├── 读取 metadata.jsonl 筛选 test split 患者
        ├── 逐患者: build_prompt() → generate_summary()
        │   (max_new_tokens=1024, temp=0.6, top_p=0.9, rep_penalty=1.2)
        ├── extract_sections() 正则提取三段:
        │   "Diagnosis:" / "Hospital Course:" / "Discharge Instructions:"
        └── 指标: ROUGE-L F1 (全文 + 分段), BERTScore (Bio_ClinicalBERT, 可选)
```

---

## 关键参数

### 训练配置 (两套)

| 参数 | Llama-3.1-8B-Instruct | Qwen3.5-0.8B |
|---|---|---|
| 启动脚本 | `scripts/train_logo.sh` | `scripts/train_logo_qwen_ds.sh` |
| GPU | 8×A800-80G | 2 GPU |
| DeepSpeed | ZeRO-3 | ZeRO-2 |
| max_seq_length | 10000 | 4096 |
| target_position_length | 65536 | 8192 |
| lora_r / alpha | 32 / 16 | 16 / 8 |
| trainable_params | embed, norm | embed, norm |
| max_steps | - | 1200 (~1.3 epochs) |
| learning_rate | 5e-7 | 2e-5 |
| beta | 2.0 | 3.0 |
| gamma_beta_ratio | 0.25 | 0.2 |
| sft_weight | 0.1 | 0.3 |
| label_smoothing | 0.0 | 0.0 |
| batch | - | 1×4grad×2GPU=8 |

### 数据参数

| 参数 | 值 | 说明 |
|---|---|---|
| chunk_size | 80 tokens | convert 时的 chunk 大小 |
| num_chunks | 16 | Context chunk 数 |
| chunk_token_size | 200 | 每个 chunk 最大 token 数 |
| max_answer_tokens | 1024 | 答案最大 token 数 |
| position_variants_per_sample | 2 | 每个样本生成几个 position 变体 |
| continuous_ratio | 0.8 | continuous vs sparse 比例 |

---

## Rejected Answer 设计

### reject_1: 数值/时间篡改 (`generate_rejected_numerical_temporal`)
1. 在 gold answer 中识别 8-15 个数值 (lab values, vitals, drug dosages)
2. 随机扰动这些数值 (保持单位不变)
3. 偏移 1-3 个时间标记 ("day 3" → "day 5")
4. 期望 token 重叠 ~92-96%，需要检查数值准确性

### reject_2: 事件替换 (`generate_rejected_event_substitution`)
1. 替换手术/操作名称 (用 CLINICAL_ENTITY_TABLE, 120+ 对)
2. 插入 1-2 个伪造的并发症
3. 添加 1 个虚假诊断
4. 删除 1 个真实诊断
5. 期望 token 重叠 ~75-85%，需要识别内容真伪

### reject_3: 段落污染 (`generate_rejected_section_contamination`)
1. 句子跨 Diagnosis / Hospital Course / Discharge Instructions 三段迁移
2. 注入 2-3 句来自其他患者的同主题内容
3. 制造药物矛盾 (说停药但在后续段落中继续使用)
4. 期望 token 重叠 ~75-85%，需要识别结构性/跨患者错误

**替换表示例** (CLINICAL_ENTITY_TABLE, 120+ 对):
```
STEMI → NSTEMI, PCI → CABG, metoprolol → atenolol,
sepsis → SIRS, improved → worsened, discharged → transferred to ICU
```

---

## 时间驱动的 Position 编码

### 时间戳解析
```python
# 绝对时间: "2119-01-30 00:00:00" → Unix timestamp
# 相对时间: "14 hours", "38 minutes later" → 累积偏移
# 锚点: 第一个绝对时间戳
```

### Position 映射 (TimeLayout)
```
Context region: [system_len, tail_start)
Chunk i position: system_len + (t_i - t_min) / (t_max - t_min) × context_size
```
- 时间间隔大的 chunk 之间 → position gap 大
- ICU 密集事件的 chunk → position 紧凑
- 无时间戳时退化为等宽 slot (Layout)

### 合成策略
- **Continuous** (80-90%): chunk 内连续 position ID, ±2% jitter
- **Sparse** (10-20%): chunk 内随机采样唯一 position, ±15% jitter
- **QA tail**: 始终 continuous

---

## 文件清单

### 当前 Pipeline 核心文件

| 文件 | 角色 |
|---|---|
| `data/convert_ds_to_logo.py` | Step 1: DS 原始数据 → LOGO 格式 (五层临床评分 + 时间戳 + 3 种 rejected 生成) |
| `data/build_logo_dataset.py` | Step 2: Tokenize + Position 合成 → 12 字段训练数据 |
| `data/position_synthesis.py` | TimeLayout / Layout + continuous / sparse / temporal position 合成 |
| `training/logo_train.py` | Step 3a: LOGOTrainer (SimPO loss + 四分支 sequential forward) |
| `training/simpo_trainer.py` | SimPOTrainer 基类 (move_to_device, get_batch_logps, compute_loss) |
| `training/custom_dataset.py` | SimPODataCollator (12 字段 batch 化) |
| `training/sft_train.py` | Step 3b: SFT 基线训练 (仅 chosen 分支, 标准 Trainer, 与 LOGO 对比性能) |
| `utils/utils.py` | 模型加载 + LoRA 配置 + Position 编码配置 + tokenizer |
| `evaluate_ds.py` | Step 4: 推理生成 + ROUGE-L + BERTScore 评测 |
| `merge_model.py` | LoRA 权重合并工具 |
| `scripts/run_ds_pipeline.sh` | 端到端流程脚本 (Step 1→2→3, Qwen3.5-0.8B) |
| `scripts/train_logo.sh` | Llama-3.1-8B 训练启动脚本 |
| `scripts/train_logo_qwen_ds.sh` | Qwen3.5-0.8B 训练启动脚本 |
| `training/config/zero2-minimal.json` | DeepSpeed ZeRO-2 配置 |
| `training/config/zero3-minimal.json` | DeepSpeed ZeRO-3 配置 |

### 未在当前 Pipeline 中使用的文件

| 文件 | 说明 |
|---|---|
| `data/ImportanceScoring.py` | 旧版评分 (依赖 modelzipper + spaCy)，未被任何脚本引用 |
| `process_ds_gold.py` | Gold summary 处理，独立脚本，不参与 pipeline |
| `data/inspect_logo_dataset.py` | 数据集检查工具，可独立运行 |
| `data/gen_pre_dis_pre_data/` | 预出院数据生成，未集成 |
| `data/positional_indices_synthesis.ipynb` | Jupyter notebook，非脚本 |
| `tmp_download_lb.py` | 临时下载脚本 |
| `training/instruct_tuning.py` | 指令微调，独立入口，未被调用 |
| `training/language_modeling.py` | 语言模型训练，独立入口，未被调用 |
| `evaluation/` | 已从磁盘删除 (LongBench, BABILong, LongPPL)，仅在 git 历史中 |

---

## 完整训练命令

```bash
cd /home/qluai/lzy/TE9HTw-/

# Step 1: 转换 DS 数据
python3 data/convert_ds_to_logo.py \
    --input_dir data/DS_long \
    --output_path data/DS_long/ds_logo_dataset_v5 \
    --chunk_size 80 --test_ratio 0.02 --seed 42 --overwrite

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

# Step 3a: LOGO 训练 (Qwen3.5-0.8B, 2 GPU)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0,1 \
deepspeed --include localhost:0,1 --master_port 33600 \
  training/logo_train.py \
  --model_name_or_path ~/models/Qwen3.5-0.8B-Instruct \
  --model_type qwen3.5 --attn_implementation flash_attention_2 \
  --max_position_embeddings 8192 --lora_r 16 --lora_alpha 8 \
  --dataset_path data/DS_long/ds_logo_tokenized_v5_qwen \
  --output_dir outputs/qwen_ds_long_v5 --max_steps 1200 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_steps 60 \
  --optim adamw_torch --max_seq_length 4096 --max_target_length 1024 \
  --beta 3.0 --gamma_beta_ratio 0.2 --sft_weight 0.3 \
  --low_rank_training True \
  --gradient_checkpointing True --bf16 True \
  --deepspeed training/config/zero2-minimal.json \
  --save_steps 200 --eval_steps 200 --logging_steps 20 \
  --load_best_model_at_end True --metric_for_best_model eval_loss \
  --save_total_limit 5 --seed 42 --report_to none

# Step 3b: SFT 基线训练 (与 LOGO 对比性能)
python3 training/sft_train.py \
  --model_name_or_path ~/models/Llama3.1-8B-Instruct \
  --dataset_path data/DS_long/ds_logo_tokenized_v5_llama \
  --output_dir outputs/llama_ds_long_sft \
  --max_seq_length 4096 --learning_rate 2e-5 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 3 --bf16 True \
  --gradient_checkpointing True --logging_steps 20 \
  --save_steps 200 --save_total_limit 3

# Step 4: 评测
python3 evaluate_ds.py \
  --model_path outputs/qwen_ds_long_v5/merged \
  --data_dir data/DS_long \
  --output_dir outputs/eval_results \
  --max_new_tokens 1024 --use_bertscore
```

---

## 已知局限

1. **ROUGE-1 替代 ROUGE-L** — chunk 评分阶段用快速 unigram overlap，精度略低于 LCS
2. **实体替换表覆盖不全** — 120+ 对，罕见诊断可能未覆盖
3. **评估指标有限** — 仅有 ROUGE-L + BERTScore，缺 CUI F-score 和 SapBERT 语义相似度
4. **embed + norm 训练** — embed_tokens + norm 层参数量很小（相比全量微调），但 embed 层参与训练可能带来词表级别的适配
5. **Qwen3.5 大 vocab** — 248K vocab 导致 sequential forward 变慢 (~42s/step)
6. **三路 reject 标签共用** — SimPO loss 对 3 个 reject 取平均，未区分不同类型的重要程度
