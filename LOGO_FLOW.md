# LOGO 代码流程文档

> 基于论文 *Long cOntext aliGnment via efficient preference Optimization* 的实现流程。

---

## 总体架构

```
原始数据 {Q, C, P}
      │
      ▼
阶段一：数据构造 (data/)
      │
      ▼
阶段二：LOGO 训练 (training/)
      │
      ▼
阶段三：评测 (evaluation/)
```

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `data/logo_data_pipeline.py` | **新建** | 端到端数据构造流水线（主入口） |
| `data/positional_indices_synthesis.py` | **新建** | 连续+稀疏 position_ids 合成 |
| `data/ImportanceScoring.py` | 已存在 | spaCy NER 重要性打分 |
| `data/gen_pre_dis_pre_data/gen_hf.py` | 已存在 | LLM API 生成 chosen/rejected 答案 |
| `data/gen_pre_dis_pre_data/post_process_critical_paths.py` | 已存在 | LLM judge 评判答案质量 |
| `training/logo_train.py` | **已修改** | LOGO 训练主入口 |
| `training/simpo_trainer.py` | **新建** | SimPO 基础训练器 |
| `training/custom_dataset.py` | **已修改** | 新增 LOGODataCollator |
| `training/scripts/llama3_logo.sh` | 已存在 | 训练启动脚本 |
| `utils/utils.py` | **新建** | 模型加载 + LoRA |
| `evaluation/chat.py` | 已存在 | FastChat 模板 |

---

## 阶段一：LOGO 训练数据构造

### 主入口

```bash
python data/logo_data_pipeline.py \
    --input_dir /data/raw_samples \
    --output_dir /data/logo_train_data \
    --tokenizer meta-llama/Meta-Llama-3-8B-Instruct \
    --chunk_size 512 \
    --num_chunks 16 \
    --delta_threshold 6.0 \
    --max_embedding_size 65536 \
    --max_seq_length 10000 \
    --max_target_length 2000 \
    --continuous_ratio 0.9 \
    --num_proc 4
```

### 核心流程

```
输入: {"question": Q, "context": C, "answer": P}

Step 1 ── chunk_context_by_tokens(C, tokenizer, 512, 16)
         │  将上下文按 tokenizer 切分为 512 token 的 chunk
         │  保留 16 个 chunk（前8 + 后8，覆盖全文首尾）
         └──→ chunk_texts: ["text_0", "text_1", ..., "text_15"]

Step 2 ── ChunkImportanceScorer.score_all_chunks(Q, chunk_texts)
         │  用 spaCy 提取命名实体 + 动词 + 名词
         │  Score(C_i) = |entities(Q) ∩ entities(C_i)|
         │  无 spaCy 时自动降级为词级重叠（支持中文等非英语数据）
         └──→ scored: [{chunk_id, score}, ...] 降序排列

Step 3 ── classify_chunks(scored, delta=6.0)
         │  score > 6  → critical（关键证据块）
         │  score ≤ 6  → irrelevant（无关干扰块）
         │  可选 top-k 模式：选最高/最低 k 个（适合非英语场景）
         └──→ critical: [{chunk_id, text, score}]
              irrelevant: [{chunk_id, text, score}]

Step 4 ── build_shared_context(critical, irrelevant)
         │  共享上下文 C' = 所有 critical + 若干 irrelevant（作为噪声）
         │  按 chunk_id 排序恢复原文顺序
         └──→ shared_chunks: ["critical_0", "irrelevant_0", "critical_1", ...]

Step 5-6 ── tokenize_logo_sample(tokenizer, shared_chunks, Q,
         │                        P_chosen, P_rejected1, P_rejected2)
         │
         │  构建 prompt: system_prompt + [chunk + beacon]*N + question
         │  构建 3 个 answer (共享同一个 prompt)
         │
         │  调用 position_indices_synthesis:
         │  ├── Continuous (90%): 同 chunk 内 position 连续，chunk 间有 gap
         │  │   例: [0..99] [5000..5123] [15000..15127]
         │  └── Sparse (10%): token 在分配区间内间隔分布
         │      例: [0,12,24,...] [5000,5020,5040,...]
         │
         └──→ 输出: 12 个字段的训练样本
```

### 输出格式

```json
{
  "chosen_input_ids":       "[system + C' + Q + P_chosen]",
  "chosen_attention_mask":  "[1,1,...,1,0,0,...]",
  "chosen_position_ids":    "[0,1,2,...,0,5000,...,0]",
  "chosen_labels":          "[-100,...,-100, tok_answer]",

  "reject_1_input_ids":     "[system + C' + Q + P_rejected1]",
  "reject_1_attention_mask":"[...]",
  "reject_1_position_ids":  "[...]",
  "reject_1_labels":        "[-100,...,-100, tok_answer]",

  "reject_2_input_ids":     "[system + C' + Q + P_rejected2]",
  "reject_2_attention_mask":"[...]",
  "reject_2_position_ids":  "[...]",
  "reject_2_labels":        "[-100,...,-100, tok_answer]"
}
```

### 关键设计点

1. **共享 prompt**：chosen 和两个 rejected **共享同一个 C'**，只有 answer 不同。模型学到的是"同一上下文中哪个答案更好"，而非"不同上下文导致不同答案"
2. **position_ids 合成**：真实输入 ~8K-10K tokens，但 position_ids 被**人为拉开**到 64K/80K 空间，模拟长序列
3. **labels 掩码**：prompt 部分全部 `-100`，只对 answer 部分计算 loss

---

## 阶段二：LOGO 偏好优化训练

### 启动命令

```bash
deepspeed --include localhost:0,1,2,3,4,5,6,7 ./training/logo_train.py \
    --output_dir=./output/llama3_logo \
    --model_name_or_path=Llama-3-8B-Instruct-80K \
    --dataset_path=/data/logo_train_data \
    --beta 10 \
    --gamma_beta_ratio 0.3 \
    --sft_weight 0.1 \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=5e-7 \
    --lora_r=32 \
    --lora_alpha=16 \
    --max_seq_length=10000 \
    --max_target_length=2000 \
    --bf16 True \
    --deepspeed=./training/config/zero3.json
```

### 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| beta | 10 | 控制 reward 缩放，β 越大 margin 越陡峭 |
| gamma_beta_ratio | 0.3 | γ/β=3/10，目标 margin γ=3 |
| sft_weight | 0.1 | SFT 正则化权重 λ |
| lora_r | 32 | LoRA 秩 |
| lora_alpha | 16 | LoRA α |
| learning_rate | 5e-7 | 学习率 |
| per_device_batch_size | 1 | 单卡 batch size |
| gradient_accumulation_steps | 8 | 梯度累积（有效 batch = 8×8 = 64） |
| max_seq_length | 10000 | 最大序列长度 |
| max_target_length | 2000 | 最大 answer 长度 |
| DeepSpeed | ZeRO-3 | 8 GPU, bf16 |
| 训练步数 | 1200 | warmup 120 步, cosine 调度 |

### 训练循环

```
1. 加载数据
   └── Dataset['train'] + Dataset['test']

2. LOGODataCollator 整理 batch
   ├── labels 填充 -100
   ├── position_ids 填充 0
   ├── input_ids / attention_mask 填充 0
   └── 统一 pad/truncate 到 max_seq_length=10000

3. concatenated_forward(model, batch)
   │
   │  将 chosen + reject_1 + reject_2 沿 batch 维拼接
   │  input_ids    = [B*C,  B*R1,  B*R2]   shape: (3B, L)
   │  attention_mask = 同上
   │  position_ids   = 同上                ← 使用合成 position_ids!
   │  labels         = 同上
   │
   │  all_logits = model(input_ids, attention_mask, position_ids=position_ids)
   │
   │  仅对 answer 部分计算 log_prob (labels != -100 的位置)
   │  → chosen_logps, reject1_logps, reject2_logps

4. LOGO Loss
   │
   │  L_pref = -log σ( β · ( log(π_w/π_l) - γ/β ) )
   │          = -log σ(10 · (chosen_logps - avg(rejected_logps) - 0.3))
   │
   │  当 chosen 比 rejected 平均高出 >0.3 时, loss → 0
   │  否则 loss → 大
   │
   │  L_SFT = CrossEntropy(chosen_logits, chosen_labels)
   │
   │  L = 0.1 · L_SFT + L_pref

5. 反向传播 → 更新 LoRA 参数
```

### LOGOTrainer 继承链

```
Trainer (HuggingFace)
  └── SimPOTrainer (training/simpo_trainer.py)
        ├── simpo_loss()           ← 无参考模型偏好损失
        ├── get_batch_logps()      ← 仅计算 answer 部分 log_prob
        ├── concatenated_forward() ← 拼接 chosen+rejected 后前向传播
        └── compute_loss()         ← 覆盖 Trainer.compute_loss
              └── LOGOTrainer (training/logo_train.py)
                    ├── simpo_loss()           ← 与父类相同
                    ├── concatenated_forward() ← 重写：1 chosen + 2 rejected
                    └── get_batch_loss_metrics() ← L = L_pref + λ·L_SFT
```

### 与标准 DPO/SimPO 的区别

| | DPO | SimPO | LOGO |
|---|---|---|---|
| Reference Model | 需要 | 不需要 | 不需要 |
| rejected 数量 | 1 | 1 | **2** |
| rejected 聚合 | 直接比较 | 直接比较 | **取平均后比较** |
| position_ids | 默认连续 | 默认连续 | **合成非连续** |
| 正则化 | 无 | 无 | **SFT loss (λ=0.1)** |
| 训练显存 | 高（需 ref model） | 低 | 低 |

---

## 阶段三：评测

```bash
# 合并 LoRA 权重
python merge_model.py \
    --base_model Llama-3-8B-Instruct-80K \
    --peft_model ./output/llama3_logo/final_checkpoint \
    --output ./output/llama3_logo_merged

# LongBench 评测
python evaluation/longbench/main.py \
    --model_path ./output/llama3_logo_merged

# NIAH / Babilong 检索评测
python evaluation/babilong/main.py \
    --model_path ./output/llama3_logo_merged

# 长文本 PPL
python evaluation/LongPPL/main.py \
    --model_path ./output/llama3_logo_merged
```

---

## 位置索引合成详解

### Continuous Chunk (90%)

```
真实 chunk:
  chunk_0: [tok_0, tok_1, ..., tok_511]   (512 tokens)
  chunk_1: [tok_0, tok_1, ..., tok_511]   (512 tokens)
  chunk_2: [tok_0, tok_1, ..., tok_511]   (512 tokens)

合成 position_ids:
  system:     [0, 1, 2, ..., S-1]          (S = system prompt 长度)
  chunk_0:    [S, S+1, S+2, ..., S+511]    (连续)
  chunk_1:    [20000, 20001, ..., 20511]   (连续, 但有 gap!)
  chunk_2:    [40000, 40001, ..., 40511]   (连续, gap 更大!)
  question:   [65024, 65025, ..., 65535]   (QA 预留空间, 连续)
  answer:     [65024+Q, ..., 65535]        (紧跟 question)

特点: 同 chunk 内连续 → 保留局部语义
     chunk 间有 gap  → 模拟长位置分布
```

### Sparse Chunk (10%)

```
真实 chunk:
  chunk_0: [tok_0, tok_1, tok_2]           (3 tokens, 分配区间 100)

合成 position_ids:
  chunk_0:    [0, 50, 99]                   (间隔分布, 不连续)

特点: token 在分配区间内均匀/随机分布
     破坏局部连续性, 但让模型暴露于更多样的位置模式
```

---

## 常见问题 / 易错点

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| position_ids 与 input_ids 长度不一致 | 构造时未对齐 | `LOGODataCollator` 会自动检查并抛出异常 |
| labels 在 prompt 部分也参与 loss | 未设为 -100 | `tokenize_logo_sample()` 已正确处理 |
| chosen/rejected 用了不同 prompt | 未共享 C' | `build_shared_context()` 确保所有 answer 共享 C' |
| rejected 太弱（全是胡言乱语） | 答案质量差 | 建议用 LLM 生成（`gen_hf.py`），而非简单截断 |
| 没有使用合成 position_ids | 直接用连续 0,1,2,... | `concatenated_forward` 会传入 position_ids |
| pad position_ids 用 -100 | position_ids 不支持负值 | `LOGODataCollator` 用 0 填充（由 attention_mask 掩码） |
