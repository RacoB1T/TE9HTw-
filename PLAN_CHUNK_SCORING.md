# Chunk 分割 & 打分修复计划

## 问题诊断

### Chunk 分割

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 1 | chunk 大小极度不均 | 单个事件（Physician note）可达 4000-17800 字，函数不拆分 | avg=1386 ± 2546, min=17, max=17861 |
| 2 | 长 chunk 天然高分 | 更多文字 = 更多关键词命中 | note chunk: 100+, lab chunk: 10 |
| 3 | 评分基准不一致 | 35 字 PTT chunk vs 4390 字 CT 报告 | 跨 chunk 比较无意义 |

### Chunk 打分

| # | 问题 | 原因 | 影响 |
|---|---|---|---|
| 4 | abnormal_results 不校验数值 | `creatinine|BUN` 权重 5，不检查值 | Cr 1.0（正常）= Cr 4.6（肾衰）同 5 分 |
| 5 | "is administered" 无差别匹配 | 几乎每个 med chunk 都命中 | CRRT 液袋排 #1 |
| 6 | 时间加成 +3 太弱 | 临床得分 100+ 面前无存在感 | 出院前关键信息进不了 critical |
| 7 | flowsheet-only 患者无叙事 chunk | 所有事件都是 lab/med/vital | 最高分 18 vs note 患者 144 |

---

## Part A: Chunk 分割修复

**目标**：所有 chunk 200-500 chars，大小均匀

**改动**：`data/convert_ds_to_logo.py` — `chunk_events()`

**方法**：两步分块，按 token 数（whitespace 估算，1 word ≈ 1.3 tokens）

1. 将每个事件的 TEXT 先按句子边界（句号、换行、分号）切成 segments，每段 < 40 tokens
2. 再将 segments 按 ~80 tokens 目标合并成 chunk，保持时间顺序
3. `chunk_size` 参数从 chars 改为 tokens，默认 80

```
Before (chars):
  event[7] = "Radiology note: ... (4390 chars)"  → chunk[7] = 4390 chars

After (tokens):
  event[7] → 按句号切 → 15 segments (~30 tokens each) → 合并成 ~6 chunks (~80 tokens each)
```

**token 估算**：
```python
def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)  # 1 word ≈ 1.3 BPE tokens
```

**需要处理的边界情况**：
- 没有句子边界的超长文本：按 40 tokens 硬切
- 短事件合并：多个 "PTT is 142. PT is 47." 合成一个 chunk
- 时间戳传递：每个 segment 继承原事件的时间戳
- `chunk_token_size` 在 `build_logo_dataset.py` 和 `convert_ds_to_logo.py` 中统一为同一单位

---

## Part B: Chunk 打分修复

**目标**：评分反映临床重要性，不受 chunk 长度影响

**改动**：`data/convert_ds_to_logo.py` — `CLINICAL_PATTERNS` + `score_chunk_clinical()`

### B1: 降低 "is administered" 权重

```
之前: weight=1, 每个 "X is administered" 计 1 分
之后: weight=0, 或者只对关键药物（抗生素、血管活性药、抗凝药）加 1 分
```

**理由**：CRRT 置换液 "Prismasate is administered" 不应和 "Ceftriaxone is administered" 同等对待。
常规给药是临床常态，不加分。仅当药物属于关键类别（抗生素/血管活性药/抗凝药/镇静药）时才 +1。

### B2: abnormal_results 加数值范围校验

```
之前: 有 "creatinine" 关键词 → +5
之后: 有 "creatinine" 且数值 > 1.2 → +5
     有 "creatinine" 且数值正常 → +1 (参考价值)
```

对常见 lab 提取数值，判断是否在正常范围内。异常 +5，正常 +1。

**实现**：给每个 `CLINICAL_PATTERN` 加一个可选的 `value_check` 函数：

```python
def _check_creatinine(text):
    m = re.search(r'Creatinine is (\d+\.?\d*)', text)
    if m: return float(m.group(1)) > 1.2
    return True  # 无法提取数值时默认为异常（避免漏掉）
```

### B3: 时间加成倍增

```
之前: 第一个和最后一个 chunk +3
之后: 第一个 chunk +5, 最后 10% 的 chunk 梯度加分（+1 到 +5）
```

**理由**：写出院小结时，医生一定会看最近的 lab/med 和出院计划。最后几个 chunk 应该获得更多加成。

### B4: 规范化临床得分（去长度偏置）

```
之前: final_score = clinical_score × (1 + ROUGE-1) + temporal_bonus
之后: final_score = (clinical_score / ln(chunk_tokens + 1)) × (1 + ROUGE-1) + temporal_bonus
```

用 `ln(chunk_tokens)` 归一化。80 token 的 chunk 和 10 token 的 chunk 在同等临床密度下得分接近。

### B5: 增加 narrative note 的识别

新增一个 pattern `narrative_note` 权重 12：

```python
(r"Physician  note:|Nursing note:|Radiology note:|General note:|"
 r"Chief Complaint:|HPI:|Assessment:|Plan:|24 Hour Events:", weight=12)
```

包含 narrative 的 chunk 基础分 +12，因为它们包含医生对病情的综合判断，比原始 lab 数据更有价值。

---

## 实施顺序

1. **Part A** — 改 `chunk_events()` 加句子级分割
2. **Part B1-B3** — 调整正则权重、增加数值校验、增加时间加成
3. **Part B4-B5** — 加长度归一化、narrative note 识别
4. 重建 DS_long 数据集 + 验证 scoring
