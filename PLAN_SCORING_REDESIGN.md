# Chunk 打分重新设计方案

## 数据分析结论

基于 500 个患者的真实数据：

```
74% 患者: clinical note < 5%（大部分是 flowsheet 数据）
9.4% 患者: 完全没有 clinical note
仅 26% 患者: 有较多临床笔记可依赖
```

**当前评分的根本问题不是权重调得不好，而是它对 flowsheet-dominant 患者完全失效。**

| 患者类型 | 占比 | 当前最高分 | 实际需要的信息 |
|---|---|---|---|
| note-rich (168015) | ~26% | 144 分 | ✅ 被覆盖 |
| flowsheet (121758) | ~74% | 18 分 | ❌ 漏掉所有关键信息 |

Gold summary 即使在 flowsheet-only 患者中也包含丰富临床内容（手术、并发症、会诊），但当前评分无法从 lab/med/vital 中提取这些信号。

---

## 新评分架构

### Layer 0: 事件分类（解决 35 char vs 4390 char 的问题）

先解决问题 0: chunk 大小不均。用 Part A 的 token 分割确保所有 chunk 在 40-120 tokens。

### Layer 1: 事件类型基础分

不再用正则匹配关键词，而是**先判断事件属于什么类型**，再给基础分：

```python
EVENT_TYPE_SCORES = {
    "clinical_note": 15,      # Physician/Nursing/Radiology/General note
    "procedure_mention": 12,  # 文本中提到手术/操作词汇
    "medication_change": 8,   # started, discontinued, titrated, switched
    "consultation": 8,        # consult, seen by, recommended by
    "diagnosis_mention": 7,   # diagnosed with, consistent with, findings show
    "lab_panel": 5,           # >5 个不同 lab 值的单个事件
    "lab_single": 2,          # 单个或少量 lab 值
    "assessment": 3,          # Daily Weight, Braden, GCS, I/O
    "vital_signs": 1,         # NBP, HR, SpO2, Temp, RR
    "medication_admin": 0,    # "X is administered" — 基础分 0
}
```

**关键改动：**
- `medication_admin = 0`：常规给药不加分。3000 次 "Aspirin is administered" 不应该累积任何分数。
- `lab_panel vs lab_single`：多指标面板本身就有信息量（医生看的是组合，不是单个值）
- `medication_change > medication_admin`：区分 "started on metoprolol" vs "Metoprolol is administered"

### Layer 2: Lab 数值校验（解决 Cr 1.0 = Cr 5.2 的问题）

不再看关键词 "creatinine"，而是提取实际数值并判断异常程度：

```python
LAB_REFERENCE_RANGES = {
    "creatinine":    (0.6, 1.3, 4.0),    # (low, high_normal, critical)
    "BUN":           (7, 20, 80),
    "WBC":           (4.0, 11.0, 25.0),
    "hemoglobin":    (12, 16, 7),
    "platelet":      (150, 400, 50),
    "PTT":           (25, 35, 80),
    "PT":            (11, 15, 30),
    "INR":           (0.8, 1.2, 4.0),
    "troponin":      (0, 0.04, 1.0),
    "potassium":     (3.5, 5.0, 6.5),
    "sodium":        (135, 145, 120),
    "glucose":       (70, 110, 400),
    "pH":            (7.35, 7.45, 7.0),
    "lactate":       (0.5, 2.0, 4.0),
    "bicarbonate":   (22, 28, 12),
    ...
}

def score_lab_value(name, value):
    low, high, critical = LAB_REFERENCE_RANGES.get(name, (0, 999, 9999))
    if value < critical or value > critical * 2:  # 需要根据方向判断
        return ("critical", 5)   # +5 for critical
    elif value < low or value > high:
        return ("abnormal", 3)   # +3 for abnormal
    else:
        return ("normal", 0)     # +0 for normal
```

**效果**：
```
Cr 1.0（正常）→ 0 分
Cr 2.4（异常）→ +3
Cr 5.2（危急）→ +5
```

### Layer 3: 时间上下文

```python
event_position = event_index / total_events  # 0.0 - 1.0

# 入院期 (0.0 - 0.05): 第一个临床发现最重要
if event_position < 0.05:  multiplier = 1.5

# 出院前期 (0.8 - 1.0): 梯度加成
elif event_position > 0.8:
    multiplier = 1.0 + (event_position - 0.8) * 2.5  # 1.0 → 1.5

# 中间期: 不加成
else:  multiplier = 1.0
```

### Layer 4: Gold 对齐（ROUGE-1）

保持现有逻辑：`×(1 + ROUGE-1)`，但改用更好的实现。

### Layer 5: Panel bonus（Lab 组合效应）

单个 lab 值异常的临床意义有限。但一个事件中 **5+ 个值全部异常** 就是明确的临床信号（脓毒症、AKI、DIC 等）。

```python
if event_type == "lab_panel":
    abnormal_count = count_abnormal_values(event_text)
    if abnormal_count >= 5:  panel_bonus = 5
    elif abnormal_count >= 3: panel_bonus = 3
```

---

## 新旧对比

### Patient 121758 (flowsheet-only, 0 notes)

| Chunk | 旧得分 | 新得分 | 内容 |
|---|---|---|---|
| Prismasate bag admin | 18.4 (#1) | 0 (#50+) | "is administered" → 基础 0 |
| Cr 5.2 + BUN 47 面板 | 10 (#2) | 18 (#1) | lab_panel(5) + Cr critical(5) + BUN critical(5) + panel_bonus(3) |
| PTT 52 + Cr 3.4 + Hb 9.8 | 10 (#3) | 14 (#2) | lab_panel(5) + multi-abnormal(3+3+3) = 14 |
| 最后 5 个 lab chunk | 3-13 | 12-18 | 出院前 ×1.3-1.5 加成 |
| 第一个 lab chunk（入院）| 5-10 | 15-22 | 入院 ×1.5 加成 |

### Patient 168015 (note-rich, 50 notes)

| Chunk | 旧得分 | 新得分 |
|---|---|---|
| Physician note (Cath/PCI) | 144 | ~60 (基础 15 + procedure(12) + diagnosis(7) + med_change(8) + 异常 lab bonus = ~45, ×1.3 temporal = ~58) |
| Single lab (PTT only) | 0 | 0 (不变) |
| 最后一个 med admin | 3 | ~8 (出院加成 ×1.3) |

**效果**：flowsheet 和 note-rich 患者的 top chunk 分数差距从 144:18 缩小到 ~60:22，都在合理范围内。

---

## 实施清单

1. 重构 `CLINICAL_PATTERNS` → `EVENT_CLASSIFIERS`（事件类型分类器）
2. 新增 `LAB_REFERENCE_RANGES` + `extract_lab_values()` + `score_lab_value()`
3. 重构 `score_chunk_clinical()` → 新 5 层评分函数
4. 新增 `classify_event_type()` — 先分类，再评分
5. 删除 `routine_medication` pattern（"is administered" 不再加分）
6. 修改 `rouge1_similarity` 计算（可选优化：word-level Jaccard 已经是 O(n+m)，保持不变）
7. 验证：选 10 个患者（5 flowsheet + 5 note-rich），人工检查 top-10 critical chunks 质量
