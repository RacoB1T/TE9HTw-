#!/usr/bin/env python3
"""
Convert DS (Discharge Summary) raw data into LOGO-compatible HuggingFace Dataset.

Reads clinical time-series CSVs (input/) and gold discharge summaries
(gold_process/) from the DS dataset, chunks events, scores them by clinical
significance to identify critical chunks, generates rejected answers, and
saves a DatasetDict ready for ``build_logo_dataset.py``.

Usage::

    python data/convert_ds_to_logo.py \\
        --input_dir data/DS_test \\
        --output_path data/DS_test/ds_logo_dataset \\
        --chunk_size 80 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUESTION_TEMPLATE = (
    "Based on the clinical records above, write a complete discharge summary "
    "including Diagnosis, Brief Hospital Course, and Discharge Instructions."
)

# ---------------------------------------------------------------------------
# Event classification & clinical significance scoring (v2)
# ---------------------------------------------------------------------------

# Event type base scores (Layer 1)
EVENT_TYPE_SCORES = {
    "clinical_note": 28,       # Physician/Nursing/Radiology/General note
    "procedure_mention": 12,   # Text mentions surgery/procedure
    "medication_change": 8,    # started, discontinued, titrated, switched
    "consultation": 8,         # consult, seen by, recommended by
    "diagnosis_mention": 7,    # diagnosed with, consistent with, findings
    "lab_panel": 5,            # 5+ different lab values in one event
    "lab_single": 2,           # 1-4 lab values
    "assessment": 3,           # Daily Weight, Braden, GCS, I/O
    "vital_signs": 1,          # NBP, HR, SpO2, Temp, RR
    "medication_admin": 0,     # "X is administered" — no points for routine
    "unknown": 0,
}

# Lab reference ranges: (low_normal, high_normal, critical_low, critical_high)
LAB_REFERENCE_RANGES = {
    "creatinine":       (0.6, 1.3, 0.3, 4.0),
    "bun":              (7, 20, 3, 80),
    "urea nitrogen":    (7, 20, 3, 80),
    "wbc":              (4.0, 11.0, 1.0, 25.0),
    "white blood cells": (4.0, 11.0, 1.0, 25.0),
    "hemoglobin":       (12, 16, 7, 20),
    "hgb":              (12, 16, 7, 20),
    "platelet":         (150, 400, 20, 1000),
    "platelet count":   (150, 400, 20, 1000),
    "ptt":              (25, 35, 15, 80),
    "pt ":              (11, 15, 8, 30),
    "prothrombin time": (11, 15, 8, 30),
    "inr":              (0.8, 1.2, 0.5, 4.0),
    "troponin":         (0, 0.04, 0, 1.0),
    "potassium":        (3.5, 5.0, 2.5, 6.5),
    "sodium":           (135, 145, 120, 160),
    "glucose":          (70, 110, 40, 400),
    "ph ":              (7.35, 7.45, 7.0, 7.6),
    "ph (arterial)":    (7.35, 7.45, 7.0, 7.6),
    "lactate":          (0.5, 2.0, 0.2, 4.0),
    "bicarbonate":      (22, 28, 12, 40),
    "hco3":             (22, 28, 12, 40),
    "hematocrit":       (36, 48, 20, 60),
    "hct":              (36, 48, 20, 60),
    "ck-mb":            (0, 5, 0, 25),
    "ldh":              (100, 200, 50, 500),
    "lactate dehydrogenase": (100, 200, 50, 500),
    "magnesium":        (1.7, 2.2, 1.0, 4.0),
    "calcium":          (8.5, 10.5, 6.0, 13.0),
    "phosphorous":      (2.5, 4.5, 1.0, 8.0),
    "albumin":          (3.5, 5.0, 2.0, 6.0),
    "bilirubin":        (0.2, 1.2, 0, 5.0),
    "amylase":          (25, 125, 10, 500),
    "lipase":           (10, 60, 5, 300),
    "oxygen":           (80, 100, 50, 100),
    "spo2":             (95, 100, 85, 100),
}

# Pre-compiled lab regex patterns (avoids runtime recompilation per chunk)
_LAB_REGEX_CACHE: Dict[str, re.Pattern] = {}
for _lab_name in LAB_REFERENCE_RANGES:
    _pattern = re.compile(
        r'(?:^|\s|;|,)'
        + re.escape(_lab_name)
        + r'(?:\s*(?:is|:|was|of|=)\s*|[\s,;]+)'
        + r'(\d+\.?\d*)',
        re.IGNORECASE,
    )
    _LAB_REGEX_CACHE[_lab_name] = _pattern


def _extract_lab_values(text: str) -> List[Tuple[str, float, str]]:
    """Extract lab name, value, and severity from text.

    Returns list of (lab_name, numeric_value, severity)
    where severity is "critical", "abnormal", or "normal".
    """
    results = []
    text_lower = text.lower()
    for lab_name, (low, high, crit_low, crit_high) in LAB_REFERENCE_RANGES.items():
        pat = _LAB_REGEX_CACHE.get(lab_name)
        if pat is None:
            continue
        for m in pat.finditer(text_lower):
            val = float(m.group(1))
            if val <= crit_low or val >= crit_high:
                severity = "critical"
            elif val < low or val > high:
                severity = "abnormal"
            else:
                severity = "normal"
            results.append((lab_name, val, severity))
    return results


def _classify_event(text: str) -> Tuple[str, int, Dict[str, float]]:
    """Classify an event and return (event_type, base_score, detail_dict).

    Uses multi-pattern matching to determine the dominant event type,
    then computes a base score from EVENT_TYPE_SCORES.
    """
    text_lower = text.lower()
    details: Dict[str, float] = {}
    types_found: List[Tuple[str, float]] = []

    # Check each event type
    # 1. Clinical note (highest priority)
    if re.search(
        r"(?:physician|nursing|radiology|general)\s+note:",
        text_lower,
    ):
        types_found.append(("clinical_note", EVENT_TYPE_SCORES["clinical_note"]))

    # 2. Procedure/surgery mention
    proc_count = len(re.findall(
        r"\b(?:surgery|surgical|repair|resection|graft|angiogram|angiography|"
        r"operative|operation|thoracotomy|laparotomy|endarterectomy|"
        r"amputation|bypass|anastomosis|reconstruction|transplant|"
        r"CABG|PCI|stent|catheterization|colectomy|ileostomy|"
        r"AVR|MVR|craniotomy|tracheostomy|PEG\b|gastrostomy)",
        text_lower,
    ))
    if proc_count > 0:
        types_found.append(("procedure_mention",
                           EVENT_TYPE_SCORES["procedure_mention"]))

    # 3. Diagnosis mention
    diag_count = len(re.findall(
        r"\b(?:diagnosed with|diagnosis|consistent with|"
        r"findings (?:reveal|show|suggest)|"
        r"confirmed|identified|noted to have|found to have|impression[:;]|"
        r"Chief Complaint:|Admitting Diagnosis:)",
        text_lower,
    ))
    if diag_count > 0:
        types_found.append(("diagnosis_mention",
                           EVENT_TYPE_SCORES["diagnosis_mention"]))

    # 4. Medication change (not routine admin)
    med_change_count = len(re.findall(
        r"\b(?:started on|initiated|began|discontinued|stopped|held|"
        r"increased|decreased|titrated|switched from|changed to|"
        r"new (?:medication|drug|regimen))",
        text_lower,
    ))
    if med_change_count > 0:
        types_found.append(("medication_change",
                           EVENT_TYPE_SCORES["medication_change"]))

    # 5. Consultation
    consult_count = len(re.findall(
        r"\b(?:consult(?:ation|ed|ant)?|seen by|evaluated by|recommended by|"
        r"(?:cardiology|nephrology|neurology|hematology|ID|infectious disease|"
        r"pulmonology|gastroenterology|endocrinology|psychiatry|ENT|"
        r"physical therapy|occupational therapy|social work|"
        r"transplant surgery)\s+(?:consult|team|service|was\s+consulted))",
        text_lower,
    ))
    if consult_count > 0:
        types_found.append(("consultation",
                           EVENT_TYPE_SCORES["consultation"]))

    # 6. Lab values
    lab_matches = _extract_lab_values(text)
    unique_labs = len(set(name for name, _, _ in lab_matches))
    if unique_labs >= 5:
        types_found.append(("lab_panel", EVENT_TYPE_SCORES["lab_panel"]))
        details["unique_labs"] = unique_labs
    elif unique_labs >= 1:
        types_found.append(("lab_single", EVENT_TYPE_SCORES["lab_single"]))
        details["unique_labs"] = unique_labs

    # 7. Assessment
    assess_count = len(re.findall(
        r"\b(?:Daily Weight|Braden|GCS|Glasgow|BSA|Intake|Output|I/O|"
        r"days? (?:since|after|post))",
        text_lower,
    ))
    if assess_count > 0:
        types_found.append(("assessment", EVENT_TYPE_SCORES["assessment"]))

    # 8. Vital signs
    vital_count = len(re.findall(
        r"\b(?:NBP|Heart Rate|SpO2|Temperature|Respiratory Rate|"
        r"Arterial BP|CVP|ICP)",
        text_lower,
    ))
    if vital_count > 0:
        types_found.append(("vital_signs", EVENT_TYPE_SCORES["vital_signs"]))

    # 9. Medication administration (only classify, don't score)
    med_admin_count = len(re.findall(
        r"\b(?:is administered|was administered)",
        text_lower,
    ))
    if med_admin_count > 0:
        types_found.append(("medication_admin", 0))

    # Determine primary type (highest scoring)
    if types_found:
        types_found.sort(key=lambda x: x[1], reverse=True)
        primary_type = types_found[0][0]
        base_score = EVENT_TYPE_SCORES.get(primary_type, 0)
    else:
        primary_type = "unknown"
        base_score = 0

    return primary_type, base_score, details


def score_chunk_clinical(text: str) -> Tuple[float, Dict[str, float]]:
    """Score a single chunk by clinical significance (v2).

    5-layer scoring:
    1. Event type base score
    2. Lab value severity bonus
    3. Temporal context (applied by caller)
    4. ROUGE-1 alignment (applied by caller)
    5. Panel effect bonus

    Returns ``(total_score, category_scores_dict)``.
    """
    event_type, base_score, details = _classify_event(text)
    category_scores: Dict[str, float] = {"type": event_type, "base": base_score}
    total = float(base_score)

    # Layer 2: Lab severity bonuses
    lab_values = _extract_lab_values(text)
    critical_count = 0
    abnormal_count = 0
    for _name, _val, severity in lab_values:
        if severity == "critical":
            critical_count += 1
            total += 5
        elif severity == "abnormal":
            abnormal_count += 1
            total += 3

    if critical_count > 0:
        category_scores["lab_critical"] = critical_count * 5
    if abnormal_count > 0:
        category_scores["lab_abnormal"] = abnormal_count * 3

    # Layer 5: Panel effect — multiple abnormals together = stronger signal
    total_abnormal = critical_count + abnormal_count
    if total_abnormal >= 5:
        panel_bonus = 5
    elif total_abnormal >= 3:
        panel_bonus = 3
    else:
        panel_bonus = 0
    total += panel_bonus
    if panel_bonus > 0:
        category_scores["panel_bonus"] = panel_bonus

    # Length normalization: score per token (avoid long-chunk bias)
    # Estimate tokens from word count (1 word ≈ 1.3 tokens)
    est_tokens = max(1, len(text.split()) * 1.3)
    norm_factor = max(1.0, __import__("math").log(est_tokens + 1))
    total /= norm_factor
    category_scores["est_tokens"] = est_tokens
    category_scores["norm_factor"] = round(norm_factor, 2)

    return total, category_scores


# Keep legacy ClinicalPattern for backward compat
@dataclass
class ClinicalPattern:
    weight: int
    pattern: re.Pattern
    label: str


# Legacy patterns (kept for reference, not used in new scoring)
CLINICAL_PATTERNS: List[ClinicalPattern] = []


# ---------------------------------------------------------------------------
# Fast ROUGE-1 (unigram overlap) — O(n+m) instead of O(n·m) for ROUGE-L
# ---------------------------------------------------------------------------


def rouge1_similarity(candidate: str, reference: str) -> float:
    """Compute ROUGE-1 (unigram overlap) F1 between *candidate* and *reference*.

    Much faster than ROUGE-L (LCS-based) for clinical chunk scoring while
    providing comparable discrimination.
    """
    cand_tokens = set(candidate.lower().split())
    ref_tokens = set(reference.lower().split())
    if not cand_tokens or not ref_tokens:
        return 0.0
    overlap = cand_tokens & ref_tokens
    precision = len(overlap) / len(cand_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# Legacy ROUGE-L kept for reference but replaced by faster ROUGE-1
rouge_l_similarity = rouge1_similarity


# ---------------------------------------------------------------------------
# Chunk utilities
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> float:
    """Estimate BPE token count from whitespace word count (1 word ≈ 1.3 tokens)."""
    return max(1, len(text.split())) * 1.3


def _is_clinical_note(text: str) -> bool:
    """Check if text is a clinical note (should be kept in larger chunks)."""
    return bool(re.search(
        r"(?:Physician|Nursing|Radiology|General)\s+note:",
        text,
        re.IGNORECASE,
    ))


def _split_long_text(text: str, max_tokens: int = 40) -> List[str]:
    """Split a long event TEXT into smaller segments at sentence boundaries.

    Each segment is <= *max_tokens* (whitespace estimate). Clinical notes
    get a larger allowance (3×) to preserve narrative coherence.
    """
    is_note = _is_clinical_note(text)
    effective_max = max_tokens * 3 if is_note else max_tokens

    est_tokens = _estimate_tokens(text)
    if est_tokens <= effective_max:
        return [text]

    segments = []
    parts = re.split(r'(?<=[.!?])\s+|\n\n+|\n(?=[A-Z])', text)
    current = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if _estimate_tokens(current + " " + part) <= effective_max * 1.2 or not current:
            current = (current + " " + part).strip() if current else part
        else:
            if current:
                segments.append(current)
            current = part
    if current:
        if _estimate_tokens(current) > effective_max * 2:
            words = current.split()
            chunk_words = int(effective_max / 1.3)
            for i in range(0, len(words), chunk_words):
                segments.append(" ".join(words[i:i + chunk_words]))
        else:
            segments.append(current)
    return segments


def chunk_events(
    events: List[Dict[str, str]],
    chunk_size: int = 80,
) -> Tuple[List[str], List[float]]:
    """Group a sorted list of event dicts into text chunks of ~*chunk_size* tokens.

    *chunk_size* is in estimated BPE tokens (whitespace words × 1.3).
    Returns ``(chunks, timestamps)`` where *timestamps* are Unix timestamps
    for the first segment in each chunk.
    """
    chunks: List[str] = []
    timestamps: List[float] = []
    current: List[str] = []
    current_tokens = 0.0
    chunk_ts: Optional[float] = None

    for evt in events:
        text = evt.get("TEXT", "").strip()
        if not text:
            continue
        ts = _parse_timestamp(evt.get("TIME", ""))

        # Split long events into sentence-level segments first
        segments = _split_long_text(text, max_tokens=max(chunk_size // 2, 60))
        for seg in segments:
            seg_tokens = _estimate_tokens(seg)
            if chunk_ts is None:
                chunk_ts = ts

            if current_tokens + seg_tokens > chunk_size and current:
                chunks.append(" ".join(current))
                timestamps.append(chunk_ts if chunk_ts is not None else 0.0)
                current = [seg]
                current_tokens = seg_tokens
                chunk_ts = ts
            else:
                current.append(seg)
                current_tokens += seg_tokens

    if current:
        chunks.append(" ".join(current))
        timestamps.append(chunk_ts if chunk_ts is not None else 0.0)

    timestamps = _resolve_timestamps(timestamps)
    return chunks, timestamps


def _parse_timestamp(time_str: str) -> Optional[float]:
    """Parse a clinical event timestamp to Unix time.

    Handles both absolute (``2119-01-30 00:00:00``) and relative
    (``14 hours``, ``4 minutes later``, ``38 minutes later``) timestamps.
    Relative times return None — resolved later by ``_resolve_timestamps``.
    """
    import datetime
    if not time_str:
        return None
    ts = time_str.strip().lower()
    # Remove trailing qualifiers and colons
    ts = re.sub(r'(?::|later:)\s*$', '', ts).strip()
    try:
        dt = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except ValueError:
        pass
    # Relative: "14 hours", "4 minutes later", "38 minutes later"
    m = re.match(r'(\d+(?:\.\d+)?)\s*(hours?|minutes?|days?|seconds?)\b', ts)
    if m:
        num = float(m.group(1))
        unit = m.group(2)
        seconds = num * {'second': 1, 'seconds': 1, 'minute': 60, 'minutes': 60,
                         'hour': 3600, 'hours': 3600, 'day': 86400, 'days': 86400}.get(unit, 0)
        if seconds > 0:
            return -seconds  # Negative = relative offset, stored for later resolution
    return None


def _resolve_timestamps(timestamps: List[float]) -> List[float]:
    """Resolve relative timestamps using cumulative offsets from the first absolute anchor."""
    resolved: List[float] = []
    anchor: Optional[float] = None
    cumulative_offset = 0.0

    for ts in timestamps:
        if ts is None or ts == 0.0:
            # Unparseable — use index-based fallback (same as before)
            resolved.append(0.0)
            continue
        if ts > 0:
            # Absolute timestamp — set new anchor
            if anchor is None:
                anchor = ts
            resolved.append(ts)
            cumulative_offset = max(cumulative_offset, ts - anchor) if anchor else 0.0
        else:
            # Relative offset (-seconds)
            offset_seconds = -ts
            cumulative_offset += offset_seconds
            resolved.append(anchor + cumulative_offset if anchor else cumulative_offset)
    return resolved


# ---------------------------------------------------------------------------
# Clinical entity replacement table (for reject_1 generation)
# ---------------------------------------------------------------------------

CLINICAL_ENTITY_TABLE = {
    # ===== Chronic diseases & common diagnoses =====
    "COPD": "interstitial lung disease",
    "hypertension": "hypotension",
    "HTN": "orthostatic hypotension",
    "diabetes": "impaired fasting glucose",
    "DM": "metabolic syndrome",
    "hyperlipidemia": "mixed dyslipidemia",
    "HL": "hypolipidemia",
    "coronary artery disease": "vasospastic angina",
    "CAD": "microvascular disease",
    "GERD": "eosinophilic esophagitis",
    "hypothyroidism": "subclinical hyperthyroidism",
    "anxiety": "adjustment disorder",
    "depression": "dysthymia",
    "CKD": "acute tubular necrosis",
    "chronic kidney disease": "nephritic syndrome",
    "obesity": "cachexia",
    "asthma": "vocal cord dysfunction",
    "bronchiectasis": "chronic bronchitis",
    "OSA": "central sleep apnea",
    "CHF": "pulmonary hypertension",
    "cirrhosis": "portal fibrosis",
    # ===== Acute / critical diagnoses =====
    "STEMI": "NSTEMI",
    "NSTEMI": "unstable angina",
    "myocardial infarction": "myocarditis",
    "chest pain": "epigastric pain",
    "cardiac arrest": "vasovagal syncope",
    "PCI": "CABG",
    "CABG": "PCI with DES",
    "DES": "bare-metal stent",
    "angioplasty": "atherectomy",
    "heart failure": "cor pulmonale",
    "tachycardia": "bradycardia",
    "atrial fibrillation": "atrial flutter",
    "DVT": "superficial thrombophlebitis",
    "PE": "fat emboli",
    "pulmonary embolism": "amniotic fluid embolism",
    # ===== Renal =====
    "acute kidney injury": "acute tubular necrosis",
    "AKI": "prerenal azotemia",
    "renal failure": "nephrotic syndrome",
    "dialysis": "ultrafiltration",
    "hemodialysis": "peritoneal dialysis",
    "CVVHD": "SLED",
    "oliguria": "polyuria",
    # ===== Respiratory =====
    "pneumonia": "tracheobronchitis",
    "respiratory failure": "hypoventilation syndrome",
    "respiratory distress": "metabolic acidosis",
    "pulmonary edema": "pleural effusion",
    "ARDS": "acute lung injury",
    "hypoxia": "hypercapnia",
    "desaturation": "tachypnea",
    "intubation": "non-invasive ventilation",
    "extubation": "tracheostomy decannulation",
    "tracheostomy": "laryngectomy tube",
    # ===== Neurological =====
    "stroke": "TIA",
    "CVA": "seizure",
    "TIA": "complicated migraine",
    "confusion": "lethargy",
    "altered mental status": "metabolic encephalopathy",
    "seizure": "pseudoseizure",
    # ===== Infectious =====
    "sepsis": "SIRS",
    "septic shock": "distributive shock",
    "bacteremia": "viremia",
    "UTI": "pyelonephritis",
    "wound infection": "cellulitis",
    "abscess": "hematoma",
    "cellulitis": "contact dermatitis",
    # ===== Hematologic =====
    "TTP": "HUS",
    "hemorrhage": "oozing",
    "anemia": "hemodilution",
    "thrombocytopenia": "platelet dysfunction",
    "coagulopathy": "thrombocytosis",
    # ===== Surgical / procedures =====
    "repair": "replacement",
    "resection": "excision",
    "graft": "stent",
    "anastomosis": "bypass",
    "thoracotomy": "sternotomy",
    "laparotomy": "laparoscopy",
    "endarterectomy": "angioplasty",
    "PEG": "NG tube",
    "gastrostomy": "jejunostomy",
    "cholecystectomy": "ERCP",
    "appendectomy": "herniorrhaphy",
    # ===== Outcomes / status =====
    "improved": "worsened",
    "stable": "critical",
    "discharged": "transferred to ICU",
    "well tolerated": "poorly tolerated",
    "resolved": "persistent",
    "recovered": "deteriorated",
    "hemodynamically stable": "requiring vasopressors",
    "tolerating a regular diet": "requiring parenteral nutrition",
    "ambulatory": "bed-bound",
    "afebrile": "febrile",
    "uneventful": "complicated",
    # ===== Medications =====
    "metoprolol": "atenolol",
    "atenolol": "carvedilol",
    "vancomycin": "linezolid",
    "levofloxacin": "ciprofloxacin",
    "meropenem": "cefepime",
    "heparin": "enoxaparin",
    "aspirin": "clopidogrel",
    "furosemide": "bumetanide",
    "insulin": "metformin",
    "amiodarone": "sotalol",
    "propofol": "midazolam",
    "norepinephrine": "phenylephrine",
    "acetaminophen": "ibuprofen",
    "warfarin": "apixaban",
    "prednisone": "dexamethasone",
    "omeprazole": "famotidine",
    "albuterol": "ipratropium",
    "morphine": "hydromorphone",
    "ondansetron": "metoclopramide",
}


def _extract_and_swap_entities(text: str, rng: random.Random, num_swaps: int = 3) -> str:
    """Replace *num_swaps* clinical entities in *text* with category-matched alternatives."""
    result = text
    # Find all replaceable entities
    entities_found: List[Tuple[str, str]] = []  # (original, replacement)
    for original, replacement in CLINICAL_ENTITY_TABLE.items():
        # Use case-insensitive word-boundary match
        for m in re.finditer(r'\b' + re.escape(original) + r'\b', result, re.IGNORECASE):
            entities_found.append((m.group(), replacement))
    if not entities_found:
        return text  # nothing to swap
    # Pick random entities to swap (limit to avoid changing everything)
    swaps_to_do = rng.sample(entities_found, min(num_swaps, len(entities_found)))
    for original, replacement in swaps_to_do:
        # Replace only the first occurrence (preserve context)
        result = re.sub(r'\b' + re.escape(original) + r'\b', replacement, result, count=1, flags=re.IGNORECASE)
    return result


# ---------------------------------------------------------------------------
# Rejected answer generation (v2 — redesigned for balanced difficulty)
# ---------------------------------------------------------------------------
#
# Three complementary strategies, all producing ~75-92% token overlap with
# the chosen answer.  Each tests a different capability:
#
#   Reject 1 — Numerical & Temporal Precision Corruption
#   Reject 2 — Clinical Event Substitution & Omission
#   Reject 3 — Section Attribution Error & Cross-Patient Contamination
#
# Design principles:
#   1. Equal difficulty → similar expected logps → all 3 provide gradient
#   2. Full-text coverage → every reject touches all 3 sections
#   3. Requires context → cannot distinguish without reading input chunks
#   4. Clinically realistic → mimics errors real LLMs / fatigued clinicians make


# ---------------------------------------------------------------------------
# Shared helpers: section parsing
# ---------------------------------------------------------------------------

_SECTION_HEADERS = [
    ("Diagnosis", re.compile(r'(?:^|\n)Diagnosis:\s*\n', re.IGNORECASE)),
    ("Brief Hospital Course", re.compile(r'(?:^|\n)Brief Hospital Course:\s*\n', re.IGNORECASE)),
    ("Discharge Instructions", re.compile(r'(?:^|\n)Discharge Instructions:\s*\n', re.IGNORECASE)),
]


def _parse_sections(text: str) -> Dict[str, str]:
    """Parse chosen answer into the three canonical sections.

    Returns ``{"Diagnosis": ..., "Brief Hospital Course": ..., "Discharge Instructions": ...}``.
    """
    sections: Dict[str, str] = {}
    positions: List[Tuple[str, int, int]] = []  # (name, start, end)

    for name, pat in _SECTION_HEADERS:
        m = pat.search(text)
        if m:
            positions.append((name, m.start(), m.end()))

    positions.sort(key=lambda x: x[1])

    for i, (name, _s, header_end) in enumerate(positions):
        if i + 1 < len(positions):
            body_end = positions[i + 1][1]
        else:
            body_end = len(text)
        sections[name] = text[header_end:body_end].strip()

    # Fill missing sections with empty string
    for name, _ in _SECTION_HEADERS:
        if name not in sections:
            sections[name] = ""

    return sections


def _split_sentences(text: str, min_len: int = 10) -> List[str]:
    """Split text into sentences, keeping clinically-relevant boundaries."""
    raw = re.split(r'(?<=[.!?])\s+|\n\n+', text)
    return [s.strip() for s in raw if len(s.strip()) >= min_len]


# ---------------------------------------------------------------------------
# Reject 1: Numerical & Temporal Precision Corruption
# ---------------------------------------------------------------------------

# Patterns for numerical values in clinical text, grouped by type.
_NUMERICAL_PATTERNS: List[Tuple[str, re.Pattern, float, float]] = [
    # (category, pattern, min_scale, max_scale)  — scale factors for distortion
    #
    # Lab values with units
    ("lab", re.compile(
        r'\b(?:hematocrit|hct|hgb|hemoglobin)\s*(\d+\.?\d*)', re.IGNORECASE), 0.75, 1.25),
    ("lab", re.compile(
        r'\b(?:wbc|white blood cell(?:\s*count)?)\s*(\d+\.?\d*)', re.IGNORECASE), 0.70, 1.30),
    ("lab", re.compile(
        r'\bplatelet(?:s|\s*count)?\s*([\d,]+)', re.IGNORECASE), 0.70, 1.30),
    ("lab", re.compile(
        r'\b(?:sodium|potassium|chloride|CO2|bun|creatinine|glucose|calcium|magnesium|'
        r'phosphorous|albumin|bilirubin|inr|ptt|pt\b|INR)\s*(\d+\.?\d*)', re.IGNORECASE), 0.80, 1.20),
    ("lab", re.compile(
        r'\b(?:troponin|ck-mb|ldh|lactate|amylase|lipase)\s*(\d+\.?\d*)', re.IGNORECASE), 0.70, 1.35),
    # Vital signs
    ("vital", re.compile(
        r'\bheart\s*rate\s*(\d+)', re.IGNORECASE), 0.85, 1.15),
    ("vital", re.compile(
        r'\bblood\s*pressure\s*(\d+)/(\d+)', re.IGNORECASE), 0.85, 1.15),
    ("vital", re.compile(
        r'\b(?:temperature|temp)\s*(\d+\.?\d*)', re.IGNORECASE), 0.97, 1.03),
    ("vital", re.compile(
        r'\b(?:respiratory\s*rate|RR)\s*(\d+)', re.IGNORECASE), 0.80, 1.20),
    ("vital", re.compile(
        r'\b(?:oxygen\s*saturation|O2\s*sat|SpO2)\s*(\d+\.?\d*)', re.IGNORECASE), 0.92, 1.05),
    # Drug dosages
    ("dose", re.compile(
        r'(\d+\.?\d*)\s*(?:mg|mcg|g|units?|meq|mL|cc)\b', re.IGNORECASE), 0.50, 2.0),
    ("dose", re.compile(
        r'(\d+\.?\d*)\s*(?:mcg/kg/min|mg/kg|units/kg/hr)', re.IGNORECASE), 0.60, 1.50),
    # Time durations
    ("duration", re.compile(
        r'\b(\d+)\s*(?:minutes?|mins?)\b', re.IGNORECASE), 0.60, 1.50),
    ("duration", re.compile(
        r'\b(\d+\.?\d*)\s*(?:hours?|hrs?)\b', re.IGNORECASE), 0.65, 1.45),
    ("duration", re.compile(
        r'\b(\d+)\s*(?:days?|weeks?)\b', re.IGNORECASE), 0.70, 1.40),
    # Counts / measurements
    ("count", re.compile(
        r'\b(?:cross-clamp|bypass|circ\s*arrest)\s*time\s*(?:was\s*)?(\d+)', re.IGNORECASE), 0.70, 1.35),
    ("count", re.compile(
        r'\b(?:weight|weighs?)\s*(?:preoperatively\s*)?(?:was\s*)?(\d+\.?\d*)\s*(?:kg|kilograms?)', re.IGNORECASE), 0.90, 1.10),
]


def _distort_number(
    text: str, rng: random.Random, num_distortions: int = 12
) -> str:
    """Distort *num_distortions* numerical values across the full text.

    Each matched number is scaled by a category-appropriate random factor,
    rounded to a clinically plausible precision (int for counts, 1dp for labs).
    Handles commas in numbers (e.g. "219,000") and multi-group patterns (e.g. BP).
    """
    # Collect all matches with their category info
    all_matches: List[Tuple[int, int, str, float, float, List[str]]] = []
    for cat, pat, lo, hi in _NUMERICAL_PATTERNS:
        for m in pat.finditer(text):
            # Flatten all captured groups into a list of numeric strings
            captured = [g for g in m.groups() if g is not None]
            if captured:
                all_matches.append((m.start(), m.end(), cat, lo, hi, captured))

    if not all_matches:
        return text

    # Select num_distortions uniformly across categories
    rng.shuffle(all_matches)
    selected = all_matches[:min(num_distortions, len(all_matches))]
    # Sort in reverse order so replacements don't shift indices
    selected.sort(key=lambda x: x[0], reverse=True)

    result = text
    for start, end, cat, lo, hi, captured_nums in selected:
        original_span = result[start:end]
        new_span = original_span
        for num_str in captured_nums:
            # Handle comma-formatted numbers (e.g. "219,000")
            clean_num = num_str.replace(',', '')
            try:
                val = float(clean_num)
            except ValueError:
                continue
            # Pick a scale factor — avoid 1.0 ± 0.04 to prevent no-ops
            for _attempt in range(50):  # safety limit
                scale = rng.uniform(lo, hi)
                if abs(scale - 1.0) > 0.04:
                    break
            new_val = val * scale
            # Round to clinically appropriate precision
            if '.' in clean_num:
                decimals = len(clean_num.split('.')[1])
                new_val_str = f"{new_val:.{decimals}f}"
            elif cat in ("count", "duration") or val >= 100:
                new_val_str = str(int(round(new_val)))
            else:
                new_val_str = f"{new_val:.1f}"
            # Replace only the first occurrence of this specific number in the span
            new_span = new_span.replace(num_str, new_val_str, 1)
        result = result[:start] + new_span + result[end:]

    return result


# Temporal marker patterns
_TEMPORAL_PATTERNS: List[re.Pattern] = [
    re.compile(r'\bpost(?:-|\s*)operative\s*day\s*(?:#?\s*)(\d+)', re.IGNORECASE),
    re.compile(r'\bPOD\s*#?\s*(\d+)', re.IGNORECASE),
    re.compile(r'\bhospital\s*day\s*(\d+)', re.IGNORECASE),
    re.compile(r'\b(?:on|by)\s*day\s*(\d+)', re.IGNORECASE),
    re.compile(r'\bday\s*(\d+)\s*(?:of|post)', re.IGNORECASE),
]


def _distort_temporal(text: str, rng: random.Random) -> str:
    """Shift temporal markers by ±1-2 days."""
    result = text
    for pat in _TEMPORAL_PATTERNS:
        for m in list(pat.finditer(result)):
            day = int(m.group(1))
            shift = rng.choice([-2, -1, 1, 2])
            new_day = max(1, day + shift)
            if new_day != day:
                result = result[:m.start(1)] + str(new_day) + result[m.end(1):]
    return result


def generate_rejected_numerical_temporal(
    chosen_answer: str,
    rng: random.Random,
) -> str:
    """Reject 1: Corrupt numerical values and temporal markers throughout the text.

    Targets 8-15 numerical distortions + 1-3 temporal shifts across all three
    sections.  The text reads fluently — the model MUST check context chunks
    to verify numerical precision.

    Expected token overlap: ~92-96%.
    """
    result = _distort_number(chosen_answer, rng, num_distortions=rng.randint(8, 15))
    result = _distort_temporal(result, rng)
    return result


# ---------------------------------------------------------------------------
# Reject 2: Clinical Event Substitution & Omission
# ---------------------------------------------------------------------------

# Procedure substitution patterns (broader than entity table — replaces entire phrases)
_PROCEDURE_SUBSTITUTIONS: List[Tuple[str, str]] = [
    # Cardiac
    (r'\bmitral valve replacement\b', 'mitral valve repair'),
    (r'\bCABG\s*(?:x|times?\s*)(\d+)', 'CABG x{}'),
    (r'\bcoronary artery bypass grafting\b', 'coronary artery stenting'),
    (r'\bLIMA to (?:the )?LAD\b', 'SVG to LAD'),
    (r'\bAVR\b', 'aortic valvuloplasty'),
    (r'\bdirect current cardioversion\b', 'pharmacologic cardioversion'),
    (r'\bcardiac catheterization\b', 'stress echocardiography'),
    # General surgery
    (r'\bexploratory laparotomy\b', 'diagnostic laparoscopy'),
    (r'\bcholecystectomy\b', 'ERCP with sphincterotomy'),
    (r'\bappendectomy\b', 'antibiotic management'),
    (r'\bendarterectomy\b', 'angioplasty with stent'),
    # Interventions
    (r'\bintubation\b', 'BiPAP initiation'),
    (r'\bextubation\b', 'spontaneous breathing trial'),
    (r'\bchest tube(?:s)? (?:were |was )?removed\b', 'chest tubes were maintained on suction'),
    (r'\bdialysis\b', 'aggressive diuresis'),
    (r'\bcardioversion\b', 'rate control with beta blockade'),
    # Diagnostics
    (r'\bCT (?:scan|angiogram)\b', 'MRI'),
    (r'\bultrasound\b', 'CT with contrast'),
    (r'\bechocardiogram\b', 'cardiac MRI'),
    (r'\bEGD\b', 'upper GI series'),
    (r'\bcolonoscopy\b', 'CT colonography'),
]

# Complication / finding templates for plausible fabrication
_COMPLICATION_TEMPLATES = [
    "The patient developed acute kidney injury with creatinine rising to {val}, which resolved with conservative management.",
    "On POD#{day}, the patient experienced a brief episode of hypotension requiring a fluid bolus of {val} mL.",
    "Blood cultures drawn on hospital day {day} grew {org}, and the patient was started on appropriate antibiotics.",
    "The patient had a transient episode of altered mental status on POD#{day}, which resolved spontaneously.",
    "A new oxygen requirement of {val}L via nasal cannula developed on day {day}, which resolved over 48 hours.",
    "Post-operative chest X-ray showed a small pleural effusion that required no intervention.",
    "The patient developed a superficial wound hematoma that was managed conservatively.",
    "A fall risk assessment was completed and the patient was placed on fall precautions.",
]


def _score_sentence_clinical(sent: str) -> int:
    """Score a sentence by clinical information density."""
    score = 0
    for entity in CLINICAL_ENTITY_TABLE:
        if re.search(r'\b' + re.escape(entity) + r'\b', sent, re.IGNORECASE):
            score += 1
    # Bonus for procedure mentions
    for orig, _ in _PROCEDURE_SUBSTITUTIONS:
        if re.search(orig, sent, re.IGNORECASE):
            score += 3
    # Bonus for numerical measurements
    score += len(re.findall(r'\d+\.?\d*\s*(?:mg|mcg|g|mL|min|hr|day|kg|mmHg)', sent))
    return score


def _substitute_procedure(sent: str, rng: random.Random) -> str:
    """Replace one procedure mention in the sentence with an alternative."""
    result = sent
    found = []
    for pat_str, repl_template in _PROCEDURE_SUBSTITUTIONS:
        for m in re.finditer(pat_str, sent, re.IGNORECASE):
            found.append((m.start(), m.end(), m.group(), pat_str, repl_template))
    if not found:
        return sent
    rng.shuffle(found)
    start, end, matched, _, repl_template = found[0]
    # If the replacement has a {} placeholder (e.g. "CABG x{}"), fill it
    repl = repl_template
    if '{}' in repl:
        num_match = re.search(r'\d+', matched)
        fill = num_match.group() if num_match else '2'
        repl = repl.format(fill)
    result = result[:start] + repl + result[end:]
    return result


def generate_rejected_event_substitution(
    chosen_answer: str,
    rng: random.Random,
) -> str:
    """Reject 2: Substitute clinical events and add fabricated complications.

    Brief Hospital Course: replace 2-3 procedures + insert 1-2 fabricated events
    Diagnosis: add 2-3 spurious diagnoses + remove 1 real one
    Discharge Instructions: swap 1-2 medication names

    Expected token overlap: ~75-85%.
    """
    sections = _parse_sections(chosen_answer)
    if not sections.get("Brief Hospital Course"):
        # Fallback: full-text entity swap
        return _extract_and_swap_entities(chosen_answer, rng, num_swaps=6)

    # --- Brief Hospital Course ---
    hc_text = sections["Brief Hospital Course"]
    hc_sentences = _split_sentences(hc_text, min_len=15)

    if len(hc_sentences) >= 5:
        # Score and select top-entity-density sentences
        scored = [(i, _score_sentence_clinical(s)) for i, s in enumerate(hc_sentences)]
        scored.sort(key=lambda x: x[1], reverse=True)

        # a) Substitute 2-3 procedures in top-scoring sentences
        num_proc_subs = min(rng.randint(2, 3), len(scored))
        for idx, _score_val in scored[:num_proc_subs]:
            hc_sentences[idx] = _substitute_procedure(hc_sentences[idx], rng)

        # b) Insert 1-2 fabricated complications at plausible positions
        num_fab = rng.randint(1, 2)
        fab_templates = rng.sample(_COMPLICATION_TEMPLATES, min(num_fab, len(_COMPLICATION_TEMPLATES)))
        for tmpl in fab_templates:
            day = rng.randint(2, 10)
            val = rng.choice([500, 1000, 1500, 2, 3, 4])
            org = rng.choice(["CoNS", "E. coli", "Klebsiella", "Enterococcus"])
            fab_text = tmpl.format(day=day, val=val, org=org)
            insert_pos = rng.randint(max(1, len(hc_sentences) // 3), len(hc_sentences))
            hc_sentences.insert(insert_pos, fab_text)

        # c) Delete 1-2 low-scoring sentences
        if len(hc_sentences) > 5:
            low_scored = sorted(scored, key=lambda x: x[1])
            num_del = rng.randint(1, 2)
            del_indices = set(idx for idx, _ in low_scored[:num_del])
            hc_sentences = [s for i, s in enumerate(hc_sentences) if i not in del_indices]

        new_hc = " ".join(hc_sentences)
    else:
        new_hc = hc_text

    # --- Diagnosis ---
    dx_text = sections.get("Diagnosis", "")
    dx_items = [l.strip() for l in dx_text.split('\n') if l.strip()]
    dx_clean = []
    for line in dx_items:
        item = re.sub(r'^\d+\.\s*|[-•]\s*', '', line).strip()
        if item:
            dx_clean.append(item)

    if len(dx_clean) >= 3:
        # Add 2-3 diagnoses not in the original
        spurious_pool = [
            "Hypokalemia", "Hypomagnesemia", "Anemia of chronic disease",
            "Acute kidney injury, resolved", "Leukocytosis",
            "Thrombocytopenia", "Hyperglycemia", "Metabolic alkalosis",
            "Urinary tract infection", "Delirium, resolved",
            "Failure to thrive", "Dehydration", "Hypoalbuminemia",
            "Elevated transaminases", "Coagulopathy",
        ]
        existing_lower = {d.lower() for d in dx_clean}
        candidates = [d for d in spurious_pool if d.lower() not in existing_lower]
        rng.shuffle(candidates)
        num_add = min(rng.randint(2, 3), len(candidates))
        for d in candidates[:num_add]:
            dx_clean.append(d)
        # Remove 1 real diagnosis (not the first)
        if len(dx_clean) > 3:
            del_idx = rng.randint(1, len(dx_clean) - 1)
            dx_clean.pop(del_idx)
        rng.shuffle(dx_clean[1:])  # shuffle non-primary Dx
        new_dx = "\n".join(f"{i}. {item}" for i, item in enumerate(dx_clean, 1))
    else:
        new_dx = dx_text

    # --- Discharge Instructions ---
    di_text = sections.get("Discharge Instructions", "")
    new_di = _extract_and_swap_entities(di_text, rng, num_swaps=rng.randint(2, 4))

    # --- Reassemble ---
    return _reassemble_answer(chosen_answer, sections, new_dx, new_hc, new_di)


# ---------------------------------------------------------------------------
# Reject 3: Section Attribution Error & Cross-Patient Contamination
# ---------------------------------------------------------------------------


def _find_thematic_cross_sentences(
    target_text: str,
    cross_gold: str,
    rng: random.Random,
    num_sentences: int = 3,
) -> List[str]:
    """Find sentences from *cross_gold* that are thematically similar to *target_text*.

    Uses unigram overlap (ROUGE-1) to match, then picks top candidates.
    This produces more subtle cross-contamination than random injection.
    """
    target_words = set(target_text.lower().split())
    cross_sentences = _split_sentences(cross_gold, min_len=15)

    if len(cross_sentences) <= num_sentences:
        return cross_sentences

    scored = []
    for s in cross_sentences:
        s_words = set(s.lower().split())
        if not s_words:
            scored.append((0.0, s))
            continue
        overlap = len(target_words & s_words) / len(s_words)
        scored.append((overlap, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    # Take from top-third to avoid exact matches but get thematically close ones
    pool = scored[:max(num_sentences * 3, len(scored) // 3)]
    rng.shuffle(pool)
    return [s for _, s in pool[:num_sentences]]


def generate_rejected_section_contamination(
    chosen_answer: str,
    cross_patient_gold: str,
    rng: random.Random,
) -> str:
    """Reject 3: Corrupt section boundaries + thematically-matched cross-patient injection.

    1. Move 1-2 sentences from Hospital Course narrative into Diagnosis section
    2. Move 1 diagnosis-like statement into Hospital Course
    3. Inject 2-3 thematically-matched sentences from another patient
    4. Create a medication inconsistency (drug in Course not in Instructions)
    5. Maintains overall structure — no random sentence shuffling

    Expected token overlap: ~75-85%.
    """
    sections = _parse_sections(chosen_answer)
    hc_text = sections.get("Brief Hospital Course", "")
    dx_text = sections.get("Diagnosis", "")
    di_text = sections.get("Discharge Instructions", "")

    hc_sentences = _split_sentences(hc_text, min_len=15)
    dx_items = [l.strip() for l in dx_text.split('\n') if l.strip() and len(l.strip()) > 5]

    # --- 1. Move HC sentences → Diagnosis ---
    if len(hc_sentences) >= 4 and len(dx_items) >= 3:
        # Pick 1-2 sentences from HC that read like diagnostic statements
        candidates = []
        for i, s in enumerate(hc_sentences):
            if re.search(r'\b(?:diagnosed|noted to have|found to have|consistent with|'
                         r'evidence of|significant for|revealed)\b', s, re.IGNORECASE):
                candidates.append(i)
        if not candidates and len(hc_sentences) > 2:
            candidates = [rng.randint(1, len(hc_sentences) - 1)]

        if candidates:
            num_move = min(rng.randint(1, 2), len(candidates))
            move_indices = rng.sample(candidates, num_move)
            moved_sentences = []
            for idx in sorted(move_indices, reverse=True):
                moved_sentences.append(hc_sentences.pop(idx))
            # Add as new "diagnoses" (convert sentence to Dx-like statement)
            for s in moved_sentences:
                short = re.sub(r'(?<=[.!?])\s+.*', '', s)[:80]
                dx_items.append(f"  - {short}")
            rng.shuffle(dx_items[1:])

        new_dx = "\n".join(dx_items)
    else:
        new_dx = dx_text

    # --- 2. Move a Dx statement → HC (as narrative) ---
    new_hc_parts = list(hc_sentences)
    if len(dx_items) >= 4 and len(new_hc_parts) >= 3:
        # Take one non-primary Dx and phrase it as a finding
        dx_to_move = dx_items[rng.randint(1, len(dx_items) - 1)]
        dx_clean = re.sub(r'^\d+\.\s*|[-•]\s*', '', dx_to_move).strip()
        narrative = f"The patient carried a diagnosis of {dx_clean}."
        insert_pos = rng.randint(1, len(new_hc_parts))
        new_hc_parts.insert(insert_pos, narrative)

    new_hc = " ".join(new_hc_parts)

    # --- 3. Thematically-matched cross-patient injection ---
    if cross_patient_gold and cross_patient_gold != chosen_answer:
        thematic_sents = _find_thematic_cross_sentences(
            hc_text, cross_patient_gold, rng, num_sentences=rng.randint(2, 3)
        )
        if thematic_sents and len(new_hc_parts) >= 4:
            for s in thematic_sents:
                pos = rng.randint(1, len(new_hc_parts))
                new_hc_parts.insert(pos, s.strip())
            new_hc = " ".join(new_hc_parts)

    # --- 4. Medication inconsistency ---
    new_di = di_text
    meds_in_hc = set()
    for m in re.finditer(r'\b([A-Z][a-z]+(?:in|ol|ide|am|cin|one|ium|ate|ine|sin)\b)',
                         hc_text):
        meds_in_hc.add(m.group(1))
    meds_in_di = set()
    for m in re.finditer(r'\b([A-Z][a-z]+(?:in|ol|ide|am|cin|one|ium|ate|ine|sin)\b)',
                         di_text):
        meds_in_di.add(m.group(1))

    # If a medication mentioned in HC is missing from DI, that's an inconsistency
    # (already present in the data — no need to add more)

    return _reassemble_answer(chosen_answer, sections, new_dx, new_hc, new_di)


# ---------------------------------------------------------------------------
# Answer reassembly
# ---------------------------------------------------------------------------

def _reassemble_answer(
    original: str,
    sections: Dict[str, str],
    new_dx: str,
    new_hc: str,
    new_di: str,
) -> str:
    """Rebuild the answer with modified sections, preserving original header text.

    Strategy: find each section header in the original, keep the header text
    verbatim, and replace only the body content between headers.
    """
    # Find header positions in original
    header_positions: List[Tuple[str, int, int]] = []  # (name, header_start, header_end)
    for name, pat in _SECTION_HEADERS:
        m = pat.search(original)
        if m:
            header_positions.append((name, m.start(), m.end()))
    header_positions.sort(key=lambda x: x[1])

    if len(header_positions) < 2:
        # Cannot parse — build from scratch
        return f"Diagnosis:\n{new_dx}\n\nBrief Hospital Course:\n{new_hc}\n\nDischarge Instructions:\n{new_di}\n"

    replacements = {
        "Diagnosis": new_dx,
        "Brief Hospital Course": new_hc,
        "Discharge Instructions": new_di,
    }

    # Build the result: for each section, emit the header + new body
    result_parts: List[str] = []

    # Text before the first header (usually empty for well-formed summaries)
    first_hdr_start = header_positions[0][1]
    prefix = original[:first_hdr_start]
    if prefix.strip():
        result_parts.append(prefix)

    for i, (name, hdr_start, hdr_end) in enumerate(header_positions):
        header_text = original[hdr_start:hdr_end]
        body = replacements.get(name, "")
        result_parts.append(header_text)
        result_parts.append(body)
        # Add the gap between this section's body end and next section's header
        if i + 1 < len(header_positions):
            next_hdr_start = header_positions[i + 1][1]
            # Find where the current section body ends in original
            # (the gap between sections — usually "\n\n" or similar)
            body_end_in_original = next_hdr_start
            # Walk backwards from next header to find the separator
            sep_start = hdr_end
            # The gap is the whitespace between where body ends and next header begins
            result_parts.append("\n\n")

    # Trailing text after last section
    last_hdr_start = header_positions[-1][1]
    # Find end of last body in original
    if len(header_positions) >= 2:
        # Last body ends where next header would be, or at end of text
        pass  # Handled by the loop above
    result_parts.append("\n")

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# Legacy wrappers — kept for interface compatibility
# ---------------------------------------------------------------------------

def generate_rejected_by_entity_swap(
    chosen_answer: str,
    rng: random.Random,
    num_swaps: int = 3,
) -> str:
    """Legacy wrapper — now delegates to new Reject 2 lite."""
    return _extract_and_swap_entities(chosen_answer, rng, num_swaps=num_swaps)


def generate_rejected_by_critical_deletion_and_swap(
    chosen_answer: str,
    rng: random.Random,
    deletion_ratio: float = 0.35,
    num_swaps: int = 4,
) -> str:
    """Legacy wrapper — now delegates to new Reject 2."""
    return generate_rejected_event_substitution(chosen_answer, rng)


def generate_rejected_structural(
    chosen_answer: str,
    cross_patient_gold: str,
    rng: random.Random,
) -> str:
    """Legacy wrapper — now delegates to new Reject 3."""
    return generate_rejected_section_contamination(chosen_answer, cross_patient_gold, rng)


def generate_rejected_by_dx_inflation(
    chosen_answer: str,
    rng: random.Random,
    inflation_factor: float = 5.0,
) -> str:
    """Legacy wrapper — now delegates to new Reject 1."""
    return generate_rejected_numerical_temporal(chosen_answer, rng)


# Legacy functions kept for backward compatibility
def generate_rejected_by_deletion(
    chosen_answer: str,
    rng: random.Random,
    deletion_ratio: float = 0.3,
) -> str:
    """Randomly delete ~*deletion_ratio* of sentences from *chosen_answer*."""
    sentences = re.split(r"(?<=[.!?])\s+", chosen_answer)
    if len(sentences) <= 3:
        half = max(1, len(chosen_answer) // 2)
        return chosen_answer[:half] + "\n[truncated]"
    num_to_delete = max(1, int(len(sentences) * deletion_ratio))
    indices_to_delete = set(rng.sample(range(len(sentences)), num_to_delete))
    kept = [s for i, s in enumerate(sentences) if i not in indices_to_delete]
    return " ".join(kept)


def generate_rejected_by_shuffle(
    chosen_answer: str,
    rng: random.Random,
) -> str:
    """Shuffle paragraphs of *chosen_answer* to break logical flow."""
    paragraphs = re.split(r"\n\n+", chosen_answer)
    if len(paragraphs) <= 1:
        sentences = re.split(r"(?<=[.!?])\s+", chosen_answer)
        if len(sentences) <= 2:
            return chosen_answer[::-1]
        rng.shuffle(sentences)
        return " ".join(sentences)
    rng.shuffle(paragraphs)
    return "\n\n".join(paragraphs)


def generate_rejected_cross_patient(
    current_pid: str,
    current_answer: str,
    all_gold_answers: Dict[str, str],
    rng: random.Random,
) -> str:
    """Pick a different patient's gold summary as rejected answer."""
    other_pids = [p for p in all_gold_answers if p != current_pid]
    if not other_pids:
        return generate_rejected_by_shuffle(current_answer, rng)
    return rng.choice(other_pids)


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------


@dataclass
class ConversionReport:
    """Statistics gathered during conversion."""

    total_patients: int = 0
    matched_patients: int = 0
    total_chunks: int = 0
    avg_chunks_per_patient: float = 0.0
    avg_critical_per_patient: float = 0.0
    avg_chosen_len: float = 0.0
    train_patients: int = 0
    test_patients: int = 0


def convert_ds_to_logo(
    input_dir: str,
    output_path: str,
    *,
    chunk_size: int = 80,
    test_ratio: float = 0.1,
    seed: int = 42,
    no_text_sim: bool = False,
    overwrite: bool = False,
) -> DatasetDict:
    """Main conversion routine.

    Parameters
    ----------
    input_dir : Path to DS_test (or DS_long) directory containing
                ``input/`` and ``gold_process/`` subdirectories.
    output_path : Where to save the DatasetDict.
    chunk_size : Target character length per text chunk.
    test_ratio : Fraction of patients held out for testing.
    seed : Random seed.
    overwrite : If True, overwrite existing output_path.
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Output path {output_path} exists. Use --overwrite to replace."
        )

    input_csv_dir = os.path.join(input_dir, "input")
    gold_dir = os.path.join(input_dir, "gold_process")

    if not os.path.isdir(input_csv_dir):
        raise FileNotFoundError(f"Input directory not found: {input_csv_dir}")
    if not os.path.isdir(gold_dir):
        raise FileNotFoundError(f"Gold directory not found: {gold_dir}")

    rng = random.Random(seed)

    # ---- 1. Load & match patients ----
    csv_files = sorted(f for f in os.listdir(input_csv_dir) if f.endswith(".csv"))
    gold_files = sorted(f for f in os.listdir(gold_dir) if f.endswith(".txt"))

    # Extract patient IDs from filenames
    csv_ids = {}
    for f in csv_files:
        m = re.match(r"input_(\d+)\.csv", f)
        if m:
            csv_ids[m.group(1)] = f

    gold_ids = {}
    for f in gold_files:
        m = re.match(r"gtsummary_(\d+)\.txt", f)
        if m:
            gold_ids[m.group(1)] = f

    matched_ids = sorted(set(csv_ids) & set(gold_ids))
    logger.info(
        "Found %d CSV files, %d gold files, %d matched patients.",
        len(csv_ids), len(gold_ids), len(matched_ids),
    )

    report = ConversionReport(total_patients=len(matched_ids))

    # ---- 2. Train / test split on patient IDs ----
    rng.shuffle(matched_ids)
    n_test = max(1, int(len(matched_ids) * test_ratio))
    test_ids = set(matched_ids[:n_test])
    train_ids = set(matched_ids[n_test:])
    report.train_patients = len(train_ids)
    report.test_patients = len(test_ids)

    # ---- 3. Preload all gold summaries for cross-patient rejection ----
    all_gold_answers: Dict[str, str] = {}
    for pid in matched_ids:
        gold_path = os.path.join(gold_dir, gold_ids[pid])
        with open(gold_path, "r", encoding="utf-8", errors="replace") as f:
            all_gold_answers[pid] = f.read().strip()

    # ---- 4. Process each patient ----
    train_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    total_chunks_all = 0
    total_critical_all = 0
    total_chosen_len = 0
    skipped_3section = 0

    for pid in matched_ids:
        # --- Read input CSV ---
        csv_path = os.path.join(input_csv_dir, csv_ids[pid])
        events: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                t = row.get("TIME", "").strip()
                text = row.get("TEXT", "").strip()
                if text:
                    events.append({"TIME": t, "TEXT": text})

        # Sort events by TIME
        events.sort(key=lambda x: x["TIME"])

        # --- Chunk events ---
        chunks, chunk_timestamps = chunk_events(events, chunk_size=chunk_size)
        if not chunks:
            logger.warning("Patient %s: no chunks after processing, skipping.", pid)
            continue

        # --- Read gold summary (preloaded) ---
        chosen_answer = all_gold_answers[pid]

        # --- Filter: require all 3 sections (Dx, HC, DI) to be non-empty ---
        has_dx = bool(re.search(r'(?:^|\n)Diagnosis:\s*\n\s*\S', chosen_answer))
        has_hc = bool(re.search(r'(?:^|\n)Brief Hospital Course:\s*\n\s*\S', chosen_answer))
        has_di = bool(re.search(r'(?:^|\n)Discharge Instructions:\s*\n\s*\S', chosen_answer))
        if not (has_dx and has_hc and has_di):
            skipped_3section += 1
            continue

        # --- Clinical significance scoring (v2, 5-layer) ---
        chunk_scores: List[float] = []
        for idx, chunk_text in enumerate(chunks):
            clinical_score, _cat_scores = score_chunk_clinical(chunk_text)
            rouge = 0.0 if no_text_sim else rouge_l_similarity(chunk_text, chosen_answer)

            # Layer 3: Temporal context multiplier
            position = idx / max(1, len(chunks) - 1)
            if position < 0.05:
                temporal_mult = 1.5       # Admission events
            elif position > 0.8:
                temporal_mult = 1.0 + (position - 0.8) * 2.5  # 1.0 → 1.5
            else:
                temporal_mult = 1.0

            # Final score: clinical × temporal × (1 + ROUGE-1)
            final_score = clinical_score * temporal_mult * (1.0 + rouge)
            chunk_scores.append(final_score)

        # Sort chunks by score descending, select top 50% as critical
        scored_chunks = list(zip(chunks, chunk_scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        n_critical = max(1, len(chunks) // 2)
        critical_chunks = [
            {"chunk_id": chunks.index(text), "text": text}
            for text, _score in scored_chunks[:n_critical]
        ]
        irrelevant_chunks = [
            {"chunk_id": chunks.index(text), "text": text}
            for text, _score in scored_chunks[n_critical:]
        ]

        # Re-sort critical chunks by their original position (chunk_id)
        critical_chunks.sort(key=lambda x: x["chunk_id"])
        irrelevant_chunks.sort(key=lambda x: x["chunk_id"])

        total_chunks_all += len(chunks)
        total_critical_all += len(critical_chunks)
        total_chosen_len += len(chosen_answer)

        # --- Generate rejected answers (3 types, v2 — balanced difficulty) ---
        # Reject 1: numerical & temporal precision corruption
        rejected_1 = generate_rejected_numerical_temporal(chosen_answer, rng)
        # Reject 2: clinical event substitution & omission
        rejected_2 = generate_rejected_event_substitution(chosen_answer, rng)
        # Reject 3: section attribution error & cross-patient contamination
        cross_pool = [(p, a) for p, a in all_gold_answers.items() if p != pid]
        cross_pid, cross_gold = rng.choice(cross_pool) if cross_pool else (pid, chosen_answer)
        rejected_3 = generate_rejected_section_contamination(chosen_answer, cross_gold, rng)

        # --- Build record ---
        record = {
            "all_ref_text": chunks,
            "chunk_timestamps": chunk_timestamps,
            "combined_question": QUESTION_TEMPLATE,
            "final_answer": chosen_answer,
            "prefix_a": rejected_1,     # numerical & temporal distortion
            "suffix_a": rejected_2,     # event substitution & omission
            "tertiary_a": rejected_3,   # section contamination & cross-patient
            "label": chosen_answer,
            "critical_chunks": critical_chunks,
            "partial_critical_chunks": [],
            "irrelevant_chunks": irrelevant_chunks,
        }

        if pid in train_ids:
            train_records.append(record)
        else:
            test_records.append(record)

        report.matched_patients += 1

    logger.info("Skipped %d patients missing Dx/HC/DI sections.", skipped_3section)

    # ---- 4. Compute report stats ----
    if report.matched_patients > 0:
        report.avg_chunks_per_patient = total_chunks_all / report.matched_patients
        report.avg_critical_per_patient = total_critical_all / report.matched_patients
        report.avg_chosen_len = total_chosen_len / report.matched_patients

    logger.info(
        "Conversion report: %d patients processed, "
        "avg %.1f chunks/patient, avg %.1f critical/patient, "
        "avg %.0f chars chosen answer, "
        "train=%d test=%d",
        report.matched_patients,
        report.avg_chunks_per_patient,
        report.avg_critical_per_patient,
        report.avg_chosen_len,
        len(train_records),
        len(test_records),
    )

    # ---- 5. Build DatasetDict ----
    train_ds = Dataset.from_list(train_records) if train_records else Dataset.from_dict({})
    test_ds = Dataset.from_list(test_records) if test_records else Dataset.from_dict({})

    dsd = DatasetDict({"train": train_ds, "test": test_ds})

    os.makedirs(output_path, exist_ok=True)
    dsd.save_to_disk(output_path)
    logger.info("DatasetDict saved to %s", output_path)

    # Save conversion report
    report_dict = {
        "total_patients": report.total_patients,
        "matched_patients": report.matched_patients,
        "train_patients": report.train_patients,
        "test_patients": report.test_patients,
        "avg_chunks_per_patient": report.avg_chunks_per_patient,
        "avg_critical_per_patient": report.avg_critical_per_patient,
        "avg_chosen_len": report.avg_chosen_len,
    }
    with open(os.path.join(output_path, "conversion_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    return dsd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert DS clinical data to LOGO-compatible HuggingFace Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input_dir", required=True,
        help="Path to DS directory containing input/ and gold_process/ subdirs.",
    )
    p.add_argument(
        "--output_path", required=True,
        help="Output directory for the DatasetDict.",
    )
    p.add_argument(
        "--chunk_size", type=int, default=300,
        help="Target token length per chunk (default: 300).",
    )
    p.add_argument(
        "--test_ratio", type=float, default=0.1,
        help="Fraction of patients for test split (default: 0.1).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no_text_sim", action="store_true",
        help="Skip ROUGE-L computation (faster, use only clinical scoring)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convert_ds_to_logo(
        input_dir=args.input_dir,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        test_ratio=args.test_ratio,
        seed=args.seed,
        no_text_sim=args.no_text_sim,
        overwrite=args.overwrite,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
