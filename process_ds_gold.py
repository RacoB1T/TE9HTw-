"""
Improved extraction of Diagnosis (Dx), Hospital Course (HC), and
Discharge Instructions (DI) from MIMIC-III discharge summaries.

Key fixes over the original evaluate_ds.py logic:
1. End markers searched ONLY after start markers
2. Added missing patterns: DISCHARGE DIAGNOSES (plural), FINAL DIAGNOSES,
   PRIMARY DIAGNOSIS, POSTOPERATIVE DIAGNOSES, de-identified variants
3. Discharge Disposition added to end patterns
4. Hospital Course variants (BY SYSTEM, BY DATES, TO DATE, HOSPITAL PROGRESS AND COURSE, de-id)
5. DI: DISCHARGE INSTRUCTIONS (all-caps), DISCHARGE PLANS, de-identified variants,
   semicolon/no-punctuation meds variants, Followup Instructions as fallback
6. Name markers ([**First Name, [**Name) in all end patterns
7. DIAGNOSIS/DIAGNOSES as fallback when near end of document
"""

import re
import os
import sys

sys.setrecursionlimit(10000)

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

DIAGNOSIS_START_PATTERNS = [
    # Standard discharge diagnosis headers
    r"Discharge Diagnosis:",
    r"DISCHARGE DIAGNOSIS:",
    r"DISCHARGE DIAGNOSES:",
    r"DISCHARGE DIAGNOSES\.:",             # typo: period before colon
    r"DISCHARGED DIAGNOSES:",
    r"FINAL DIAGNOSIS:",
    r"FINAL DIAGNOSES:",
    r"FINAL DIAGNOSIS/PROBLEM LIST:",
    r"DIAGNOSIS ON DISCHARGE:",
    r"DISCHARGE DIAGNOSIS/CPT CODES:",
    # Other diagnosis headers (used in some summaries as discharge dx)
    r"PRIMARY DIAGNOSIS:",
    r"Primary diagnosis:",
    r"Primary Diagnoses:",
    r"POSTOPERATIVE DIAGNOSES:",
    r"POSTOPERATIVE DIAGNOSIS:",
    # De-identified: e.g. [**Location (un) **] Diagnosis:
    r"\[\*\*[^*]+\*\*\]\s*Diagnosis:",
]

DIAGNOSIS_END_PATTERNS = [
    r"Discharge Condition:",
    r"CONDITION ON DISCHARGE:",
    r"Discharge Disposition:",
    r"DISCHARGE STATUS:",
    r"Discharge Instructions:",
    r"DISCHARGE INSTRUCTIONS:",
    r"DISCHARGE INSTRUCTIONS/FOLLOWUP:",
    r"DISCHARGE INSTRUCTIONS-FOLLOWUP:",
    r"General Discharge Instructions:",
    r"Followup Instructions:",
    r"FOLLOW-UP INSTRUCTIONS:",
    r"FOLLOW-UP PLANS:",
    r"FOLLOW-UP APPOINTMENT:",
    r"FOLLOW-UP:",
    r"RECOMMENDED FOLLOWUP:",
    r"RECOMMENDED FOLLOW-UP",
    r"Discharge Medications:",
    r"DISCHARGE MEDICATIONS:",
    r"MEDICATIONS ON DISCHARGE:",
    r"MEDICATIONS AT THE TIME OF DISCHARGE:",
    r"MEDICATIONS AT TIME OF DISCHARGE:",
    r"Discharge Labs:",
    r"DISCHARGE LABORATORY DATA:",
    r"Discharge Physical Exam:",
    r"DISCHARGE PHYSICAL EXAMINATION:",
    r"Discharge Disposition:",
    r"DISCHARGE DISPOSITION:",
    # De-identification fallbacks
    r"\[\*\*First Name",
    r"\[\*\*Name",
]

HOSPITAL_COURSE_START_PATTERNS = [
    r"Brief Hospital Course:",
    r"BRIEF HOSPITAL COURSE BY DATES:",
    r"BRIEF SUMMARY OF HOSPITAL COURSE",
    r"HOSPITAL COURSE:",                  # standard
    r"HOSPITAL COURSE\s*[-:]",            # dash or colon: HOSPITAL COURSE - / HOSPITAL COURSE:
    r"HOSPITAL PROGRESS AND COURSE:",     # compound header
    r"HOSPITAL COURSE BY SYSTEM",
    r"HOSPITAL COURSE BY REVIEW",
    r"HOSPITAL COURSE BY ISSUE",
    r"HOSPITAL COURSE BY PROBLEM",
    r"HOSPITAL COURSE TO",
    r"HOSPITAL COURSE SUMMARY",
    r"HOSPITAL COURSE WHILE",
    r"HOSPITAL COURSE ON",
    r"SUMMARY OF HOSPITAL COURSE",
    r"CONCISE SUMMARY OF HOSPITAL COURSE",
    r"Hospital course by problem:",
    r"Hospital Course by Problem:",
    r"Hospital Course by System:",
    r"Hospital Course:",
    r"Hospital Course Summary:",
    r"Hospital Course",
    # De-identified: e.g. HO[**Last Name ...**] COURSE: (= HOSPITAL COURSE:)
    r"HO\[\*\*[^*]+\*\*\]\s*COURSE",
]

HOSPITAL_COURSE_END_PATTERNS = [
    r"Discharge Medications:",
    r"DISCHARGE MEDICATIONS:",
    r"MEDICATIONS ON DISCHARGE:",
    r"MEDICATIONS AT THE TIME OF DISCHARGE:",
    r"MEDICATIONS AT TIME OF DISCHARGE:",
    r"DISCHARGE STATUS:",
    r"Discharge Disposition:",
    r"DISCHARGE DISPOSITION:",
    r"Medications on Admission:",
    r"MEDICATIONS ON ADMISSION:",
    r"CONDITION ON DISCHARGE:",
    r"Discharge Condition:",
    r"DISCHARGE DIAGNOSIS:",
    r"DISCHARGE DIAGNOSES:",
    r"Discharge Diagnosis:",
    r"FINAL DIAGNOSIS:",
    r"FINAL DIAGNOSES:",
    r"PRIMARY DIAGNOSIS:",
    r"Discharge Instructions:",
    r"DISCHARGE INSTRUCTIONS:",
    r"DISCHARGE INSTRUCTIONS/FOLLOWUP:",
    r"DISCHARGE INSTRUCTIONS-FOLLOWUP:",
    r"Discharge Labs:",
    r"DISCHARGE LABORATORY DATA:",
    r"Discharge Physical Exam:",
    r"DISCHARGE PHYSICAL EXAMINATION:",
    r"Followup Instructions:",
    r"FOLLOW-UP INSTRUCTIONS:",
    r"FOLLOW-UP PLANS:",
    # De-identification
    r"\[\*\*First Name",
    r"\[\*\*Name",
]

DISCHARGE_INSTRUCTIONS_START_PATTERNS = [
    r"Discharge Instructions:",
    r"DISCHARGE INSTRUCTIONS:",
    r"DISCHARGE INSTRUCTIONS/FOLLOWUP:",
    r"DISCHARGE INSTRUCTIONS-FOLLOWUP:",
    r"DISCHARGE INSTRUCTIONS/FOLLOWUP PLANS:",
    r"DISCHARGE PLAN:",
    r"DISCHARGE PLANS:",
    r"RECOMMENDED FOLLOWUP:",
    r"General Discharge Instructions:",
    r"FOLLOW-UP INSTRUCTIONS:",
    r"Followup Instructions:",          # fallback when no explicit DI header
    # De-identified DISCHARGE INSTRUCTIONS: e.g. DI[**Last Name ...**]E INSTRUCTIONS:
    r"DI\[\*\*[^*]+\*\*\]E\s+INSTRUCTIONS:",
    # De-identified DISCHARGE PLAN: e.g. DI[**...**] PLAN:
    r"DI\[\*\*[^*]+\*\*\]\s+PLAN:",
]

DISCHARGE_INSTRUCTIONS_END_PATTERNS = [
    r"Followup Instructions:",
    r"FOLLOW-UP INSTRUCTIONS:",
    r"RECOMMENDED FOLLOW-UP",
    r"FOLLOW-UP PLANS:",
    r"FOLLOW-UP:",
    r"Discharge Condition:",
    r"CONDITION ON DISCHARGE:",
    r"Discharge Disposition:",
    r"DISCHARGE STATUS:",
    # De-identification fallbacks
    r"\[\*\*First Name",
    r"\[\*\*Name",
]

DISCHARGE_MEDICATIONS_START_PATTERNS = [
    r"Discharge Medications:",
    r"DISCHARGE MEDICATIONS:",
    r"DISCHARGE MEDICATIONS;",            # semicolon variant
    r"DISCHARGE MEDICATIONS\s*[\n;]",     # no-colon / semicolon variants
    r"MEDICATIONS AT THE TIME OF DISCHARGE:",
    r"MEDICATIONS AT TIME OF DISCHARGE:",
    r"MEDICATIONS ON DISCHARGE:",
    r"MEDICATIONS ON DISCHARGE;",
    r"Discharge Medications from[^:\n]*:",   # e.g. "Discharge Medications from [date]:"
    r"Discharge Medications to[^:\n]*:",     # e.g. "Discharge Medications to be dictated..."
]

DISCHARGE_MEDICATIONS_END_PATTERNS = [
    r"Discharge Disposition:",
    r"DISCHARGE STATUS:",
    r"FOLLOW-UP PLANS:",
    r"FOLLOW-UP APPOINTMENT:",
    r"FOLLOW-UP:",
    r"Followup Instructions:",
    r"FOLLOW-UP INSTRUCTIONS:",
    r"Discharge Condition:",
    r"CONDITION ON DISCHARGE:",
    r"Discharge Instructions:",
    r"DISCHARGE INSTRUCTIONS:",
    r"DISCHARGE INSTRUCTIONS/FOLLOWUP:",
    r"DISCHARGE INSTRUCTIONS-FOLLOWUP:",
    r"Discharge Diagnosis:",
    r"DISCHARGE DIAGNOSIS:",
    r"DISCHARGE DIAGNOSES:",
    r"the patient's condition",
    # De-identification fallbacks
    r"\[\*\*First Name",
    r"\[\*\*Name",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _search_after(text, start_pos, patterns):
    """Search for the first occurrence of any pattern AFTER start_pos.
    Returns (match_start_position, match_end_position) or (None, None).
    """
    if start_pos >= len(text):
        return None, None
    search_text = text[start_pos:]
    best_start = float("inf")
    best_match = None
    for pat in patterns:
        m = re.search(pat, search_text)
        if m and m.start() < best_start:
            best_start = m.start()
            best_match = m
    if best_match:
        return start_pos + best_match.start(), start_pos + best_match.end()
    return None, None


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_sections(text):
    """
    Extract Diagnosis, Hospital Course, Discharge Instructions
    from a MIMIC-III discharge summary.
    """
    sections = {
        "Diagnosis": "",
        "Hospital Course": "",
        "Discharge Instructions": "",
    }

    # ---- 1. Find Hospital Course ----
    hc_start_match = None
    hc_start_pos = float("inf")
    for pat in HOSPITAL_COURSE_START_PATTERNS:
        m = re.search(pat, text)
        if m and m.start() < hc_start_pos:
            hc_start_pos = m.start()
            hc_start_match = m

    if hc_start_match:
        hc_content_start = hc_start_match.end()
        hc_end_start, _ = _search_after(text, hc_content_start, HOSPITAL_COURSE_END_PATTERNS)
        if hc_end_start is not None:
            sections["Hospital Course"] = text[hc_content_start:hc_end_start].strip()
        else:
            sections["Hospital Course"] = text[hc_content_start:].strip()

    # ---- 2. Find Diagnosis ----
    dx_start_match = None
    dx_start_pos = float("inf")
    for pat in DIAGNOSIS_START_PATTERNS:
        m = re.search(pat, text)
        if m and m.start() < dx_start_pos:
            dx_start_pos = m.start()
            dx_start_match = m

    # Fallback: if no discharge-specific diagnosis header found,
    # look for standalone "DIAGNOSIS:" / "DIAGNOSES:" in the last 25% of doc
    if not dx_start_match:
        doc_len = len(text)
        fallback_start = int(doc_len * 0.75)
        for pat in [r"(?<=\n)DIAGNOSIS:", r"(?<=\n)DIAGNOSES:"]:
            for m in re.finditer(pat, text[fallback_start:]):
                abs_pos = fallback_start + m.start()
                # Skip pathology/lab headers (often have date patterns nearby)
                prefix = text[max(0, abs_pos - 30):abs_pos]
                if re.search(r"(?:\d{2,4}[-/]\d{2})", prefix):
                    continue
                if abs_pos < dx_start_pos:
                    dx_start_pos = abs_pos
                    dx_start_match = m

    if dx_start_match:
        dx_content_start = dx_start_match.end()
        dx_end_start, _ = _search_after(text, dx_content_start, DIAGNOSIS_END_PATTERNS)
        if dx_end_start is not None:
            sections["Diagnosis"] = text[dx_content_start:dx_end_start].strip()
        else:
            sections["Diagnosis"] = text[dx_content_start:].strip()

    # ---- 3. Find Discharge Instructions ----
    di_start_match = None
    di_start_pos = float("inf")
    for pat in DISCHARGE_INSTRUCTIONS_START_PATTERNS:
        m = re.search(pat, text)
        if m and m.start() < di_start_pos:
            di_start_pos = m.start()
            di_start_match = m

    discharge_instructions_text = ""

    if di_start_match:
        di_content_start = di_start_match.end()
        di_end_start, _ = _search_after(text, di_content_start, DISCHARGE_INSTRUCTIONS_END_PATTERNS)
        if di_end_start is not None:
            discharge_instructions_text = text[di_content_start:di_end_start].strip()
        else:
            discharge_instructions_text = text[di_content_start:].strip()

    # ---- 4. Append Discharge Medications to DI ----
    meds_start_match = None
    meds_start_pos = float("inf")
    for pat in DISCHARGE_MEDICATIONS_START_PATTERNS:
        m = re.search(pat, text)
        if m and m.start() < meds_start_pos:
            meds_start_pos = m.start()
            meds_start_match = m

    if meds_start_match:
        meds_content_start = meds_start_match.end()
        meds_end_start, _ = _search_after(text, meds_content_start, DISCHARGE_MEDICATIONS_END_PATTERNS)
        if meds_end_start is not None:
            meds_text = text[meds_content_start:meds_end_start].strip()
        else:
            meds_text = text[meds_content_start:].strip()
        if meds_text:
            if discharge_instructions_text:
                discharge_instructions_text += "\n" + meds_text
            else:
                discharge_instructions_text = meds_text

    sections["Discharge Instructions"] = discharge_instructions_text

    return sections


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def main():
    gold_dir = "/home/qluai/lzy/TE9HTw-/data/DS_long/gold"
    output_dir = "/home/qluai/lzy/TE9HTw-/data/DS_long/gold_process"

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(
        f for f in os.listdir(gold_dir) if re.match(r"gtsummary_\d+\.txt", f)
    )
    n = len(files)
    print(f"Found {n} gold files")

    counts = {
        "Diagnosis": {"nonempty": 0, "empty": 0},
        "Hospital Course": {"nonempty": 0, "empty": 0},
        "Discharge Instructions": {"nonempty": 0, "empty": 0},
    }

    for filename in files:
        base_id = re.match(r"gtsummary_(\d+)\.txt", filename).group(1)

        with open(os.path.join(gold_dir, filename), "r") as f:
            text = f.read()

        sections = extract_sections(text)

        # Write merged file: gtsummary_{id}.txt with three sections
        out_path = os.path.join(output_dir, f"gtsummary_{base_id}.txt")
        with open(out_path, "w") as f:
            f.write(f"Diagnosis:\n{sections['Diagnosis']}\n\n")
            f.write(f"Brief Hospital Course:\n{sections['Hospital Course']}\n\n")
            f.write(f"Discharge Instructions:\n{sections['Discharge Instructions']}\n")

        for section_name, content in sections.items():
            if content.strip():
                counts[section_name]["nonempty"] += 1
            else:
                counts[section_name]["empty"] += 1

    print(f"\nProcessed {n} files → {output_dir}/")
    print(f"{'Section':<30} {'Non-empty':>10} {'Empty':>10} {'Fill rate':>10}")
    print("-" * 60)
    for section in ["Diagnosis", "Hospital Course", "Discharge Instructions"]:
        nonempty = counts[section]["nonempty"]
        empty = counts[section]["empty"]
        rate = nonempty / n * 100
        print(f"{section:<30} {nonempty:>10} {empty:>10} {rate:>9.1f}%")


if __name__ == "__main__":
    main()
