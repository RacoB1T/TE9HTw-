#!/usr/bin/env python3
"""
Evaluate trained LOGO model on DS discharge summary generation.

Metrics:
- ROUGE-L: lexical overlap with ground truth
- BERTScore: semantic similarity (optional, requires bert-score)

Usage:
    python evaluate_ds.py \
        --model_path outputs/llama_ds_long/merged_model \
        --test_dir data/DS_long \
        --output_dir outputs/eval_results \
        --max_samples 73
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rouge import Rouge
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section extraction from generated text (model outputs "Diagnosis:\n...")
# ---------------------------------------------------------------------------

SECTION_PATTERNS_GENERATED = {
    "Diagnosis": re.compile(
        r"(?:^|\n)(?:\*{0,2})D(?:iagnosis|IAGNOSIS)(?:\*{0,2})\s*:\s*\n?",
        re.IGNORECASE,
    ),
    "Hospital Course": re.compile(
        r"(?:^|\n)(?:\*{0,2})(?:Brief\s*)?H(?:ospital|OSPITAL)\s*C(?:ourse|OURSE)(?:\s*Summary)?(?:\*{0,2})\s*:\s*\n?",
        re.IGNORECASE,
    ),
    "Discharge Instructions": re.compile(
        r"(?:^|\n)(?:\*{0,2})D(?:ischarge|ISCHARGE)\s*(?:I(?:nstructions|NSTRUCTIONS)|S(?:ummary|UMMARY))(?:/\s*Followup)?(?:\*{0,2})\s*:\s*\n?",
        re.IGNORECASE,
    ),
}

SECTION_PATTERNS_GOLD = {
    "Diagnosis": re.compile(r"(?:^|\n)Diagnosis:\s*\n", re.IGNORECASE),
    "Hospital Course": re.compile(r"(?:^|\n)Brief Hospital Course:\s*\n", re.IGNORECASE),
    "Discharge Instructions": re.compile(
        r"(?:^|\n)Discharge Instructions:\s*\n", re.IGNORECASE
    ),
}


def extract_sections(text: str, is_ground_truth: bool = False) -> Dict[str, str]:
    """Extract Diagnosis / Hospital Course / Discharge Instructions sections."""
    patterns = SECTION_PATTERNS_GOLD if is_ground_truth else SECTION_PATTERNS_GENERATED
    sections: Dict[str, str] = {}
    section_names = ["Diagnosis", "Hospital Course", "Discharge Instructions"]

    for i, name in enumerate(section_names):
        match = patterns[name].search(text)
        if not match:
            sections[name] = ""
            continue

        start = match.end()
        # Find the next section header
        end = len(text)
        for j in range(i + 1, len(section_names)):
            next_match = patterns[section_names[j]].search(text, pos=start)
            if next_match:
                end = min(end, next_match.start())
        sections[name] = text[start:end].strip()

    return sections


# ---------------------------------------------------------------------------
# Prompt builder (matches the training format)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE_LLAMA3 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Below are some references. Read them carefully and answer the question using the references.<|eot_id|><|start_header_id|>user<|end_header_id|>

References:
{chunks_text}

Question:
Based on the clinical records above, write a complete discharge summary including Diagnosis, Brief Hospital Course, and Discharge Instructions.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

PROMPT_TEMPLATE_LLAMA2 = """<s>[INST] <<SYS>>
Below are some references. Read them carefully and answer the question using the references.
<</SYS>>

References:
{chunks_text}

Question:
Based on the clinical records above, write a complete discharge summary including Diagnosis, Brief Hospital Course, and Discharge Instructions. [/INST] """


def get_prompt_template(model_path: str) -> str:
    """Auto-select prompt template based on model path."""
    if "Llama2" in model_path or "llama-2" in model_path.lower() or "llama2" in model_path.lower():
        return PROMPT_TEMPLATE_LLAMA2
    return PROMPT_TEMPLATE_LLAMA3


def chunk_events(events: List[Dict[str, str]], chunk_size: int = 300) -> List[str]:
    """Group sorted events into text chunks of ~chunk_size chars."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for evt in events:
        text = evt.get("TEXT", "").strip()
        if not text:
            continue
        if current_len + len(text) > chunk_size and current:
            chunks.append(" ".join(current))
            current = [text]
            current_len = len(text)
        else:
            current.append(text)
            current_len += len(text)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_prompt(events: List[Dict[str, str]], model_path: str = "", max_chunks: int = 16) -> str:
    """Build inference prompt from clinical events — no chunking, direct concatenation.

    All event texts are joined in chronological order.  No ``[Chunk N]``
    labels are inserted, and no sampling is applied — the model sees the
    full clinical record (truncated by the tokenizer if needed).
    """
    events.sort(key=lambda x: x.get("TIME", ""))
    # Concatenate all event texts in time order
    event_texts = []
    for evt in events:
        text = evt.get("TEXT", "").strip()
        if text:
            event_texts.append(text)
    references_text = "\n\n".join(event_texts)

    template = get_prompt_template(model_path)
    return template.format(chunks_text=references_text)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_summary(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """Generate a discharge summary from the prompt."""
    # Truncate from the LEFT to keep the most recent clinical events.
    # Use a generous input budget so the model sees as much context as possible.
    input_budget = 4096 - max_new_tokens - 50  # 50-token safety margin
    tokenizer.truncation_side = "left"         # keep the END (most recent events)
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=input_budget,
        padding=False,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.6,            # higher temp breaks self-reinforcing loops
        top_p=0.9,                  # nucleus sampling: cut low-prob tail
        repetition_penalty=1.2,     # stronger anti-repetition
        min_new_tokens=200,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode only the generated part (remove prompt tokens)
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_rouge(
    generated: Dict[str, str],
    gold: Dict[str, str],
) -> Dict[str, float]:
    """Compute ROUGE-L F1 for full text and per-section."""
    rouge = Rouge()
    results = {}

    # Full text
    gen_full = "\n\n".join(generated.values())
    gold_full = "\n\n".join(gold.values())
    if gen_full.strip() and gold_full.strip():
        scores = rouge.get_scores(gen_full, gold_full)
        results["full"] = scores[0]["rouge-l"]["f"] * 100

    # Per section
    for section in generated:
        if generated[section].strip() and gold[section].strip():
            try:
                scores = rouge.get_scores(generated[section], gold[section])
                results[section] = scores[0]["rouge-l"]["f"] * 100
            except Exception:
                results[section] = 0.0
        else:
            results[section] = 0.0

    return results


# ---------------------------------------------------------------------------
# BERTScore with Bio_ClinicalBERT
# ---------------------------------------------------------------------------

_bertscore_model = None
_bertscore_tokenizer = None


def _load_bertscore():
    """Lazy-load Bio_ClinicalBERT for semantic similarity evaluation."""
    global _bertscore_model, _bertscore_tokenizer
    if _bertscore_model is None:
        from transformers import AutoModel, AutoTokenizer
        logger.info("Loading Bio_ClinicalBERT for BERTScore...")
        _bertscore_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        _bertscore_model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        _bertscore_model.eval()
        if torch.cuda.is_available():
            _bertscore_model = _bertscore_model.cuda()
        logger.info("Bio_ClinicalBERT loaded.")


@torch.no_grad()
def _embed_text(text: str, max_length: int = 512) -> torch.Tensor:
    """Get mean-pooled embedding for a text using Bio_ClinicalBERT."""
    _load_bertscore()
    device = next(_bertscore_model.parameters()).device
    inputs = _bertscore_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length, padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = _bertscore_model(**inputs)
    # Mean pool over token dimension (excluding [CLS] and [SEP])
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    embeddings = outputs.last_hidden_state * attention_mask
    pooled = embeddings.sum(dim=1) / attention_mask.sum(dim=1)
    return pooled  # (1, hidden_dim)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two embedding vectors."""
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return (a_norm * b_norm).sum(dim=-1).item()


def evaluate_bertscore(
    generated: Dict[str, str],
    gold: Dict[str, str],
) -> Dict[str, float]:
    """Compute BERTScore (cosine similarity of Bio_ClinicalBERT embeddings).

    Evaluates both full text and per-section.
    """
    results = {}

    # Full text
    gen_full = "\n\n".join(generated.values())
    gold_full = "\n\n".join(gold.values())
    if gen_full.strip() and gold_full.strip():
        gen_emb = _embed_text(gen_full)
        gold_emb = _embed_text(gold_full)
        results["full"] = cosine_similarity(gen_emb, gold_emb) * 100

    # Per section
    for section in generated:
        if generated[section].strip() and gold[section].strip():
            gen_emb = _embed_text(generated[section])
            gold_emb = _embed_text(gold[section])
            results[section] = cosine_similarity(gen_emb, gold_emb) * 100
        else:
            results[section] = 0.0

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--test_dir", required=True, help="Path to DS_test or DS_long dir with input/ and gold_process/")
    parser.add_argument("--output_dir", default="outputs/eval_results")
    parser.add_argument("--max_samples", type=int, default=73, help="Max test samples to evaluate")
    parser.add_argument("--max_chunks", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    logger.info("Loading merged model from %s ...", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    logger.info("Model loaded. Device: %s", model.device)

    # --- Find test patients ---
    input_dir = os.path.join(args.test_dir, "input")
    gold_dir = os.path.join(args.test_dir, "gold_process")

    # Get test patient IDs from conversion metadata
    meta_path = os.path.join(args.test_dir, "ds_logo_dataset", "metadata.jsonl")
    test_ids: List[str] = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            for line in f:
                m = json.loads(line)
                if m.get("split") == "test":
                    test_ids.append(str(m["sample_id"]))
    else:
        # Fallback: use the gold_process/test split
        gold_files = [f for f in os.listdir(gold_dir) if f.endswith(".txt")]
        test_ids = [re.match(r"gtsummary_(\d+)\.txt", f).group(1)
                    for f in gold_files
                    if re.match(r"gtsummary_(\d+)\.txt", f)]

    # Also try loading from the tokenized metadata
    if not test_ids:
        meta_alt = os.path.join(args.test_dir, "ds_logo_tokenized_llama", "metadata.jsonl")
        if os.path.exists(meta_alt):
            with open(meta_alt) as f:
                for line in f:
                    m = json.loads(line)
                    if m.get("split") == "test":
                        test_ids.append(str(m["sample_id"]))

    test_ids = sorted(set(test_ids))[:args.max_samples]
    logger.info("Evaluating %d test patients", len(test_ids))
    if not test_ids:
        logger.error("No test patients found!")
        sys.exit(1)

    # --- Evaluate ---
    all_rouge: Dict[str, List[float]] = {"full": [], "Diagnosis": [], "Hospital Course": [], "Discharge Instructions": []}
    all_bertscore: Dict[str, List[float]] = {"full": [], "Diagnosis": [], "Hospital Course": [], "Discharge Instructions": []}
    results_detail: List[Dict] = []

    for pid in tqdm(test_ids, desc="Evaluating"):
        csv_path = os.path.join(input_dir, f"input_{pid}.csv")
        gold_path = os.path.join(gold_dir, f"gtsummary_{pid}.txt")

        if not os.path.exists(csv_path) or not os.path.exists(gold_path):
            logger.warning("Skipping %s: missing file", pid)
            continue

        # Read input events
        events: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                t = row.get("TIME", "").strip()
                text = row.get("TEXT", "").strip()
                if text:
                    events.append({"TIME": t, "TEXT": text})

        # Read gold summary
        with open(gold_path, "r", encoding="utf-8", errors="replace") as f:
            gold_text = f.read().strip()

        # Generate
        prompt = build_prompt(events, model_path=args.model_path, max_chunks=args.max_chunks)
        generated = generate_summary(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)

        # Extract sections
        gen_sections = extract_sections(generated, is_ground_truth=False)
        gold_sections = extract_sections(gold_text, is_ground_truth=True)

        # Compute ROUGE + BERTScore
        rouge_scores = evaluate_rouge(gen_sections, gold_sections)
        bert_scores = evaluate_bertscore(gen_sections, gold_sections)

        # Save generated summary
        out_path = os.path.join(args.output_dir, f"gen_{pid}.txt")
        with open(out_path, "w") as f:
            f.write(generated)

        detail = {
            "patient_id": pid,
            "rouge": rouge_scores,
            "bertscore": bert_scores,
            "generated_len": len(generated),
            "gold_len": len(gold_text),
        }
        results_detail.append(detail)
        for key, val in rouge_scores.items():
            if key in all_rouge:
                all_rouge[key].append(val)
        for key, val in bert_scores.items():
            if key in all_bertscore:
                all_bertscore[key].append(val)

    # --- Report ---
    print("\n" + "=" * 60)
    print("Evaluation Results — ROUGE-L F1 (%)")
    print("=" * 60)
    for section, scores in all_rouge.items():
        if scores:
            print(f"  {section:<30} {np.mean(scores):6.2f} ± {np.std(scores):5.2f}")
        else:
            print(f"  {section:<30}  N/A")

    print("\n" + "=" * 60)
    print("Evaluation Results — BERTScore (%)")
    print("=" * 60)
    for section, scores in all_bertscore.items():
        if scores:
            print(f"  {section:<30} {np.mean(scores):6.2f} ± {np.std(scores):5.2f}")
        else:
            print(f"  {section:<30}  N/A")

    # Save results
    report = {
        "model_path": args.model_path,
        "num_samples": len(results_detail),
        "rouge_overall": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_rouge.items() if v},
        "bertscore_overall": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_bertscore.items() if v},
        "details": results_detail,
    }
    report_path = os.path.join(args.output_dir, "eval_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {report_path}")
    print(f"Generated summaries saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
