#!/usr/bin/env python3
"""
LOGO Training Data Builder.

Converts post-processed QA data (with critical-path chunks) into
tokenized, position-aware training samples ready for ``LOGOTrainer``.

Usage::

    python data/build_logo_dataset.py \\
        --input_path /data/pre-process \\
        --output_path ./logo_train_data \\
        --tokenizer_path meta-llama/Meta-Llama-3-8B-Instruct \\
        --context_mode existing \\
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import datasets
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# Ensure repo root is importable (for position_synthesis, etc.)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.position_synthesis import (
    Layout,
    PositionValidationError,
    TimeLayout,
    compute_layout,
    derive_variant_seed,
    synthesize_continuous_positions,
    synthesize_qa_tail_positions,
    synthesize_sparse_positions,
    synthesize_temporal_positions,
    validate_overall_positions,
    validate_positions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_12_FIELDS = frozenset(
    {
        "chosen_input_ids",
        "chosen_attention_mask",
        "chosen_position_ids",
        "chosen_labels",
        "reject_1_input_ids",
        "reject_1_attention_mask",
        "reject_1_position_ids",
        "reject_1_labels",
        "reject_2_input_ids",
        "reject_2_attention_mask",
        "reject_2_position_ids",
        "reject_2_labels",
    }
)

DEFAULT_SYSTEM_MESSAGE = (
    "Below are some references. Read them carefully and "
    "answer the question using the references."
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PromptParts:
    """Tokenized components of a single prompt."""

    system_tokens: List[int]
    """All tokens preceding the first chunk (BOS + system header + content + EOT)."""

    chunk_tokens: List[List[int]]
    """Tokenized chunk texts, one list per chunk (no special tokens)."""

    user_framing_tokens: List[List[int]]
    """
    Tokens between chunks (references header, numbering, separators).
    ``user_framing_tokens[i]`` is the text between ``chunk_tokens[i-1]``
    and ``chunk_tokens[i]``. The last entry is the text after the last
    chunk (separator + question + EOT).
    """

    assistant_header_tokens: List[int]
    """Tokens for the assistant turn header.

    For Llama-3 this is::

        <|start_header_id|>assistant<|end_header_id|>\\n\\n
    """

    @property
    def num_chunks(self) -> int:
        return len(self.chunk_tokens)

    @property
    def system_len(self) -> int:
        return len(self.system_tokens)

    @property
    def chunk_lens(self) -> List[int]:
        return [len(c) for c in self.chunk_tokens]

    @property
    def assistant_header_len(self) -> int:
        return len(self.assistant_header_tokens)

    def build_shared_prefix_input_ids(self) -> List[int]:
        """Build the full shared-prefix token sequence (no answers)."""
        ids: List[int] = list(self.system_tokens)
        for i in range(self.num_chunks):
            if i == 0:
                ids.extend(self.user_framing_tokens[0])
            ids.extend(self.chunk_tokens[i])
            ids.extend(self.user_framing_tokens[i + 1])
        ids.extend(self.assistant_header_tokens)
        return ids

    def build_shared_prefix_attention_mask(self) -> List[int]:
        return [1] * len(self.build_shared_prefix_input_ids())

    def build_shared_prefix_labels(self) -> List[int]:
        return [-100] * len(self.build_shared_prefix_input_ids())

    def build_prefix_position_ids(
        self,
        system_positions: List[int],
        chunk_positions: List[List[int]],
        question_positions: List[int],
        assistant_header_positions: List[int],
    ) -> List[int]:
        """Assemble prefix position IDs from pre-computed parts."""
        ids: List[int] = list(system_positions)
        for i in range(self.num_chunks):
            if i == 0:
                ids.extend([0] * len(self.user_framing_tokens[0]))  # placeholder, overwritten
            ids.extend(chunk_positions[i])
            ids.extend([0] * len(self.user_framing_tokens[i + 1]))
        ids.extend(assistant_header_positions)
        return ids


@dataclass
class NormalizedSample:
    """Single input sample after field normalization."""

    sample_id: str
    source_name: str
    context_chunks: List[str]
    question: str
    chosen_answer: str
    rejected_answer_1: str
    rejected_answer_2: str
    label: str = ""
    chunk_timestamps: List[float] = field(default_factory=list)
    critical_chunks: List[dict] = field(default_factory=list)
    partial_critical_chunks: List[dict] = field(default_factory=list)
    irrelevant_chunks: List[dict] = field(default_factory=list)


@dataclass
class BuildReport:
    """Statistics gathered during dataset construction."""

    total_input_samples: int = 0
    success_samples: int = 0
    dropped_samples: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    train_samples: int = 0
    test_samples: int = 0
    continuous_count: int = 0
    sparse_count: int = 0
    avg_real_length: float = 0.0
    max_real_length: int = 0
    avg_max_position: float = 0.0
    avg_chosen_answer_len: float = 0.0
    avg_rejected_1_len: float = 0.0
    avg_rejected_2_len: float = 0.0


# ---------------------------------------------------------------------------
# Prompt adapters
# ---------------------------------------------------------------------------


class PromptAdapter(ABC):
    """Abstract prompt adapter — decomposes a chat template into token parts."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, system_message: str = DEFAULT_SYSTEM_MESSAGE):
        self.tokenizer = tokenizer
        self.system_message = system_message

    @abstractmethod
    def build_prompt_parts(self, context_chunks: List[str], question: str) -> PromptParts:
        ...

    @abstractmethod
    def tokenize_answer(self, answer: str) -> List[int]:
        """Tokenize an answer with appropriate EOS/EOT termination."""
        ...


class Llama3PromptAdapter(PromptAdapter):
    """Prompt adapter for Meta Llama-3 Instruct format."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, system_message: str = DEFAULT_SYSTEM_MESSAGE):
        super().__init__(tokenizer, system_message)
        # Cache special token IDs
        self.bos_id: int = tokenizer.bos_token_id or tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
        self.eot_id: int = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        self.eos_id: int = tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids("<|end_of_text|>")

    # ------------------------------------------------------------------
    def build_prompt_parts(self, context_chunks: List[str], question: str) -> PromptParts:
        tk = self.tokenizer

        # --- system ---
        system_header_ids = tk.encode(
            "<|start_header_id|>system<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        system_content_ids = tk.encode(self.system_message, add_special_tokens=False)
        eot_ids = tk.encode("<|eot_id|>", add_special_tokens=False)
        system_tokens = [self.bos_id] + system_header_ids + system_content_ids + eot_ids

        # --- user header ---
        user_header_ids = tk.encode(
            "<|start_header_id|>user<|end_header_id|>\n\n",
            add_special_tokens=False,
        )

        # --- chunks ---
        chunk_tokens = [
            tk.encode(chunk_text, add_special_tokens=False)
            for chunk_text in context_chunks
        ]

        # --- user framing (text between chunks) ---
        # Structure:
        #   framing[0]:  "References:\n[Chunk 1]\n"
        #   framing[1]:  "\n\n[Chunk 2]\n"
        #   framing[2]:  "\n\n[Chunk 3]\n"
        #   ...
        #   framing[N]:  "\n\nQuestion:\n{question}"
        user_framing_tokens: List[List[int]] = []
        for i in range(len(context_chunks) + 1):
            if i == 0:
                text = "References:\n[Chunk 1]\n"
            elif i < len(context_chunks):
                text = f"\n\n[Chunk {i + 1}]\n"
            else:
                text = f"\n\nQuestion:\n{question}"
            user_framing_tokens.append(tk.encode(text, add_special_tokens=False))

        # Prepend user_header to first framing, append EOT to last
        user_framing_tokens[0] = user_header_ids + user_framing_tokens[0]
        user_framing_tokens[-1] = user_framing_tokens[-1] + eot_ids

        # --- assistant header ---
        assistant_header_tokens = tk.encode(
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )

        return PromptParts(
            system_tokens=system_tokens,
            chunk_tokens=chunk_tokens,
            user_framing_tokens=user_framing_tokens,
            assistant_header_tokens=assistant_header_tokens,
        )

    # ------------------------------------------------------------------
    def tokenize_answer(self, answer: str) -> List[int]:
        tk = self.tokenizer
        answer_ids = tk.encode(answer, add_special_tokens=False)

        # Append EOT + EOS if not already present
        if not answer_ids or answer_ids[-1] != self.eot_id:
            answer_ids.append(self.eot_id)
        if answer_ids[-1] != self.eos_id:
            answer_ids.append(self.eos_id)

        return answer_ids


class Qwen35PromptAdapter(PromptAdapter):
    """Prompt adapter for Qwen3.5 ChatML format.

    Qwen3.5 uses ChatML delimiters::

        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        References:
        [Chunk 1]
        ...
        Question:
        {question}<|im_end|>
        <|im_start|>assistant

        {answer}<|im_end|>
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, system_message: str = DEFAULT_SYSTEM_MESSAGE):
        super().__init__(tokenizer, system_message)
        # Qwen3.5 special tokens
        self.im_start_id: int = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id: int = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # Qwen3.5 has no BOS token
        self.bos_id: int = tokenizer.bos_token_id  # None for Qwen3.5

    # ------------------------------------------------------------------
    def build_prompt_parts(self, context_chunks: List[str], question: str) -> PromptParts:
        tk = self.tokenizer

        # --- system ---
        # Format: <|im_start|>system\n{message}<|im_end|>\n
        system_tokens = (
            [self.im_start_id]
            + tk.encode("system\n", add_special_tokens=False)
            + tk.encode(self.system_message, add_special_tokens=False)
            + [self.im_end_id]
            + tk.encode("\n", add_special_tokens=False)
        )

        # --- user header ---
        # Format: <|im_start|>user\n
        user_header_ids = (
            [self.im_start_id]
            + tk.encode("user\n", add_special_tokens=False)
        )

        # --- chunks ---
        chunk_tokens = [
            tk.encode(chunk_text, add_special_tokens=False)
            for chunk_text in context_chunks
        ]

        # --- user framing (text between chunks) ---
        user_framing_tokens: List[List[int]] = []
        for i in range(len(context_chunks) + 1):
            if i == 0:
                text = "References:\n[Chunk 1]\n"
            elif i < len(context_chunks):
                text = f"\n\n[Chunk {i + 1}]\n"
            else:
                text = f"\n\nQuestion:\n{question}"
            user_framing_tokens.append(tk.encode(text, add_special_tokens=False))

        # Prepend user_header to first framing, append <|im_end|> + newline to last
        user_framing_tokens[0] = user_header_ids + user_framing_tokens[0]
        user_framing_tokens[-1] = user_framing_tokens[-1] + [self.im_end_id] + tk.encode("\n", add_special_tokens=False)

        # --- assistant header ---
        # Format: <|im_start|>assistant\n
        # No <think> block — we set enable_thinking=False for training
        assistant_header_tokens = (
            [self.im_start_id]
            + tk.encode("assistant\n", add_special_tokens=False)
        )

        return PromptParts(
            system_tokens=system_tokens,
            chunk_tokens=chunk_tokens,
            user_framing_tokens=user_framing_tokens,
            assistant_header_tokens=assistant_header_tokens,
        )

    # ------------------------------------------------------------------
    def tokenize_answer(self, answer: str) -> List[int]:
        tk = self.tokenizer
        answer_ids = tk.encode(answer, add_special_tokens=False)

        # Append <|im_end|> as EOS if not already present
        if not answer_ids or answer_ids[-1] != self.im_end_id:
            answer_ids.append(self.im_end_id)

        return answer_ids


# ---------------------------------------------------------------------------
# Prompt adapter registry
# ---------------------------------------------------------------------------

PROMPT_ADAPTERS = {
    "llama-3": Llama3PromptAdapter,
    "qwen3.5": Qwen35PromptAdapter,
}
# Answer normalization
# ---------------------------------------------------------------------------


def _normalize_answer(text: str) -> str:
    """Lightweight answer normalization for deduplication."""
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    # Strip trailing EOS/EOT markers
    for marker in ["<|eot_id|>", "<|end_of_text|>", "</s>", "<|end|>"]:
        while t.endswith(marker):
            t = t[: -len(marker)].strip()
    return t


def _answers_equivalent(a: str, b: str) -> bool:
    return _normalize_answer(a).lower() == _normalize_answer(b).lower()


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


class LogoDatasetBuilder:
    """Build a LOGO training DatasetDict from post-processed QA data."""

    def __init__(
        self,
        tokenizer_path: str,
        output_path: str,
        *,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_type: str = "llama-3",
        context_mode: str = "existing",
        max_seq_length: int = 10000,
        real_reference_tokens: int = 8192,
        target_position_length: int = 65536,
        num_chunks: int = 16,
        chunk_token_size: int = 512,
        max_answer_tokens: int = 512,
        position_variants_per_sample: int = 2,
        continuous_ratio: float = 0.9,
        test_ratio: float = 0.02,
        seed: int = 42,
        num_proc: int = 1,
        strict: bool = False,
        overwrite: bool = False,
        deduplicate_answers: bool = True,
    ):
        self.output_path = output_path
        self.model_type = model_type
        self.context_mode = context_mode
        self.max_seq_length = max_seq_length
        self.real_reference_tokens = real_reference_tokens
        self.target_position_length = target_position_length
        self.num_chunks = num_chunks
        self.chunk_token_size = chunk_token_size
        self.max_answer_tokens = max_answer_tokens
        self.position_variants_per_sample = position_variants_per_sample
        self.continuous_ratio = continuous_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.num_proc = num_proc
        self.strict = strict
        self.overwrite = overwrite
        self.deduplicate_answers = deduplicate_answers

        # Load tokenizer
        if tokenizer is not None:
            logger.info("Using provided tokenizer instance.")
            self.tokenizer = tokenizer
        else:
            logger.info("Loading tokenizer from %s ...", tokenizer_path)
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, trust_remote_code=True
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = (
                self.tokenizer.eos_token_id
                or self.tokenizer.unk_token_id
                or 0
            )

        adapter_cls = PROMPT_ADAPTERS.get(self.model_type)
        if adapter_cls is None:
            raise ValueError(
                f"Unknown model_type '{self.model_type}'. "
                f"Available: {list(PROMPT_ADAPTERS.keys())}"
            )
        self.prompt_adapter = adapter_cls(self.tokenizer)

        # State
        self.report = BuildReport()

    # ------------------------------------------------------------------
    # Input loading & normalization
    # ------------------------------------------------------------------

    def load_input_data(self, input_path: str) -> List[NormalizedSample]:
        """Load all post-process datasets from *input_path* and normalize fields."""
        logger.info("Loading input data from %s ...", input_path)

        # Collect all datasets (single dir or multiple subdirs)
        dataset_items: List[Dict[str, Any]] = []
        if os.path.isdir(input_path):
            # Try loading as a single Dataset
            try:
                ds = datasets.load_from_disk(input_path)
                if isinstance(ds, DatasetDict):
                    # Loaded a DatasetDict with train/test splits
                    for split_name, split_ds in ds.items():
                        for row in split_ds:
                            row["_source_name"] = split_name
                            dataset_items.append(row)
                        logger.info(
                            "  Loaded %d samples from split '%s'", len(split_ds), split_name
                        )
                else:
                    for row in ds:
                        dataset_items.append(row)
                    logger.info("Loaded %d samples from single dataset.", len(dataset_items))
            except Exception:
                # Try loading each subdirectory
                for subdir in sorted(os.listdir(input_path)):
                    subpath = os.path.join(input_path, subdir)
                    if os.path.isdir(subpath) and not subdir.startswith("."):
                        try:
                            ds = datasets.load_from_disk(subpath)
                            for row in ds:
                                row["_source_name"] = subdir
                                dataset_items.append(row)
                            logger.info(
                                "  Loaded %d samples from %s", len(ds), subdir
                            )
                        except Exception as exc:
                            logger.warning("  Skipping %s: %s", subdir, exc)
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if not dataset_items:
            raise ValueError(f"No valid dataset found at {input_path}")

        # Normalize fields
        samples: List[NormalizedSample] = []
        for idx, item in enumerate(dataset_items):
            source = item.get("_source_name", "unknown")

            # Context chunks (handle both legacy strings and dicts from gen_hf.py)
            all_ref = item.get("all_ref_text", [])
            if isinstance(all_ref, str):
                all_ref = [all_ref]
            chunks: List[str] = []
            for c in all_ref:
                if c is None:
                    continue
                if isinstance(c, dict):
                    # New format: {"chunk_id": int, "text": str} from gen_hf.py
                    text = str(c.get("text", c.get("chunk", ""))).strip()
                else:
                    text = str(c).strip()
                if text:
                    chunks.append(text)

            # Question
            question = str(item.get("combined_question", "")).strip()

            # Answers
            chosen = str(item.get("final_answer", "")).strip()
            rejected_1 = str(item.get("prefix_a", "")).strip()

            # suffix_a: accept both correct and typo
            rejected_2 = str(
                item.get("suffix_a", item.get("siffix_a", ""))
            ).strip()

            # Label (optional)
            label = str(item.get("label", "")).strip()

            # Chunk timestamps (for time-driven position encoding)
            ts_raw = item.get("chunk_timestamps", [])
            if isinstance(ts_raw, list) and ts_raw:
                chunk_ts = [float(t) if t is not None else 0.0 for t in ts_raw]
            else:
                chunk_ts = []

            sample_id = hashlib.md5(
                f"{source}:{idx}:{question}".encode()
            ).hexdigest()[:12]

            # Paper-mode chunk fields (lists of {"chunk_id": int, "text": str} dicts)
            critical_chunks = item.get("critical_chunks", [])
            if not isinstance(critical_chunks, list):
                critical_chunks = []
            partial_critical_chunks = item.get("partial_critical_chunks", [])
            if not isinstance(partial_critical_chunks, list):
                partial_critical_chunks = []
            irrelevant_chunks = item.get("irrelevant_chunks", [])
            if not isinstance(irrelevant_chunks, list):
                irrelevant_chunks = []

            samples.append(
                NormalizedSample(
                    sample_id=sample_id,
                    source_name=source,
                    context_chunks=chunks,
                    question=question,
                    chosen_answer=chosen,
                    rejected_answer_1=rejected_1,
                    rejected_answer_2=rejected_2,
                    label=label,
                    chunk_timestamps=chunk_ts,
                    critical_chunks=critical_chunks,
                    partial_critical_chunks=partial_critical_chunks,
                    irrelevant_chunks=irrelevant_chunks,
                )
            )

        logger.info("Normalized %d total samples.", len(samples))
        self.report.total_input_samples = len(samples)
        return samples

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_sample(self, sample: NormalizedSample) -> Optional[str]:
        """Return ``None`` if valid, else a reason string for dropping."""
        # Required fields
        if not sample.context_chunks:
            return "empty_context_chunks"
        if not sample.question:
            return "empty_question"
        if not sample.chosen_answer:
            return "empty_chosen_answer"
        if not sample.rejected_answer_1:
            return "empty_rejected_1"
        if not sample.rejected_answer_2:
            return "empty_rejected_2"

        # Answer deduplication
        if self.deduplicate_answers:
            if _answers_equivalent(sample.chosen_answer, sample.rejected_answer_1):
                return "chosen_equals_rejected_1"
            if _answers_equivalent(sample.chosen_answer, sample.rejected_answer_2):
                return "chosen_equals_rejected_2"
            if _answers_equivalent(sample.rejected_answer_1, sample.rejected_answer_2):
                return "rejected_1_equals_rejected_2"

        return None

    # ------------------------------------------------------------------
    # Shared chunks
    # ------------------------------------------------------------------

    def build_shared_chunks(self, sample: NormalizedSample) -> List[str]:
        """Return the shared context chunks for all three branches."""
        if self.context_mode == "existing":
            return self._build_existing_chunks(sample)
        elif self.context_mode == "paper":
            return self._build_paper_chunks(sample)
        else:
            raise ValueError(f"Unknown context_mode: {self.context_mode}")

    def _build_existing_chunks(self, sample: NormalizedSample) -> List[str]:
        """existing mode: use all_ref_text as-is, deduplicate."""
        # Remove duplicates while preserving order
        seen: set = set()
        unique: List[str] = []
        for c in sample.context_chunks:
            key = c.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[: self.num_chunks]

    def _build_paper_chunks(self, sample: NormalizedSample) -> List[str]:
        """paper mode: combine critical + sampled irrelevant chunks.

        Merges all critical_chunks with a random sample of irrelevant_chunks,
        sorts by chunk_id to restore original document order, deduplicates,
        and limits to ``num_chunks``. All three answers share this same context.
        """
        critical: List[dict] = (
            list(sample.critical_chunks) if sample.critical_chunks else []
        )
        irrelevant: List[dict] = (
            list(sample.irrelevant_chunks) if sample.irrelevant_chunks else []
        )

        # Sample irrelevant chunks to fill up to num_chunks
        num_irrelevant = max(0, self.num_chunks - len(critical))
        if num_irrelevant > 0 and irrelevant:
            rng = random.Random(self.seed)
            sampled_irrelevant = rng.sample(
                irrelevant, min(num_irrelevant, len(irrelevant))
            )
        else:
            sampled_irrelevant = []

        # Combine and sort by chunk_id to restore original document order
        all_chunks = critical + sampled_irrelevant
        all_chunks.sort(
            key=lambda x: x.get("chunk_id", 0) if isinstance(x, dict) else 0
        )

        # Extract text, deduplicate while preserving order
        seen: set = set()
        result: List[str] = []
        for c in all_chunks:
            if isinstance(c, dict):
                text = str(c.get("text", c.get("chunk", ""))).strip()
            else:
                text = str(c).strip()
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                result.append(text)

        return result[: self.num_chunks]

    # ------------------------------------------------------------------
    # Length control
    # ------------------------------------------------------------------

    def _control_chunk_length(self, chunk_tokens: List[int]) -> List[int]:
        """Truncate a single chunk to ``chunk_token_size`` tokens."""
        if len(chunk_tokens) > self.chunk_token_size:
            return chunk_tokens[: self.chunk_token_size]
        return chunk_tokens

    def _control_answer_length(self, answer_tokens: List[int]) -> List[int]:
        """Truncate answer to ``max_answer_tokens`` tokens."""
        if len(answer_tokens) > self.max_answer_tokens:
            return answer_tokens[: self.max_answer_tokens]
        return answer_tokens

    def _compute_budget(self, pp: PromptParts) -> Tuple[int, int, int, int]:
        """Return ``(system_len, question_len, assistant_header_len, max_answer_len)``.

        ``question_len`` = total framing tokens (everything between chunks + question text).
        """
        system_len = pp.system_len
        # framing tokens: first is user_header + "References:\n[Chunk 1]\n",
        # middle are "\n\n[Chunk N]\n", last is "\n\nQuestion:\n{question}" + eot
        question_len = sum(len(f) for f in pp.user_framing_tokens)
        assistant_header_len = pp.assistant_header_len
        return system_len, question_len, assistant_header_len

    def _control_total_length(
        self,
        pp: PromptParts,
        chosen_tokens: List[int],
        rejected_1_tokens: List[int],
        rejected_2_tokens: List[int],
    ) -> Tuple[PromptParts, List[int], List[int], List[int]]:
        """Ensure total length of every branch <= max_seq_length.

        Priority: truncate answers, then chunks, then (as last resort) question.
        System and assistant header are never truncated.
        """
        system_len, question_len, assistant_header_len = self._compute_budget(pp)
        chunk_total = sum(pp.chunk_lens)

        max_answer = max(
            len(chosen_tokens), len(rejected_1_tokens), len(rejected_2_tokens)
        )

        total = system_len + question_len + chunk_total + assistant_header_len + max_answer

        if total <= self.max_seq_length:
            return pp, chosen_tokens, rejected_1_tokens, rejected_2_tokens

        # 1. Truncate answers
        overhead = total - self.max_seq_length
        if max_answer > 0:
            trim = min(overhead, max_answer - 1)  # keep at least 1 answer token
            if trim > 0:
                chosen_tokens = chosen_tokens[: max(1, len(chosen_tokens) - trim)]
                rejected_1_tokens = rejected_1_tokens[: max(1, len(rejected_1_tokens) - trim)]
                rejected_2_tokens = rejected_2_tokens[: max(1, len(rejected_2_tokens) - trim)]
                overhead -= trim

        if overhead <= 0:
            return pp, chosen_tokens, rejected_1_tokens, rejected_2_tokens

        # 2. Truncate individual chunks
        for i in range(pp.num_chunks):
            if overhead <= 0:
                break
            cl = pp.chunk_lens[i]
            if cl > 1:
                trim = min(overhead, cl - 1)
                pp.chunk_tokens[i] = pp.chunk_tokens[i][: cl - trim]
                overhead -= trim

        if overhead <= 0:
            return pp, chosen_tokens, rejected_1_tokens, rejected_2_tokens

        # 3. Last resort: truncate question framing (keep at least 10 tokens)
        if question_len > 10 and overhead > 0:
            trim = min(overhead, question_len - 10)
            # Trim from the last framing segment (question text)
            last_framing = pp.user_framing_tokens[-1]
            if len(last_framing) > trim:
                pp.user_framing_tokens[-1] = last_framing[: len(last_framing) - trim]
            overhead -= trim

        return pp, chosen_tokens, rejected_1_tokens, rejected_2_tokens

    # ------------------------------------------------------------------
    # Single variant construction
    # ------------------------------------------------------------------

    def build_single_variant(
        self,
        sample: NormalizedSample,
        variant_index: int,
        strategy: str,  # "continuous" or "sparse"
    ) -> Optional[Dict[str, Any]]:
        """Build one position variant for a single sample.

        Returns a dict with exactly 12 keys, or ``None`` if the sample
        should be dropped.
        """
        variant_seed = derive_variant_seed(self.seed, sample.sample_id, variant_index)

        # 1. Shared chunks
        shared_chunks = self.build_shared_chunks(sample)
        if not shared_chunks:
            return None

        # 2. Build prompt parts
        pp = self.prompt_adapter.build_prompt_parts(shared_chunks, sample.question)

        # 3. Tokenize and truncate answers
        chosen_tokens = self._control_answer_length(
            self.prompt_adapter.tokenize_answer(sample.chosen_answer)
        )
        rejected_1_tokens = self._control_answer_length(
            self.prompt_adapter.tokenize_answer(sample.rejected_answer_1)
        )
        rejected_2_tokens = self._control_answer_length(
            self.prompt_adapter.tokenize_answer(sample.rejected_answer_2)
        )

        # 4. Control chunk lengths
        pp.chunk_tokens = [self._control_chunk_length(c) for c in pp.chunk_tokens]

        # 5. Overall length control
        pp, chosen_tokens, rejected_1_tokens, rejected_2_tokens = self._control_total_length(
            pp, chosen_tokens, rejected_1_tokens, rejected_2_tokens
        )

        # 6. Build position IDs preserving temporal order for clinical data.
        #
        # The token sequence is:
        #   system | framing[0] chunk[0] framing[1] chunk[1] ... framing[N] | assistant_header | answer
        #
        # Position layout (time-driven when timestamps available):
        #   system:  [0, system_len)                                     continuous
        #   context: [system_len, tail_start)   time-proportional slots  temporal
        #   tail:    [tail_start, K_target)                              continuous

        system_len, question_len, assistant_header_len = self._compute_budget(pp)

        # Group sizes: one per chunk (framing[i] + chunk[i])
        group_lens: List[int] = []
        for ci in range(pp.num_chunks):
            f_idx = 0 if ci == 0 else ci
            group_lens.append(len(pp.user_framing_tokens[f_idx]) + pp.chunk_lens[ci])

        # Last framing + assistant_header are in the tail
        tail_prefix_len = len(pp.user_framing_tokens[-1]) + assistant_header_len
        max_answer = max(len(chosen_tokens), len(rejected_1_tokens), len(rejected_2_tokens))
        tail_total = tail_prefix_len + max_answer

        tail_start = self.target_position_length - tail_total
        tail_start = max(tail_start, system_len + sum(group_lens))

        # Map chunk timestamps if available (preserve clinical temporal order)
        use_time_driven = bool(sample.chunk_timestamps) and len(sample.chunk_timestamps) == len(sample.context_chunks)
        group_timestamps: List[float] = []
        if use_time_driven:
            # Map each shared chunk back to its original timestamp
            for chunk_text in shared_chunks:
                try:
                    orig_idx = sample.context_chunks.index(chunk_text)
                    group_timestamps.append(sample.chunk_timestamps[orig_idx])
                except (ValueError, IndexError):
                    use_time_driven = False
                    break

        num_groups = len(group_lens)

        # Fallback to index-based ordering if no timestamps
        if not use_time_driven:
            group_timestamps = [float(i) for i in range(num_groups)]

        # Build position IDs using time-driven layout
        time_layout = TimeLayout(
            system_len=system_len,
            chunk_lens=list(group_lens),
            chunk_timestamps=list(group_timestamps),
            question_len=question_len,
            assistant_header_len=assistant_header_len,
            max_answer_len=max_answer,
            K_target=self.target_position_length,
        )

        rng = random.Random(variant_seed)
        if use_time_driven and len(group_lens) > 0:
            if strategy == "sparse":
                # Time-driven with larger jitter for sparse variety
                group_positions = synthesize_temporal_positions(
                    group_lens, list(time_layout.chunk_starts), seed=variant_seed,
                    jitter_ratio=0.15,  # larger jitter for sparse
                )
            else:
                group_positions = synthesize_temporal_positions(
                    group_lens, list(time_layout.chunk_starts), seed=variant_seed
                )
            group_positions = [g.tolist() for g in group_positions]
        else:
            # Equal-slot fallback
            context_size = time_layout.context_region_size
            num_groups = len(group_lens)
            slot_boundaries: List[Tuple[int, int]] = []
            if num_groups > 0 and context_size > 0:
                slot_width = context_size // num_groups
                leftover = context_size % num_groups
                cursor = system_len
                for gi in range(num_groups):
                    extra = 1 if gi < leftover else 0
                    start = cursor
                    end = cursor + slot_width + extra
                    slot_boundaries.append((start, end))
                    cursor = end
                    group_positions.append(selected)

        # Build shared prefix position IDs sequentially
        def _build_branch_positions(answer_len: int) -> List[int]:
            pos: List[int] = []

            # System
            pos.extend(range(system_len))

            # Chunk groups (framing + chunk interleaved)
            for ci in range(num_groups):
                f_idx = 0 if ci == 0 else ci
                framing_len = len(pp.user_framing_tokens[f_idx])
                chunk_len = pp.chunk_lens[ci]
                group_pos = group_positions[ci]
                # First framing_len positions go to framing tokens
                pos.extend(group_pos[:framing_len])
                # Remaining go to chunk tokens
                pos.extend(group_pos[framing_len:framing_len + chunk_len])

            # Last framing + assistant_header (tail region)
            last_framing_len = len(pp.user_framing_tokens[-1])
            tail_cursor = tail_start
            for _ in range(last_framing_len):
                pos.append(tail_cursor)
                tail_cursor += 1
            for _ in range(assistant_header_len):
                pos.append(tail_cursor)
                tail_cursor += 1

            # Answer (continues from tail_cursor)
            for _ in range(answer_len):
                pos.append(tail_cursor)
                tail_cursor += 1

            return pos

        # Build full input_ids
        prefix_ids = pp.build_shared_prefix_input_ids()
        prefix_attn = pp.build_shared_prefix_attention_mask()

        chosen_position_ids = _build_branch_positions(len(chosen_tokens))
        r1_position_ids = _build_branch_positions(len(rejected_1_tokens))
        r2_position_ids = _build_branch_positions(len(rejected_2_tokens))

        chosen_input_ids = prefix_ids + chosen_tokens
        chosen_attention_mask = prefix_attn + [1] * len(chosen_tokens)
        chosen_labels = [-100] * len(prefix_ids) + chosen_tokens

        r1_input_ids = prefix_ids + rejected_1_tokens
        r1_attention_mask = prefix_attn + [1] * len(rejected_1_tokens)
        r1_labels = [-100] * len(prefix_ids) + rejected_1_tokens

        r2_input_ids = prefix_ids + rejected_2_tokens
        r2_attention_mask = prefix_attn + [1] * len(rejected_2_tokens)
        r2_labels = [-100] * len(prefix_ids) + rejected_2_tokens

        # 11. Per-sample validation
        result = {
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_position_ids": chosen_position_ids,
            "chosen_labels": chosen_labels,
            "reject_1_input_ids": r1_input_ids,
            "reject_1_attention_mask": r1_attention_mask,
            "reject_1_position_ids": r1_position_ids,
            "reject_1_labels": r1_labels,
            "reject_2_input_ids": r2_input_ids,
            "reject_2_attention_mask": r2_attention_mask,
            "reject_2_position_ids": r2_position_ids,
            "reject_2_labels": r2_labels,
        }

        if not self._validate_variant(result, sample.sample_id):
            return None

        # Metadata (not in training dataset)
        result["_sample_id"] = sample.sample_id
        result["_source_name"] = sample.source_name
        result["_variant_id"] = variant_index
        result["_position_strategy"] = strategy
        result["_real_token_length"] = len(chosen_input_ids)
        result["_max_position_id"] = max(
            max(chosen_position_ids), max(r1_position_ids), max(r2_position_ids)
        )
        result["_num_chunks"] = pp.num_chunks
        result["_chosen_answer_length"] = len(chosen_tokens)
        result["_rejected_1_length"] = len(rejected_1_tokens)
        result["_rejected_2_length"] = len(rejected_2_tokens)

        return result

    def _validate_variant(self, result: Dict[str, Any], sample_id: str) -> bool:
        """Run per-sample assertions. Returns True if all pass."""
        try:
            # 1. Field completeness
            actual_keys = {k for k in result if not k.startswith("_")}
            if actual_keys != EXPECTED_12_FIELDS:
                logger.warning(
                    "%s: expected 12 fields, got %d. Extra: %s, Missing: %s",
                    sample_id,
                    len(actual_keys),
                    actual_keys - EXPECTED_12_FIELDS,
                    EXPECTED_12_FIELDS - actual_keys,
                )
                return False

            # 2. Length consistency per branch
            for prefix in ("chosen", "reject_1", "reject_2"):
                ids_len = len(result[f"{prefix}_input_ids"])
                attn_len = len(result[f"{prefix}_attention_mask"])
                pos_len = len(result[f"{prefix}_position_ids"])
                labels_len = len(result[f"{prefix}_labels"])
                if not (ids_len == attn_len == pos_len == labels_len):
                    logger.warning(
                        "%s: length mismatch in %s branch: ids=%d attn=%d pos=%d labels=%d",
                        sample_id, prefix, ids_len, attn_len, pos_len, labels_len,
                    )
                    return False

            # 3. Max length
            if len(result["chosen_input_ids"]) > self.max_seq_length:
                logger.warning(
                    "%s: chosen length %d > max_seq_length %d",
                    sample_id, len(result["chosen_input_ids"]), self.max_seq_length,
                )
                return False

            # 4. Position validity
            for prefix in ("chosen", "reject_1", "reject_2"):
                pos = result[f"{prefix}_position_ids"]
                for p in pos:
                    if not (0 <= p < self.target_position_length):
                        logger.warning(
                            "%s: position %d out of range in %s", sample_id, p, prefix
                        )
                        return False

            # 5. Shared prompt: first len(prefix_ids) tokens must match
            chosen_ids = result["chosen_input_ids"]
            r1_ids = result["reject_1_input_ids"]
            r2_ids = result["reject_2_input_ids"]
            # Find where answers diverge
            min_len = min(len(chosen_ids), len(r1_ids), len(r2_ids))
            # The answer part is at the end; prefix is the shared part
            # We can verify by checking the shortest sequence's full overlap
            last_divergence = -1
            for i in range(min_len):
                if chosen_ids[i] != r1_ids[i] or chosen_ids[i] != r2_ids[i]:
                    last_divergence = i
            # Actually, let's be less strict: just check that the beginning
            # (system + user message part) matches
            # The diverging point should be after the assistant header

            # 6. Labels: prompt = -100, answer = input_ids
            for prefix in ("chosen", "reject_1", "reject_2"):
                ids = result[f"{prefix}_input_ids"]
                labels = result[f"{prefix}_labels"]
                answer_region = False
                for i, (tid, lbl) in enumerate(zip(ids, labels)):
                    if lbl != -100:
                        answer_region = True
                        if lbl != tid:
                            logger.warning(
                                "%s: label mismatch at pos %d in %s: id=%d label=%d",
                                sample_id, i, prefix, tid, lbl,
                            )
                            return False
                    elif answer_region:
                        # Once we enter answer region, all subsequent labels should be != -100
                        pass

            # 7. Answers not empty
            for prefix in ("chosen", "reject_1", "reject_2"):
                labels = result[f"{prefix}_labels"]
                answer_labels = [l for l in labels if l != -100]
                if not answer_labels:
                    logger.warning("%s: empty answer labels in %s", sample_id, prefix)
                    return False

            return True

        except Exception as exc:
            logger.warning("%s: validation exception: %s", sample_id, exc)
            return False

    # ------------------------------------------------------------------
    # Split processing
    # ------------------------------------------------------------------

    def _assign_strategies(self, total_variants: int) -> List[str]:
        """Return a list of 'continuous'/'sparse' strings with global ratio."""
        rng = random.Random(self.seed)
        n_continuous = round(total_variants * self.continuous_ratio)
        n_sparse = total_variants - n_continuous
        strategies = ["continuous"] * n_continuous + ["sparse"] * n_sparse
        rng.shuffle(strategies)
        return strategies

    def build_split(
        self,
        samples: List[NormalizedSample],
        split_name: str,
    ) -> Tuple[Dataset, List[Dict[str, Any]]]:
        """Process a train or test split.

        Returns ``(training_dataset, metadata_list)``.
        """
        total_variants = len(samples) * self.position_variants_per_sample
        strategies = self._assign_strategies(total_variants)
        strategy_iter = iter(strategies)

        records: List[Dict[str, Any]] = []
        metadata: List[Dict[str, Any]] = []

        for sample in samples:
            for vi in range(self.position_variants_per_sample):
                strategy = next(strategy_iter)
                result = self.build_single_variant(sample, vi, strategy)
                if result is not None:
                    # Separate metadata from training fields
                    meta = {k: v for k, v in result.items() if k.startswith("_")}
                    meta["split"] = split_name
                    metadata.append(meta)

                    record = {k: v for k, v in result.items() if not k.startswith("_")}
                    records.append(record)

                    if strategy == "continuous":
                        self.report.continuous_count += 1
                    else:
                        self.report.sparse_count += 1

        if split_name == "train":
            self.report.train_samples = len(records)
        else:
            self.report.test_samples = len(records)

        self.report.success_samples += len(records)

        logger.info(
            "Split '%s': %d samples → %d records (%d continuous, %d sparse)",
            split_name,
            len(samples),
            len(records),
            sum(1 for m in metadata if m["_position_strategy"] == "continuous"),
            sum(1 for m in metadata if m["_position_strategy"] == "sparse"),
        )

        # Convert to Dataset
        if records:
            ds = Dataset.from_list(records)
        else:
            ds = Dataset.from_dict({k: [] for k in EXPECTED_12_FIELDS})

        return ds, metadata

    # ------------------------------------------------------------------
    # Main build
    # ------------------------------------------------------------------

    def build(self, input_path: str) -> DatasetDict:
        """Run the full build pipeline."""
        if os.path.exists(self.output_path) and not self.overwrite:
            raise FileExistsError(
                f"Output path {self.output_path} exists. Use --overwrite to replace."
            )

        t_start = time.time()

        # 1. Load
        samples = self.load_input_data(input_path)

        # 2. Validate & filter
        valid_samples: List[NormalizedSample] = []
        for s in samples:
            reason = self.validate_sample(s)
            if reason is None:
                valid_samples.append(s)
            else:
                self.report.drop_reasons[reason] = self.report.drop_reasons.get(reason, 0) + 1
                self.report.dropped_samples += 1

        logger.info(
            "Validation: %d valid, %d dropped (reasons: %s)",
            len(valid_samples),
            self.report.dropped_samples,
            dict(self.report.drop_reasons),
        )

        if not valid_samples:
            raise ValueError("No valid samples after validation.")

        # 3. Split by sample_id
        rng = random.Random(self.seed)
        sample_ids = sorted(set(s.sample_id for s in valid_samples))
        rng.shuffle(sample_ids)

        n_test = max(1, int(len(sample_ids) * self.test_ratio))
        test_ids = set(sample_ids[:n_test])
        train_ids = set(sample_ids[n_test:])

        train_samples = [s for s in valid_samples if s.sample_id in train_ids]
        test_samples = [s for s in valid_samples if s.sample_id in test_ids]

        # Verify no leakage
        assert train_ids.isdisjoint(test_ids), "train/test sample_id leakage!"

        logger.info(
            "Split: %d train samples (%d unique IDs), %d test samples (%d unique IDs)",
            len(train_samples), len(train_ids),
            len(test_samples), len(test_ids),
        )

        # 4. Build each split
        train_ds, train_meta = self.build_split(train_samples, "train")
        test_ds, test_meta = self.build_split(test_samples, "test")

        all_metadata = train_meta + test_meta

        # 5. Compute report stats
        if all_metadata:
            lengths = [m["_real_token_length"] for m in all_metadata]
            self.report.avg_real_length = sum(lengths) / len(lengths)
            self.report.max_real_length = max(lengths)
            positions = [m["_max_position_id"] for m in all_metadata]
            self.report.avg_max_position = sum(positions) / len(positions)
            self.report.avg_chosen_answer_len = sum(
                m["_chosen_answer_length"] for m in all_metadata
            ) / len(all_metadata)
            self.report.avg_rejected_1_len = sum(
                m["_rejected_1_length"] for m in all_metadata
            ) / len(all_metadata)
            self.report.avg_rejected_2_len = sum(
                m["_rejected_2_length"] for m in all_metadata
            ) / len(all_metadata)

        # 6. Save
        dataset_dict = DatasetDict({"train": train_ds, "test": test_ds})
        os.makedirs(self.output_path, exist_ok=True)
        dataset_dict.save_to_disk(self.output_path)
        logger.info("Dataset saved to %s", self.output_path)

        # 7. Save config & report
        config = {
            "tokenizer_path": self.tokenizer.name_or_path,
            "input_path": input_path,
            "seed": self.seed,
            "max_seq_length": self.max_seq_length,
            "target_position_length": self.target_position_length,
            "num_chunks": self.num_chunks,
            "chunk_token_size": self.chunk_token_size,
            "max_answer_tokens": self.max_answer_tokens,
            "position_variants_per_sample": self.position_variants_per_sample,
            "continuous_ratio": self.continuous_ratio,
            "context_mode": self.context_mode,
            "test_ratio": self.test_ratio,
            "strict": self.strict,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_seconds": round(time.time() - t_start, 1),
        }
        with open(os.path.join(self.output_path, "build_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        report_dict = asdict(self.report)
        report_dict.pop("drop_reasons", None)
        report_dict["drop_reasons"] = dict(self.report.drop_reasons)
        with open(os.path.join(self.output_path, "build_report.json"), "w") as f:
            json.dump(report_dict, f, indent=2)

        # Save metadata
        with open(os.path.join(self.output_path, "metadata.jsonl"), "w") as f:
            for m in all_metadata:
                # Strip leading underscores from keys
                clean = {k.lstrip("_"): v for k, v in m.items()}
                f.write(json.dumps(clean) + "\n")

        logger.info("Config, report, and metadata saved.")

        return dataset_dict


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOGO Training Data Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input_path", required=True, help="Path to post-processed dataset(s)")
    p.add_argument("--output_path", required=True, help="Output directory for DatasetDict")
    p.add_argument("--tokenizer_path", required=True, help="HuggingFace tokenizer path")
    p.add_argument(
        "--model_type", default="llama-3", choices=list(PROMPT_ADAPTERS.keys()),
        help="Model type for prompt format (default: llama-3)"
    )
    p.add_argument(
        "--context_mode", default="existing", choices=["existing", "paper"],
        help="Context construction mode (default: existing)"
    )
    p.add_argument("--max_seq_length", type=int, default=10000)
    p.add_argument("--real_reference_tokens", type=int, default=8192)
    p.add_argument("--target_position_length", type=int, default=65536)
    p.add_argument("--num_chunks", type=int, default=16)
    p.add_argument("--chunk_token_size", type=int, default=512)
    p.add_argument("--max_answer_tokens", type=int, default=512)
    p.add_argument("--position_variants_per_sample", type=int, default=2)
    p.add_argument("--continuous_ratio", type=float, default=0.9)
    p.add_argument("--test_ratio", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_proc", type=int, default=1)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--no_deduplicate", dest="deduplicate_answers", action="store_false",
        help="Disable answer deduplication"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    builder = LogoDatasetBuilder(
        tokenizer_path=args.tokenizer_path,
        output_path=args.output_path,
        model_type=args.model_type,
        context_mode=args.context_mode,
        max_seq_length=args.max_seq_length,
        real_reference_tokens=args.real_reference_tokens,
        target_position_length=args.target_position_length,
        num_chunks=args.num_chunks,
        chunk_token_size=args.chunk_token_size,
        max_answer_tokens=args.max_answer_tokens,
        position_variants_per_sample=args.position_variants_per_sample,
        continuous_ratio=args.continuous_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        num_proc=args.num_proc,
        strict=args.strict,
        overwrite=args.overwrite,
        deduplicate_answers=args.deduplicate_answers,
    )

    dataset_dict = builder.build(input_path=args.input_path)
    logger.info("Done! Train: %d, Test: %d", len(dataset_dict["train"]), len(dataset_dict["test"]))


if __name__ == "__main__":
    main()
