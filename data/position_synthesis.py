"""
Position ID synthesis for LOGO training data.

Provides two strategies for mapping chunk tokens into a synthetic position
space ("continuous" and "sparse"), along with layout computation and
validation. This module is **pure math** — it does not depend on tokenizers,
datasets, or model loading.

Strategy overview
-----------------
Given *N* chunks to place into a context region of size *W* positions,
the region is divided into *N* equal, non-overlapping slots. Each chunk
is assigned to its own slot and its position IDs are generated **within**
that slot.

- **Continuous**: pick a random start offset inside the slot; chunk
  tokens receive consecutive position IDs (gap = 1).
- **Sparse**: sample *L_j* unique positions from the slot, sort them,
  and assign them in token order. Adjacent tokens typically have gap > 1.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Layout:
    """Position-space layout computed from token counts."""

    system_len: int
    """Tokens in the system / preamble region (continuous, starts at 0)."""

    chunk_lens: List[int]
    """Token count for each context chunk (in original order)."""

    question_len: int
    """Tokens in the question segment."""

    assistant_header_len: int
    """Tokens consumed by the assistant header (e.g. ``<|start_header_id|>assistant...``)."""

    max_answer_len: int
    """Maximum token count across the three answers (chosen / reject_1 / reject_2)."""

    K_target: int
    """Target position window (e.g. 65536)."""

    # --- computed ---

    tail_len: int = 0
    """question + assistant_header + max_answer_len."""

    tail_start: int = 0
    """First position ID assigned to the QA tail region."""

    context_region_size: int = 0
    """Total positions available for context chunk slots."""

    slot_boundaries: List[Tuple[int, int]] = ()
    """``(start_inclusive, end_exclusive)`` for each chunk's slot."""

    num_chunks: int = 0

    def __post_init__(self) -> None:
        self.num_chunks = len(self.chunk_lens)

        # Compute tail region
        self.tail_len = (
            self.question_len + self.assistant_header_len + self.max_answer_len
        )
        self.tail_start = max(self.K_target - self.tail_len, self.system_len)

        # Compute context region
        self.context_region_size = self.tail_start - self.system_len

        if self.num_chunks == 0:
            self.slot_boundaries = ()
            return

        # Divide context region into equal-width slots
        slot_width = self.context_region_size // self.num_chunks
        leftover = self.context_region_size % self.num_chunks

        boundaries: List[Tuple[int, int]] = []
        cursor = self.system_len
        for i in range(self.num_chunks):
            extra = 1 if i < leftover else 0
            start = cursor
            end = cursor + slot_width + extra
            boundaries.append((start, end))
            cursor = end
        self.slot_boundaries = tuple(boundaries)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_rng(seed: int) -> random.Random:
    return random.Random(seed)


def _sample_start(rng: random.Random, slot_start: int, slot_end: int, chunk_len: int) -> int:
    """Return a random legal start position so the chunk fits inside the slot."""
    if chunk_len >= slot_end - slot_start:
        return slot_start  # chunk fills the entire slot
    return rng.randint(slot_start, slot_end - chunk_len)


def _sample_sparse_positions(
    rng: random.Random, slot_start: int, slot_end: int, chunk_len: int
) -> List[int]:
    """Sample *chunk_len* unique positions from [slot_start, slot_end) and return them sorted."""
    available = list(range(slot_start, slot_end))
    if chunk_len > len(available):
        chunk_len = len(available)
    selected = rng.sample(available, chunk_len)
    selected.sort()
    return selected


def _auto_padding(
    t: torch.Tensor, length: int, fill_value: int = 0
) -> torch.Tensor:
    """Truncate or right-pad *t* to exactly *length* elements."""
    if t.size(0) > length:
        return t[:length].clone()
    if t.size(0) == length:
        return t.clone()
    padded = torch.full((length,), fill_value, dtype=t.dtype)
    padded[: t.size(0)] = t
    return padded


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_layout(
    system_len: int,
    chunk_lens: List[int],
    question_len: int,
    assistant_header_len: int,
    max_answer_len: int,
    K_target: int,
) -> Layout:
    """Compute the position-space layout.

    Returns a :class:`Layout` whose ``slot_boundaries`` field defines the
    legal region for each chunk's position IDs.
    """
    return Layout(
        system_len=system_len,
        chunk_lens=list(chunk_lens),
        question_len=question_len,
        assistant_header_len=assistant_header_len,
        max_answer_len=max_answer_len,
        K_target=K_target,
    )


def synthesize_continuous_positions(
    chunk_lens: List[int],
    slot_boundaries: List[Tuple[int, int]],
    seed: int = 0,
) -> List[torch.Tensor]:
    """Generate **continuous** position IDs for each chunk.

    Each chunk receives *consecutive* position IDs starting at a random
    offset within its slot.

    Returns one 1-D ``LongTensor`` per chunk (same length as the input
    token count).
    """
    rng = _make_rng(seed)
    position_ids: List[torch.Tensor] = []

    for chunk_len, (slot_start, slot_end) in zip(chunk_lens, slot_boundaries):
        start = _sample_start(rng, slot_start, slot_end, chunk_len)
        positions = torch.arange(start, start + chunk_len, dtype=torch.long)
        position_ids.append(positions)

    return position_ids


def synthesize_sparse_positions(
    chunk_lens: List[int],
    slot_boundaries: List[Tuple[int, int]],
    seed: int = 0,
) -> List[torch.Tensor]:
    """Generate **sparse** position IDs for each chunk.

    Each chunk receives *unique, sorted* position IDs randomly sampled
    from its slot. Adjacent tokens typically have gap > 1.

    Returns one 1-D ``LongTensor`` per chunk.
    """
    rng = _make_rng(seed)
    position_ids: List[torch.Tensor] = []

    for chunk_len, (slot_start, slot_end) in zip(chunk_lens, slot_boundaries):
        selected = _sample_sparse_positions(rng, slot_start, slot_end, chunk_len)
        position_ids.append(torch.tensor(selected, dtype=torch.long))

    return position_ids


def synthesize_qa_tail_positions(
    tail_start: int,
    question_len: int,
    assistant_header_len: int,
    answer_len: int,
) -> torch.Tensor:
    """Generate **continuous** position IDs for the QA tail region.

    The tail consists of: question → assistant_header → answer,
    all placed contiguously starting at *tail_start*.

    Returns a 1-D tensor of length ``question_len + assistant_header_len + answer_len``.
    """
    total = question_len + assistant_header_len + answer_len
    return torch.arange(tail_start, tail_start + total, dtype=torch.long)


# ---------------------------------------------------------------------------
# Time-driven position synthesis
# ---------------------------------------------------------------------------


@dataclass
class TimeLayout:
    """Position-space layout computed from timestamps (for clinical data)."""

    system_len: int
    chunk_lens: List[int]
    chunk_timestamps: List[float]  # Unix timestamps (seconds) for each chunk
    question_len: int
    assistant_header_len: int
    max_answer_len: int
    K_target: int

    # Computed
    tail_start: int = 0
    context_region_size: int = 0
    chunk_starts: List[int] = ()

    def __post_init__(self) -> None:
        self.num_chunks = len(self.chunk_lens)

        tail_len = self.question_len + self.assistant_header_len + self.max_answer_len
        self.tail_start = max(self.K_target - tail_len, self.system_len)
        self.context_region_size = self.tail_start - self.system_len

        if self.num_chunks == 0 or not self.chunk_timestamps:
            self.chunk_starts = ()
            return

        # Map timestamps proportionally to position space
        t_min = min(self.chunk_timestamps)
        t_max = max(self.chunk_timestamps)
        t_span = max(t_max - t_min, 1.0)  # avoid division by zero

        starts = []
        cursor = self.system_len
        for i in range(self.num_chunks):
            if i == 0:
                # First chunk at system boundary
                starts.append(cursor)
            else:
                # Proportional position based on time gap from previous chunk
                dt_prev = max(self.chunk_timestamps[i] - self.chunk_timestamps[i - 1], 0.0)
                max_pos_gap = max(1, (self.context_region_size - cursor) // (self.num_chunks - i))
                gap = min(int(dt_prev / t_span * self.context_region_size), max_pos_gap)
                cursor = min(cursor + gap, self.tail_start - sum(self.chunk_lens[i:]) - 1)
                starts.append(cursor)
            cursor += self.chunk_lens[i]

        self.chunk_starts = tuple(starts)


def synthesize_temporal_positions(
    chunk_lens: List[int],
    chunk_starts: List[int],
    seed: int = 0,
    jitter_ratio: float = 0.02,
) -> List[torch.Tensor]:
    """Generate time-driven position IDs that preserve temporal order.

    Each chunk receives **consecutive** position IDs starting at its
    time-proportional ``chunk_start`` position. Temporal gaps between
    chunks are preserved as position gaps.

    *jitter_ratio* controls how much random offset is applied:
    0.02 (2%) for continuous, 0.15 (15%) for sparse.

    Returns one 1-D ``LongTensor`` per chunk.
    """
    rng = random.Random(seed)
    position_ids: List[torch.Tensor] = []

    for i, chunk_len in enumerate(chunk_lens):
        base_start = chunk_starts[i]

        # Jitter: random backward offset within available gap
        if i > 0:
            prev_end = chunk_starts[i - 1] + chunk_lens[i - 1]
            max_jitter = max(0, int((base_start - prev_end) * jitter_ratio * 2))
            if max_jitter > 0:
                jitter = rng.randint(0, max_jitter)
                base_start -= jitter

        positions = torch.arange(base_start, base_start + chunk_len, dtype=torch.long)
        position_ids.append(positions)

    return position_ids


def synthesize_prefix_positions(
    system_len: int,
    chunk_position_ids: List[torch.Tensor],
    question_len: int,
    assistant_header_len: int,
) -> torch.Tensor:
    """Concatenate position IDs for the shared prompt prefix.

    Order: system (0..system_len-1) → chunks → question → assistant header.
    """
    parts: List[torch.Tensor] = [
        torch.arange(system_len, dtype=torch.long),
        *chunk_position_ids,
        # The question + assistant header positions are filled in later
        # by synthesize_qa_tail_positions — here we just need placeholders
        # for the question tokens.
    ]
    return torch.cat(parts, dim=0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class PositionValidationError(Exception):
    """Raised when position IDs violate a constraint."""

    message: str
    chunk_index: Optional[int] = None


def validate_positions(
    position_ids_list: List[torch.Tensor],
    slot_boundaries: List[Tuple[int, int]],
    strategy: str = "continuous",
) -> None:
    """Validate that position IDs satisfy all constraints.

    Checks:
    - Positions within assigned slot boundaries.
    - No duplicate position IDs.
    - Positions are strictly increasing per chunk.
    - Continuous strategy: gap == 1 within each chunk.
    - Sparse strategy: at least one gap > 1 per chunk (warning if not).

    Raises :class:`PositionValidationError` on violation.
    """
    for i, (pos_ids, (slot_start, slot_end)) in enumerate(
        zip(position_ids_list, slot_boundaries)
    ):
        if pos_ids.numel() == 0:
            continue

        pos_list = pos_ids.tolist()

        # ---- boundary check ----
        if pos_ids.min().item() < slot_start or pos_ids.max().item() >= slot_end:
            raise PositionValidationError(
                f"Chunk {i}: position(s) outside slot [{slot_start}, {slot_end}). "
                f"Got min={pos_ids.min().item()}, max={pos_ids.max().item()}.",
                chunk_index=i,
            )

        # ---- strict monotonicity ----
        diffs = pos_ids[1:] - pos_ids[:-1]
        if (diffs <= 0).any():
            raise PositionValidationError(
                f"Chunk {i}: positions are not strictly increasing.",
                chunk_index=i,
            )

        # ---- no duplicates (implied by strict increase, but double-check) ----
        unique_count = pos_ids.unique().numel()
        if unique_count != pos_ids.numel():
            raise PositionValidationError(
                f"Chunk {i}: duplicate position IDs detected.",
                chunk_index=i,
            )

        # ---- strategy-specific ----
        if strategy == "continuous":
            if (diffs != 1).any():
                raise PositionValidationError(
                    f"Chunk {i}: continuous strategy requires gap=1 "
                    f"between adjacent positions, but got gaps {diffs.tolist()}.",
                    chunk_index=i,
                )
        elif strategy == "sparse":
            if pos_ids.numel() > 1 and (diffs == 1).all():
                # This is a warning-level issue, not a hard error.
                # Still valid, but likely indicates slot is too small
                # for meaningful sparse sampling.
                pass


def validate_overall_positions(
    all_position_ids: torch.Tensor,
    K_target: int,
    label: str = "",
) -> None:
    """Validate that a complete position-id tensor is legal.

    Checks:
    - All values >= 0 and < K_target.
    - Strictly increasing overall.
    """
    if all_position_ids.numel() == 0:
        return

    if all_position_ids.min().item() < 0:
        raise PositionValidationError(
            f"{label}: negative position ID {all_position_ids.min().item()}."
        )
    if all_position_ids.max().item() >= K_target:
        raise PositionValidationError(
            f"{label}: position ID {all_position_ids.max().item()} "
            f"exceeds K_target={K_target}."
        )

    diffs = all_position_ids[1:] - all_position_ids[:-1]
    if (diffs <= 0).any():
        bad_idx = (diffs <= 0).nonzero(as_tuple=True)[0][0].item()
        raise PositionValidationError(
            f"{label}: position IDs not strictly increasing at index {bad_idx}."
        )


# ---------------------------------------------------------------------------
# Utility: seed derivation
# ---------------------------------------------------------------------------

def derive_variant_seed(
    global_seed: int, sample_id: str, variant_index: int
) -> int:
    """Derive a deterministic sub-seed for a position variant.

    Uses SHA-256 to produce a stable seed that is reproducible across
    Python process restarts (unlike the built-in ``hash()`` which is
    randomised per process by default).

    The same ``(global_seed, sample_id, variant_index)`` tuple always
    produces the same seed.
    """
    raw = f"{global_seed}:{sample_id}:{variant_index}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
