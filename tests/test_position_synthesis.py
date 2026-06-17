"""
Tests for ``data/position_synthesis.py``.
"""

import pytest
import torch

from data.position_synthesis import (
    Layout,
    PositionValidationError,
    compute_layout,
    derive_variant_seed,
    synthesize_continuous_positions,
    synthesize_qa_tail_positions,
    synthesize_sparse_positions,
    validate_overall_positions,
    validate_positions,
)


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------

class TestLayout:
    def test_basic(self):
        layout = compute_layout(
            system_len=10,
            chunk_lens=[5, 5, 5],
            question_len=20,
            assistant_header_len=5,
            max_answer_len=30,
            K_target=1000,
        )
        assert layout.system_len == 10
        assert layout.num_chunks == 3
        assert layout.tail_len == 20 + 5 + 30  # question + header + max_answer
        assert layout.tail_start == 1000 - layout.tail_len
        assert layout.context_region_size == layout.tail_start - 10

    def test_slot_boundaries_non_overlapping(self):
        layout = compute_layout(10, [5, 5, 5], 20, 5, 30, 1000)
        boundaries = layout.slot_boundaries
        assert len(boundaries) == 3
        for i in range(len(boundaries) - 1):
            assert boundaries[i][1] <= boundaries[i + 1][0]  # no overlap

    def test_slot_boundaries_cover_context_region(self):
        layout = compute_layout(10, [3, 7], 20, 5, 30, 1000)
        b = layout.slot_boundaries
        assert b[0][0] == 10  # starts at system_len
        assert b[-1][1] == layout.tail_start  # ends at tail_start

    def test_zero_chunks(self):
        layout = compute_layout(10, [], 20, 5, 30, 1000)
        assert layout.num_chunks == 0
        assert layout.slot_boundaries == ()

    def test_single_chunk_fills_entire_region(self):
        layout = compute_layout(5, [10], 20, 5, 30, 1000)
        assert len(layout.slot_boundaries) == 1
        s, e = layout.slot_boundaries[0]
        assert s == 5
        assert e == layout.tail_start


# ---------------------------------------------------------------------------
# Continuous synthesis
# ---------------------------------------------------------------------------

class TestContinuousSynthesis:
    def test_basic(self):
        slot_boundaries = [(0, 100), (100, 200), (200, 300)]
        chunk_lens = [10, 15, 20]
        result = synthesize_continuous_positions(chunk_lens, slot_boundaries, seed=42)
        assert len(result) == 3
        for i, t in enumerate(result):
            assert t.numel() == chunk_lens[i]
            assert t.dtype == torch.long

    def test_within_slot(self):
        slot_boundaries = [(0, 50), (50, 100)]
        chunk_lens = [10, 10]
        result = synthesize_continuous_positions(chunk_lens, slot_boundaries, seed=42)
        for t, (s, e) in zip(result, slot_boundaries):
            assert t.min().item() >= s
            assert t.max().item() < e

    def test_consecutive_within_chunk(self):
        slot_boundaries = [(0, 100)]
        chunk_lens = [20]
        result = synthesize_continuous_positions(chunk_lens, slot_boundaries, seed=0)
        diffs = result[0][1:] - result[0][:-1]
        assert (diffs == 1).all(), f"Expected gap=1, got gaps {diffs.tolist()}"

    def test_chunk_fills_slot(self):
        """When chunk_len == slot_width, it should start at slot_start."""
        slot_boundaries = [(10, 20)]
        chunk_lens = [10]
        result = synthesize_continuous_positions(chunk_lens, slot_boundaries, seed=0)
        assert result[0][0].item() == 10

    def test_validate_passes_continuous(self):
        slot_boundaries = [(0, 50)]
        pos = synthesize_continuous_positions([10], slot_boundaries, seed=0)
        validate_positions(pos, slot_boundaries, strategy="continuous")

    def test_validate_fails_continuous_with_gaps(self):
        slot_boundaries = [(0, 50)]
        # Manually construct a position that would fail continuous check
        pos = [torch.tensor([0, 2, 4], dtype=torch.long)]
        with pytest.raises(PositionValidationError, match="continuous strategy"):
            validate_positions(pos, slot_boundaries, strategy="continuous")


# ---------------------------------------------------------------------------
# Sparse synthesis
# ---------------------------------------------------------------------------

class TestSparseSynthesis:
    def test_basic(self):
        slot_boundaries = [(0, 100), (100, 200)]
        chunk_lens = [5, 8]
        result = synthesize_sparse_positions(chunk_lens, slot_boundaries, seed=42)
        assert len(result) == 2
        for i, t in enumerate(result):
            assert t.numel() == chunk_lens[i]
            assert t.dtype == torch.long

    def test_within_slot(self):
        slot_boundaries = [(0, 50), (50, 100)]
        chunk_lens = [5, 5]
        result = synthesize_sparse_positions(chunk_lens, slot_boundaries, seed=42)
        for t, (s, e) in zip(result, slot_boundaries):
            assert t.min().item() >= s
            assert t.max().item() < e

    def test_strictly_increasing(self):
        slot_boundaries = [(0, 200)]
        chunk_lens = [30]
        result = synthesize_sparse_positions(chunk_lens, slot_boundaries, seed=0)
        diffs = result[0][1:] - result[0][:-1]
        assert (diffs > 0).all()

    def test_no_duplicates(self):
        slot_boundaries = [(0, 50)]
        chunk_lens = [30]
        result = synthesize_sparse_positions(chunk_lens, slot_boundaries, seed=0)
        assert result[0].unique().numel() == result[0].numel()

    def test_validate_passes_sparse(self):
        slot_boundaries = [(0, 200)]
        pos = synthesize_sparse_positions([30], slot_boundaries, seed=0)
        validate_positions(pos, slot_boundaries, strategy="sparse")

    def test_validate_fails_out_of_bounds(self):
        slot_boundaries = [(0, 50)]
        pos = [torch.tensor([0, 10, 60], dtype=torch.long)]  # 60 > 49
        with pytest.raises(PositionValidationError, match="outside slot"):
            validate_positions(pos, slot_boundaries, strategy="sparse")

    def test_validate_fails_non_increasing(self):
        slot_boundaries = [(0, 100)]
        pos = [torch.tensor([10, 5, 20], dtype=torch.long)]  # not increasing
        with pytest.raises(PositionValidationError, match="strictly increasing"):
            validate_positions(pos, slot_boundaries, strategy="sparse")


# ---------------------------------------------------------------------------
# Seed reproducibility
# ---------------------------------------------------------------------------

class TestSeedReproducibility:
    def test_continuous_same_seed_same_result(self):
        sb = [(0, 100), (100, 200)]
        cl = [10, 10]
        r1 = synthesize_continuous_positions(cl, sb, seed=42)
        r2 = synthesize_continuous_positions(cl, sb, seed=42)
        for t1, t2 in zip(r1, r2):
            assert torch.equal(t1, t2)

    def test_continuous_different_seed_different_result(self):
        sb = [(0, 1000)]
        cl = [10]
        r1 = synthesize_continuous_positions(cl, sb, seed=1)
        r2 = synthesize_continuous_positions(cl, sb, seed=2)
        # Should differ with high probability
        assert not torch.equal(r1[0], r2[0])

    def test_sparse_same_seed_same_result(self):
        sb = [(0, 200)]
        cl = [30]
        r1 = synthesize_sparse_positions(cl, sb, seed=42)
        r2 = synthesize_sparse_positions(cl, sb, seed=42)
        for t1, t2 in zip(r1, r2):
            assert torch.equal(t1, t2)

    def test_sparse_different_seed_different_result(self):
        sb = [(0, 500)]
        cl = [50]
        r1 = synthesize_sparse_positions(cl, sb, seed=42)
        r2 = synthesize_sparse_positions(cl, sb, seed=99)
        assert not torch.equal(r1[0], r2[0])

    def test_derive_variant_seed_deterministic(self):
        s1 = derive_variant_seed(42, "sample_001", 0)
        s2 = derive_variant_seed(42, "sample_001", 0)
        assert s1 == s2

    def test_derive_variant_seed_different(self):
        s1 = derive_variant_seed(42, "sample_001", 0)
        s2 = derive_variant_seed(42, "sample_001", 1)
        assert s1 != s2


# ---------------------------------------------------------------------------
# QA tail positions
# ---------------------------------------------------------------------------

class TestQATail:
    def test_continuous(self):
        pos = synthesize_qa_tail_positions(
            tail_start=1000, question_len=10, assistant_header_len=5, answer_len=20
        )
        assert pos.numel() == 35
        assert pos[0].item() == 1000
        assert pos[-1].item() == 1034
        diffs = pos[1:] - pos[:-1]
        assert (diffs == 1).all()


# ---------------------------------------------------------------------------
# Overall validation
# ---------------------------------------------------------------------------

class TestOverallValidation:
    def test_passes(self):
        pos = torch.arange(0, 100, dtype=torch.long)
        validate_overall_positions(pos, K_target=200, label="test")

    def test_fails_negative(self):
        pos = torch.tensor([-1, 0, 1], dtype=torch.long)
        with pytest.raises(PositionValidationError, match="negative"):
            validate_overall_positions(pos, K_target=200)

    def test_fails_exceeds_target(self):
        pos = torch.arange(0, 500, dtype=torch.long)
        with pytest.raises(PositionValidationError, match="exceeds K_target"):
            validate_overall_positions(pos, K_target=200)

    def test_fails_non_increasing(self):
        pos = torch.tensor([0, 2, 1], dtype=torch.long)
        with pytest.raises(PositionValidationError, match="strictly increasing"):
            validate_overall_positions(pos, K_target=100)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_chunks(self):
        """Zero chunks should produce empty result."""
        result = synthesize_continuous_positions([], [], seed=0)
        assert result == []

    def test_single_chunk(self):
        sb = [(0, 100)]
        cl = [10]
        result = synthesize_continuous_positions(cl, sb, seed=0)
        assert len(result) == 1
        assert result[0].numel() == 10

    def test_chunk_zero_tokens(self):
        """A zero-token chunk should produce empty position tensor."""
        sb = [(0, 50)]
        cl = [0]
        result = synthesize_continuous_positions(cl, sb, seed=0)
        assert result[0].numel() == 0

    def test_chunk_larger_than_slot_continuous(self):
        """When chunk > slot, it fills the entire slot."""
        sb = [(0, 5)]
        cl = [10]
        result = synthesize_continuous_positions(cl, sb, seed=0)
        assert result[0].numel() == 10
        # Should start at slot_start and fill as much as possible
        assert result[0][0].item() == 0

    def test_chunk_larger_than_slot_sparse(self):
        """When chunk > slot in sparse mode, use all available positions."""
        sb = [(0, 5)]
        cl = [10]
        result = synthesize_sparse_positions(cl, sb, seed=0)
        # Can only use min(chunk_len, slot_width) positions
        assert result[0].numel() <= 5

    def test_many_small_chunks(self):
        n = 16
        sb = [(i * 50, (i + 1) * 50) for i in range(n)]
        cl = [20] * n
        result = synthesize_continuous_positions(cl, sb, seed=42)
        assert len(result) == n
        for t, (s, e) in zip(result, sb):
            assert t.min().item() >= s
            assert t.max().item() < e

    def test_target_small_position_window(self):
        """K_target=100 with system+chunks+qa fitting."""
        layout = compute_layout(
            system_len=5,
            chunk_lens=[5, 5],
            question_len=10,
            assistant_header_len=5,
            max_answer_len=10,
            K_target=100,
        )
        assert layout.tail_start > layout.system_len
        assert layout.tail_start < 100
