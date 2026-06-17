"""
Tests for ``data/build_logo_dataset.py``.
"""

import json
import os

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from data.build_logo_dataset import (
    EXPECTED_12_FIELDS,
    LogoDatasetBuilder,
    NormalizedSample,
    _answers_equivalent,
    _normalize_answer,
)
from training.custom_dataset import SimPODataCollator


# Tokenizer path for tests (only used as identifier; mock is injected)
_TOKENIZER_PATH = "mock-llama3-tokenizer"


def _make_mock_dataset(num_samples: int = 10) -> Dataset:
    """Create a minimal post-process-like dataset for testing."""
    records = []
    for i in range(num_samples):
        records.append(
            {
                "all_ref_text": [
                    f"Reference chunk {i}_0",
                    f"Reference chunk {i}_1",
                    f"Reference chunk {i}_2",
                ],
                "combined_question": f"What is the capital of France sample {i}",
                "final_answer": f"Paris",
                "prefix_a": f"Berlin",
                "suffix_a": f"Rome",
                "label": f"Paris",
            }
        )
    return Dataset.from_list(records)


def _make_builder(tmp_path, mock_tokenizer, **kwargs):
    """Create a LogoDatasetBuilder with mock tokenizer."""
    defaults = dict(
        tokenizer_path=_TOKENIZER_PATH,
        output_path=str(tmp_path / "output"),
        position_variants_per_sample=1,
        seed=42,
        num_chunks=3,
        max_seq_length=4096,
        target_position_length=8192,
        chunk_token_size=128,
    )
    defaults.update(kwargs)
    defaults["tokenizer"] = mock_tokenizer
    return LogoDatasetBuilder(**defaults)


# ---------------------------------------------------------------------------
# Answer normalization
# ---------------------------------------------------------------------------


class TestAnswerNormalization:
    def test_strip_whitespace(self):
        assert _normalize_answer("  hello world  ") == "hello world"

    def test_collapse_whitespace(self):
        assert _normalize_answer("hello    world") == "hello world"

    def test_strip_eot(self):
        assert _normalize_answer("Paris<|eot_id|>") == "Paris"

    def test_case_insensitive_match(self):
        assert _answers_equivalent("Paris", "paris")

    def test_different_answers(self):
        assert not _answers_equivalent("Paris", "Berlin")


# ---------------------------------------------------------------------------
# Field normalization
# ---------------------------------------------------------------------------


class TestFieldNormalization:
    def test_suffix_a_typo_compat(self):
        item = {
            "all_ref_text": ["chunk1"],
            "combined_question": "q",
            "final_answer": "a1",
            "prefix_a": "a2",
            "siffix_a": "a3",
        }
        rejected_2 = str(item.get("suffix_a", item.get("siffix_a", ""))).strip()
        assert rejected_2 == "a3"

    def test_suffix_a_preferred_over_siffix(self):
        item = {
            "all_ref_text": ["chunk1"],
            "combined_question": "q",
            "final_answer": "a1",
            "prefix_a": "a2",
            "suffix_a": "correct",
            "siffix_a": "typo",
        }
        rejected_2 = str(item.get("suffix_a", item.get("siffix_a", ""))).strip()
        assert rejected_2 == "correct"


# ---------------------------------------------------------------------------
# Builder: input loading
# ---------------------------------------------------------------------------


class TestBuilderLoad:
    def test_load_single_dataset(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "test_data")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        samples = builder.load_input_data(ds_path)
        assert len(samples) == 10
        assert all(isinstance(s, NormalizedSample) for s in samples)

    def test_samples_have_unique_ids(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "test_data")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        samples = builder.load_input_data(ds_path)
        ids = [s.sample_id for s in samples]
        assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Builder: validation
# ---------------------------------------------------------------------------


class TestBuilderValidation:
    def test_valid_sample_passes(self, tmp_path, mock_tokenizer):
        builder = _make_builder(tmp_path, mock_tokenizer)
        sample = NormalizedSample(
            sample_id="test1",
            source_name="src",
            context_chunks=["chunk1"],
            question="q?",
            chosen_answer="a1",
            rejected_answer_1="a2",
            rejected_answer_2="a3",
        )
        assert builder.validate_sample(sample) is None

    def test_empty_context_fails(self, tmp_path, mock_tokenizer):
        builder = _make_builder(tmp_path, mock_tokenizer)
        sample = NormalizedSample("t", "s", [], "q", "a1", "a2", "a3")
        assert builder.validate_sample(sample) == "empty_context_chunks"

    def test_duplicate_answers_dropped(self, tmp_path, mock_tokenizer):
        builder = _make_builder(tmp_path, mock_tokenizer)
        sample = NormalizedSample("t", "s", ["c"], "q", "a1", "a1", "a3")
        assert builder.validate_sample(sample) == "chosen_equals_rejected_1"

    def test_duplicate_can_be_disabled(self, tmp_path, mock_tokenizer):
        builder = _make_builder(tmp_path, mock_tokenizer, deduplicate_answers=False)
        sample = NormalizedSample("t", "s", ["c"], "q", "a1", "a1", "a3")
        assert builder.validate_sample(sample) is None


# ---------------------------------------------------------------------------
# Builder: end-to-end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_build_and_save(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(
            tmp_path, mock_tokenizer,
            position_variants_per_sample=2,
            continuous_ratio=0.5,
            test_ratio=0.2,
        )
        dsd = builder.build(ds_path)

        assert isinstance(dsd, DatasetDict)
        assert "train" in dsd
        assert "test" in dsd
        assert len(dsd["train"]) > 0
        assert len(dsd["test"]) > 0

    def test_output_has_12_fields(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        dsd = builder.build(ds_path)

        for split in ("train", "test"):
            for row in dsd[split]:
                actual_keys = set(row.keys())
                assert actual_keys == EXPECTED_12_FIELDS, (
                    f"{split}: expected {EXPECTED_12_FIELDS}, got {actual_keys}"
                )

    def test_save_load_roundtrip(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        builder.build(ds_path)

        reloaded = load_from_disk(builder.output_path)
        assert isinstance(reloaded, DatasetDict)
        assert "train" in reloaded
        assert "test" in reloaded

    def test_aux_files_written(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        builder.build(ds_path)

        assert os.path.isfile(os.path.join(builder.output_path, "build_config.json"))
        assert os.path.isfile(os.path.join(builder.output_path, "build_report.json"))
        assert os.path.isfile(os.path.join(builder.output_path, "metadata.jsonl"))


# ---------------------------------------------------------------------------
# Shared prompt verification
# ---------------------------------------------------------------------------


class TestSharedPrompt:
    def test_shared_prompt_prefix(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        dsd = builder.build(ds_path)

        for row in dsd["train"]:
            c_ids = row["chosen_input_ids"]
            r1_ids = row["reject_1_input_ids"]
            r2_ids = row["reject_2_input_ids"]

            diverge_idx = None
            for i in range(min(len(c_ids), len(r1_ids))):
                if c_ids[i] != r1_ids[i]:
                    diverge_idx = i
                    break

            if diverge_idx is not None and diverge_idx > 0:
                assert c_ids[:diverge_idx] == r1_ids[:diverge_idx]
                assert c_ids[:diverge_idx] == r2_ids[:diverge_idx]

                c_labels = row["chosen_labels"]
                assert all(l == -100 for l in c_labels[:diverge_idx])


# ---------------------------------------------------------------------------
# Labels verification
# ---------------------------------------------------------------------------


class TestLabels:
    def test_prompt_labels_are_neg100(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        dsd = builder.build(ds_path)

        for row in dsd["train"]:
            for prefix in ("chosen", "reject_1", "reject_2"):
                labels = row[f"{prefix}_labels"]
                ids = row[f"{prefix}_input_ids"]

                # All -100 labels should be in the prompt prefix
                in_answer = False
                for tid, lbl in zip(ids, labels):
                    if lbl != -100:
                        in_answer = True
                        assert lbl == tid, f"Label mismatch: {lbl} != {tid}"

                answer_labels = [l for l in labels if l != -100]
                assert len(answer_labels) > 0, f"{prefix} has no answer labels"


# ---------------------------------------------------------------------------
# Position validation
# ---------------------------------------------------------------------------


class TestPositionValidation:
    def test_positions_in_range(self, tmp_path, mock_tokenizer):
        target = 8192
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer, target_position_length=target)
        dsd = builder.build(ds_path)

        for row in dsd["train"]:
            for prefix in ("chosen", "reject_1", "reject_2"):
                positions = row[f"{prefix}_position_ids"]
                assert all(0 <= p < target for p in positions), (
                    f"Position out of range: min={min(positions)}, max={max(positions)}"
                )

    def test_positions_increasing(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        dsd = builder.build(ds_path)

        for row in dsd["train"]:
            for prefix in ("chosen", "reject_1", "reject_2"):
                positions = row[f"{prefix}_position_ids"]
                for i in range(1, len(positions)):
                    assert positions[i] > positions[i - 1], (
                        f"Position not increasing at index {i}: "
                        f"{positions[i-1]} >= {positions[i]}"
                    )


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


class TestTrainTestSplit:
    def test_no_sample_id_leakage(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(
            tmp_path, mock_tokenizer,
            position_variants_per_sample=2,
            test_ratio=0.3,
        )
        builder.build(ds_path)

        meta_path = os.path.join(builder.output_path, "metadata.jsonl")
        train_ids = set()
        test_ids = set()
        with open(meta_path) as f:
            for line in f:
                m = json.loads(line)
                sid = m["sample_id"]
                if m["split"] == "train":
                    train_ids.add(sid)
                else:
                    test_ids.add(sid)

        assert train_ids.isdisjoint(test_ids), "Sample IDs leaked between train/test!"


# ---------------------------------------------------------------------------
# Seed determinism
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    def test_same_seed_same_output(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        out1 = str(tmp_path / "out1")
        out2 = str(tmp_path / "out2")

        for out in (out1, out2):
            builder = _make_builder(
                tmp_path, mock_tokenizer,
                output_path=out,
            )
            builder.build(ds_path)

        dsd1 = load_from_disk(out1)
        dsd2 = load_from_disk(out2)

        for split in ("train", "test"):
            for r1, r2 in zip(dsd1[split], dsd2[split]):
                for key in EXPECTED_12_FIELDS:
                    assert r1[key] == r2[key], f"Mismatch in {key}"


# ---------------------------------------------------------------------------
# Overwrite protection
# ---------------------------------------------------------------------------


class TestOverwriteProtection:
    def test_refuses_overwrite_by_default(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        out_path = str(tmp_path / "output")
        builder1 = _make_builder(tmp_path, mock_tokenizer, output_path=out_path)
        builder1.build(ds_path)

        builder2 = _make_builder(tmp_path, mock_tokenizer, output_path=out_path, overwrite=False)
        with pytest.raises(FileExistsError):
            builder2.build(ds_path)

    def test_overwrite_flag_works(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        out_path = str(tmp_path / "output")
        builder1 = _make_builder(tmp_path, mock_tokenizer, output_path=out_path)
        builder1.build(ds_path)

        builder2 = _make_builder(tmp_path, mock_tokenizer, output_path=out_path, overwrite=True)
        builder2.build(ds_path)


# ---------------------------------------------------------------------------
# SimPODataCollator compatibility
# ---------------------------------------------------------------------------


class TestCollatorCompatibility:
    def test_collate_two_samples(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer)
        dsd = builder.build(ds_path)

        samples = list(dsd["train"].select(range(2)))
        assert len(samples) == 2

        collator = SimPODataCollator(max_seq_length=4096)
        batch = collator(samples)

        for prefix in ("chosen", "reject_1", "reject_2"):
            ids = batch[f"{prefix}_input_ids"]
            attn = batch[f"{prefix}_attention_mask"]
            pos = batch[f"{prefix}_position_ids"]
            labels = batch[f"{prefix}_labels"]

            assert ids.ndim == 2
            assert ids.shape[0] == 2
            assert attn.shape == ids.shape
            assert pos.shape == ids.shape
            assert labels.shape == ids.shape

            # Labels at padding positions should be -100
            pad_mask = attn == 0
            if pad_mask.any():
                assert (labels[pad_mask] == -100).all(), (
                    f"{prefix}: labels at padding positions should be -100"
                )


# ---------------------------------------------------------------------------
# Smoke training data check
# ---------------------------------------------------------------------------


class TestSmokeTraining:
    def test_data_loads_and_collates(self, tmp_path, mock_tokenizer):
        ds = _make_mock_dataset(10)
        ds_path = str(tmp_path / "input")
        ds.save_to_disk(ds_path)

        builder = _make_builder(tmp_path, mock_tokenizer, max_seq_length=2048, target_position_length=4096)
        dsd = builder.build(ds_path)

        if len(dsd["train"]) < 2:
            pytest.skip("Not enough training samples")

        reloaded = load_from_disk(builder.output_path)
        assert "train" in reloaded

        collator = SimPODataCollator(max_seq_length=2048)
        batch = collator([reloaded["train"][0]])

        assert batch["chosen_input_ids"].shape[0] == 1
        assert batch["reject_1_input_ids"].shape[0] == 1
        assert batch["reject_2_input_ids"].shape[0] == 1
        assert batch["chosen_input_ids"].shape == batch["chosen_labels"].shape
