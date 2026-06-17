#!/usr/bin/env python3
"""
LOGO Dataset Inspector.

Validates and displays statistics for a built LOGO training dataset.
Usage::

    python data/inspect_logo_dataset.py --dataset_path ./logo_train_data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

from datasets import DatasetDict, load_from_disk

# Expected 12 training fields
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

BRANCH_PREFIXES = ("chosen", "reject_1", "reject_2")


def load_dataset(path: str) -> DatasetDict:
    """Load a DatasetDict from disk."""
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Dataset path not found: {path}")
    dsd = load_from_disk(path)
    if not isinstance(dsd, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dsd).__name__}")
    return dsd


def check_field_completeness(dsd: DatasetDict) -> List[str]:
    """Verify every sample has exactly the 12 expected fields."""
    errors: List[str] = []
    for split_name, ds in dsd.items():
        for i, row in enumerate(ds):
            actual = set(row.keys())
            if actual != EXPECTED_12_FIELDS:
                missing = EXPECTED_12_FIELDS - actual
                extra = actual - EXPECTED_12_FIELDS
                errors.append(
                    f"[{split_name}:{i}] field mismatch: "
                    f"missing={missing}, extra={extra}"
                )
    if not errors:
        print(f"  [PASS] Field completeness: all samples have exactly 12 fields.")
    else:
        print(f"  [FAIL] Field completeness: {len(errors)} errors:")
        for e in errors[:10]:
            print(f"    {e}")
    return errors


def check_length_consistency(dsd: DatasetDict) -> List[str]:
    """Check that each branch has consistent lengths."""
    errors: List[str] = []
    for split_name, ds in dsd.items():
        for i, row in enumerate(ds):
            for prefix in BRANCH_PREFIXES:
                ids_len = len(row[f"{prefix}_input_ids"])
                attn_len = len(row[f"{prefix}_attention_mask"])
                pos_len = len(row[f"{prefix}_position_ids"])
                labels_len = len(row[f"{prefix}_labels"])
                if not (ids_len == attn_len == pos_len == labels_len):
                    errors.append(
                        f"[{split_name}:{i}] {prefix}: "
                        f"ids={ids_len}, attn={attn_len}, "
                        f"pos={pos_len}, labels={labels_len}"
                    )
    if not errors:
        print(f"  [PASS] Length consistency: all branches consistent.")
    else:
        print(f"  [FAIL] Length consistency: {len(errors)} errors.")
    return errors


def check_position_validity(dsd: DatasetDict, K_target: int = 65536) -> List[str]:
    """Check position IDs are in range and strictly increasing."""
    errors: List[str] = []
    for split_name, ds in dsd.items():
        for i, row in enumerate(ds):
            for prefix in BRANCH_PREFIXES:
                positions = row[f"{prefix}_position_ids"]
                if not positions:
                    continue
                # Range check
                if min(positions) < 0 or max(positions) >= K_target:
                    errors.append(
                        f"[{split_name}:{i}] {prefix}: "
                        f"position out of range [{min(positions)}, {max(positions)}]"
                    )
                # Monotonicity check
                for j in range(1, len(positions)):
                    if positions[j] <= positions[j - 1]:
                        errors.append(
                            f"[{split_name}:{i}] {prefix}: "
                            f"not increasing at idx {j}: "
                            f"{positions[j-1]} >= {positions[j]}"
                        )
                        break
    if not errors:
        print(f"  [PASS] Position validity: all positions OK.")
    else:
        print(f"  [FAIL] Position validity: {len(errors)} errors.")
    return errors


def check_shared_prompt(dsd: DatasetDict) -> List[str]:
    """Check that the three branches share the same prompt prefix."""
    errors: List[str] = []
    for split_name, ds in dsd.items():
        for i, row in enumerate(ds):
            c_ids = row["chosen_input_ids"]
            r1_ids = row["reject_1_input_ids"]
            r2_ids = row["reject_2_input_ids"]

            # Find divergence point
            min_len = min(len(c_ids), len(r1_ids), len(r2_ids))
            diverge = None
            for j in range(min_len):
                if c_ids[j] != r1_ids[j] or c_ids[j] != r2_ids[j]:
                    diverge = j
                    break

            if diverge is not None and diverge > 0:
                # Before divergence, input_ids should match
                if (
                    c_ids[:diverge] != r1_ids[:diverge]
                    or c_ids[:diverge] != r2_ids[:diverge]
                ):
                    errors.append(
                        f"[{split_name}:{i}] shared prefix mismatch at token {diverge}"
                    )

                # Before divergence, labels should be -100
                c_labels = row["chosen_labels"]
                for j in range(diverge):
                    if c_labels[j] != -100:
                        errors.append(
                            f"[{split_name}:{i}] non -100 label "
                            f"in shared prefix at idx {j}"
                        )
                        break

    if not errors:
        print(f"  [PASS] Shared prompt: all three branches share prefix.")
    else:
        print(f"  [FAIL] Shared prompt: {len(errors)} errors.")
    return errors


def check_labels(dsd: DatasetDict) -> List[str]:
    """Check that prompt tokens have label=-100 and answer tokens have label==input_id."""
    errors: List[str] = []
    for split_name, ds in dsd.items():
        for i, row in enumerate(ds):
            for prefix in BRANCH_PREFIXES:
                ids = row[f"{prefix}_input_ids"]
                labels = row[f"{prefix}_labels"]

                answer_labels = [l for l in labels if l != -100]
                if not answer_labels:
                    errors.append(
                        f"[{split_name}:{i}] {prefix}: no non -100 labels"
                    )
                    continue

                # Check label == input_id wherever label != -100
                for j, (tid, lbl) in enumerate(zip(ids, labels)):
                    if lbl != -100 and lbl != tid:
                        errors.append(
                            f"[{split_name}:{i}] {prefix}: "
                            f"label mismatch at idx {j}: "
                            f"input_id={tid}, label={lbl}"
                        )

    if not errors:
        print(f"  [PASS] Labels: prompt=-100, answer=input_ids.")
    else:
        print(f"  [FAIL] Labels: {len(errors)} errors.")
    return errors


def print_statistics(dsd: DatasetDict) -> None:
    """Print distributions of lengths, positions, and answer sizes."""
    print("\n--- Statistics ---")
    for split_name, ds in dsd.items():
        print(f"\n  Split: {split_name}  ({len(ds)} records)")

        total_lens = []
        max_positions = []
        answer_lens: Dict[str, List[int]] = {
            "chosen": [], "rejected_1": [], "rejected_2": []
        }

        for row in ds:
            total_lens.append(len(row["chosen_input_ids"]))
            max_positions.append(max(row["chosen_position_ids"]))

            for prefix in BRANCH_PREFIXES:
                labels = row[f"{prefix}_labels"]
                ans_len = len([l for l in labels if l != -100])
                key = prefix.replace("reject_", "rejected_")
                answer_lens[key].append(ans_len)

        def _print_dist(name: str, values: List[int]) -> None:
            if not values:
                return
            print(
                f"    {name:24s}: "
                f"min={min(values):5d}  max={max(values):5d}  "
                f"avg={sum(values)/len(values):7.1f}  "
                f"med={sorted(values)[len(values)//2]:5d}"
            )

        _print_dist("total_length", total_lens)
        _print_dist("max_position_id", max_positions)
        for key in ("chosen", "rejected_1", "rejected_2"):
            _print_dist(f"{key}_answer_len", answer_lens[key])

    # Read metadata if available
    print()


def print_sample(dsd: DatasetDict, idx: int = 0, split: str = "train") -> None:
    """Decode and display a random sample's token sequences."""
    ds = dsd[split]
    if idx >= len(ds):
        idx = len(ds) - 1
    if idx < 0:
        return

    row = ds[idx]
    print(f"\n--- Sample {idx} ({split}) ---")

    for prefix in BRANCH_PREFIXES:
        ids = row[f"{prefix}_input_ids"]
        labels = row[f"{prefix}_labels"]
        positions = row[f"{prefix}_position_ids"]

        # Show the answer region (non -100 labels)
        answer_start = next(
            (j for j, l in enumerate(labels) if l != -100), len(labels)
        )
        print(f"\n  [{prefix}]")
        print(f"    total_length: {len(ids)}")
        print(f"    answer_start: {answer_start}")
        print(f"    position_range: [{min(positions)}, {max(positions)}]")
        print(f"    first_answer_ids: {ids[answer_start:answer_start+20]}")
        print(
            f"    first_answer_pos: {positions[answer_start:answer_start+20]}"
        )


def full_inspection(dataset_path: str, K_target: int = 65536) -> int:
    """Run all checks. Returns number of errors found."""
    print(f"Inspecting dataset: {dataset_path}")
    dsd = load_dataset(dataset_path)
    print(f"  Splits: {list(dsd.keys())}")
    for k, v in dsd.items():
        print(f"    {k}: {len(v)} records")

    print("\n--- Validation ---")
    all_errors: List[str] = []
    all_errors.extend(check_field_completeness(dsd))
    all_errors.extend(check_length_consistency(dsd))
    all_errors.extend(check_position_validity(dsd, K_target))
    all_errors.extend(check_shared_prompt(dsd))
    all_errors.extend(check_labels(dsd))

    print_statistics(dsd)
    print_sample(dsd, idx=0)

    if all_errors:
        print(f"\n  Total errors: {len(all_errors)}")
    else:
        print(f"\n  [OK] All checks passed!")

    return len(all_errors)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LOGO Dataset Inspector")
    p.add_argument("--dataset_path", required=True, help="Path to built DatasetDict")
    p.add_argument(
        "--K_target", type=int, default=65536, help="Target position window"
    )
    p.add_argument("--sample_idx", type=int, default=0, help="Sample index to display")
    p.add_argument("--split", default="train", help="Split to display sample from")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_errors = full_inspection(args.dataset_path, K_target=args.K_target)
    sys.exit(0 if n_errors == 0 else 1)


if __name__ == "__main__":
    main()
