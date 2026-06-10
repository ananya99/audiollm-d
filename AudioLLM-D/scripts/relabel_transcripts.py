#!/usr/bin/env python
"""Rewrite pause labels in processed transcripts using MFA-measured durations.

For each clip, reads the original .txt transcript and the corresponding
_alignment.json produced by postprocess_mfa.py.  Within each clip, all
non-zero-duration pause tokens are ranked by measured duration and re-labelled
so that shortest durations get [short-pause] and longest get [long-pause].
The count of each pause type is preserved (no additions or deletions).
Zero-duration (no_gap) pause tokens keep their original label.

Corrected transcripts are written to --output-dir with the same filename.

Usage:
    python scripts/relabel_transcripts.py \
        --transcript-dir merged/merged_data_processed/transcripts \
        --alignment-dir  mfa/processed \
        --output-dir     merged/merged_data_processed_relabelled/transcripts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_PAUSE_TOKENS = ["[short-pause]", "[medium-pause]", "[long-pause]"]
_PAUSE_RE = re.compile(
    r"\[(?:short-pause|medium-pause|long-pause)\]"
)
_PAUSE_ORDER = ["[short-pause]", "[medium-pause]", "[long-pause]"]


def relabel(original_text: str, alignment: list[dict]) -> str:
    """Return transcript text with pause labels corrected by measured duration.

    The sequence and count of pause tokens is preserved exactly.
    """
    # Collect pause records that have a real duration (exclude no_gap).
    pause_records = [
        r for r in alignment
        if r.get("type") == "silence" and r["end"] > r["start"]
    ]

    if not pause_records:
        return original_text

    # Count original label types from alignment (not from text — same thing,
    # but alignment is already tokenised so it's authoritative).
    label_counts = Counter(r["word"] for r in pause_records)

    # Sort pause records by ascending measured duration.
    sorted_records = sorted(pause_records, key=lambda r: r["end"] - r["start"])

    # Build the new label sequence (shortest → [short-pause], longest → [long-pause]).
    new_labels: list[str] = []
    for label in _PAUSE_ORDER:
        new_labels.extend([label] * label_counts.get(label, 0))

    # Map each sorted record to its new label; re-index by original position.
    # We need to apply labels in TEXT ORDER, so sort back by original position.
    # Use enumerate over sorted_records to track assignment, then re-sort by
    # the order records appear in the alignment list.

    # Build (alignment_index, new_label) pairs.
    # alignment_index = position of the record in the original alignment list.
    record_positions = {
        id(r): i for i, r in enumerate(alignment) if r.get("type") == "silence" and r["end"] > r["start"]
    }
    # sorted_records is ordered by duration; zip with new_labels gives assignment.
    label_by_pos: dict[int, str] = {}
    for record, new_label in zip(sorted_records, new_labels):
        pos = record_positions[id(record)]
        label_by_pos[pos] = new_label

    # Walk alignment in text order; collect new labels for non-zero pause tokens.
    relabelled_in_order: list[str] = []
    for i, r in enumerate(alignment):
        if r.get("type") == "silence" and r["end"] > r["start"]:
            relabelled_in_order.append(label_by_pos[i])
        elif r.get("type") == "silence":
            # Zero-duration: keep original label unchanged.
            relabelled_in_order.append(r["word"])

    # Now substitute into the original text in order using regex.
    pause_iter = iter(relabelled_in_order)

    def _replace(m: re.Match) -> str:
        try:
            return next(pause_iter)
        except StopIteration:
            return m.group(0)

    return _PAUSE_RE.sub(_replace, original_text)


def process_directory(
    transcript_dir: Path,
    alignment_dir: Path,
    output_dir: Path,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(transcript_dir.glob("*.txt"))

    n_ok = 0
    n_skipped = 0

    for txt_path in txt_files:
        stem = txt_path.stem
        json_path = alignment_dir / (stem + "_alignment.json")

        if not json_path.exists():
            print(f"[skip] {stem}: no alignment JSON found.", file=sys.stderr)
            n_skipped += 1
            # Write original unchanged so output dir is complete.
            (output_dir / txt_path.name).write_text(
                txt_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            continue

        original = txt_path.read_text(encoding="utf-8")
        alignment = json.load(open(json_path, encoding="utf-8"))
        corrected = relabel(original, alignment)
        (output_dir / txt_path.name).write_text(corrected, encoding="utf-8")
        n_ok += 1

    return n_ok, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-dir", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    n_ok, n_skipped = process_directory(
        transcript_dir=args.transcript_dir,
        alignment_dir=args.alignment_dir,
        output_dir=args.output_dir,
    )
    print(f"[done] relabelled {n_ok} transcripts, {n_skipped} skipped (original copied).")


if __name__ == "__main__":
    main()
