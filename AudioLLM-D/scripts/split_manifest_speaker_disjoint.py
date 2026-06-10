"""Create train/test manifests with speaker-disjoint groups.

The input manifest is expected to contain clip-level rows. Rows are grouped by
speaker before splitting, so every clip from a speaker lands in exactly one
split. For TalkBank-style task IDs such as ``BBW8512a`` and ``BBW8512b``, the
default speaker normalization strips the trailing task letter and groups both
as ``BBW8512``.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


_CLIP_SUFFIX_RE = re.compile(r"_(\d+)$")
_TASK_SUFFIX_RE = re.compile(r"(?<=\d)[A-Za-z]$")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    manifest = Path(path).expanduser().resolve()
    with manifest.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _materialize_file(src_value: Any, dst_dir: Path, *, mode: str) -> str:
    if not isinstance(src_value, str) or not src_value.strip():
        return src_value

    src = Path(src_value).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Referenced file does not exist: {src}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        return str(dst.resolve())

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        raise ValueError(f"Unsupported materialize mode: {mode}")

    return str(dst.resolve())


def infer_speaker_id(row: dict[str, Any], *, strip_task_suffix: bool = True) -> str:
    """Return the participant id used for speaker-disjoint grouping."""

    raw_id = row.get("speaker_id") or row.get("source_id") or row.get("clip_id")
    if not raw_id:
        audio_path = row.get("audio_path")
        if not audio_path:
            raise ValueError(f"Cannot infer speaker id for row without ids/audio_path: {row}")
        raw_id = Path(str(audio_path)).stem

    speaker_id = str(raw_id)
    speaker_id = _CLIP_SUFFIX_RE.sub("", speaker_id)
    if strip_task_suffix:
        speaker_id = _TASK_SUFFIX_RE.sub("", speaker_id)
    return speaker_id


def _split_groups(
    grouped_rows: dict[str, list[dict[str, Any]]],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1")

    speaker_ids = list(grouped_rows)
    rng = random.Random(seed)
    rng.shuffle(speaker_ids)

    total_rows = sum(len(rows) for rows in grouped_rows.values())
    target_test_rows = total_rows * test_ratio

    test_speakers: set[str] = set()
    current_test_rows = 0
    for speaker_id in sorted(speaker_ids, key=lambda sid: len(grouped_rows[sid]), reverse=True):
        if not test_speakers:
            test_speakers.add(speaker_id)
            current_test_rows += len(grouped_rows[speaker_id])
            continue

        candidate_rows = current_test_rows + len(grouped_rows[speaker_id])
        current_gap = abs(target_test_rows - current_test_rows)
        candidate_gap = abs(target_test_rows - candidate_rows)
        if candidate_gap <= current_gap or current_test_rows < target_test_rows:
            test_speakers.add(speaker_id)
            current_test_rows = candidate_rows

    train_speakers = set(grouped_rows) - test_speakers
    if not train_speakers:
        raise ValueError("Split produced no train speakers; reduce --test-ratio or add data")
    return train_speakers, test_speakers


def split_manifest(
    *,
    manifest: str | Path,
    output_dir: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
    strip_task_suffix: bool = True,
    materialize_files: bool = False,
    file_mode: str = "symlink",
) -> dict[str, Any]:
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest}")

    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[infer_speaker_id(row, strip_task_suffix=strip_task_suffix)].append(row)

    train_speakers, test_speakers = _split_groups(
        grouped_rows,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    output_root = Path(output_dir).expanduser().resolve()
    for row in rows:
        speaker_id = infer_speaker_id(row, strip_task_suffix=strip_task_suffix)
        split = "test" if speaker_id in test_speakers else "train"
        out_row = dict(row)
        out_row["split"] = split
        out_row["speaker_id"] = speaker_id
        if materialize_files:
            split_root = output_root / split
            out_row["audio_path"] = _materialize_file(
                out_row.get("audio_path"),
                split_root / "audio",
                mode=file_mode,
            )
            out_row["transcript_path"] = _materialize_file(
                out_row.get("transcript_path"),
                split_root / "transcripts",
                mode=file_mode,
            )
        if split == "test":
            test_rows.append(out_row)
        else:
            train_rows.append(out_row)

    write_jsonl(train_rows, output_root / "train" / "manifest.jsonl")
    write_jsonl(test_rows, output_root / "test" / "manifest.jsonl")

    overlap = sorted(train_speakers & test_speakers)
    summary = {
        "input_manifest": str(Path(manifest).expanduser().resolve()),
        "output_dir": str(output_root),
        "seed": seed,
        "test_ratio": test_ratio,
        "strip_task_suffix": strip_task_suffix,
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "total_speakers": len(grouped_rows),
        "train_speakers": len(train_speakers),
        "test_speakers": len(test_speakers),
        "speaker_overlap": overlap,
        "materialize_files": materialize_files,
        "file_mode": file_mode if materialize_files else None,
    }
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input combined JSONL manifest.")
    parser.add_argument("--output-dir", required=True, help="Directory for train/test manifests.")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep-task-suffix",
        action="store_true",
        help="Do not collapse IDs like BBW8512a/BBW8512b into one speaker.",
    )
    parser.add_argument(
        "--materialize-files",
        action="store_true",
        help="Also create split audio/transcript folders and update paths in manifests.",
    )
    parser.add_argument(
        "--file-mode",
        choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="How to materialize files when --materialize-files is set.",
    )
    args = parser.parse_args()

    summary = split_manifest(
        manifest=args.manifest,
        output_dir=args.output_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        strip_task_suffix=not args.keep_task_suffix,
        materialize_files=args.materialize_files,
        file_mode=args.file_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
