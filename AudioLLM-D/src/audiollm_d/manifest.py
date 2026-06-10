"""Build JSONL manifests from DementiaBank/ADReSS-style directories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from audiollm_d.chat_parser import read_chat_transcript

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
TRANSCRIPT_EXTENSIONS = {".cha", ".txt"}


@dataclass(frozen=True)
class ManifestRecord:
    """One supervised transcription example."""

    audio_path: str
    text: str
    speaker_id: str
    split: str
    transcript_path: str | None = None
    alignment_path: str | None = None
    diagnosis: str | None = None
    mmse: str | None = None
    age: str | None = None
    sex: str | None = None


def _index_by_stem(paths: Iterable[Path]) -> dict[str, Path]:
    return {path.stem: path for path in sorted(paths)}


def find_audio_files(audio_dir: Path) -> list[Path]:
    return [
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]


def find_transcript_files(transcript_dir: Path) -> list[Path]:
    return [
        path
        for path in transcript_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in TRANSCRIPT_EXTENSIONS
    ]


def infer_split(path: Path, default_split: str) -> str:
    """Infer split from parent directory names when datasets already provide one."""

    parts = {part.lower() for part in path.parts}
    for split in ("train", "validation", "val", "dev", "test"):
        if split in parts:
            return "validation" if split in {"val", "dev"} else split
    return default_split


def build_manifest_records(
    *,
    audio_dir: str | Path,
    transcript_dir: str | Path,
    default_split: str = "train",
    alignment_dir: str | Path | None = None,
) -> list[ManifestRecord]:
    """Pair audio and transcripts by filename stem and return manifest records."""

    audio_root = Path(audio_dir).expanduser().resolve()
    transcript_root = Path(transcript_dir).expanduser().resolve()
    transcripts = _index_by_stem(find_transcript_files(transcript_root))
    alignment_root = Path(alignment_dir).expanduser().resolve() if alignment_dir else None
    records: list[ManifestRecord] = []

    for audio_path in find_audio_files(audio_root):
        transcript_path = transcripts.get(audio_path.stem)
        if transcript_path is None:
            continue

        if transcript_path.suffix.lower() == ".cha":
            transcript = read_chat_transcript(transcript_path)
            text = transcript.text
            metadata = transcript.metadata
        else:
            text = transcript_path.read_text(encoding="utf-8").strip()
            metadata = {}

        if not text:
            continue

        alignment_path: str | None = None
        if alignment_root is not None:
            candidate = alignment_root / f"{audio_path.stem}_alignment.json"
            if candidate.exists():
                alignment_path = str(candidate)

        records.append(
            ManifestRecord(
                audio_path=str(audio_path.resolve()),
                text=text,
                speaker_id=audio_path.stem,
                split=infer_split(audio_path, default_split),
                transcript_path=str(transcript_path.resolve()),
                alignment_path=alignment_path,
                diagnosis=metadata.get("diagnosis"),
                mmse=metadata.get("mmse"),
                age=metadata.get("age"),
                sex=metadata.get("sex"),
            )
        )

    return records


def write_jsonl(records: Iterable[ManifestRecord], output_path: str | Path) -> None:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, help="Directory containing wav/mp3/flac audio.")
    parser.add_argument("--transcript-dir", required=True, help="Directory containing .cha or .txt files.")
    parser.add_argument("--output", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--default-split", default="train")
    parser.add_argument(
        "--alignment-dir",
        default=None,
        help="Directory of MFA alignment JSONs (<stem>_alignment.json). "
        "When provided, alignment_path is populated in each manifest record.",
    )
    args = parser.parse_args()

    records = build_manifest_records(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        default_split=args.default_split,
        alignment_dir=args.alignment_dir,
    )
    write_jsonl(records, args.output)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
