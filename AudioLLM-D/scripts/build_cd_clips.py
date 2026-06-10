"""Create <=30s stitched PAR clips with aligned transcript text from CHAT files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from audiollm_d.chat_parser import extract_timed_speaker_utterances, read_chat_transcript

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    source_id: str
    clip_index: int
    audio_path: str
    transcript_path: str
    text: str
    split: str
    diagnosis: str | None = None
    mmse: str | None = None
    age: str | None = None
    sex: str | None = None


def _find_audio_files(audio_dir: Path) -> list[Path]:
    return sorted(
        path for path in audio_dir.iterdir() if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def _write_jsonl(records: list[ClipRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def build_cd_clips(
    *,
    audio_dir: str | Path,
    transcript_dir: str | Path,
    output_dir: str | Path,
    speaker: str = "PAR",
    split: str = "train",
    pause_seconds: float = 0.6,
    max_clip_seconds: float = 30.0,
    target_sr: int | None = None,
) -> list[ClipRecord]:
    audio_root = Path(audio_dir).expanduser().resolve()
    transcript_root = Path(transcript_dir).expanduser().resolve()
    out_root = Path(output_dir).expanduser().resolve()

    audio_out = out_root / "audio"
    text_out = out_root / "transcripts"
    audio_out.mkdir(parents=True, exist_ok=True)
    text_out.mkdir(parents=True, exist_ok=True)

    pause_seconds = max(0.0, pause_seconds)
    if max_clip_seconds <= 0:
        raise ValueError("max_clip_seconds must be > 0")

    records: list[ClipRecord] = []
    for audio_path in _find_audio_files(audio_root):
        chat_path = transcript_root / f"{audio_path.stem}.cha"
        if not chat_path.exists():
            continue

        timed_utterances = extract_timed_speaker_utterances(chat_path, speaker=speaker)
        if not timed_utterances:
            continue

        transcript = read_chat_transcript(chat_path)
        metadata = transcript.metadata

        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        audio = audio.astype(np.float32, copy=False)
        total_samples = len(audio)

        pause_samples = int(round(pause_seconds * sr))
        pause = np.zeros(pause_samples, dtype=np.float32)
        max_clip_samples = int(round(max_clip_seconds * sr))

        clip_idx = 0
        current_parts: list[np.ndarray] = []
        current_text: list[str] = []
        current_len = 0

        def flush_clip() -> None:
            nonlocal clip_idx, current_parts, current_text, current_len
            if not current_parts or not current_text:
                current_parts = []
                current_text = []
                current_len = 0
                return

            clip_audio = np.concatenate(current_parts)
            clip_text = " ".join(current_text).strip()
            if not clip_text:
                current_parts = []
                current_text = []
                current_len = 0
                return

            clip_id = f"{audio_path.stem}_{clip_idx:03d}"
            audio_dst = audio_out / f"{clip_id}.wav"
            text_dst = text_out / f"{clip_id}.txt"
            sf.write(str(audio_dst), clip_audio, sr)
            text_dst.write_text(clip_text + "\n", encoding="utf-8")

            records.append(
                ClipRecord(
                    clip_id=clip_id,
                    source_id=audio_path.stem,
                    clip_index=clip_idx,
                    audio_path=str(audio_dst.resolve()),
                    transcript_path=str(text_dst.resolve()),
                    text=clip_text,
                    split=split,
                    diagnosis=metadata.get("diagnosis"),
                    mmse=metadata.get("mmse"),
                    age=metadata.get("age"),
                    sex=metadata.get("sex"),
                )
            )

            clip_idx += 1
            current_parts = []
            current_text = []
            current_len = 0

        for utterance in timed_utterances:
            start = min(int(utterance.start_ms / 1000 * sr), total_samples)
            end = min(int(utterance.end_ms / 1000 * sr), total_samples)
            if end <= start:
                continue

            segment = audio[start:end]
            if len(segment) == 0:
                continue

            additional = len(segment) + (pause_samples if current_parts and pause_samples > 0 else 0)
            if current_parts and current_len + additional > max_clip_samples:
                flush_clip()

            if current_parts and pause_samples > 0:
                current_parts.append(pause)
                current_len += pause_samples

            current_parts.append(segment)
            current_text.append(utterance.text)
            current_len += len(segment)

        flush_clip()

    _write_jsonl(records, out_root / "manifest.jsonl")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, help="Directory with source audio files.")
    parser.add_argument("--transcript-dir", required=True, help="Directory with matching .cha transcripts.")
    parser.add_argument("--output-dir", required=True, help="Output directory for clips and manifest.")
    parser.add_argument("--speaker", default="PAR")
    parser.add_argument("--split", default="train")
    parser.add_argument("--pause-seconds", type=float, default=0.6)
    parser.add_argument("--max-clip-seconds", type=float, default=30.0)
    parser.add_argument("--target-sr", type=int, default=None)
    args = parser.parse_args()

    records = build_cd_clips(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        output_dir=args.output_dir,
        speaker=args.speaker,
        split=args.split,
        pause_seconds=args.pause_seconds,
        max_clip_seconds=args.max_clip_seconds,
        target_sr=args.target_sr,
    )
    print(f"Wrote {len(records)} clips to {Path(args.output_dir).expanduser().resolve()}")


if __name__ == "__main__":
    main()
