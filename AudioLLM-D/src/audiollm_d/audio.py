"""Audio extraction utilities for CHAT-aligned recordings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from audiollm_d.chat_parser import SpeakerSegment, extract_speaker_timestamps


def extract_speaker_audio(
    audio_path: str | Path,
    chat_path: str | Path,
    *,
    speaker: str = "PAR",
    output_path: str | Path | None = None,
    target_sr: int | None = None,
    pause_seconds: float = 0.6,
) -> tuple[np.ndarray, int]:
    """Extract and stitch the segments of *audio_path* where *speaker* is talking.

    Timestamps are read from the CHAT file at *chat_path*. Segments without a
    timestamp in the CHAT file are skipped. The stitched audio is returned as a
    1-D float32 numpy array together with the sample rate.

    Args:
        audio_path: Path to the recording (.wav or .mp3).
        chat_path:  Matching CHAT transcript (.cha).
        speaker:    CLAN speaker code to keep (default ``"PAR"``).
        output_path: If given, write the stitched audio here as a WAV file.
        target_sr:  Resample to this rate. ``None`` keeps the native rate.

    Returns:
        ``(stitched, sr)`` — float32 array and sample rate in Hz.
    """
    import librosa  # lazy import — heavy dependency

    segs: list[SpeakerSegment] = extract_speaker_timestamps(chat_path, speaker=speaker)
    if not segs:
        raise ValueError(f"No timestamped {speaker} segments found in {chat_path}")

    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    total_samples = len(audio)

    clips: list[np.ndarray] = []
    for seg in segs:
        start = int(seg.start_ms / 1000 * sr)
        end = int(seg.end_ms / 1000 * sr)
        # Clamp to actual audio length and skip zero-length slices.
        start = min(start, total_samples)
        end = min(end, total_samples)
        if end > start:
            clips.append(audio[start:end])

    if not clips:
        raise ValueError(f"All {speaker} segments fell outside the audio duration ({total_samples / sr:.1f}s)")

    pause_samples = int(round(max(0.0, pause_seconds) * sr))
    pause = np.zeros(pause_samples, dtype=np.float32)
    stitched_parts: list[np.ndarray] = []
    for i, clip in enumerate(clips):
        if i > 0 and pause_samples > 0:
            stitched_parts.append(pause)
        stitched_parts.append(clip.astype(np.float32, copy=False))
    stitched = np.concatenate(stitched_parts)

    if output_path is not None:
        sf.write(str(output_path), stitched, sr)

    return stitched, sr


def extract_speaker_audio_clips(
    audio_path: str | Path,
    chat_path: str | Path,
    *,
    speaker: str = "PAR",
    target_sr: int | None = None,
    pause_seconds: float = 0.6,
    max_clip_seconds: float = 30.0,
    output_dir: str | Path | None = None,
) -> tuple[list[np.ndarray], int]:
    """Extract speaker-only audio and split it into clips up to ``max_clip_seconds``.

    A silence gap of ``pause_seconds`` is inserted between stitched turns to sound natural.
    """
    import librosa  # lazy import — heavy dependency

    segs: list[SpeakerSegment] = extract_speaker_timestamps(chat_path, speaker=speaker)
    if not segs:
        raise ValueError(f"No timestamped {speaker} segments found in {chat_path}")

    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    total_samples = len(audio)

    max_samples = int(max_clip_seconds * sr)
    if max_samples <= 0:
        raise ValueError("max_clip_seconds must be > 0")

    pause_samples = int(round(max(0.0, pause_seconds) * sr))
    pause = np.zeros(pause_samples, dtype=np.float32)

    merged: list[np.ndarray] = []
    current_parts: list[np.ndarray] = []
    current_len = 0

    for seg in segs:
        start = min(int(seg.start_ms / 1000 * sr), total_samples)
        end = min(int(seg.end_ms / 1000 * sr), total_samples)
        if end <= start:
            continue
        segment = audio[start:end].astype(np.float32, copy=False)
        segment_len = len(segment)

        additional = segment_len + (pause_samples if current_parts and pause_samples > 0 else 0)
        if current_parts and current_len + additional > max_samples:
            merged.append(np.concatenate(current_parts))
            current_parts = [segment]
            current_len = segment_len
            continue

        if current_parts and pause_samples > 0:
            current_parts.append(pause)
            current_len += pause_samples
        current_parts.append(segment)
        current_len += segment_len

    if current_parts:
        merged.append(np.concatenate(current_parts))

    if not merged:
        raise ValueError(f"All {speaker} segments fell outside the audio duration ({total_samples / sr:.1f}s)")

    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        stem = Path(audio_path).stem
        for i, clip in enumerate(merged):
            sf.write(str(destination / f"{stem}_{i:03d}.wav"), clip, sr)

    return merged, sr
