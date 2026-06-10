"""Dataset and collation code for Qwen2-Audio supervised fine-tuning."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from audiollm_d.chat import (
    TRAIN_SYSTEM_PROMPT,
    TRAIN_TRANSCRIPTION_INSTRUCTION,
    build_transcription_conversation,
    format_chat,
)
from audiollm_d.ctc_head import encoder_output_lengths
from audiollm_d.ctc_labels import CtcVocab, collapse_runs
from audiollm_d.manifest import read_jsonl

NOISE_AUGMENTATION_FLAG_COLUMN = "__apply_noise_augmentation__"

# All augmentation type names accepted in augmentation_types config.
AUGMENTATION_TYPES = frozenset({"white_noise", "babble", "reverb", "speed", "codec"})

# Whisper encoder output frame rate: 16kHz / (hop=160 * stride=2) = 50 Hz → 20 ms/frame.
_ENCODER_FRAMES_PER_SECOND: float = 50.0
# Maximum encoder output frames for a 30-second clip (Qwen2-Audio pads all audio to 30s).
_MAX_ENCODER_FRAMES: int = 1500

# Per-frame class labels for encoder supervision.
# Class 0 = regular speech; classes 1..N = one class per disfluency token (SPECIAL_TOKENS order).
# Class -1 = ignore (frames beyond actual audio end).
from audiollm_d.special_tokens import SPECIAL_TOKENS as _SPECIAL_TOKENS

_DISFLUENCY_FRAME_CLASS: dict[str, int] = {
    token: i + 1 for i, token in enumerate(_SPECIAL_TOKENS)
}
ENCODER_NUM_CLASSES: int = 1 + len(_SPECIAL_TOKENS)  # 9 for the current token set


def _build_encoder_frame_labels(alignment: list[dict]) -> "torch.Tensor":
    """Return a long tensor of shape [_MAX_ENCODER_FRAMES] with per-frame class labels.

    Class 0 = speech; classes 1–8 = individual disfluency tokens (SPECIAL_TOKENS order);
    -1 = ignore (beyond actual audio end).
    """
    import torch

    labels = torch.full((_MAX_ENCODER_FRAMES,), -1, dtype=torch.long)
    if not alignment:
        return labels

    audio_end = alignment[-1]["end"]
    valid_frames = min(int(audio_end * _ENCODER_FRAMES_PER_SECOND) + 1, _MAX_ENCODER_FRAMES)
    labels[:valid_frames] = 0  # default: regular speech

    for rec in alignment:
        cls = _DISFLUENCY_FRAME_CLASS.get(rec.get("word", ""))
        if cls is None:
            continue
        s = max(0, int(rec["start"] * _ENCODER_FRAMES_PER_SECOND))
        e = min(_MAX_ENCODER_FRAMES, int(rec["end"] * _ENCODER_FRAMES_PER_SECOND))
        if s < e:
            labels[s:e] = cls

    return labels


def load_audio(path: str | Path, sampling_rate: int) -> np.ndarray:
    """Load mono audio at the model feature extractor sampling rate."""

    import librosa

    audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return audio


def apply_white_noise_at_snr(audio: np.ndarray, *, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Mix white Gaussian noise into audio at a target SNR in dB."""
    if audio.size == 0:
        return audio
    signal_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    if signal_rms <= 1e-12:
        return audio

    noise = rng.standard_normal(audio.shape).astype(np.float32, copy=False)
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if noise_rms <= 1e-12:
        return audio

    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    mixed = audio + scaled_noise
    return np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)


def _mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale noise to target SNR and mix into signal. Both must be float32, same length."""
    signal_rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float64)))
    if signal_rms <= 1e-12:
        return signal
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    if noise_rms <= 1e-12:
        return signal
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    mixed = signal + noise * (target_noise_rms / noise_rms)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32, copy=False)


def _speech_shaped_noise(n_samples: int, sr: int, num_sources: int, rng: np.random.Generator) -> np.ndarray:
    """Generate unit-RMS bandlimited noise shaped like speech (300–3400 Hz) via FFT masking.

    Summing several independent sources approximates the spectral envelope of
    overlapping talkers without requiring external noise files.
    """
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sr)
    speech_band = (freqs >= 300.0) & (freqs <= 3400.0)
    accumulated = np.zeros(n_samples, dtype=np.float64)
    for _ in range(num_sources):
        raw = rng.standard_normal(n_samples)
        spectrum = np.fft.rfft(raw)
        spectrum *= speech_band
        accumulated += np.fft.irfft(spectrum, n=n_samples)
    rms = float(np.sqrt(np.mean(np.square(accumulated))))
    if rms < 1e-12:
        return accumulated.astype(np.float32)
    return (accumulated / rms).astype(np.float32)


def apply_babble_noise(
    audio: np.ndarray,
    *,
    snr_db: float,
    sr: int,
    num_sources: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add simulated multi-talker babble (speech-shaped noise) at the given SNR.

    Noise is generated on-the-fly from bandpass-filtered Gaussian noise, so no
    external babble corpus is required.
    """
    if audio.size == 0:
        return audio
    noise = _speech_shaped_noise(len(audio), sr, num_sources, rng)
    return _mix_at_snr(audio, noise, snr_db)


def apply_rir_reverb(
    audio: np.ndarray,
    *,
    rt60_ms: float,
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convolve audio with a synthetic exponential-decay room impulse response.

    The RIR is constructed as decaying Gaussian noise: h[n] = exp(-6.9*n/N) * w[n]
    where N = rt60_samples.  Output is RMS-normalised to match the input level so
    training loss is not affected by loudness differences.
    """
    if audio.size == 0:
        return audio
    rt60_samples = int(rt60_ms * 1e-3 * sr)
    if rt60_samples < 2:
        return audio

    t = np.arange(rt60_samples, dtype=np.float64)
    decay = np.exp(-6.9 * t / rt60_samples)
    rir = (decay * rng.standard_normal(rt60_samples)).astype(np.float32)
    peak = np.max(np.abs(rir))
    if peak < 1e-12:
        return audio
    rir /= peak

    reverbed = np.convolve(audio, rir)[: len(audio)].astype(np.float32)
    # Restore original RMS so the loss scale stays consistent.
    in_rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    out_rms = float(np.sqrt(np.mean(np.square(reverbed), dtype=np.float64)))
    if out_rms > 1e-12:
        reverbed *= in_rms / out_rms
    return np.clip(reverbed, -1.0, 1.0)


def apply_speed_perturbation(audio: np.ndarray, *, rate: float, sr: int) -> np.ndarray:
    """Change speech speed by resampling (rate < 1 = slower, rate > 1 = faster).

    Speed and pitch are both shifted, which is the standard practice for ASR
    augmentation.  The output length changes: slow audio gets more samples,
    fast audio gets fewer — the Qwen2-Audio feature extractor handles variable
    lengths naturally.
    """
    import librosa

    if audio.size == 0 or abs(rate - 1.0) < 1e-4:
        return audio
    # Treating audio as if sampled at sr*rate and resampling to sr yields
    # len(audio)/rate output samples, changing playback duration.
    resampled = librosa.resample(audio, orig_sr=int(round(sr * rate)), target_sr=sr)
    return resampled.astype(np.float32)


def apply_codec_compression(audio: np.ndarray, *, sr: int, codec_rate: int) -> np.ndarray:
    """Simulate telephony/codec degradation by downsampling then upsampling.

    Resampling to a low intermediate rate (e.g. 8 kHz) and back introduces the
    anti-aliasing distortion and bandwidth limitation that occurs in codec pipelines.
    """
    import librosa

    if audio.size == 0 or codec_rate >= sr:
        return audio
    degraded = librosa.resample(audio, orig_sr=sr, target_sr=codec_rate)
    restored = librosa.resample(degraded, orig_sr=codec_rate, target_sr=sr)
    # Rounding may produce ±1 sample drift; trim or pad to the original length.
    n = len(audio)
    if len(restored) >= n:
        return restored[:n].astype(np.float32)
    return np.pad(restored, (0, n - len(restored))).astype(np.float32)


def load_manifest_dataset(
    manifest_path: str | Path,
    *,
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    manifest = Path(manifest_path).expanduser().resolve()
    records = read_jsonl(manifest)
    for index, record in enumerate(records):
        for key in ("audio_path", "transcript_path"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                path = Path(value).expanduser()
                if not path.is_absolute():
                    raise ValueError(
                        f"Manifest {manifest} row {index} has non-absolute {key}: {value}"
                    )
                record[key] = str(path.resolve())
    if max_samples is not None:
        records = records[:max_samples]
    return records


@dataclass
class Qwen2AudioTranscriptionCollator:
    """Collate JSONL manifest rows into Qwen2-Audio training tensors."""

    processor: Any
    audio_column: str = "audio_path"
    text_column: str = "text"
    sampling_rate: int | None = None
    ignore_index: int = -100
    # Overall probability that a flagged training sample is augmented.
    noise_augmentation_prob: float = 0.0
    # Augmentation types to sample from uniformly; see AUGMENTATION_TYPES.
    augmentation_types: list[str] = field(default_factory=lambda: ["white_noise"])
    # White-noise parameters.
    noise_snr_db_min: float = 12.0
    noise_snr_db_max: float = 30.0
    # Babble parameters.
    babble_snr_db_min: float = 10.0
    babble_snr_db_max: float = 25.0
    babble_num_sources: int = 4
    # Reverb parameters (RT60 in milliseconds).
    reverb_rt60_ms_min: float = 200.0
    reverb_rt60_ms_max: float = 800.0
    # Speed-perturbation parameters.
    speed_min: float = 0.85
    speed_max: float = 1.15
    # Codec-compression parameter.
    codec_resample_rate: int = 8000
    noise_augmentation_flag_column: str = NOISE_AUGMENTATION_FLAG_COLUMN
    noise_seed: int | None = None
    # Column in manifest rows that holds the path to the MFA alignment JSON.
    # When present, silence_frame_labels and disfl_window_labels are built.
    alignment_column: str = "alignment_path"
    # Window parameters for the sliding-window disfluency encoder labels.
    # Must match DisfluencyEncoderConfig.window_size_s / stride_s.
    disfl_window_size_s: float = 0.5
    disfl_stride_s: float = 0.1
    # Phase 2: when set, attach frame-level CTC targets to each batch. ``ctc_labels_dir``
    # holds the per-clip ``{clip_id}.npy`` files from build_ctc_labels.py; ``ctc_vocab`` is
    # the matching CtcVocab. Both None -> collator behaves exactly as in Phase 1 (no CTC keys).
    ctc_labels_dir: str | Path | None = None
    ctc_vocab: CtcVocab | None = None
    clip_id_column: str = "clip_id"

    def __post_init__(self) -> None:
        if self.sampling_rate is None:
            self.sampling_rate = int(self.processor.feature_extractor.sampling_rate)
        tokenizer = self.processor.tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.noise_augmentation_prob = float(np.clip(self.noise_augmentation_prob, 0.0, 1.0))
        if self.noise_snr_db_min > self.noise_snr_db_max:
            self.noise_snr_db_min, self.noise_snr_db_max = self.noise_snr_db_max, self.noise_snr_db_min
        if self.babble_snr_db_min > self.babble_snr_db_max:
            self.babble_snr_db_min, self.babble_snr_db_max = self.babble_snr_db_max, self.babble_snr_db_min
        if self.reverb_rt60_ms_min > self.reverb_rt60_ms_max:
            self.reverb_rt60_ms_min, self.reverb_rt60_ms_max = self.reverb_rt60_ms_max, self.reverb_rt60_ms_min
        if self.speed_min > self.speed_max:
            self.speed_min, self.speed_max = self.speed_max, self.speed_min
        unknown = [t for t in self.augmentation_types if t not in AUGMENTATION_TYPES]
        if unknown:
            raise ValueError(f"Unknown augmentation type(s): {unknown}. Valid: {sorted(AUGMENTATION_TYPES)}")
        self._augmentation_types = list(self.augmentation_types) or ["white_noise"]
        self._noise_rng = np.random.default_rng(self.noise_seed)
        self._ctc_enabled = self.ctc_labels_dir is not None and self.ctc_vocab is not None
        if self._ctc_enabled:
            self.ctc_labels_dir = Path(self.ctc_labels_dir)

    def _apply_augmentation(self, audio: np.ndarray) -> np.ndarray:
        """Pick one augmentation type uniformly at random and apply it."""
        aug_type = str(self._noise_rng.choice(self._augmentation_types))
        sr = int(self.sampling_rate)
        if aug_type == "white_noise":
            snr_db = float(self._noise_rng.uniform(self.noise_snr_db_min, self.noise_snr_db_max))
            return apply_white_noise_at_snr(audio, snr_db=snr_db, rng=self._noise_rng)
        if aug_type == "babble":
            snr_db = float(self._noise_rng.uniform(self.babble_snr_db_min, self.babble_snr_db_max))
            return apply_babble_noise(audio, snr_db=snr_db, sr=sr, num_sources=self.babble_num_sources, rng=self._noise_rng)
        if aug_type == "reverb":
            rt60_ms = float(self._noise_rng.uniform(self.reverb_rt60_ms_min, self.reverb_rt60_ms_max))
            return apply_rir_reverb(audio, rt60_ms=rt60_ms, sr=sr, rng=self._noise_rng)
        if aug_type == "speed":
            rate = float(self._noise_rng.uniform(self.speed_min, self.speed_max))
            return apply_speed_perturbation(audio, rate=rate, sr=sr)
        if aug_type == "codec":
            return apply_codec_compression(audio, sr=sr, codec_rate=self.codec_resample_rate)
        return audio

    def _load_frame_labels(self, clip_id: str) -> np.ndarray:
        path = Path(self.ctc_labels_dir) / f"{clip_id}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"CTC frame labels not found for clip {clip_id!r} at {path}. "
                f"Run scripts/build_ctc_labels.py for this manifest."
            )
        return np.load(path)

    def _prompt_length(self, prompt_text: str) -> int:
        tokenizer = self.processor.tokenizer
        return len(tokenizer(prompt_text, add_special_tokens=False).input_ids)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        audios: list[np.ndarray] = []
        for feature in features:
            audio = load_audio(feature[self.audio_column], int(self.sampling_rate))
            should_try_augment = bool(feature.get(self.noise_augmentation_flag_column, False))
            if (
                should_try_augment
                and self.noise_augmentation_prob > 0.0
                and self._noise_rng.random() < self.noise_augmentation_prob
            ):
                audio = self._apply_augmentation(audio)
            audios.append(audio)
        full_texts: list[str] = []
        prompt_lengths: list[int] = []

        for feature in features:
            target_text = str(feature[self.text_column]).strip()
            prompt_conversation = build_transcription_conversation(
                text=None,
                system_prompt=TRAIN_SYSTEM_PROMPT,
                instruction=TRAIN_TRANSCRIPTION_INSTRUCTION,
            )
            full_conversation = build_transcription_conversation(
                text=target_text,
                system_prompt=TRAIN_SYSTEM_PROMPT,
                instruction=TRAIN_TRANSCRIPTION_INSTRUCTION,
            )
            prompt_text = format_chat(
                self.processor,
                prompt_conversation,
                add_generation_prompt=True,
            )
            full_text = format_chat(
                self.processor,
                full_conversation,
                add_generation_prompt=False,
            )
            prompt_lengths.append(self._prompt_length(prompt_text))
            full_texts.append(full_text)

        batch = self.processor(
            text=full_texts,
            audio=audios,
            sampling_rate=int(self.sampling_rate),
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels[labels == pad_token_id] = self.ignore_index

        for row_idx, prompt_length in enumerate(prompt_lengths):
            labels[row_idx, : min(prompt_length, labels.shape[1])] = self.ignore_index

        batch["labels"] = labels

        # Build per-sample labels from MFA alignment JSONs (one JSON read per sample).
        import torch
        from audiollm_d.disfluency_encoder import build_window_labels

        sil_label_list = []
        win_label_list = []
        any_alignment = False
        for feature in features:
            alignment_path = feature.get(self.alignment_column)
            alignment: list[dict] | None = None
            if alignment_path:
                try:
                    alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
                    any_alignment = True
                except (OSError, json.JSONDecodeError):
                    pass
            sil_label_list.append(_build_encoder_frame_labels(alignment or []))
            win_label_list.append(
                build_window_labels(
                    alignment or [],
                    window_size_s=self.disfl_window_size_s,
                    stride_s=self.disfl_stride_s,
                )
            )
        if any_alignment:
            batch["silence_frame_labels"] = torch.stack(sil_label_list)
            batch["disfl_window_labels"] = torch.stack(win_label_list)

        if self._ctc_enabled:
            self._attach_ctc_targets(batch, features)
        return batch

    def _attach_ctc_targets(self, batch: dict[str, Any], features: list[dict[str, Any]]) -> None:
        """Add CTC keys to ``batch`` from the per-clip frame labels.

        Encoder valid frame counts ``T_i`` are derived from ``feature_attention_mask`` (the same
        formula the model uses). Each clip's frame labels are asserted to match ``T_i`` within ±1
        (the #1 silent bug guard), truncated to exactly ``T_i`` so lengths line up, then
        run-length-collapsed into the CTC target sequence.
        """
        import torch

        input_lengths = encoder_output_lengths(batch["feature_attention_mask"])  # (B,) long
        max_t = int(input_lengths.max().item()) if len(input_lengths) else 0

        targets: list[int] = []
        target_lengths: list[int] = []
        frame_label_rows = torch.full((len(features), max_t), -1, dtype=torch.long)

        for row_idx, feature in enumerate(features):
            clip_id = feature.get(self.clip_id_column)
            if clip_id is None:
                raise KeyError(
                    f"Collator needs '{self.clip_id_column}' on each row for CTC labels; "
                    f"row {row_idx} has keys {sorted(feature)}."
                )
            frame_labels = self._load_frame_labels(str(clip_id)).astype(np.int64)
            t_i = int(input_lengths[row_idx].item())
            n_lab = len(frame_labels)
            # Length-changing augmentation (speed perturbation, up to ~±18%) alters the encoder
            # frame count at train time while the precomputed labels are on the clean-audio grid,
            # so t_i and n_lab can differ by many frames (reverb does NOT — it truncates back to the
            # original length). The CTC *target* is the run-length-collapsed sequence, which is
            # invariant to uniform time-scaling, so collapse the clean labels for the target and
            # only pad/truncate the (loss-irrelevant) frame buffer to t_i. A gross mismatch well
            # beyond the speed range signals a real wrong/stale label, not augmentation.
            tol = max(t_i, n_lab) // 4 + 4
            assert abs(t_i - n_lab) <= tol, (
                f"{clip_id}: encoder frames {t_i} vs label frames {n_lab} differ by more than "
                f"{tol} (beyond speed-augmentation range); likely a wrong/stale CTC label — "
                f"regenerate with build_ctc_labels_vad.py."
            )
            collapsed = collapse_runs(frame_labels)
            targets.extend(collapsed)
            target_lengths.append(len(collapsed))
            if t_i > 0:
                if n_lab >= t_i:
                    buf = frame_labels[:t_i]
                else:  # sped-down clip: pad buffer with the final frame's class out to t_i.
                    buf = np.concatenate(
                        [frame_labels, np.full(t_i - n_lab, frame_labels[-1], dtype=frame_labels.dtype)]
                    )
                frame_label_rows[row_idx, :t_i] = torch.from_numpy(buf)

        batch["ctc_input_lengths"] = input_lengths
        batch["ctc_targets"] = torch.tensor(targets, dtype=torch.long)
        batch["ctc_target_lengths"] = torch.tensor(target_lengths, dtype=torch.long)
        batch["ctc_frame_labels"] = frame_label_rows
