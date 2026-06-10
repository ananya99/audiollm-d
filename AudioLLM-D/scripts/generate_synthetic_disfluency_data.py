#!/usr/bin/env python
"""Generate synthetic disfluency clips using neural TTS + token event insertion.

Preview mode creates one sample per special token. Pauses are true digital
silence with speech before and after so they reflect realistic context.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import edge_tts
import librosa
import numpy as np
import soundfile as sf

from audiollm_d.special_tokens import SPECIAL_TOKENS


VOICE_POOL = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
    "en-AU-NatashaNeural",
]
_CHATTERBOX_MODEL: Any | None = None
_CHATTERBOX_DEVICE: str | None = None
_CHATTERBOX_VARIANT: str = "turbo"
_CHATTERBOX_AUDIO_PROMPT_PATH: str | None = None
_CHATTERBOX_MODELS: dict[str, Any] = {}


@dataclass(frozen=True)
class SyntheticRecord:
    clip_id: str
    audio_path: str
    text: str
    speaker_id: str
    split: str
    source: str
    token_focus: str


@dataclass(frozen=True)
class VoiceStyle:
    style_id: str
    voice: str
    rate_choices: tuple[str, ...]
    pitch_choices: tuple[str, ...]


VOICE_STYLES: tuple[VoiceStyle, ...] = (
    VoiceStyle("older_female", "en-US-JennyNeural", ("-18%", "-14%", "-10%"), ("-4Hz", "-2Hz", "+0Hz")),
    VoiceStyle("older_male", "en-US-DavisNeural", ("-20%", "-16%", "-12%"), ("-4Hz", "-2Hz", "+0Hz")),
    VoiceStyle("neutral_female", "en-US-AriaNeural", ("-4%", "+0%", "+4%"), ("-1Hz", "+0Hz", "+1Hz")),
    VoiceStyle("neutral_male", "en-US-GuyNeural", ("-4%", "+0%", "+4%"), ("-1Hz", "+0Hz", "+1Hz")),
    VoiceStyle("younger_female", "en-AU-NatashaNeural", ("+8%", "+12%", "+16%"), ("+2Hz", "+3Hz", "+4Hz")),
    VoiceStyle("younger_female_uk", "en-GB-SoniaNeural", ("+6%", "+10%", "+14%"), ("+1Hz", "+2Hz", "+3Hz")),
)

DEFAULT_HOUR_TARGET_MINUTES: dict[str, float] = {
    "[long-pause]": 16.0,
    "[unclear-speech]": 12.0,
    "[short-pause]": 8.0,
    "[medium-pause]": 7.0,
    "[laughter]": 6.0,
    "[non-verbal]": 5.0,
    "[uh]": 2.0,
    "[um]": 2.0,
    "[mixed]": 2.0,
}

GENERIC_CONTEXTS: tuple[tuple[str, str], ...] = (
    ("I was talking about something from earlier", "and then I continued in the same way"),
    ("I started to explain the situation clearly", "and then I finished the thought"),
    ("I was describing what happened that day", "and after that I kept speaking"),
    ("I tried to remember all the details", "and then I moved on to the next part"),
    ("I was telling a simple story", "and then I carried on naturally"),
    ("I was answering a question slowly", "and then I completed the sentence"),
    ("I paused while thinking for a moment", "and then I resumed speaking"),
    ("I was giving a short explanation", "and then I added one more detail"),
)

COOKIE_THEFT_CONTEXTS: tuple[tuple[str, str], ...] = (
    ("The boy is reaching for the cookie jar", "and the stool looks unstable"),
    ("The girl is watching her brother at the counter", "and she seems ready to help"),
    ("The mother is busy at the sink", "while water is spilling onto the floor"),
    ("The kitchen looks crowded in this picture", "and everyone is doing something different"),
    ("One child is taking cookies from the jar", "and crumbs may fall from the counter"),
    ("The sink is overflowing in the background", "while the children focus on the cookies"),
    ("I can see the mother washing dishes", "and she does not notice the children"),
    ("The scene shows children near the cupboard", "while the mother keeps working at the sink"),
)

GENERIC_SUBJECTS: tuple[str, ...] = (
    "I",
    "We",
    "My friend",
    "The speaker",
    "The person",
)
GENERIC_PRE_VERBS: tuple[str, ...] = (
    "was explaining",
    "started describing",
    "kept talking about",
    "tried to summarize",
    "was reflecting on",
    "was describing",
)
GENERIC_PRE_OBJECTS: tuple[str, ...] = (
    "what happened earlier",
    "the situation from this morning",
    "the details of the event",
    "the main idea carefully",
    "the story in simple words",
    "the sequence of events",
)
GENERIC_POST_STARTS: tuple[str, ...] = (
    "and then",
    "after that",
    "so then",
    "and afterwards",
    "and finally",
)
GENERIC_POST_VERBS: tuple[str, ...] = (
    "continued speaking",
    "finished the thought",
    "added one more detail",
    "moved to the next point",
    "wrapped up the sentence",
    "kept the explanation going",
)

COOKIE_SUBJECTS: tuple[str, ...] = (
    "The boy",
    "The girl",
    "The mother",
    "The children",
    "The scene",
)
COOKIE_PRE_VERBS: tuple[str, ...] = (
    "is reaching toward",
    "is standing near",
    "is looking at",
    "is focused on",
    "is moving close to",
)
COOKIE_PRE_OBJECTS: tuple[str, ...] = (
    "the cookie jar",
    "the kitchen counter",
    "the sink area",
    "the cupboard",
    "the stool by the counter",
)
COOKIE_POST_STARTS: tuple[str, ...] = (
    "while",
    "and",
    "as",
    "at the same time",
)
COOKIE_POST_VERBS: tuple[str, ...] = (
    "water spills from the sink",
    "the stool looks unstable",
    "the mother keeps washing dishes",
    "the children watch each other",
    "crumbs may fall from the counter",
)


def _normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    max_abs = float(np.max(np.abs(audio)))
    if max_abs < 1e-8:
        return audio.astype(np.float32)
    return (audio * (peak / max_abs)).astype(np.float32)


def _fade(audio: np.ndarray, sr: int, fade_ms: float = 12.0) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    n = int(sr * fade_ms * 1e-3)
    if n <= 0 or audio.size < 2 * n:
        return audio.astype(np.float32)
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = audio.astype(np.float32, copy=True)
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def _concat(segments: list[np.ndarray], sr: int, gap_s: float = 0.025) -> np.ndarray:
    gap = np.zeros(int(sr * gap_s), dtype=np.float32)
    out: list[np.ndarray] = []
    for idx, seg in enumerate(segments):
        out.append(_fade(seg.astype(np.float32), sr))
        if idx < len(segments) - 1 and gap.size > 0:
            out.append(gap)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


async def _synth_to_mp3(text: str, *, voice: str, rate: str, pitch: str, output: Path) -> None:
    communicate = edge_tts.Communicate(text, voice=voice, rate=rate, pitch=pitch)
    await communicate.save(str(output))


def _tts_audio(
    text: str,
    *,
    sr: int,
    voice: str,
    cache_dir: Path,
    backend: str,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    chatterbox_exaggeration: float | None = None,
    chatterbox_cfg_weight: float | None = None,
    chatterbox_temperature: float | None = None,
    chatterbox_variant_override: str | None = None,
    chatterbox_audio_prompt_override: str | None = None,
) -> np.ndarray:
    if backend == "chatterbox":
        return _tts_audio_chatterbox(
            text=text,
            sr=sr,
            cache_dir=cache_dir,
            voice=voice,
            variant=chatterbox_variant_override or _CHATTERBOX_VARIANT,
            audio_prompt_path=(
                chatterbox_audio_prompt_override
                if chatterbox_audio_prompt_override is not None
                else _CHATTERBOX_AUDIO_PROMPT_PATH
            ),
            exaggeration=chatterbox_exaggeration,
            cfg_weight=chatterbox_cfg_weight,
            temperature=chatterbox_temperature,
        )
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{voice}|{rate}|{pitch}|{sr}|{text}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    mp3_path = cache_dir / f"{h}.mp3"
    if not mp3_path.exists():
        # Edge TTS can intermittently return empty audio for a specific voice.
        # Retry with fallback voices before failing.
        last_error: Exception | None = None
        for candidate_voice in (voice, "en-US-AriaNeural", "en-US-GuyNeural"):
            try:
                asyncio.run(
                    _synth_to_mp3(
                        text,
                        voice=candidate_voice,
                        rate=rate,
                        pitch=pitch,
                        output=mp3_path,
                    )
                )
                if mp3_path.exists() and mp3_path.stat().st_size > 0:
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        if not mp3_path.exists() or mp3_path.stat().st_size == 0:
            if last_error is not None:
                raise last_error
            raise RuntimeError(f"Edge TTS failed to synthesize text: {text[:80]!r}")
    try:
        audio, _ = librosa.load(str(mp3_path), sr=sr, mono=True)
    except Exception:  # noqa: BLE001
        # Rare transient where the file was not readable immediately after save.
        if mp3_path.exists():
            mp3_path.unlink()
        asyncio.run(
            _synth_to_mp3(
                text,
                voice="en-US-AriaNeural",
                rate=rate,
                pitch=pitch,
                output=mp3_path,
            )
        )
        audio, _ = librosa.load(str(mp3_path), sr=sr, mono=True)
    return _normalize(audio.astype(np.float32), peak=0.9)


def _load_chatterbox_model(*, variant: str):
    global _CHATTERBOX_DEVICE, _CHATTERBOX_MODELS
    if variant in _CHATTERBOX_MODELS:
        return _CHATTERBOX_MODELS[variant]
    try:
        import torch
        from chatterbox.tts import ChatterboxTTS
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except Exception as exc:
        raise RuntimeError(
            "Chatterbox backend requested but chatterbox-tts is not importable. "
            "Install with: pip install chatterbox-tts"
        ) from exc
    _CHATTERBOX_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if variant == "turbo":
        _CHATTERBOX_MODELS[variant] = ChatterboxTurboTTS.from_pretrained(device=_CHATTERBOX_DEVICE)
    else:
        _CHATTERBOX_MODELS[variant] = ChatterboxTTS.from_pretrained(device=_CHATTERBOX_DEVICE)
    return _CHATTERBOX_MODELS[variant]


def _tts_audio_chatterbox(
    *,
    text: str,
    sr: int,
    cache_dir: Path,
    voice: str,
    variant: str,
    audio_prompt_path: str | None,
    exaggeration: float | None,
    cfg_weight: float | None,
    temperature: float | None,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (
        f"chatterbox|{variant}|{audio_prompt_path}|{exaggeration}|{cfg_weight}|"
        f"{temperature}|{sr}|{text}"
    )
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    wav_cache = cache_dir / f"{h}.wav"
    if wav_cache.exists():
        audio, _ = librosa.load(str(wav_cache), sr=sr, mono=True)
        return _normalize(audio.astype(np.float32), peak=0.9)

    model = _load_chatterbox_model(variant=variant)
    generate_kwargs: dict[str, Any] = {}
    resolved_prompt = audio_prompt_path
    if variant == "turbo" and not resolved_prompt:
        resolved_prompt = _ensure_turbo_audio_prompt(cache_dir=cache_dir, sr=sr, voice=voice)
    if resolved_prompt:
        generate_kwargs["audio_prompt_path"] = resolved_prompt
    if exaggeration is not None:
        generate_kwargs["exaggeration"] = float(exaggeration)
    if cfg_weight is not None:
        generate_kwargs["cfg_weight"] = float(cfg_weight)
    if temperature is not None:
        generate_kwargs["temperature"] = float(temperature)
    wav = model.generate(text, **generate_kwargs)
    # Chatterbox returns torch tensor (shape [T] or [1,T]).
    wav_np = wav.detach().cpu().float().numpy()
    if wav_np.ndim == 2:
        wav_np = wav_np[0]
    out_sr = int(getattr(model, "sr", sr))
    wav_np = wav_np.astype(np.float32, copy=False)
    if out_sr != sr:
        wav_np = librosa.resample(wav_np, orig_sr=out_sr, target_sr=sr).astype(np.float32)
    wav_np = _normalize(wav_np, peak=0.9)
    sf.write(wav_cache, wav_np, sr)
    return wav_np


def _ensure_turbo_audio_prompt(*, cache_dir: Path, sr: int, voice: str) -> str:
    """Create a neutral >5s reference clip for Turbo when none is provided."""
    prompt_wav = cache_dir / "turbo_auto_prompt.wav"
    if prompt_wav.exists():
        existing, _ = librosa.load(str(prompt_wav), sr=sr, mono=True)
        if len(existing) / float(sr) > 5.2:
            return str(prompt_wav)
    prompt_mp3 = cache_dir / "turbo_auto_prompt.mp3"
    prompt_text = (
        "I am speaking in a calm and natural voice. "
        "This is a longer reference sample so the model can match my speaking style. "
        "I continue talking for a few more seconds in the same tone and pace."
    )
    asyncio.run(
        _synth_to_mp3(
            prompt_text,
            voice=voice,
            rate="+0%",
            pitch="+0Hz",
            output=prompt_mp3,
        )
    )
    prompt_audio, _ = librosa.load(str(prompt_mp3), sr=sr, mono=True)
    prompt_audio = prompt_audio.astype(np.float32)
    # Ensure >5s by padding with trailing silence if needed.
    min_samples = int(5.4 * sr)
    if len(prompt_audio) < min_samples:
        prompt_audio = np.pad(prompt_audio, (0, min_samples - len(prompt_audio)))
    sf.write(prompt_wav, _normalize(prompt_audio, peak=0.9), sr)
    return str(prompt_wav)


def _ensure_issue30_laugh_prompt(*, cache_dir: Path, sr: int, voice: str) -> str:
    """Create a 'bubbly' reference clip as suggested in issue #30 discussion."""
    prompt_wav = cache_dir / "issue30_laugh_prompt.wav"
    if prompt_wav.exists():
        return str(prompt_wav)
    prompt_mp3 = cache_dir / "issue30_laugh_prompt.mp3"
    prompt_text = (
        "This is so funny, haha, I cannot stop laughing. "
        "Everything sounds cheerful and lively in this sample."
    )
    asyncio.run(
        _synth_to_mp3(
            prompt_text,
            voice=voice,
            rate="+6%",
            pitch="+2Hz",
            output=prompt_mp3,
        )
    )
    prompt_audio, _ = librosa.load(str(prompt_mp3), sr=sr, mono=True)
    sf.write(prompt_wav, _normalize(prompt_audio.astype(np.float32), peak=0.9), sr)
    return str(prompt_wav)


def _silence(duration_s: float, sr: int) -> np.ndarray:
    return np.zeros(max(1, int(duration_s * sr)), dtype=np.float32)


def _degrade_unclear_speech(audio: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    n = len(audio)
    if n == 0:
        return audio.astype(np.float32)
    # Stronger garbling without additive white noise:
    # severe low-pass + codec-like resampling + temporal dropouts.
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = np.fft.rfft(audio.astype(np.float64))
    cutoff = rng.uniform(500.0, 1100.0)
    low_pass_mask = np.where(freqs <= cutoff, 1.0, np.exp(-(freqs - cutoff) / 260.0))
    muffled = np.fft.irfft(spectrum * low_pass_mask, n=n).astype(np.float32)
    codec_rate = int(rng.choice([2400, 3000, 3600]))
    codec = librosa.resample(muffled, orig_sr=sr, target_sr=codec_rate)
    codec = librosa.resample(codec, orig_sr=codec_rate, target_sr=sr).astype(np.float32)
    if len(codec) >= n:
        degraded = codec[:n]
    else:
        degraded = np.pad(codec, (0, n - len(codec)))
    chunk = int(0.03 * sr)
    if chunk > 0:
        for start in range(0, n, chunk):
            if rng.random() < 0.18:
                degraded[start : min(start + chunk, n)] *= rng.uniform(0.03, 0.2)
    return _normalize(_fade(degraded, sr), peak=0.72)


def _token_template(token: str) -> tuple[str, str, str]:
    templates = {
        "[short-pause]": (
            "I was trying to explain what happened",
            token,
            "and then I continued with the same thought",
        ),
        "[medium-pause]": (
            "I started talking about it earlier today",
            token,
            "and after that I kept going with the story",
        ),
        "[long-pause]": (
            "I wanted to describe the situation clearly",
            token,
            "and then I picked up the sentence again",
        ),
        "[unclear-speech]": (
            "I said a part that was difficult to hear",
            token,
            "then I repeated the rest in a normal way",
        ),
        "[laughter]": (
            "I was telling a simple story",
            token,
            "and then I returned to speaking normally",
        ),
        "[non-verbal]": (
            "I paused while I was talking",
            token,
            "and then I carried on with the sentence",
        ),
        "[uh]": (
            "I was trying to remember the details",
            token,
            "and then I finished the thought",
        ),
        "[um]": (
            "I was speaking a little slowly",
            token,
            "and after that I completed the sentence",
        ),
    }
    return templates[token]


def _compose_generic_context(rng: np.random.Generator) -> tuple[str, str]:
    pre = (
        f"{rng.choice(GENERIC_SUBJECTS)} {rng.choice(GENERIC_PRE_VERBS)} "
        f"{rng.choice(GENERIC_PRE_OBJECTS)}"
    )
    post = f"{rng.choice(GENERIC_POST_STARTS)} {rng.choice(GENERIC_POST_VERBS)}"
    return pre, post


def _compose_cookie_context(rng: np.random.Generator) -> tuple[str, str]:
    pre = (
        f"{rng.choice(COOKIE_SUBJECTS)} {rng.choice(COOKIE_PRE_VERBS)} "
        f"{rng.choice(COOKIE_PRE_OBJECTS)}"
    )
    post = f"{rng.choice(COOKIE_POST_STARTS)} {rng.choice(COOKIE_POST_VERBS)}"
    return pre, post


def _sample_text_triplet(
    token: str,
    rng: np.random.Generator,
    *,
    used_texts: set[str] | None = None,
) -> tuple[str, str, str]:
    # Prefer compositional contexts for high uniqueness, with fallback to
    # curated templates for robustness.
    for _ in range(80):
        if rng.random() < 0.45:
            pre, post = _compose_cookie_context(rng)
        else:
            pre, post = _compose_generic_context(rng)
        candidate = f"{pre} {token} {post}"
        if used_texts is None or candidate not in used_texts:
            return (pre, token, post)

    contexts = COOKIE_THEFT_CONTEXTS if rng.random() < 0.45 else GENERIC_CONTEXTS
    pre, post = contexts[int(rng.integers(0, len(contexts)))]
    return (pre, token, post)


def _sample_voice_style(rng: np.random.Generator) -> VoiceStyle:
    return VOICE_STYLES[int(rng.integers(0, len(VOICE_STYLES)))]


def _sample_rate_pitch(style: VoiceStyle, rng: np.random.Generator) -> tuple[str, str]:
    rate = style.rate_choices[int(rng.integers(0, len(style.rate_choices)))]
    pitch = style.pitch_choices[int(rng.integers(0, len(style.pitch_choices)))]
    return rate, pitch


def _time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    return librosa.effects.time_stretch(audio.astype(np.float32), rate=rate).astype(np.float32)


def _event_candidates(event_bank_dir: Path | None, token: str) -> list[Path]:
    if event_bank_dir is None or not event_bank_dir.exists():
        return []
    token_key = token.strip("[]").replace("-", "_")
    all_wavs = sorted(event_bank_dir.glob("*.wav"))
    matches = [p for p in all_wavs if token_key in p.stem.lower().replace("-", "_")]
    if matches:
        return matches
    # Fallbacks for naming variations.
    if token == "[non-verbal]":
        return [p for p in all_wavs if "non" in p.stem.lower() or "verbal" in p.stem.lower()]
    if token == "[laughter]":
        return [p for p in all_wavs if "laugh" in p.stem.lower() or "chuckle" in p.stem.lower()]
    return []


def _sample_event_from_bank(
    *,
    candidates: list[Path],
    sr: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    if not candidates:
        return None
    chosen = candidates[int(rng.integers(0, len(candidates)))]
    audio, _ = librosa.load(str(chosen), sr=sr, mono=True)
    audio = audio.astype(np.float32)
    # Trim leading/trailing silence if present in source clips.
    intervals = librosa.effects.split(audio, top_db=28)
    if len(intervals) > 0:
        audio = np.concatenate([audio[s:e] for s, e in intervals]).astype(np.float32)
    if audio.size == 0:
        return None
    # Gentle variation to avoid exact copy artifacts.
    audio = _time_stretch(audio, rate=float(rng.uniform(0.9, 1.1)))
    audio = librosa.effects.pitch_shift(
        audio.astype(np.float32),
        sr=sr,
        n_steps=float(rng.uniform(-1.0, 1.0)),
    ).astype(np.float32)
    audio *= float(rng.uniform(0.8, 1.05))
    return _normalize(_fade(audio, sr), peak=0.82)


def _fit_event_duration(
    audio: np.ndarray,
    *,
    sr: int,
    rng: np.random.Generator,
    min_s: float,
    max_s: float,
) -> np.ndarray:
    """Stretch/repeat/trim an event to a target duration window."""
    target = float(rng.uniform(min_s, max_s))
    target_n = max(1, int(target * sr))
    if audio.size == 0:
        return np.zeros(target_n, dtype=np.float32)
    out = audio.astype(np.float32)
    if len(out) < target_n:
        pieces = [out]
        while sum(len(p) for p in pieces) < target_n:
            pieces.append(np.zeros(int(sr * rng.uniform(0.04, 0.10)), dtype=np.float32))
            pieces.append(out)
        out = np.concatenate(pieces)
    if len(out) > target_n:
        out = out[:target_n]
    return _normalize(_fade(out, sr), peak=0.82)


def _realistic_laughter(
    *,
    sr: int,
    voice: str,
    cache_dir: Path,
    tts_backend: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build laughter-like vocalization from varied syllables, not plain 'ha ha ha'."""
    if tts_backend == "chatterbox":
        # Force non-lexical event generation first (Turbo tags), because
        # sentence prompts can suppress laugh acoustics.
        tag_prompt = str(rng.choice(["[laugh]", "[chuckle]", "[laugh] [chuckle] [laugh]"]))
        laughed = _tts_audio(
            tag_prompt,
            sr=sr,
            voice=voice,
            cache_dir=cache_dir,
            backend=tts_backend,
            chatterbox_variant_override="turbo",
            chatterbox_temperature=0.9,
        )
        intervals = librosa.effects.split(laughed, top_db=30)
        if len(intervals) > 0:
            # Keep all voiced regions, not just one, to preserve laugh rhythm.
            chunks = [laughed[int(s) : int(e)] for s, e in intervals if int(e) - int(s) > int(0.04 * sr)]
            if chunks:
                laughed = np.concatenate(chunks).astype(np.float32)
        if len(laughed) > int(0.18 * sr):
            return _fit_event_duration(laughed, sr=sr, rng=rng, min_s=1.2, max_s=2.2)

        # Fallback to issue #30 text+prompt strategy when tags fail to emit.
        issue30_prompt = _ensure_issue30_laugh_prompt(cache_dir=cache_dir, sr=sr, voice=voice)
        laugh_text = str(
            rng.choice(
                [
                    "This is so funny hahaha I cannot stop laughing!",
                    "That was hilarious haha, I am laughing so hard right now!",
                ]
            )
        )
        laughed = _tts_audio(
            laugh_text,
            sr=sr,
            voice=voice,
            cache_dir=cache_dir,
            backend=tts_backend,
            chatterbox_variant_override="standard",
            chatterbox_audio_prompt_override=issue30_prompt,
            chatterbox_exaggeration=float(rng.uniform(0.9, 1.3)),
            chatterbox_cfg_weight=float(rng.uniform(0.35, 0.55)),
            chatterbox_temperature=0.8,
        )
        intervals = librosa.effects.split(laughed, top_db=30)
        if len(intervals) > 0:
            best = max(intervals, key=lambda it: int(it[1] - it[0]))
            laughed = laughed[int(best[0]) : int(best[1])].astype(np.float32)
        return _fit_event_duration(laughed, sr=sr, rng=rng, min_s=1.2, max_s=2.2)

    if tts_backend == "chatterbox" and _CHATTERBOX_VARIANT == "turbo":
        # README: paralinguistic tags are native to Chatterbox-Turbo.
        # Use tag-only prompts to avoid lexical words around the event.
        tagged = _tts_audio(
            str(rng.choice(["[laugh]", "[chuckle]", "[laugh] [chuckle]"])),
            sr=sr,
            voice=voice,
            cache_dir=cache_dir,
            backend=tts_backend,
        )
        intervals = librosa.effects.split(tagged, top_db=30)
        if len(intervals) > 0:
            # Keep the longest voiced region to reduce trailing artifacts.
            best = max(intervals, key=lambda it: int(it[1] - it[0]))
            tagged = tagged[int(best[0]) : int(best[1])].astype(np.float32)
        return _fit_event_duration(tagged, sr=sr, rng=rng, min_s=1.2, max_s=2.2)

    syllables = [
        _tts_audio("ha", sr=sr, voice=voice, cache_dir=cache_dir, backend=tts_backend, rate="+10%"),
        _tts_audio("heh", sr=sr, voice=voice, cache_dir=cache_dir, backend=tts_backend, rate="+6%"),
        _tts_audio("huh", sr=sr, voice=voice, cache_dir=cache_dir, backend=tts_backend, rate="+2%"),
    ]
    bursts: list[np.ndarray] = []
    burst_count = int(rng.integers(5, 9))
    for _ in range(burst_count):
        base = syllables[int(rng.integers(0, len(syllables)))]
        seg = _time_stretch(base, rate=float(rng.uniform(0.82, 1.25)))
        semitones = float(rng.uniform(-2.5, 3.0))
        seg = librosa.effects.pitch_shift(seg.astype(np.float32), sr=sr, n_steps=semitones).astype(
            np.float32
        )
        seg = seg * float(rng.uniform(0.65, 1.0))
        bursts.append(_fade(seg, sr))
        bursts.append(_silence(float(rng.uniform(0.03, 0.09)), sr))
    breath_len = int(sr * rng.uniform(0.10, 0.22))
    breath = rng.standard_normal(breath_len).astype(np.float32)
    breath = np.convolve(breath, np.ones(64, dtype=np.float32) / 64.0, mode="same")
    breath /= max(np.max(np.abs(breath)), 1e-6)
    breath *= float(rng.uniform(0.05, 0.14))
    out = np.concatenate([_fade(breath, sr), *bursts]) if bursts else breath
    out = _normalize(_fade(out.astype(np.float32), sr), peak=0.8)
    return _fit_event_duration(out, sr=sr, rng=rng, min_s=1.2, max_s=2.2)


def _event_audio(
    token: str,
    *,
    sr: int,
    voice: str,
    cache_dir: Path,
    event_bank_dir: Path | None,
    strict_real_events: bool,
    tts_backend: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if token == "[short-pause]":
        return _silence(rng.uniform(0.18, 0.34), sr)
    if token == "[medium-pause]":
        return _silence(rng.uniform(0.45, 0.85), sr)
    if token == "[long-pause]":
        return _silence(rng.uniform(1.2, 2.4), sr)
    if token == "[unclear-speech]":
        clear = _tts_audio(
            "I am speaking but this part should be unclear now",
            sr=sr,
            voice=voice,
            cache_dir=cache_dir,
            backend=tts_backend,
            rate="-4%",
            pitch="-1Hz",
        )
        return _degrade_unclear_speech(clear, sr, rng)
    if token == "[laughter]":
        bank_clip = _sample_event_from_bank(
            candidates=_event_candidates(event_bank_dir, token),
            sr=sr,
            rng=rng,
        )
        if bank_clip is not None:
            return _fit_event_duration(bank_clip, sr=sr, rng=rng, min_s=1.2, max_s=2.2)
        if strict_real_events:
            raise ValueError(
                "No real laughter clip found in event bank. "
                "Provide --event-bank-dir with good laughter wav samples."
            )
        return _realistic_laughter(
            sr=sr, voice=voice, cache_dir=cache_dir, tts_backend=tts_backend, rng=rng
        )
    if token == "[non-verbal]":
        bank_clip = _sample_event_from_bank(
            candidates=_event_candidates(event_bank_dir, token),
            sr=sr,
            rng=rng,
        )
        if bank_clip is not None:
            return bank_clip
        if strict_real_events:
            raise ValueError(
                "No real non-verbal clip found in event bank. "
                "Provide --event-bank-dir with good non-verbal wav samples."
            )
        if tts_backend == "chatterbox":
            cue_text = str(
                rng.choice(
                    [
                        "[cough] [cough]",
                        "[cough] [cough] [cough]",
                        "[cough] er",
                        "er er",
                    ]
                )
            )
            nonverbal = _tts_audio(
                cue_text,
                sr=sr,
                voice=voice,
                cache_dir=cache_dir,
                backend=tts_backend,
            )
            return _fit_event_duration(nonverbal, sr=sr, rng=rng, min_s=0.8, max_s=1.8)
        # Edge fallback: synthesize varied non-verbal vocal cues so this class
        # does not collapse to only "hmm".
        cues = [
            "er",
            "mm-hmm",
            "cough",
            "clears throat",
            "hmm",
        ]
        segs: list[np.ndarray] = []
        n_cues = int(rng.integers(1, 4))
        for _ in range(n_cues):
            cue = str(rng.choice(cues))
            seg = _tts_audio(
                cue,
                sr=sr,
                voice=voice,
                cache_dir=cache_dir,
                backend=tts_backend,
                rate=f"{int(rng.integers(-15, 10)):+d}%",
                pitch=f"{int(rng.integers(-5, 3)):+d}Hz",
            )
            segs.append(seg)
            if rng.random() < 0.8:
                segs.append(_silence(float(rng.uniform(0.04, 0.12)), sr))
        nonverbal = _concat(segs, sr=sr, gap_s=0.0) if segs else _silence(0.5, sr)
        return _fit_event_duration(nonverbal, sr=sr, rng=rng, min_s=0.8, max_s=1.8)
    if token == "[uh]":
        return _tts_audio("uh", sr=sr, voice=voice, cache_dir=cache_dir, backend=tts_backend)
    if token == "[um]":
        return _tts_audio("um", sr=sr, voice=voice, cache_dir=cache_dir, backend=tts_backend)
    raise ValueError(f"Unsupported token: {token}")


def _build_token_clip(
    token: str,
    *,
    sr: int,
    voice: str,
    rate: str,
    pitch: str,
    cache_dir: Path,
    event_bank_dir: Path | None,
    strict_real_events: bool,
    tts_backend: str,
    text_triplet: tuple[str, str, str] | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    pre_text, token_text, post_text = text_triplet or _token_template(token)

    pre = _tts_audio(
        pre_text,
        sr=sr,
        voice=voice,
        cache_dir=cache_dir,
        backend=tts_backend,
        rate=rate,
        pitch=pitch,
    )
    event = _event_audio(
        token,
        sr=sr,
        voice=voice,
        cache_dir=cache_dir,
        event_bank_dir=event_bank_dir,
        strict_real_events=strict_real_events,
        tts_backend=tts_backend,
        rng=rng,
    )
    post = _tts_audio(
        post_text,
        sr=sr,
        voice=voice,
        cache_dir=cache_dir,
        backend=tts_backend,
        rate=rate,
        pitch=pitch,
    )
    full_text = f"{pre_text} {token_text} {post_text}"
    # Add a slightly longer post-cue pause after non-verbal and filler events.
    if token in {"[non-verbal]", "[um]", "[uh]"}:
        clip = _concat([pre, event, _silence(float(rng.uniform(0.10, 0.18)), sr), post], sr=sr, gap_s=0.03)
    else:
        clip = _concat([pre, event, post], sr=sr, gap_s=0.03)
    return _normalize(clip, peak=0.92), full_text


def _build_mixed_clip(
    *,
    sr: int,
    voice: str,
    rate: str,
    pitch: str,
    cache_dir: Path,
    event_bank_dir: Path | None,
    strict_real_events: bool,
    tts_backend: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    pre_text, _, post_text = _sample_text_triplet("[short-pause]", rng, used_texts=None)
    mixed_tokens = ["[long-pause]", "[unclear-speech]", "[non-verbal]", "[laughter]"]
    t1 = str(rng.choice(mixed_tokens))
    t2 = str(rng.choice(mixed_tokens))
    pre = _tts_audio(pre_text, sr=sr, voice=voice, cache_dir=cache_dir, backend=tts_backend, rate=rate, pitch=pitch)
    ev1 = _event_audio(
        t1,
        sr=sr,
        voice=voice,
        cache_dir=cache_dir,
        event_bank_dir=event_bank_dir,
        strict_real_events=strict_real_events,
        tts_backend=tts_backend,
        rng=rng,
    )
    ev2 = _event_audio(
        t2,
        sr=sr,
        voice=voice,
        cache_dir=cache_dir,
        event_bank_dir=event_bank_dir,
        strict_real_events=strict_real_events,
        tts_backend=tts_backend,
        rng=rng,
    )
    post = _tts_audio(post_text, sr=sr, voice=voice, cache_dir=cache_dir, backend=tts_backend, rate=rate, pitch=pitch)
    text = f"{pre_text} {t1} {t2} {post_text}"
    clip = _concat(
        [pre, ev1, _silence(float(rng.uniform(0.06, 0.12)), sr), ev2, _silence(float(rng.uniform(0.10, 0.18)), sr), post],
        sr=sr,
        gap_s=0.03,
    )
    return _normalize(clip, peak=0.92), text


def _write_manifest(rows: list[SyntheticRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def generate_preview(
    *,
    output_dir: Path,
    sr: int,
    split: str,
    speaker_prefix: str,
    event_bank_dir: Path | None,
    strict_real_events: bool,
    tts_backend: str,
    seed: int,
) -> list[SyntheticRecord]:
    rng = np.random.default_rng(seed)
    audio_dir = output_dir / "audio"
    tts_cache_dir = output_dir / "_tts_cache"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SyntheticRecord] = []

    for idx, token in enumerate(SPECIAL_TOKENS, start=1):
        clip_id = f"preview_{idx:02d}_{token.strip('[]').replace('-', '_')}"
        voice = VOICE_POOL[(idx - 1) % len(VOICE_POOL)]
        style = VOICE_STYLES[(idx - 1) % len(VOICE_STYLES)]
        rate, pitch = _sample_rate_pitch(style, rng)
        audio, text = _build_token_clip(
            token,
            sr=sr,
            voice=voice,
            rate=rate,
            pitch=pitch,
            cache_dir=tts_cache_dir,
            event_bank_dir=event_bank_dir,
            strict_real_events=strict_real_events,
            tts_backend=tts_backend,
            text_triplet=None,
            rng=rng,
        )
        audio_path = audio_dir / f"{clip_id}.wav"
        sf.write(audio_path, audio, sr)
        rows.append(
            SyntheticRecord(
                clip_id=clip_id,
                audio_path=str(audio_path.resolve()),
                text=text,
                speaker_id=f"{speaker_prefix}_{idx:02d}",
                split=split,
                source=f"synthetic_disfluency_generator_v4_{tts_backend}",
                token_focus=token,
            )
        )
    _write_manifest(rows, output_dir / "manifest.jsonl")
    return rows


def generate_hour_dataset(
    *,
    output_dir: Path,
    sr: int,
    split: str,
    speaker_prefix: str,
    event_bank_dir: Path | None,
    strict_real_events: bool,
    tts_backend: str,
    seed: int,
) -> list[SyntheticRecord]:
    rng = np.random.default_rng(seed)
    audio_dir = output_dir / "audio"
    tts_cache_dir = output_dir / "_tts_cache"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows: list[SyntheticRecord] = []
    clip_idx = 0
    used_texts: set[str] = set()

    for token_focus, minutes in DEFAULT_HOUR_TARGET_MINUTES.items():
        target_s = float(minutes) * 60.0
        accumulated = 0.0
        while accumulated < target_s:
            style = _sample_voice_style(rng)
            rate, pitch = _sample_rate_pitch(style, rng)
            if token_focus == "[mixed]":
                audio, text = _build_mixed_clip(
                    sr=sr,
                    voice=style.voice,
                    rate=rate,
                    pitch=pitch,
                    cache_dir=tts_cache_dir,
                    event_bank_dir=event_bank_dir,
                    strict_real_events=strict_real_events,
                    tts_backend=tts_backend,
                    rng=rng,
                )
                clip_id = f"hour_{clip_idx:05d}_mixed"
                speaker_id = f"{speaker_prefix}_{style.style_id}"
            else:
                text_triplet = _sample_text_triplet(token_focus, rng, used_texts=used_texts)
                audio, text = _build_token_clip(
                    token_focus,
                    sr=sr,
                    voice=style.voice,
                    rate=rate,
                    pitch=pitch,
                    cache_dir=tts_cache_dir,
                    event_bank_dir=event_bank_dir,
                    strict_real_events=strict_real_events,
                    tts_backend=tts_backend,
                    text_triplet=text_triplet,
                    rng=rng,
                )
                clip_id = f"hour_{clip_idx:05d}_{token_focus.strip('[]').replace('-', '_')}"
                speaker_id = f"{speaker_prefix}_{style.style_id}"

            audio_path = audio_dir / f"{clip_id}.wav"
            sf.write(audio_path, audio, sr)
            dur = len(audio) / float(sr)
            accumulated += dur
            rows.append(
                SyntheticRecord(
                    clip_id=clip_id,
                    audio_path=str(audio_path.resolve()),
                    text=text,
                    speaker_id=speaker_id,
                    split=split,
                    source=f"synthetic_disfluency_generator_v5_{tts_backend}",
                    token_focus=token_focus,
                )
            )
            used_texts.add(text)
            clip_idx += 1

    _write_manifest(rows, output_dir / "manifest.jsonl")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preview", "hour"], default="preview")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--tts-backend",
        choices=["edge", "chatterbox"],
        default="edge",
        help="TTS backend for context speech and filler event generation.",
    )
    parser.add_argument(
        "--chatterbox-variant",
        choices=["turbo", "standard"],
        default="turbo",
        help="Chatterbox model variant when --tts-backend chatterbox.",
    )
    parser.add_argument(
        "--chatterbox-audio-prompt",
        default=None,
        help="Optional reference wav path passed as audio_prompt_path for Chatterbox generation.",
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--speaker-prefix", default="syn")
    parser.add_argument(
        "--event-bank-dir",
        default=None,
        help="Optional directory of real event wavs (e.g., laughter/non-verbal seeds).",
    )
    parser.add_argument(
        "--strict-real-events",
        action="store_true",
        help="Require real event-bank clips for [laughter] and [non-verbal].",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_bank_dir = (
        Path(args.event_bank_dir).expanduser().resolve() if args.event_bank_dir else None
    )
    global _CHATTERBOX_VARIANT, _CHATTERBOX_AUDIO_PROMPT_PATH
    _CHATTERBOX_VARIANT = str(args.chatterbox_variant)
    _CHATTERBOX_AUDIO_PROMPT_PATH = (
        str(Path(args.chatterbox_audio_prompt).expanduser().resolve())
        if args.chatterbox_audio_prompt
        else None
    )
    if args.mode == "preview":
        rows = generate_preview(
            output_dir=output_dir,
            sr=args.sample_rate,
            split=args.split,
            speaker_prefix=args.speaker_prefix,
            event_bank_dir=event_bank_dir,
            strict_real_events=args.strict_real_events,
            tts_backend=args.tts_backend,
            seed=args.seed,
        )
    else:
        rows = generate_hour_dataset(
            output_dir=output_dir,
            sr=args.sample_rate,
            split=args.split,
            speaker_prefix=args.speaker_prefix,
            event_bank_dir=event_bank_dir,
            strict_real_events=args.strict_real_events,
            tts_backend=args.tts_backend,
            seed=args.seed,
        )
    print(f"Wrote {len(rows)} {args.mode} clips to {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
