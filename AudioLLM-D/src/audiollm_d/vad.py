"""CrisperWhisper VAD silence handling for the v5 relabel experiment.

v5 changes exactly one variable from the failed v4: the *source* of the auxiliary CTC head's
frame labels. v4 derived pause labels from the CHAT transcript (via the MFA alignment JSONs —
see ``ctc_labels.py``); v5 derives them from acoustically-grounded silence segments detected by
CrisperWhisper (a Whisper-large with accurate verbatim word timestamps).

This module is the torch-free, GPU-free core shared by the three v5 scripts:

  * Stage 0 (``scripts/crisperwhisper_vad.py``): turn CrisperWhisper word timestamps into a
    per-clip silence timeline — :func:`silences_from_word_timestamps`, :func:`save_silences`.
  * Stage 1 / GATE (``scripts/vad_chat_diff.py``): compare CHAT-marked pauses against the VAD
    silences — :func:`chat_pauses_from_alignment`, :func:`match_chat_to_vad`,
    :func:`vad_only_segments`, :func:`calibrate_thresholds`.
  * Stage 2 (``scripts/build_ctc_labels_vad.py``): bin VAD silences by the Stage-1 thresholds and
    rasterize onto the 25 fps frame grid — :func:`classify_duration`,
    :func:`rasterize_vad_silences`.

Everything here is deterministic and unit-tested (``tests/test_vad.py``) so the pipeline can be
validated on CPU before GPU time is spent on Stage 0 inference and Stage 3 retraining.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# Reuse the frame-grid primitives so the v5 labels live on exactly the same grid as v4's. The
# CTC vocab is shared too; v5 simply instantiates it with the pause-only class list below.
from audiollm_d.ctc_labels import (
    FRAME_SECONDS,
    SPEECH_CLASS,
    CtcVocab,
    n_frames_for_duration,
)

# The CHAT bracket tokens that mark a pause, as they appear in the alignment JSON "word" field,
# mapped to the v5 CTC pause class they correspond to. These are the ONLY events the VAD head
# supervises — laughter / non-verbal / unclear / uh / um stay with the LM head (VAD can't label
# them; per the ticket the CTC vocab carries no non-silent tokens).
CHAT_PAUSE_TOKENS: dict[str, str] = {
    "[short-pause]": "short-pause",
    "[medium-pause]": "medium-pause",
    "[long-pause]": "long-pause",
}

# v5 CTC event classes (``speech`` first by convention; ``blank`` appended by CtcVocab). This is
# the deliberately reduced, silence-only vocab. If Stage 1 finds the three pause categories are
# not acoustically separable, collapse this (e.g. ``["speech", "pause"]`` or
# ``["speech", "pause", "long-pause"]``) and pass --classes to Stage 2 accordingly.
V5_PAUSE_CLASSES: tuple[str, ...] = (SPEECH_CLASS, "short-pause", "medium-pause", "long-pause")


def v5_vocab(classes: Sequence[str] | None = None) -> CtcVocab:
    """The v5 CTC vocab (pause-only). ``classes`` overrides the default for a Stage-1 collapse."""
    return CtcVocab.from_classes(tuple(classes) if classes else V5_PAUSE_CLASSES)


# ---------------------------------------------------------------------------------------------
# Stage 0: CrisperWhisper word timestamps -> silence timeline
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SilenceSegment:
    """A single detected non-speech span, in seconds. ``duration == end - start`` (validated)."""

    start: float
    end: float
    duration: float

    @classmethod
    def from_span(cls, start: float, end: float) -> "SilenceSegment":
        start = float(start)
        end = float(end)
        return cls(start=start, end=end, duration=max(0.0, end - start))

    def to_json(self) -> dict[str, float]:
        return {
            "start_s": round(self.start, 4),
            "end_s": round(self.end, 4),
            "duration_s": round(self.duration, 4),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SilenceSegment":
        return cls(
            start=float(payload["start_s"]),
            end=float(payload["end_s"]),
            duration=float(payload["duration_s"]),
        )


def silences_from_word_timestamps(
    words: Sequence[dict[str, Any]],
    clip_duration: float,
    *,
    min_silence_s: float = 0.10,
    include_edges: bool = True,
    edge_min_silence_s: float | None = None,
) -> list[SilenceSegment]:
    """Derive silence segments from CrisperWhisper word-level timestamps.

    CrisperWhisper returns a list of words, each with a ``(start, end)`` timestamp (in seconds).
    The silences are the *gaps* between consecutive words, plus optionally the leading gap
    (clip start -> first word) and trailing gap (last word -> clip end). A gap counts as a
    silence only if it is at least ``min_silence_s`` long (edges use ``edge_min_silence_s`` when
    given, else ``min_silence_s``); shorter gaps are normal co-articulation, not pauses.

    ``words`` items must expose ``start``/``end`` floats (see :func:`normalize_word_timestamps`
    which adapts the various shapes the HF pipeline / CrisperWhisper emit). Words are sorted by
    start time defensively. Returns silences in time order.
    """
    edge_floor = min_silence_s if edge_min_silence_s is None else edge_min_silence_s
    spans = [
        (float(w["start"]), float(w["end"]))
        for w in words
        if w.get("start") is not None and w.get("end") is not None
    ]
    spans.sort(key=lambda s: s[0])

    silences: list[SilenceSegment] = []
    if not spans:
        # No words at all -> the whole clip is "silence" (almost certainly a load/inference
        # failure; Stage 0 flags clips with zero silences OR zero words for review).
        if clip_duration > 0:
            silences.append(SilenceSegment.from_span(0.0, clip_duration))
        return silences

    if include_edges:
        first_start = spans[0][0]
        if first_start >= edge_floor:
            silences.append(SilenceSegment.from_span(0.0, first_start))

    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        gap = next_start - prev_end
        if gap >= min_silence_s:
            silences.append(SilenceSegment.from_span(prev_end, next_start))

    if include_edges:
        last_end = spans[-1][1]
        if clip_duration - last_end >= edge_floor:
            silences.append(SilenceSegment.from_span(last_end, clip_duration))

    return silences


def _exact_zero_runs(x: Any, min_len_samples: int) -> list[tuple[int, int]]:
    """Contiguous runs of *exactly* 0.0 samples at least ``min_len_samples`` long, as (start, end).

    These are digital-silence artifacts: real recordings have a nonzero noise floor and essentially
    never contain a long solid block of literal zeros. In this corpus they are the constant 0.6 s
    pads ``build_cd_clips.py`` inserts between stitched turns (``audio.py``, ``pause_seconds=0.6``).
    """
    import numpy as np

    if min_len_samples <= 0:
        return []
    z = (np.asarray(x) == 0.0).astype(np.int8)
    if z.sum() == 0:
        return []
    d = np.diff(np.concatenate([[0], z, [0]]))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s) >= min_len_samples]


def _subtract_intervals(
    spans: list[tuple[int, int]], cuts: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Remove every ``cuts`` interval from every ``spans`` interval, splitting as needed."""
    if not cuts:
        return spans
    out: list[tuple[int, int]] = []
    for s, e in spans:
        pieces = [(s, e)]
        for cs, ce in cuts:
            nxt: list[tuple[int, int]] = []
            for ps, pe in pieces:
                if ce <= ps or cs >= pe:  # no overlap
                    nxt.append((ps, pe))
                    continue
                if cs > ps:
                    nxt.append((ps, cs))
                if ce < pe:
                    nxt.append((ce, pe))
            pieces = nxt
        out.extend(pieces)
    return out


def detect_silences_from_waveform(
    samples: Any,
    sample_rate: int,
    *,
    min_silence_s: float = 0.10,
    top_db: float = 30.0,
    frame_ms: float = 30.0,
    hop_ms: float = 10.0,
    min_speech_s: float = 0.05,
    include_edges: bool = True,
    drop_zero_inserts: bool = False,
    zero_insert_min_s: float = 0.40,
) -> list[SilenceSegment]:
    """Acoustically-grounded silence detection straight from the waveform (energy VAD).

    This is the v5 silence source. CrisperWhisper word-timestamp *gaps* are unusable here: the
    model packs pause time into word/space-token durations, leaving inter-word gaps ~0, so a
    gap-based extractor finds nothing. Instead we read silence directly from short-time energy.

    Algorithm (deterministic, numpy-only so it stays CPU-testable):

    1. Frame the signal (``frame_ms`` window, ``hop_ms`` hop) and take per-frame RMS.
    2. A frame is *speech* when its energy is within ``top_db`` dB of the clip's reference level
       (95th-percentile RMS — robust to clicks vs a plain max).
    3. Coalesce speech frames into sample intervals; drop speech islands shorter than
       ``min_speech_s`` (clicks/coughs that would otherwise split a real pause).
    4. The silences are the complement of the speech intervals within ``[0, duration]`` (leading
       and trailing edges included unless ``include_edges`` is False), keeping only spans
       >= ``min_silence_s``.

    With ``drop_zero_inserts`` the synthetic stitch pads (contiguous *exact*-zero runs
    >= ``zero_insert_min_s``) are *excised* from the detected silences before the min-length
    filter -- so a real pause that abuts a 0.6 s pad keeps its genuine portion instead of being
    dropped wholesale. This is what makes the v5 acoustic-vs-CHAT contrast meaningful on the
    stitched clips: it removes the transcript-derived artifact and keeps the real acoustic pauses.

    Returns silences in time order. Output is byte-for-byte the same ``SilenceSegment`` shape the
    rest of the v5 pipeline (Stage 1 diff, Stage 2 rasterize) already consumes.
    """
    import numpy as np

    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    n = x.shape[0]
    if n == 0 or sample_rate <= 0:
        return []
    duration = n / float(sample_rate)

    frame = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    n_frames = 1 + max(0, (n - frame) // hop)

    # Per-frame RMS energy.
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame]
        rms[i] = math.sqrt(float(np.mean(seg * seg))) if seg.size else 0.0

    ref = float(np.percentile(rms, 95)) if n_frames else 0.0
    if ref <= 0.0:
        # Entire clip is silent (or empty) -> one silence spanning the whole clip.
        return [SilenceSegment.from_span(0.0, duration)] if duration > 0 else []

    eps = 1e-10
    db = 20.0 * np.log10(np.maximum(rms, eps) / ref)
    speech_mask = db > -float(top_db)

    # Speech frames -> sample intervals, then merge overlaps (frame > hop makes them overlap).
    speech_intervals: list[list[int]] = []
    in_run = False
    run_start = 0
    for i in range(n_frames):
        if speech_mask[i] and not in_run:
            in_run, run_start = True, i
        elif not speech_mask[i] and in_run:
            in_run = False
            speech_intervals.append([run_start * hop, min((i - 1) * hop + frame, n)])
    if in_run:
        speech_intervals.append([run_start * hop, n])

    # Drop short speech islands (clicks) so they don't fragment a genuine pause.
    min_speech_samples = int(round(min_speech_s * sample_rate))
    speech_intervals = [iv for iv in speech_intervals if (iv[1] - iv[0]) >= min_speech_samples]

    # Merge any intervals that now touch/overlap.
    merged: list[list[int]] = []
    for iv in speech_intervals:
        if merged and iv[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], iv[1])
        else:
            merged.append(list(iv))

    # Silences = gaps between speech intervals (plus edges) within [0, n].
    silences: list[SilenceSegment] = []
    cursor = 0
    for s, e in merged:
        if s > cursor:
            silences.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < n:
        silences.append((cursor, n))

    # Excise synthetic digital-silence inserts (the constant turn-stitch pads), keeping any real
    # pause that surrounds them. Done before the min-length filter so split remnants are re-checked.
    if drop_zero_inserts:
        inserts = _exact_zero_runs(x, int(round(zero_insert_min_s * sample_rate)))
        silences = _subtract_intervals(silences, inserts)

    out: list[SilenceSegment] = []
    min_silence_samples = int(round(min_silence_s * sample_rate))
    for s, e in silences:
        if not include_edges and (s == 0 or e == n):
            continue
        if (e - s) >= min_silence_samples:
            out.append(SilenceSegment.from_span(s / sample_rate, e / sample_rate))
    return out


def normalize_word_timestamps(raw_words: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    """Adapt CrisperWhisper / HF-pipeline word dicts to ``{"word", "start", "end"}`` floats.

    The HF ASR pipeline with ``return_timestamps="word"`` yields chunks shaped
    ``{"text": "okay", "timestamp": (0.0, 1.86)}``; some CrisperWhisper paths instead yield
    ``{"word": ..., "start": ..., "end": ...}``. Words whose timestamp is incomplete (a trailing
    ``None`` end is common on the last chunk) are dropped — they can't bound a gap. Kept separate
    from :func:`silences_from_word_timestamps` so the parsing is independently testable.
    """
    out: list[dict[str, float]] = []
    for w in raw_words:
        text = w.get("word", w.get("text", ""))
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            ts = w.get("timestamp")
            if ts is not None and len(ts) == 2:
                start, end = ts
        if start is None or end is None:
            continue
        out.append({"word": str(text).strip(), "start": float(start), "end": float(end)})
    return out


def save_silences(path: str | Path, clip_id: str, segments: Sequence[SilenceSegment]) -> None:
    """Write a per-clip silence file: ``{"clip_id", "n_segments", "segments": [...]}``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "clip_id": clip_id,
        "n_segments": len(segments),
        "segments": [s.to_json() for s in segments],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_silences(path: str | Path) -> list[SilenceSegment]:
    """Read a per-clip silence file written by :func:`save_silences`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [SilenceSegment.from_json(s) for s in payload["segments"]]


# ---------------------------------------------------------------------------------------------
# Stage 1: CHAT pauses vs VAD silences (the GATE)
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatPause:
    """A transcript-marked pause from the alignment JSON, with its CHAT category and span."""

    category: str  # "short-pause" | "medium-pause" | "long-pause"
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def chat_pauses_from_alignment(alignment: Sequence[dict[str, Any]]) -> list[ChatPause]:
    """Extract CHAT-marked pauses (short/medium/long) from one MFA alignment JSON.

    The alignment is the same artifact v4 consumed: a list of ``{"word", "start", "end",
    [type]}`` segments. We keep only the three pause brackets; their ``start``/``end`` give the
    transcript-anchored location used to find the temporally-overlapping VAD silence.
    """
    pauses: list[ChatPause] = []
    for seg in alignment:
        category = CHAT_PAUSE_TOKENS.get(seg.get("word", ""))
        if category is None:
            continue
        pauses.append(ChatPause(category=category, start=float(seg["start"]), end=float(seg["end"])))
    return pauses


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Length of the temporal overlap of ``[a_start, a_end)`` and ``[b_start, b_end)`` (>= 0)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


@dataclass
class ChatVadMatch:
    """One CHAT pause and the VAD silence (if any) that overlaps it most."""

    pause: ChatPause
    matched: bool
    vad_duration: float | None  # measured duration of the best-overlapping VAD silence
    overlap_s: float


def match_chat_to_vad(
    chat_pauses: Sequence[ChatPause],
    vad_silences: Sequence[SilenceSegment],
    *,
    min_overlap_s: float = 0.02,
) -> list[ChatVadMatch]:
    """For each CHAT pause, find the VAD silence it overlaps most and record that silence's
    *measured* duration.

    A pause is ``matched`` when the best overlap is at least ``min_overlap_s``. The recorded
    ``vad_duration`` is the full measured duration of the matched VAD silence (the acoustic
    truth we want to threshold on), not the overlap length.
    """
    matches: list[ChatVadMatch] = []
    for pause in chat_pauses:
        best_overlap = 0.0
        best_seg: SilenceSegment | None = None
        for seg in vad_silences:
            ov = interval_overlap(pause.start, pause.end, seg.start, seg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_seg = seg
        matched = best_seg is not None and best_overlap >= min_overlap_s
        matches.append(
            ChatVadMatch(
                pause=pause,
                matched=matched,
                vad_duration=best_seg.duration if (matched and best_seg) else None,
                overlap_s=best_overlap,
            )
        )
    return matches


def vad_only_segments(
    chat_pauses: Sequence[ChatPause],
    vad_silences: Sequence[SilenceSegment],
    *,
    min_overlap_s: float = 0.02,
) -> list[SilenceSegment]:
    """VAD silences that overlap *no* CHAT pause — silence the transcript never marked.

    A high VAD-only rate is the signal that relabeling adds information (premise holds); a near-
    zero rate is the STOP condition (VAD ~= CHAT -> labels were never the problem).
    """
    extra: list[SilenceSegment] = []
    for seg in vad_silences:
        overlaps_any = any(
            interval_overlap(seg.start, seg.end, p.start, p.end) >= min_overlap_s
            for p in chat_pauses
        )
        if not overlaps_any:
            extra.append(seg)
    return extra


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (``q`` in [0, 100]); no numpy dependency for the core."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (q / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


@dataclass
class CategoryStats:
    """Summary of the measured VAD durations for one CHAT pause category."""

    category: str
    n: int
    mean: float
    median: float
    p10: float
    p25: float
    p75: float
    p90: float

    def to_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "n": self.n,
            "mean": round(self.mean, 4),
            "median": round(self.median, 4),
            "p10": round(self.p10, 4),
            "p25": round(self.p25, 4),
            "p75": round(self.p75, 4),
            "p90": round(self.p90, 4),
        }


def summarize_category_durations(durations: Sequence[float], category: str) -> CategoryStats:
    """Percentile summary of one category's matched VAD durations."""
    vals = [float(d) for d in durations]
    n = len(vals)
    mean = sum(vals) / n if n else float("nan")
    return CategoryStats(
        category=category,
        n=n,
        mean=mean,
        median=_percentile(vals, 50),
        p10=_percentile(vals, 10),
        p25=_percentile(vals, 25),
        p75=_percentile(vals, 75),
        p90=_percentile(vals, 90),
    )


def calibrate_thresholds(per_category_durations: dict[str, Sequence[float]]) -> dict[str, float]:
    """Derive ``{short_max, medium_max}`` duration boundaries from the per-category VAD
    distributions (Stage-1 percentile decision, replacing v4's 0.3/1.0/2.5 guess).

    Boundary between two adjacent categories = midpoint of (lower category's p75, upper
    category's p25) when both are populated — the natural separation point if the distributions
    roughly separate. Falls back to the midpoint of the two medians, then to the old guess, so
    the function is always defined even on sparse data. A silence with measured duration ``d``
    is then ``short-pause`` if ``d < short_max``, ``medium-pause`` if ``d < medium_max``, else
    ``long-pause`` (see :func:`classify_duration`).
    """
    short = [float(x) for x in per_category_durations.get("short-pause", [])]
    medium = [float(x) for x in per_category_durations.get("medium-pause", [])]
    long = [float(x) for x in per_category_durations.get("long-pause", [])]

    def boundary(lower: list[float], upper: list[float], fallback: float) -> float:
        if lower and upper:
            sep = (_percentile(lower, 75) + _percentile(upper, 25)) / 2.0
            if math.isfinite(sep) and sep > 0:
                return round(sep, 3)
            mid = (_percentile(lower, 50) + _percentile(upper, 50)) / 2.0
            if math.isfinite(mid) and mid > 0:
                return round(mid, 3)
        return fallback

    short_max = boundary(short, medium, fallback=1.0)
    medium_max = boundary(medium, long, fallback=2.5)
    # Keep them ordered even if the distributions invert (heavy overlap case).
    if medium_max <= short_max:
        medium_max = round(short_max + 0.5, 3)
    return {"short_max": short_max, "medium_max": medium_max}


def save_thresholds(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_thresholds(path: str | Path) -> dict[str, float]:
    """Load a Stage-1 threshold file; returns the ``{short_max, medium_max}`` mapping."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("thresholds", payload)


# ---------------------------------------------------------------------------------------------
# Stage 2: VAD silences -> 25 fps frame labels
# ---------------------------------------------------------------------------------------------


def classify_duration(duration: float, thresholds: dict[str, float]) -> str:
    """Bin a measured silence ``duration`` into a pause class using Stage-1 thresholds.

    ``duration < short_max`` -> ``short-pause``; ``< medium_max`` -> ``medium-pause``; else
    ``long-pause``. When the vocab is collapsed (e.g. binary), Stage 2 only emits classes that
    exist in its vocab — see :func:`rasterize_vad_silences`.
    """
    short_max = thresholds["short_max"]
    medium_max = thresholds["medium_max"]
    if duration < short_max:
        return "short-pause"
    if duration < medium_max:
        return "medium-pause"
    return "long-pause"


def rasterize_vad_silences(
    silences: Sequence[SilenceSegment],
    n_frames: int,
    vocab: CtcVocab,
    thresholds: dict[str, float],
):
    """Rasterize VAD silences into dense per-frame class ids of length ``n_frames``.

    Every frame defaults to ``speech``; each silence is binned by measured duration
    (:func:`classify_duration`) and painted onto the frames whose centre time
    ``(f + 0.5) * 0.04`` falls in ``[start, end)`` — identical frame math to v4's
    ``ctc_labels.rasterize_alignment`` so the two label sets are directly comparable. A bin
    whose class is absent from a collapsed ``vocab`` folds to the longest class present (so a
    binary {speech, pause} vocab still fires on every silence).

    Returns an ``int8`` numpy array. ``blank`` never appears.
    """
    import numpy as np

    class_to_index = vocab.class_to_index
    speech_id = class_to_index[SPEECH_CLASS]
    labels = np.full(n_frames, speech_id, dtype=np.int8)
    if n_frames == 0:
        return labels

    # Collapsed-vocab fallback: prefer the exact class; otherwise fold to the FIRST silence
    # class in vocab order so no silence is silently dropped to speech. This makes the two
    # intended collapses behave correctly: binary ["speech","pause"] -> every bin becomes
    # "pause"; 3-way ["speech","pause","long-pause"] -> "long-pause" matches exactly and
    # short/medium fold to "pause".
    silence_classes = [c for c in vocab.classes if c != SPEECH_CLASS]

    def resolve_class(name: str) -> int:
        if name in class_to_index:
            return class_to_index[name]
        if silence_classes:
            return class_to_index[silence_classes[0]]
        return speech_id

    for seg in silences:
        cls_name = classify_duration(seg.duration, thresholds)
        cls_id = resolve_class(cls_name)
        if cls_id == speech_id:
            continue
        start_f = int(math.ceil(seg.start / FRAME_SECONDS - 0.5))
        end_f = int(math.ceil(seg.end / FRAME_SECONDS - 0.5))
        start_f = max(start_f, 0)
        end_f = min(end_f, n_frames)
        if end_f > start_f:
            labels[start_f:end_f] = cls_id
    return labels


__all__ = [
    "CHAT_PAUSE_TOKENS",
    "V5_PAUSE_CLASSES",
    "v5_vocab",
    "SilenceSegment",
    "detect_silences_from_waveform",
    "silences_from_word_timestamps",
    "normalize_word_timestamps",
    "save_silences",
    "load_silences",
    "ChatPause",
    "chat_pauses_from_alignment",
    "interval_overlap",
    "ChatVadMatch",
    "match_chat_to_vad",
    "vad_only_segments",
    "CategoryStats",
    "summarize_category_durations",
    "calibrate_thresholds",
    "save_thresholds",
    "load_thresholds",
    "classify_duration",
    "rasterize_vad_silences",
    "n_frames_for_duration",
]
