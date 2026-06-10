"""Utilities for reading DementiaBank/TalkBank CHAT transcripts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .special_tokens import (
    SHORT_PAUSE_TOKEN,
    MEDIUM_PAUSE_TOKEN,
    LONG_PAUSE_TOKEN,
    UNCLEAR_SPEECH_TOKEN,
    FILLER_UH_TOKEN,
    FILLER_UM_TOKEN,
    LAUGHTER_TOKEN,
    NON_VERBAL_TOKEN,
)

# Strip CLAN bullet timestamps (\x15NNN_NNN\x15) and bare NNN_NNN line-end timestamps.
_CONTROL_RE = re.compile(r"\x15\d+_\d+\x15")
_TIMESTAMP_RE = re.compile(r"\b\d+_\d+\b")

# [: target form] — strip the annotation, keep the original spoken word.
# e.g. "outta [: out of]" → "outta"
_TARGET_FORM_RE = re.compile(r"\s*\[:\s*[^\]]+?\s*\]")

# Retrace markers: <attempt> [/] or [//] — keep the spoken words, strip CHAT markup.
# Both the attempt and the correction stay in the transcript to match the audio.
_RETRACE_RE = re.compile(r"<([^>]*)>\s*\[/{1,2}\]")

# Remaining [scope] annotations (post-codes [+ gram], etc.)
_SCOPE_RE = re.compile(r"\[[^\]]*\]")

# &+word fragments (e.g. &+dri, &+sh): strip the &+ marker, keep the spoken part.
_FRAG_RE = re.compile(r"&\+(\S+)")

# CHAT tokens to normalize after ordinary CHAT cleanup.
_CHAT_TOKEN_MAP: dict[str, str] = {
    # Pauses and trailing pause markers.
    "(.)": SHORT_PAUSE_TOKEN,
    "(..)": MEDIUM_PAUSE_TOKEN,
    "(...)": LONG_PAUSE_TOKEN,
    "+.": SHORT_PAUSE_TOKEN,
    "+..": MEDIUM_PAUSE_TOKEN,
    "+...": LONG_PAUSE_TOKEN,
    "xxx": UNCLEAR_SPEECH_TOKEN,
    # High-frequency filled pauses become ordinary Whisper-friendly words.
    "&uh": FILLER_UH_TOKEN,
    "&-uh": FILLER_UH_TOKEN,
    "uhuh": FILLER_UH_TOKEN,
    "&um": FILLER_UM_TOKEN,
    "&-um": FILLER_UM_TOKEN,
    "&-you_know": "you know",
    "you_know": "you know",
    "&-like": "like",
    # Non-lexical vocalizations that are not encoded as &= event tokens.
    "&hm": NON_VERBAL_TOKEN,
    "&-hm": NON_VERBAL_TOKEN,
    "&mm": NON_VERBAL_TOKEN,
    "&-mm": NON_VERBAL_TOKEN,
    "&er": NON_VERBAL_TOKEN,
    "&-er": NON_VERBAL_TOKEN,
    # Only laughter-like &= events get their own class.
    "&=laughs": LAUGHTER_TOKEN,
    "&=chuckles": LAUGHTER_TOKEN,
}
_CHAT_TOKEN_RE = re.compile(
    r"(?<!\S)("
    + "|".join(re.escape(token) for token in sorted(_CHAT_TOKEN_MAP, key=len, reverse=True))
    + r")(?![\w&=()+[\]<>@-])"
)


def _normalize_chat_token(m: re.Match[str]) -> str:
    return _CHAT_TOKEN_MAP[m.group(1)]


# Every other CHAT event token maps to [non-verbal]. Examples seen in data include
# &=sighs, &=clears:throat, &=throat:clears, &=coughs, &=groans, &=claps,
# &=head:shake, &=vocalizes, &=mumbles, and other long-tail event labels.
_CHAT_EVENT_RE = re.compile(r"(?<!\S)&=[^\s]+")


def _normalize_chat_event(m: re.Match[str]) -> str:
    return NON_VERBAL_TOKEN


# Bare &word: keep mapped fillers (e.g. &um, &uh) as-is; strip & from everything else.
_BARE_AMP_RE = re.compile(r"&(\w+)")
_BARE_FILLERS: frozenset[str] = frozenset(
    token[1:] for token in _CHAT_TOKEN_MAP
    if token.startswith("&") and len(token) > 1 and token[1] not in ("=", "-", "+")
)

def _bare_amp_sub(m: re.Match[str]) -> str:
    word = m.group(1)
    return m.group(0) if word in _BARE_FILLERS else word


# Words containing any character outside this set are replaced with xxx.
_ALLOWED_CHARS: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?;:'-_"
    "&=()+[]<>@"
)
_TOKEN_RE = re.compile(r"\S+")


def _sanitize_token(m: re.Match[str]) -> str:
    return "xxx" if any(ch not in _ALLOWED_CHARS for ch in m.group(0)) else m.group(0)


# Consecutive word@l (letter-name) tokens → hyphen-joined spelling.
# e.g. "c@l u@l p@l" → "c-u-p", "c@l o@l o@l k@l i@l e@l" → "c-o-o-k-i-e"
_AT_L_SEQ_RE = re.compile(r"\w+@l(?:\s+\w+@l)*")

# Other @code suffixes (e.g. @n neologism, @o onomatopoeia, @u unclear):
# strip the marker, keep the spoken part.
_AT_CODE_RE = re.compile(r"(\w+)@[a-z]+")


def _collapse_letter_seq(m: re.Match[str]) -> str:
    return "-".join(re.findall(r"(\w+)@l", m.group(0)))


# Trailing partial-word elisions: word(letters) → wordletters (e.g. tryin(g) → trying).
_PARTIAL_WORD_RE = re.compile(r"(\w+)\((\w*)\)")

# Leading partial-word elisions: (letters)word → word (e.g. (be)cause → cause).
# The speaker omitted the leading sounds; only the spoken portion is kept.
_LEADING_ELISION_RE = re.compile(r"\(\w+\)(\w+)")

# Space before sentence-final punctuation — normalise to "word." not "word ."
_TERMINAL_SPACE_RE = re.compile(r"\s+([.!?])$")

# Stray <> angle-bracket characters left after retrace/scope removal.
_ANGLE_RE = re.compile(r"[<>]")

# Utterance-overlap and CHAT continuation prefix markers (+< +> +, +^ etc.)
_OVERLAP_MARKER_RE = re.compile(r"^\s*\+[<>,^~]\s*")
_STANDALONE_PLUS_RE = re.compile(r"(?<!\S)\+(?!\S)")

_DEPENDENT_TIER_RE = re.compile(r"^%[A-Za-z]+:")
_MULTISPACE_RE = re.compile(r"\s+")

# Matches both CLAN bullet (\x15NNN_NNN\x15) and bare (NNN_NNN) timestamps anywhere in a line.
# We take the last match per line since timestamps always trail the utterance content.
_LINE_TS_RE = re.compile(r"\x15?(\d+)_(\d+)\x15?")

@dataclass(frozen=True)
class SpeakerSegment:
    """A single speaker turn with its audio timestamp range."""

    speaker: str
    start_ms: int
    end_ms: int

    @property
    def start_s(self) -> float:
        return self.start_ms / 1000

    @property
    def end_s(self) -> float:
        return self.end_ms / 1000


@dataclass(frozen=True)
class TimedUtterance:
    """A cleaned speaker utterance aligned to a timestamp range."""

    speaker: str
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.end_ms - self.start_ms) / 1000)


def extract_speaker_timestamps(
    path: str | Path,
    speaker: str = "PAR",
) -> list[SpeakerSegment]:
    """Return timestamp ranges for each turn by *speaker* that carries a timestamp.

    CHAT files embed audio offsets as ``start_ms_end_ms`` at the end of utterance
    lines. Turns without a timestamp (e.g. overlap continuations) are silently skipped.
    """
    transcript_path = Path(path)
    segments: list[SpeakerSegment] = []
    active_speaker: str | None = None
    last_ts: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal active_speaker, last_ts
        if active_speaker == speaker and last_ts is not None:
            segments.append(SpeakerSegment(speaker=speaker, start_ms=last_ts[0], end_ms=last_ts[1]))
        active_speaker = None
        last_ts = None

    for raw_line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("*"):
            flush()
            code, _, content = line.partition(":")
            active_speaker = code.lstrip("*")
            m = _LINE_TS_RE.search(content)
            last_ts = (int(m.group(1)), int(m.group(2))) if m else None
        elif _DEPENDENT_TIER_RE.match(line) or line.startswith("@"):
            flush()
        elif active_speaker is not None:
            # Continuation line — may carry the timestamp for a multi-line utterance.
            m = _LINE_TS_RE.search(line)
            if m:
                last_ts = (int(m.group(1)), int(m.group(2)))

    flush()
    return segments


def extract_timed_speaker_utterances(
    path: str | Path,
    speaker: str = "PAR",
) -> list[TimedUtterance]:
    """Return cleaned speaker utterances with CHAT timestamp boundaries."""
    transcript_path = Path(path)
    utterances: list[TimedUtterance] = []
    active_speaker: str | None = None
    active_text: list[str] = []
    active_ts: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal active_speaker, active_text, active_ts
        if active_speaker == speaker and active_text and active_ts is not None:
            cleaned = clean_chat_utterance(" ".join(active_text))
            if cleaned:
                utterances.append(
                    TimedUtterance(
                        speaker=speaker,
                        start_ms=active_ts[0],
                        end_ms=active_ts[1],
                        text=cleaned,
                    )
                )
        active_speaker = None
        active_text = []
        active_ts = None

    for raw_line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("*"):
            flush()
            code, _, content = line.partition(":")
            active_speaker = code.lstrip("*")
            active_text = [content]
            ts_match = _LINE_TS_RE.search(content)
            active_ts = (int(ts_match.group(1)), int(ts_match.group(2))) if ts_match else None
            continue
        if _DEPENDENT_TIER_RE.match(line) or line.startswith("@"):
            flush()
            continue
        if active_speaker is not None:
            active_text.append(line)
            ts_match = _LINE_TS_RE.search(line)
            if ts_match:
                active_ts = (int(ts_match.group(1)), int(ts_match.group(2)))

    flush()
    return utterances


@dataclass(frozen=True)
class ChatTranscript:
    """Participant utterances and metadata extracted from one ``.cha`` file."""

    path: Path
    participant_id: str | None
    metadata: dict[str, str]
    utterances: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.utterances).strip()


def parse_id_tier(line: str) -> dict[str, str]:
    """Parse a CHAT ``@ID`` tier into useful metadata fields.

    DementiaBank participant rows look like:
    ``@ID: eng|Pitt|PAR|57;|male|ProbableAD||Participant|18||``.
    """
    if not line.startswith("@ID:"):
        return {}

    parts = [part.strip() for part in line.removeprefix("@ID:").strip().split("|")]
    fields = ["language", "corpus", "speaker", "age", "sex", "diagnosis", "group", "role", "mmse"]
    metadata = {field: value for field, value in zip(fields, parts) if value}
    if "age" in metadata:
        metadata["age"] = metadata["age"].rstrip(";")
    return metadata


def clean_chat_utterance(text: str) -> str:
    """Normalize one CHAT utterance while preserving clinically useful disfluencies.

    Cleaning order matters — each step assumes the previous ones have run:
    1.  Remove CLAN bullet/line-end timestamps.
    2.  Strip [: target] annotations; keep the original spoken word.
    3.  Strip <> and [/]/[//] retrace markup; keep both the attempt and correction.
    4.  Strip &+ fragments and unknown bare-& prefixes, keep spoken part.
    5.  Remove remaining [scope] annotations.
    6.  Strip overlap/continuation markers (+< +> etc.) before angle-bracket removal.
    7.  Strip leftover <> angle brackets.
    8.  Expand trailing partial-word elisions: tryin(g) → trying.
    9.  Expand leading partial-word elisions: (be)cause → cause.
    10. Collapse whitespace; attach terminal punctuation; strip leading/trailing spaces.
    11. Preserve pause markers (.)/(..)/(...), xxx unintelligible markers, and normalize selected fillers/events.
    """
    text = _CONTROL_RE.sub(" ", text)
    text = _TIMESTAMP_RE.sub(" ", text)

    # Strip [: target] annotations; the original spoken word stays.
    text = _TARGET_FORM_RE.sub("", text)

    # Keep the spoken words from retrace sequences; only strip the CHAT markup.
    text = _RETRACE_RE.sub(r"\1", text)

    # &+ fragments: strip marker, keep spoken part.
    # Mapped bare-& fillers (e.g. &um, &uh), &- fillers, and &= events are left as-is
    # until the final normalization pass below.
    # Other bare-& words: strip the & and keep the spoken part.
    text = _FRAG_RE.sub(r"\1", text)
    text = _BARE_AMP_RE.sub(_bare_amp_sub, text)

    text = _SCOPE_RE.sub(" ", text)

    # Strip overlap/continuation markers before _ANGLE_RE removes the < or >.
    text = _OVERLAP_MARKER_RE.sub("", text)

    text = _ANGLE_RE.sub("", text)

    # Expand trailing elisions: tryin(g) → trying.
    text = _PARTIAL_WORD_RE.sub(r"\1\2", text)

    # Expand leading elisions: (be)cause → cause.
    text = _LEADING_ELISION_RE.sub(r"\1", text)

    # @l letter-name sequences → hyphen-joined spelling (must run before _AT_CODE_RE).
    # Other @code suffixes: strip marker, keep spoken part.
    text = _AT_L_SEQ_RE.sub(_collapse_letter_seq, text)
    text = _AT_CODE_RE.sub(r"\1", text)

    # Replace any token still containing non-allowed characters (e.g. IPA) with xxx.
    text = _TOKEN_RE.sub(_sanitize_token, text)

    for old, new in {
        "\t":    " ",
        "\n":    " ",
        "+/.":   " ",      # interrupted utterance
        "+//.":  " ",      # self-interrupted utterance
        "+/?":   " ",      # interrupted question
        "+//?":  " ",      # self-interrupted question
        "‡":     " ",      # lazy overlap marker
    }.items():
        text = text.replace(old, new)

    text = _STANDALONE_PLUS_RE.sub(" ", text)
    text = text.replace("*PAR:", " ")
    text = _MULTISPACE_RE.sub(" ", text)
    text = text.strip(" ,\t\n")
    text = _CHAT_TOKEN_RE.sub(_normalize_chat_token, text)
    text = _CHAT_EVENT_RE.sub(_normalize_chat_event, text)
    text = _MULTISPACE_RE.sub(" ", text)
    return _TERMINAL_SPACE_RE.sub(r"\1", text)


def read_chat_transcript(path: str | Path) -> ChatTranscript:
    """Read a CHAT file and return participant-only transcript text."""
    transcript_path = Path(path)
    metadata: dict[str, str] = {}
    participant_id: str | None = None
    utterances: list[str] = []
    active_speaker: str | None = None
    active_text: list[str] = []

    def flush_active() -> None:
        nonlocal active_speaker, active_text
        if active_speaker == "PAR" and active_text:
            cleaned = clean_chat_utterance(" ".join(active_text))
            if cleaned:
                utterances.append(cleaned)
        active_speaker = None
        active_text = []

    for raw_line in transcript_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.startswith("@ID:") and "|PAR|" in line:
            metadata.update(parse_id_tier(line))
            participant_id = transcript_path.stem
            continue
        if line.startswith("*"):
            flush_active()
            speaker, _, content = line.partition(":")
            active_speaker = speaker.lstrip("*")
            active_text = [content]
            continue
        if _DEPENDENT_TIER_RE.match(line) or line.startswith("@"):
            flush_active()
            continue
        if active_speaker is not None:
            active_text.append(line)

    flush_active()
    return ChatTranscript(
        path=transcript_path,
        participant_id=participant_id,
        metadata=metadata,
        utterances=utterances,
    )
