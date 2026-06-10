#!/usr/bin/env python
"""Post-process MFA TextGrid outputs into per-token alignment JSON.

Anchors MFA timestamps back to the ORIGINAL processed transcripts — the output
JSON contains one record per token in the original .txt, in order.

For each token type:
  - Regular words / contractions: timestamp from the MFA speech interval.
  - [um], [uh], [laughter], [non-verbal], [unclear-speech]: MFA aligns these as
    _um/_uh/_laugh/_nvb/_unk respectively, so they get real intervals.
    [unclear-speech] specifically maps to the `spn` (spoken-noise) phone — MFA
    assigns it the audio span that best matches the signal; accuracy is lower
    than for real words but it gives a usable time range.
  - [short-pause], [medium-pause], [long-pause]: no MFA interval exists (they
    were stripped from the .lab). Their timestamp is the silence gap between the
    surrounding MFA speech intervals.

Usage:
    python scripts/postprocess_mfa.py \
        --textgrid-dir mfa/alignments \
        --transcript-dir merged/merged_data_processed/transcripts \
        --output-dir mfa/processed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Token constants (must match special_tokens.py and prepare_mfa_input.py)
# ---------------------------------------------------------------------------
_PAUSE_TOKENS = {"[short-pause]", "[medium-pause]", "[long-pause]"}

_BRACKET_TO_MFA: dict[str, str] = {
    "[um]": "_um",
    "[uh]": "_uh",
    "[laughter]": "_laugh",
    "[non-verbal]": "_nvb",
    "[unclear-speech]": "_unk",
}
_MFA_TO_BRACKET = {v: k for k, v in _BRACKET_TO_MFA.items()}

_ALL_BRACKET_TOKENS = _PAUSE_TOKENS | set(_BRACKET_TO_MFA)

_BRACKET_RE = re.compile(
    r"\[(?:"
    + "|".join(
        re.escape(t[1:-1])
        for t in sorted(_ALL_BRACKET_TOKENS, key=len, reverse=True)
    )
    + r")\]"
)

_SILENCE_LABELS = {"", "sp", "sil", "<eps>"}

# Duration thresholds from CLAUDE.md project spec (seconds).
_SHORT_MIN = 0.3
_MEDIUM_MIN = 1.0
_LONG_MIN = 2.5


# ---------------------------------------------------------------------------
# Transcript tokenizer
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split transcript into tokens, preserving bracket tokens as single units."""
    tokens: list[str] = []
    pos = 0
    for m in _BRACKET_RE.finditer(text):
        # Flush any plain words before this bracket token.
        chunk = text[pos:m.start()]
        # Strip punctuation and split on whitespace.
        chunk = re.sub(r"[,.\?!;]", " ", chunk)
        chunk = re.sub(r":+", "", chunk)  # CHAT prolongation colons
        for w in chunk.split():
            tokens.append(w.lower())
        tokens.append(m.group(0))
        pos = m.end()
    # Flush remainder.
    chunk = text[pos:]
    chunk = re.sub(r"[,.\?!;]", " ", chunk)
    chunk = re.sub(r":+", "", chunk)
    for w in chunk.split():
        tokens.append(w.lower())
    return tokens


def _mfa_form(token: str) -> str | None:
    """Return the word as it appears in the .lab / TextGrid, or None for pause tokens."""
    if token in _PAUSE_TOKENS:
        return None
    return _BRACKET_TO_MFA.get(token, token)


# ---------------------------------------------------------------------------
# TextGrid parser
# ---------------------------------------------------------------------------

def _parse_textgrid_words(tg_path: Path) -> tuple[list[tuple[float, float, str]], float]:
    """Return (intervals, total_duration) for the words tier.

    intervals: list of (xmin, xmax, label) for every interval.
    total_duration: TextGrid xmax (full clip length in seconds).
    """
    text = tg_path.read_text(encoding="utf-8", errors="replace")

    # Extract top-level xmax (clip duration).
    tg_xmax_match = re.search(r"^xmax\s*=\s*([\d.eE+\-]+)", text, re.MULTILINE)
    total_duration = float(tg_xmax_match.group(1)) if tg_xmax_match else 0.0

    words_block = re.search(
        r'name\s*=\s*"words".*?(?=name\s*=\s*"|$)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if words_block is None:
        raise ValueError(f"No 'words' tier in {tg_path}")

    interval_re = re.compile(
        r"xmin\s*=\s*([\d.eE+\-]+)\s+"
        r"xmax\s*=\s*([\d.eE+\-]+)\s+"
        r'text\s*=\s*"([^"]*)"',
        re.DOTALL,
    )
    intervals = [
        (float(m.group(1)), float(m.group(2)), m.group(3).strip())
        for m in interval_re.finditer(words_block.group(0))
    ]
    return intervals, total_duration


# ---------------------------------------------------------------------------
# Core alignment: anchor TextGrid intervals back to original transcript tokens
# ---------------------------------------------------------------------------

def align_transcript(
    original_text: str,
    tg_intervals: list[tuple[float, float, str]],
) -> list[dict]:
    """Produce one alignment record per token in original_text.

    Speech tokens consume MFA intervals in order.
    Pause tokens consume the silence gap that spans their position in the original.
    """
    orig_tokens = _tokenize(original_text)
    if not orig_tokens:
        return []

    # Separate speech intervals (have a word label) from silence intervals.
    # We keep ALL intervals in order so we can extract silence gaps.
    # speech_queue: (xmin, xmax, label) for non-silence intervals only.
    speech_queue: list[tuple[float, float, str]] = [
        (s, e, l) for s, e, l in tg_intervals if l not in _SILENCE_LABELS
    ]

    # Build a flat list of (start, end) for all silence spans so we can look up
    # which silence falls between two given speech intervals.
    # Simpler: keep all intervals ordered and track 'prev_speech_end' as we walk.

    # We'll walk speech_queue in order, one item per non-pause token.
    speech_idx = 0
    records: list[dict] = []

    # Track the end time of the last consumed speech interval.
    last_speech_end: float = tg_intervals[0][0] if tg_intervals else 0.0

    # For each pause token we need the gap between prev speech end and next speech start.
    # Build a lookahead: for each position in orig_tokens that is a pause token,
    # find the index of the next non-pause token.

    for i, tok in enumerate(orig_tokens):
        if tok in _PAUSE_TOKENS:
            # Silence token: timestamp = gap between surrounding speech intervals.
            # "Start" = end of last consumed speech interval.
            # "End"   = start of next upcoming speech interval (if any).
            next_speech_start: float | None = None
            if speech_idx < len(speech_queue):
                next_speech_start = speech_queue[speech_idx][0]

            gap_start = last_speech_end
            gap_end = next_speech_start if next_speech_start is not None else last_speech_end

            if gap_end > gap_start:
                records.append({
                    "word": tok,
                    "start": round(gap_start, 4),
                    "end": round(gap_end, 4),
                    "type": "silence",
                })
            else:
                # Degenerate gap (back-to-back speech): emit with zero width so
                # the token is still present in the JSON at the right position.
                records.append({
                    "word": tok,
                    "start": round(gap_start, 4),
                    "end": round(gap_start, 4),
                    "type": "silence",
                    "note": "no_gap",
                })
        else:
            # Speech token: consume next interval from MFA queue.
            mfa_expected = _mfa_form(tok)  # e.g. "_um", "okay"

            if speech_idx >= len(speech_queue):
                # MFA ran out of intervals — alignment failure for remaining tokens.
                records.append({"word": tok, "aligned": False})
                continue

            xmin, xmax, label = speech_queue[speech_idx]

            # Sanity-check: label should match expected (with MFA lowercasing).
            # If not, it's likely a text-normalization difference — accept it
            # but flag it so callers can filter if needed.
            label_clean = label.strip().lower()
            expected_clean = (mfa_expected or "").strip().lower()
            ok = (label_clean == expected_clean)

            rec: dict = {
                "word": tok,
                "start": round(xmin, 4),
                "end": round(xmax, 4),
            }
            if not ok:
                rec["mfa_label"] = label  # for debugging mismatches
            records.append(rec)

            last_speech_end = xmax
            speech_idx += 1

    return records


# ---------------------------------------------------------------------------
# Continuity post-processing
# ---------------------------------------------------------------------------

def make_continuous(records: list[dict], total_duration: float) -> list[dict]:
    """Bridge any silence gap between consecutive tokens so the timeline is gapless.

    Clips are stitched from multiple utterances, so MFA produces silence gaps at
    utterance boundaries that don't correspond to any pause token in the transcript.
    We absorb each such gap by extending the preceding token's 'end' to the next
    token's 'start', making the token segmentation cover every audio frame.

    Pause tokens (type == 'silence') already span their gap via align_transcript,
    so this function only extends tokens where a gap remains after that step.

    The last token is extended to total_duration so the full clip is covered.
    """
    if not records:
        return records

    result = [dict(r) for r in records]

    for i in range(len(result) - 1):
        gap = round(result[i + 1]["start"] - result[i]["end"], 6)
        if gap > 0:
            result[i]["end"] = result[i + 1]["start"]

    # Extend last token to clip end.
    if total_duration > 0 and result[-1]["end"] < total_duration:
        result[-1]["end"] = round(total_duration, 4)

    return result


# ---------------------------------------------------------------------------
# Directory-level processing
# ---------------------------------------------------------------------------

def postprocess_directory(
    textgrid_dir: Path,
    transcript_dir: Path,
    output_dir: Path,
) -> tuple[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tg_files = sorted(textgrid_dir.rglob("*.TextGrid"))

    n_ok = 0
    n_err = 0

    for tg_path in tg_files:
        stem = tg_path.stem
        txt_path = transcript_dir / (stem + ".txt")
        if not txt_path.exists():
            print(f"[warn] {stem}: no transcript found at {txt_path}", file=sys.stderr)
            n_err += 1
            continue

        try:
            intervals, total_duration = _parse_textgrid_words(tg_path)
        except Exception as exc:
            print(f"[error] {stem}: TextGrid parse failed: {exc}", file=sys.stderr)
            n_err += 1
            continue

        if not intervals:
            print(f"[warn] {stem}: empty TextGrid, skipping.", file=sys.stderr)
            n_err += 1
            continue

        original_text = txt_path.read_text(encoding="utf-8").strip()
        records = align_transcript(original_text, intervals)
        records = make_continuous(records, total_duration)

        if not records:
            print(f"[warn] {stem}: no alignment records produced.", file=sys.stderr)
            n_err += 1
            continue

        json_path = output_dir / (stem + "_alignment.json")
        json_path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        n_ok += 1

    return n_ok, n_err


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--textgrid-dir", required=True, type=Path)
    parser.add_argument("--transcript-dir", required=True, type=Path,
                        help="Directory with original processed .txt transcripts.")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    n_ok, n_err = postprocess_directory(
        textgrid_dir=args.textgrid_dir,
        transcript_dir=args.transcript_dir,
        output_dir=args.output_dir,
    )
    print(f"[done] processed {n_ok} TextGrids, {n_err} errors/warnings.")


if __name__ == "__main__":
    main()
