#!/usr/bin/env python
"""Prepare MFA corpus directory from AudioLLM-D processed transcripts.

For each clip:
  - Strips pause tokens ([short-pause], [medium-pause], [long-pause]) — MFA derives
    these from silence intervals in post-processing.
  - Maps vocalized/non-verbal bracket tokens to underscore-prefixed MFA words
    (_um, _uh, _laugh, _nvb, _unk) that have entries in mfa/custom_dict.txt.
  - Strips CHAT prolongation colons (ba:ll → ball) and sentence punctuation.
  - Writes a .lab file alongside a symlink to the .wav.

Usage:
    python scripts/prepare_mfa_input.py \
        --audio-dir merged/merged_data_processed/audio \
        --transcript-dir merged/merged_data_processed/transcripts \
        --output-dir mfa/corpus
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Pause tokens: stripped entirely (MFA finds silences itself).
_PAUSE_TOKENS = {
    "[short-pause]",
    "[medium-pause]",
    "[long-pause]",
}

# Bracket tokens that survive into MFA input, mapped to underscore-prefixed words.
# The underscore prefix ensures no collision with real English dictionary words.
_BRACKET_TO_MFA: dict[str, str] = {
    "[um]": "_um",
    "[uh]": "_uh",
    "[laughter]": "_laugh",
    "[non-verbal]": "_nvb",
    "[unclear-speech]": "_unk",
}

# Single regex for all bracket tokens (longest-first, non-capturing group).
_BRACKET_RE = re.compile(
    r"\[(?:"
    + "|".join(
        re.escape(tok[1:-1])
        for tok in sorted(_PAUSE_TOKENS | set(_BRACKET_TO_MFA.keys()), key=len, reverse=True)
    )
    + r")\]"
)


def _replace_bracket_token(m: re.Match) -> str:
    tok = m.group(0)
    if tok in _PAUSE_TOKENS:
        return ""
    return _BRACKET_TO_MFA.get(tok, "")


def _clean_for_mfa(text: str) -> str:
    """Convert a processed transcript line to MFA-ready word sequence."""
    # Replace bracket tokens.
    text = _BRACKET_RE.sub(_replace_bracket_token, text)
    # Strip CHAT prolongation colons within words (ba:ll → ball, she: → she).
    text = re.sub(r":+", "", text)
    # Strip sentence-final punctuation that MFA can't look up.
    text = re.sub(r"[,.\?!;]", " ", text)
    # Collapse whitespace.
    text = " ".join(text.split())
    return text


def prepare_corpus(
    audio_dir: Path,
    transcript_dir: Path,
    output_dir: Path,
    *,
    audio_ext: str = ".wav",
    dry_run: bool = False,
) -> tuple[int, int]:
    """Create output_dir with .lab files and audio symlinks.

    Returns (n_written, n_skipped).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_files = sorted(transcript_dir.glob("*.txt"))

    n_written = 0
    n_skipped = 0
    skipped_names: list[str] = []

    for txt_path in txt_files:
        stem = txt_path.stem
        audio_path = audio_dir / (stem + audio_ext)
        if not audio_path.exists():
            print(f"[skip] No audio for {stem}", file=sys.stderr)
            n_skipped += 1
            skipped_names.append(stem)
            continue

        text = txt_path.read_text(encoding="utf-8").strip()
        cleaned = _clean_for_mfa(text)

        if not cleaned:
            print(f"[skip] Empty transcript after cleaning: {stem}", file=sys.stderr)
            n_skipped += 1
            skipped_names.append(stem)
            continue

        if dry_run:
            print(f"{stem}: {cleaned}")
            n_written += 1
            continue

        lab_path = output_dir / (stem + ".lab")
        lab_path.write_text(cleaned + "\n", encoding="utf-8")

        wav_link = output_dir / (stem + audio_ext)
        if not wav_link.exists():
            wav_link.symlink_to(audio_path.resolve())

        n_written += 1

    return n_written, n_skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--transcript-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--audio-ext",
        default=".wav",
        help="Audio file extension (default: .wav)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cleaned transcripts without writing files.",
    )
    args = parser.parse_args()

    n_written, n_skipped = prepare_corpus(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        output_dir=args.output_dir,
        audio_ext=args.audio_ext,
        dry_run=args.dry_run,
    )

    action = "would write" if args.dry_run else "wrote"
    print(f"[done] {action} {n_written} corpus entries, skipped {n_skipped}.")


if __name__ == "__main__":
    main()
