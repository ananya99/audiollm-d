"""Read-only disfluency-token count reports for normalized CHAT transcripts."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from audiollm_d.chat_parser import (
    extract_timed_speaker_utterances,
    read_chat_transcript,
)
from audiollm_d.special_tokens import DISFLUENCY_TOKENS

_TOKEN_COUNT_RE_CACHE: dict[str, re.Pattern[str]] = {}


@dataclass(frozen=True)
class DisfluencyTokenReport:
    """Read-only count report over normalized participant transcript text."""

    paths: list[Path]
    files_scanned: int
    utterances_scanned: int
    tokens: list[str]
    counts: Counter[str]

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _count_token(text: str, token: str) -> int:
    pattern = _TOKEN_COUNT_RE_CACHE.get(token)
    if pattern is None:
        if re.fullmatch(r"\w+", token):
            # Count plain words like "uh", "um", and "xxx" only as standalone tokens.
            pattern = re.compile(rf"(?<![\w'-]){re.escape(token)}(?![\w'-])")
        else:
            pattern = re.compile(re.escape(token))
        _TOKEN_COUNT_RE_CACHE[token] = pattern
    return len(pattern.findall(text))


def count_disfluency_tokens_in_text(
    text: str,
    tokens: Iterable[str] = DISFLUENCY_TOKENS,
) -> Counter[str]:
    """Count final normalized disfluency tokens in *text* without mutating data."""
    return Counter({token: count for token in tokens if (count := _count_token(text, token))})


def _iter_chat_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    return sorted(root.rglob("*.cha"))


def count_disfluency_tokens(
    path: str | Path,
    speaker: str = "PAR",
    tokens: Iterable[str] = DISFLUENCY_TOKENS,
) -> DisfluencyTokenReport:
    """Parse CHAT files read-only and report final normalized disfluency-token counts."""
    token_list = list(tokens)
    counts: Counter[str] = Counter()
    utterances_scanned = 0
    paths = _iter_chat_paths(path)

    for chat_path in paths:
        if speaker == "PAR":
            transcript = read_chat_transcript(chat_path)
            utterances = transcript.utterances
        else:
            utterances = [
                utterance.text
                for utterance in extract_timed_speaker_utterances(chat_path, speaker=speaker)
            ]
        utterances_scanned += len(utterances)
        for utterance in utterances:
            counts.update(count_disfluency_tokens_in_text(utterance, token_list))

    return DisfluencyTokenReport(
        paths=paths,
        files_scanned=len(paths),
        utterances_scanned=utterances_scanned,
        tokens=token_list,
        counts=counts,
    )


def format_disfluency_token_report(report: DisfluencyTokenReport) -> str:
    """Format a disfluency-token count report as a small Markdown table."""
    lines = [
        f"Files scanned: {report.files_scanned}",
        f"Utterances scanned: {report.utterances_scanned}",
        f"Total disfluency tokens: {report.total}",
        "",
        "| Final token | Count |",
        "|---|---:|",
    ]
    for token in report.tokens:
        lines.append(f"| `{token}` | {report.counts.get(token, 0)} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="CHAT file or directory of .cha files to scan.")
    parser.add_argument(
        "--speaker",
        default="PAR",
        help="Speaker tier to scan for the report. Defaults to PAR.",
    )
    args = parser.parse_args()

    report = count_disfluency_tokens(args.path, speaker=args.speaker)
    print(format_disfluency_token_report(report))


if __name__ == "__main__":
    main()
