#!/usr/bin/env python
"""List pause-style CHAT tokens and their frequencies from transcript files."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

# Standard CHAT parenthetical pauses: (.), (..), (...), and any longer dot-run variants.
PAREN_PAUSE_RE = re.compile(r"\(\.+\)")

# Plus-prefixed punctuation-only markers often used with pauses/abandonments: +..., +/., +//., +..?, ...
PLUS_PUNCT_TOKEN_RE = re.compile(r"\+[^\s\t]+")
PLUS_PUNCT_ONLY_RE = re.compile(r"^\+[./?,!]+$")


def scan_pause_tokens(transcript_dir: str | Path) -> dict:
    root = Path(transcript_dir)
    if not root.exists():
        raise FileNotFoundError(f"Transcript directory does not exist: {root}")

    files = sorted(path for path in root.rglob("*.cha") if path.is_file())

    pause_counter: Counter[str] = Counter()
    pause_examples: dict[str, list[str]] = defaultdict(list)

    for path in files:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line_tokens: list[str] = []

            line_tokens.extend(PAREN_PAUSE_RE.findall(line))
            for token in PLUS_PUNCT_TOKEN_RE.findall(line):
                if PLUS_PUNCT_ONLY_RE.match(token):
                    line_tokens.append(token)

            for token in line_tokens:
                pause_counter[token] += 1
                if len(pause_examples[token]) < 3:
                    pause_examples[token].append(f"{path.name}: {line.strip()}")

    return {
        "transcript_dir": str(root),
        "files_scanned": len(files),
        "pause_token_occurrences": sum(pause_counter.values()),
        "unique_pause_tokens": sorted(pause_counter.keys()),
        "pause_token_counts": dict(sorted(pause_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "pause_token_examples": dict(pause_examples),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcript-dir",
        required=True,
        help="Directory containing .cha transcripts (searched recursively).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the full results JSON.",
    )
    args = parser.parse_args()

    results = scan_pause_tokens(args.transcript_dir)

    print(f"Scanned files: {results['files_scanned']}")
    print(f"Total pause-token occurrences: {results['pause_token_occurrences']}")
    print(f"Unique pause tokens: {len(results['unique_pause_tokens'])}")
    if results["unique_pause_tokens"]:
        print("Pause token set: " + " ".join(results["unique_pause_tokens"]))
    print("Pause token counts:")
    for token, count in results["pause_token_counts"].items():
        print(f"  {token}: {count}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {output_path}")


if __name__ == "__main__":
    main()
