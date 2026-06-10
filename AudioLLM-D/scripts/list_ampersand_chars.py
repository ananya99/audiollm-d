#!/usr/bin/env python
"""List unique characters that appear after '&' in CHAT transcripts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Capture contiguous non-whitespace after '&' (e.g. &=laughs, &-uh, &+dri).
AMP_TOKEN_RE = re.compile(r"&([^\s]+)")
PAR_UTTERANCE_RE = re.compile(r"^\*PAR:\s*(.*)$")


def scan_ampersand_patterns(transcript_dir: str | Path) -> dict:
    root = Path(transcript_dir)
    if not root.exists():
        raise FileNotFoundError(f"Transcript directory does not exist: {root}")

    files = sorted(path for path in root.rglob("*.cha") if path.is_file())
    token_counter: Counter[str] = Counter()
    char_counter: Counter[str] = Counter()
    first_char_counter: Counter[str] = Counter()

    for path in files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        current_tier = "none"

        for line in text.splitlines():
            par_match = PAR_UTTERANCE_RE.match(line)
            if par_match:
                current_tier = "par"
                line_to_scan = par_match.group(1)
            elif line.startswith("*"):
                current_tier = "other"
                continue
            elif line.startswith("%"):
                current_tier = "other"
                continue
            elif line.startswith("@"):
                current_tier = "none"
                continue
            elif line.startswith((" ", "\t")) and current_tier == "par":
                line_to_scan = line.strip()
            else:
                continue

            for match in AMP_TOKEN_RE.finditer(line_to_scan):
                token = match.group(1)
                if not token:
                    continue
                token_counter[token] += 1
                first_char_counter[token[0]] += 1
                char_counter.update(token)

    return {
        "transcript_dir": str(root),
        "files_scanned": len(files),
        "ampersand_token_occurrences": sum(token_counter.values()),
        "unique_tokens_after_ampersand": sorted(token_counter.keys()),
        "unique_chars_after_ampersand": sorted(char_counter.keys()),
        "unique_first_chars_after_ampersand": sorted(first_char_counter.keys()),
        "token_counts": dict(sorted(token_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "char_counts": dict(sorted(char_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
        "first_char_counts": dict(sorted(first_char_counter.items(), key=lambda kv: (-kv[1], kv[0]))),
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

    results = scan_ampersand_patterns(args.transcript_dir)

    print(f"Scanned files: {results['files_scanned']}")
    print(f"Total '&...' token occurrences: {results['ampersand_token_occurrences']}")
    print(
        "Unique first chars after '&': "
        + (" ".join(results["unique_first_chars_after_ampersand"]) or "(none)")
    )
    print(
        "Unique chars after '&': "
        + (" ".join(results["unique_chars_after_ampersand"]) or "(none)")
    )
    print(
        "Unique tokens after '&': "
        f"{len(results['unique_tokens_after_ampersand'])}"
    )
    print("Top 20 '&...' tokens by frequency:")
    for token, count in list(results["token_counts"].items())[:20]:
        print(f"  &{token}: {count}")

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to: {output_path}")


if __name__ == "__main__":
    main()
