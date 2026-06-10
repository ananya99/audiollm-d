#!/usr/bin/env python
"""Token-frequency audit for the disfluency tokens (ticket 1.1).

Counts each of the 8 disfluency tokens in a train manifest's ``text`` field and
confirms each maps to a *single* tokenizer id after the special tokens are
registered (un-registered, bracketed tokens fragment into several subwords).

Outputs a Markdown frequency table. Tokens below ``--min-count`` are flagged
"report-only, not expected to learn" and should be excluded from the optimized
metric and noted as a data limitation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from audiollm_d.special_tokens import DISFLUENCY_TOKENS, SPECIAL_TOKENS


def count_tokens_in_manifest(manifest_path: str | Path, tokens: list[str]) -> tuple[Counter, int, int]:
    """Exact-string count of each token in the manifest ``text`` field.

    The disfluency tokens are bracketed (e.g. ``[uh]``) so exact substring counts
    are unambiguous — brackets never occur inside ordinary words.
    """
    counts: Counter[str] = Counter()
    n_rows = 0
    speakers: set[str] = set()
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            speakers.add(str(row.get("source_id") or row.get("speaker_id") or ""))
            text = str(row.get("text", ""))
            for token in tokens:
                c = text.count(token)
                if c:
                    counts[token] += c
    return counts, n_rows, len(speakers)


def check_tokenizer_single_ids(model_id: str, tokens: list[str]) -> dict[str, dict]:
    """Return per-token tokenization before and after registering the special tokens."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = processor.tokenizer

    before = {tok: tokenizer.encode(tok, add_special_tokens=False) for tok in tokens}
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    after = {tok: tokenizer.encode(tok, add_special_tokens=False) for tok in tokens}

    return {
        tok: {
            "before_pieces": len(before[tok]),
            "after_ids": after[tok],
            "single_id": len(after[tok]) == 1,
        }
        for tok in tokens
    }


def build_report(
    *,
    train_counts: Counter,
    train_rows: int,
    train_speakers: int,
    test_counts: Counter | None,
    tokens: list[str],
    min_count: int,
    tokenizer_info: dict[str, dict] | None,
) -> str:
    lines = [
        "# 1.1 Token frequency audit",
        "",
        f"Train manifest: **{train_rows} clips, {train_speakers} speakers**.",
        "",
    ]
    header = "| Token | Train count |"
    sep = "|---|---:|"
    if test_counts is not None:
        header += " Test count |"
        sep += "---:|"
    if tokenizer_info is not None:
        header += " Base pieces | Single ID after register? |"
        sep += "---:|:--:|"
    header += " Flag |"
    sep += ":--|"
    lines += [header, sep]

    for token in sorted(tokens, key=lambda t: train_counts.get(t, 0), reverse=True):
        train_n = train_counts.get(token, 0)
        row = f"| `{token}` | {train_n} |"
        if test_counts is not None:
            row += f" {test_counts.get(token, 0)} |"
        if tokenizer_info is not None:
            info = tokenizer_info.get(token, {})
            row += f" {info.get('before_pieces', '?')} |"
            row += " yes |" if info.get("single_id") else " **NO** |"
        flag = "report-only (<%d)" % min_count if train_n < min_count else "ok"
        row += f" {flag} |"
        lines.append(row)

    flagged = [t for t in tokens if train_counts.get(t, 0) < min_count]
    lines += ["", f"Total disfluency tokens (train): **{sum(train_counts.values())}**.", ""]
    if flagged:
        lines.append(
            f"⚠️ Below ~{min_count} instances (report-only, exclude from optimized metric): "
            + ", ".join(f"`{t}`" for t in flagged)
        )
    else:
        lines.append(f"All tokens have ≥ {min_count} instances — none flagged report-only by frequency.")
    if tokenizer_info is not None:
        not_single = [t for t, i in tokenizer_info.items() if not i["single_id"]]
        if not_single:
            lines.append(
                "⚠️ Not single tokenizer IDs even after registration: "
                + ", ".join(f"`{t}`" for t in not_single)
            )
        else:
            lines.append("All tokens map to a single tokenizer ID once the special tokens are registered.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--no-tokenizer", action="store_true", help="Skip the tokenizer single-ID check.")
    parser.add_argument("--out", default=None, help="Optional path to write the Markdown report.")
    args = parser.parse_args()

    train_counts, train_rows, train_speakers = count_tokens_in_manifest(
        args.train_manifest, DISFLUENCY_TOKENS
    )
    test_counts = None
    if args.test_manifest:
        test_counts, _, _ = count_tokens_in_manifest(args.test_manifest, DISFLUENCY_TOKENS)

    tokenizer_info = None
    if not args.no_tokenizer:
        tokenizer_info = check_tokenizer_single_ids(args.model, DISFLUENCY_TOKENS)

    report = build_report(
        train_counts=train_counts,
        train_rows=train_rows,
        train_speakers=train_speakers,
        test_counts=test_counts,
        tokens=DISFLUENCY_TOKENS,
        min_count=args.min_count,
        tokenizer_info=tokenizer_info,
    )
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report + "\n", encoding="utf-8")
        print(f"\nWrote report to: {out_path}")


if __name__ == "__main__":
    main()
