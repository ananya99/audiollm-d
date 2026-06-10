"""Inspect MFA alignment quality for a manifest.

Checks:
  1. Coverage   — how many samples have alignment files.
  2. Durations  — per-token mean/p5/p95 of disfluency token durations.
  3. Labels     — window label class distribution (speech vs each disfluency class).
  4. Boundaries — gaps and overlaps between consecutive words.

Usage:
    python scripts/inspect_mfa_labels.py \
        --manifest /scratch/merged_data_80_20/train/manifest.jsonl \
        --alignment-dir /scratch/mfa/mfa_processed \
        [--max-samples 200]

If the manifest already has an ``alignment_path`` column that column is used
directly.  Otherwise ``--alignment-dir`` is required: the script resolves each
sample's alignment file as ``{alignment_dir}/{clip_id}_alignment.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audiollm_d.disfluency_encoder import build_window_labels, compute_max_windows
from audiollm_d.manifest import read_jsonl
from audiollm_d.special_tokens import SPECIAL_TOKENS


# ── helpers ────────────────────────────────────────────────────────────────────

_CLASS_NAMES = ["speech"] + SPECIAL_TOKENS  # index 0 = speech, 1-8 = disfluency


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _fmt(v: float) -> str:
    return f"{v:.3f}s" if not (v != v) else "  n/a "


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--alignment-dir", default=None,
                        help="Directory containing {clip_id}_alignment.json files.")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else None

    records = read_jsonl(args.manifest)
    if args.max_samples:
        records = records[: args.max_samples]

    n_total = len(records)
    n_with_alignment = 0
    n_missing = 0
    n_unreadable = 0

    # Per disfluency token: list of durations (seconds).
    token_durations: dict[str, list[float]] = defaultdict(list)

    # Window label counts: index = class (0=speech, 1-8=disfluency, -1=ignored).
    window_counts: dict[int, int] = defaultdict(int)

    # Boundary gap/overlap stats.
    gaps: list[float] = []
    overlaps: list[float] = []

    n_windows = compute_max_windows()

    for record in records:
        alignment_path = record.get("alignment_path")
        if not alignment_path and alignment_dir is not None:
            clip_id = record.get("clip_id") or Path(record.get("audio_path", "")).stem
            alignment_path = str(alignment_dir / f"{clip_id}_alignment.json")
        if not alignment_path:
            n_missing += 1
            continue

        try:
            alignment: list[dict] = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            n_unreadable += 1
            continue

        n_with_alignment += 1

        # 1. Token duration stats.
        for tok in SPECIAL_TOKENS:
            for rec in alignment:
                if rec.get("word") == tok:
                    dur = rec["end"] - rec["start"]
                    token_durations[tok].append(dur)

        # 2. Window label distribution.
        labels = build_window_labels(alignment)
        for cls_val in labels.tolist():
            window_counts[int(cls_val)] += 1

        # 3. Boundary gaps and overlaps (consecutive word pairs).
        for prev, curr in zip(alignment[:-1], alignment[1:]):
            delta = curr["start"] - prev["end"]
            if delta > 0.005:      # gap > 5ms
                gaps.append(delta)
            elif delta < -0.005:   # overlap > 5ms
                overlaps.append(-delta)

    # ── report ────────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print(f"MANIFEST: {args.manifest}")
    print(f"Samples examined: {n_total}")
    print("=" * 60)

    # Coverage.
    print("\n── 1. ALIGNMENT COVERAGE ──────────────────────────────────")
    print(f"  With alignment file : {n_with_alignment:>5}  ({100*n_with_alignment/n_total:.1f}%)")
    print(f"  No alignment_path   : {n_missing:>5}  ({100*n_missing/n_total:.1f}%)")
    print(f"  Unreadable file     : {n_unreadable:>5}  ({100*n_unreadable/n_total:.1f}%)")

    if n_with_alignment == 0:
        print("\n  No alignments found — nothing more to check.")
        return

    # Token durations.
    print("\n── 2. DISFLUENCY TOKEN DURATION DISTRIBUTION ──────────────")
    header = f"  {'token':<22}  {'count':>6}  {'p5':>7}  {'mean':>7}  {'p95':>7}  {'max':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for tok in SPECIAL_TOKENS:
        durs = token_durations[tok]
        if not durs:
            print(f"  {tok:<22}  {'0':>6}")
            continue
        mean = sum(durs) / len(durs)
        p5   = _percentile(durs, 5)
        p95  = _percentile(durs, 95)
        mx   = max(durs)
        flag = "  ← check" if p5 < 0.02 or mx > 5.0 else ""
        print(f"  {tok:<22}  {len(durs):>6}  {_fmt(p5)}  {_fmt(mean)}  {_fmt(p95)}  {_fmt(mx)}{flag}")

    # Window label distribution.
    print("\n── 3. WINDOW LABEL DISTRIBUTION ───────────────────────────")
    total_windows = sum(window_counts.values())
    valid_windows = total_windows - window_counts.get(-1, 0)
    disfl_windows = sum(window_counts.get(c, 0) for c in range(1, len(SPECIAL_TOKENS) + 1))
    print(f"  Total windows        : {total_windows:>8,}  (across {n_with_alignment} samples)")
    print(f"  Valid (non-ignored)  : {valid_windows:>8,}  ({100*valid_windows/max(total_windows,1):.1f}%)")
    print(f"  Speech (class 0)     : {window_counts.get(0,0):>8,}  ({100*window_counts.get(0,0)/max(valid_windows,1):.1f}% of valid)")
    print(f"  Disfluency (any)     : {disfl_windows:>8,}  ({100*disfl_windows/max(valid_windows,1):.1f}% of valid)")
    print(f"  Ignored (class -1)   : {window_counts.get(-1,0):>8,}")
    print()
    for i, tok in enumerate(SPECIAL_TOKENS, start=1):
        cnt = window_counts.get(i, 0)
        pct = 100.0 * cnt / max(valid_windows, 1)
        print(f"    class {i}  {tok:<22}  {cnt:>8,}  ({pct:.2f}%)")

    # Boundary quality.
    print("\n── 4. BOUNDARY GAPS / OVERLAPS ────────────────────────────")
    if gaps:
        print(f"  Gaps   (>5ms) : {len(gaps):>6}  mean={_fmt(sum(gaps)/len(gaps))}  p95={_fmt(_percentile(gaps, 95))}  max={_fmt(max(gaps))}")
    else:
        print("  Gaps   (>5ms) : none")
    if overlaps:
        print(f"  Overlaps(>5ms): {len(overlaps):>6}  mean={_fmt(sum(overlaps)/len(overlaps))}  p95={_fmt(_percentile(overlaps, 95))}  max={_fmt(max(overlaps))}")
    else:
        print("  Overlaps(>5ms): none")

    print("\n── INTERPRETATION GUIDE ────────────────────────────────────")
    print("  Token durations:")
    print("    [uh]/[um]         healthy: 0.05–0.50s, bad if p5 < 20ms or max > 2s")
    print("    [short-pause]     healthy: 0.30–1.00s")
    print("    [medium-pause]    healthy: 1.00–2.50s")
    print("    [long-pause]      healthy: > 2.50s")
    print("  Window label rate:")
    print("    Disfluency < 1%   → very sparse labels; hard to learn from")
    print("    Disfluency > 20%  → suspiciously high; check alignment")
    print("  Boundaries:")
    print("    Many overlaps     → MFA failed to segment cleanly")
    print("    Gaps p95 > 500ms  → large silent regions unaccounted for")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
