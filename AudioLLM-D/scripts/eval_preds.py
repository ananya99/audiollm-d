#!/usr/bin/env python
"""Compute WER and disfluency token metrics on one/few shot predictions."""

import argparse
import json
import re
from pathlib import Path

from audiollm_d.metrics import compute_disfluency_token_metrics, compute_wer
from audiollm_d.special_tokens import DISFLUENCY_TOKENS, SPECIAL_TOKENS

# ── Normalization —───────────────────────────────────────────────────────────

_DICT_PREFIX_RE   = re.compile(r"^\s*\{\s*['\"]?([A-Za-z_ ]+)['\"]?\s*:\s*(.+?)\s*\}\s*$")
_PREFIXED_TEXT_RE = re.compile(r"^\s*([A-Za-z_ ]{1,48})\s*:\s*(.+)$")
_TRANSCRIPT_KEYS  = (
    "transcript", "transcription", "transciript",
    "content", "audio_content", "text", "output", "prediction",
)


def _unwrap_outer_quotes(text: str) -> str:
    out = text.strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {"'", '"'}:
        return out[1:-1].strip()
    return out


def normalize_prediction_for_wer(text: str) -> str:
    """Remove chat-template artifacts"""
    cleaned = text
    for marker in ("<|im_end|>", "<|im_start|>"):
        cleaned = cleaned.replace(marker, " ")
    cleaned = " ".join(cleaned.split()).strip()
    cleaned = _unwrap_outer_quotes(cleaned)
    dict_match = _DICT_PREFIX_RE.match(cleaned)
    if dict_match:
        key   = dict_match.group(1).strip().lower()
        value = dict_match.group(2).strip()
        if any(t in key for t in _TRANSCRIPT_KEYS):
            cleaned = _unwrap_outer_quotes(value)
    prefix_match = _PREFIXED_TEXT_RE.match(cleaned)
    if prefix_match:
        key   = prefix_match.group(1).strip().lower()
        value = prefix_match.group(2).strip()
        if any(t in key for t in _TRANSCRIPT_KEYS):
            cleaned = _unwrap_outer_quotes(value)
    return cleaned


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="/scratch/yasaman/results/zero_shot_predictions_train.jsonl")
    parser.add_argument("--output", default="/scratch/yasaman/results/zero_shot_metrics_train.json")
    args = parser.parse_args()

    rows = []
    with open(args.predictions) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    print(f"Loaded {len(rows)} predictions")

    valid = [r for r in rows if not r["raw_prediction"].startswith("ERROR")]
    skipped = len(rows) - len(valid)
    if skipped:
        print(f"Skipping {skipped} errored rows")
    print(f"Evaluating {len(valid)} valid predictions")

    if not valid:
        print("No valid predictions."); return

    references  = [str(r["text"]).strip() for r in valid]
    predictions = [normalize_prediction_for_wer(r["raw_prediction"]) for r in valid]

    wer_value = compute_wer(references, predictions)
    disfluency_metrics = compute_disfluency_token_metrics(
        references, predictions, DISFLUENCY_TOKENS
    )

    print(f"\n{'='*60}")
    print(f"WER                          : {wer_value:.6f}  ({wer_value*100:.2f}%)")
    print(f"{'='*60}")

    print(f"\nDisfluency token metrics (all tokens incl. [uh]/[um]):")
    print(f"  P/R/F1         : "
          f"{disfluency_metrics['disfluency_token_precision']:.4f} / "
          f"{disfluency_metrics['disfluency_token_recall']:.4f} / "
          f"{disfluency_metrics['disfluency_token_f1']:.4f}")
    print(f"  Hallucination  : {disfluency_metrics['disfluency_token_hallucination_rate']:.4f}")
    print(f"  Miss rate      : {disfluency_metrics['disfluency_token_miss_rate']:.4f}")
    print(f"  Exact set acc  : {disfluency_metrics['disfluency_token_exact_set_accuracy']:.4f}")
    print(f"  Order-aware F1 : {disfluency_metrics['disfluency_token_order_f1']:.4f}")

    print(f"\nPer-token breakdown (all disfluency tokens):")
    print(f"  {'token':<22} {'P':>8} {'R':>8} {'F1':>8} {'ref':>7} {'pred':>7}")
    print("  " + "-" * 62)
    for token, stats in disfluency_metrics["per_token_stats"].items():
        print(
            f"  {token:<22} "
            f"{stats['precision']:>8.4f} "
            f"{stats['recall']:>8.4f} "
            f"{stats['f1']:>8.4f} "
            f"{stats['ref_total']:>7} "
            f"{stats['pred_total']:>7}"
        )

    metrics = {
        "num_samples":           len(valid),
        "wer":                   wer_value,
        "disfluency_metrics":    disfluency_metrics,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n Saved metrics → {args.output}")
    


if __name__ == "__main__":
    main()