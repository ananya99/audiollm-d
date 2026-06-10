#!/usr/bin/env python
"""Run transcription over a manifest and report Word Error Rate (WER)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from audiollm_d.inference import load_for_inference, transcribe_with_loaded_model
from audiollm_d.manifest import read_jsonl
from audiollm_d.metrics import compute_disfluency_token_metrics, compute_wer
from audiollm_d.special_tokens import DISFLUENCY_TOKENS


_DICT_PREFIX_RE = re.compile(r"^\s*\{\s*['\"]?([A-Za-z_ ]+)['\"]?\s*:\s*(.+?)\s*\}\s*$")
_PREFIXED_TEXT_RE = re.compile(r"^\s*([A-Za-z_ ]{1,48})\s*:\s*(.+)$")
_TRANSCRIPT_KEYS = (
    "transcript",
    "transcription",
    "transciript",  # observed misspelling in generations
    "content",
    "audio_content",
    "text",
    "output",
    "prediction",
)


def _unwrap_outer_quotes(text: str) -> str:
    out = text.strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {"'", '"'}:
        return out[1:-1].strip()
    return out


def normalize_prediction_for_wer(text: str) -> str:
    """Remove chat-template artifacts from decoded model output before WER."""
    cleaned = text
    for marker in ("<|im_end|>", "<|im_start|>"):
        cleaned = cleaned.replace(marker, " ")
    cleaned = " ".join(cleaned.split()).strip()

    # If model wraps transcript in paired quotes, unwrap once.
    cleaned = _unwrap_outer_quotes(cleaned)

    # Handle dict-like generations: {'transcription': '...'}.
    dict_match = _DICT_PREFIX_RE.match(cleaned)
    if dict_match:
        key = dict_match.group(1).strip().lower()
        value = dict_match.group(2).strip()
        if any(token in key for token in _TRANSCRIPT_KEYS):
            cleaned = _unwrap_outer_quotes(value)

    # Handle prefixed generations: "TRANSCRIPT: ...", "Transciript: ...", etc.
    prefix_match = _PREFIXED_TEXT_RE.match(cleaned)
    if prefix_match:
        key = prefix_match.group(1).strip().lower()
        value = prefix_match.group(2).strip()
        if any(token in key for token in _TRANSCRIPT_KEYS):
            cleaned = _unwrap_outer_quotes(value)

    return cleaned


def evaluate_manifest_wer(
    *,
    manifest_path: str | Path,
    model_name_or_path: str,
    adapter_path: str | None = None,
    max_new_tokens: int = 256,
    max_samples: int | None = None,
    system_prompt: str | None = None,
    instruction: str | None = None,
) -> tuple[float, list[dict]]:
    rows = read_jsonl(manifest_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")

    loaded = load_for_inference(model_name_or_path, adapter_path)

    references: list[str] = []
    predictions: list[str] = []
    outputs: list[dict] = []

    for idx, row in enumerate(rows):
        audio_path = row["audio_path"]
        ref = str(row["text"]).strip()
        raw_hyp = transcribe_with_loaded_model(
            audio_path=audio_path,
            model=loaded.model,
            processor=loaded.processor,
            max_new_tokens=max_new_tokens,
            system_prompt=system_prompt,
            instruction=instruction,
            disfl_encoder=loaded.disfl_encoder,
            disfl_alpha=loaded.disfl_alpha,
        )
        hyp = normalize_prediction_for_wer(raw_hyp)
        references.append(ref)
        predictions.append(hyp)
        outputs.append(
            {
                "index": idx,
                "audio_path": audio_path,
                "reference": ref,
                "raw_prediction": raw_hyp,
                "prediction_for_wer": hyp,
                "speaker_id": row.get("speaker_id"),
                "clip_id": row.get("clip_id"),
                "source_id": row.get("source_id"),
            }
        )
        print(f"[{idx + 1}/{len(rows)}] {Path(audio_path).name}")

    return compute_wer(references, predictions), outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSONL manifest with audio_path and text.")
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional JSONL path to save per-sample references and predictions.",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional JSON path to save summary metrics.",
    )
    args = parser.parse_args()

    wer_value, outputs = evaluate_manifest_wer(
        manifest_path=args.manifest,
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples,
        system_prompt=args.system_prompt,
        instruction=args.instruction,
    )

    references = [str(row["reference"]) for row in outputs]
    predictions = [str(row["prediction_for_wer"]) for row in outputs]
    disfluency_token_metrics = compute_disfluency_token_metrics(
        references, predictions, DISFLUENCY_TOKENS
    )

    print(f"WER: {wer_value:.6f}")
    print(
        "Disfluency-token P/R/F1 (count): "
        f"{disfluency_token_metrics['disfluency_token_precision']:.6f} / "
        f"{disfluency_token_metrics['disfluency_token_recall']:.6f} / "
        f"{disfluency_token_metrics['disfluency_token_f1']:.6f}"
    )
    print(
        f"Edit-aligned (order-aware) F1: "
        f"{disfluency_token_metrics['disfluency_token_order_f1']:.6f}"
    )

    if args.predictions_out:
        pred_path = Path(args.predictions_out)
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as f:
            for row in outputs:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote predictions to: {pred_path}")

    if args.metrics_out:
        metrics = {
            "manifest": args.manifest,
            "model": args.model,
            "adapter": args.adapter,
            "num_samples": len(outputs),
            "wer": wer_value,
            "disfluency_token_metrics": disfluency_token_metrics,
        }
        metrics_path = Path(args.metrics_out)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
