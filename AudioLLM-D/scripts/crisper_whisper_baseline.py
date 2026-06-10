#!/usr/bin/env python
"""CrisperWhisper baseline evaluation on AudioLLM-D manifests."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from transformers import logging as hf_logging
from transformers import pipeline

from audiollm_d.config import load_train_config
from audiollm_d.dataset import load_manifest_dataset
from audiollm_d.metrics import compute_disfluency_token_metrics, compute_wer
from audiollm_d.special_tokens import SPECIAL_TOKENS

_FILLER_TOKEN_TO_WORD: dict[str, str] = {
    "[uh]": "uh",
    "[um]": "um",
}
_FILLER_METRIC_TOKENS = ["[uh]", "[um]"]


def _configure_transformers_logging() -> None:
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)


def _normalize_for_crisper_wer(text: str, special_tokens: list[str]) -> str:
    """Normalize transcript text for CrisperWhisper WER computation.

    CrisperWhisper explicitly models fillers as lexical words ("uh", "um"), so we
    translate filler markers to words and remove all other disfluency markers.
    """
    cleaned = str(text)

    # Handle bracketed filler tokens from references or generations in any case:
    # [uh], [UH], [Um], etc.
    cleaned = re.sub(r"\[\s*uh\s*\]", " uh ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*um\s*\]", " um ", cleaned, flags=re.IGNORECASE)

    # Remove project special tokens except filler markers already translated above.
    filler_markers = set(_FILLER_TOKEN_TO_WORD.keys())
    for token in special_tokens:
        if token.lower() in filler_markers:
            continue
        cleaned = cleaned.replace(token, " ")

    return " ".join(cleaned.split()).strip()


def _canonicalize_fillers_for_metrics(text: str) -> str:
    """Canonicalize filler words to marker tokens used by metrics helpers."""
    cleaned = str(text).lower()
    cleaned = re.sub(r"\[\s*uh\s*\]", " [uh] ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[\s*um\s*\]", " [um] ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\buh\b", " [uh] ", cleaned)
    cleaned = re.sub(r"\bum\b", " [um] ", cleaned)
    return " ".join(cleaned.split()).strip()


def _build_uh_um_metrics(references: list[str], predictions: list[str]) -> dict[str, object]:
    metrics = compute_disfluency_token_metrics(references, predictions, _FILLER_METRIC_TOKENS)
    per_token_stats = metrics["per_token_stats"]
    order_per_token_stats = metrics["per_token_order_aware_stats"]
    return {
        "bag_of_tokens": {
            "precision": metrics["disfluency_token_precision"],
            "recall": metrics["disfluency_token_recall"],
            "f1": metrics["disfluency_token_f1"],
            "tp": metrics["disfluency_token_tp"],
            "pred_total": metrics["disfluency_token_pred_total"],
            "ref_total": metrics["disfluency_token_ref_total"],
            "per_token": {
                "uh": per_token_stats["[uh]"],
                "um": per_token_stats["[um]"],
            },
        },
        "order_preserved": {
            "precision": metrics["disfluency_token_order_precision"],
            "recall": metrics["disfluency_token_order_recall"],
            "f1": metrics["disfluency_token_order_f1"],
            "tp": metrics["disfluency_token_order_tp"],
            "pred_total": metrics["disfluency_token_order_pred_total"],
            "ref_total": metrics["disfluency_token_order_ref_total"],
            "per_token": {
                "uh": order_per_token_stats["[uh]"],
                "um": order_per_token_stats["[um]"],
            },
        },
    }


def _resolve_manifest_path(config_path: str | Path, split: str) -> str:
    config = load_train_config(config_path)
    if split == "train":
        return config.data.train_manifest
    if not config.data.validation_manifest:
        raise ValueError(
            "The config does not define data.validation_manifest. "
            "Either add it or run with --split train."
        )
    return config.data.validation_manifest


def _resolve_default_output_paths(
    *,
    config_path: str | Path,
    split: str,
    predictions_out: str | None,
    metrics_out: str | None,
) -> tuple[Path, Path]:
    config = load_train_config(config_path)
    output_root = (
        Path(config.output_dir).expanduser().resolve() / "baseline_eval" / "crisper_whisper"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    pred_path = (
        Path(predictions_out).expanduser().resolve()
        if predictions_out
        else output_root / f"{split}_predictions.jsonl"
    )
    metrics_path = (
        Path(metrics_out).expanduser().resolve()
        if metrics_out
        else output_root / f"{split}_metrics.json"
    )
    return pred_path, metrics_path


def _resolve_special_tokens(config_path: str | Path) -> list[str]:
    config = load_train_config(config_path)
    seen: set[str] = set()
    ordered: list[str] = []
    for token in [*config.special_tokens, *SPECIAL_TOKENS]:
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def _resolve_pipeline_device(device_arg: str) -> int:
    if device_arg == "auto":
        return 0 if torch.cuda.is_available() else -1
    if device_arg == "cpu":
        return -1
    if device_arg.startswith("cuda:"):
        return int(device_arg.split(":", maxsplit=1)[1])
    raise ValueError(
        f"Unsupported --device value: {device_arg}. "
        "Use auto, cpu, or cuda:<index>."
    )


def evaluate_crisper_whisper_baseline(
    *,
    config_path: str | Path,
    split: str,
    model_name: str,
    language: str | None,
    batch_size: int,
    max_samples: int | None,
    device: str,
    predictions_out: str | None,
    metrics_out: str | None,
) -> dict[str, object]:
    _configure_transformers_logging()
    manifest_path = _resolve_manifest_path(config_path, split=split)
    special_tokens = _resolve_special_tokens(config_path)
    rows = load_manifest_dataset(manifest_path, max_samples=max_samples)
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")

    pipeline_device = _resolve_pipeline_device(device)
    dtype = torch.float16 if pipeline_device >= 0 else torch.float32
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        chunk_length_s=30,
        torch_dtype=dtype,
        device=pipeline_device,
    )

    generate_kwargs: dict[str, object] = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language

    pred_path, metrics_path = _resolve_default_output_paths(
        config_path=config_path,
        split=split,
        predictions_out=predictions_out,
        metrics_out=metrics_out,
    )

    references: list[str] = []
    predictions: list[str] = []
    prediction_rows: list[dict[str, object]] = []

    for start in range(0, len(rows), max(batch_size, 1)):
        batch = rows[start : start + max(batch_size, 1)]
        audio_paths = [str(row["audio_path"]) for row in batch]
        outputs = asr(
            audio_paths,
            batch_size=max(batch_size, 1),
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        for offset, (row, output) in enumerate(zip(batch, outputs, strict=True)):
            reference_raw = str(row["text"])
            prediction_raw = str(output.get("text", "")).strip()
            reference_norm = _normalize_for_crisper_wer(reference_raw, special_tokens)
            prediction_norm = _normalize_for_crisper_wer(prediction_raw, special_tokens)
            references.append(reference_norm)
            predictions.append(prediction_norm)
            prediction_rows.append(
                {
                    "audio_path": row["audio_path"],
                    "reference_raw": reference_raw,
                    "prediction_raw": prediction_raw,
                    "reference_normalized": reference_norm,
                    "prediction_normalized": prediction_norm,
                }
            )
            print(f"[{start + offset + 1}/{len(rows)}] {Path(str(row['audio_path'])).name}")

    wer_value = compute_wer(references, predictions)
    filler_metric_references = [_canonicalize_fillers_for_metrics(text) for text in references]
    filler_metric_predictions = [_canonicalize_fillers_for_metrics(text) for text in predictions]
    uh_um_metrics = _build_uh_um_metrics(filler_metric_references, filler_metric_predictions)
    metrics: dict[str, object] = {
        "config": str(Path(config_path).expanduser().resolve()),
        "split": split,
        "manifest": manifest_path,
        "model": model_name,
        "language": language,
        "num_samples": len(rows),
        "wer": wer_value,
        "uh_um_metrics": uh_um_metrics,
        "normalization_notes": {
            "mapped_to_words": {"[uh]": "uh", "[um]": "um"},
            "stripped_special_tokens_except_fillers": [
                token for token in special_tokens if token.lower() not in {"[uh]", "[um]"}
            ],
            "uh_um_metrics_canonicalization": "word-boundary uh/um and [uh]/[um] are mapped to [uh]/[um]",
        },
    }

    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", encoding="utf-8") as f:
        for row in prediction_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote predictions to: {pred_path}")

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote metrics to: {metrics_path}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/qwen2_audio_lora.yaml",
        help="Training config YAML used to source manifest paths and special tokens.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "all"),
        default="validation",
        help="Which manifest path from the config to evaluate. Use 'all' for train+validation.",
    )
    parser.add_argument(
        "--model",
        default="nyrahealth/CrisperWhisper",
        help="Hugging Face model checkpoint name/path.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="ASR language (for example 'en'). Use empty string for auto language detection.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device selector: auto, cpu, or cuda:<index>.",
    )
    parser.add_argument(
        "--predictions-out",
        default=None,
        help="Optional JSONL output path for per-sample references/predictions.",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional JSON output path for aggregate WER.",
    )
    args = parser.parse_args()

    language = args.language.strip() or None
    if args.split == "all" and (args.predictions_out or args.metrics_out):
        raise ValueError(
            "When using --split all, omit --predictions-out/--metrics-out so each split "
            "can write to its own default output file."
        )
    splits = ("train", "validation") if args.split == "all" else (args.split,)
    for split_name in splits:
        metrics = evaluate_crisper_whisper_baseline(
            config_path=args.config,
            split=split_name,
            model_name=args.model,
            language=language,
            batch_size=max(args.batch_size, 1),
            max_samples=args.max_samples,
            device=args.device,
            predictions_out=args.predictions_out,
            metrics_out=args.metrics_out,
        )

        print(f"[{split_name}] WER: {float(metrics['wer']):.6f}")
        uh_um_metrics = metrics["uh_um_metrics"]
        bag = uh_um_metrics["bag_of_tokens"]
        order = uh_um_metrics["order_preserved"]
        print(
            f"[{split_name}] uh/um bag-of-tokens P/R/F1: "
            f"{float(bag['precision']):.6f} / {float(bag['recall']):.6f} / {float(bag['f1']):.6f}"
        )
        print(
            f"[{split_name}] uh/um order-preserved P/R/F1: "
            f"{float(order['precision']):.6f} / {float(order['recall']):.6f} / {float(order['f1']):.6f}"
        )
        print(f"[{split_name}] Samples: {metrics['num_samples']}")
        print(f"[{split_name}] Manifest: {metrics['manifest']}")


if __name__ == "__main__":
    main()
