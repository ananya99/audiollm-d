#!/usr/bin/env python
"""Whisper-large baseline evaluation on AudioLLM-D manifests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import logging as hf_logging
from transformers import pipeline

from audiollm_d.config import load_train_config
from audiollm_d.dataset import load_manifest_dataset
from audiollm_d.metrics import compute_wer
from audiollm_d.special_tokens import SPECIAL_TOKENS


def _configure_transformers_logging() -> None:
    # Whisper emits several non-actionable generation warnings by default.
    # Keep script output focused on evaluation progress/metrics.
    hf_logging.set_verbosity_error()
    logging.getLogger("transformers").setLevel(logging.ERROR)


def _normalize_strip_special_tokens(text: str, special_tokens: list[str]) -> str:
    cleaned = str(text)
    for token in special_tokens:
        cleaned = cleaned.replace(token, " ")
    return " ".join(cleaned.split()).strip()


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
    output_root = Path(config.output_dir).expanduser().resolve() / "baseline_eval" / "whisper_large"
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
    # Keep configured tokens first, then include any package-level defaults.
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


def evaluate_whisper_baseline(
    *,
    config_path: str | Path,
    split: str,
    whisper_model: str,
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
        model=whisper_model,
        device=pipeline_device,
        torch_dtype=dtype,
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

    effective_batch_size = max(batch_size, 1)
    for start in range(0, len(rows), effective_batch_size):
        batch = rows[start : start + effective_batch_size]
        audio_paths = [str(row["audio_path"]) for row in batch]
        outputs = asr(
            audio_paths,
            batch_size=effective_batch_size,
            return_timestamps=False,
            generate_kwargs=generate_kwargs,
        )
        if isinstance(outputs, dict):
            outputs = [outputs]

        for offset, (row, output) in enumerate(zip(batch, outputs, strict=True), start=1):
            index = start + offset
            reference_raw = str(row["text"])
            prediction_raw = str(output.get("text", "")).strip()
            reference_norm = _normalize_strip_special_tokens(reference_raw, special_tokens)
            prediction_norm = _normalize_strip_special_tokens(prediction_raw, special_tokens)
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
            print(f"[{index}/{len(rows)}] {Path(str(row['audio_path'])).name}")

    wer_value = compute_wer(references, predictions)
    metrics: dict[str, object] = {
        "config": str(Path(config_path).expanduser().resolve()),
        "split": split,
        "manifest": manifest_path,
        "whisper_model": whisper_model,
        "language": language,
        "num_samples": len(rows),
        "wer": wer_value,
        "special_tokens_removed_for_normalization": special_tokens,
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
        "--whisper-model",
        default="openai/whisper-large",
        help="Hugging Face Whisper checkpoint name/path.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Whisper language (for example 'en'). Use empty string for auto language detection.",
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
    results: dict[str, dict[str, object]] = {}
    for split_name in splits:
        metrics = evaluate_whisper_baseline(
            config_path=args.config,
            split=split_name,
            whisper_model=args.whisper_model,
            language=language,
            batch_size=max(args.batch_size, 1),
            max_samples=args.max_samples,
            device=args.device,
            predictions_out=args.predictions_out,
            metrics_out=args.metrics_out,
        )
        results[split_name] = metrics
        print(f"[{split_name}] WER: {float(metrics['wer']):.6f}")
        print(f"[{split_name}] Samples: {metrics['num_samples']}")
        print(f"[{split_name}] Manifest: {metrics['manifest']}")


if __name__ == "__main__":
    main()
