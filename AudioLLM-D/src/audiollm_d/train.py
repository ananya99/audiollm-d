"""Fine-tune Qwen2-Audio-Instruct for dementia speech transcription with LoRA."""

from __future__ import annotations

import argparse
import contextlib
import os
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from audiollm_d.config import TrainConfig, load_train_config
from audiollm_d.ctc_head import AuxiliaryCtcHead, locate_audio_tower
from audiollm_d.ctc_labels import CtcVocab
from audiollm_d.dataset import (
    ENCODER_NUM_CLASSES,
    NOISE_AUGMENTATION_FLAG_COLUMN,
    Qwen2AudioTranscriptionCollator,
    load_manifest_dataset,
)
from audiollm_d.disfluency_encoder import DisfluencyEncoder, WINDOW_NUM_CLASSES
from audiollm_d.inference import transcribe_batch_with_loaded_model
from audiollm_d.metrics import compute_disfluency_token_metrics, compute_wer
from audiollm_d.model_setup import (
    find_audio_encoder as _find_audio_encoder,
    find_whisper_encoder as _find_whisper_encoder,
    get_encoder_hidden_dim as _get_encoder_hidden_dim,
    replace_qwen_audio_encoder_from_whisper as _replace_qwen_audio_encoder_from_whisper,
)
from audiollm_d.special_tokens import DISFLUENCY_TOKENS, register_tokens_with_init
# os.environ["WANDB_MODE"] = "disabled"

add_disfluency_tokens = register_tokens_with_init
_DICT_PREFIX_RE = re.compile(r"^\s*\{\s*['\"]?([A-Za-z_ ]+)['\"]?\s*:\s*(.+?)\s*\}\s*$")
_PREFIXED_TEXT_RE = re.compile(r"^\s*([A-Za-z_ ]{1,48})\s*:\s*(.+)$")
_TRANSCRIPT_KEYS = (
    "transcript",
    "transcription",
    "transciript",
    "content",
    "audio_content",
    "text",
    "output",
    "prediction",
)


class _TeeTextIO:
    """Mirror writes to terminal and a log file."""

    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror = mirror

    def write(self, data: str) -> int:
        self._primary.write(data)
        self._mirror.write(data)
        return len(data)

    def flush(self) -> None:
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._primary.fileno()

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


@contextlib.contextmanager
def _tee_stdout_stderr(log_path: Path):
    """Duplicate stdout/stderr into a persistent log file."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_f:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = _TeeTextIO(original_stdout, log_f)
        sys.stderr = _TeeTextIO(original_stderr, log_f)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def _bucket_parameter(name: str) -> str:
    lname = name.lower()
    if "lora_" in lname:
        return "lora_adapters"
    if "multi_modal_projector" in lname:
        return "multi_modal_projector"
    if "embed_tokens" in lname:
        return "embed_tokens"
    if "lm_head" in lname:
        return "lm_head"
    if "language_model" in lname:
        return "language_model_backbone"
    if "audio" in lname or "encoder" in lname:
        return "audio_stack"
    return "other"


def _optimizer_stack_for_parameter(name: str) -> str:
    """Classify trainable PEFT parameters into stack-specific LR groups."""

    lname = name.lower()
    if "audio_tower" in lname or "audio_encoder" in lname:
        return "audio_encoder"
    if "language_model" in lname:
        return "llm"
    return "other"


def _report_trainable_parameters(
    model,
    *,
    old_vocab_size: int,
    current_tokenizer_vocab_size: int,
    disfluency_token_ids: list[int],
) -> None:
    total_params = 0
    trainable_params = 0
    bucket_counts: dict[str, int] = defaultdict(int)
    trainable_param_names: list[str] = []

    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
            bucket_counts[_bucket_parameter(name)] += count
            trainable_param_names.append(name)

    newly_added_token_ids = [tid for tid in disfluency_token_ids if tid >= old_vocab_size]
    masked_names = getattr(model, "_old_vocab_masked_parameters", set())
    effective_bucket_counts = dict(bucket_counts)
    if newly_added_token_ids:
        new_rows = max(current_tokenizer_vocab_size - old_vocab_size, 0)
        embed_weight = model.get_input_embeddings().weight
        if "embed_tokens" in effective_bucket_counts:
            effective_bucket_counts["embed_tokens"] = new_rows * embed_weight.shape[1]

        if "lm_head" in effective_bucket_counts:
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None and output_embeddings.weight.data_ptr() == embed_weight.data_ptr():
                # Weight tying means there are no additional lm_head parameters beyond embed_tokens.
                effective_bucket_counts["lm_head"] = 0
            elif output_embeddings is not None:
                effective_bucket_counts["lm_head"] = new_rows * output_embeddings.weight.shape[1]

    for bucket, count in sorted(effective_bucket_counts.items(), key=lambda item: item[1], reverse=True):
        print(f"[param-check]   - {bucket}: {count:,}")

    print(
        f"[param-check] Vocab old={old_vocab_size:,}, new_tokens={len(newly_added_token_ids):,}, "
        f"current_tokenizer_vocab={current_tokenizer_vocab_size:,}, "
        f"current_embedding_vocab={model.get_input_embeddings().weight.shape[0]:,}"
    )
    if newly_added_token_ids:
        effective_trainable = sum(effective_bucket_counts.values())
        eff_pct = (100.0 * effective_trainable / total_params) if total_params else 0.0
        print(
            f"[param-check] Effective trainable (actual updates, row-masked): {effective_trainable:,} / "
            f"{total_params:,} ({eff_pct:.4f}%)"
        )
    if "embed_tokens" in masked_names:
        print("[param-check] embed_tokens old rows frozen via gradient mask.")
    if "lm_head" in masked_names:
        print("[param-check] lm_head old rows frozen via gradient mask.")


def _extract_layer_index(module_name: str, marker: str) -> int | None:
    match = re.search(rf"{re.escape(marker)}\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def _extract_any_stack_layer_index(module_name: str) -> int | None:
    match = re.search(r"(?:layers|blocks)\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


def _unwrap_outer_quotes(text: str) -> str:
    out = text.strip()
    if len(out) >= 2 and out[0] == out[-1] and out[0] in {"'", '"'}:
        return out[1:-1].strip()
    return out


def _normalize_prediction_for_wer(text: str) -> str:
    cleaned = text
    for marker in ("<|im_end|>", "<|im_start|>"):
        cleaned = cleaned.replace(marker, " ")
    cleaned = " ".join(cleaned.split()).strip()
    cleaned = _unwrap_outer_quotes(cleaned)

    dict_match = _DICT_PREFIX_RE.match(cleaned)
    if dict_match:
        key = dict_match.group(1).strip().lower()
        value = dict_match.group(2).strip()
        if any(token in key for token in _TRANSCRIPT_KEYS):
            cleaned = _unwrap_outer_quotes(value)

    prefix_match = _PREFIXED_TEXT_RE.match(cleaned)
    if prefix_match:
        key = prefix_match.group(1).strip().lower()
        value = prefix_match.group(2).strip()
        if any(token in key for token in _TRANSCRIPT_KEYS):
            cleaned = _unwrap_outer_quotes(value)
    return cleaned


def _evaluate_model_on_records(
    *,
    label: str,
    model,
    processor,
    records: list[dict[str, Any]],
    manifest_name: str,
    audio_column: str,
    text_column: str,
    max_new_tokens: int,
    max_samples: int | None,
    eval_generation_batch_size: int,
    eval_generation_do_sample: bool,
    eval_generation_temperature: float,
) -> dict[str, Any]:
    eval_rows = records[:max_samples] if max_samples is not None else records
    if not eval_rows:
        raise ValueError(f"[eval:{label}] No rows available for evaluation.")

    print(f"[eval:{label}] Evaluating {len(eval_rows)} rows from {manifest_name}.")
    model_was_training = model.training
    model.eval()
    references: list[str] = []
    predictions: list[str] = []
    prediction_rows: list[dict[str, Any]] = []
    batch_size = max(int(eval_generation_batch_size), 1)
    try:
        for start_idx in range(0, len(eval_rows), batch_size):
            batch_rows = eval_rows[start_idx : start_idx + batch_size]
            audio_paths = [row[audio_column] for row in batch_rows]
            batch_hyps = transcribe_batch_with_loaded_model(
                audio_paths=audio_paths,
                model=model,
                processor=processor,
                max_new_tokens=max_new_tokens,
                do_sample=eval_generation_do_sample,
                temperature=eval_generation_temperature,
            )
            for offset, (row, raw_hyp) in enumerate(zip(batch_rows, batch_hyps, strict=True)):
                audio_path = row[audio_column]
                reference = str(row[text_column]).strip()
                hyp = _normalize_prediction_for_wer(raw_hyp)
                references.append(reference)
                predictions.append(hyp)
                prediction_rows.append(
                    {
                        "audio_path": audio_path,
                        "reference": reference,
                        "prediction": hyp,
                        "speaker_id": row.get("speaker_id") or row.get("source_id"),
                        "clip_id": row.get("clip_id"),
                    }
                )
                print(
                    f"[eval:{label}] [{start_idx + offset + 1}/{len(eval_rows)}] {Path(audio_path).name}"
                )
    finally:
        if model_was_training:
            model.train()

    wer_value = compute_wer(references, predictions)
    disfluency_token_metrics = compute_disfluency_token_metrics(references, predictions, DISFLUENCY_TOKENS)
    print(f"[eval:{label}] WER: {wer_value:.6f}")
    print(
        f"[eval:{label}] Disfluency-token P/R/F1: "
        f"{disfluency_token_metrics['disfluency_token_precision']:.6f} / "
        f"{disfluency_token_metrics['disfluency_token_recall']:.6f} / "
        f"{disfluency_token_metrics['disfluency_token_f1']:.6f}"
    )
    return {
        "num_samples": len(eval_rows),
        "wer": wer_value,
        "disfluency_token_metrics": disfluency_token_metrics,
        "predictions": prediction_rows,
    }


def _export_loss_curves(trainer, *, output_dir: str | Path) -> None:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    history = list(getattr(trainer.state, "log_history", []) or [])
    grad_accum_steps = max(int(getattr(trainer.args, "gradient_accumulation_steps", 1) or 1), 1)

    train_points: list[dict[str, float]] = []
    train_points_raw: list[dict[str, float]] = []
    eval_points: list[dict[str, float]] = []
    for row in history:
        if "step" not in row:
            continue
        step = float(row["step"])
        if "loss" in row and "eval_loss" not in row:
            raw_loss = float(row["loss"])
            train_points_raw.append({"step": step, "loss": raw_loss})
            train_points.append({"step": step, "loss": raw_loss / grad_accum_steps})
        if "eval_loss" in row:
            eval_points.append({"step": step, "loss": float(row["eval_loss"])})

    payload = {
        "metadata": {
            "gradient_accumulation_steps": grad_accum_steps,
            "train_loss_is_normalized_by_grad_accumulation": True,
        },
        "train_loss": train_points,
        "train_loss_raw": train_points_raw,
        "eval_loss": eval_points,
    }
    loss_json = output_root / "loss_curves.json"
    loss_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[loss-plot] Wrote loss data to: {loss_json}")

    if not train_points and not eval_points:
        print("[loss-plot] No loss points found in trainer.state.log_history; skipping plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[loss-plot] matplotlib not installed; skipping PNG plot generation.")
        return

    plt.figure(figsize=(9, 5))
    if train_points:
        plt.plot([p["step"] for p in train_points], [p["loss"] for p in train_points], label="train_loss")
    if eval_points:
        plt.plot([p["step"] for p in eval_points], [p["loss"] for p in eval_points], label="eval_loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Train vs Eval Loss (train normalized by grad accumulation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = output_root / "loss_curves.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()
    print(f"[loss-plot] Wrote loss plot to: {plot_path}")


def _run_lora_init_sanity_check(
    *,
    config: TrainConfig,
    model,
    processor,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]] | None,
    train_manifest_name: str,
    test_manifest_name: str,
) -> None:
    requested_split = config.lora_init_sanity_check_split.strip().lower()
    if requested_split not in {"train", "test"}:
        raise ValueError(
            "lora_init_sanity_check_split must be either 'train' or 'test'. "
            f"Got: {config.lora_init_sanity_check_split!r}"
        )

    if requested_split == "test" and eval_records is not None:
        split_name = "test"
        records = eval_records
        manifest_name = test_manifest_name
    else:
        split_name = "train"
        records = train_records
        manifest_name = train_manifest_name
        if requested_split == "test":
            print("[sanity:lora-init] No validation/test split found; falling back to train split.")

    print(f"[sanity:lora-init] Running base-vs-LoRA check on '{split_name}' split.")
    with_lora = _evaluate_model_on_records(
        label=f"sanity_with_untrained_lora_{split_name}",
        model=model,
        processor=processor,
        records=records,
        manifest_name=manifest_name,
        audio_column=config.data.audio_column,
        text_column=config.data.text_column,
        max_new_tokens=config.lora_init_sanity_check_max_new_tokens,
        max_samples=config.lora_init_sanity_check_max_samples,
        eval_generation_batch_size=config.eval_generation_batch_size,
        eval_generation_do_sample=config.eval_generation_do_sample,
        eval_generation_temperature=config.eval_generation_temperature,
    )

    if not hasattr(model, "disable_adapter"):
        print(
            "[sanity:lora-init] Model does not expose disable_adapter(); "
            "skipping base-model comparison."
        )
        return

    with model.disable_adapter():
        base_no_lora = _evaluate_model_on_records(
            label=f"sanity_base_model_{split_name}",
            model=model,
            processor=processor,
            records=records,
            manifest_name=manifest_name,
            audio_column=config.data.audio_column,
            text_column=config.data.text_column,
            max_new_tokens=config.lora_init_sanity_check_max_new_tokens,
            max_samples=config.lora_init_sanity_check_max_samples,
            eval_generation_batch_size=config.eval_generation_batch_size,
            eval_generation_do_sample=config.eval_generation_do_sample,
            eval_generation_temperature=config.eval_generation_temperature,
        )

    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with_lora_predictions = with_lora.pop("predictions", None)
    base_predictions = base_no_lora.pop("predictions", None)

    if with_lora_predictions:
        with_lora_path = output_root / f"sanity_with_untrained_lora_{split_name}_predictions.jsonl"
        with with_lora_path.open("w", encoding="utf-8") as f:
            for row in with_lora_predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[sanity:lora-init] Wrote predictions to: {with_lora_path}")

    if base_predictions:
        base_path = output_root / f"sanity_base_model_{split_name}_predictions.jsonl"
        with base_path.open("w", encoding="utf-8") as f:
            for row in base_predictions:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[sanity:lora-init] Wrote predictions to: {base_path}")

    summary = {
        "split": split_name,
        "manifest": manifest_name,
        "num_samples": with_lora.get("num_samples"),
        "base_model": base_no_lora,
        "with_untrained_lora": with_lora,
        "delta_with_lora_minus_base": {
            "wer": with_lora["wer"] - base_no_lora["wer"],
            "disfluency_token_precision": with_lora["disfluency_token_metrics"]["disfluency_token_precision"]
            - base_no_lora["disfluency_token_metrics"]["disfluency_token_precision"],
            "disfluency_token_recall": with_lora["disfluency_token_metrics"]["disfluency_token_recall"]
            - base_no_lora["disfluency_token_metrics"]["disfluency_token_recall"],
            "disfluency_token_f1": with_lora["disfluency_token_metrics"]["disfluency_token_f1"]
            - base_no_lora["disfluency_token_metrics"]["disfluency_token_f1"],
        },
    }
    summary_path = output_root / "lora_init_sanity_check.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sanity:lora-init] Wrote summary to: {summary_path}")


def _resolve_lora_target_modules(
    model,
    *,
    requested_suffixes: list[str],
    audio_encoder_top_half_only: bool,
    audio_encoder_layers: list[int] | None = None,
    audio_encoder_target_modules: list[str] | None = None,
) -> list[str]:
    # audio_encoder_layers (explicit list) overrides audio_encoder_top_half_only.
    # audio_encoder_target_modules (e.g. ["q_proj", "v_proj"]) further restricts
    # which suffixes get LoRA on the encoder, independent of the LM target list.
    requested = set(requested_suffixes)
    encoder_suffix_filter = (
        set(audio_encoder_target_modules) if audio_encoder_target_modules else None
    )
    print(f"[lora-check] Requested target suffixes: {sorted(requested)}")
    if encoder_suffix_filter is not None:
        print(
            f"[lora-check] Encoder suffix filter: {sorted(encoder_suffix_filter)} "
            f"(overrides target_modules for audio_tower only)"
        )
    if audio_encoder_layers is not None:
        print(
            f"[lora-check] Explicit audio_encoder_layers: {sorted(audio_encoder_layers)} "
            f"(overrides audio_encoder_top_half_only)"
        )
    resolved: list[str] = []
    audio_resolved: list[str] = []
    llm_attn_resolved: list[str] = []
    llm_mlp_resolved: list[str] = []
    audio_candidates: list[str] = []
    audio_layer_ids: set[int] = set()

    for module_name, _ in model.named_modules():
        suffix = module_name.rsplit(".", 1)[-1]
        if suffix not in requested:
            continue

        if "encoder.layers." in module_name and ".self_attn." in module_name:
            if suffix not in {"q_proj", "k_proj", "v_proj", "out_proj"}:
                continue
            layer_idx = _extract_layer_index(module_name, "encoder.layers")
            if layer_idx is None:
                continue
            # Defer to audio_candidates so the top_audio_ids layer filter (computed after this
            # loop) is applied uniformly in the second pass below.
            audio_candidates.append(module_name)
            audio_layer_ids.add(layer_idx)
            continue

        llm_like_layer = (
            "language_model.model.layers." in module_name
            or "language_model.layers." in module_name
            or (
                "model.layers." in module_name
                and "encoder.layers." not in module_name
                and "audio" not in module_name.lower()
            )
        )
        if ".self_attn." in module_name and suffix in {"q_proj", "k_proj", "v_proj", "out_proj"} and not llm_like_layer:
            audio_candidates.append(module_name)
            if (idx := _extract_any_stack_layer_index(module_name)) is not None:
                audio_layer_ids.add(idx)
            continue

        if llm_like_layer:
            if ".self_attn." in module_name and suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
                resolved.append(module_name)
                llm_attn_resolved.append(module_name)
                continue
            if ".mlp." in module_name and suffix in {"gate_proj", "up_proj", "down_proj"}:
                resolved.append(module_name)
                llm_mlp_resolved.append(module_name)

    sorted_audio_layer_ids = sorted(audio_layer_ids)
    if audio_encoder_layers is not None:
        # Explicit list wins. Warn about any requested layers that don't exist.
        requested_layer_set = set(audio_encoder_layers)
        missing = requested_layer_set - set(sorted_audio_layer_ids)
        if missing:
            print(
                f"[lora-check] WARNING: audio_encoder_layers requested {sorted(missing)} "
                f"but encoder only has layers {sorted_audio_layer_ids}; those will be silently dropped."
            )
        top_audio_ids = requested_layer_set & set(sorted_audio_layer_ids)
        filter_active = True
    elif audio_encoder_top_half_only and sorted_audio_layer_ids:
        midpoint = len(sorted_audio_layer_ids) // 2
        top_audio_ids = set(sorted_audio_layer_ids[midpoint:])
        filter_active = True
    else:
        top_audio_ids = set(sorted_audio_layer_ids)
        filter_active = False

    print(f"[lora-check] Audio encoder layer ids discovered: {sorted_audio_layer_ids}")
    if filter_active:
        print(f"[lora-check] Audio encoder layers selected: {sorted(top_audio_ids)}")
    else:
        print("[lora-check] Audio encoder layer filter disabled; all layers eligible.")

    for module_name in audio_candidates:
        layer_idx = _extract_any_stack_layer_index(module_name)
        suffix = module_name.rsplit(".", 1)[-1]
        # Optional per-stack suffix filter for the encoder.
        if encoder_suffix_filter is not None and suffix not in encoder_suffix_filter:
            continue
        if not filter_active or layer_idx is None or layer_idx in top_audio_ids:
            resolved.append(module_name)
            audio_resolved.append(module_name)

    deduped = sorted(set(resolved))
    if not deduped:
        print(
            "[lora-check] WARNING: dynamic LoRA target resolution found no exact module paths; "
            "falling back to suffix-based matching from config.lora.target_modules."
        )
        print(f"[lora-check] Fallback target suffixes used by PEFT: {requested_suffixes}")
        return requested_suffixes

    print(f"[lora-check] Resolved exact module paths: {len(deduped)}")
    print(
        f"[lora-check]   audio encoder self-attn: {len(set(audio_resolved))}, "
        f"llm self-attn: {len(set(llm_attn_resolved))}, "
        f"llm mlp: {len(set(llm_mlp_resolved))}"
    )
    sample_count = 12
    for module_name in deduped[:sample_count]:
        print(f"[lora-check]   module: {module_name}")
    if len(deduped) > sample_count:
        print(f"[lora-check]   ... and {len(deduped) - sample_count} more modules")
    return deduped


def _register_old_vocab_gradient_masks(model, *, old_vocab_size: int) -> None:
    def _make_hook():
        def hook(grad):
            if grad is None:
                return grad
            grad = grad.clone()
            grad[:old_vocab_size] = 0
            return grad

        return hook

    masked_names: set[str] = set()
    input_weight = model.get_input_embeddings().weight
    input_weight.register_hook(_make_hook())
    masked_names.add("embed_tokens")

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None:
        output_weight = output_embeddings.weight
        if output_weight.data_ptr() != input_weight.data_ptr():
            output_weight.register_hook(_make_hook())
        masked_names.add("lm_head")

    setattr(model, "_old_vocab_masked_parameters", masked_names)


def _count_target_layer_parameters(model, target_module_names: list[str]) -> int:
    modules = dict(model.named_modules())
    total = 0
    for name in target_module_names:
        module = modules.get(name)
        if module is None:
            continue
        total += sum(param.numel() for param in module.parameters(recurse=False))
    return total


def _count_lora_adapter_parameters(model, target_module_names: list[str]) -> int:
    target_set = set(target_module_names)
    total = 0
    for module_name, module in model.named_modules():
        normalized = module_name.removeprefix("base_model.model.")
        if normalized not in target_set:
            continue
        if hasattr(module, "lora_A"):
            for adapter_layer in module.lora_A.values():
                total += sum(param.numel() for param in adapter_layer.parameters())
        if hasattr(module, "lora_B"):
            for adapter_layer in module.lora_B.values():
                total += sum(param.numel() for param in adapter_layer.parameters())
    return total


def _split_lora_targets_by_stack(target_module_names: list[str]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"audio": [], "llm": [], "other": []}
    for module_name in target_module_names:
        suffix = module_name.rsplit(".", 1)[-1]
        llm_like_layer = (
            "language_model.model.layers." in module_name
            or "language_model.layers." in module_name
            or (
                "model.layers." in module_name
                and "encoder.layers." not in module_name
                and "audio" not in module_name.lower()
            )
        )
        if "encoder.layers." in module_name and ".self_attn." in module_name and suffix in {"q_proj", "k_proj", "v_proj", "out_proj"}:
            buckets["audio"].append(module_name)
            continue

        if ".self_attn." in module_name and suffix in {"q_proj", "k_proj", "v_proj", "out_proj"} and not llm_like_layer:
            buckets["audio"].append(module_name)
            continue

        if llm_like_layer and (
            (".self_attn." in module_name and suffix in {"q_proj", "k_proj", "v_proj", "o_proj"})
            or (".mlp." in module_name and suffix in {"gate_proj", "up_proj", "down_proj"})
        ):
            buckets["llm"].append(module_name)
            continue

        buckets["other"].append(module_name)
    return buckets


def _filter_lora_target_modules(
    resolved_target_modules: list[str],
    *,
    enable_audio_lora: bool,
    enable_llm_lora: bool = True,
) -> list[str]:
    if enable_audio_lora and enable_llm_lora:
        return resolved_target_modules

    buckets = _split_lora_targets_by_stack(resolved_target_modules)
    keep: set[str] = set()
    if enable_audio_lora:
        keep.update(buckets["audio"])
    if enable_llm_lora:
        keep.update(buckets["llm"])
    if not enable_audio_lora and not enable_llm_lora:
        return []
    return [module_name for module_name in resolved_target_modules if module_name in keep]


def _build_modules_to_save(
    requested_modules: list[str],
    *,
    train_multi_modal_projector: bool,
) -> list[str]:
    modules = list(requested_modules)
    if not train_multi_modal_projector:
        modules = [name for name in modules if name != "multi_modal_projector"]
    return modules


def _find_silence_head(model):
    """Return the silence_head Linear module after PEFT wrapping."""
    # PEFT wraps modules_to_save under base_model.model.<name> or similar.
    for name, module in model.named_modules():
        if name.endswith("silence_head") and hasattr(module, "weight"):
            return module
    return None


def _find_encoder_layer(model, layer_idx: int):
    """Return a specific transformer layer of the Whisper audio encoder.

    Tries audio_tower.encoder.layers[layer_idx] first (Qwen2-Audio layout),
    then audio_tower.layers[layer_idx] as fallback.  Returns None if not found.
    """
    audio_tower = _find_audio_encoder(model)
    if audio_tower is None:
        return None
    for layers_obj in (
        getattr(audio_tower, "encoder", None),
        audio_tower,
    ):
        if layers_obj is None:
            continue
        layers = getattr(layers_obj, "layers", None)
        if layers is not None and layer_idx < len(layers):
            return layers[layer_idx]
    return None


def _get_encoder_hidden_dim(model) -> int:
    """Return the audio encoder hidden dimension (d_model) from config."""
    for obj in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if obj is None:
            continue
        audio_cfg = getattr(getattr(obj, "config", None), "audio_config", None)
        if audio_cfg is not None:
            for attr in ("d_model", "hidden_size", "encoder_hidden_size"):
                dim = getattr(audio_cfg, attr, None)
                if dim is not None:
                    return int(dim)
    return 1280  # Whisper-large-v2 fallback for Qwen2-Audio-7B


def _find_whisper_encoder(model):
    """Return a Whisper encoder module from a seq2seq ASR model object."""
    candidate = model
    for _ in range(5):
        inner = getattr(candidate, "model", None)
        if inner is not None and hasattr(inner, "encoder"):
            return inner.encoder
        if hasattr(candidate, "encoder"):
            return candidate.encoder
        advanced = False
        for attr in ("base_model", "model"):
            nxt = getattr(candidate, attr, None)
            if nxt is not None and nxt is not candidate:
                candidate = nxt
                advanced = True
                break
        if not advanced:
            break
    return None


def _assert_audio_encoder_compatibility(*, target_audio_cfg, source_cfg, source_model_name: str) -> None:
    """Fail early when source Whisper encoder shape does not match Qwen audio tower."""
    required_equal = (
        ("d_model", getattr(target_audio_cfg, "d_model", None), getattr(source_cfg, "d_model", None)),
        ("encoder_layers", getattr(target_audio_cfg, "encoder_layers", None), getattr(source_cfg, "encoder_layers", None)),
        ("encoder_attention_heads", getattr(target_audio_cfg, "encoder_attention_heads", None), getattr(source_cfg, "encoder_attention_heads", None)),
        ("num_mel_bins", getattr(target_audio_cfg, "num_mel_bins", None), getattr(source_cfg, "num_mel_bins", None)),
        ("max_source_positions", getattr(target_audio_cfg, "max_source_positions", None), getattr(source_cfg, "max_source_positions", None)),
    )
    mismatches = [
        f"{name}: target={target_val}, source={source_val}"
        for name, target_val, source_val in required_equal
        if target_val is not None and source_val is not None and int(target_val) != int(source_val)
    ]
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Audio encoder replacement aborted due to incompatible Whisper configs "
            f"for source model {source_model_name}: {mismatch_text}"
        )


def _replace_qwen_audio_encoder_from_whisper(model, *, source_model_name: str, use_bf16: bool) -> None:
    """Load Whisper encoder weights from source model into Qwen2-Audio audio_tower."""
    import torch
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

    print(f"[audio-init] Loading source Whisper model: {source_model_name}")
    source_cfg = AutoConfig.from_pretrained(source_model_name, trust_remote_code=True)
    target_audio_cfg = getattr(model.config, "audio_config", None)
    if target_audio_cfg is None:
        raise ValueError("Target model has no audio_config; expected Qwen2-Audio architecture.")
    _assert_audio_encoder_compatibility(
        target_audio_cfg=target_audio_cfg,
        source_cfg=source_cfg,
        source_model_name=source_model_name,
    )

    source_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        source_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    )
    source_encoder = _find_whisper_encoder(source_model)
    target_encoder = _find_audio_encoder(model)
    if source_encoder is None:
        raise ValueError(
            f"Could not find Whisper encoder module in source model: {source_model_name}"
        )
    if target_encoder is None:
        raise ValueError("Could not find Qwen2-Audio audio_tower in target model.")

    load_result = target_encoder.load_state_dict(source_encoder.state_dict(), strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "Audio encoder replacement produced key mismatch. "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    print("[audio-init] Replaced Qwen2-Audio audio_tower with source Whisper encoder weights.")

    # Free source model memory before wrapping in PEFT.
    del source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_param_group_optimizer(model, lrs, weight_decay: float):
    """Build a 4-group AdamW (encoder/projector/LM/new_tokens) so each component
    gets its own LR. Used when TrainConfig.param_group_lrs is set."""

    import torch

    buckets: dict[str, list] = {"enc": [], "proj": [], "lm": [], "new_tok": [], "ctc": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # The embed/lm_head modules_to_save clones are the "new_tok" group
        # (their gradient is masked elsewhere to update only the new rows).
        if "ctc_head" in name:
            buckets["ctc"].append(p)
        elif "embed_tokens.modules_to_save" in name or "lm_head.modules_to_save" in name:
            buckets["new_tok"].append(p)
        elif "multi_modal_projector" in name:
            buckets["proj"].append(p)
        elif ("audio_tower" in name or "audio_encoder" in name) and "lora_" in name:
            buckets["enc"].append(p)
        else:
            buckets["lm"].append(p)

    groups = [
        {"params": buckets["enc"], "lr": lrs.encoder_lora},
        {"params": buckets["proj"], "lr": lrs.projector},
        {"params": buckets["lm"], "lr": lrs.lm_lora},
        {"params": buckets["new_tok"], "lr": lrs.new_tokens},
    ]
    if buckets["ctc"]:
        groups.append({"params": buckets["ctc"], "lr": lrs.ctc_head})
    print(
        "[optim] param-group AdamW: "
        f"enc={sum(p.numel() for p in buckets['enc']):,} @ {lrs.encoder_lora:.0e}, "
        f"proj={sum(p.numel() for p in buckets['proj']):,} @ {lrs.projector:.0e}, "
        f"lm={sum(p.numel() for p in buckets['lm']):,} @ {lrs.lm_lora:.0e}, "
        f"new_tok={sum(p.numel() for p in buckets['new_tok']):,} @ {lrs.new_tokens:.0e}"
        + (f", ctc={sum(p.numel() for p in buckets['ctc']):,} @ {lrs.ctc_head:.0e}" if buckets["ctc"] else "")
    )
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def configure_wandb(config: TrainConfig) -> None:
    """Apply config-backed W&B settings before Trainer initializes integrations."""

    wandb = config.wandb
    env_values = {
        "WANDB_PROJECT": wandb.project,
        "WANDB_ENTITY": wandb.entity,
        "WANDB_NAME": wandb.name,
        "WANDB_MODE": wandb.mode,
        "WANDB_DIR": wandb.dir,
        "WANDB_LOG_MODEL": (
            str(wandb.log_model).lower() if wandb.log_model is not None else None
        ),
        "WANDB_TAGS": ",".join(wandb.tags) if wandb.tags else None,
    }
    for key, value in env_values.items():
        if value:
            os.environ[key] = value


def _is_wandb_enabled(report_to: str | list[str]) -> bool:
    if isinstance(report_to, str):
        targets = [part.strip().lower() for part in report_to.split(",") if part.strip()]
    else:
        targets = [str(item).strip().lower() for item in report_to]
    return "wandb" in targets


def _maybe_init_wandb(config: TrainConfig) -> None:
    if not _is_wandb_enabled(config.report_to):
        return
    try:
        import wandb
    except ImportError:
        print("[wandb] report_to includes wandb, but wandb package is not installed.")
        return

    if wandb.run is None:
        wandb_kwargs: dict[str, Any] = {
            "project": config.wandb.project,
            "entity": config.wandb.entity,
            "name": config.wandb.name,
            "tags": config.wandb.tags or None,
        }
        wandb.init(**{k: v for k, v in wandb_kwargs.items() if v is not None})

    _maybe_log_wandb_config_file(config)


def _maybe_log_wandb_config_file(config: TrainConfig) -> None:
    if not _is_wandb_enabled(config.report_to):
        return

    config_path = getattr(config, "_path", None)
    if not config_path:
        return

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        print(f"[wandb] Config file not found, skipping file log: {path}")
        return

    try:
        import wandb
    except ImportError:
        return

    if wandb.run is None:
        return

    wandb.save(str(path), base_path=str(path.parent), policy="now")
    print(f"[wandb] Logged config file: {path}")


def load_model_and_processor(config: TrainConfig):
    """Load Qwen2-Audio and wrap it with PEFT LoRA adapters."""

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

    quantization_config = None
    if config.use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        )

    processor = AutoProcessor.from_pretrained(config.model_name_or_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if config.bf16 else None,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    if config.audio_encoder_init_model:
        _replace_qwen_audio_encoder_from_whisper(
            model,
            source_model_name=config.audio_encoder_init_model,
            use_bf16=config.bf16,
        )

    tokenizer_vocab_before = len(processor.tokenizer)
    disfluency_token_ids = add_disfluency_tokens(model, processor, config.disfluency_tokens)
    tokenizer_vocab_after = len(processor.tokenizer)
    new_tokens_added = max(tokenizer_vocab_after - tokenizer_vocab_before, 0)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    resolved_target_modules = _resolve_lora_target_modules(
        model,
        requested_suffixes=config.lora.target_modules,
        audio_encoder_top_half_only=config.lora.audio_encoder_top_half_only,
        audio_encoder_layers=config.lora.audio_encoder_layers,
        audio_encoder_target_modules=config.lora.audio_encoder_target_modules,
    )
    if not config.lora.enable_audio_lora or not config.lora.enable_llm_lora:
        before_count = len(resolved_target_modules)
        resolved_target_modules = _filter_lora_target_modules(
            resolved_target_modules,
            enable_audio_lora=config.lora.enable_audio_lora,
            enable_llm_lora=config.lora.enable_llm_lora,
        )
        print(
            "[lora-check] LoRA stack filter applied: "
            f"enable_audio_lora={config.lora.enable_audio_lora}, "
            f"enable_llm_lora={config.lora.enable_llm_lora}; "
            f"dropped {before_count - len(resolved_target_modules)} modules, "
            f"keeping {len(resolved_target_modules)} modules."
        )
    lora_target_layer_params = _count_target_layer_parameters(model, resolved_target_modules)
    modules_to_save = _build_modules_to_save(
        config.lora.modules_to_save,
        train_multi_modal_projector=config.lora.train_multi_modal_projector,
    )
    if not config.lora.train_multi_modal_projector and "multi_modal_projector" in config.lora.modules_to_save:
        print(
            "[lora-check] train_multi_modal_projector=false; "
            "excluding multi_modal_projector from modules_to_save."
        )
    if new_tokens_added == 0:
        modules_to_save = [m for m in modules_to_save if m not in {"embed_tokens", "lm_head"}]
        print(
            "[param-check] No new tokens were added; excluding embed_tokens and lm_head "
            "from modules_to_save."
        )

    # Lever 4: create the silence head and attach it to the base model BEFORE PEFT
    # wrapping so that modules_to_save causes PEFT to save its weights in checkpoints.
    silence_head = None
    if config.silence_aux_weight > 0.0:
        from torch import nn

        hidden_dim = _get_encoder_hidden_dim(model)
        model_dtype = torch.bfloat16 if config.bf16 else (torch.float16 if config.fp16 else torch.float32)
        silence_head = nn.Linear(hidden_dim, ENCODER_NUM_CLASSES, bias=True, dtype=model_dtype)
        nn.init.normal_(silence_head.weight, std=0.02)
        nn.init.zeros_(silence_head.bias)
        model.silence_head = silence_head
        modules_to_save = [*modules_to_save, "silence_head"]
        print(
            f"[silence-head] Attached disfluency frame classifier: "
            f"Linear({hidden_dim}, {ENCODER_NUM_CLASSES}). modules_to_save updated."
        )

    # When the user wants a different LoRA rank for the encoder than for the LM
    # (v2 design: encoder r=8, LM r=16), build PEFT's rank_pattern/alpha_pattern
    # dicts so each resolved encoder module overrides the global r/alpha.
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}
    if config.lora.audio_encoder_rank is not None or config.lora.audio_encoder_alpha is not None:
        target_buckets_for_rank = _split_lora_targets_by_stack(resolved_target_modules)
        enc_modules = target_buckets_for_rank["audio"]
        enc_rank = config.lora.audio_encoder_rank
        enc_alpha = config.lora.audio_encoder_alpha
        if enc_rank is not None and enc_alpha is None:
            enc_alpha = enc_rank * 2  # match LoRA paper default ratio
        for m in enc_modules:
            if enc_rank is not None:
                rank_pattern[m] = enc_rank
            if enc_alpha is not None:
                alpha_pattern[m] = enc_alpha
        print(
            f"[lora-check] Encoder-specific rank/alpha applied to {len(enc_modules)} modules: "
            f"r={enc_rank}, alpha={enc_alpha} (LM stays at r={config.lora.r}, alpha={config.lora.alpha})"
        )

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=resolved_target_modules,
        modules_to_save=modules_to_save,
        use_dora=config.lora.use_dora,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    # After PEFT wrapping, retrieve the silence head from the new model graph.
    if silence_head is not None:
        silence_head = _find_silence_head(model)

    lora_adapter_params = _count_lora_adapter_parameters(model, resolved_target_modules)
    target_buckets = _split_lora_targets_by_stack(resolved_target_modules)
    audio_lora_adapter_params = _count_lora_adapter_parameters(model, target_buckets["audio"])
    llm_lora_adapter_params = _count_lora_adapter_parameters(model, target_buckets["llm"])
    other_lora_adapter_params = _count_lora_adapter_parameters(model, target_buckets["other"])
    print(f"[lora-check] Base-layer params where LoRA is attached: {lora_target_layer_params:,}")
    print(f"[lora-check] Trainable LoRA adapter params (A+B): {lora_adapter_params:,}")
    print(f"[lora-check] Trainable LoRA adapter params (audio): {audio_lora_adapter_params:,}")
    print(f"[lora-check] Trainable LoRA adapter params (llm): {llm_lora_adapter_params:,}")
    if target_buckets["other"]:
        print(f"[lora-check] Trainable LoRA adapter params (other): {other_lora_adapter_params:,}")
    if new_tokens_added > 0:
        _register_old_vocab_gradient_masks(model, old_vocab_size=tokenizer_vocab_before)
    _report_trainable_parameters(
        model,
        old_vocab_size=tokenizer_vocab_before,
        current_tokenizer_vocab_size=tokenizer_vocab_after,
        disfluency_token_ids=disfluency_token_ids,
    )
    # Sliding-window disfluency encoder (separate module, not part of PEFT graph).
    disfl_encoder = None
    if config.disfl_encoder is not None and config.disfl_aux_weight > 0.0:
        enc_cfg = config.disfl_encoder
        hidden_dim = _get_encoder_hidden_dim(model)
        disfl_encoder = DisfluencyEncoder(
            hidden_dim=enc_cfg.hidden_dim,
            num_layers=enc_cfg.num_layers,
            num_heads=enc_cfg.num_heads,
            window_size_s=enc_cfg.window_size_s,
            stride_s=enc_cfg.stride_s,
            out_dim=hidden_dim,
        )
        from audiollm_d.disfluency_encoder import compute_max_windows
        n_windows = compute_max_windows(enc_cfg.window_size_s, enc_cfg.stride_s)
        total_params = sum(p.numel() for p in disfl_encoder.parameters())
        win_enc_params = sum(p.numel() for p in disfl_encoder.window_encoder.parameters())
        ctx_params = sum(p.numel() for p in disfl_encoder.context_transformer.parameters())
        cls_params = sum(p.numel() for p in disfl_encoder.classifier.parameters())
        proj_params = sum(p.numel() for p in disfl_encoder.out_proj.parameters())
        print(f"[disfl-encoder] Sliding-window DisfluencyEncoder created:")
        print(f"[disfl-encoder]   window={enc_cfg.window_size_s}s, stride={enc_cfg.stride_s}s → {n_windows} windows per 30s clip")
        print(f"[disfl-encoder]   hidden_dim={enc_cfg.hidden_dim}, layers={enc_cfg.num_layers}, heads={enc_cfg.num_heads}, out_dim={hidden_dim}")
        print(f"[disfl-encoder]   alpha=0.01 (learnable, clamp=[0.005,0.1]); LM gradients flow to encoder")
        print(f"[disfl-encoder]   Parameter breakdown:")
        print(f"[disfl-encoder]     window_encoder (conv+norm):  {win_enc_params:>8,}")
        print(f"[disfl-encoder]     context_transformer:         {ctx_params:>8,}")
        print(f"[disfl-encoder]     classifier (9-class):        {cls_params:>8,}")
        print(f"[disfl-encoder]     out_proj (→ Whisper dim):    {proj_params:>8,}")
        print(f"[disfl-encoder]   Total trainable params:        {total_params:>8,}")

        if config.disfl_encoder_pretrained_path:
            import torch as _torch
            raw_state = _torch.load(config.disfl_encoder_pretrained_path, map_location="cpu")
            # Drop keys whose shape doesn't match the current model (e.g. binary
            # pre-training classifier vs 9-class joint training classifier).
            model_shapes = {k: v.shape for k, v in disfl_encoder.state_dict().items()}
            state = {k: v for k, v in raw_state.items() if k not in model_shapes or v.shape == model_shapes[k]}
            dropped = [k for k in raw_state if k not in state]
            missing, unexpected = disfl_encoder.load_state_dict(state, strict=False)
            print(f"[disfl-encoder] Loaded pretrained weights from {config.disfl_encoder_pretrained_path}")
            if dropped:
                print(f"[disfl-encoder]   shape-mismatched keys (re-init): {dropped}")
            if missing:
                print(f"[disfl-encoder]   missing keys (re-init): {missing}")
            if unexpected:
                print(f"[disfl-encoder]   unexpected keys (ignored): {unexpected}")

    return model, processor, disfluency_token_ids, silence_head, disfl_encoder


def _resolve_token_to_ids(tokenizer, token: str) -> list[int]:
    """Resolve a single disfluency token string to its tokenizer id(s).

    Registered special tokens resolve to one id directly. As a fallback (e.g. a
    token that was never added to the vocab) we encode the raw and space-prefixed
    forms and keep them only if they collapse to a single id.
    """
    unk_id = getattr(tokenizer, "unk_token_id", None)
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, int) and token_id != unk_id:
        return [token_id]

    candidate_ids: set[int] = set()
    variants = [token]
    if re.fullmatch(r"\w+", token):
        variants.append(f" {token}")
    space_ids = tokenizer.encode(" ", add_special_tokens=False)
    for variant in variants:
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if not ids:
            continue
        if variant.startswith(" ") and space_ids and ids[: len(space_ids)] == space_ids:
            ids = ids[len(space_ids):]
        if len(ids) == 1:
            candidate_ids.add(ids[0])
    return sorted(candidate_ids)


def _resolve_disfluency_token_ids(tokenizer, disfluency_tokens: list[str]) -> list[int]:
    """Resolve disfluency tokens to token IDs for loss weighting."""
    weighted_ids: set[int] = set()
    unresolved: list[str] = []
    for token in disfluency_tokens:
        ids = _resolve_token_to_ids(tokenizer, token)
        if ids:
            weighted_ids.update(ids)
        else:
            unresolved.append(token)

    ordered_ids = sorted(weighted_ids)
    print(f"[loss-weight] Disfluency tokens requested: {disfluency_tokens}")
    print(f"[loss-weight] Resolved weighted token IDs: {ordered_ids}")
    if unresolved:
        print(f"[loss-weight] WARNING: could not resolve these disfluency tokens to single token IDs: {unresolved}")
    return ordered_ids


def _compute_window_class_weights(
    train_records: list[dict[str, Any]],
    *,
    alignment_dir: str | None,
    n_classes: int,
) -> "torch.Tensor | None":
    """Compute sqrt-inverse-frequency class weights from all training alignments.

    Returns a float32 tensor of shape [n_classes] (mean-normalized so typical
    weights are O(1)), or None when no alignment files are found.
    """
    import math
    import torch
    from audiollm_d.disfluency_encoder import build_window_labels
    from audiollm_d.special_tokens import SPECIAL_TOKENS

    counts: list[int] = [0] * n_classes
    n_processed = 0

    for record in train_records:
        alignment_path = record.get("alignment_path")
        if not alignment_path and alignment_dir:
            clip_id = record.get("clip_id") or Path(record.get("audio_path", "")).stem
            alignment_path = str(Path(alignment_dir) / f"{clip_id}_alignment.json")
        if not alignment_path:
            continue
        try:
            alignment = json.loads(Path(alignment_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for cls_val in build_window_labels(alignment).tolist():
            if 0 <= int(cls_val) < n_classes:
                counts[int(cls_val)] += 1
        n_processed += 1

    if n_processed == 0:
        print("[disfl-encoder] No alignment files found; window loss will be unweighted.")
        return None

    raw = [1.0 / math.sqrt(max(c, 1)) for c in counts]
    # Normalise so speech (class 0) = 1.0 → all disfluency classes > 1.0.
    weights = [w / raw[0] for w in raw]

    class_names = ["speech"] + SPECIAL_TOKENS
    print(f"[disfl-encoder] Global window class weights (speech=1.0, {n_processed} samples):")
    for i, (name, w, cnt) in enumerate(zip(class_names, weights, counts)):
        print(f"[disfl-encoder]   class {i}  {name:<22}  count={cnt:>8,}  weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


def _compute_inverse_frequency_token_weights(
    *,
    train_records: list[dict[str, Any]],
    text_column: str,
    tokenizer,
    token_ids: list[int],
    base_weight: float,
) -> dict[int, float]:
    """Compute per-token weights proportional to inverse sqrt(train frequency)."""
    import math

    if not token_ids:
        return {}

    tracked_ids = set(token_ids)
    counts: Counter[int] = Counter()
    for row in train_records:
        text = str(row[text_column])
        encoded_ids = tokenizer.encode(text, add_disfluency_tokens=False)
        for token_id in encoded_ids:
            if token_id in tracked_ids:
                counts[token_id] += 1

    raw_weights: dict[int, float] = {
        token_id: 1.0 / math.sqrt(max(counts.get(token_id, 0), 1)) for token_id in token_ids
    }
    mean_raw_weight = sum(raw_weights.values()) / len(raw_weights)
    normalized_weights: dict[int, float] = {
        token_id: (raw_weight / mean_raw_weight) * base_weight
        for token_id, raw_weight in raw_weights.items()
    }

    print("[loss-weight] Per-token inverse-frequency weighting enabled.")
    print(f"[loss-weight] Base mean disfluency weight: {base_weight:.4f}")
    for token_id in token_ids:
        token_str = tokenizer.convert_ids_to_tokens(int(token_id))
        print(
            f"[loss-weight]   token={token_str!r:<20} id={token_id:<8} count={counts.get(token_id, 0):<8} "
            f"weight={normalized_weights[token_id]:.6f}"
        )
    return normalized_weights


def _resolve_disfluency_token_weights(
    tokenizer,
    disfluency_tokens: list[str],
    weights_by_token: dict[str, float] | None,
    default_weight: float,
) -> dict[int, float]:
    """Build the ``{token_id: alpha}`` map used by the (tiered) focal loss.

    Each disfluency token gets either its explicit per-token weight (frequency
    tier) or the uniform ``default_weight`` fallback. The resolved weight vector
    is logged so the tiering is auditable at startup.
    """
    weights_by_token = weights_by_token or {}
    weight_by_id: dict[int, float] = {}
    unresolved: list[str] = []
    log_rows: list[tuple[str, float, list[int]]] = []

    for token in disfluency_tokens:
        weight = float(weights_by_token.get(token, default_weight))
        ids = _resolve_token_to_ids(tokenizer, token)
        if not ids:
            unresolved.append(token)
            continue
        for token_id in ids:
            # If a subword id is shared across tokens, keep the larger weight.
            weight_by_id[token_id] = max(weight_by_id.get(token_id, 0.0), weight)
        log_rows.append((token, weight, ids))

    print("[loss-weight] Frequency-tiered disfluency weights (alpha):")
    for token, weight, ids in sorted(log_rows, key=lambda r: r[1], reverse=True):
        print(f"[loss-weight]   {token!r:18} alpha={weight:<5} ids={ids}")
    if unresolved:
        print(f"[loss-weight] WARNING: could not resolve these disfluency tokens to single token IDs: {unresolved}")
    return weight_by_id


class WeightedTokenTrainer:
    """Factory for a Trainer subclass that applies the tiered focal disfluency loss,
    plus (Phase 2) the auxiliary CTC objective on the encoder features."""

    @staticmethod
    def build(
        token_weights_by_id: dict[int, float],
        silence_head=None,
        silence_aux_weight: float = 0.0,
        silence_encoder_layer: int = -1,
        label_smoothing_factor: float = 0.0,
        disfl_encoder=None,
        disfl_aux_weight: float = 0.0,
        disfl_lr_multiplier: float = 10.0,
        disfl_class_weights=None,
        audio_encoder_learning_rate: float | None = None,
        llm_learning_rate: float | None = None,
        gamma: float = 0.0,
        focal_ids=None,
        ctc_head=None,
        ctc_alpha: float = 0.0,
    ):
        import torch
        import torch.nn.functional as F
        from transformers import Trainer

        weighted_ids = sorted(token_weights_by_id.keys())
        weighted_values = [float(token_weights_by_id[token_id]) for token_id in weighted_ids]
        weighted_ids_tensor = torch.tensor(weighted_ids, dtype=torch.long)
        weighted_values_tensor = torch.tensor(weighted_values, dtype=torch.float32)
        _silence_head = silence_head
        _silence_aux_weight = silence_aux_weight
        _silence_encoder_layer = silence_encoder_layer
        _label_smoothing = label_smoothing_factor
        _disfl_encoder = disfl_encoder
        _disfl_aux_weight = disfl_aux_weight
        _disfl_lr_multiplier = disfl_lr_multiplier
        _disfl_class_weights = disfl_class_weights  # global sqrt-inv-freq tensor or None
        _audio_encoder_learning_rate = audio_encoder_learning_rate
        _llm_learning_rate = llm_learning_rate
        # Focal-loss focusing parameter. 0.0 == plain (tiered) weighted CE, i.e.
        # byte-identical to main's baseline LM loss; >0 applies the (1-p)^gamma
        # down-weighting of easy tokens on top of the per-token weights.
        _gamma = float(gamma)
        # Phase 2 auxiliary CTC head on the audio encoder features.
        # None / alpha == 0 -> disabled (loss path is unchanged from Phase 1).
        _ctc_head = ctc_head
        _ctc_alpha = float(ctc_alpha)
        # Optional restriction of the focal (1-p)^gamma factor to specific token ids
        # (v3.5: focal on the rare brackets only, plain weighted CE elsewhere). None
        # applies focal to every target. Has no effect when gamma == 0.
        _focal_ids = {int(i) for i in focal_ids} if focal_ids else None
        ctc_enabled = ctc_head is not None and ctc_alpha > 0.0
        # Learnable injection scale. Clamped to [0.005, 0.1] so LM gradient can
        # calibrate signal strength without collapsing to zero.
        _disfl_alpha = (
            torch.tensor(0.01, requires_grad=True)
            if disfl_encoder is not None else None
        )

        class _Trainer(Trainer):
            # Accumulators for sub-loss logging; reset on each log() call.
            _lm_loss_sum: float = 0.0
            _aux_loss_sum: float = 0.0
            _disfl_loss_sum: float = 0.0
            _ctc_loss_sum: float = 0.0
            _log_steps: int = 0

            def log(self, logs: dict, start_time=None) -> None:
                if self._log_steps > 0:
                    logs["lm_loss"] = round(self._lm_loss_sum / self._log_steps, 6)
                    if _silence_head is not None and _silence_aux_weight > 0.0:
                        logs["aux_frame_loss"] = round(self._aux_loss_sum / self._log_steps, 6)
                    if _disfl_encoder is not None and _disfl_aux_weight > 0.0:
                        logs["disfl_window_loss"] = round(self._disfl_loss_sum / self._log_steps, 6)
                    if ctc_enabled:
                        logs["ctc_loss"] = round(self._ctc_loss_sum / self._log_steps, 6)
                    if _disfl_alpha is not None:
                        logs["disfl_alpha"] = round(_disfl_alpha.item(), 5)
                    self._lm_loss_sum = 0.0
                    self._aux_loss_sum = 0.0
                    self._disfl_loss_sum = 0.0
                    self._ctc_loss_sum = 0.0
                    self._log_steps = 0
                super().log(logs, start_time) if start_time is not None else super().log(logs)

            def compute_loss(
                self,
                model,
                inputs,
                return_outputs: bool = False,
                num_items_in_batch=None,
            ):
                # Pull the CTC side-channel tensors out before the model forward.
                ctc_targets = inputs.pop("ctc_targets", None)
                ctc_input_lengths = inputs.pop("ctc_input_lengths", None)
                ctc_target_lengths = inputs.pop("ctc_target_lengths", None)
                inputs.pop("ctc_frame_labels", None)  # debug/viz only; not used in the loss

                labels = inputs.pop("labels")
                # Lever 4: pop silence labels before the model forward pass.
                silence_labels = inputs.pop("silence_frame_labels", None)

                # Register a one-shot forward hook to capture encoder hidden states.
                # When silence_encoder_layer >= 0, hook that specific transformer layer
                # (local acoustic features); otherwise hook the full encoder output.
                _enc_store: dict[str, Any] = {}
                _hook_handle = None
                if _silence_head is not None and _silence_aux_weight > 0.0 and silence_labels is not None:
                    if _silence_encoder_layer >= 0:
                        hook_target = _find_encoder_layer(model, _silence_encoder_layer)
                    else:
                        hook_target = _find_audio_encoder(model)
                    if hook_target is not None:
                        def _enc_hook(mod, inp, out):
                            hs = (
                                out.last_hidden_state
                                if hasattr(out, "last_hidden_state")
                                else (out[0] if isinstance(out, tuple) else out)
                            )
                            _enc_store["enc"] = hs

                        _hook_handle = hook_target.register_forward_hook(_enc_hook)

                # Sliding-window disfluency encoder: run before model forward,
                # inject residual into Whisper encoder output via a hook.
                _disfl_residual = None
                _win_logits = None
                _disfl_hook_handle = None
                window_labels = inputs.pop("disfl_window_labels", None)
                if _disfl_encoder is not None and _disfl_aux_weight > 0.0:
                    input_features = inputs.get("input_features")
                    if input_features is not None:
                        _disfl_residual, _win_logits = _disfl_encoder(input_features)
                        encoder = _find_audio_encoder(model)
                        if encoder is not None:
                            def _disfl_residual_hook(mod, inp, out):
                                hs = (
                                    out.last_hidden_state
                                    if hasattr(out, "last_hidden_state")
                                    else out[0]
                                )
                                T = min(hs.size(1), _disfl_residual.size(1))
                                # Learnable alpha (clamped to [0.005, 0.1]) calibrates
                                # injection strength. No detach: LM gradients flow back
                                # to both alpha and the encoder for end-to-end calibration.
                                _alpha = _disfl_alpha.clamp(0.005, 0.1).to(hs.device).to(hs.dtype)
                                enhanced = hs + _alpha * _disfl_residual[:, :T, :].to(hs.dtype)
                                if hasattr(out, "last_hidden_state"):
                                    return type(out)(last_hidden_state=enhanced)
                                return (enhanced,) + out[1:]

                            _disfl_hook_handle = encoder.register_forward_hook(
                                _disfl_residual_hook
                            )

                outputs = model(**inputs)
                if _hook_handle is not None:
                    _hook_handle.remove()
                if _disfl_hook_handle is not None:
                    _disfl_hook_handle.remove()

                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                flat_logits = shift_logits.view(-1, shift_logits.size(-1))
                flat_targets = shift_labels.view(-1)
                log_probs = F.log_softmax(flat_logits, dim=-1)
                hard_losses = F.nll_loss(log_probs, flat_targets.clamp_min(0), reduction="none")
                if _label_smoothing > 0.0:
                    smooth_term = _label_smoothing
                    smooth_losses = -log_probs.mean(dim=-1)
                    losses = (1.0 - _label_smoothing) * hard_losses + _label_smoothing * smooth_losses
                else:
                    losses = hard_losses

                if _gamma > 0.0:
                    # Focal loss: down-weight easy (confident) tokens by (1 - p_true)^gamma,
                    # where p_true = exp(-nll) is the model's probability on the gold token.
                    # Mirrors audiollm_d.loss.disfluency_focal_loss and reduces to the plain
                    # weighted CE above when gamma == 0 (so main's baseline is preserved).
                    p_true = torch.exp(-hard_losses)
                    focal = (1.0 - p_true).clamp_min(0.0).pow(_gamma)
                    if _focal_ids is not None:
                        # v3.5: restrict focal modulation to the listed token ids; every
                        # other target keeps plain weighted CE (focal factor = 1.0).
                        fids = torch.tensor(
                            sorted(_focal_ids), dtype=torch.long, device=flat_targets.device
                        )
                        is_focal = (flat_targets[..., None] == fids).any(dim=-1)
                        focal = torch.where(is_focal, focal, torch.ones_like(focal))
                    losses = losses * focal

                flat_labels = flat_targets
                weights = torch.ones_like(losses)
                valid_mask = flat_labels.ne(-100)
                if weighted_ids_tensor.numel():
                    device_weighted_ids = weighted_ids_tensor.to(flat_labels.device)
                    device_weighted_values = weighted_values_tensor.to(losses.device)
                    for token_id, token_weight in zip(
                        device_weighted_ids, device_weighted_values, strict=True
                    ):
                        weights = torch.where(flat_labels.eq(token_id), weights * token_weight, weights)

                lm_loss = (losses * weights * valid_mask).sum() / valid_mask.sum().clamp_min(1)
                total_loss = lm_loss
                self._lm_loss_sum += lm_loss.detach().item()
                self._log_steps += 1

                # Lever 4: auxiliary encoder disfluency frame classification loss.
                # 9-class cross-entropy: class 0 = speech, classes 1-8 = individual
                # disfluency tokens (SPECIAL_TOKENS order), -1 = ignore.
                if (
                    _silence_head is not None
                    and _silence_aux_weight > 0.0
                    and silence_labels is not None
                    and "enc" in _enc_store
                ):
                    enc_out = _enc_store["enc"]  # [B, T_enc, D]
                    frame_logits = _silence_head(enc_out)  # [B, T_enc, C]
                    sil_lbls = silence_labels.to(enc_out.device)  # [B, 1500]
                    T = min(frame_logits.size(1), sil_lbls.size(1))
                    frame_logits = frame_logits[:, :T, :]  # [B, T, C]
                    sil_lbls = sil_lbls[:, :T]             # [B, T]
                    valid = sil_lbls.ne(-1)
                    if valid.any():
                        # Inverse-frequency class weights to handle speech-class dominance.
                        n_classes = frame_logits.size(-1)
                        counts = torch.bincount(
                            sil_lbls[valid].clamp_min(0), minlength=n_classes
                        ).float().clamp_min(1.0)
                        class_weights = (valid.sum().float() / (n_classes * counts)).to(frame_logits.dtype)
                        aux_loss = F.cross_entropy(
                            frame_logits.reshape(-1, n_classes),
                            sil_lbls.reshape(-1),
                            weight=class_weights,
                            ignore_index=-1,
                        )
                        self._aux_loss_sum += aux_loss.detach().item()
                        total_loss = lm_loss + _silence_aux_weight * aux_loss

                # Sliding-window disfluency encoder window classification loss.
                if (
                    _disfl_encoder is not None
                    and _disfl_aux_weight > 0.0
                    and _win_logits is not None
                    and window_labels is not None
                ):
                    win_lbls = window_labels.to(_win_logits.device)
                    valid = win_lbls.ne(-1)
                    if valid.any():
                        n_cls = _win_logits.size(-1)
                        # Global sqrt-inv-freq class weights (stable across batches).
                        cw = (
                            _disfl_class_weights.to(_win_logits.device, _win_logits.dtype)
                            if _disfl_class_weights is not None
                            else None
                        )
                        disfl_loss = F.cross_entropy(
                            _win_logits.reshape(-1, n_cls),
                            win_lbls.reshape(-1),
                            weight=cw,
                            ignore_index=-1,
                        )
                        self._disfl_loss_sum += disfl_loss.detach().item()
                        total_loss = total_loss + _disfl_aux_weight * disfl_loss

                # Phase 2 auxiliary CTC loss on the audio-encoder features. The model
                # forward above populated the head's captured features via its hook.
                if ctc_enabled and ctc_targets is not None:
                    ctc_log_probs = _ctc_head.forward_from_capture()  # (B, T, V) fp32
                    ctc_loss = _ctc_head.ctc_loss(
                        ctc_log_probs,
                        ctc_targets.to(ctc_log_probs.device),
                        ctc_input_lengths.to("cpu"),  # cudnn CTC wants lengths on CPU
                        ctc_target_lengths.to("cpu"),
                    )
                    total_loss = total_loss + _ctc_alpha * ctc_loss
                    self._ctc_loss_sum += float(ctc_loss.detach())

                return (total_loss, outputs) if return_outputs else total_loss

            def create_optimizer(self):
                from torch.optim import AdamW as TorchAdamW

                if self.optimizer is not None:
                    return self.optimizer

                use_stack_specific_lrs = (
                    _audio_encoder_learning_rate is not None
                    or _llm_learning_rate is not None
                )
                if _disfl_encoder is None and not use_stack_specific_lrs:
                    return super().create_optimizer()

                disfl_ids = (
                    {id(p) for p in _disfl_encoder.parameters()}
                    if _disfl_encoder is not None
                    else set()
                )
                model_params_by_stack: dict[str, list] = {
                    "audio_encoder": [],
                    "llm": [],
                    "other": [],
                }
                for name, param in self.model.named_parameters():
                    if not param.requires_grad or id(param) in disfl_ids:
                        continue
                    model_params_by_stack[_optimizer_stack_for_parameter(name)].append(param)

                disfl_params = (
                    [p for p in _disfl_encoder.parameters() if p.requires_grad]
                    if _disfl_encoder is not None
                    else []
                )
                # alpha trains at the base LR alongside LoRA — it's a calibration
                # parameter, not an encoder parameter.
                alpha_params = [_disfl_alpha] if _disfl_alpha is not None else []
                if alpha_params:
                    model_params_by_stack["other"].extend(alpha_params)

                stack_lrs = {
                    "audio_encoder": _audio_encoder_learning_rate or self.args.learning_rate,
                    "llm": _llm_learning_rate or self.args.learning_rate,
                    "other": self.args.learning_rate,
                }
                param_groups = []
                for stack_name in ("audio_encoder", "llm", "other"):
                    params = model_params_by_stack[stack_name]
                    if not params:
                        continue
                    param_groups.append(
                        {
                            "params": params,
                            "lr": stack_lrs[stack_name],
                            "weight_decay": self.args.weight_decay,
                            "name": stack_name,
                        }
                    )

                if disfl_params:
                    param_groups.append(
                        {
                            "params": disfl_params,
                            "lr": self.args.learning_rate * _disfl_lr_multiplier,
                            "weight_decay": self.args.weight_decay,
                            "name": "disfl_encoder",
                        }
                    )
                for group in param_groups:
                    n_params = sum(p.numel() for p in group["params"])
                    print(
                        "[optimizer] "
                        f"group={group['name']} params={n_params:,} "
                        f"lr={group['lr']:.2e} weight_decay={group['weight_decay']}"
                    )
                self.optimizer = TorchAdamW(param_groups)
                return self.optimizer

            def save_model(self, output_dir: str | None = None, **kwargs) -> None:
                super().save_model(output_dir, **kwargs)
                if _disfl_encoder is not None:
                    import pathlib
                    save_dir = pathlib.Path(output_dir or self.args.output_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(_disfl_encoder.state_dict(), save_dir / "disfl_encoder.pt")
                    if _disfl_alpha is not None:
                        torch.save({"alpha": _disfl_alpha.detach().cpu()}, save_dir / "disfl_alpha.pt")
                    print(f"[disfl-encoder] Saved encoder weights to {save_dir / 'disfl_encoder.pt'} "
                          f"(alpha={_disfl_alpha.item():.4f})" if _disfl_alpha is not None
                          else f"[disfl-encoder] Saved encoder weights to {save_dir / 'disfl_encoder.pt'}")

        return _Trainer


class CurriculumMixedTrainDataset:
    """Epoch-aware synthetic/original mixture for single-run curriculum training."""

    def __init__(
        self,
        *,
        synthetic_records: list[dict[str, Any]],
        original_records: list[dict[str, Any]],
        synthetic_only_epochs: int,
        ramp_epochs: int,
        final_original_ratio: float,
        seed: int,
    ) -> None:
        if not synthetic_records:
            raise ValueError("Curriculum dataset requires at least one synthetic record.")
        if not original_records:
            raise ValueError("Curriculum dataset requires at least one original record.")
        self._synthetic = synthetic_records
        self._original = original_records
        self._synthetic_only_epochs = max(int(synthetic_only_epochs), 0)
        self._ramp_epochs = max(int(ramp_epochs), 0)
        self._final_original_ratio = float(min(max(final_original_ratio, 0.0), 1.0))
        self._seed = int(seed)
        self._epoch_idx = 0
        self._epoch_size = max(len(self._synthetic), len(self._original))

    def __len__(self) -> int:
        return self._epoch_size

    def set_epoch(self, epoch_idx: int) -> None:
        self._epoch_idx = max(int(epoch_idx), 0)

    def current_original_ratio(self) -> float:
        if self._epoch_idx < self._synthetic_only_epochs:
            return 0.0
        if self._ramp_epochs <= 0:
            return self._final_original_ratio
        ramp_step = self._epoch_idx - self._synthetic_only_epochs + 1
        progress = min(max(ramp_step / float(self._ramp_epochs), 0.0), 1.0)
        return self._final_original_ratio * progress

    def current_mix_summary(self) -> str:
        original_ratio = self.current_original_ratio()
        synthetic_ratio = 1.0 - original_ratio
        return (
            f"epoch={self._epoch_idx + 1} "
            f"synthetic_ratio={synthetic_ratio:.3f} "
            f"original_ratio={original_ratio:.3f}"
        )

    def _sample_from_original(self, idx: int) -> bool:
        ratio = self.current_original_ratio()
        if ratio <= 0.0:
            return False
        if ratio >= 1.0:
            return True
        # Deterministic per (seed, epoch, idx) to keep behavior reproducible.
        draw = (
            (int(idx) * 1103515245 + self._epoch_idx * 12345 + self._seed)
            & 0xFFFFFFFF
        ) / 4294967296.0
        return draw < ratio

    def __getitem__(self, idx: int) -> dict[str, Any]:
        use_original = self._sample_from_original(idx)
        source = self._original if use_original else self._synthetic
        # Epoch-dependent offset to avoid repeating the same prefix each epoch.
        source_idx = (int(idx) + self._epoch_idx * 9973) % len(source)
        return source[source_idx]


def _setup_ctc_head(model, config: TrainConfig):
    """Build the auxiliary CTC head, attach it to ``model``, and install the encoder hook.

    Returns ``(head, vocab, labels_dir)``. The vocab is read from the label artifact's
    ``ctc_vocab.json`` (source of truth); a ``config.ctc.classes`` override must agree with it.
    The head is registered as ``model.ctc_head`` so it appears in ``named_parameters()`` and the
    optimizer; the forward hook on ``audio_tower`` captures post-pooling features each forward.
    """
    labels_root = Path(config.ctc.labels_dir)
    vocab_json = labels_root / "ctc_vocab.json"
    if vocab_json.exists():
        ctc_vocab = CtcVocab.from_json(json.loads(vocab_json.read_text(encoding="utf-8")))
        if config.ctc.classes:
            configured = CtcVocab.from_classes(config.ctc.classes)
            if tuple(configured.classes) != tuple(ctc_vocab.classes):
                raise ValueError(
                    f"config.ctc.classes {configured.classes} disagree with the artifact vocab "
                    f"{ctc_vocab.classes} in {vocab_json}. Regenerate labels or fix the config."
                )
    else:
        ctc_vocab = CtcVocab.from_classes(config.ctc.classes)

    audio_tower, d_encoder = locate_audio_tower(model)
    head = AuxiliaryCtcHead.build(d_encoder, ctc_vocab.vocab_size, ctc_vocab.blank_index)
    device = next(audio_tower.parameters()).device
    head.to(device)
    head.register_capture_hook(audio_tower)
    model.ctc_head = head  # register as a submodule -> shows up in named_parameters()/optimizer

    labels_dir = labels_root / "labels"
    n_params = sum(p.numel() for p in head.parameters())
    print(
        f"[ctc] Enabled. d_encoder={d_encoder}, vocab={ctc_vocab.classes} "
        f"(blank={ctc_vocab.blank_index}), alpha={config.ctc.alpha}"
    )
    print(f"[ctc] Head params: {n_params:,} on {device}; labels={labels_dir}")
    return head, ctc_vocab, str(labels_dir)


def train(config: TrainConfig) -> None:
    from datasets import Dataset
    from transformers import TrainerCallback, TrainingArguments, set_seed

    from audiollm_d.dataset import (
        NOISE_AUGMENTATION_FLAG_COLUMN,
        Qwen2AudioTranscriptionCollator,
        load_manifest_dataset,
    )

    set_seed(config.seed)
    configure_wandb(config)
    _maybe_init_wandb(config)
    model, processor, disfluency_token_ids, silence_head, disfl_encoder = load_model_and_processor(config)

    ctc_head = None
    ctc_vocab = None
    ctc_labels_dir = None
    if config.ctc.enabled:
        ctc_head, ctc_vocab, ctc_labels_dir = _setup_ctc_head(model, config)

    synthetic_train_records = load_manifest_dataset(
        config.data.train_manifest,
        max_samples=config.data.max_train_samples,
    )
    original_train_records = None
    if config.data.curriculum_original_manifest:
        original_train_records = load_manifest_dataset(
            config.data.curriculum_original_manifest,
            max_samples=config.data.max_curriculum_original_samples,
        )
        print(
            "[curriculum] Enabled single-run curriculum with synthetic manifest "
            f"{config.data.train_manifest} and original manifest "
            f"{config.data.curriculum_original_manifest}."
        )
        print(
            "[curriculum] Schedule: synthetic-only epochs="
            f"{config.data.curriculum_synthetic_only_epochs}, "
            f"ramp_epochs={config.data.curriculum_ramp_epochs}, "
            f"final_original_ratio={config.data.curriculum_final_original_ratio:.3f}."
        )

    train_records = (
        synthetic_train_records + (original_train_records or [])
        if original_train_records
        else synthetic_train_records
    )
    eval_records = None
    if config.data.validation_manifest:
        eval_records = load_manifest_dataset(
            config.data.validation_manifest,
            max_samples=config.data.max_eval_samples,
        )
        print(
            f"[eval] Validation/test loss tracking enabled every {config.eval_steps} steps "
            f"using {config.data.validation_manifest}."
        )

    synthetic_manifest_name = config.data.train_manifest
    original_manifest_name = config.data.curriculum_original_manifest
    train_manifest_name = (
        f"{config.data.train_manifest} + {config.data.curriculum_original_manifest}"
        if original_train_records
        else config.data.train_manifest
    )
    test_manifest_name = (
        config.data.validation_manifest
        if config.data.validation_manifest
        else config.data.train_manifest
    )
    collator = Qwen2AudioTranscriptionCollator(
        processor=processor,
        audio_column=config.data.audio_column,
        text_column=config.data.text_column,
        sampling_rate=config.data.sampling_rate,
        noise_augmentation_prob=config.data.noise_augmentation_prob,
        augmentation_types=config.data.augmentation_types,
        noise_snr_db_min=config.data.noise_snr_db_min,
        noise_snr_db_max=config.data.noise_snr_db_max,
        babble_snr_db_min=config.data.babble_snr_db_min,
        babble_snr_db_max=config.data.babble_snr_db_max,
        babble_num_sources=config.data.babble_num_sources,
        reverb_rt60_ms_min=config.data.reverb_rt60_ms_min,
        reverb_rt60_ms_max=config.data.reverb_rt60_ms_max,
        speed_min=config.data.speed_min,
        speed_max=config.data.speed_max,
        codec_resample_rate=config.data.codec_resample_rate,
        disfl_window_size_s=config.disfl_encoder.window_size_s if config.disfl_encoder else 0.5,
        disfl_stride_s=config.disfl_encoder.stride_s if config.disfl_encoder else 0.1,
        ctc_labels_dir=ctc_labels_dir,
        ctc_vocab=ctc_vocab,
    )
    if config.data.noise_augmentation_prob > 0.0:
        types_str = ", ".join(config.data.augmentation_types)
        print(
            f"[augment] Train-only augmentation enabled: "
            f"p={config.data.noise_augmentation_prob:.3f}, types=[{types_str}]"
        )

    if config.run_lora_init_sanity_check:
        _run_lora_init_sanity_check(
            config=config,
            model=model,
            processor=processor,
            train_records=train_records,
            eval_records=eval_records,
            train_manifest_name=train_manifest_name,
            test_manifest_name=test_manifest_name,
        )

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_records else "no",
        eval_steps=config.eval_steps if eval_records else None,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16,
        fp16=config.fp16,
        label_smoothing_factor=config.label_smoothing_factor,
        report_to=config.report_to,
        run_name=config.wandb.name,
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
    )

    synthetic_with_noise = [
        {**row, NOISE_AUGMENTATION_FLAG_COLUMN: True}
        for row in synthetic_train_records
    ]
    original_with_noise = [
        {**row, NOISE_AUGMENTATION_FLAG_COLUMN: True}
        for row in (original_train_records or [])
    ]
    if original_train_records:
        curriculum_dataset = CurriculumMixedTrainDataset(
            synthetic_records=synthetic_with_noise,
            original_records=original_with_noise,
            synthetic_only_epochs=config.data.curriculum_synthetic_only_epochs,
            ramp_epochs=config.data.curriculum_ramp_epochs,
            final_original_ratio=config.data.curriculum_final_original_ratio,
            seed=config.seed,
        )
        train_dataset = curriculum_dataset
    else:
        train_dataset = Dataset.from_list(synthetic_with_noise)
        curriculum_dataset = None
    synthetic_eval_dataset = Dataset.from_list(synthetic_train_records)
    original_eval_dataset = Dataset.from_list(original_train_records) if original_train_records else None
    eval_dataset = Dataset.from_list(eval_records) if eval_records else None

    # --- Disfluency loss weighting: two candidate sources (REBASE DECISION POINT) ---
    # `token_weight_map`: our frequency-tiered / explicit weights from
    #   config.disfluency_loss_weights (+ uniform fallback). Logged here for audit.
    # `token_weights_by_id`: main's corpus inverse-frequency weights.
    # The trainer is currently fed `token_weights_by_id` (main's proven baseline);
    # focal (gamma / focal_tokens) is layered on top. To switch the active source to
    # our tiered weights, pass `token_weights_by_id=token_weight_map` into build() below.
    token_weight_map = _resolve_disfluency_token_weights(
        processor.tokenizer,
        config.disfluency_tokens,
        config.disfluency_loss_weights,
        config.disfluency_loss_weight,
    )
    token_weights_by_id = _compute_inverse_frequency_token_weights(
        train_records=train_records,
        text_column=config.data.text_column,
        tokenizer=processor.tokenizer,
        token_ids=disfluency_token_ids,
        base_weight=config.disfluency_loss_weight,
    )
    disfl_class_weights = None
    if disfl_encoder is not None:
        model_device = next(model.parameters()).device
        disfl_encoder = disfl_encoder.to(model_device)
        print(
            f"[disfl-encoder] Moved to device={model_device}, "
            f"disfl_aux_weight={config.disfl_aux_weight}, "
            f"lr_multiplier={config.disfl_encoder_lr_multiplier}× "
            f"(effective_lr={config.learning_rate * config.disfl_encoder_lr_multiplier:.2e})"
        )
        disfl_class_weights = _compute_window_class_weights(
            train_records,
            alignment_dir=config.disfl_alignment_dir,
            n_classes=WINDOW_NUM_CLASSES,
        )

    print(f"[loss-weight] focal_gamma={config.focal_gamma}")
    focal_ids = None
    if config.focal_tokens:
        focal_ids = _resolve_disfluency_token_ids(processor.tokenizer, config.focal_tokens)
        print(
            f"[loss-weight] focal restricted to {config.focal_tokens} "
            f"-> ids={focal_ids} (all other tokens get flat alpha-weighted CE)"
        )
    trainer_cls = WeightedTokenTrainer.build(
        token_weights_by_id=token_weights_by_id,
        silence_head=silence_head,
        silence_aux_weight=config.silence_aux_weight,
        silence_encoder_layer=config.silence_encoder_layer,
        label_smoothing_factor=config.label_smoothing_factor,
        disfl_encoder=disfl_encoder,
        disfl_aux_weight=config.disfl_aux_weight,
        disfl_lr_multiplier=config.disfl_encoder_lr_multiplier,
        disfl_class_weights=disfl_class_weights,
        audio_encoder_learning_rate=config.audio_encoder_learning_rate,
        llm_learning_rate=config.llm_learning_rate,
        gamma=config.focal_gamma,
        focal_ids=focal_ids,
        ctc_head=ctc_head,
        ctc_alpha=config.ctc.alpha if config.ctc.enabled else 0.0,
    )
    if silence_head is not None:
        layer_desc = (
            f"layer {config.silence_encoder_layer}"
            if config.silence_encoder_layer >= 0
            else "final encoder output"
        )
        print(
            f"[silence-head] Encoder supervision enabled: aux_weight={config.silence_aux_weight}, "
            f"hook_target={layer_desc}."
        )
    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    if config.param_group_lrs is not None:
        # Build optimizer manually so each component gets its own LR. Scheduler
        # stays None → Trainer creates a default cosine schedule that scales all
        # groups together (relative LR ratios are preserved).
        optimizer = _build_param_group_optimizer(
            model, config.param_group_lrs, config.weight_decay
        )
        trainer_kwargs["optimizers"] = (optimizer, None)
    trainer = trainer_cls(**trainer_kwargs)
    if curriculum_dataset is not None:
        class CurriculumEpochCallback(TrainerCallback):
            def on_train_begin(self, args, state, control, **kwargs):
                curriculum_dataset.set_epoch(0)
                print(f"[curriculum] {curriculum_dataset.current_mix_summary()}")
                return control

            def on_epoch_begin(self, args, state, control, **kwargs):
                epoch_idx = int(state.epoch) if state.epoch is not None else 0
                curriculum_dataset.set_epoch(epoch_idx)
                print(f"[curriculum] {curriculum_dataset.current_mix_summary()}")
                return control

        trainer.add_callback(CurriculumEpochCallback())

    pre_metrics: dict[str, Any] = {}
    post_metrics: dict[str, Any] = {}
    pre_loss_metrics: dict[str, Any] = {}
    post_loss_metrics: dict[str, Any] = {}
    epoch_metrics: dict[str, Any] = {}
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    def _adapter_context(*, disable_lora: bool):
        if disable_lora and hasattr(model, "disable_adapter"):
            return model.disable_adapter()
        if disable_lora:
            print(
                "[eval] Requested LoRA-disabled evaluation, but model does not expose "
                "disable_adapter(); evaluating with adapters enabled."
            )
        return contextlib.nullcontext()

    def _write_split_predictions(phase_metrics: dict[str, Any], *, phase_name: str) -> dict[str, Any]:
        cleaned_metrics: dict[str, Any] = {}
        for split_name, split_payload in phase_metrics.items():
            if not isinstance(split_payload, dict):
                cleaned_metrics[split_name] = split_payload
                continue

            payload_copy = dict(split_payload)
            predictions = payload_copy.pop("predictions", None)
            if predictions:
                pred_path = output_root / f"{phase_name}_{split_name}_predictions.jsonl"
                with pred_path.open("w", encoding="utf-8") as f:
                    for row in predictions:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[eval] Wrote predictions to: {pred_path}")
            cleaned_metrics[split_name] = payload_copy
        return cleaned_metrics

    def _run_pre_post_metrics(
        phase: str,
        *,
        disable_lora: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        split_metrics: dict[str, Any] = {}
        split_loss_metrics: dict[str, Any] = {}

        with _adapter_context(disable_lora=disable_lora):
            split_metrics["synthetic_train"] = _evaluate_model_on_records(
                label=f"{phase}_synthetic_train",
                model=model,
                processor=processor,
                records=synthetic_train_records,
                manifest_name=synthetic_manifest_name,
                audio_column=config.data.audio_column,
                text_column=config.data.text_column,
                max_new_tokens=config.pre_post_eval_max_new_tokens,
                max_samples=config.pre_post_eval_max_samples,
                eval_generation_batch_size=config.eval_generation_batch_size,
                eval_generation_do_sample=config.eval_generation_do_sample,
                eval_generation_temperature=config.eval_generation_temperature,
            )
            synthetic_loss_prefix = f"{phase}_synthetic_train"
            split_loss_metrics["synthetic_train"] = trainer.evaluate(
                eval_dataset=synthetic_eval_dataset,
                metric_key_prefix=synthetic_loss_prefix,
            )
            synthetic_loss_key = f"{synthetic_loss_prefix}_loss"
            if synthetic_loss_key in split_loss_metrics["synthetic_train"]:
                print(
                    "[eval:"
                    f"{phase}_synthetic_train] Loss: "
                    f"{split_loss_metrics['synthetic_train'][synthetic_loss_key]:.6f}"
                )

            if original_train_records is not None and original_eval_dataset is not None:
                split_metrics["original_train"] = _evaluate_model_on_records(
                    label=f"{phase}_original_train",
                    model=model,
                    processor=processor,
                    records=original_train_records,
                    manifest_name=original_manifest_name or "original_train_manifest",
                    audio_column=config.data.audio_column,
                    text_column=config.data.text_column,
                    max_new_tokens=config.pre_post_eval_max_new_tokens,
                    max_samples=config.pre_post_eval_max_samples,
                    eval_generation_batch_size=config.eval_generation_batch_size,
                    eval_generation_do_sample=config.eval_generation_do_sample,
                    eval_generation_temperature=config.eval_generation_temperature,
                )
                original_loss_prefix = f"{phase}_original_train"
                split_loss_metrics["original_train"] = trainer.evaluate(
                    eval_dataset=original_eval_dataset,
                    metric_key_prefix=original_loss_prefix,
                )
                original_loss_key = f"{original_loss_prefix}_loss"
                if original_loss_key in split_loss_metrics["original_train"]:
                    print(
                        "[eval:"
                        f"{phase}_original_train] Loss: "
                        f"{split_loss_metrics['original_train'][original_loss_key]:.6f}"
                    )

            # If a validation/test split exists, evaluate it as well.
            if eval_records is not None and eval_dataset is not None:
                split_metrics["test"] = _evaluate_model_on_records(
                    label=f"{phase}_test",
                    model=model,
                    processor=processor,
                    records=eval_records,
                    manifest_name=test_manifest_name,
                    audio_column=config.data.audio_column,
                    text_column=config.data.text_column,
                    max_new_tokens=config.pre_post_eval_max_new_tokens,
                    max_samples=config.pre_post_eval_max_samples,
                    eval_generation_batch_size=config.eval_generation_batch_size,
                    eval_generation_do_sample=config.eval_generation_do_sample,
                    eval_generation_temperature=config.eval_generation_temperature,
                )
                test_loss_prefix = f"{phase}_test"
                split_loss_metrics["test"] = trainer.evaluate(
                    eval_dataset=eval_dataset,
                    metric_key_prefix=test_loss_prefix,
                )
                test_loss_key = f"{test_loss_prefix}_loss"
                if test_loss_key in split_loss_metrics["test"]:
                    print(f"[eval:{phase}_test] Loss: {split_loss_metrics['test'][test_loss_key]:.6f}")

        return split_metrics, split_loss_metrics

    def _log_generation_metrics_to_trainer(
        *,
        split_metrics: dict[str, Any],
        split_loss_metrics: dict[str, Any],
        metric_namespace: str,
        epoch_idx: int,
    ) -> None:
        payload: dict[str, float] = {
            f"{metric_namespace}/epoch": float(epoch_idx),
        }
        for split_name, metrics in split_metrics.items():
            if not isinstance(metrics, dict):
                continue
            if "wer" in metrics:
                payload[f"{metric_namespace}/{split_name}_wer"] = float(metrics["wer"])
            special = metrics.get("disfluency_token_metrics")
            if isinstance(special, dict):
                for key in ("disfluency_token_precision", "disfluency_token_recall", "disfluency_token_f1"):
                    if key in special:
                        payload[f"{metric_namespace}/{split_name}_{key}"] = float(special[key])

        for split_name, metrics in split_loss_metrics.items():
            if not isinstance(metrics, dict):
                continue
            loss_key = next(
                (key for key in metrics.keys() if key.endswith(f"_{split_name}_loss")),
                None,
            )
            if loss_key is not None:
                payload[f"{metric_namespace}/{split_name}_loss"] = float(metrics[loss_key])

        trainer.log(payload)

    def _run_epoch_eval(epoch_idx: int) -> None:
        phase_name = f"epoch_{epoch_idx:03d}"
        split_metrics, split_loss_metrics = _run_pre_post_metrics(phase_name)
        stripped_metrics = _write_split_predictions(split_metrics, phase_name=phase_name)
        epoch_metrics[phase_name] = {
            "metrics": stripped_metrics,
            "loss_metrics": split_loss_metrics,
        }
        _log_generation_metrics_to_trainer(
            split_metrics=stripped_metrics,
            split_loss_metrics=split_loss_metrics,
            metric_namespace="epoch_eval",
            epoch_idx=epoch_idx,
        )

    class EpochEndEvalCallback(TrainerCallback):
        def __init__(self):
            self._last_epoch_run: int = 0
            self._interval: int = max(int(config.epoch_eval_interval), 1)
            self._start_epoch: int = max(int(config.eval_start_epoch), 0)

        def on_epoch_end(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return control
            if state.epoch is None:
                return control
            if float(state.epoch) >= float(args.num_train_epochs):
                return control
            epoch_idx = int(round(float(state.epoch)))
            if epoch_idx <= self._last_epoch_run:
                return control
            if epoch_idx < self._start_epoch:
                return control
            if epoch_idx % self._interval != 0:
                return control
            self._last_epoch_run = epoch_idx
            print(
                f"[eval] Running generation metrics for epoch {epoch_idx} "
                f"(every {self._interval} epochs)."
            )
            _run_epoch_eval(epoch_idx)
            return control

    trainer.add_callback(EpochEndEvalCallback())

    if config.run_pre_post_train_eval:
        pre_metrics, pre_loss_metrics = _run_pre_post_metrics("before", disable_lora=True)
        pre_metrics = _write_split_predictions(pre_metrics, phase_name="before_train")
    trainer.train()
    if config.run_pre_post_train_eval:
        post_metrics, post_loss_metrics = _run_pre_post_metrics("after")
        post_metrics = _write_split_predictions(post_metrics, phase_name="after_train")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)
    if ctc_head is not None:
        import torch

        ctc_path = output_root / "ctc_head.pt"
        torch.save(ctc_head.state_dict(), ctc_path)
        (output_root / "ctc_vocab.json").write_text(
            json.dumps(ctc_vocab.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[ctc] Saved auxiliary CTC head to {ctc_path}")
    if config.run_pre_post_train_eval and pre_metrics and post_metrics:
        comparison = {
            "train_manifest": train_manifest_name,
            "synthetic_manifest": synthetic_manifest_name,
            "original_manifest": original_manifest_name,
            "test_manifest": test_manifest_name,
            "before_train": pre_metrics,
            "after_train": post_metrics,
            "before_train_loss_metrics": pre_loss_metrics,
            "after_train_loss_metrics": post_loss_metrics,
            "epoch_metrics": epoch_metrics,
        }
        metrics_path = output_root / "pre_post_train_eval_metrics.json"
        metrics_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] Wrote pre/post train metrics to: {metrics_path}")
    _export_loss_curves(trainer, output_dir=config.output_dir)
    config_path = getattr(config, "_path", None)
    if config_path:
        Path(config.output_dir, "train_config.yaml").write_text(
            Path(config_path).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML training config.")
    args = parser.parse_args()
    config = load_train_config(args.config)
    setattr(config, "_path", args.config)
    log_path = Path(config.output_dir) / "train.log"
    with _tee_stdout_stderr(log_path):
        print(f"[logging] Mirroring stdout/stderr to: {log_path}")
        train(config)


if __name__ == "__main__":
    main()
