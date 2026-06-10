"""Load the configured Qwen2-Audio PEFT model and print trainable parameters."""

from __future__ import annotations

import argparse
from collections import Counter

from audiollm_d.config import load_train_config
from audiollm_d.train import load_model_and_processor


def _bucket_parameter(name: str) -> str:
    lowered = name.lower()
    if "lora_" in lowered or "dora" in lowered:
        return "peft_adapter"
    if "modules_to_save" in lowered:
        return "modules_to_save"
    if "embed_tokens" in lowered:
        return "embed_tokens"
    if "lm_head" in lowered:
        return "lm_head"
    if "multi_modal_projector" in lowered:
        return "multi_modal_projector"
    if "audio" in lowered or "encoder" in lowered:
        return "audio_stack"
    if "language_model" in lowered or "model.layers" in lowered:
        return "language_model"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/qwen2_audio_dora.yaml",
        help="Training YAML config used to construct the PEFT model.",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Print only parameter names, without shapes/counts.",
    )
    args = parser.parse_args()

    config = load_train_config(args.config)
    model, _processor, _disfluency_token_ids = load_model_and_processor(config)

    print("=== CHECKING ACTIVE DORA/LORA MODULES ===")
    trainable_total = 0
    total_params = 0
    trainable_count = 0
    bucket_counts: Counter[str] = Counter()

    for name, param in model.named_parameters():
        total_params += param.numel()
        if not param.requires_grad:
            continue

        trainable_count += 1
        trainable_total += param.numel()
        bucket_counts[_bucket_parameter(name)] += param.numel()

        if args.names_only:
            print(f"Trainable layer: {name}")
        else:
            shape = tuple(param.shape)
            print(f"Trainable layer: {name} | shape={shape} | params={param.numel():,}")

    pct = (100.0 * trainable_total / total_params) if total_params else 0.0
    print("=== SUMMARY ===")
    print(f"Trainable tensors: {trainable_count:,}")
    print(f"Trainable params: {trainable_total:,}")
    print(f"Total params: {total_params:,}")
    print(f"Trainable percent: {pct:.6f}%")
    print("Trainable params by bucket:")
    for bucket, count in bucket_counts.most_common():
        print(f"  {bucket}: {count:,}")


if __name__ == "__main__":
    main()
