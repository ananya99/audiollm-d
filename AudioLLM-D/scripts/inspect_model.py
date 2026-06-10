#!/usr/bin/env python
"""Inspect Qwen2-Audio model architecture using torchinfo and save results to disk."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/model_architecture",
        help="Directory to write summary text files.",
    )
    return parser.parse_args()


def dump_config(model_id: str, out_dir: Path) -> None:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    lines = [
        f"Model: {model_id}",
        "",
        "── Audio encoder (Whisper) ────────────────────────────────",
        f"  encoder_layers      : {cfg.audio_config.encoder_layers}",
        f"  d_model             : {cfg.audio_config.d_model}",
        f"  encoder_ffn_dim     : {cfg.audio_config.encoder_ffn_dim}",
        f"  encoder_attention_heads: {cfg.audio_config.encoder_attention_heads}",
        f"  num_mel_bins        : {cfg.audio_config.num_mel_bins}",
        f"  max_source_positions: {cfg.audio_config.max_source_positions}",
        "",
        "── LM backbone (Qwen2) ────────────────────────────────────",
        f"  num_hidden_layers   : {cfg.text_config.num_hidden_layers}",
        f"  hidden_size         : {cfg.text_config.hidden_size}",
        f"  intermediate_size   : {cfg.text_config.intermediate_size}",
        f"  num_attention_heads : {cfg.text_config.num_attention_heads}",
        f"  num_key_value_heads : {cfg.text_config.num_key_value_heads}",
        f"  vocab_size          : {cfg.text_config.vocab_size}",
        f"  max_position_embeddings: {cfg.text_config.max_position_embeddings}",
        "",
        "── Projection ─────────────────────────────────────────────",
        f"  audio hidden → LM hidden: {cfg.audio_config.d_model} → {cfg.text_config.hidden_size}",
        "",
        "── Full config ────────────────────────────────────────────",
        str(cfg),
    ]

    text = "\n".join(lines)
    print(text)
    (out_dir / "config.txt").write_text(text, encoding="utf-8")
    print(f"\nSaved → {out_dir / 'config.txt'}")


def dump_torchinfo(model_id: str, out_dir: Path) -> None:
    import torch
    from torchinfo import summary
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    dtype = torch.bfloat16

    print(f"Loading {model_id} …")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)  # noqa: F841
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=dtype,
    )
    model.eval()
    print("Model loaded.\n")

    sections: dict[str, str] = {}

    # ── Full model ────────────────────────────────────────────────
    stat = summary(model, depth=3, col_names=["num_params", "trainable"], verbose=0)
    sections["full_model"] = str(stat)

    # ── Audio encoder ─────────────────────────────────────────────
    stat = summary(model.audio_tower, depth=2, col_names=["num_params", "trainable"], verbose=0)
    sections["audio_encoder"] = str(stat)

    # ── Multi-modal projector (with dummy input for shapes) ───────
    d_model = model.config.audio_config.d_model
    proj_device = next(model.multi_modal_projector.parameters()).device
    dummy = torch.randn(1, 50, d_model, dtype=dtype, device=proj_device)
    stat = summary(
        model.multi_modal_projector,
        input_data=dummy,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,
    )
    sections["projector"] = str(stat)

    # ── LM backbone ───────────────────────────────────────────────
    stat = summary(model.language_model, depth=2, col_names=["num_params", "trainable"], verbose=0)
    sections["lm_backbone"] = str(stat)

    labels = {
        "full_model": "FULL MODEL  (depth=3)",
        "audio_encoder": "AUDIO ENCODER  (Whisper, depth=2)",
        "projector": "MULTI-MODAL PROJECTOR  (audio dim → LM dim)",
        "lm_backbone": "LM BACKBONE  (Qwen2, depth=2)",
    }

    for key, text in sections.items():
        header = f"{'=' * 60}\n{labels[key]}\n{'=' * 60}\n"
        full = header + text
        print(full)
        path = out_dir / f"{key}.txt"
        path.write_text(full, encoding="utf-8")
        print(f"Saved → {path}\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_config(args.model, out_dir)
    dump_torchinfo(args.model, out_dir)
    print("Done. All files written to:", out_dir.resolve())


if __name__ == "__main__":
    main()
