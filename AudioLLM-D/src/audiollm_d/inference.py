"""Run transcription with a base or LoRA-adapted Qwen2-Audio model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from audiollm_d.chat import (
    INFERENCE_SYSTEM_PROMPT,
    INFERENCE_TRANSCRIPTION_INSTRUCTION,
    build_transcription_conversation,
    format_chat,
)
from audiollm_d.config import DisfluencyEncoderConfig, load_train_config
from audiollm_d.dataset import load_audio
from audiollm_d.model_setup import (
    disfluency_encoder_hook,
    load_disfluency_encoder_for_inference,
    replace_qwen_audio_encoder_from_whisper,
)
from audiollm_d.special_tokens import SPECIAL_TOKENS, register_tokens_with_init


@dataclass
class LoadedModel:
    model: Any
    processor: Any
    disfl_encoder: Any | None = None
    disfl_alpha: float | None = None


def _register_special_tokens(model, processor) -> None:
    new_ids = register_tokens_with_init(model, processor, SPECIAL_TOKENS)
    print(f"Special tokens: {len(SPECIAL_TOKENS)} total, {len(new_ids)} registered.")


def _strip_framework_special_tokens(text: str, tokenizer) -> str:
    """Remove all tokenizer special tokens except the CHAT disfluency tokens."""
    to_remove = set(tokenizer.all_special_tokens) - set(SPECIAL_TOKENS)
    for tok in sorted(to_remove, key=len, reverse=True):
        text = text.replace(tok, "")
    return text.strip()


def resolve_inference_setup(
    *,
    audio_encoder_init_model: str | None = None,
    adapter_path: str | Path | None = None,
    train_config_path: str | Path | None = None,
    disfl_encoder_checkpoint: str | Path | None = None,
    disfl_encoder_config: DisfluencyEncoderConfig | None = None,
) -> dict[str, Any]:
    """Resolve encoder-init and optional DisfluencyEncoder paths from CLI/config."""
    resolved_audio_encoder_init = audio_encoder_init_model
    resolved_disfl_config = disfl_encoder_config
    resolved_disfl_checkpoint = (
        Path(disfl_encoder_checkpoint) if disfl_encoder_checkpoint is not None else None
    )

    if train_config_path is not None:
        train_config = load_train_config(train_config_path)
        if resolved_audio_encoder_init is None:
            resolved_audio_encoder_init = train_config.audio_encoder_init_model
        if resolved_disfl_config is None:
            resolved_disfl_config = train_config.disfl_encoder

    if adapter_path is not None:
        adapter_dir = Path(adapter_path)
        if resolved_disfl_checkpoint is None and (adapter_dir / "disfl_encoder.pt").exists():
            resolved_disfl_checkpoint = adapter_dir

    return {
        "audio_encoder_init_model": resolved_audio_encoder_init,
        "disfl_encoder_checkpoint": resolved_disfl_checkpoint,
        "disfl_encoder_config": resolved_disfl_config,
    }


def load_for_inference(
    model_name_or_path: str,
    adapter_path: str | Path | None = None,
    *,
    audio_encoder_init_model: str | None = None,
    train_config_path: str | Path | None = None,
    disfl_encoder_checkpoint: str | Path | None = None,
    disfl_encoder_config: DisfluencyEncoderConfig | None = None,
    use_bf16: bool | None = None,
) -> LoadedModel:
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    setup = resolve_inference_setup(
        audio_encoder_init_model=audio_encoder_init_model,
        adapter_path=adapter_path,
        train_config_path=train_config_path,
        disfl_encoder_checkpoint=disfl_encoder_checkpoint,
        disfl_encoder_config=disfl_encoder_config,
    )
    use_bf16 = torch.cuda.is_available() if use_bf16 is None else use_bf16

    processor_path = adapter_path or model_name_or_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    )

    if setup["audio_encoder_init_model"]:
        replace_qwen_audio_encoder_from_whisper(
            model,
            source_model_name=setup["audio_encoder_init_model"],
            use_bf16=use_bf16,
        )

    if adapter_path:
        from peft import PeftModel

        target_vocab_size = len(processor.tokenizer)
        current_vocab_size = model.get_input_embeddings().weight.shape[0]
        if current_vocab_size != target_vocab_size:
            print(
                "Resizing base model token embeddings "
                f"from {current_vocab_size} to {target_vocab_size} before loading adapter."
            )
            model.resize_token_embeddings(target_vocab_size)
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    _register_special_tokens(model, processor)

    disfl_encoder = None
    disfl_alpha = None
    if setup["disfl_encoder_checkpoint"] is not None:
        disfl_encoder, disfl_alpha = load_disfluency_encoder_for_inference(
            checkpoint_dir=setup["disfl_encoder_checkpoint"],
            model=model,
            disfl_encoder_config=setup["disfl_encoder_config"],
        )

    return LoadedModel(
        model=model,
        processor=processor,
        disfl_encoder=disfl_encoder,
        disfl_alpha=disfl_alpha,
    )


def generate_with_audio_enhancements(
    model,
    inputs: dict[str, Any],
    *,
    disfl_encoder=None,
    disfl_alpha: float | None = None,
    **generate_kwargs,
):
    import torch

    with disfluency_encoder_hook(
        model=model,
        input_features=inputs.get("input_features"),
        disfl_encoder=disfl_encoder,
        disfl_alpha=disfl_alpha if disfl_alpha is not None else 0.01,
    ):
        with torch.inference_mode():
            return model.generate(**inputs, **generate_kwargs)


def transcribe_with_loaded_model(
    *,
    audio_path: str | Path,
    model,
    processor,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    system_prompt: str | None = None,
    instruction: str | None = None,
    disfl_encoder=None,
    disfl_alpha: float | None = None,
) -> str:
    return transcribe_batch_with_loaded_model(
        audio_paths=[audio_path],
        model=model,
        processor=processor,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        system_prompt=system_prompt,
        instruction=instruction,
        disfl_encoder=disfl_encoder,
        disfl_alpha=disfl_alpha,
    )[0]


def transcribe_batch_with_loaded_model(
    *,
    audio_paths: list[str | Path],
    model,
    processor,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    system_prompt: str | None = None,
    instruction: str | None = None,
    disfl_encoder=None,
    disfl_alpha: float | None = None,
) -> list[str]:
    if not audio_paths:
        return []

    conversation = build_transcription_conversation(
        text=None,
        system_prompt=system_prompt or INFERENCE_SYSTEM_PROMPT,
        instruction=instruction or INFERENCE_TRANSCRIPTION_INSTRUCTION,
    )
    prompt = format_chat(processor, conversation, add_generation_prompt=True)
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    audios = [load_audio(path, sampling_rate) for path in audio_paths]
    inputs = processor(
        text=[prompt] * len(audios),
        audio=audios,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generate_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature

    generated_ids = generate_with_audio_enhancements(
        model,
        inputs,
        disfl_encoder=disfl_encoder,
        disfl_alpha=disfl_alpha,
        **generate_kwargs,
    )
    generated_ids = generated_ids[:, inputs["input_ids"].size(1) :]
    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return [_strip_framework_special_tokens(text, processor.tokenizer) for text in decoded]


def transcribe(
    *,
    audio_path: str | Path,
    model_name_or_path: str,
    adapter_path: str | None = None,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    system_prompt: str | None = None,
    instruction: str | None = None,
    audio_encoder_init_model: str | None = None,
    train_config_path: str | Path | None = None,
) -> str:
    loaded = load_for_inference(
        model_name_or_path,
        adapter_path,
        audio_encoder_init_model=audio_encoder_init_model,
        train_config_path=train_config_path,
    )
    return transcribe_with_loaded_model(
        audio_path=audio_path,
        model=loaded.model,
        processor=loaded.processor,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        system_prompt=system_prompt,
        instruction=instruction,
        disfl_encoder=loaded.disfl_encoder,
        disfl_alpha=loaded.disfl_alpha,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument(
        "--audio-encoder-init-model",
        default=None,
        help="Whisper checkpoint used to initialize audio_tower during training.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Optional training YAML; reads audio_encoder_init_model automatically.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--instruction", default=None)
    args = parser.parse_args()
    print(
        transcribe(
            audio_path=args.audio,
            model_name_or_path=args.model,
            adapter_path=args.adapter,
            max_new_tokens=args.max_new_tokens,
            system_prompt=args.system_prompt,
            instruction=args.instruction,
            audio_encoder_init_model=args.audio_encoder_init_model,
            train_config_path=args.train_config,
        )
    )


if __name__ == "__main__":
    main()
