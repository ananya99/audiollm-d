"""Prompt construction for Qwen2-Audio transcription fine-tuning and inference."""

from __future__ import annotations

TRAIN_SYSTEM_PROMPT = (
    "You are a clinical speech transcription assistant. Transcribe the participant's "
    "speech exactly, preserving fillers, repetitions, false starts, and explicit pause tokens. "
    "Do not add any explanation, notes, or extra lines."
)

TRAIN_TRANSCRIPTION_INSTRUCTION = (
    "Transcribe this dementia speech sample. Preserve lexical disfluencies such as um, uh, "
    "repetitions, abandoned words, and pause tokens if present."
)

INFERENCE_SYSTEM_PROMPT = (
    "You are a clinical speech transcription assistant. Transcribe the participant's "
    "speech exactly, preserving fillers, repetitions, false starts, and explicit pause tokens. "
    "Return exactly one line in this format: TRANSCRIPTION: {content}. "
    "Do not add any explanation, notes, or extra lines."
)

INFERENCE_TRANSCRIPTION_INSTRUCTION = (
    "Transcribe this dementia speech sample. Preserve lexical disfluencies such as um, uh, "
    "repetitions, abandoned words, and pause tokens if present. "
    "Output format must be: TRANSCRIPTION: {content}"
)


def build_transcription_conversation(
    text: str | None = None,
    *,
    instruction: str = TRAIN_TRANSCRIPTION_INSTRUCTION,
    system_prompt: str = TRAIN_SYSTEM_PROMPT,
) -> list[dict]:
    """Build the chat conversation expected by Qwen2-Audio-Instruct.

    If ``text`` is omitted, the conversation ends after the user prompt and is suitable for
    generation. If ``text`` is provided, it includes the assistant target for supervised training.
    """

    conversation: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "local_audio"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    if text is not None:
        conversation.append({"role": "assistant", "content": text})
    return conversation


def format_chat(processor, conversation: list[dict], *, add_generation_prompt: bool) -> str:
    """Apply the model chat template with a small compatibility fallback."""

    try:
        return processor.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
    except TypeError:
        tokenizer = getattr(processor, "tokenizer", processor)
        return tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )
