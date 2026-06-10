#!/usr/bin/env python
"""Transcribe dementia speech with disfluency special tokens injected into the prompt.

Registers disfluency tokens into the tokenizer vocabulary, builds an
enhanced prompt that lists every token and its meaning, runs inference, then annotates
the output to show which special tokens were detected.

Usage
-----
python scripts/transcribe.py \\
    --audio sample.wav \\
    --model Qwen/Qwen2-Audio-7B-Instruct \\
    [--adapter outputs/qwen2-audio-dementia-lora] \\
    [--max-new-tokens 256]
"""

from __future__ import annotations

import argparse
import re
import textwrap
from collections import Counter
from pathlib import Path

from audiollm_d.chat_parser import (
    COUGH_TOKEN,
    LAUGHTER_TOKEN,
    LONG_PAUSE_TOKEN,
    MEDIUM_PAUSE_TOKEN,
    NOISE_TOKEN,
    SHORT_PAUSE_TOKEN,
    SIGH_TOKEN,
    UNINTELLIGIBLE_TOKEN,
    VOCAL_NOISE_TOKEN,
)
from audiollm_d.special_tokens import SPECIAL_TOKENS

# Matches "TRANSCRIPTION:" or "TRANSCRIPT:" with any capitalisation/spacing.
_PREFIX_RE = re.compile(r"TRANSCRI(?:PTION|PT)\s*:\s*", re.IGNORECASE)
def _extract_transcription(raw: str) -> str:
    """Strip the TRANSCRIPTION: prefix the model is asked to emit."""
    m = _PREFIX_RE.search(raw)
    return raw[m.end():].strip() if m else raw.strip()


# ── prompt construction ───────────────────────────────────────────────────────

_PAUSE_DESCRIPTIONS = {
    SHORT_PAUSE_TOKEN: "short pause  (0.3–1.0 s)",
    MEDIUM_PAUSE_TOKEN: "medium pause (1.0–2.5 s)",
    LONG_PAUSE_TOKEN: "long pause   (>2.5 s)",
}
_EVENT_TOKENS = [LAUGHTER_TOKEN, SIGH_TOKEN, COUGH_TOKEN, VOCAL_NOISE_TOKEN, NOISE_TOKEN]
_TOKEN_NAMES = {
    SHORT_PAUSE_TOKEN: "short_pause",
    MEDIUM_PAUSE_TOKEN: "medium_pause",
    LONG_PAUSE_TOKEN: "long_pause",
    UNINTELLIGIBLE_TOKEN: "indecipherable",
    LAUGHTER_TOKEN: "laughter",
    SIGH_TOKEN: "sigh",
    COUGH_TOKEN: "cough",
    VOCAL_NOISE_TOKEN: "vocal_noise",
    NOISE_TOKEN: "noise",
}


def _build_token_legend() -> str:
    """Return a human-readable legend of every special token for use in the prompt."""
    lines: list[str] = []

    lines.append("Pause markers:")
    for token, description in _PAUSE_DESCRIPTIONS.items():
        lines.append(f"  {token:<14} {description}")

    lines.append("Unintelligible speech:")
    lines.append(f"  {UNINTELLIGIBLE_TOKEN:<14} speech that cannot be understood")

    lines.append("Filler words:")
    lines.append("  uh  um")

    lines.append("Non-verbal events:")
    event_tokens = "  " + "  ".join(_EVENT_TOKENS)
    lines.append(textwrap.fill(event_tokens, width=80, subsequent_indent="  "))

    return "\n".join(lines)


def _build_few_shot_examples() -> str:
    """Return few-shot transcription examples that demonstrate correct token usage."""
    examples = [
        ("[short example: filler + pause]",
         f"the uh {SHORT_PAUSE_TOKEN} cookie is falling off the stool"),
        ("[short example: repetition + mixed pauses]",
         f"she's she's {MEDIUM_PAUSE_TOKEN} washing the dishes and uh {SHORT_PAUSE_TOKEN} the water is over"),
        ("[full multi-sentence transcription — transcribe the ENTIRE clip like this]",
         f"um the the {SHORT_PAUSE_TOKEN} boy is {LONG_PAUSE_TOKEN} trying to get cookies "
         f"{SHORT_PAUSE_TOKEN} for uh himself {MEDIUM_PAUSE_TOKEN} and also one for his sister "
         f"{SHORT_PAUSE_TOKEN} {UNINTELLIGIBLE_TOKEN} his sister is telling him "
         f"{SHORT_PAUSE_TOKEN} to be very quiet {SHORT_PAUSE_TOKEN} she's going shh"),
    ]

    lines = []
    for label, content in examples:
        lines.append(f"  {label}")
        lines.append(f"  TRANSCRIPTION: {content}")
    return "\n".join(lines)

# TODO: Add flags to use baseline, zero-shot or few shot prompt variants. 

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a clinical speech transcription assistant specialised in dementia speech.
    Transcribe the participant's speech exactly, preserving all disfluencies using the
    special tokens listed below. Do not paraphrase or omit any spoken content.
    Return exactly one line in this format: TRANSCRIPTION: {content}
    Do not add explanations, notes, or extra lines.

    Disfluency token legend:
""") + textwrap.indent(_build_token_legend(), "    ") + textwrap.dedent("""

    Few-shot examples (correct output format):
""") + _build_few_shot_examples()

_INSTRUCTION = textwrap.dedent("""\
    Transcribe this dementia speech sample using the disfluency tokens from the system prompt.
    Write filled pauses as plain words: uh and um.
    Insert (.) (..) (...) wherever you hear a silence — do not omit them.
    Mark every unintelligible word as xxx.
    Output EXACTLY one line in this format: TRANSCRIPTION: {content}\
""")


# ── output annotation ─────────────────────────────────────────────────────────

def annotate_output(transcription: str) -> str:
    """Return a summary of which special tokens appear in the transcription."""
    counts = Counter(tok for tok in SPECIAL_TOKENS if tok in transcription)
    if not counts:
        return "  (none)"
    lines = [
        f"  {tok:<20} → {_TOKEN_NAMES[tok]:<20} [{n}×]"
        for tok, n in sorted(counts.items(), key=lambda x: SPECIAL_TOKENS.index(x[0]))
    ]
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="Path to audio file.")
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct",
                        help="Base model name or local path.")
    parser.add_argument("--adapter", default=None,
                        help="Optional LoRA adapter directory (fine-tuned checkpoint).")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.adapter and not Path(args.adapter).exists():
        parser.error(
            f"Adapter path does not exist: {args.adapter}\n"
            "Either train a LoRA adapter first, or omit --adapter to run with the base model."
        )

    print("Loading model …")
    from audiollm_d.inference import load_for_inference, transcribe_with_loaded_model

    # // Loads the model and processor, and registers special tokens in the tokenizer.
    loaded = load_for_inference(args.model, args.adapter)

    print("── Prompt (system) ──────────────────────────────────────────────────────────")
    print(_SYSTEM_PROMPT)
    print()

    print("── Running inference ────────────────────────────────────────────────────────")
    result = transcribe_with_loaded_model(
        audio_path=args.audio,
        model=loaded.model,
        processor=loaded.processor,
        max_new_tokens=args.max_new_tokens,
        system_prompt=_SYSTEM_PROMPT,
        instruction=_INSTRUCTION,
        disfl_encoder=loaded.disfl_encoder,
        disfl_alpha=loaded.disfl_alpha,
    )

    print()
    transcription = _extract_transcription(result)

    print("── Result ────────────────────────────────────────────────────────────")
    print(result)
    print()
    print("── Transcription ────────────────────────────────────────────────────────────")
    print(transcription)
    print()
    print("── Special tokens detected ──────────────────────────────────────────────────")
    print(annotate_output(transcription))


if __name__ == "__main__":
    main()
