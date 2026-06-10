#!/usr/bin/env python
"""Few-shot baseline inference with Qwen2-Audio-7B-Instruct."""

import argparse
import json
import textwrap
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# ── Token definitions ─────────────────────────────────────────────────────────

SHORT_PAUSE_TOKEN    = "[short-pause]"
MEDIUM_PAUSE_TOKEN   = "[medium-pause]"
LONG_PAUSE_TOKEN     = "[long-pause]"
UNCLEAR_SPEECH_TOKEN = "[unclear-speech]"
LAUGHTER_TOKEN       = "[laughter]"
NON_VERBAL_TOKEN     = "[non-verbal]"
FILLER_UH_TOKEN      = "[uh]"
FILLER_UM_TOKEN      = "[um]"

SPECIAL_TOKENS: list[str] = [
    SHORT_PAUSE_TOKEN,
    MEDIUM_PAUSE_TOKEN,
    LONG_PAUSE_TOKEN,
    UNCLEAR_SPEECH_TOKEN,
    LAUGHTER_TOKEN,
    NON_VERBAL_TOKEN,
    FILLER_UH_TOKEN,
    FILLER_UM_TOKEN,
]
DISFLUENCY_TOKENS: list[str] = SPECIAL_TOKENS

# ── Prompt ────────────────────────────────────────────────────────────────────

def _build_token_legend() -> str:
    return "\n".join([
        "Pause markers:",
        f"  {SHORT_PAUSE_TOKEN:<20} short pause  (0.3-1.0 s)",
        f"  {MEDIUM_PAUSE_TOKEN:<20} medium pause (1.0-2.5 s)",
        f"  {LONG_PAUSE_TOKEN:<20} long pause   (>2.5 s)",
        "Unintelligible speech:",
        f"  {UNCLEAR_SPEECH_TOKEN:<20} speech that cannot be understood",
        "Filler words:",
        f"  {FILLER_UH_TOKEN:<20} filler uh",
        f"  {FILLER_UM_TOKEN:<20} filler um",
        "Non-verbal events:",
        f"  {LAUGHTER_TOKEN:<20} laughter or chuckling",
        f"  {NON_VERBAL_TOKEN:<20} sighs, coughs, throat-clearing, vocal noises",
    ])


def _build_few_shot_examples() -> str:
    sp  = SHORT_PAUSE_TOKEN
    mp  = MEDIUM_PAUSE_TOKEN
    lp  = LONG_PAUSE_TOKEN
    xxx = UNCLEAR_SPEECH_TOKEN
    lau = LAUGHTER_TOKEN
    nv  = NON_VERBAL_TOKEN
    uh  = FILLER_UH_TOKEN   # now "[uh]"
    um  = FILLER_UM_TOKEN   # now "[um]"

    examples = [
        ("[filler + short pause]",
         f"the {uh} {sp} cookie is falling off the stool"),
        ("[repetition + mixed pauses]",
         f"she's she's {mp} washing the dishes and {uh} {sp} the water is over"),
        ("[long pause + unclear speech + non-verbal]",
         f"{um} {lp} I can see {xxx} and the boy is {sp} reaching up {nv}"),
        ("[full multi-sentence clip — transcribe the ENTIRE clip like this]",
         f"{um} the the {sp} boy is {lp} trying to get cookies {sp} for {uh} himself "
         f"{mp} and also one for his sister {sp} {xxx} his sister is telling him "
         f"{sp} to be very quiet {sp} she's going shh {lau}"),
    ]

    lines = []
    for label, content in examples:
        lines.append(f"  [{label}]")
        lines.append(f"  TRANSCRIPTION: {content}")
    return "\n".join(lines)


SYSTEM_PROMPT = (
    textwrap.dedent("""\
        You are a clinical speech transcription assistant specialised in dementia speech.
        Transcribe the participant's speech exactly, preserving all disfluencies using the
        special tokens listed below. Do not paraphrase or omit any spoken content.
        Return exactly one line in this format: TRANSCRIPTION: <your transcription here>
        Do not add explanations, notes, or extra lines.

        Disfluency token legend:
    """)
    + textwrap.indent(_build_token_legend(), "    ")
    + textwrap.dedent("""

        Few-shot examples (correct output format):
    """)
    + _build_few_shot_examples()
)

INSTRUCTION = textwrap.dedent("""\
    Transcribe this dementia speech sample using the disfluency tokens from the system prompt.
    Use [uh] for filler uh and [um] for filler um - NOT plain words.
    Insert [short-pause] [medium-pause] [long-pause] for silences of the right duration.
    Mark unintelligible speech as [unclear-speech].
    Use [laughter] for laughing, [non-verbal] for coughs, sighs, throat-clearing.
    Output EXACTLY one line: TRANSCRIPTION: <your transcription here>\
""")

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: Path) -> list[dict]:
    records = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_audio_path(audio_path: str, data_root: Path) -> Path:
    p = Path(audio_path)
    if p.is_absolute() and p.exists():
        return p
    resolved = data_root / p
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Audio not found: {audio_path} (tried {resolved})")


def load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    audio, sr = sf.read(str(path))
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(model_name: str):
    print(f"Loading {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    allocated = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  device : {next(model.parameters()).device}")
    print(f"  VRAM   : {allocated:.1f} GB")
    return model, processor


def transcribe(audio: np.ndarray, model, processor, max_new_tokens: int = 512) -> str:
    """Single audio — few-shot examples are in the system prompt as text only."""
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "local"},
                {"type": "text",  "text": INSTRUCTION},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    sr = int(processor.feature_extractor.sampling_rate)
    inputs = processor(
        text=prompt, audio=[audio], sampling_rate=sr,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated, skip_special_tokens=False, clean_up_tokenization_spaces=False,
    )[0].strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",       default="/scratch/merged_data_80_20/train/manifest.jsonl")
    parser.add_argument("--data-root",      default="/scratch")
    parser.add_argument("--output",         default="/scratch/yasaman/results/few_shot_predictions_train.jsonl")
    parser.add_argument("--model",          default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-samples",    type=int, default=None)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_manifest(Path(args.manifest))
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Loaded {len(records)} records")
    if not records:
        print("No records found."); return

    model, processor = load_model(args.model)

    with open(output_path, "w") as out_f:
        for i, row in enumerate(records):
            clip_id = row.get("clip_id", str(i))
            print(f"[{i+1}/{len(records)}] {clip_id}", end=" ... ", flush=True)
            try:
                audio_path = resolve_audio_path(row["audio_path"], Path(args.data_root))
                raw = transcribe(load_audio(audio_path), model, processor, args.max_new_tokens)
            except FileNotFoundError as e:
                print(f"SKIP — {e}"); continue
            except Exception as e:
                print(f"ERROR — {e}"); raw = f"ERROR: {e}"

            out_f.write(json.dumps({
                "clip_id": clip_id, "source_id": row.get("source_id"),
                "audio_path": str(audio_path), "text": row["text"],
                "raw_prediction": raw, "split": row.get("split"),
                "diagnosis": row.get("diagnosis"), "mmse": row.get("mmse"),
                "age": row.get("age"), "sex": row.get("sex"),
            }, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"done  |  ref: {row['text'][:60]}...")

    print(f"\n Saved → {output_path}")


if __name__ == "__main__":
    main()