#!/usr/bin/env python
"""Run audio-question answering inference — one model call per question."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from audiollm_d.chat import format_chat
from audiollm_d.dataset import load_audio
from audiollm_d.inference import generate_with_audio_enhancements, load_for_inference


QA_DATASET_PATH = Path("/scratch/qa_dataset/qa_test_full.jsonl")

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = Path(f"/scratch/yasaman/qa_results/qa_predictions_{time_stamp}.json")

# SYSTEM_PROMPT = """\
# You are an expert in dementia speech. 
# Your job is to answer a single question about a speech recording of a person suffering from dementia speaking.
# Listen to the audio and answer the given question below.
# Be concise. Follow the format instruction exactly. 
# You must respond in English only.
# """

SYSTEM_PROMPT = """
You are an expert dementia speech analyst.

You will receive:
1. An audio recording of a person with dementia speaking.
2. A transcript generated from the recording.
3. A question about the recording.

Important instructions:

- Use BOTH the audio and transcript.
- The transcript may contain speech disfluencies, repetitions, fillers, false starts, pauses, word searches, and dementia-related speech phenomena.
- Treat these disfluencies as meaningful information when the question asks about them.
- For content questions, focus on the speaker's intended meaning rather than disfluencies.
- The transcript is only an aid and may contain recognition errors.
- Do not infer information that is not present in the audio or transcript.
- Be conservative and evidence-based.
- Follow the required output format exactly.
- Respond in English only.
"""

QUESTION_PROMPTS_TRANSCRIBE_THEN_ANSWER = {
    "yes_no": (
        "{question}\n\n"
        "First, transcribe what you hear in this recording exactly, preserving all "
        "fillers (uh, um), pauses, and unclear speech.\n"
        "Then, use your transcription as a guiding reference to answer the question.\n\n"
        "You MUST respond in exactly this two-line format:\n"
        "Transcription: <what you heard>\n"
        "Answer: yes or no\n\n"
        "Do NOT skip Transcription. Do NOT skip the Answer line.\n\n"
        "Example:\n"
        "Transcription: [uh] this is Cinderella. those are her two stepsisters.\n"
        "Answer: no\n\n"
        "Now start to transcribe and answer the question about the audio you just heard, DO NOT skip the answer:"
    ),
    "multiple_choice": (
        "{question}\n\n"
        "First, transcribe what you hear in this recording exactly, preserving all "
        "fillers (uh, um), pauses, and unclear speech.\n"
        "Then, use your transcription as a guiding reference to answer the question.\n\n"
        "You MUST respond in exactly this two-line format:\n"
        "Transcription: <what you heard>\n"
        "Answer: A, B, C, or D\n\n"
        "Do NOT skip Transcription. Do NOT skip the Answer line.\n\n"
        "Example:\n"
        "Transcription: and she's [um] whisked away with some horses.\n"
        "Answer: D\n\n"
        "Now start to transcribe and answer the question about the audio you just heard, DO NOT skip the answer:"
    ),
    "count": (
        "{question}\n\n"
        "First, transcribe what you hear in this recording exactly, preserving all "
        "fillers (uh, um), pauses, and unclear speech.\n"
        "Then, use your transcription as a guiding reference to answer the question.\n\n"
        "You MUST respond in exactly this two-line format:\n"
        "Transcription: <what you heard>\n"
        "Answer: a single integer\n\n"
        "Do NOT skip Transcription. Do NOT skip the Answer line.\n\n"
        "Example:\n"
        "Transcription: [uh] she visited by [um] someone [uh] who tells her [um] to go.\n"
        "Answer: 4\n\n"
        "Now start to transcribe and answer the question about the audio you just heard, DO NOT skip the answer:"
    ),
}

QUESTION_PROMPTS_COT = {
    "yes_no": (
        "{question}\n\n"
        "You MUST respond in exactly this two-line format and no other format:\n"
        "Reasoning: <1-2 sentences describing what you hear relevant to this question>\n"
        "Answer: yes or no\n\n"
        "Do NOT skip the Answer line.\n\n"
        "Example:\n"
        "Reasoning: The participant says 'uh' at the start and 'um' in the middle.\n"
        "Answer: yes\n\n"
        "Now respond for the audio you just heard:"
    ),
    "multiple_choice": (
        "{question}\n\n"
        "You MUST respond in exactly this two-line format and no other format:\n"
        "Reasoning: <1-2 sentences describing what you hear relevant to this question>\n"
        "Answer: A, B, C, or D\n\n"
        "Do NOT skip the Answer line.\n\n"
        "Example:\n"
        "Reasoning: The participant says she is whisked away with some horses.\n"
        "Answer: D"
        "Now respond for the audio you just heard:"
    ),
    "count": (
        "{question}\n\n"
        "You MUST respond in exactly this two-line format and no other format:\n"
        "Reasoning: <list each instance you hear, e.g. 'uh at the start, um in the middle'>\n"
        "Answer: a single integer\n\n"
        "Do NOT skip the Answer line.\n\n"
        "Example:\n"
        "Reasoning: uh at the start, um in the middle, uh near the end.\n"
        "Answer: 3"
        "Now respond for the audio you just heard:"
    ),
}

QUESTION_PROMPTS = {
    "yes_no": (
        "{question}\n\n"
        "Answer with exactly one word: yes or no (lowercase only).\n"
        "Answer 'no' unless you clearly hear the relevant word or concept spoken in this clip."
    ),
    "multiple_choice": (
        "{question}\n\n"
        "Answer with exactly one uppercase letter: A, B, C, or D.\n"
        "Do not include the option text."
    ),
    "count": (
        "{question}\n\n"
        "Answer with a single integer. Count only from this clip's audio.\n"
        "Do not include any words or explanation."
    ),
}

# QUESTION_PROMPTS_WITH_CONTEXT = {
#     "yes_no": (
#         "{question}\n\n"
#         "Here is a transcription of the audio to help you:\n"
#         "Transcription: {transcript}\n\n"
#         "Answer with exactly one word: yes or no (lowercase only).\n"
#     ),
#     "multiple_choice": (
#         "{question}\n\n"
#         "Here is a transcription of the audio to help you:\n"
#         "Transcription: {transcript}\n\n"
#         "Answer with exactly one uppercase letter: A, B, C, or D.\n"
#         "Do not include the option text."
#     ),
#     "count": (
#         "{question}\n\n"
#         "Here is a transcription of the audio to help you:\n"
#         "Transcription: {transcript}\n\n"
#         "Answer with a single integer. Count only from the transcription above.\n"
#         "Do not include any words or explanation."
#     ),
# }

QUESTION_PROMPTS_WITH_CONTEXT = {}

QUESTION_PROMPTS_WITH_CONTEXT["yes_no"] = """
Question:
{question}

Transcript:
{transcript}

Instructions:
- Use both the audio and transcript.
- Ignore filler words, hesitations, and repetitions unless they are relevant to the question.
- Answer only if supported by evidence in the recording.
- Output exactly one word:

yes
or
no
"""

QUESTION_PROMPTS_WITH_CONTEXT["multiple_choice"] = """
Question:
{question}

Transcript:
{transcript}

Instructions:
- Use both the audio and transcript.
- Consider the meaning of the speaker's utterances.
- Ignore irrelevant disfluencies.
- Select the option best supported by the recording.

Output exactly one uppercase letter:
A
B
C
or
D
"""

QUESTION_PROMPTS_WITH_CONTEXT["count"] = """
Question:
{question}

Transcript:
{transcript}

Instructions:
- Count only information directly observable in the recording.
- Use the transcript as evidence and verify with audio when possible.
- Do not estimate.
- If disfluency markers are being counted, count every explicit occurrence.
- If content items are being counted, count only distinct valid occurrences.

Output exactly one integer.
"""

DISFLUENCY_GUIDE = """
Special transcription markers may appear:

[short-pause],[medium-pause],[long-pause] : silent pauses of varying lengths
[uh],[um] : filler expressions 'um' and 'uh'
[laughter] : laughter sound
[non-verbal] : non-verbal vocal sound (cough, sniff, etc.)
[unclear-speech] : speech that is too unclear to transcribe

These markers are intentional annotations and should be treated as valid evidence when answering questions about speech behaviour.
"""

DEFAULT_PROMPT = "{question}\n\nAnswer concisely."

def extract_cot_answer(raw: str) -> str:
    """Extract answer from any format the model produces."""
    # 1. Multiline "Answer: X" on its own line
    match = re.search(r"(?i)^answer\s*:\s*(.+)$", raw, flags=re.MULTILINE)
    if match:
        return match.group(1).strip().strip("<>")

    # 2. Inline "Answer: X" anywhere in the text
    match = re.search(r"(?i)answer\s*:\s*(.+?)(?:\.|$)", raw)
    if match:
        return match.group(1).strip().strip("<>")

    # 3. Dict-like {'Answer': 'X'} or {"Answer": "X"}
    match = re.search(r"""['"]\s*[Aa]nswer\s*['"]\s*:\s*['"]\s*([^'"]+?)\s*['"]""", raw)
    if match:
        return match.group(1).strip().strip("<>")

    # 4. Plain answer with no reasoning — single short token
    stripped = raw.strip().strip("<>").rstrip(".")
    if len(stripped.split()) <= 4:
        return stripped

    # 5. No answer found
    return ""

def load_finetuned_transcriptions(path: Path) -> dict[str, str]:
    """Load fine-tuned model transcriptions keyed by audio_path."""
    transcriptions = {}
    text = path.read_text(encoding="utf-8").strip()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        audio_path = row.get("audio_path")
        prediction = row.get("prediction", "").strip()
        if audio_path and prediction:
            transcriptions[str(audio_path)] = prediction
    return transcriptions

def read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return rows
    rows: list[dict] = []
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        row, end = decoder.raw_decode(text, idx)
        rows.append(row)
        idx = end
    return rows


def load_qa_dataset(path: Path) -> list[dict]:
    """Load JSONL QA rows — one record per question, no grouping."""
    records = []
    for row in read_jsonl(path):
        audio_path = row.get("audio_path")
        if not audio_path:
            raise ValueError(f"Missing audio_path in QA row: {row}")
        records.append({
            "clip_id": Path(audio_path).stem,
            "source_id": row.get("source_id"),
            "audio_path": audio_path,
            "transcript": row.get("transcript"),
            "question": row["question"],
            "gold_answer": row["answer"],
            "answer_type": row.get("answer_type", "unknown"),
            "category": row.get("category"),
        })
    return sorted(records, key=lambda r: (r["clip_id"], r["answer_type"]))


def build_single_question_conversation(
    question: str,
    answer_type: str,
    *,
    use_cot: bool = False,
    transcribe_first: bool = False,
    context_transcript: str | None = None,
) -> list[dict]:
    if transcribe_first:
        prompts = QUESTION_PROMPTS_TRANSCRIBE_THEN_ANSWER
    elif context_transcript is not None:
        prompts = QUESTION_PROMPTS_WITH_CONTEXT
    elif use_cot:
        prompts = QUESTION_PROMPTS_COT
    else:
        prompts = QUESTION_PROMPTS

    template = prompts.get(answer_type, DEFAULT_PROMPT)

    if context_transcript is not None:
        prompt_text = DISFLUENCY_GUIDE + "\n" + template.format(
            question=question, transcript=context_transcript
        )
    else:
        prompt_text = template.format(question=question)

    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "local_audio"},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    if transcribe_first:
        conversation.append({"role": "assistant", "content": "Transcription:"})
    return conversation


def answer_single_question(
    *,
    audio_path: str | Path,
    question: str,
    answer_type: str,
    model,
    processor,
    max_new_tokens: int,
    disfl_encoder=None,
    disfl_alpha: float | None = None,
    use_cot: bool = False,
    transcribe_first: bool = False,
    context_transcript: str | None = None,
) -> tuple[str, str]:
    conversation = build_single_question_conversation(
        question,
        answer_type,
        use_cot=use_cot,
        transcribe_first=transcribe_first,
        context_transcript=context_transcript,
    )
    add_generation_prompt = not transcribe_first
    prompt = format_chat(processor, conversation, add_generation_prompt=add_generation_prompt)
    # print(f"--- PROMPT ---\n{prompt}\n--- END PROMPT ---", flush=True)
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    audio = load_audio(audio_path, sampling_rate)
    inputs = processor(
        text=prompt, audio=[audio], sampling_rate=sampling_rate,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    generated_ids = generate_with_audio_enhancements(
        model,
        inputs,
        disfl_encoder=disfl_encoder,
        disfl_alpha=disfl_alpha,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
    raw = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    prediction = extract_cot_answer(raw) if (use_cot or transcribe_first) else raw
    return raw, prediction

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--audio-encoder-init-model", default=None)
    parser.add_argument("--train-config", type=Path, default=None)
    parser.add_argument("--disfl-encoder-checkpoint", type=Path, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens-cot", type=int, default=512)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--qa-dataset", type=Path, default=QA_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--chain-of-thought",
        action="store_true",
        help="Use chain-of-thought prompting and extract final answer from response.",
    )
    parser.add_argument(
        "--transcribe-first",
        action="store_true",
        help="Ask model to transcribe audio then use its transcription to answer.",
    )
    parser.add_argument(
        "--finetuned-transcriptions",
        type=Path,
        default=None,
        help="JSONL file with fine-tuned model transcriptions to use as context.",
    )
    parser.add_argument(
        "--use-gold-transcripts",
        action="store_true",
        help="Use gold transcripts from the QA dataset as context.",
    )
    args = parser.parse_args()

    if args.chain_of_thought or args.transcribe_first:
        effective_max_new_tokens = args.max_new_tokens_cot
    else:
        effective_max_new_tokens = args.max_new_tokens

    records = load_qa_dataset(args.qa_dataset)

    if args.clip_id is not None:
        records = [r for r in records if r["clip_id"] == args.clip_id]
        if not records:
            raise ValueError(f"Clip ID not found: {args.clip_id}")

    if args.max_questions is not None:
        records = records[:args.max_questions]

    gold_transcriptions: dict[str, str] = {
        record["audio_path"]: record["transcript"]
        for record in records
        if record.get("transcript")
    }

    print(f"Loading model: {args.model}")
    loaded = load_for_inference(
        args.model,
        args.adapter,
        audio_encoder_init_model=args.audio_encoder_init_model,
        train_config_path=args.train_config,
        disfl_encoder_checkpoint=args.disfl_encoder_checkpoint,
    )
    if loaded.disfl_encoder is not None:
        print(f"Loaded DisfluencyEncoder (alpha={loaded.disfl_alpha:.4f})")

    finetuned_transcriptions: dict[str, str] = {}
    if args.finetuned_transcriptions is not None:
        finetuned_transcriptions = load_finetuned_transcriptions(args.finetuned_transcriptions)
        print(f"Loaded {len(finetuned_transcriptions)} fine-tuned transcriptions")

    if args.transcribe_first:
        mode = "transcribe-first"
    elif args.chain_of_thought:
        mode = "chain-of-thought"
    elif args.finetuned_transcriptions:
        mode = "finetuned-transcript-context"
    elif args.use_gold_transcripts:
        mode = "gold-transcript-context"
    else:
        mode = "direct"
    print(f"Running {mode} inference on {len(records)} questions")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []
    for i, record in enumerate(records, 1):
        print(
            f"[{i}/{len(records)}] {record['clip_id']} | {record['answer_type']} | "
            f"{record['question'][:60]}",
            flush=True,
        )
        raw_prediction = ""
        prediction = ""
        error = None
        context_transcript = None

        if args.finetuned_transcriptions is not None:
            context_transcript = finetuned_transcriptions.get(str(record["audio_path"]))
            if context_transcript is None:
                print(f"  WARNING: no transcription found for {record['audio_path']}", flush=True)
        elif args.use_gold_transcripts:
            context_transcript = gold_transcriptions.get(str(record["audio_path"]))

        try:
            raw_prediction, prediction = answer_single_question(
                audio_path=record["audio_path"],
                question=record["question"],
                answer_type=record["answer_type"],
                model=loaded.model,
                processor=loaded.processor,
                max_new_tokens=effective_max_new_tokens,
                disfl_encoder=loaded.disfl_encoder,
                disfl_alpha=loaded.disfl_alpha,
                use_cot=args.chain_of_thought,
                transcribe_first=args.transcribe_first,
                context_transcript=context_transcript,
            )
            print(f"  gold={record['gold_answer']!r}  pred={prediction!r}", flush=True)
            if args.chain_of_thought or args.transcribe_first:
                print(f"  reasoning={raw_prediction!r}", flush=True)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {error}", flush=True)

        output_rows.append({
            "clip_id": record["clip_id"],
            "source_id": record.get("source_id"),
            "audio_path": record["audio_path"],
            "transcript": record.get("transcript"),
            "context_transcript": context_transcript,
            "question": record["question"],
            "category": record.get("category"),
            "answer_type": record["answer_type"],
            "gold_answer": record["gold_answer"],
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "error": error,
        })

    args.output.write_text(
        json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(output_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()