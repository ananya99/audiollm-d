#!/usr/bin/env python
"""Run audio-question answering inference over the hardcoded curated QA dataset."""

from __future__ import annotations

import argparse
import ast
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
OUTPUT_PATH = Path(
    f"/scratch/yasaman/qa_results/qa_predictions_{time_stamp}.json"
)

ANSWER_TYPE_FORMAT_TAGS = {
    "yes_no": "[yes/no — answer: yes or no only]",
    "multiple_choice": "[multiple choice — single letter A, B, C, or D only]",
    "count": "[count — single integer only]",
}

SYSTEM_PROMPT = """\
You are answering questions about a speech recording of a participant retelling a story \
or describing a scene.

Listen to the audio and answer each question using only what is audible in THIS recording. \
Each clip is different and short. Do not reuse answers from earlier clips or from prompt \
examples. Do not transcribe the entire clip. Do not infer diagnosis, demographics, or \
other details not directly supported by the audio.

Questions fall into two categories:
- content: what the participant says happened (characters, actions, objects, events)
- disfluency: how the participant speaks (fillers like uh/um, pauses, repetitions, \
non-verbal sounds, unclear speech)

═══ FORMAT RULES — FOLLOW EXACTLY ═══

yes/no questions:
  Output exactly one word: yes or no (lowercase, no punctuation, no explanation).
  Answer ONLY based on what you actually hear in this clip.
  Most clips do NOT contain most story elements. Default to "no" unless you clearly \
hear the word or concept spoken.

  RIGHT:
    Clip: "she's washing the floor"
    Q: Does the participant mention a fairy godmother? → no

    Clip: "that's the fairy godmother who uses magic on her"
    Q: Does the participant mention a fairy godmother? → yes

  WRONG:
    "The participant does not mention a fairy godmother." (sentence, not yes/no)
    "yes" when the concept is part of the story but not spoken in this clip

multiple choice questions:
  Output exactly one uppercase letter: A, B, C, or D.
  Never output the option text, a phrase, or a sentence.

  RIGHT: B
  WRONG: "B. She meets a prince"  /  "She meets a prince"

count questions:
  Output a single Arabic numeral derived by counting from this clip's audio.
  No words, no explanation, no sentence.

  RIGHT: 4
  WRONG: "The participant says uh four times."  /  "four"

═══ COUNTING RULES — DO NOT MIX CATEGORIES ═══

Count ONLY what the question names:
- "uh" or "um" → count audible filler tokens only (uh and um are separate tokens; \
count each occurrence)
- pauses → count pauses only when the question asks about pauses
- non-verbal sounds → count coughs, sighs, throat-clearing, and similar non-word \
vocalizations only
- unintelligible speech → count unclear-speech events only

Do not combine categories. A pause is not a filler. A non-verbal sound is not a filler.

═══ JSON OUTPUT RULES ═══

You will be given a numbered list of questions and an answer skeleton like this:
  [{"question_index": 0, "answer": ""}, {"question_index": 1, "answer": ""}, ...]

Rules:
1. Fill every empty answer in the skeleton. Do not skip any question_index.
2. The question_index in your output must match the skeleton exactly.
3. The answer format must match the tag on that specific question_index — not any other.
4. Return only a valid JSON array. No text before or after the array.
5. Do not stop early. Every object in the skeleton must appear in your output.
"""

GROUPED_QUESTION_PROMPT = """\
Answer all {question_count} numbered questions about THIS audio clip. You MUST return exactly \
{question_count} answers with question_index 0 through {last_question_index}.

{questions}

Each "answer" value must follow the tag on that question:
- yes/no → exactly "yes" or "no" (lowercase, no punctuation, no explanation)
- multiple choice → exactly one letter: A, B, C, or D (uppercase only; never the option text)
- count → a single integer string derived from the audio (for example, "4" or "7")

Every answer must be based on what you hear in this clip. Do not copy placeholder values, \
reuse one answer for multiple questions, or repeat answers from other clips.

Return only a valid JSON array. Fill in every empty answer in this skeleton from the audio:
{answer_skeleton}

You must include exactly one object for every question_index from 0 through {last_question_index}. \
Do not stop early. Do not include explanations outside the JSON array.
"""


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
    """Load JSONL QA rows and group questions by audio clip."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(path):
        audio_path = row.get("audio_path")
        if not audio_path:
            raise ValueError(f"Missing audio_path in QA row: {row}")
        grouped[str(audio_path)].append(row)

    records: list[dict] = []
    for audio_path, rows in grouped.items():
        clip_id = Path(audio_path).stem
        source_id = rows[0].get("source_id")
        records.append(
            {
                "id": clip_id,
                "audio_path": audio_path,
                "source_id": source_id,
                "transcript": rows[0].get("transcript"),
                "questions": [
                    {
                        "question": row["question"],
                        "answer": row["answer"],
                        "answer_type": row.get("answer_type", "unknown"),
                        "category": row.get("category"),
                    }
                    for row in rows
                ],
            }
        )
    return sorted(records, key=lambda record: record["id"])


def load_inputs(dataset_path: Path = QA_DATASET_PATH) -> list[dict]:
    return load_qa_dataset(dataset_path)


def format_question_line(question_index: int, question: dict) -> str:
    answer_type = str(question.get("answer_type") or "unknown").strip().lower()
    tag = ANSWER_TYPE_FORMAT_TAGS.get(answer_type, "[answer concisely]")
    return f"{question_index}. {tag} {question['question'].strip()}"


def build_grouped_question_text(questions: list[dict]) -> str:
    type_order = {"yes_no": 0, "multiple_choice": 1, "count": 2}
    sorted_questions = sorted(
        enumerate(questions),
        key=lambda x: type_order.get(x[1].get("answer_type", ""), 99)
    )
    numbered = "\n".join(
        format_question_line(display_idx, question)
        for display_idx, (orig_idx, question) in enumerate(sorted_questions)
    )
    answer_skeleton = json.dumps(
        [{"question_index": display_idx, "answer": ""}
         for display_idx in range(len(sorted_questions))],
        indent=2,
    )
    display_to_orig = {
        display_idx: orig_idx 
        for display_idx, (orig_idx, _) in enumerate(sorted_questions)
    }
    return GROUPED_QUESTION_PROMPT.format(
        questions=numbered,
        question_count=len(questions),
        last_question_index=len(questions) - 1,
        answer_skeleton=answer_skeleton,
    ), display_to_orig


def build_qa_conversation(questions: list[dict]) -> tuple[list[dict], dict[int, int]]:
    question_text, display_to_orig = build_grouped_question_text(questions)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "local_audio"},
                {"type": "text", "text": question_text},
            ],
        },
    ]
    return conversation, display_to_orig


def answer_questions_via_server(
    *,
    server_url: str,
    audio_path: str | Path,
    questions: list[dict],
    max_new_tokens: int,
) -> str:
    import urllib.error
    import urllib.request

    payload = json.dumps(
        {
            "audio_path": str(audio_path),
            "questions": questions,
            "max_new_tokens": max_new_tokens,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{server_url.rstrip('/')}/qa",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Inference server error ({exc.code}): {detail}") from exc
    if "error" in body:
        raise RuntimeError(f"Inference server error: {body['error']}")
    return str(body["raw_prediction"]).strip()


def answer_questions(
    *,
    audio_path: str | Path,
    questions: list[dict],
    model,
    processor,
    max_new_tokens: int,
    disfl_encoder=None,
    disfl_alpha: float | None = None,
) -> tuple[str, dict[int, int]]:
    conversation, display_to_orig = build_qa_conversation(questions)
    prompt = format_chat(processor, conversation, add_generation_prompt=True)
    sampling_rate = int(processor.feature_extractor.sampling_rate)
    audio = load_audio(audio_path, sampling_rate)
    inputs = processor(
        text=prompt, audio=[audio], sampling_rate=sampling_rate,
        return_tensors="pt", padding=True,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    generated_ids = generate_with_audio_enhancements(
        model,
        inputs,
        disfl_encoder=disfl_encoder,
        disfl_alpha=disfl_alpha,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    generated_ids = generated_ids[:, inputs["input_ids"].size(1):]
    result = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0].strip()
    print(f"[RAW OUTPUT] {result}", flush=True)
    return result, display_to_orig

def parse_grouped_answers(raw_prediction: str) -> dict[int, str]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw_prediction, flags=re.I).strip().rstrip("`").strip()
    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    candidate = match.group() if match else cleaned
    parsed = None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            pass

    answers = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict) or "question_index" not in item or "answer" not in item:
                continue
            try:
                question_index = int(item["question_index"])
            except (TypeError, ValueError):
                continue
            answers[question_index] = str(item["answer"]).strip()
        return answers

    # Some models emit Python-like objects but fail to escape quotes inside answers,
    # e.g. "answer': 'The speaker says 'uh' four times.'". Recover complete
    # answers using object boundaries instead of treating inner quotes as delimiters.
    tolerant_pattern = re.compile(
        r"[\"']question_index[\"']\s*:\s*(\d+)\s*,\s*"
        r"[\"']answer[\"']\s*:\s*(?P<quote>[\"'])"
        r"(?P<answer>.*?)(?P=quote)\s*}\s*(?=,\s*{|\s*])",
        flags=re.DOTALL,
    )
    for match in tolerant_pattern.finditer(candidate):
        answers[int(match.group(1))] = match.group("answer").strip()
    if answers:
        return answers
    if not answers:
        print(f"[PARSE FAILURE]\n{raw_prediction}\n[END PARSE FAILURE]", flush=True)
        raise ValueError("Could not recover indexed answers from model response.")
    return answers


def iter_clip_tasks(
    qa_records: list[dict],
    max_questions: int | None,
    clip_id: str | None = None,
):
    remaining = max_questions
    for record in qa_records:
        if clip_id is not None and record["id"] != clip_id:
            continue
        questions = record["questions"]
        if remaining is not None:
            if remaining <= 0:
                return
            questions = questions[:remaining]
            remaining -= len(questions)
        if questions:
            yield record, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Optional training YAML; reads audio_encoder_init_model and disfl_encoder config.",
    )
    parser.add_argument(
        "--disfl-encoder-checkpoint",
        type=Path,
        default=None,
        help="Checkpoint directory containing disfl_encoder.pt (defaults to --adapter).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--clip-id",
        default=None,
        help="Run inference only for this transcript/clip ID.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional global question limit. Questions remain grouped by clip.",
    )
    parser.add_argument(
        "--qa-dataset",
        type=Path,
        default=QA_DATASET_PATH,
        help="JSONL QA dataset with audio_path, question, answer, and answer_type fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Path to write per-question prediction rows as a JSON array.",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="Use a running inference server (from scripts/serve_inference.py) instead of loading the model.",
    )
    args = parser.parse_args()

    qa_records = load_inputs(args.qa_dataset)
    if args.clip_id is not None and args.clip_id not in {record["id"] for record in qa_records}:
        raise ValueError(f"Clip ID not found in QA dataset: {args.clip_id}")
    tasks = list(
        iter_clip_tasks(
            qa_records,
            args.max_questions,
            clip_id=args.clip_id,
        )
    )
    if not tasks:
        raise ValueError("No QA questions found.")

    total_questions = sum(len(questions) for _, questions in tasks)
    print(f"Grouped inference calls: {len(tasks)} for {total_questions} questions")
    loaded = None
    if args.server_url:
        print(f"Using inference server: {args.server_url}")
    else:
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
    args.output.parent.mkdir(parents=True, exist_ok=True)

    completed = 0
    errors = 0
    output_rows: list[dict] = []
    for task_index, (record, questions) in enumerate(tasks, 1):
        audio_path = record["audio_path"]
        print(f"[{task_index}/{len(tasks)}] {record['id']} ({len(questions)} questions)", flush=True)
        raw_prediction = ""
        group_error = None
        answers = {}
        display_to_orig = {i: i for i in range(len(questions))}
        try:
            if args.server_url:
                raw_prediction = answer_questions_via_server(
                    server_url=args.server_url,
                    audio_path=audio_path,
                    questions=questions,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                raw_prediction, display_to_orig = answer_questions(
                    audio_path=audio_path,
                    questions=questions,
                    model=loaded.model,
                    processor=loaded.processor,
                    max_new_tokens=args.max_new_tokens,
                    disfl_encoder=loaded.disfl_encoder,
                    disfl_alpha=loaded.disfl_alpha,
                )
            print(f"--- raw model output: {record['id']} ---")
            print(raw_prediction)
            print("--- end raw model output ---", flush=True)
            display_answers = parse_grouped_answers(raw_prediction)
            answers = {
                display_to_orig[display_idx]: answer
                for display_idx, answer in display_answers.items()
                if display_idx in display_to_orig
            }
        except Exception as exc:
            group_error = f"{type(exc).__name__}: {exc}"

        for question_index, question in enumerate(questions):
            prediction = answers.get(question_index, "")
            error = group_error
            if error is None and question_index not in answers:
                error = f"Missing answer for question_index {question_index}"
            if error is None:
                completed += 1
            else:
                errors += 1
            output_rows.append({
                "clip_id": record["id"],
                "source_id": record.get("source_id"),
                "audio_path": audio_path,
                "transcript": record.get("transcript"),
                "question_index": question_index,
                "question": question["question"],
                "category": question.get("category"),
                "answer_type": question.get("answer_type"),
                "gold_answer": question["answer"],
                "prediction": prediction,
                "raw_prediction": raw_prediction, 
                "error": error,
            })

    args.output.write_text(
        json.dumps(output_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {completed} predictions with {errors} errors to {args.output}")


if __name__ == "__main__":
    main()
