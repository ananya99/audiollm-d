"""Align reference disfluency misses to greedy decode steps and their probabilities."""

from __future__ import annotations

from typing import Any

from audiollm_d.special_tokens import DISFLUENCY_TOKENS

SPECIAL_SET = set(DISFLUENCY_TOKENS)


def align_sequences(ref_tokens: list[str], pred_tokens: list[str]) -> list[tuple[str | None, str | None]]:
    """Global token alignment via Levenshtein backtrace."""
    n, m = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == pred_tokens[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    aligned: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref_tokens[i - 1] == pred_tokens[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                aligned.append((ref_tokens[i - 1], pred_tokens[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            aligned.append((ref_tokens[i - 1], None))
            i -= 1
            continue
        aligned.append((None, pred_tokens[j - 1]))
        j -= 1
    aligned.reverse()
    return aligned


def build_word_end_step_map(trace: dict[str, Any]) -> list[int]:
    """Map each word index in decoded_text.split() to the decode step that finished it."""
    words = trace["decoded_text"].split()
    steps = trace["steps"]
    word_end_steps: list[int] = []
    w_idx = 0
    for step in steps:
        partial = "".join(item["chosen_token"] for item in steps[: step["step"] + 1])
        completed = partial.split()
        while w_idx < len(words) and w_idx < len(completed) and completed[w_idx] == words[w_idx]:
            word_end_steps.append(step["step"])
            w_idx += 1
    return word_end_steps


def _miss_step_window_bounds(
    *,
    pred_index: int | None,
    word_end_steps: list[int],
    num_steps: int,
) -> tuple[int, int]:
    """Decode-step range where a skipped reference token could have been emitted."""
    if not word_end_steps:
        return 0, max(num_steps - 1, 0)

    if pred_index is None or pred_index <= 0:
        start = 0
    else:
        prev_end = word_end_steps[pred_index - 1]
        start = min(prev_end + 1, num_steps - 1)

    if pred_index is not None and pred_index < len(word_end_steps):
        end = max(start, word_end_steps[pred_index] - 1)
    else:
        end = word_end_steps[-1]

    return start, min(end, num_steps - 1)


def _best_step_for_missed_token(
    trace: dict[str, Any],
    *,
    ref_tok: str,
    start: int,
    end: int,
) -> int:
    """Pick the step in [start, end] with highest softmax prob for ref_tok."""
    best_step = start
    best_prob = -1.0
    for step in trace["steps"]:
        idx = step["step"]
        if start <= idx <= end:
            prob = step["disfluency_probs"][ref_tok]
            if prob > best_prob:
                best_prob = prob
                best_step = idx
    return best_step


def _decode_step_for_miss(
    *,
    ref_tok: str,
    pred_tok: str | None,
    pred_index: int | None,
    word_end_steps: list[int],
    num_steps: int,
    trace: dict[str, Any],
) -> tuple[int, int, int]:
    """Return (decode_step, window_start, window_end) for a missed reference token."""
    if not word_end_steps:
        return 0, 0, max(num_steps - 1, 0)

    if pred_tok is None:
        window_start, window_end = _miss_step_window_bounds(
            pred_index=pred_index,
            word_end_steps=word_end_steps,
            num_steps=num_steps,
        )
        decode_step = _best_step_for_missed_token(
            trace,
            ref_tok=ref_tok,
            start=window_start,
            end=window_end,
        )
        return decode_step, window_start, window_end

    if pred_tok is not None and pred_index is not None and pred_index < len(word_end_steps):
        step = word_end_steps[pred_index]
        return step, step, step

    fallback = min(word_end_steps[-1], num_steps - 1)
    return fallback, fallback, fallback


def find_disfluency_misses(reference: str, trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Find reference disfluency tokens that were not emitted and link them to decode steps."""
    ref_tokens = reference.split()
    pred_tokens = trace["decoded_text"].split()
    word_end_steps = build_word_end_step_map(trace)
    aligned = align_sequences(ref_tokens, pred_tokens)

    misses: list[dict[str, Any]] = []
    ref_index = 0
    pred_index = 0

    for ref_tok, pred_tok in aligned:
        if ref_tok in SPECIAL_SET and pred_tok != ref_tok:
            if pred_tok is None:
                miss_type = "skipped"
            elif pred_tok in SPECIAL_SET:
                miss_type = "wrong_type"
            else:
                miss_type = "substituted"

            decode_step, window_start, window_end = _decode_step_for_miss(
                ref_tok=ref_tok,
                pred_tok=pred_tok,
                pred_index=pred_index if pred_tok is not None else pred_index,
                word_end_steps=word_end_steps,
                num_steps=trace["num_steps"],
                trace=trace,
            )
            step_data = trace["steps"][decode_step] if decode_step is not None else None

            misses.append(
                {
                    "audio_path": trace["audio_path"],
                    "ref_token": ref_tok,
                    "pred_token": pred_tok,
                    "miss_type": miss_type,
                    "ref_index": ref_index,
                    "pred_index": pred_index if pred_tok is not None else pred_index,
                    "decode_step": decode_step,
                    "window_start": window_start,
                    "window_end": window_end,
                    "chosen_token": step_data["chosen_token"] if step_data else None,
                    "chosen_prob": step_data["chosen_prob"] if step_data else None,
                    "prob_missed_token": step_data["disfluency_probs"][ref_tok] if step_data else None,
                    "logit_missed_token": step_data["disfluency_logits"][ref_tok] if step_data else None,
                    "max_disfluency_token": step_data["max_disfluency_token"] if step_data else None,
                    "max_disfluency_prob": step_data["max_disfluency_prob"] if step_data else None,
                    "any_disfluency_prob": step_data["any_disfluency_prob"] if step_data else None,
                    "all_disfluency_probs": step_data["disfluency_probs"] if step_data else None,
                }
            )

        if ref_tok is not None:
            ref_index += 1
        if pred_tok is not None:
            pred_index += 1

    return misses


def collect_all_misses(
    traces: list[dict[str, Any]],
    references: dict[str, str],
) -> list[dict[str, Any]]:
    all_misses: list[dict[str, Any]] = []
    for trace in traces:
        reference = references.get(trace["audio_path"], trace.get("reference", ""))
        all_misses.extend(find_disfluency_misses(reference, trace))
    return all_misses


def miss_step_window(trace: dict[str, Any], decode_step: int, radius: int = 2) -> list[dict[str, Any]]:
    """Return decode steps in [decode_step-radius, decode_step+radius] with probabilities."""
    rows = []
    for step in trace["steps"]:
        if decode_step - radius <= step["step"] <= decode_step + radius:
            rows.append(step)
    return rows
