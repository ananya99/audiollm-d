"""Question-type-aware metrics for audio QA evaluation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from enum import Enum
from typing import Any


class MetricKind(str, Enum):
    COUNT = "count"
    YES_NO = "yes_no"
    MULTIPLE_CHOICE = "multiple_choice"
    SHORT_PHRASE = "short_phrase"

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

def _parse_metric_kind(metric_kind: str | MetricKind) -> MetricKind:
    if isinstance(metric_kind, MetricKind):
        return metric_kind
    return MetricKind(str(metric_kind).strip().lower())


def parse_yes_no(text: str) -> str | None:
    cleaned = _normalize_whitespace(text.lower())
    if not cleaned:
        return None
    if re.match(r"^(yes|yeah|yep)\b", cleaned):
        return "yes"
    if re.match(r"^no\b", cleaned):
        return "no"
    if re.search(r"\bone or more\b|\bat least one\b|\bseveral\b|\bmultiple\b", cleaned):
        return "yes"
    if re.search(r"\bdoes not\b|\bdo not\b|\bdid not\b|\bdoesn't\b|\bdon't\b|\bdidn't\b|\bwithout\b|\bnone\b|\bzero\b", cleaned):
        return "no"
    if re.search(r"\b(yes|yeah|yep)\b", cleaned):
        return "yes"
    if re.search(r"\bno\b", cleaned):
        return "no"
    return None


def parse_multiple_choice(text: str) -> str | None:
    """Return A, B, C, or D when a choice letter can be inferred."""
    cleaned = text.strip().upper()
    if re.fullmatch(r"[ABCD]", cleaned):
        return cleaned
    match = re.search(r"\b([ABCD])\b", cleaned)
    return match.group(1) if match else None


def evaluate_multiple_choice(gold_answer: str, prediction: str) -> dict[str, Any]:
    gold_choice = parse_multiple_choice(gold_answer)
    pred_choice = parse_multiple_choice(prediction)
    correct = (
        gold_choice is not None
        and pred_choice is not None
        and gold_choice == pred_choice
    )
    return {
        "metric_kind": MetricKind.MULTIPLE_CHOICE.value,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "gold_choice": gold_choice,
        "pred_choice": pred_choice,
    }


def evaluate_yes_no(gold_answer: str, prediction: str) -> dict[str, Any]:
    gold_label = parse_yes_no(gold_answer)
    pred_label = parse_yes_no(prediction)
    correct = gold_label is not None and pred_label is not None and gold_label == pred_label
    return {
        "metric_kind": MetricKind.YES_NO.value,
        "correct": correct,
        "score": 1.0 if correct else 0.0,
        "gold_label": gold_label,
        "pred_label": pred_label,
    }


def extract_count(text: str) -> int | None:
    """Extract the first explicit count from a count-style answer."""
    cleaned = _normalize_whitespace(text.lower())
    if not cleaned:
        return None

    digit_match = re.search(r"\b(\d+)\b", cleaned)
    if digit_match:
        return int(digit_match.group(1))

    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", cleaned):
            return value
    return None


def _count_score(gold_count: int | None, pred_count: int | None) -> float:
    if gold_count is None or pred_count is None:
        return 0.0

    diff = abs(gold_count - pred_count)
    return max(0.0, 1.0 - diff / max(abs(gold_count), 1))


def evaluate_count(gold_answer: str, prediction: str) -> dict[str, Any]:
    gold_count = extract_count(gold_answer)
    pred_count = extract_count(prediction)
    correct = gold_count is not None and pred_count is not None and gold_count == pred_count
    off_by_one = (
        gold_count is not None
        and pred_count is not None
        and abs(gold_count - pred_count) <= 1
    )
    diff = abs(gold_count - pred_count) if gold_count is not None and pred_count is not None else None
    sq_err = diff ** 2 if diff is not None else None
    return {
        "metric_kind": MetricKind.COUNT.value,
        "correct": correct,
        "off_by_one": off_by_one,
        "score": _count_score(gold_count, pred_count),
        "gold_count": gold_count,
        "pred_count": pred_count,
        "count_difference": diff,
        "squared_error": sq_err,
    }


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split()).strip()


def _strip_punctuation(text: str) -> str:
    return re.sub(r"[^\w\s'-]", " ", text)


def normalize_answer_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    return _normalize_whitespace(_strip_punctuation(text.lower()))


def extract_quoted_phrases(text: str) -> list[str]:
    """Pull quoted spans from an answer, preserving order."""
    phrases: list[str] = []
    for pattern in (r'"([^"]+)"', r"'([^']+)'"):
        phrases.extend(re.findall(pattern, text))
    return [_normalize_whitespace(phrase) for phrase in phrases if phrase.strip()]


def evaluate_short_phrase(gold_answer: str, prediction: str) -> dict[str, Any]:
    gold_phrases = extract_quoted_phrases(gold_answer)
    gold_text = normalize_answer_text(gold_phrases[0] if gold_phrases else gold_answer)
    pred_text = normalize_answer_text(prediction)

    exact_match = gold_text == pred_text and bool(gold_text)
    contains = bool(gold_text) and gold_text in pred_text
    reverse_contains = bool(pred_text) and pred_text in gold_text
    correct = exact_match or contains
    score = 1.0 if exact_match else (0.75 if contains or reverse_contains else 0.0)
    return {
        "metric_kind": MetricKind.SHORT_PHRASE.value,
        "correct": correct,
        "score": score,
        "exact_match": exact_match,
        "contains": contains or reverse_contains,
        "gold_phrase": gold_text,
        "pred_phrase": pred_text,
    }


_EVALUATORS = {
    MetricKind.COUNT: evaluate_count,
    MetricKind.YES_NO: evaluate_yes_no,
    MetricKind.MULTIPLE_CHOICE: evaluate_multiple_choice,
    MetricKind.SHORT_PHRASE: evaluate_short_phrase,
}

def evaluate_qa_pair(
    *,
    gold_answer: str,
    prediction: str,
    metric_kind: str | MetricKind,
    question_type: str | None = None,
) -> dict[str, Any]:
    """Score one prediction against its gold answer using the provided metric kind."""
    kind = _parse_metric_kind(metric_kind)
    result = _EVALUATORS[kind](gold_answer, prediction)
    if question_type is not None:
        result["question_type"] = question_type
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _aggregate_bucket(
    evaluated_rows: Sequence[dict[str, Any]],
    *,
    num_empty: int = 0,
    metric_kind: str | None = None,
) -> dict[str, Any]:
    scores = [float(row["score"]) for row in evaluated_rows]
    correct = [bool(row["correct"]) for row in evaluated_rows]
    payload = {
        "total": len(evaluated_rows) + num_empty,
        "num_empty": num_empty,
        "num_evaluated": len(evaluated_rows),
        "count": len(evaluated_rows),
        "accuracy": _mean([1.0 if v else 0.0 for v in correct]),
        "mean_score": _mean(scores),
    }
    if metric_kind is not None:
        payload["metric_kind"] = metric_kind

    if metric_kind == MetricKind.COUNT.value:
        off_by_one = [row["off_by_one"] for row in evaluated_rows if "off_by_one" in row]
        sq_errors = [row["squared_error"] for row in evaluated_rows
                     if row.get("squared_error") is not None]
        abs_errors = [row["count_difference"] for row in evaluated_rows
                      if row.get("count_difference") is not None]
        payload["off_by_one_accuracy"] = _mean([1.0 if v else 0.0 for v in off_by_one])
        payload["mse"] = _mean(sq_errors) if sq_errors else None
        payload["mae"] = _mean(abs_errors) if abs_errors else None

    return payload


def evaluate_predictions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate a batch of QA prediction rows and aggregate by type and metric."""
    evaluated_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    empty_by_type: dict[str, int] = defaultdict(int)
    empty_by_metric_kind: dict[str, int] = defaultdict(int)
    empty_by_category: dict[str, int] = defaultdict(int)

    for row in rows:
        prediction = str(row.get("prediction", "")).strip()
        gold_answer = str(row.get("gold_answer") or row.get("answer", "")).strip()
        metric_kind = row.get("metric_kind")
        question_type = str(row.get("type") or row.get("answer_type", "unknown")).strip()
        category = str(row.get("category") or "unknown").strip()

        if not prediction:
            skipped_rows.append({"reason": "empty_prediction", "row": row})
            empty_by_type[question_type] += 1
            empty_by_category[category] += 1
            if metric_kind:
                empty_by_metric_kind[str(metric_kind)] += 1
            continue
        if not gold_answer:
            skipped_rows.append({"reason": "empty_gold_answer", "row": row})
            continue
        if not metric_kind:
            skipped_rows.append({"reason": "missing_metric_kind", "row": row})
            continue

        result = evaluate_qa_pair(
            gold_answer=gold_answer,
            prediction=prediction,
            metric_kind=metric_kind,
            question_type=question_type,
        )
        evaluated_rows.append(
            {
                "clip_id": row.get("clip_id"),
                "question_index": row.get("question_index"),
                "question": row.get("question"),
                "category": category,
                "type": question_type,
                "gold_answer": gold_answer,
                "prediction": prediction,
                **result,
            }
        )

    scores = [float(row["score"]) for row in evaluated_rows]
    correct = [bool(row["correct"]) for row in evaluated_rows]
    num_empty = sum(empty_by_category.values())

    by_type: dict[str, dict[str, Any]] = {}
    type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        type_groups[str(row["type"])].append(row)

    all_types = sorted(set(type_groups) | set(empty_by_type))
    for question_type in all_types:
        group = type_groups.get(question_type, [])
        by_type[question_type] = _aggregate_bucket(
            group,
            num_empty=empty_by_type.get(question_type, 0),
            metric_kind=group[0]["metric_kind"] if group else None,
        )

    by_metric_kind: dict[str, dict[str, Any]] = {}
    metric_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        metric_groups[str(row["metric_kind"])].append(row)

    all_metric_kinds = sorted(set(metric_groups) | set(empty_by_metric_kind))
    for group_key in all_metric_kinds:
        group = metric_groups.get(group_key, [])
        by_metric_kind[group_key] = _aggregate_bucket(
            group,
            num_empty=empty_by_metric_kind.get(group_key, 0),
            metric_kind=group_key,
        )

    by_category: dict[str, dict[str, Any]] = {}
    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in evaluated_rows:
        category_groups[str(row["category"])].append(row)

    all_categories = sorted(set(category_groups) | set(empty_by_category))
    for category in all_categories:
        group = category_groups.get(category, [])
        by_category[category] = _aggregate_bucket(
            group,
            num_empty=empty_by_category.get(category, 0),
        )

    return {
        "num_input_rows": len(rows),
        "num_evaluated": len(evaluated_rows),
        "num_empty_predictions": num_empty,
        "num_skipped": len(skipped_rows),
        "overall_accuracy": _mean([1.0 if value else 0.0 for value in correct]),
        "overall_mean_score": _mean(scores),
        "by_type": by_type,
        "by_metric_kind": by_metric_kind,
        "by_category": by_category,
        "rows": evaluated_rows,
        "skipped": skipped_rows,
    }
