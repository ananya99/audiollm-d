"""Metrics helpers for transcription evaluation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from jiwer import wer


def compute_wer(references: Sequence[str], predictions: Sequence[str]) -> float:
    """Compute corpus WER for paired reference and predicted transcripts."""
    if len(references) != len(predictions):
        raise ValueError(
            f"references and predictions must have same length, got {len(references)} and {len(predictions)}"
        )
    return wer(list(references), list(predictions))


def extract_disfluency_token_counts(text: str, disfluency_tokens: Sequence[str]) -> Counter[str]:
    """Count occurrences of each special token in a transcript string."""
    counts = Counter()
    for token in disfluency_tokens:
        n = text.count(token)
        if n > 0:
            counts[token] = n
    return counts


def _tokenize_for_alignment(text: str) -> list[str]:
    return text.split()


def _compute_precision_recall_f1(tp: int, pred_total: int, ref_total: int) -> tuple[float, float, float]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / ref_total if ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def _align_token_sequences(ref_tokens: Sequence[str], pred_tokens: Sequence[str]) -> list[tuple[str | None, str | None]]:
    """Global token alignment via edit-distance DP (Levenshtein backtrace)."""
    n = len(ref_tokens)
    m = len(pred_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref_tokens[i - 1] == pred_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # delete from ref
                dp[i][j - 1] + 1,  # insert into ref (from pred)
                dp[i - 1][j - 1] + sub_cost,  # substitute / match
            )

    aligned: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = 0 if ref_tokens[i - 1] == pred_tokens[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
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


def compute_order_aware_disfluency_token_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    disfluency_tokens: Sequence[str],
) -> dict[str, Any]:
    """Compute order-aware disfluency precision/recall/F1 using token alignment."""
    if len(references) != len(predictions):
        raise ValueError(
            f"references and predictions must have same length, got {len(references)} and {len(predictions)}"
        )

    special_set = set(disfluency_tokens)
    tp = 0
    fp = 0
    fn = 0
    per_token_tp = Counter()
    per_token_pred = Counter()
    per_token_ref = Counter()

    for ref, pred in zip(references, predictions, strict=True):
        ref_tokens = _tokenize_for_alignment(ref)
        pred_tokens = _tokenize_for_alignment(pred)
        aligned_pairs = _align_token_sequences(ref_tokens, pred_tokens)

        for ref_tok, pred_tok in aligned_pairs:
            ref_label = ref_tok if ref_tok in special_set else None
            pred_label = pred_tok if pred_tok in special_set else None

            if ref_label is None and pred_label is None:
                continue
            if ref_label is not None and pred_label is not None:
                if ref_label == pred_label:
                    tp += 1
                    per_token_tp[ref_label] += 1
                    per_token_pred[pred_label] += 1
                    per_token_ref[ref_label] += 1
                else:
                    # Wrong disfluency type at this aligned position.
                    fp += 1
                    fn += 1
                    per_token_pred[pred_label] += 1
                    per_token_ref[ref_label] += 1
                continue
            if pred_label is not None:
                fp += 1
                per_token_pred[pred_label] += 1
            if ref_label is not None:
                fn += 1
                per_token_ref[ref_label] += 1

    pred_total = tp + fp
    ref_total = tp + fn
    precision, recall, f1 = _compute_precision_recall_f1(tp, pred_total, ref_total)

    per_token_order_aware_stats: dict[str, dict[str, float | int]] = {}
    for token in disfluency_tokens:
        token_tp = int(per_token_tp.get(token, 0))
        token_pred_total = int(per_token_pred.get(token, 0))
        token_ref_total = int(per_token_ref.get(token, 0))
        token_precision, token_recall, token_f1 = _compute_precision_recall_f1(
            token_tp, token_pred_total, token_ref_total
        )
        per_token_order_aware_stats[token] = {
            "tp": token_tp,
            "pred_total": token_pred_total,
            "ref_total": token_ref_total,
            "precision": token_precision,
            "recall": token_recall,
            "f1": token_f1,
        }

    return {
        "disfluency_token_order_tp": tp,
        "disfluency_token_order_pred_total": pred_total,
        "disfluency_token_order_ref_total": ref_total,
        "disfluency_token_order_precision": precision,
        "disfluency_token_order_recall": recall,
        "disfluency_token_order_f1": f1,
        "per_token_order_aware_stats": per_token_order_aware_stats,
    }


def compute_per_token_stats(
    references: Sequence[str],
    predictions: Sequence[str],
    disfluency_tokens: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """Compute per-token precision/recall/F1 from corpus-level token counts."""
    if len(references) != len(predictions):
        raise ValueError(
            f"references and predictions must have same length, got {len(references)} and {len(predictions)}"
        )

    per_token_ref = Counter()
    per_token_pred = Counter()
    per_token_tp = Counter()

    for ref, pred in zip(references, predictions, strict=True):
        ref_counts = extract_disfluency_token_counts(ref, disfluency_tokens)
        pred_counts = extract_disfluency_token_counts(pred, disfluency_tokens)
        per_token_ref.update(ref_counts)
        per_token_pred.update(pred_counts)
        for token in disfluency_tokens:
            per_token_tp[token] += min(ref_counts.get(token, 0), pred_counts.get(token, 0))

    stats: dict[str, dict[str, float | int]] = {}
    for token in disfluency_tokens:
        tp = int(per_token_tp.get(token, 0))
        pred_total = int(per_token_pred.get(token, 0))
        ref_total = int(per_token_ref.get(token, 0))
        precision, recall, f1 = _compute_precision_recall_f1(tp, pred_total, ref_total)
        stats[token] = {
            "tp": tp,
            "pred_total": pred_total,
            "ref_total": ref_total,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return stats


def compute_disfluency_token_metrics(
    references: Sequence[str],
    predictions: Sequence[str],
    disfluency_tokens: Sequence[str],
) -> dict[str, Any]:
    """Compute corpus-level metrics that target special-token transcription quality."""
    if len(references) != len(predictions):
        raise ValueError(
            f"references and predictions must have same length, got {len(references)} and {len(predictions)}"
        )

    corpus_tp = 0
    corpus_pred_total = 0
    corpus_ref_total = 0
    exact_set_matches = 0
    any_presence_matches = 0

    for ref, pred in zip(references, predictions, strict=True):
        ref_counts = extract_disfluency_token_counts(ref, disfluency_tokens)
        pred_counts = extract_disfluency_token_counts(pred, disfluency_tokens)
        ref_set = set(ref_counts.keys())
        pred_set = set(pred_counts.keys())

        if ref_set == pred_set:
            exact_set_matches += 1
        if bool(ref_set) == bool(pred_set):
            any_presence_matches += 1

        corpus_ref_total += sum(ref_counts.values())
        corpus_pred_total += sum(pred_counts.values())
        for token in disfluency_tokens:
            corpus_tp += min(ref_counts.get(token, 0), pred_counts.get(token, 0))

    precision = corpus_tp / corpus_pred_total if corpus_pred_total else 0.0
    recall = corpus_tp / corpus_ref_total if corpus_ref_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    num_samples = len(references)
    exact_set_accuracy = exact_set_matches / num_samples if num_samples else 0.0
    any_disfluency_token_accuracy = any_presence_matches / num_samples if num_samples else 0.0
    hallucination_rate = (
        (corpus_pred_total - corpus_tp) / corpus_pred_total if corpus_pred_total else 0.0
    )
    miss_rate = (corpus_ref_total - corpus_tp) / corpus_ref_total if corpus_ref_total else 0.0
    order_aware = compute_order_aware_disfluency_token_metrics(references, predictions, disfluency_tokens)

    return {
        "num_samples": num_samples,
        "disfluency_token_precision": precision,
        "disfluency_token_recall": recall,
        "disfluency_token_f1": f1,
        "disfluency_token_exact_set_accuracy": exact_set_accuracy,
        "any_disfluency_token_accuracy": any_disfluency_token_accuracy,
        "disfluency_token_hallucination_rate": hallucination_rate,
        "disfluency_token_miss_rate": miss_rate,
        "disfluency_token_tp": corpus_tp,
        "disfluency_token_pred_total": corpus_pred_total,
        "disfluency_token_ref_total": corpus_ref_total,
        "per_token_stats": compute_per_token_stats(references, predictions, disfluency_tokens),
        **order_aware,
    }


def bootstrap_disfluency_metrics_ci(
    references: Sequence[str],
    predictions: Sequence[str],
    disfluency_tokens: Sequence[str],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Clip-level bootstrap CIs for WER and order-aware disfluency F1/P/R (ticket 2.5).

    Resamples the (reference, prediction) clip pairs with replacement ``n_boot`` times and
    recomputes each metric, returning the point estimate plus the percentile ``[alpha/2,
    1-alpha/2]`` interval. The point estimate is the metric on the full (unresampled) set.
    Reused for both the Phase 1 winner and Phase 2 rows so the comparison reports identical
    intervals.
    """
    import random as _random

    references = list(references)
    predictions = list(predictions)
    n = len(references)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty prediction set.")

    def _metrics(refs, preds):
        order = compute_order_aware_disfluency_token_metrics(refs, preds, disfluency_tokens)
        return {
            "wer": compute_wer(refs, preds),
            "edit_f1": order["disfluency_token_order_f1"],
            "edit_precision": order["disfluency_token_order_precision"],
            "edit_recall": order["disfluency_token_order_recall"],
        }

    point = _metrics(references, predictions)
    samples: dict[str, list[float]] = {k: [] for k in point}
    rng = _random.Random(seed)
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        refs = [references[i] for i in idx]
        preds = [predictions[i] for i in idx]
        for key, value in _metrics(refs, preds).items():
            samples[key].append(value)

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    out: dict[str, dict[str, float]] = {}
    for key, point_value in point.items():
        ordered = sorted(samples[key])
        lo = ordered[max(0, int(lo_q * (n_boot - 1)))]
        hi = ordered[min(n_boot - 1, int(hi_q * (n_boot - 1)))]
        out[key] = {"point": float(point_value), "lo": float(lo), "hi": float(hi)}
    return out
