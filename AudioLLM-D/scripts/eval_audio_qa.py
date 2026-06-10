#!/usr/bin/env python
"""Evaluate audio QA predictions against gold answers using question-type-aware metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audiollm_d.qa_metrics import evaluate_predictions


DEFAULT_PREDICTIONS_PATH = Path(
    "/scratch/yasaman/qa_results/qa_predictions.json"
)

ANSWER_TYPE_TO_METRIC_KIND = {
    "count": "count",
    "yes_no": "yes_no",
    "multiple_choice": "multiple_choice",
}


def read_predictions(path: Path) -> list[dict]:
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
    if rows:
        return rows

    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric_kind_for_row(row: dict) -> str | None:
    if row.get("metric_kind"):
        return str(row["metric_kind"])
    answer_type = str(row.get("answer_type") or row.get("type", "")).strip().lower()
    return ANSWER_TYPE_TO_METRIC_KIND.get(answer_type)


def prepare_prediction_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Attach metric_kind to prediction rows before evaluation."""
    prepared: list[dict] = []
    skipped: list[dict] = []
    for row in rows:
        metric_kind = metric_kind_for_row(row)
        if metric_kind is None:
            skipped.append(
                {
                    "reason": "unknown_answer_type",
                    "answer_type": row.get("answer_type") or row.get("type"),
                    "row": row,
                }
            )
            continue
        prepared.append({**row, "metric_kind": metric_kind})
    return prepared, skipped


def _row_key(row: dict) -> tuple:
    return (row.get("clip_id"), row.get("question_index"))


def annotate_predictions_with_evaluation(
    rows: list[dict],
    metrics: dict,
    *,
    pre_skipped: list[dict] | None = None,
) -> list[dict]:
    """Attach per-row evaluation verdicts to the original prediction rows."""
    evaluations: dict[tuple, dict] = {}

    for evaluated in metrics.get("rows", []):
        evaluations[_row_key(evaluated)] = {
            "correct": bool(evaluated["correct"]),
            "score": float(evaluated["score"]),
        }

    for skipped in metrics.get("skipped", []):
        row = skipped["row"]
        evaluations[_row_key(row)] = {
            "correct": None,
            "score": None,
            "skipped": True,
            "reason": skipped["reason"],
        }

    for skipped in pre_skipped or []:
        row = skipped["row"]
        evaluations[_row_key(row)] = {
            "correct": None,
            "score": None,
            "skipped": True,
            "reason": skipped["reason"],
        }

    annotated: list[dict] = []
    for row in rows:
        updated = {key: value for key, value in row.items() if key != "evaluation"}
        evaluation = evaluations.get(_row_key(row))
        if evaluation is not None:
            updated["evaluation"] = evaluation
        annotated.append(updated)
    return annotated


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _print_summary(metrics: dict, *, pre_skipped: list[dict] | None = None) -> None:
    total_skipped = metrics["num_skipped"] + (len(pre_skipped) if pre_skipped else 0)
    print(f"Evaluated {metrics['num_evaluated']} / {metrics['num_input_rows']} rows")
    print(f"Empty predictions: {metrics.get('num_empty_predictions', 0)}")
    if total_skipped:
        print(f"Skipped {total_skipped} rows")
    if pre_skipped:
        print(f"  {len(pre_skipped)} rows had unsupported answer_type values")

    print(f"\nOverall accuracy (non-empty predictions): {metrics['overall_accuracy']:.4f}")
    print(f"Overall mean score (non-empty predictions): {metrics['overall_mean_score']:.4f}")

    print("\nBy category:")
    print(
        f"  {'category':<14} {'total':>6} {'empty':>6} {'scored':>6} "
        f"{'accuracy':>10} {'mean_score':>12}"
    )
    print("  " + "-" * 62)
    for category, payload in metrics.get("by_category", {}).items():
        print(
            f"  {category:<14} {payload['total']:>6} {payload['num_empty']:>6} "
            f"{payload['num_evaluated']:>6} {payload['accuracy']:>10.4f} "
            f"{payload['mean_score']:>12.4f}"
        )

    print("\nBy metric kind (non-empty predictions only):")
    print(
        f"  {'metric':<14} {'total':>6} {'empty':>6} {'scored':>6} "
        f"{'accuracy':>10} {'mean_score':>12}"
    )
    print("  " + "-" * 62)
    for metric_kind, payload in metrics["by_metric_kind"].items():
        line = (
            f"  {metric_kind:<14} {payload['total']:>6} {payload['num_empty']:>6} "
            f"{payload['num_evaluated']:>6} {payload['accuracy']:>10.4f} "
            f"{payload['mean_score']:>12.4f}"
        )
        if metric_kind == "count":
            obo = payload.get("off_by_one_accuracy")
            mae = payload.get("mae")
            mse = payload.get("mse")
            line += f"  obo={obo:.4f}  mae={mae:.2f}  mse={mse:.2f}"
        print(line)

    print("\nBy answer type (non-empty predictions only):")
    print(
        f"  {'answer_type':<28} {'metric':<14} {'total':>6} {'empty':>6} {'scored':>6} "
        f"{'accuracy':>10} {'mean_score':>12}"
    )
    print("  " + "-" * 92)
    for answer_type, payload in metrics["by_type"].items():
        metric_kind = payload.get("metric_kind") or "-"
        print(
            f"  {answer_type:<28} {metric_kind:<14} {payload['total']:>6} "
            f"{payload['num_empty']:>6} {payload['num_evaluated']:>6} "
            f"{payload['accuracy']:>10.4f} {payload['mean_score']:>12.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preds",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
        help="JSON or JSONL file with gold_answer, prediction, and answer_type fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the aggregate metrics JSON report.",
    )
    parser.add_argument(
        "--no-update-preds",
        action="store_true",
        help="Do not write per-row evaluation verdicts back to --preds.",
    )
    parser.add_argument(
        "--include-rows",
        action="store_true",
        help="Include per-row evaluation details in the written JSON report.",
    )
    args = parser.parse_args()

    rows = read_predictions(args.preds)
    prepared_rows, pre_skipped = prepare_prediction_rows(rows)
    metrics = evaluate_predictions(prepared_rows)
    _print_summary(metrics, pre_skipped=pre_skipped)

    if not args.no_update_preds:
        annotated_rows = annotate_predictions_with_evaluation(
            rows,
            metrics,
            pre_skipped=pre_skipped,
        )
        write_predictions(args.preds, annotated_rows)
        print(f"\nUpdated predictions with evaluation verdicts: {args.preds}")

    if args.output is not None:
        report = dict(metrics)
        if pre_skipped:
            report["pre_skipped"] = pre_skipped
        if not args.include_rows:
            report.pop("rows", None)
            report.pop("skipped", None)
            report.pop("pre_skipped", None)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote metrics report to {args.output}")


if __name__ == "__main__":
    main()
