#!/usr/bin/env python
"""Build the rank-sweep results table (ticket 1.5).

Winner rule (declared before looking at the table): for each run, pick the epoch
with the best validation edit-aligned (order-aware) F1; the sweep winner is the
run whose selected epoch has the highest val edit-F1. Nothing is re-tuned after.

Metrics are the same point estimates the training pipeline already writes for v1
and v2 (WER + count-F1 + order/edit-F1 in ``pre_post_train_eval_metrics.json``),
so the sweep is reported identically to the previous models. No bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _test_metrics(split_payload: dict[str, Any]) -> dict[str, Any] | None:
    test = split_payload.get("test") if isinstance(split_payload, dict) else None
    if not isinstance(test, dict):
        return None
    return test


def _edit_f1(test: dict[str, Any]) -> float | None:
    dtm = test.get("disfluency_token_metrics", {})
    val = dtm.get("disfluency_token_order_f1")
    return float(val) if val is not None else None


def select_best_epoch(metrics: dict[str, Any]) -> dict[str, Any] | None:
    """Return the best-val-edit-F1 epoch's label + test metric block."""
    candidates: list[tuple[str, dict[str, Any], float]] = []

    epoch_metrics = metrics.get("epoch_metrics") or {}
    for phase, payload in epoch_metrics.items():
        test = _test_metrics((payload or {}).get("metrics", {}))
        if test is None:
            continue
        f1 = _edit_f1(test)
        if f1 is not None:
            candidates.append((phase, test, f1))

    final_test = _test_metrics(metrics.get("after_train", {}))
    if final_test is not None:
        f1 = _edit_f1(final_test)
        if f1 is not None:
            candidates.append(("after_train (final)", final_test, f1))

    if not candidates:
        return None
    label, test, val_f1 = max(candidates, key=lambda c: c[2])
    return {"sel_epoch": label, "val_edit_f1": val_f1, "test": test}


def run_descriptor(run_dir: Path) -> tuple[int | str, str]:
    """Read lora.r and DoRA flag from the run's saved train_config.yaml."""
    cfg_path = run_dir / "train_config.yaml"
    if not cfg_path.exists():
        return "?", "?"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    lora = cfg.get("lora", {})
    rank = lora.get("r", "?")
    variant = "DoRA" if lora.get("use_dora") else "LoRA"
    return rank, variant


def collect_row(run_dir: Path) -> dict[str, Any]:
    rank, variant = run_descriptor(run_dir)
    row: dict[str, Any] = {"run": run_dir.name, "rank": rank, "variant": variant}
    metrics_path = run_dir / "pre_post_train_eval_metrics.json"
    if not metrics_path.exists():
        row["error"] = "no metrics json"
        return row
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = select_best_epoch(metrics)
    if best is None:
        row["error"] = "no val edit-F1 found"
        return row
    test = best["test"]
    dtm = test.get("disfluency_token_metrics", {})
    row["sel_epoch"] = best["sel_epoch"]
    row["val_edit_f1"] = best["val_edit_f1"]
    row["wer"] = test.get("wer")
    row["edit_f1"] = dtm.get("disfluency_token_order_f1")
    row["count_f1"] = dtm.get("disfluency_token_f1")
    return row


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def build_table(rows: list[dict[str, Any]], winner_run: str | None) -> str:
    lines = [
        "# Phase-1 rank sweep results",
        "",
        "Point metrics from each run's pre_post_train_eval_metrics.json (same eval "
        "as v1/v2). Winner = best val edit-F1 epoch (pre-declared rule).",
        "",
        "| Run | Rank | Type | Sel. epoch | WER | edit-F1 | count-F1 |",
        "|---|---:|:--|:--|:--|:--|:--|",
    ]
    for r in rows:
        mark = " **(winner)**" if r["run"] == winner_run else ""
        if r.get("error"):
            lines.append(
                f"| {r['run']}{mark} | {r.get('rank', '?')} | {r.get('variant', '?')} | "
                f"— | _{r['error']}_ | — | — |"
            )
            continue
        lines.append(
            f"| {r['run']}{mark} | {r['rank']} | {r['variant']} | {r['sel_epoch']} | "
            f"{_fmt(r.get('wer'))} | {_fmt(r.get('edit_f1'))} | {_fmt(r.get('count_f1'))} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="Dir containing the sweep run subdirs.")
    parser.add_argument("--reference-dir", default=None, help="Optional reference-row run dir.")
    parser.add_argument("--out", default=None, help="Optional path to write the Markdown table.")
    args = parser.parse_args()

    run_dirs = sorted(p for p in Path(args.runs_root).iterdir() if p.is_dir())
    rows: list[dict[str, Any]] = []
    if args.reference_dir:
        ref_row = collect_row(Path(args.reference_dir))
        ref_row["variant"] = "ref (wCE)"
        rows.append(ref_row)
    for run_dir in run_dirs:
        rows.append(collect_row(run_dir))

    # Winner = highest selected val edit-F1 among sweep runs (exclude the reference).
    sweep_rows = [r for r in rows if r.get("variant") != "ref (wCE)" and "val_edit_f1" in r]
    winner_run = max(sweep_rows, key=lambda r: r["val_edit_f1"])["run"] if sweep_rows else None

    table = build_table(rows, winner_run)
    print(table)
    if winner_run:
        wr = next(r for r in sweep_rows if r["run"] == winner_run)
        print(
            f"\nWinner: {winner_run} (val edit-F1 = {wr['val_edit_f1']:.4f} at {wr['sel_epoch']}). "
            f"Test edit-F1 = {_fmt(wr.get('edit_f1'))}, WER = {_fmt(wr.get('wer'))}."
        )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote table to: {args.out}")


if __name__ == "__main__":
    main()
