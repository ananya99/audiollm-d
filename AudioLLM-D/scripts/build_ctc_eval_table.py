"""Phase 1 vs Phase 2 comparison table with bootstrap CIs (ticket 2.5).

The metric is the LM head's text output (CTC is auxiliary and does not change generation), so
this reads the ``*_predictions.jsonl`` files the trainer already writes (each row has
``reference`` + ``prediction``) and computes WER + order-aware (edit-aligned) disfluency F1 with
clip-level bootstrap CIs for each run. It also breaks F1 down per token so the Phase 2 gain can
be shown to land on the bracket-pause tokens, not on [uh]/[um].

Example:
    python scripts/build_ctc_eval_table.py \
        --run "Phase 1 (v3)=/scratch/.../qwen2-audio-dementia-lora-v3/after_train_test_predictions.jsonl" \
        --run "Phase 2 (v4 a=0.3)=/scratch/.../qwen2-audio-dementia-lora-v4-ctc-a03/after_train_test_predictions.jsonl" \
        --output table.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from audiollm_d.manifest import read_jsonl
from audiollm_d.metrics import (
    bootstrap_disfluency_metrics_ci,
    compute_per_token_order_stats,
)
from audiollm_d.special_tokens import DISFLUENCY_TOKENS

PAUSE_TOKENS = ["[short-pause]", "[medium-pause]", "[long-pause]"]


def _load_pairs(path: str) -> tuple[list[str], list[str]]:
    rows = read_jsonl(path)
    refs = [str(r.get("reference", "")).strip() for r in rows]
    preds = [str(r.get("prediction", "")).strip() for r in rows]
    return refs, preds


def _fmt_ci(stat: dict[str, float]) -> str:
    return f"{stat['point']:.3f} [{stat['lo']:.3f}, {stat['hi']:.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PREDICTIONS_JSONL",
        help="Repeatable. First run is treated as the Phase 1 baseline.",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="Write markdown here (else stdout).")
    args = parser.parse_args()

    runs: list[tuple[str, str]] = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run must be LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        runs.append((label.strip(), path.strip()))

    lines: list[str] = []
    lines.append("## Phase 1 vs Phase 2 — LM-head metrics (bootstrap 95% CI)\n")
    lines.append("| Run | WER ↓ | Edit-F1 ↑ | Edit-P | Edit-R |")
    lines.append("| --- | --- | --- | --- | --- |")

    per_token_rows: dict[str, dict[str, float]] = {}
    pause_only_rows: list[str] = []
    for label, path in runs:
        refs, preds = _load_pairs(path)
        ci = bootstrap_disfluency_metrics_ci(
            refs, preds, DISFLUENCY_TOKENS, n_boot=args.n_boot, seed=args.seed
        )
        lines.append(
            f"| {label} | {_fmt_ci(ci['wer'])} | {_fmt_ci(ci['edit_f1'])} "
            f"| {_fmt_ci(ci['edit_precision'])} | {_fmt_ci(ci['edit_recall'])} |"
        )
        # Pause-only edit-F1 (the DoD target): restrict the token set to the 3 pauses.
        pause_ci = bootstrap_disfluency_metrics_ci(
            refs, preds, PAUSE_TOKENS, n_boot=args.n_boot, seed=args.seed
        )
        pause_only_rows.append(f"| {label} | {_fmt_ci(pause_ci['edit_f1'])} |")
        per_token_rows[label] = {
            tok: stats["f1"]
            for tok, stats in compute_per_token_order_stats(refs, preds, DISFLUENCY_TOKENS).items()
        }

    lines.append("\n### Bracket-pause edit-F1 (the DoD target)\n")
    lines.append("| Run | Pause edit-F1 |")
    lines.append("| --- | --- |")
    lines.extend(pause_only_rows)

    lines.append("\n### Per-token order-aware F1\n")
    header = "| Token | " + " | ".join(label for label, _ in runs) + " |"
    lines.append(header)
    lines.append("| --- " * (len(runs) + 1) + "|")
    for tok in DISFLUENCY_TOKENS:
        cells = " | ".join(f"{per_token_rows[label].get(tok, 0.0):.3f}" for label, _ in runs)
        marker = " *(pause)*" if tok in PAUSE_TOKENS else ""
        lines.append(f"| {tok}{marker} | {cells} |")

    table = "\n".join(lines) + "\n"
    if args.output:
        Path(args.output).write_text(table, encoding="utf-8")
        print(f"[eval-table] Wrote {args.output}")
    else:
        print(table)


if __name__ == "__main__":
    main()
