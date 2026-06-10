"""Stage 1 / GATE (v5): diff CHAT-marked pauses against CrisperWhisper VAD silences.

This is the decision gate for the whole experiment. For each pause the CHAT transcript marked
(read from the MFA alignment JSONs — same artifact v4 used), we find the temporally-overlapping
VAD silence and record its measured duration, then compute three quantities:

  1. Per-CHAT-category VAD-duration distributions: (.) short / (..) medium / (...) long.
  2. Agreement rate: fraction of CHAT pauses that have a corresponding VAD silence.
  3. VAD-only rate: how often VAD finds silence where CHAT marked nothing.

It then emits a one-line GATE verdict and a calibrated threshold decision:

  * separate + meaningful VAD-only  -> PROCEED (premise holds). Thresholds from percentiles.
  * heavy overlap                   -> COLLAPSE classes, then proceed (see --collapse-* hints).
  * VAD ~= CHAT, little VAD-only     -> STOP. Relabeling adds nothing; pivot to synthetic splicing.

Outputs (to --output-dir): ``stage1_report.md``, ``stage1_thresholds.json`` (consumed by
Stage 2), ``stage1_stats.json``, and (if matplotlib is available and not --no-plots) duration
histograms — a paper figure.

Example:
    python scripts/vad_chat_diff.py \
        --vad-dir /scratch/nathan/lora_run/v5_vad/vad_silences \
        --alignment-dir /scratch/sathvik/mfa_processed \
        --manifest /scratch/merged_data_80_20/train/manifest.jsonl \
        --output-dir /scratch/nathan/lora_run/v5_vad
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from audiollm_d.manifest import read_jsonl
from audiollm_d.vad import (
    calibrate_thresholds,
    chat_pauses_from_alignment,
    load_silences,
    match_chat_to_vad,
    summarize_category_durations,
    vad_only_segments,
)

CATEGORIES = ["short-pause", "medium-pause", "long-pause"]
# GATE thresholds (tunable). These encode the ticket's qualitative decision rule numerically so
# the verdict is reproducible; the human still confirms from the plots.
AGREEMENT_HIGH = 0.85  # CHAT pauses mostly backed by VAD silence
VAD_ONLY_MEANINGFUL = 0.15  # >= this share of VAD silences are CHAT-unmarked -> new info
SEPARATION_MIN = 0.10  # min gap (s) between adjacent category medians to call them "separable"


def _alignment_path(alignment_dir: Path, clip_id: str) -> Path:
    return alignment_dir / f"{clip_id}_alignment.json"


def _plot_distributions(per_cat: dict[str, list[float]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"[stage1] matplotlib unavailable ({exc}); skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"short-pause": "tab:green", "medium-pause": "tab:orange", "long-pause": "tab:red"}
    for cat in CATEGORIES:
        vals = per_cat.get(cat, [])
        if vals:
            ax.hist(vals, bins=40, alpha=0.5, label=f"{cat} (n={len(vals)})", color=colors[cat])
    ax.set_xlabel("measured VAD silence duration (s)")
    ax.set_ylabel("count")
    ax.set_title("CHAT pause category vs CrisperWhisper VAD silence duration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[stage1] Wrote distribution plot -> {out_path}")


def _decide_gate(agreement: float, vad_only_rate: float, stats: dict[str, dict]) -> tuple[str, str]:
    """Return ``(verdict, one_line)``. verdict in {PROCEED, COLLAPSE, STOP}."""
    medians = [stats[c]["median"] for c in CATEGORIES if stats.get(c, {}).get("n", 0) > 0]
    separable = (
        len(medians) >= 2
        and all(
            (medians[i + 1] - medians[i]) >= SEPARATION_MIN for i in range(len(medians) - 1)
        )
    )

    if vad_only_rate < VAD_ONLY_MEANINGFUL and agreement >= AGREEMENT_HIGH:
        return (
            "STOP",
            f"VAD ~= CHAT (agreement {agreement:.2f}, VAD-only {vad_only_rate:.2f}). Relabeling "
            "adds little information -> STOP and pivot to synthetic silence-splicing.",
        )
    if separable:
        return (
            "PROCEED",
            f"Categories separate (medians {medians}) and VAD-only {vad_only_rate:.2f} is "
            "meaningful -> premise holds. Use the calibrated thresholds and proceed to Stage 2.",
        )
    return (
        "COLLAPSE",
        f"Category distributions overlap (medians {medians}); not cleanly separable. Collapse to "
        "fewer classes (e.g. speech/pause/long-pause or binary pause), pass --classes to Stage 2, "
        "then proceed.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vad-dir", required=True, help="Stage 0 output: vad_silences/{clip_id}.json")
    parser.add_argument("--alignment-dir", default="/scratch/sathvik/mfa_processed")
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-overlap", type=float, default=0.02, help="Min CHAT/VAD overlap (s) to match.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    vad_dir = Path(args.vad_dir)
    alignment_dir = Path(args.alignment_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clip_ids: list[str] = []
    seen: set[str] = set()
    for manifest in args.manifest:
        for row in read_jsonl(manifest):
            cid = row["clip_id"]
            if cid not in seen:
                seen.add(cid)
                clip_ids.append(cid)

    per_cat_durations: dict[str, list[float]] = defaultdict(list)
    n_chat_total = 0
    n_chat_matched = 0
    n_vad_total = 0
    n_vad_only = 0
    missing_vad: list[str] = []
    missing_align: list[str] = []

    for clip_id in clip_ids:
        vad_path = vad_dir / f"{clip_id}.json"
        align_path = _alignment_path(alignment_dir, clip_id)
        if not vad_path.exists():
            missing_vad.append(clip_id)
            continue
        if not align_path.exists():
            missing_align.append(clip_id)
            continue

        vad_silences = load_silences(vad_path)
        alignment = json.loads(align_path.read_text(encoding="utf-8"))
        chat_pauses = chat_pauses_from_alignment(alignment)

        matches = match_chat_to_vad(chat_pauses, vad_silences, min_overlap_s=args.min_overlap)
        for m in matches:
            n_chat_total += 1
            if m.matched and m.vad_duration is not None:
                n_chat_matched += 1
                per_cat_durations[m.pause.category].append(m.vad_duration)

        n_vad_total += len(vad_silences)
        n_vad_only += len(vad_only_segments(chat_pauses, vad_silences, min_overlap_s=args.min_overlap))

    agreement = n_chat_matched / n_chat_total if n_chat_total else 0.0
    vad_only_rate = n_vad_only / n_vad_total if n_vad_total else 0.0

    stats = {
        cat: summarize_category_durations(per_cat_durations.get(cat, []), cat).to_json()
        for cat in CATEGORIES
    }
    thresholds = calibrate_thresholds(per_cat_durations)
    verdict, one_line = _decide_gate(agreement, vad_only_rate, stats)

    # --- write artifacts ---------------------------------------------------------------------
    stats_payload = {
        "n_clips_used": len(clip_ids) - len(missing_vad) - len(missing_align),
        "n_chat_pauses": n_chat_total,
        "n_chat_matched": n_chat_matched,
        "agreement_rate": round(agreement, 4),
        "n_vad_silences": n_vad_total,
        "n_vad_only": n_vad_only,
        "vad_only_rate": round(vad_only_rate, 4),
        "category_stats": stats,
        "missing_vad": missing_vad,
        "missing_alignment": missing_align,
    }
    (out_dir / "stage1_stats.json").write_text(
        json.dumps(stats_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    threshold_payload = {
        "verdict": verdict,
        "decision": one_line,
        "thresholds": thresholds,
        "rule": "duration < short_max -> short-pause; < medium_max -> medium-pause; else long-pause",
        "derived_from": {c: stats[c] for c in CATEGORIES},
    }
    (out_dir / "stage1_thresholds.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if not args.no_plots:
        _plot_distributions(per_cat_durations, out_dir / "stage1_distributions.png")

    report = _render_report(stats_payload, thresholds, verdict, one_line)
    (out_dir / "stage1_report.md").write_text(report, encoding="utf-8")

    print(report)
    print(f"\n[stage1] Artifacts in {out_dir}: stage1_report.md, stage1_thresholds.json, stage1_stats.json")
    if missing_vad or missing_align:
        print(f"[stage1] NOTE: skipped {len(missing_vad)} missing-VAD, {len(missing_align)} missing-alignment clips.")
    print(f"[stage1] GATE = {verdict}")


def _render_report(stats: dict, thresholds: dict, verdict: str, one_line: str) -> str:
    cat_rows = "\n".join(
        f"| {c} | {stats['category_stats'][c]['n']} | "
        f"{stats['category_stats'][c]['median']} | "
        f"{stats['category_stats'][c]['p25']} | {stats['category_stats'][c]['p75']} | "
        f"{stats['category_stats'][c]['mean']} |"
        for c in CATEGORIES
    )
    return f"""# Stage 1 — CHAT vs CrisperWhisper VAD diff (GATE)

**GATE verdict: `{verdict}`**

> {one_line}

## The three quantities

- **Agreement rate** (CHAT pauses with a matching VAD silence): **{stats['agreement_rate']:.3f}**
  ({stats['n_chat_matched']}/{stats['n_chat_pauses']})
- **VAD-only rate** (VAD silences CHAT never marked): **{stats['vad_only_rate']:.3f}**
  ({stats['n_vad_only']}/{stats['n_vad_silences']})
- **Clips used:** {stats['n_clips_used']}

## Per-CHAT-category VAD silence duration (seconds)

| category | n | median | p25 | p75 | mean |
|---|---|---|---|---|---|
{cat_rows}

See `stage1_distributions.png` for the full histograms (paper figure).

## Threshold decision (Stage 2 input)

`stage1_thresholds.json`:

- `short_max = {thresholds['short_max']}` — duration < short_max -> **short-pause**
- `medium_max = {thresholds['medium_max']}` — duration < medium_max -> **medium-pause**; else **long-pause**

(Replaces v4's hand-picked 0.3 / 1.0 / 2.5.)

## What this gate forces

- `PROCEED` — premise holds: run Stage 2 with the thresholds above, then Stage 3 retrain.
- `COLLAPSE` — categories not separable: reduce the CTC vocab (`--classes speech pause` or
  `--classes speech pause long-pause`) in Stage 2, then proceed.
- `STOP` — VAD ~= CHAT: relabeling won't add information. Report the negative result and pivot
  effort to synthetic silence-splicing (the separate next experiment).
"""


if __name__ == "__main__":
    main()
