"""Stage 2 (v5): generate CTC frame labels from CrisperWhisper VAD silences.

Converts each clip's VAD silence timeline (Stage 0) into a 25 fps frame-label sequence, binning
silences by the Stage-1 calibrated duration thresholds. The CTC vocab is silence-only —
``{speech, short-pause, medium-pause, long-pause, blank}`` — with NO non-silent tokens (laughter,
non-verbal, unclear, uh, um stay with the LM head; VAD can't label them).

The output format is byte-for-byte what the v4 pipeline already consumes
(``labels/{clip_id}.npy`` int8 + ``ctc_vocab.json`` + ``meta.jsonl``), so Stage 3 retraining
needs only a new ``ctc.labels_dir`` — the ONLY difference from v4 is the source of the labels.

Acceptance (built in): label length == ``ceil(audio_duration / 0.04)`` == encoder output length
to within +/-1 (asserted; the off-by-one is the classic silent bug). Spot-check overlays render
labels + VAD silences against the waveform for the manual check.

Example:
    python scripts/build_ctc_labels_vad.py \
        --manifest /scratch/merged_data_80_20/train/manifest.jsonl \
                   /scratch/merged_data_80_20/test/manifest.jsonl \
        --vad-dir /scratch/nathan/lora_run/v5_vad/vad_silences \
        --thresholds /scratch/nathan/lora_run/v5_vad/stage1_thresholds.json \
        --output-dir /scratch/nathan/lora_run/v5_ctc_labels \
        --spot-check 5

    # collapsed vocab when Stage 1 says categories aren't separable:
    #   --classes speech pause            (binary)
    #   --classes speech pause long-pause (3-way)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from audiollm_d.ctc_labels import collapse_runs, n_frames_for_duration
from audiollm_d.manifest import read_jsonl
from audiollm_d.vad import (
    classify_duration,
    load_silences,
    load_thresholds,
    rasterize_vad_silences,
    v5_vocab,
)


def _clip_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / float(info.samplerate)


def _spot_check_plot(clip_id, audio_path, frame_labels, silences, vocab, out_path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wav, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t_wav = np.arange(len(wav)) / sr
    t_lab = (np.arange(len(frame_labels)) + 0.5) * 0.04

    fig, (ax_w, ax_l) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)
    ax_w.plot(t_wav, wav, lw=0.4, color="steelblue")
    for s in silences:
        ax_w.axvspan(s.start, s.end, color="orange", alpha=0.25)
    ax_w.set_ylabel("waveform\n(orange=VAD silence)")
    ax_w.set_title(f"{clip_id} — v5 VAD CTC frame labels (25 fps)")

    ax_l.step(t_lab, frame_labels, where="mid", color="darkorange", lw=1.0)
    ax_l.set_ylim(-0.5, len(vocab.classes) - 0.5)
    ax_l.set_yticks(range(len(vocab.classes)))
    ax_l.set_yticklabels(vocab.classes)
    ax_l.set_xlabel("time (s)")
    ax_l.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", nargs="+", required=True)
    parser.add_argument("--vad-dir", required=True, help="Stage 0 output: vad_silences/{clip_id}.json")
    parser.add_argument("--thresholds", required=True, help="Stage 1 output: stage1_thresholds.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="CTC event classes (speech first). Omit for v5 default "
        "[speech, short-pause, medium-pause, long-pause]. Use for a Stage-1 collapse.",
    )
    parser.add_argument("--spot-check", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocab = v5_vocab(args.classes)
    thresholds = load_thresholds(args.thresholds)
    vad_dir = Path(args.vad_dir)
    out_dir = Path(args.output_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ctc-vad] vocab={vocab.classes} blank_index={vocab.blank_index}")
    print(f"[ctc-vad] thresholds={thresholds}")

    rows: dict[str, dict] = {}
    for manifest in args.manifest:
        for row in read_jsonl(manifest):
            rows.setdefault(row["clip_id"], row)
    print(f"[ctc-vad] {len(rows)} unique clips across {len(args.manifest)} manifest(s).")

    spot_clips: set[str] = set()
    if args.spot_check > 0:
        rng = random.Random(args.seed)
        spot_clips = set(rng.sample(sorted(rows), min(args.spot_check, len(rows))))
        (out_dir / "spot_check").mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    class_hist_total: Counter[str] = Counter()
    silence_class_total: Counter[str] = Counter()
    n_done = 0
    meta_path = out_dir / "meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as meta_f:
        for clip_id, row in sorted(rows.items()):
            vad_path = vad_dir / f"{clip_id}.json"
            if not vad_path.exists():
                missing.append(clip_id)
                continue
            silences = load_silences(vad_path)
            duration = _clip_duration_seconds(row["audio_path"])
            n_frames = n_frames_for_duration(duration)
            frame_labels = rasterize_vad_silences(silences, n_frames, vocab, thresholds)

            # Acceptance assertion: saved length == ceil(duration / 0.04).
            assert len(frame_labels) == n_frames == n_frames_for_duration(duration), (
                f"{clip_id}: label length {len(frame_labels)} != ceil(dur/0.04) {n_frames}"
            )

            np.save(labels_dir / f"{clip_id}.npy", frame_labels)
            hist = Counter(vocab.index_to_class[int(c)] for c in frame_labels)
            class_hist_total.update(hist)
            for s in silences:
                silence_class_total[classify_duration(s.duration, thresholds)] += 1

            meta_f.write(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "n_frames": int(n_frames),
                        "duration": round(duration, 4),
                        "n_silences": len(silences),
                        "target_len": len(collapse_runs(frame_labels)),
                        "class_histogram": dict(hist),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_done += 1

            if clip_id in spot_clips:
                _spot_check_plot(
                    clip_id,
                    row["audio_path"],
                    frame_labels,
                    silences,
                    vocab,
                    out_dir / "spot_check" / f"{clip_id}.png",
                )

    (out_dir / "ctc_vocab.json").write_text(
        json.dumps(vocab.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[ctc-vad] Wrote {n_done} label files to {labels_dir}")
    print(f"[ctc-vad] Wrote vocab to {out_dir / 'ctc_vocab.json'} and meta to {meta_path}")
    total = sum(class_hist_total.values()) or 1
    print("[ctc-vad] Corpus frame-class distribution:")
    for cls_name, count in class_hist_total.most_common():
        print(f"[ctc-vad]   {cls_name:16} {count:>9,}  ({100.0 * count / total:5.2f}%)")
    print("[ctc-vad] Silence-segment bin counts (by measured duration):")
    for cls_name, count in silence_class_total.most_common():
        print(f"[ctc-vad]   {cls_name:16} {count:>7,}")
    if spot_clips:
        print(f"[ctc-vad] Spot-check overlays: {out_dir / 'spot_check'} ({len(spot_clips)} clips)")
    if missing:
        raise SystemExit(
            f"[ctc-vad] ERROR: {len(missing)} clips missing VAD silence files (e.g. {missing[:5]}). "
            f"Looked in {vad_dir}. Run Stage 0 (scripts/acoustic_vad.py) for these clips."
        )


if __name__ == "__main__":
    main()
