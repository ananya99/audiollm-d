"""Generate frame-level CTC labels for every clip (Phase 2.1).

Reads the per-clip MFA alignment JSONs and rasterizes each onto the 25 fps frame grid (one
class id per 40 ms frame), saving an int8 ``.npy`` per clip plus a ``ctc_vocab.json`` and a
``meta.jsonl``. Optionally overlays labels on a few waveforms for the manual sanity check.

Acceptance (built in): every train/val clip gets a label sequence whose length equals
``ceil(audio_duration / 0.04)``; the run aborts if any clip is missing its alignment.

Example:
    python scripts/build_ctc_labels.py \
        --manifest /scratch/merged_data_80_20/train/manifest.jsonl \
                   /scratch/merged_data_80_20/test/manifest.jsonl \
        --alignment-dir /scratch/sathvik/mfa_processed \
        --output-dir /scratch/nathan/lora_run/ctc_labels \
        --spot-check 5
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from audiollm_d.ctc_labels import (
    CtcVocab,
    collapse_runs,
    n_frames_for_duration,
    rasterize_alignment,
)
from audiollm_d.manifest import read_jsonl

DEFAULT_ALIGNMENT_DIR = "/scratch/sathvik/mfa_processed"


def _alignment_path(alignment_dir: Path, clip_id: str) -> Path:
    return alignment_dir / f"{clip_id}_alignment.json"


def _clip_duration_seconds(audio_path: str) -> float:
    info = sf.info(audio_path)
    return info.frames / float(info.samplerate)


def _spot_check_plot(
    clip_id: str,
    audio_path: str,
    frame_labels: np.ndarray,
    vocab: CtcVocab,
    out_path: Path,
) -> None:
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
    ax_w.set_ylabel("waveform")
    ax_w.set_title(f"{clip_id} — CTC frame labels (25 fps)")

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
    parser.add_argument("--manifest", nargs="+", required=True, help="One or more manifest JSONL files.")
    parser.add_argument("--alignment-dir", default=DEFAULT_ALIGNMENT_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="CTC event classes (speech first). Omit for the full default vocab.",
    )
    parser.add_argument("--spot-check", type=int, default=0, help="Render N random overlay PNGs.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    vocab = CtcVocab.from_classes(args.classes)
    alignment_dir = Path(args.alignment_dir)
    out_dir = Path(args.output_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Gather all clips across the provided manifests (dedup by clip_id).
    rows: dict[str, dict] = {}
    for manifest in args.manifest:
        for row in read_jsonl(manifest):
            clip_id = row["clip_id"]
            rows.setdefault(clip_id, row)

    print(f"[ctc-labels] vocab={vocab.classes} blank_index={vocab.blank_index}")
    print(f"[ctc-labels] {len(rows)} unique clips across {len(args.manifest)} manifest(s).")

    missing: list[str] = []
    meta_path = out_dir / "meta.jsonl"
    class_hist_total: Counter[str] = Counter()
    n_done = 0
    spot_clips = set()
    if args.spot_check > 0:
        rng = random.Random(args.seed)
        spot_clips = set(rng.sample(sorted(rows), min(args.spot_check, len(rows))))
        spot_dir = out_dir / "spot_check"
        spot_dir.mkdir(parents=True, exist_ok=True)

    with meta_path.open("w", encoding="utf-8") as meta_f:
        for clip_id, row in sorted(rows.items()):
            align_path = _alignment_path(alignment_dir, clip_id)
            if not align_path.exists():
                missing.append(clip_id)
                continue
            alignment = json.loads(align_path.read_text(encoding="utf-8"))
            duration = _clip_duration_seconds(row["audio_path"])
            n_frames = n_frames_for_duration(duration)
            frame_labels = rasterize_alignment(alignment, n_frames, vocab)

            # Acceptance assertion: saved length == ceil(duration / 0.04).
            assert len(frame_labels) == n_frames == n_frames_for_duration(duration), (
                f"{clip_id}: label length {len(frame_labels)} != ceil(dur/0.04) {n_frames}"
            )

            np.save(labels_dir / f"{clip_id}.npy", frame_labels)
            hist = Counter(vocab.index_to_class[int(c)] for c in frame_labels)
            class_hist_total.update(hist)
            target_len = len(collapse_runs(frame_labels))
            meta_f.write(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "n_frames": int(n_frames),
                        "duration": round(duration, 4),
                        "target_len": target_len,
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
                    vocab,
                    out_dir / "spot_check" / f"{clip_id}.png",
                )

    (out_dir / "ctc_vocab.json").write_text(
        json.dumps(vocab.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[ctc-labels] Wrote {n_done} label files to {labels_dir}")
    print(f"[ctc-labels] Wrote vocab to {out_dir / 'ctc_vocab.json'} and meta to {meta_path}")
    total = sum(class_hist_total.values()) or 1
    print("[ctc-labels] Corpus frame-class distribution:")
    for cls_name, count in class_hist_total.most_common():
        print(f"[ctc-labels]   {cls_name:16} {count:>9,}  ({100.0 * count / total:5.2f}%)")
    if spot_clips:
        print(f"[ctc-labels] Spot-check overlays: {out_dir / 'spot_check'} ({len(spot_clips)} clips)")
    if missing:
        raise SystemExit(
            f"[ctc-labels] ERROR: {len(missing)} clips missing alignments (e.g. {missing[:5]}). "
            f"Looked in {alignment_dir}."
        )


if __name__ == "__main__":
    main()
