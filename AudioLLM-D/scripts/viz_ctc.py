"""Per-frame CTC argmax visualizations for sanity checking (ticket 2.4).

For a handful of validation clips, run the (adapted) audio encoder, apply the trained CTC head,
take the per-frame argmax over classes, and plot it — overlaid on the waveform and against the
2.1 ground-truth frame labels. By ~epoch 5 the predictions should fire as discrete spikes at
pause/event locations and be ``blank``/``speech`` elsewhere; if it is ``blank`` everywhere, the
vocab is too imbalanced or alpha is too low (see the ticket).

Example:
    python scripts/viz_ctc.py \
        --model-dir /scratch/nathan/lora_run/outputs/qwen2-audio-dementia-lora-v4-ctc-a03 \
        --manifest /scratch/merged_data_80_20/test/manifest.jsonl \
        --ctc-labels-dir /scratch/nathan/lora_run/ctc_labels \
        --num-clips 5 --output-dir <model-dir>/ctc_viz
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from audiollm_d.ctc_head import AuxiliaryCtcHead, encoder_output_lengths, locate_audio_tower
from audiollm_d.ctc_labels import CtcVocab
from audiollm_d.dataset import load_audio
from audiollm_d.manifest import read_jsonl


def _load_ctc_head(model, model_dir: Path, vocab: CtcVocab):
    import torch

    audio_tower, d_encoder = locate_audio_tower(model)
    head = AuxiliaryCtcHead.build(d_encoder, vocab.vocab_size, vocab.blank_index)
    state = torch.load(model_dir / "ctc_head.pt", map_location="cpu")
    head.load_state_dict(state)
    head.to(next(audio_tower.parameters()).device)
    head.eval()
    return head, audio_tower


def _predict_frames(model, head, audio_tower, processor, audio_path: str):
    """Return (pred_classes[T], T) for one clip by running the adapted encoder + head."""
    import torch

    wav = load_audio(audio_path, int(processor.feature_extractor.sampling_rate))
    batch = processor(
        text="<|audio_bos|><|AUDIO|><|audio_eos|>",
        audio=[wav],
        sampling_rate=int(processor.feature_extractor.sampling_rate),
        return_tensors="pt",
        padding=True,
    )
    device = next(audio_tower.parameters()).device
    input_features = batch["input_features"].to(device)
    t_valid = int(encoder_output_lengths(batch["feature_attention_mask"])[0])
    with torch.no_grad():
        audio_out = audio_tower(input_features)
        feats = getattr(audio_out, "last_hidden_state", None)
        if feats is None:
            feats = audio_out[0]
        logits = head.proj(feats.to(head.proj.weight.dtype))  # (1, 750, V)
        pred = logits.float().argmax(-1)[0, :t_valid].cpu().numpy()
    return pred, t_valid, wav


def _plot(clip_id, wav, sr, pred, gt, vocab, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels_axis = list(vocab.classes) + ["blank"]
    t_wav = np.arange(len(wav)) / sr
    t_pred = (np.arange(len(pred)) + 0.5) * 0.04

    fig, (ax_w, ax_p) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax_w.plot(t_wav, wav, lw=0.4, color="steelblue")
    ax_w.set_ylabel("waveform")
    ax_w.set_title(f"{clip_id} — CTC per-frame argmax (pred) vs ground truth")

    if gt is not None:
        n = min(len(gt), len(t_pred))
        ax_p.step(t_pred[:n], gt[:n], where="mid", color="gray", lw=1.2, alpha=0.6, label="ground truth")
    ax_p.step(t_pred, pred, where="mid", color="crimson", lw=1.0, label="prediction")
    ax_p.set_ylim(-0.5, len(labels_axis) - 0.5)
    ax_p.set_yticks(range(len(labels_axis)))
    ax_p.set_yticklabels(labels_axis)
    ax_p.set_xlabel("time (s)")
    ax_p.grid(True, axis="y", alpha=0.3)
    ax_p.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Trained adapter dir (contains ctc_head.pt).")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--ctc-labels-dir", default=None, help="build_ctc_labels.py output root for GT overlay.")
    parser.add_argument("--num-clips", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # Lazy import so the heavy load path mirrors transcribe.py exactly.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from transcribe import load_model_and_processor  # type: ignore

    import json

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir) if args.output_dir else model_dir / "ctc_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_json = model_dir / "ctc_vocab.json"
    if not vocab_json.exists() and args.ctc_labels_dir:
        vocab_json = Path(args.ctc_labels_dir) / "ctc_vocab.json"
    vocab = CtcVocab.from_json(json.loads(Path(vocab_json).read_text(encoding="utf-8")))

    model, processor = load_model_and_processor(str(model_dir), args.base_model)
    head, audio_tower = _load_ctc_head(model, model_dir, vocab)

    rows = read_jsonl(args.manifest)
    rng = random.Random(args.seed)
    chosen = rng.sample(rows, min(args.num_clips, len(rows)))

    for row in chosen:
        clip_id = row.get("clip_id", Path(row["audio_path"]).stem)
        pred, t_valid, wav = _predict_frames(model, head, audio_tower, processor, row["audio_path"])
        gt = None
        if args.ctc_labels_dir:
            gt_path = Path(args.ctc_labels_dir) / "labels" / f"{clip_id}.npy"
            if gt_path.exists():
                gt = np.load(gt_path)[:t_valid]
        _plot(clip_id, wav, int(processor.feature_extractor.sampling_rate), pred, gt, vocab, out_dir / f"{clip_id}.png")
        # Quick textual summary of what the head is firing (blank-collapse early warning).
        from collections import Counter

        hist = Counter(vocab.index_to_class[int(c)] for c in pred)
        print(f"[viz] {clip_id}: T={t_valid} pred_hist={dict(hist)} -> {out_dir / f'{clip_id}.png'}")

    print(f"[viz] Wrote {len(chosen)} CTC visualizations to {out_dir}")


if __name__ == "__main__":
    main()
