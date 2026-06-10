"""Pre-train the DisfluencyEncoder on MFA window-level alignment labels.

No 7B model needed — the encoder (~2M params) trains purely on mel spectrograms
supervised by MFA alignment labels.  The saved weights are later loaded via the
``disfl_encoder_pretrained_path`` config field in the joint LoRA training run.

Alignment files are resolved in this order for each sample:
  1. ``alignment_path`` column in the manifest row (preferred).
  2. ``--alignment-dir``/{clip_id}_alignment.json  (fallback).

Outputs (in --output-dir):
    disfl_encoder_best.pt   — lowest val-loss checkpoint
    disfl_encoder_final.pt  — weights after all epochs
    train_log.jsonl         — per-epoch metrics

Usage (manifest already has alignment_path):
    python scripts/train_disfl_encoder.py \
        --manifest     /scratch/merged_data_80_20/train/manifest.jsonl \
        --val-manifest /scratch/merged_data_80_20/test/manifest.jsonl \
        --output-dir   /scratch/sathvik/disfl_encoder_pretrain \
        [--epochs 30] [--batch-size 8] [--lr 3e-4]

Usage (alignment_path absent from manifest):
    python scripts/train_disfl_encoder.py \
        --manifest     /scratch/merged_data_80_20/train/manifest.jsonl \
        --val-manifest /scratch/merged_data_80_20/test/manifest.jsonl \
        --alignment-dir /scratch/mfa/mfa_processed \
        --output-dir   /scratch/sathvik/disfl_encoder_pretrain
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audiollm_d.dataset import load_audio
from audiollm_d.disfluency_encoder import (
    WINDOW_NUM_CLASSES,
    DisfluencyEncoder,
    build_window_labels,
)
from audiollm_d.manifest import read_jsonl
from audiollm_d.special_tokens import SPECIAL_TOKENS

_SAMPLING_RATE = 16_000
_CLASS_NAMES = ["speech"] + SPECIAL_TOKENS

# Pause-class merge support: short/medium/long-pause (indices 1,2,3) → class 1.
_PAUSE_MERGE_LUT = torch.tensor([0, 1, 1, 1, 2, 3, 4, 5, 6], dtype=torch.long)
_CLASS_NAMES_MERGED = (
    ["speech", "pause"]
    + [tok for tok in SPECIAL_TOKENS if tok not in ("[short-pause]", "[medium-pause]", "[long-pause]")]
)
_MERGED_NUM_CLASSES = len(_CLASS_NAMES_MERGED)  # 7

# Binary mode: speech(0) vs any disfluency(1).
_CLASS_NAMES_BINARY = ["speech", "disfluency"]
_BINARY_NUM_CLASSES = 2


def _merge_pause_labels(labels: torch.Tensor) -> torch.Tensor:
    """Remap short/medium/long-pause (classes 1,2,3) → 1; shift remaining down."""
    valid = labels >= 0
    out = labels.clone()
    out[valid] = _PAUSE_MERGE_LUT[labels[valid]]
    return out


def _binarize_labels(labels: torch.Tensor) -> torch.Tensor:
    """Remap all disfluency classes (≥1) → 1; speech(0) and ignored(-1) unchanged."""
    out = labels.clone()
    out[(labels > 0)] = 1
    return out


# ── dataset ───────────────────────────────────────────────────────────────────

class _MFADataset(Dataset):
    """Returns (mel [128, 3000], window_labels [N_wins]) pairs."""

    def __init__(
        self,
        records: list[dict],
        feature_extractor,
        *,
        alignment_dir: Path | None,
        window_size_s: float,
        stride_s: float,
        merge_pauses: bool = False,
        min_token_duration_s: float = 0.05,
        speech_keep_prob: float = 0.1,
        binary: bool = False,
    ) -> None:
        self.records = records
        self.fe = feature_extractor
        self.alignment_dir = alignment_dir
        self.window_size_s = window_size_s
        self.stride_s = stride_s
        self.merge_pauses = merge_pauses
        self.min_token_duration_s = min_token_duration_s
        self.speech_keep_prob = speech_keep_prob
        self.binary = binary

    def _alignment_path(self, record: dict) -> Path | None:
        p = record.get("alignment_path")
        if p:
            return Path(p)
        if self.alignment_dir:
            clip_id = record.get("clip_id") or Path(record.get("audio_path", "")).stem
            return self.alignment_dir / f"{clip_id}_alignment.json"
        return None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[idx]

        audio = load_audio(record["audio_path"], _SAMPLING_RATE)
        result = self.fe(
            audio,
            sampling_rate=_SAMPLING_RATE,
            return_tensors="pt",
            padding="max_length",
        )
        mel: torch.Tensor = result["input_features"].squeeze(0)  # [128, 3000]

        alignment: list[dict] = []
        p = self._alignment_path(record)
        if p is not None:
            try:
                alignment = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass

        win_labels = build_window_labels(
            alignment,
            window_size_s=self.window_size_s,
            stride_s=self.stride_s,
            min_token_duration_s=self.min_token_duration_s,
            speech_keep_prob=self.speech_keep_prob,
        )
        if self.binary:
            win_labels = _binarize_labels(win_labels)
        elif self.merge_pauses:
            win_labels = _merge_pause_labels(win_labels)
        return mel, win_labels


# ── helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(
    encoder: DisfluencyEncoder,
    loader: DataLoader,
    device: torch.device,
    *,
    n_classes: int,
    class_names: list[str],
) -> dict[str, float]:
    encoder.eval()
    total_loss = 0.0
    n_batches = 0
    per_class_correct = [0] * n_classes
    per_class_total = [0] * n_classes

    for mel, win_lbls in loader:
        mel = mel.to(device)
        win_lbls = win_lbls.to(device)
        _, logits = encoder(mel)
        valid = win_lbls.ne(-1)
        if not valid.any():
            continue
        loss = F.cross_entropy(
            logits.reshape(-1, n_classes),
            win_lbls.reshape(-1),
            ignore_index=-1,
        )
        total_loss += loss.item()
        n_batches += 1
        preds = logits.argmax(dim=-1)
        for c in range(n_classes):
            mask = win_lbls == c
            per_class_correct[c] += (preds[mask] == c).sum().item()
            per_class_total[c] += mask.sum().item()

    encoder.train()
    metrics: dict[str, float] = {"val_loss": total_loss / max(n_batches, 1)}
    for c, name in enumerate(class_names):
        acc = per_class_correct[c] / max(per_class_total[c], 1)
        metrics[f"val_acc_{name.strip('[]').replace('-', '_')}"] = acc
    return metrics


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--alignment-dir", default=None,
                        help="Fallback dir for {clip_id}_alignment.json. "
                             "Not needed when the manifest has an alignment_path column.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-extractor", default="Qwen/Qwen2-Audio-7B-Instruct",
                        help="HF model name whose feature extractor computes mel specs. "
                             "Must use 128 mel bins to match DisfluencyEncoder.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    # Encoder architecture (must match the config used in joint training).
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--window-size-s", type=float, default=1.0)
    parser.add_argument("--stride-s", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    # Label preprocessing.
    parser.add_argument("--binary", action="store_true",
                        help="Collapse to 2 classes: speech(0) vs any_disfluency(1). "
                             "Most robust option for small datasets — encoder learns "
                             "'where is disfluency' and joint LM training refines types.")
    parser.add_argument("--merge-pauses", action="store_true",
                        help="Merge short/medium/long-pause into one 'pause' class "
                             "(acoustically indistinguishable in mel spectrograms). "
                             "Ignored when --binary is set.")
    parser.add_argument("--min-token-duration", type=float, default=0.05,
                        help="Discard disfluency tokens shorter than this (seconds). "
                             "Filters near-zero-duration MFA artefacts.")
    parser.add_argument("--speech-keep-prob", type=float, default=0.1,
                        help="Fraction of speech-only windows kept as class-0 training "
                             "examples; the rest become -1 (ignored in loss). "
                             "Eliminates class imbalance without needing class weights.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else None

    if args.binary:
        n_classes = _BINARY_NUM_CLASSES
        class_names = _CLASS_NAMES_BINARY
    elif args.merge_pauses:
        n_classes = _MERGED_NUM_CLASSES
        class_names = _CLASS_NAMES_MERGED
    else:
        n_classes = WINDOW_NUM_CLASSES
        class_names = _CLASS_NAMES
    print(f"[disfl-pretrain] Classes: {n_classes}  binary={args.binary}  "
          f"merge_pauses={args.merge_pauses}  "
          f"min_token_duration={args.min_token_duration}s  "
          f"speech_keep_prob={args.speech_keep_prob}")

    # Feature extractor only — no 7B model needed.
    from transformers import AutoFeatureExtractor
    print(f"[disfl-pretrain] Loading feature extractor: {args.feature_extractor}")
    fe = AutoFeatureExtractor.from_pretrained(args.feature_extractor)

    train_records = read_jsonl(args.manifest)
    if args.max_samples:
        train_records = train_records[: args.max_samples]
    val_records = read_jsonl(args.val_manifest) if args.val_manifest else []
    print(f"[disfl-pretrain] Train samples: {len(train_records)}, Val samples: {len(val_records)}")

    ds_kwargs = dict(
        alignment_dir=alignment_dir,
        window_size_s=args.window_size_s,
        stride_s=args.stride_s,
        merge_pauses=args.merge_pauses,
        min_token_duration_s=args.min_token_duration,
        speech_keep_prob=args.speech_keep_prob,
        binary=args.binary,
    )
    train_ds = _MFADataset(train_records, fe, **ds_kwargs)
    # Val set uses speech_keep_prob=1.0 so every speech window is evaluated.
    val_ds_kwargs = {**ds_kwargs, "speech_keep_prob": 1.0}
    val_ds = _MFADataset(val_records, fe, **val_ds_kwargs) if val_records else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
                  if val_ds else None)

    # ── startup diagnostic ────────────────────────────────────────────────────
    _probe_n = min(20, len(train_records))
    _n_align_ok, _n_align_missing, _total_valid, _total_windows = 0, 0, 0, 0
    for _rec in train_records[:_probe_n]:
        _p = train_ds._alignment_path(_rec)
        if _p is None or not _p.exists():
            _n_align_missing += 1
            continue
        try:
            import json as _json
            _aln = _json.loads(_p.read_text(encoding="utf-8"))
            _n_align_ok += 1
            _lbls = build_window_labels(
                _aln,
                window_size_s=args.window_size_s,
                stride_s=args.stride_s,
                min_token_duration_s=args.min_token_duration,
                speech_keep_prob=1.0,  # count all windows, not sampled
            )
            if args.binary:
                _lbls = _binarize_labels(_lbls)
            elif args.merge_pauses:
                _lbls = _merge_pause_labels(_lbls)
            _total_valid += (_lbls >= 0).sum().item()
            _total_windows += _lbls.numel()
        except Exception:
            _n_align_missing += 1
    print(f"[disfl-pretrain] Alignment probe ({_probe_n} samples): "
          f"found={_n_align_ok}  missing/unreadable={_n_align_missing}")
    if _n_align_ok > 0:
        print(f"[disfl-pretrain]   valid windows: {_total_valid}/{_total_windows} "
              f"({100*_total_valid/max(_total_windows,1):.1f}%) — "
              f"with speech_keep_prob={args.speech_keep_prob} expect ~"
              f"{int(_total_valid * (args.speech_keep_prob + (1-args.speech_keep_prob) * 0.18))}"
              f" per epoch")
    if _n_align_missing == _probe_n:
        print("[disfl-pretrain] WARNING: no alignment files found — all windows will be -1 "
              "and no loss will be computed. Check alignment_path in the manifest.")
    # ─────────────────────────────────────────────────────────────────────────

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[disfl-pretrain] Device: {device}")

    encoder = DisfluencyEncoder(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        window_size_s=args.window_size_s,
        stride_s=args.stride_s,
        num_classes=n_classes,
    ).to(device)
    print(f"[disfl-pretrain] Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs

    def _lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        t = (step - args.warmup_steps) / max(total_steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    best_disfl_acc = -1.0
    log_path = output_dir / "train_log.jsonl"
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(1, args.epochs + 1):
        encoder.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, (mel, win_lbls) in enumerate(train_loader):
            mel = mel.to(device)
            win_lbls = win_lbls.to(device)
            valid = win_lbls.ne(-1)
            if not valid.any():
                continue

            _, logits = encoder(mel)
            loss = F.cross_entropy(
                logits.reshape(-1, n_classes),
                win_lbls.reshape(-1),
                ignore_index=-1,
            )
            (loss / args.grad_accum).backward()
            epoch_loss += loss.item()
            epoch_batches += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 20 == 0:
                    print(
                        f"[disfl-pretrain] step={global_step:>5}  "
                        f"loss={loss.item():.4f}  "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )

        avg_train_loss = epoch_loss / max(epoch_batches, 1)
        if epoch_batches == 0:
            print(f"[disfl-pretrain] WARNING epoch={epoch}: no valid batches — "
                  "all window labels are -1. Check alignment files.")
        log_row: dict[str, Any] = {"epoch": epoch, "train_loss": avg_train_loss}

        if val_loader is not None:
            val_metrics = _evaluate(
                encoder, val_loader, device,
                n_classes=n_classes, class_names=class_names,
            )
            log_row.update(val_metrics)
            val_loss = val_metrics["val_loss"]
            # Mean accuracy across all disfluency classes (skip speech at index 0).
            disfl_accs = [
                val_metrics[f"val_acc_{n.strip('[]').replace('-', '_')}"]
                for n in class_names[1:]
            ]
            disfl_acc = sum(disfl_accs) / len(disfl_accs) if disfl_accs else 0.0
            speech_acc = val_metrics.get("val_acc_speech", 0.0)
            balanced_acc = (speech_acc + disfl_acc) / 2.0
            log_row["val_disfl_acc_mean"] = disfl_acc
            log_row["val_balanced_acc"] = balanced_acc
            acc_parts = "  ".join(
                f"{n}={val_metrics.get('val_acc_' + n.strip('[]').replace('-', '_'), 0.0):.2f}"
                for n in class_names
            )
            print(
                f"[disfl-pretrain] epoch={epoch}/{args.epochs}  "
                f"train={avg_train_loss:.4f}  val={val_loss:.4f}  "
                f"acc({acc_parts})  disfl_mean={disfl_acc:.3f}  balanced={balanced_acc:.3f}"
            )
            if balanced_acc > best_disfl_acc:
                best_disfl_acc = balanced_acc
                torch.save(encoder.state_dict(), output_dir / "disfl_encoder_best.pt")
                print(f"[disfl-pretrain]   → best balanced_acc={balanced_acc:.3f}  saved disfl_encoder_best.pt")
        else:
            print(f"[disfl-pretrain] epoch={epoch}/{args.epochs}  train_loss={avg_train_loss:.4f}")

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

    torch.save(encoder.state_dict(), output_dir / "disfl_encoder_final.pt")
    print(f"[disfl-pretrain] Saved final → {output_dir / 'disfl_encoder_final.pt'}")
    if val_loader is not None:
        print(f"[disfl-pretrain] Best disfl_acc={best_disfl_acc:.3f} → {output_dir / 'disfl_encoder_best.pt'}")


if __name__ == "__main__":
    main()
