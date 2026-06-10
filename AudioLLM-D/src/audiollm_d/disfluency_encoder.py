"""Sliding-window disfluency encoder for per-frame residual injection into Whisper."""
from __future__ import annotations

import random as _random

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiollm_d.special_tokens import SPECIAL_TOKENS as _SPECIAL_TOKENS

# Mel spectrogram constants (shared with Whisper preprocessing).
_MEL_HOP_MS: float = 10.0       # milliseconds per mel frame (hop=160 at 16kHz)
_WHISPER_N_MELS: int = 128       # Whisper-large-v2 mel bins
_WHISPER_MAX_FRAMES: int = 1500  # encoder output frames for 30s audio

# Class mapping: 0 = speech, 1..N = one class per disfluency token.
_DISFLUENCY_WINDOW_CLASS: dict[str, int] = {
    token: i + 1 for i, token in enumerate(_SPECIAL_TOKENS)
}
WINDOW_NUM_CLASSES: int = 1 + len(_SPECIAL_TOKENS)  # 9


def _mel_frames(seconds: float) -> int:
    return int(round(seconds * 1000.0 / _MEL_HOP_MS))


def compute_max_windows(window_size_s: float = 0.5, stride_s: float = 0.1) -> int:
    """Number of sliding windows for 30s audio (= Whisper's padded length)."""
    total = _mel_frames(30.0)         # 3000
    wf = _mel_frames(window_size_s)   # 50 for 0.5s
    sf = _mel_frames(stride_s)        # 10 for 0.1s
    return (total - wf) // sf + 1     # 296


def build_window_labels(
    alignment: list[dict],
    *,
    window_size_s: float = 0.5,
    stride_s: float = 0.1,
    max_windows: int | None = None,
    token_overlap_threshold: float = 0.5,
    min_token_duration_s: float = 0.05,
    speech_keep_prob: float = 1.0,
) -> torch.Tensor:
    """Build window-level class labels from an MFA alignment record list.

    A window is labeled with a disfluency class when ≥ token_overlap_threshold
    fraction of that token's duration falls inside the window.  This is
    scale-invariant: a 0.1s [uh] token fully inside a 1.0s window is still
    labeled [uh], unlike the old window-fraction criterion.

    Speech-only windows (no disfluency token passes the threshold) are kept as
    class 0 with probability speech_keep_prob; otherwise they become -1
    (ignored in the loss).  Windows beyond the audio end are always -1.

    Disfluency tokens shorter than min_token_duration_s are discarded to filter
    near-zero-duration MFA artefacts.

    Args:
        alignment: list of {"word": str, "start": float, "end": float} dicts.
        window_size_s: window length in seconds.
        stride_s: window stride in seconds.
        max_windows: tensor size; defaults to compute_max_windows().
        token_overlap_threshold: fraction of the TOKEN that must fall inside a
            window to assign the disfluency label (not fraction of window).
        min_token_duration_s: disfluency tokens shorter than this are discarded.
        speech_keep_prob: probability of keeping a speech-only window as class 0;
            remaining speech windows become -1.  1.0 = keep all (old behaviour).

    Returns:
        LongTensor of shape [max_windows] with values in {-1, 0, 1, …, 8}.
    """
    if max_windows is None:
        max_windows = compute_max_windows(window_size_s, stride_s)

    labels = torch.full((max_windows,), -1, dtype=torch.long)
    if not alignment:
        return labels

    # Strip near-zero-duration disfluency annotations (MFA alignment artefacts).
    filtered = [
        r for r in alignment
        if r.get("word", "") not in _DISFLUENCY_WINDOW_CLASS
        or (r["end"] - r["start"]) >= min_token_duration_s
    ]
    if not filtered:
        return labels
    alignment = filtered

    audio_end = alignment[-1]["end"]

    for win_idx in range(max_windows):
        t_start = win_idx * stride_s
        t_end = t_start + window_size_s
        if t_start >= audio_end:
            break  # remaining windows are silence padding → leave as -1

        # Find the disfluency token most contained within this window.
        best_cls = 0
        best_tok_frac = 0.0
        for rec in alignment:
            cls = _DISFLUENCY_WINDOW_CLASS.get(rec.get("word", ""))
            if cls is None:
                continue
            token_dur = rec["end"] - rec["start"]
            if token_dur <= 0:
                continue
            overlap = max(0.0, min(t_end, rec["end"]) - max(t_start, rec["start"]))
            tok_frac = overlap / token_dur
            if tok_frac > best_tok_frac:
                best_tok_frac = tok_frac
                best_cls = cls

        if best_cls > 0 and best_tok_frac >= token_overlap_threshold:
            labels[win_idx] = best_cls
        elif speech_keep_prob >= 1.0 or _random.random() < speech_keep_prob:
            labels[win_idx] = 0
        # else: remains -1 (ignored in loss)

    return labels


class _WindowEncoder(nn.Module):
    """Encode one mel window [*, n_mels, T_win] into a fixed-size vector [*, D]."""

    def __init__(self, n_mels: int, hidden_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_mels, T]
        x = F.gelu(self.conv1(x))   # [B, D, T]
        x = F.gelu(self.conv2(x))   # [B, D, T//2]
        return self.norm(x.mean(dim=-1))  # [B, D]


class DisfluencyEncoder(nn.Module):
    """Sliding-window disfluency encoder.

    Takes the Whisper log-mel input_features [B, 128, 3000] directly from the
    batch, encodes overlapping windows, and produces:

      disfl_residual  [B, 1500, out_dim]  — added to Whisper encoder output
      window_logits   [B, N_windows, C]   — supervised by MFA window labels

    The residual is scaled by a learnable parameter alpha (initialised to 0 so
    the Whisper baseline is unaffected at the start of training).
    """

    def __init__(
        self,
        n_mels: int = _WHISPER_N_MELS,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        window_size_s: float = 0.5,
        stride_s: float = 0.1,
        num_classes: int = WINDOW_NUM_CLASSES,
        out_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window_frames = _mel_frames(window_size_s)   # 50
        self.stride_frames = _mel_frames(stride_s)         # 10

        self.window_encoder = _WindowEncoder(n_mels, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(
        self, input_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input_features: [B, n_mels, T_mel]
        Returns:
            residual:      [B, 1500, out_dim]  — unscaled; caller applies injection scale
            window_logits: [B, N_windows, num_classes]
        """
        B, C, T = input_features.shape

        if T < self.window_frames:
            input_features = F.pad(input_features, (0, self.window_frames - T))
            T = self.window_frames

        # Sliding window extraction via unfold.
        # [B, C, T] → unfold → [B, C, N_win, W]
        wins = input_features.unfold(-1, self.window_frames, self.stride_frames)
        N = wins.size(2)
        # [B, C, N, W] → [B*N, C, W]
        wins = wins.permute(0, 2, 1, 3).reshape(B * N, C, self.window_frames)

        # Per-window encoding (shared weights).
        win_feats = self.window_encoder(wins).reshape(B, N, self.hidden_dim)  # [B, N, D]

        # Cross-window context.
        ctx = self.context_transformer(win_feats)   # [B, N, D]

        # Supervision logits for MFA window labels.
        window_logits = self.classifier(ctx)         # [B, N, C]

        # Project and upsample to Whisper frame resolution (1500).
        proj = self.out_proj(ctx).transpose(1, 2)    # [B, out_dim, N]
        residual = F.interpolate(
            proj, size=_WHISPER_MAX_FRAMES, mode="linear", align_corners=False
        ).transpose(1, 2)                            # [B, 1500, out_dim]

        return residual, window_logits
