"""Unit tests for the sliding-window DisfluencyEncoder and window label builder."""

from __future__ import annotations

import torch
import pytest

from audiollm_d.disfluency_encoder import (
    DisfluencyEncoder,
    WINDOW_NUM_CLASSES,
    build_window_labels,
    compute_max_windows,
)
from audiollm_d.special_tokens import SPECIAL_TOKENS


# ── helpers ───────────────────────────────────────────────────────────────────

N_MELS = 128
T_MEL = 3000   # 30s padded, as produced by Whisper processor
OUT_DIM = 1280
BATCH = 2


@pytest.fixture()
def encoder() -> DisfluencyEncoder:
    return DisfluencyEncoder(n_mels=N_MELS, hidden_dim=64, num_layers=1, num_heads=4, out_dim=OUT_DIM)


@pytest.fixture()
def fake_features() -> torch.Tensor:
    """Fake input_features matching Whisper processor output shape."""
    return torch.randn(BATCH, N_MELS, T_MEL)


# ── DisfluencyEncoder ─────────────────────────────────────────────────────────

def test_output_shapes(encoder: DisfluencyEncoder, fake_features: torch.Tensor) -> None:
    n_windows = compute_max_windows()
    residual, logits = encoder(fake_features)
    assert residual.shape == (BATCH, 1500, OUT_DIM), f"residual shape: {residual.shape}"
    assert logits.shape == (BATCH, n_windows, WINDOW_NUM_CLASSES), f"logits shape: {logits.shape}"


def test_output_is_finite(encoder: DisfluencyEncoder, fake_features: torch.Tensor) -> None:
    _, logits = encoder(fake_features)
    assert logits.isfinite().all(), "window_logits contain NaN or Inf"


def test_gradients_flow_through_residual(encoder: DisfluencyEncoder, fake_features: torch.Tensor) -> None:
    """Gradients flow through the residual path into encoder weights."""
    optimizer = torch.optim.SGD(encoder.parameters(), lr=1.0)
    residual, _ = encoder(fake_features)
    loss = (residual - 1.0).pow(2).mean()
    before = [p.clone() for p in encoder.out_proj.parameters()]
    loss.backward()
    optimizer.step()
    after = [p for p in encoder.out_proj.parameters()]
    assert any(not b.equal(a) for b, a in zip(before, after)), "out_proj did not update — gradient not flowing"


def test_window_logits_have_correct_num_classes(encoder: DisfluencyEncoder, fake_features: torch.Tensor) -> None:
    _, logits = encoder(fake_features)
    assert logits.size(-1) == WINDOW_NUM_CLASSES == 1 + len(SPECIAL_TOKENS)


def test_short_audio_does_not_crash(encoder: DisfluencyEncoder) -> None:
    """Input shorter than one window should be padded gracefully."""
    short = torch.randn(1, N_MELS, 20)   # 20 mel frames ≪ 50-frame window
    residual, logits = encoder(short)
    assert residual.shape[0] == 1
    assert logits.shape[0] == 1


# ── build_window_labels ───────────────────────────────────────────────────────

def test_empty_alignment_all_ignored() -> None:
    labels = build_window_labels([])
    assert (labels == -1).all()


def test_speech_only_within_audio_is_zero() -> None:
    alignment = [{"word": "the", "start": 0.0, "end": 0.5},
                 {"word": "boy", "start": 0.5, "end": 1.0}]
    labels = build_window_labels(alignment)
    # Windows within the audio duration should be class 0 (speech).
    audio_end = 1.0
    n_valid = int(audio_end / 0.1)   # windows at t_start 0.0, 0.1, …, 0.9
    assert (labels[:n_valid] == 0).all(), f"expected 0, got {labels[:n_valid]}"
    # Windows beyond audio end should be -1.
    assert (labels[n_valid:] == -1).all()


def test_disfluency_window_labeled_correctly() -> None:
    uh_class = 1 + SPECIAL_TOKENS.index("[uh]")
    # [uh] token spans 0.3s → 0.7s, overlapping ≥30% with window [0.2, 0.7]
    alignment = [
        {"word": "the", "start": 0.0, "end": 0.2},
        {"word": "[uh]", "start": 0.3, "end": 0.7},
        {"word": "boy",  "start": 0.8, "end": 1.2},
    ]
    labels = build_window_labels(alignment)
    # Window index 3 spans [0.3, 0.8] — overlaps with [uh] by 0.4s / 0.5s = 80%
    assert labels[3].item() == uh_class, f"expected {uh_class}, got {labels[3].item()}"


def test_small_token_fraction_not_labeled_as_disfluency() -> None:
    uh_class = 1 + SPECIAL_TOKENS.index("[uh]")
    # [uh] spans 0.0→2.0s (duration 2.0s).  Window 0 [0.0, 0.5s] captures only
    # 0.5/2.0 = 25% of the token — below the 50% token-fraction threshold.
    alignment = [
        {"word": "[uh]", "start": 0.0, "end": 2.0},
    ]
    labels = build_window_labels(alignment, speech_keep_prob=1.0)
    assert labels[0].item() != uh_class, "25% token fraction should not produce disfluency label"
    assert labels[0].item() == 0  # class 0 (speech)

def test_high_token_fraction_labeled_as_disfluency() -> None:
    uh_class = 1 + SPECIAL_TOKENS.index("[uh]")
    # [uh] spans 0.3→0.6s (duration 0.3s).  Window 3 [0.3, 0.8s] captures all
    # 0.3/0.3 = 100% of the token — above the 50% threshold.
    alignment = [
        {"word": "the", "start": 0.0,  "end": 0.3},
        {"word": "[uh]", "start": 0.3, "end": 0.6},
        {"word": "boy",  "start": 0.6, "end": 1.2},
    ]
    labels = build_window_labels(alignment, speech_keep_prob=1.0)
    assert labels[3].item() == uh_class, f"expected {uh_class}, got {labels[3].item()}"


def test_window_beyond_audio_is_ignored() -> None:
    alignment = [{"word": "hi", "start": 0.0, "end": 0.5}]
    labels = build_window_labels(alignment)
    # Windows starting at t > 0.5s should all be -1.
    assert labels[5].item() == -1   # window index 5 starts at 0.5s (just at edge)
    assert labels[10].item() == -1


def test_compute_max_windows_default() -> None:
    n = compute_max_windows(window_size_s=0.5, stride_s=0.1)
    assert n == 296, f"expected 296, got {n}"
