"""Tests for the audio augmentation functions in dataset.py."""

from __future__ import annotations

import numpy as np
import pytest

from audiollm_d.dataset import (
    AUGMENTATION_TYPES,
    apply_babble_noise,
    apply_codec_compression,
    apply_rir_reverb,
    apply_speed_perturbation,
    apply_white_noise_at_snr,
)

SR = 16_000
RNG = np.random.default_rng(0)


def _sine(freq: float = 440.0, duration: float = 1.0, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))


# ---------------------------------------------------------------------------
# Shared contracts for every augmentation
# ---------------------------------------------------------------------------

class _AugContractBase:
    """Mixin: augmented audio must be float32, non-silent, and clipped to [-1, 1]."""

    def _augment(self, audio: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def test_dtype_is_float32(self):
        out = self._augment(_sine())
        assert out.dtype == np.float32

    def test_output_clipped(self):
        audio = _sine(amplitude=0.9)
        out = self._augment(audio)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_empty_audio_passthrough(self):
        empty = np.array([], dtype=np.float32)
        out = self._augment(empty)
        assert out.size == 0

    def test_silent_audio_passthrough(self):
        silent = np.zeros(SR, dtype=np.float32)
        out = self._augment(silent)
        assert _rms(out) < 1e-6


# ---------------------------------------------------------------------------
# White noise
# ---------------------------------------------------------------------------

class TestWhiteNoise(_AugContractBase):
    def _augment(self, audio):
        return apply_white_noise_at_snr(audio, snr_db=20.0, rng=np.random.default_rng(1))

    def test_snr_approximated(self):
        audio = _sine(amplitude=0.5, duration=2.0)
        rng = np.random.default_rng(42)
        noisy = apply_white_noise_at_snr(audio, snr_db=20.0, rng=rng)
        clean_power = np.mean(np.square(audio.astype(np.float64)))
        noise_power = np.mean(np.square((noisy - audio).astype(np.float64)))
        measured_snr_db = 10.0 * np.log10(clean_power / noise_power)
        assert abs(measured_snr_db - 20.0) < 1.5, f"SNR off: {measured_snr_db:.2f} dB"

    def test_adds_noise(self):
        audio = _sine()
        noisy = apply_white_noise_at_snr(audio, snr_db=0.0, rng=np.random.default_rng(7))
        assert not np.allclose(audio, noisy)


# ---------------------------------------------------------------------------
# Babble
# ---------------------------------------------------------------------------

class TestBabble(_AugContractBase):
    def _augment(self, audio):
        return apply_babble_noise(audio, snr_db=15.0, sr=SR, num_sources=4, rng=np.random.default_rng(2))

    def test_adds_noise(self):
        audio = _sine(duration=2.0)
        noisy = apply_babble_noise(audio, snr_db=0.0, sr=SR, num_sources=4, rng=np.random.default_rng(3))
        assert not np.allclose(audio, noisy)

    def test_snr_roughly_correct(self):
        audio = _sine(amplitude=0.5, duration=4.0)
        rng = np.random.default_rng(99)
        noisy = apply_babble_noise(audio, snr_db=20.0, sr=SR, num_sources=1, rng=rng)
        clean_power = np.mean(np.square(audio.astype(np.float64)))
        noise_power = np.mean(np.square((noisy - audio).astype(np.float64)))
        measured_snr_db = 10.0 * np.log10(clean_power / noise_power)
        assert abs(measured_snr_db - 20.0) < 2.0, f"SNR off: {measured_snr_db:.2f} dB"

    def test_num_sources_reproducibility(self):
        audio = _sine()
        out_a = apply_babble_noise(audio, snr_db=10.0, sr=SR, num_sources=4, rng=np.random.default_rng(5))
        out_b = apply_babble_noise(audio, snr_db=10.0, sr=SR, num_sources=4, rng=np.random.default_rng(5))
        np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# Reverb
# ---------------------------------------------------------------------------

class TestReverb(_AugContractBase):
    def _augment(self, audio):
        return apply_rir_reverb(audio, rt60_ms=400.0, sr=SR, rng=np.random.default_rng(4))

    def test_same_length(self):
        audio = _sine()
        out = apply_rir_reverb(audio, rt60_ms=300.0, sr=SR, rng=np.random.default_rng(10))
        assert len(out) == len(audio)

    def test_rms_preserved_approximately(self):
        audio = _sine(amplitude=0.5, duration=2.0)
        out = apply_rir_reverb(audio, rt60_ms=500.0, sr=SR, rng=np.random.default_rng(11))
        assert abs(_rms(out) - _rms(audio)) < 0.05

    def test_modifies_audio(self):
        audio = _sine()
        out = apply_rir_reverb(audio, rt60_ms=600.0, sr=SR, rng=np.random.default_rng(12))
        assert not np.allclose(audio, out)

    def test_short_rt60_passthrough(self):
        audio = _sine()
        out = apply_rir_reverb(audio, rt60_ms=0.0, sr=SR, rng=np.random.default_rng(13))
        np.testing.assert_array_equal(out, audio)


# ---------------------------------------------------------------------------
# Speed perturbation
# ---------------------------------------------------------------------------

class TestSpeedPerturbation:
    def test_faster_shorter(self):
        audio = _sine(duration=1.0)
        out = apply_speed_perturbation(audio, rate=1.2, sr=SR)
        assert len(out) < len(audio)

    def test_slower_longer(self):
        audio = _sine(duration=1.0)
        out = apply_speed_perturbation(audio, rate=0.8, sr=SR)
        assert len(out) > len(audio)

    def test_rate_one_passthrough(self):
        audio = _sine(duration=1.0)
        out = apply_speed_perturbation(audio, rate=1.0, sr=SR)
        np.testing.assert_array_equal(out, audio)

    def test_dtype_float32(self):
        out = apply_speed_perturbation(_sine(), rate=0.9, sr=SR)
        assert out.dtype == np.float32

    def test_empty_passthrough(self):
        empty = np.array([], dtype=np.float32)
        out = apply_speed_perturbation(empty, rate=1.1, sr=SR)
        assert out.size == 0

    def test_length_matches_formula(self):
        audio = _sine(duration=1.0)
        rate = 1.1
        out = apply_speed_perturbation(audio, rate=rate, sr=SR)
        expected_len = int(round(len(audio) / rate))
        # librosa may round to ±1 sample
        assert abs(len(out) - expected_len) <= 2


# ---------------------------------------------------------------------------
# Codec compression
# ---------------------------------------------------------------------------

class TestCodecCompression(_AugContractBase):
    def _augment(self, audio):
        return apply_codec_compression(audio, sr=SR, codec_rate=8000)

    def test_same_length(self):
        audio = _sine(duration=1.0)
        out = apply_codec_compression(audio, sr=SR, codec_rate=8000)
        assert len(out) == len(audio)

    def test_high_frequency_attenuated(self):
        # 6 kHz sine is above the 8 kHz codec's Nyquist (4 kHz) → should be attenuated.
        audio = _sine(freq=6000.0, duration=1.0, amplitude=0.5)
        out = apply_codec_compression(audio, sr=SR, codec_rate=8000)
        assert _rms(out) < _rms(audio) * 0.5

    def test_low_frequency_preserved(self):
        # 300 Hz sine is well within the codec's passband → mostly preserved.
        audio = _sine(freq=300.0, duration=1.0, amplitude=0.5)
        out = apply_codec_compression(audio, sr=SR, codec_rate=8000)
        assert _rms(out) > _rms(audio) * 0.7

    def test_passthrough_when_codec_rate_ge_sr(self):
        audio = _sine()
        out = apply_codec_compression(audio, sr=SR, codec_rate=SR)
        np.testing.assert_array_equal(out, audio)


# ---------------------------------------------------------------------------
# Augmentation type registry
# ---------------------------------------------------------------------------

def test_augmentation_types_set():
    assert "white_noise" in AUGMENTATION_TYPES
    assert "babble" in AUGMENTATION_TYPES
    assert "reverb" in AUGMENTATION_TYPES
    assert "speed" in AUGMENTATION_TYPES
    assert "codec" in AUGMENTATION_TYPES


# ---------------------------------------------------------------------------
# Collator validation
# ---------------------------------------------------------------------------

def test_collator_rejects_unknown_augmentation_type():
    import types

    from audiollm_d.dataset import Qwen2AudioTranscriptionCollator

    # Build a minimal fake processor so __post_init__ doesn't crash on attr access.
    proc = types.SimpleNamespace(
        feature_extractor=types.SimpleNamespace(sampling_rate=SR),
        tokenizer=types.SimpleNamespace(pad_token_id=0),
    )
    with pytest.raises(ValueError, match="Unknown augmentation type"):
        Qwen2AudioTranscriptionCollator(
            processor=proc,
            augmentation_types=["white_noise", "nonexistent"],
        )
