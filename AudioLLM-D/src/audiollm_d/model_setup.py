"""Shared Qwen2-Audio model setup helpers for training and inference."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from audiollm_d.config import DisfluencyEncoderConfig


def find_audio_encoder(model):
    """Return the Qwen2AudioEncoder, unwrapping PEFT layers as needed."""
    candidate = model
    for _ in range(5):
        if hasattr(candidate, "audio_tower"):
            return candidate.audio_tower
        advanced = False
        for attr in ("base_model", "model"):
            nxt = getattr(candidate, attr, None)
            if nxt is not None and nxt is not candidate:
                candidate = nxt
                advanced = True
                break
        if not advanced:
            break
    return None


def get_encoder_hidden_dim(model) -> int:
    """Return the audio encoder hidden dimension (d_model) from config."""
    for obj in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if obj is None:
            continue
        audio_cfg = getattr(getattr(obj, "config", None), "audio_config", None)
        if audio_cfg is not None:
            for attr in ("d_model", "hidden_size", "encoder_hidden_size"):
                dim = getattr(audio_cfg, attr, None)
                if dim is not None:
                    return int(dim)
    return 1280


def find_whisper_encoder(model):
    """Return a Whisper encoder module from a seq2seq ASR model object."""
    candidate = model
    for _ in range(5):
        inner = getattr(candidate, "model", None)
        if inner is not None and hasattr(inner, "encoder"):
            return inner.encoder
        if hasattr(candidate, "encoder"):
            return candidate.encoder
        advanced = False
        for attr in ("base_model", "model"):
            nxt = getattr(candidate, attr, None)
            if nxt is not None and nxt is not candidate:
                candidate = nxt
                advanced = True
                break
        if not advanced:
            break
    return None


def assert_audio_encoder_compatibility(*, target_audio_cfg, source_cfg, source_model_name: str) -> None:
    """Fail early when source Whisper encoder shape does not match Qwen audio tower."""
    required_equal = (
        ("d_model", getattr(target_audio_cfg, "d_model", None), getattr(source_cfg, "d_model", None)),
        ("encoder_layers", getattr(target_audio_cfg, "encoder_layers", None), getattr(source_cfg, "encoder_layers", None)),
        ("encoder_attention_heads", getattr(target_audio_cfg, "encoder_attention_heads", None), getattr(source_cfg, "encoder_attention_heads", None)),
        ("num_mel_bins", getattr(target_audio_cfg, "num_mel_bins", None), getattr(source_cfg, "num_mel_bins", None)),
        ("max_source_positions", getattr(target_audio_cfg, "max_source_positions", None), getattr(source_cfg, "max_source_positions", None)),
    )
    mismatches = [
        f"{name}: target={target_val}, source={source_val}"
        for name, target_val, source_val in required_equal
        if target_val is not None and source_val is not None and int(target_val) != int(source_val)
    ]
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Audio encoder replacement aborted due to incompatible Whisper configs "
            f"for source model {source_model_name}: {mismatch_text}"
        )


def replace_qwen_audio_encoder_from_whisper(model, *, source_model_name: str, use_bf16: bool) -> None:
    """Load Whisper encoder weights from source model into Qwen2-Audio audio_tower."""
    import torch
    from transformers import AutoConfig, AutoModelForSpeechSeq2Seq

    print(f"[audio-init] Loading source Whisper model: {source_model_name}")
    source_cfg = AutoConfig.from_pretrained(source_model_name, trust_remote_code=True)
    target_audio_cfg = getattr(model.config, "audio_config", None)
    if target_audio_cfg is None:
        raise ValueError("Target model has no audio_config; expected Qwen2-Audio architecture.")
    assert_audio_encoder_compatibility(
        target_audio_cfg=target_audio_cfg,
        source_cfg=source_cfg,
        source_model_name=source_model_name,
    )

    source_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        source_model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else None,
    )
    source_encoder = find_whisper_encoder(source_model)
    target_encoder = find_audio_encoder(model)
    if source_encoder is None:
        raise ValueError(
            f"Could not find Whisper encoder module in source model: {source_model_name}"
        )
    if target_encoder is None:
        raise ValueError("Could not find Qwen2-Audio audio_tower in target model.")

    load_result = target_encoder.load_state_dict(source_encoder.state_dict(), strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "Audio encoder replacement produced key mismatch. "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    print("[audio-init] Replaced Qwen2-Audio audio_tower with source Whisper encoder weights.")

    del source_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_disfluency_encoder_for_inference(
    *,
    checkpoint_dir: str | Path,
    model,
    disfl_encoder_config: DisfluencyEncoderConfig | None = None,
):
    """Load DisfluencyEncoder + alpha saved alongside a training checkpoint."""
    import torch
    from audiollm_d.disfluency_encoder import DisfluencyEncoder

    checkpoint_dir = Path(checkpoint_dir)
    disfl_path = checkpoint_dir / "disfl_encoder.pt"
    if not disfl_path.exists():
        return None, None

    enc_cfg = disfl_encoder_config or DisfluencyEncoderConfig()
    hidden_dim = get_encoder_hidden_dim(model)
    disfl_encoder = DisfluencyEncoder(
        hidden_dim=enc_cfg.hidden_dim,
        num_layers=enc_cfg.num_layers,
        num_heads=enc_cfg.num_heads,
        window_size_s=enc_cfg.window_size_s,
        stride_s=enc_cfg.stride_s,
        out_dim=hidden_dim,
    )
    state = torch.load(disfl_path, map_location="cpu")
    disfl_encoder.load_state_dict(state)
    disfl_encoder.eval()

    alpha_value = 0.01
    alpha_path = checkpoint_dir / "disfl_alpha.pt"
    if alpha_path.exists():
        alpha_data = torch.load(alpha_path, map_location="cpu")
        alpha_tensor = alpha_data["alpha"] if isinstance(alpha_data, dict) else alpha_data
        alpha_value = float(alpha_tensor.item())

    device = next(model.parameters()).device
    disfl_encoder.to(device)
    print(
        f"[disfl-encoder] Loaded inference encoder from {disfl_path} "
        f"(alpha={alpha_value:.4f})"
    )
    return disfl_encoder, alpha_value


@contextmanager
def disfluency_encoder_hook(
    *,
    model,
    input_features,
    disfl_encoder,
    disfl_alpha: float,
) -> Iterator[None]:
    """Inject DisfluencyEncoder residual into the audio encoder during forward."""
    if disfl_encoder is None or input_features is None:
        yield
        return

    encoder = find_audio_encoder(model)
    if encoder is None:
        yield
        return

    residual, _ = disfl_encoder(input_features)
    alpha = max(0.005, min(0.1, float(disfl_alpha)))

    def _residual_hook(_mod, _inp, out):
        hs = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        length = min(hs.size(1), residual.size(1))
        scaled = alpha * residual[:, :length, :].to(device=hs.device, dtype=hs.dtype)
        enhanced = hs.clone()
        enhanced[:, :length, :] = hs[:, :length, :] + scaled
        if hasattr(out, "last_hidden_state"):
            return type(out)(last_hidden_state=enhanced)
        return (enhanced,) + out[1:]

    handle = encoder.register_forward_hook(_residual_hook)
    try:
        yield
    finally:
        handle.remove()
