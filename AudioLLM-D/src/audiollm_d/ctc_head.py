"""Auxiliary CTC head on the post-pooling Qwen2-Audio encoder features (Phase 2.2).

Qwen2-Audio's audio tower (a Whisper-style encoder) emits, per clip, a ``(B, 750, d_encoder)``
tensor of *post-pooling, pre-projection* features at 25 fps — exactly the grid the 2.1 frame
labels live on. We attach a single ``Linear(d_encoder -> |ctc vocab|)`` head producing
frame log-probs for the standard CTC loss.

The features are captured with a forward hook on ``audio_tower`` so we reuse the encoder pass
the LM forward already runs — no second encoder forward, which keeps the joint objective close
to the cost of the LM-only objective.

Frame-count contract (the ticket's #1 silent bug): the encoder always pads mel to 3000 and so
emits a fixed 750 frames, but the *valid* per-clip length is
``_get_feat_extract_output_lengths(feature_attention_mask.sum(-1))``. We pass that as CTC's
``input_lengths`` so the padded tail is ignored; ``encoder_output_lengths`` re-implements the
(verified) formula so the data loader can assert it against the 2.1 label length.
"""

from __future__ import annotations


def encoder_output_lengths(feature_attention_mask):
    """Valid post-pooling frame count per clip, from the mel ``feature_attention_mask``.

    Mirrors ``Qwen2AudioEncoder._get_feat_extract_output_lengths``: conv2 (stride 2) then the
    avg-pool (kernel/stride 2). Returns a 1-D long tensor of shape ``(B,)``.
    """
    input_lengths = feature_attention_mask.sum(-1)
    input_lengths = (input_lengths - 1) // 2 + 1  # after conv stride-2
    output_lengths = (input_lengths - 2) // 2 + 1  # after avg-pool kernel/stride 2
    return output_lengths.long()


def locate_audio_tower(model):
    """Return ``(audio_tower_module, d_encoder)`` for a plain or PEFT-wrapped Qwen2-Audio model."""
    audio_tower = None
    for name, module in model.named_modules():
        if name.endswith("audio_tower"):
            audio_tower = module
            break
    if audio_tower is None:
        raise RuntimeError("Could not locate an 'audio_tower' submodule on the model.")

    d_encoder = None
    config = getattr(audio_tower, "config", None)
    if config is not None:
        d_encoder = getattr(config, "d_model", None)
    if d_encoder is None and hasattr(audio_tower, "conv1"):
        d_encoder = audio_tower.conv1.out_channels  # conv1 maps n_mels -> d_model
    if d_encoder is None:
        raise RuntimeError("Could not infer d_encoder (audio_config.d_model) from audio_tower.")
    return audio_tower, int(d_encoder)


class AuxiliaryCtcHead:
    """``nn.Module`` factory holding the linear CTC head, capture hook, and loss helper.

    Implemented as a builder rather than a hard ``import torch`` at module scope so the rest of
    the package (config, label tooling, tests) stays importable without torch.
    """

    @staticmethod
    def build(d_encoder: int, vocab_size: int, blank_index: int):
        import torch.nn as nn
        import torch.nn.functional as F

        class _AuxiliaryCtcHead(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Linear(d_encoder, vocab_size)
                self.blank_index = int(blank_index)
                self.vocab_size = int(vocab_size)
                # Non-parameter scratch state; underscore-prefixed so nn.Module does not try to
                # register the captured tensor / handle as a buffer or submodule.
                self._captured = None
                self._hook_handle = None

            # --- feature capture -------------------------------------------------
            def register_capture_hook(self, audio_tower) -> None:
                """Install a forward hook that stashes the encoder's last_hidden_state."""
                if self._hook_handle is not None:
                    self._hook_handle.remove()

                def _hook(_module, _inputs, output):
                    feats = getattr(output, "last_hidden_state", None)
                    if feats is None:
                        feats = output[0] if isinstance(output, (tuple, list)) else output
                    self._captured = feats
                    return output

                self._hook_handle = audio_tower.register_forward_hook(_hook)

            def remove_hook(self) -> None:
                if self._hook_handle is not None:
                    self._hook_handle.remove()
                    self._hook_handle = None

            def has_capture(self) -> bool:
                return self._captured is not None

            # --- forward / loss --------------------------------------------------
            def forward_from_capture(self):
                """Consume the captured features -> frame log-probs ``(B, T, V)`` in float32."""
                feats = self._captured
                if feats is None:
                    raise RuntimeError(
                        "CTC head has no captured encoder features; was the audio_tower hook "
                        "installed and did the model forward run with input_features?"
                    )
                self._captured = None  # release the reference each step
                logits = self.proj(feats.to(self.proj.weight.dtype))
                return F.log_softmax(logits.float(), dim=-1)

            def ctc_loss(self, log_probs, targets, input_lengths, target_lengths):
                """Standard CTC loss. ``log_probs`` is ``(B, T, V)`` -> transposed to ``(T, B, V)``."""
                log_probs_tbv = log_probs.transpose(0, 1).contiguous()
                return F.ctc_loss(
                    log_probs_tbv,
                    targets,
                    input_lengths,
                    target_lengths,
                    blank=self.blank_index,
                    zero_infinity=True,
                    reduction="mean",
                )

        return _AuxiliaryCtcHead()
