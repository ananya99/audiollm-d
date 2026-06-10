"""Pipeline integration tests — no model download required."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from audiollm_d.dataset import Qwen2AudioTranscriptionCollator
from audiollm_d.special_tokens import SPECIAL_TOKENS
from audiollm_d.train import WeightedTokenTrainer


# ── shared helpers ────────────────────────────────────────────────────────────

def _make_tokenizer() -> PreTrainedTokenizerFast:
    base = Tokenizer(WordLevel(vocab={"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=base, unk_token="[UNK]", pad_token="[PAD]")
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


class _FakeProcessor:
    """Minimal processor stub backed by the lightweight tokenizer.

    apply_chat_template returns "SYS USR" for the prompt and
    "SYS USR <assistant content>" for the full conversation so that
    prompt_length is always exactly 2 tokens.
    """

    _PROMPT = "SYS USR"

    def __init__(self) -> None:
        self.tokenizer = _make_tokenizer()
        self.feature_extractor = MagicMock()
        self.feature_extractor.sampling_rate = 16000

    def apply_chat_template(
        self, conversation: list[dict], add_generation_prompt: bool = False, tokenize: bool = False
    ) -> str:
        has_assistant = any(m["role"] == "assistant" for m in conversation)
        if has_assistant:
            target = next(m["content"] for m in conversation if m["role"] == "assistant")
            return f"{self._PROMPT} {target}"
        return self._PROMPT

    def __call__(self, text, audio=None, sampling_rate=None, return_tensors=None, padding=None):
        return self.tokenizer(text, return_tensors="pt", padding=True, add_special_tokens=False)


@pytest.fixture()
def fake_processor() -> _FakeProcessor:
    return _FakeProcessor()


@pytest.fixture()
def fake_audio(tmp_path) -> str:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"")  # content irrelevant — load_audio is patched
    return str(path)


# ── WeightedTokenTrainer ──────────────────────────────────────────────────────

def _compute_loss(trainer_cls, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Call compute_loss without a fully initialised Trainer (self is unused in the body)."""
    model = MagicMock()
    outputs = MagicMock()
    outputs.logits = logits
    model.return_value = outputs
    instance = object.__new__(trainer_cls)
    return instance.compute_loss(model, {"labels": labels})


def test_weighted_loss_upweights_special_token_positions() -> None:
    vocab_size, special_id = 20, 15
    # Uniform logits → every position has the same base loss.
    # Upweighting the position where label == special_id must raise total loss.
    logits = torch.zeros(1, 6, vocab_size)
    labels = torch.tensor([[0, 1, 2, special_id, 3, 4]])

    loss_w1 = _compute_loss(WeightedTokenTrainer.build({special_id: 1.0}), logits.clone(), labels.clone())
    loss_w2 = _compute_loss(WeightedTokenTrainer.build({special_id: 2.0}), logits.clone(), labels.clone())

    assert loss_w2 > loss_w1


def test_weighted_loss_ignores_masked_positions() -> None:
    vocab_size = 20
    logits = torch.zeros(1, 6, vocab_size)
    labels = torch.tensor([[0, -100, -100, 1, 2, 3]])

    loss = _compute_loss(WeightedTokenTrainer.build({}), logits, labels)

    assert loss.isfinite() and loss > 0


def test_all_masked_labels_give_zero_loss() -> None:
    vocab_size = 20
    logits = torch.zeros(1, 4, vocab_size)
    labels = torch.full((1, 4), -100)

    loss = _compute_loss(WeightedTokenTrainer.build({}), logits, labels)

    assert loss.item() == 0.0


# ── Collator ──────────────────────────────────────────────────────────────────

def test_collator_masks_prompt_tokens_in_labels(
    fake_processor: _FakeProcessor, fake_audio: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "audiollm_d.dataset.load_audio",
        lambda path, sr: np.zeros(sr, dtype=np.float32),
    )

    collator = Qwen2AudioTranscriptionCollator(processor=fake_processor, sampling_rate=16000)
    batch = collator([{"audio_path": fake_audio, "text": "the boy is xxx"}])
    labels = batch["labels"]

    # _FakeProcessor always returns "SYS USR" as the prompt → 2 tokens.
    prompt_len = len(fake_processor.tokenizer("SYS USR", add_special_tokens=False).input_ids)
    assert (labels[0, :prompt_len] == -100).all(), "Prompt tokens are not masked in labels"


def test_collator_special_tokens_appear_in_labels(
    fake_processor: _FakeProcessor, fake_audio: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "audiollm_d.dataset.load_audio",
        lambda path, sr: np.zeros(sr, dtype=np.float32),
    )

    collator = Qwen2AudioTranscriptionCollator(processor=fake_processor, sampling_rate=16000)
    batch = collator([{"audio_path": fake_audio, "text": "the boy is (.) [laughter] taking cookies"}])
    labels = batch["labels"]

    pause_id = fake_processor.tokenizer.convert_tokens_to_ids("(.)")
    laughter_id = fake_processor.tokenizer.convert_tokens_to_ids("[laughter]")

    unmasked = labels[labels != -100]
    assert pause_id in unmasked, "(.) token ID missing from non-masked labels"
    assert laughter_id in unmasked, "[laughter] token ID missing from non-masked labels"
