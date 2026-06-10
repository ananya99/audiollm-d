from __future__ import annotations

import torch

from audiollm_d import inference


class _FakeFeatureExtractor:
    sampling_rate = 16000


class _FakeProcessor:
    def __init__(self) -> None:
        self.feature_extractor = _FakeFeatureExtractor()
        self.tokenizer = type("Tok", (), {"all_special_tokens": []})()
        self.last_text = None
        self.last_audio = None

    def __call__(self, *, text, audio, sampling_rate, return_tensors, padding):
        self.last_text = text
        self.last_audio = audio
        batch = len(audio)
        return {
            "input_ids": torch.ones((batch, 3), dtype=torch.long),
            "attention_mask": torch.ones((batch, 3), dtype=torch.long),
        }

    def apply_chat_template(self, conversation, add_generation_prompt, tokenize):
        return "prompt"

    def batch_decode(self, generated_ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        batch = generated_ids.shape[0]
        return [f"pred_{i}" for i in range(batch)]


class _FakeModel:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def generate(self, **inputs):
        batch = inputs["input_ids"].shape[0]
        # prompt length is 3, so append 2 generated ids
        return torch.ones((batch, 5), dtype=torch.long)


def test_transcribe_batch_with_loaded_model_batches_audio_inputs(monkeypatch) -> None:
    monkeypatch.setattr(inference, "load_audio", lambda path, sr: [str(path), sr])

    model = _FakeModel()
    processor = _FakeProcessor()
    outputs = inference.transcribe_batch_with_loaded_model(
        audio_paths=["a.wav", "b.wav", "c.wav"],
        model=model,
        processor=processor,
        max_new_tokens=16,
    )

    assert outputs == ["pred_0", "pred_1", "pred_2"]
    assert isinstance(processor.last_text, list)
    assert len(processor.last_text) == 3
    assert len(processor.last_audio) == 3
