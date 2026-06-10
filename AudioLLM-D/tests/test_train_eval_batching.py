from __future__ import annotations

from audiollm_d import train


class _FakeModel:
    def __init__(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


def test_evaluate_model_on_records_uses_generation_batches(monkeypatch) -> None:
    calls: list[list[str]] = []
    audio_to_ref = {
        "a.wav": "hello &um there",
        "b.wav": "we (.) pause",
        "c.wav": "all good",
        "d.wav": "more data",
        "e.wav": "final row",
    }

    def _fake_batch_transcribe(*, audio_paths, model, processor, max_new_tokens, do_sample, temperature):
        calls.append(list(audio_paths))
        return [audio_to_ref[path] for path in audio_paths]

    monkeypatch.setattr(train, "transcribe_batch_with_loaded_model", _fake_batch_transcribe)

    records = [{"audio_path": path, "text": text} for path, text in audio_to_ref.items()]
    out = train._evaluate_model_on_records(
        label="unit",
        model=_FakeModel(),
        processor=object(),
        records=records,
        manifest_name="fake.jsonl",
        audio_column="audio_path",
        text_column="text",
        max_new_tokens=32,
        max_samples=None,
        eval_generation_batch_size=2,
        eval_generation_do_sample=False,
        eval_generation_temperature=1.0,
    )

    assert [len(c) for c in calls] == [2, 2, 1]
    assert out["num_samples"] == 5
    assert len(out["predictions"]) == 5
