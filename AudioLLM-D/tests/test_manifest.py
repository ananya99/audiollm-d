from pathlib import Path

from audiollm_d.manifest import build_manifest_records, read_jsonl, write_jsonl


def test_build_manifest_pairs_audio_and_chat_by_stem(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio" / "train"
    transcript_dir = tmp_path / "transcripts"
    audio_dir.mkdir(parents=True)
    transcript_dir.mkdir()

    (audio_dir / "sample1.wav").write_bytes(b"not a real wav but enough for manifest pairing")
    (transcript_dir / "sample1.cha").write_text(
        "\n".join(
            [
                "@ID: eng|Pitt|PAR|75;|male|ProbableAD||Participant|21||",
                "*PAR:\tum I see a cookie jar (..).",
            ]
        ),
        encoding="utf-8",
    )

    records = build_manifest_records(
        audio_dir=audio_dir,
        transcript_dir=transcript_dir,
        default_split="train",
    )

    assert len(records) == 1
    assert records[0].speaker_id == "sample1"
    assert records[0].split == "train"
    assert records[0].diagnosis == "ProbableAD"
    assert "[medium-pause]" in records[0].text


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    transcript_dir = tmp_path / "transcripts"
    output = tmp_path / "manifest.jsonl"
    audio_dir.mkdir()
    transcript_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"x")
    (transcript_dir / "a.txt").write_text("hello um world", encoding="utf-8")

    records = build_manifest_records(audio_dir=audio_dir, transcript_dir=transcript_dir)
    write_jsonl(records, output)

    loaded = read_jsonl(output)
    assert loaded[0]["text"] == "hello um world"
    assert loaded[0]["audio_path"].endswith("a.wav")
