from pathlib import Path

from audiollm_d.config import load_train_config


def test_load_train_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "model_name_or_path: Qwen/Qwen2-Audio-7B-Instruct",
                "output_dir: outputs/test",
                "data:",
                "  train_manifest: data/train.jsonl",
                "lora:",
                "  r: 8",
                "  alpha: 16",
                "  use_dora: true",
                "audio_encoder_learning_rate: 0.00002",
                "llm_learning_rate: 0.00008",
                "report_to: wandb",
                "wandb:",
                "  project: audiollm-d-test",
                "  name: test-run",
                "  tags:",
                "    - abc",
            ]
        ),
        encoding="utf-8",
    )

    config = load_train_config(config_path)

    assert config.output_dir == str((tmp_path / "outputs/test").resolve())
    assert config.data.train_manifest == str((tmp_path / "data/train.jsonl").resolve())
    assert config.lora.r == 8
    assert config.lora.alpha == 16
    assert config.lora.use_dora is True
    assert config.audio_encoder_learning_rate == 0.00002
    assert config.llm_learning_rate == 0.00008
    assert config.report_to == "wandb"
    assert config.wandb.project == "audiollm-d-test"
    assert config.wandb.name == "test-run"
    assert config.wandb.tags == ["abc"]
