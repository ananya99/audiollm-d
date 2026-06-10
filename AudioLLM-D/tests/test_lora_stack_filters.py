from audiollm_d.train import (
    _build_modules_to_save,
    _filter_lora_target_modules,
    _optimizer_stack_for_parameter,
    _split_lora_targets_by_stack,
)


def test_filter_lora_target_modules_llm_only() -> None:
    resolved = [
        "audio_tower.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.mlp.gate_proj",
    ]
    filtered = _filter_lora_target_modules(resolved, enable_audio_lora=False)
    buckets = _split_lora_targets_by_stack(filtered)

    assert filtered == [
        "language_model.model.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.mlp.gate_proj",
    ]
    assert buckets["audio"] == []
    assert len(buckets["llm"]) == 2


def test_filter_lora_target_modules_preserves_order() -> None:
    resolved = [
        "audio_tower.layers.1.self_attn.k_proj",
        "language_model.model.layers.1.self_attn.k_proj",
        "audio_tower.layers.0.self_attn.q_proj",
        "language_model.model.layers.0.self_attn.q_proj",
    ]
    filtered = _filter_lora_target_modules(resolved, enable_audio_lora=False)

    assert filtered == [
        "language_model.model.layers.1.self_attn.k_proj",
        "language_model.model.layers.0.self_attn.q_proj",
    ]


def test_build_modules_to_save_projector_flag() -> None:
    requested = ["embed_tokens", "lm_head", "multi_modal_projector"]

    assert _build_modules_to_save(requested, train_multi_modal_projector=True) == requested
    assert _build_modules_to_save(requested, train_multi_modal_projector=False) == [
        "embed_tokens",
        "lm_head",
    ]


def test_optimizer_stack_for_parameter() -> None:
    assert (
        _optimizer_stack_for_parameter(
            "base_model.model.audio_tower.encoder.layers.0.self_attn.q_proj.lora_A.default.weight"
        )
        == "audio_encoder"
    )
    assert (
        _optimizer_stack_for_parameter(
            "base_model.model.language_model.model.layers.0.self_attn.q_proj.lora_A.default.weight"
        )
        == "llm"
    )
    assert (
        _optimizer_stack_for_parameter(
            "base_model.model.multi_modal_projector.modules_to_save.default.weight"
        )
        == "other"
    )
