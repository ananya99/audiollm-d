"""Configuration objects for AudioLLM-D training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from audiollm_d.special_tokens import DISFLUENCY_TOKENS, SPECIAL_TOKENS


@dataclass
class DataConfig:
    train_manifest: str = "data/train.jsonl"
    validation_manifest: str | None = None
    # Optional second manifest used for single-run curriculum training.
    # When set, train_manifest is treated as phase-1 (synthetic) data and
    # curriculum_original_manifest as phase-2 (original/real) data.
    curriculum_original_manifest: str | None = None
    # Optional cap for the curriculum original-manifest records.
    max_curriculum_original_samples: int | None = None
    # Keep training on train_manifest only for the first N epochs.
    curriculum_synthetic_only_epochs: int = 0
    # Linearly ramp original-data sampling over this many epochs.
    curriculum_ramp_epochs: int = 0
    # Target sampling ratio for original data after the ramp finishes.
    curriculum_final_original_ratio: float = 1.0
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    audio_column: str = "audio_path"
    text_column: str = "text"
    sampling_rate: int | None = None
    # Probability that any individual training sample is augmented.
    noise_augmentation_prob: float = 0.0
    # Which augmentation types to draw from uniformly.  Options:
    # "white_noise", "babble", "reverb", "speed", "codec".
    augmentation_types: list[str] = field(default_factory=lambda: ["white_noise"])
    # White-noise parameters (SNR in dB).
    noise_snr_db_min: float = 12.0
    noise_snr_db_max: float = 30.0
    # Babble (speech-shaped noise) parameters.
    babble_snr_db_min: float = 10.0
    babble_snr_db_max: float = 25.0
    babble_num_sources: int = 4
    # Room-impulse-response reverb parameters (RT60 in milliseconds).
    reverb_rt60_ms_min: float = 200.0
    reverb_rt60_ms_max: float = 800.0
    # Speed-perturbation parameters (multiplicative rate).
    speed_min: float = 0.85
    speed_max: float = 1.15
    # Codec-compression parameter (intermediate sample rate in Hz).
    codec_resample_rate: int = 8000


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # When false, LoRA adapters are attached only to language_model layers.
    enable_audio_lora: bool = True
    # When false, LoRA adapters are not attached to language_model layers.
    enable_llm_lora: bool = True
    # When false, multi_modal_projector is excluded from modules_to_save (fully frozen).
    train_multi_modal_projector: bool = True
    audio_encoder_top_half_only: bool = True
    use_dora: bool = False
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    modules_to_save: list[str] = field(
        default_factory=lambda: ["embed_tokens", "lm_head", "multi_modal_projector"]
    )
    # Optional v2-style encoder LoRA overrides. When any of these are set, the
    # audio_tower gets a different LoRA configuration than the LM. Lets you do
    # e.g. "LM at r=16 on q/v/k/o + MLP, encoder at r=8 on q/v only, layers 24-31"
    # in a single config. audio_encoder_layers overrides audio_encoder_top_half_only.
    audio_encoder_layers: list[int] | None = None
    audio_encoder_rank: int | None = None
    audio_encoder_alpha: int | None = None
    audio_encoder_target_modules: list[str] | None = None


@dataclass
class ParamGroupLRs:
    """Per-component learning rates for AdamW. When set on TrainConfig, builds a
    multi-group optimizer instead of HF Trainer's single-LR default. Lets you push
    the new-token rows hard while staying gentle on the pretrained encoder."""

    encoder_lora: float = 1.0e-4
    projector: float = 1.0e-3
    lm_lora: float = 2.0e-4
    new_tokens: float = 1.0e-3
    # Phase 2 auxiliary CTC head. Same tier as new-token embeddings / projector — it is
    # randomly initialised and trains from scratch.
    ctc_head: float = 1.0e-3


@dataclass
class CtcConfig:
    """Phase 2 auxiliary CTC head on the post-pooling audio-encoder features.

    When ``enabled``, training adds ``alpha * CTC`` to the focal LM loss. ``labels_dir`` points
    at the build_ctc_labels.py output (the dir containing ``labels/{clip_id}.npy`` +
    ``ctc_vocab.json``). ``classes`` overrides the default full vocab; None uses
    ``CtcVocab.default()``. The head's learning rate is ``param_group_lrs.ctc_head``."""

    enabled: bool = False
    labels_dir: str | None = None
    alpha: float = 0.3
    classes: list[str] | None = None


@dataclass
class DisfluencyEncoderConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    window_size_s: float = 0.5
    stride_s: float = 0.1


@dataclass
class WandBConfig:
    project: str | None = None
    entity: str | None = None
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    mode: str | None = None
    dir: str | None = None
    log_model: str | bool | None = None


@dataclass
class TrainConfig:
    model_name_or_path: str = "Qwen/Qwen2-Audio-7B-Instruct"
    # Optional Whisper-family checkpoint to initialize Qwen2-Audio's audio_tower.
    audio_encoder_init_model: str | None = None
    output_dir: str = "outputs/qwen2-audio-dementia-lora"
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    special_tokens: list[str] = field(default_factory=lambda: SPECIAL_TOKENS)
    disfluency_tokens: list[str] = field(default_factory=lambda: DISFLUENCY_TOKENS)
    # Uniform fallback weight applied to every disfluency token when
    # disfluency_loss_weights is not set (preserves the v2 weighted-CE behavior).
    disfluency_loss_weight: float = 2.0
    label_smoothing_factor: float = 0.0
    # Weight for the encoder silence-frame BCE auxiliary loss (0 = disabled).
    silence_aux_weight: float = 0.0
    # Which encoder transformer layer to apply the frame classification loss on.
    # -1 = final encoder output (last_hidden_state); 0..N = specific layer index.
    # Lower layers encode local acoustic features; prefer values in [4, 16] for
    # a 32-layer Whisper encoder to avoid conflicting with LM-facing representations.
    silence_encoder_layer: int = -1
    # Sliding-window disfluency encoder (None = disabled).
    disfl_encoder: DisfluencyEncoderConfig | None = None
    # Weight for the window-level disfluency classification auxiliary loss.
    disfl_aux_weight: float = 0.0
    # Learning-rate multiplier for the DisfluencyEncoder (trained from scratch).
    disfl_encoder_lr_multiplier: float = 10.0
    # Directory containing {clip_id}_alignment.json files.  Used to compute
    # global class weights for the window CE loss before training starts.
    # Falls back to the alignment_path column in the manifest when None.
    disfl_alignment_dir: str | None = None
    # Path to a pre-trained DisfluencyEncoder state_dict (.pt file).
    # When set, encoder weights are loaded before joint LoRA training starts.
    disfl_encoder_pretrained_path: str | None = None
    # Frequency-tiered per-token alpha weights, keyed by token string. When set,
    # overrides the uniform weight for the listed tokens; any disfluency token not
    # listed falls back to disfluency_loss_weight. Logged at startup.
    disfluency_loss_weights: dict[str, float] | None = None
    # Focal-loss focusing parameter. gamma=0 -> plain (tiered) weighted CE;
    # gamma=2 -> focal loss with the (1-p)^gamma down-weighting of easy tokens.
    focal_gamma: float = 0.0
    # Optional list of token strings the focal modulation is RESTRICTED to (v3.5).
    # When None (default) focal applies to every target — the v3 behaviour. When set,
    # only these tokens are focal-modulated; all others keep flat (alpha) weighted CE.
    focal_tokens: list[str] | None = None
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    # Optional stack-specific rates. When unset, the global learning_rate is used.
    audio_encoder_learning_rate: float | None = None
    llm_learning_rate: float | None = None
    num_train_epochs: float = 3.0
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    use_4bit: bool = False
    seed: int = 42
    report_to: str | list[str] = "none"
    run_pre_post_train_eval: bool = False
    # How often (in epochs) to run generation-based eval (WER + edit-F1) during
    # training. The sweep selects the best-epoch checkpoint from these points, so
    # set it to align with save_steps when you need per-checkpoint selection.
    epoch_eval_interval: int = 5
    # Earliest epoch at which the in-training generation eval is allowed to run.
    # 0 (default) keeps the original behaviour (eval from the first interval hit).
    # Set e.g. 10 to skip the cheap-but-slow early evals and only score the tail.
    eval_start_epoch: int = 0
    pre_post_eval_max_samples: int | None = None
    pre_post_eval_max_new_tokens: int = 256
    eval_generation_batch_size: int = 4
    eval_generation_do_sample: bool = False
    eval_generation_temperature: float = 1.0
    run_lora_init_sanity_check: bool = False
    lora_init_sanity_check_max_samples: int = 16
    lora_init_sanity_check_split: str = "test"
    lora_init_sanity_check_max_new_tokens: int = 256
    wandb: WandBConfig = field(default_factory=WandBConfig)
    # When set, build a multi-group AdamW (encoder/projector/LM/new_tokens/ctc_head)
    # instead of HF Trainer's single-LR default. None preserves main's behavior.
    param_group_lrs: ParamGroupLRs | None = None
    # Phase 2 auxiliary CTC head. Disabled by default (Phase 1 configs unaffected).
    ctc: CtcConfig = field(default_factory=CtcConfig)


def _construct_dataclass(cls, values: dict[str, Any]):
    field_types = {
        field_name: field_def.type for field_name, field_def in cls.__dataclass_fields__.items()
    }
    kwargs = {}
    for key, value in values.items():
        if key == "data" and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(DataConfig, value)
        elif key == "lora" and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(LoRAConfig, value)
        elif key == "wandb" and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(WandBConfig, value)
        elif key == "disfl_encoder" and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(DisfluencyEncoderConfig, value)
        elif key == "param_group_lrs" and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(ParamGroupLRs, value)
        elif key == "ctc" and isinstance(value, dict):
            kwargs[key] = _construct_dataclass(CtcConfig, value)
        elif key in field_types:
            kwargs[key] = value
    return cls(**kwargs)


def _resolve_config_path(path_value: str, *, config_dir: Path) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return str(path.resolve())


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_train_config_path(path: str | Path) -> Path:
    """Resolve a training YAML path from cwd, project root, or configs/."""
    raw = Path(path).expanduser()
    stem = raw.stem or raw.name
    project_root = _project_root()

    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                Path.cwd() / raw,
                project_root / raw,
                project_root / "configs" / raw.name,
                project_root / "configs" / stem,
            ]
        )

    expanded: list[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if candidate.suffix not in {".yaml", ".yml"}:
            expanded.append(candidate.with_suffix(".yaml"))
            expanded.append(candidate.with_suffix(".yml"))

    seen: set[Path] = set()
    for candidate in expanded:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(
        f"Training config not found: {path}. "
        f"Tried under cwd ({Path.cwd()}), project root ({project_root}), "
        "and configs/ with optional .yaml extension."
    )


def load_train_config(path: str | Path) -> TrainConfig:
    config_path = resolve_train_config_path(path)
    with config_path.open("r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}
    config = _construct_dataclass(TrainConfig, values)
    config_dir = config_path.parent
    config.output_dir = _resolve_config_path(config.output_dir, config_dir=config_dir)
    config.data.train_manifest = _resolve_config_path(config.data.train_manifest, config_dir=config_dir)
    if config.data.curriculum_original_manifest:
        config.data.curriculum_original_manifest = _resolve_config_path(
            config.data.curriculum_original_manifest,
            config_dir=config_dir,
        )
    if config.data.validation_manifest:
        config.data.validation_manifest = _resolve_config_path(
            config.data.validation_manifest,
            config_dir=config_dir,
        )
    if config.ctc.labels_dir:
        config.ctc.labels_dir = _resolve_config_path(config.ctc.labels_dir, config_dir=config_dir)
    return config
