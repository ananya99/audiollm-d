# AudioLLM-D

LoRA fine-tuning scaffold for adapting `Qwen/Qwen2-Audio-7B-Instruct` to dementia
speech transcription.

The project proposal targets DementiaBank Pitt-style data: audio paired with CHAT
transcripts, speaker-disjoint splits, and transcription that preserves fillers, repetitions,
and false starts. Public dataset descriptions indicate that Pitt/DementiaBank
uses `.wav` audio and `.cha` CHAT transcripts, while ADReSS-style releases often organize
Cookie Theft audio, normalized chunks, transcripts, and demographic metadata into train/test
folders.

## Install

```bash
cd AudioLLM-D
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For 4-bit training, install a CUDA-compatible `bitsandbytes` separately and set
`use_4bit: true` in the config. Full Qwen2-Audio-7B training expects a GPU environment.

## Expected Data

The manifest format is JSONL with one transcription example per line:

```json
{"audio_path": "/path/to/001.wav", "text": "uh the boy is (.) taking cookies", "speaker_id": "001", "split": "train"}
```

To build a manifest from paired audio/transcript folders:

```bash
python scripts/build_manifest.py \
  --audio-dir /path/to/audio/train \
  --transcript-dir /path/to/transcripts/train \
  --output data/train.jsonl \
  --default-split train
```

The parser keeps only participant (`*PAR:`) utterances from CHAT files and preserves CHAT pause
markers (`(.)`, `(..)`, `(...)`) as raw text so pause-token handling can be implemented in tokenizer logic.
If you later produce Montreal Forced Aligner segment boundaries, write those segment-level rows
directly into the same JSONL schema.

## Train LoRA

Edit `configs/qwen2_audio_lora.yaml`, then run:

```bash
python scripts/train_lora.py --config configs/qwen2_audio_lora.yaml
```

The training code:

- Initializes added token embeddings from the mean existing embedding.
- Applies PEFT LoRA to Qwen attention/MLP projection modules.
- Keeps embeddings, LM head, and multimodal projection trainable via `modules_to_save`.
- Masks the user prompt in labels so loss is applied to assistant transcription tokens.

## Inference

```bash
python scripts/transcribe.py \
  --audio /path/to/sample.wav \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --adapter outputs/qwen2-audio-dementia-lora
```

## Tests

```bash
pytest
```

The tests cover CHAT parsing, manifest construction, and config loading without downloading
the model.
