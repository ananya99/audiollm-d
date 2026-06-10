# AudioLLM-D — Disfluency-Aware Dementia Speech Transcription

> CS-552 (MNLP) Open Project · Milestone 2 · Team *lordofthestrings*

Fine-tuning [`Qwen/Qwen2-Audio-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
with LoRA to transcribe dementia speech **while preserving disfluency markers** — pauses,
fillers, repetitions, and false starts — which are clinically meaningful signals that standard
ASR systems discard.

---

## ⚠️ Dataset is private — not included in this repository

This project trains on the **DementiaBank Pitt Corpus** (TalkBank, DOI `10.21415/CQCW-1F92`)
and **ADReSS-IS2020** data. **These datasets are not, and must not be, committed to GitHub.**

- They are governed by the [TalkBank Data Use Agreement](https://talkbank.org/share/rules.html)
  and contain **protected clinical/health speech data**. Redistribution is prohibited.
- Access requires **TalkBank membership credentials** — you must request access and download
  the data yourself.
- All data paths are already excluded via [.gitignore](.gitignore):
  `data/`, `ADReSS-IS2020-data/`, `preprocessed_data/`, `processed_data/`, `outputs/`, `reports/`.

**Before cloning, configuring, or pushing: never `git add` anything under these paths.**
Each contributor downloads the corpus locally (see [docs/download_data.md](docs/download_data.md))
and points the configs at their own local copy. Only code, configs, notebooks, and aggregate
artifacts (e.g. token-distribution stats) belong in version control.

---

## What this project does

| | |
|---|---|
| **Task** | Speech-to-text transcription of dementia patient recordings, preserving disfluencies |
| **Base model** | Qwen2-Audio-7B-Instruct (audio-language model) |
| **Fine-tuning** | LoRA / DoRA adapters via [PEFT](https://github.com/huggingface/peft) |
| **Key idea** | Special disfluency tokens + Focal Loss + an auxiliary CTC head to handle rare markers |
| **Evaluation** | Word Error Rate (WER), order-aware disfluency F1, and a question-type-aware QA pipeline |

**Disfluency tokens** such as `[short-pause]`, `[medium-pause]`, `[long-pause]`, `[unclear-speech]`,
and `[laughter]` are added to the vocabulary. Because these tokens are rare, training combines a
**disfluency-aware focal loss** (upweighting hard/rare tokens) with an **auxiliary CTC head** that
performs frame-level disfluency classification on the audio encoder features.

## Repository layout

```text
.
├── AudioLLM-D/          # Main package: training, inference, losses, metrics, data tooling
│   ├── src/audiollm_d/  # Library code (train, inference, loss, ctc_head, qa_metrics, ...)
│   ├── configs/         # YAML experiment configs (LoRA variants, baselines)
│   ├── scripts/         # CLI entry points (build_manifest, train_lora, transcribe)
│   └── tests/           # Unit tests (CHAT parsing, manifests, config — no model download)
├── notebooks/           # One notebook per team member (milestone deliverable)
├── docs/                # Data download, dataset analysis, CHAT parsing notes
├── report/              # Milestone report and deliverables
├── rcp_support/         # EPFL RCP cluster guide (runai jobs, Docker images)
└── outputs/, data/      # Local-only — gitignored (see warning above)
```

## Quickstart

> Requires Python ≥ 3.10 and (for training) a CUDA GPU. Full Qwen2-Audio-7B fine-tuning is
> GPU-bound — see [rcp_support/](rcp_support) for running on the EPFL RCP cluster.

```bash
cd AudioLLM-D
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 1. Get the data (private — see warning above)

Follow [docs/download_data.md](docs/download_data.md) to download the DementiaBank Pitt corpus
into a local `data/` folder. Then build a JSONL manifest from paired audio/transcript folders:

```bash
python scripts/build_manifest.py \
  --audio-dir /path/to/audio/train \
  --transcript-dir /path/to/transcripts/train \
  --output data/train.jsonl \
  --default-split train
```

Manifest schema (one transcription example per line):

```json
{"audio_path": "/path/to/001.wav", "text": "uh the boy is (.) taking cookies", "speaker_id": "001", "split": "train"}
```

The parser keeps only participant (`*PAR:`) utterances and preserves CHAT pause markers
(`(.)`, `(..)`, `(...)`) as raw text for tokenizer-level disfluency handling.

### 2. Train

```bash
python scripts/train_lora.py --config configs/qwen2_audio_lora.yaml
```

The training pipeline applies PEFT LoRA to Qwen attention/MLP projections, keeps embeddings,
LM head, and the multimodal projection trainable, masks the user prompt in labels, and applies
the disfluency-focal loss (+ optional CTC head). Several config variants live in `configs/`
(e.g. `qwen2_audio_lora_v3_5.yaml`, `crisper_whisper_baseline.yaml`, `whisper_large_baseline.yaml`).

### 3. Transcribe

```bash
python scripts/transcribe.py \
  --audio /path/to/sample.wav \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --adapter outputs/qwen2-audio-dementia-lora
```

### Run tests

```bash
cd AudioLLM-D && pytest
```

Tests cover CHAT parsing, manifest construction, and config loading without downloading the model.

## Key dependencies

`torch ≥ 2.3` · `transformers ≥ 4.45` · `peft ≥ 0.12` · `accelerate ≥ 0.33` ·
`datasets ≥ 2.20` · `librosa ≥ 0.10` · `jiwer ≥ 3.0` · `wandb` — see
[AudioLLM-D/pyproject.toml](AudioLLM-D/pyproject.toml) for the full list.

## Team

| Member | SCIPER |
|---|---|
| Ananya Gupta | 390080 |
| Sathvik Bhagavan | 389191 |
| Tuan Linh Phan | 409551 |
| Yasaman Noorikhah | 396449 |

## Acknowledgements & data citation

- **DementiaBank** / TalkBank Pitt Corpus — Becker et al., *The natural history of Alzheimer's
  disease.* (DOI `10.21415/CQCW-1F92`). Used under the TalkBank Data Use Agreement.
- **ADReSS-IS2020** Challenge dataset (Luz et al.).
- Base model: Qwen2-Audio (Alibaba/Qwen). Optional encoder init from CrisperWhisper.

Project documentation and rubric: see the
[open project description](https://docs.google.com/document/d/1NI4UKsasYuFLxOGGzsAweCbW0XOEtc59/edit).
