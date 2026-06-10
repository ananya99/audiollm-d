# Data Analysis — DementiaBank Pitt Corpus

## Overview

The dataset is the **DementiaBank Pitt Corpus**, a longitudinal speech corpus pairing audio recordings with CHAT-format transcripts. Subjects performed four elicitation tasks across multiple clinical sessions. The corpus is used here for dementia detection / transcription fine-tuning.

**DOI:** 10.21415/CQCW-1F92  
**Language:** English  
**Country:** United States  
**Date:** 2025 (noise-reduced release)  
**Creators:** Boller, Francois; Becker, James

---

## Directory Structure

```text
data/dementia/
  PItt-data.xlsx          # Participant clinical metadata (510 subjects, 108 cols)
  Clin-CLAN.docx          # Clinician guide: CHAT transcription & CLAN analysis
  Pitt_transcripts/       # CHAT .cha transcript files (mirrors audio structure)
    0metadata.cdc         # Corpus metadata (title, creators, DOI)
    CHAT.pdf              # Official CHAT format manual (TalkBank)
    Control/
      cookie/ fluency/ recall/ sentence/
    Dementia/
      cookie/ fluency/ recall/ sentence/
  pitt/                   # Audio files
    CLAN.pdf              # Official CLAN analysis tool manual
    Control/              # Healthy control subjects (1.4 GB)
      cookie/
        0wav/             # Noise-removed WAV variants
      fluency/ recall/ sentence/
    Dementia/             # Dementia subjects (12 GB)
      cookie/
        0wav/
      fluency/ recall/ sentence/
    0extra/               # Noise estimation files (68 MB)
      Control_noise/
      Dementia_noise/
      noiseRemoval_ADReSS/
      noiseRemoval_ADReSS.zip
      Readme.md
```

---

## File Counts

### Audio files (`pitt/`)

| Format | Count |
|--------|------:|
| MP3 (original recordings) | 1,290 |
| WAV signal (noise-removed, in `0wav/`) | 551 |
| WAV noise (reference, in `0extra/`) | 551 |
| **Total audio files** | **2,392** |

### MP3 recordings by group and task

| Group | cookie | fluency | recall | sentence | **Total** |
|-------|-------:|--------:|-------:|---------:|---------:|
| Control | 243 | 2 | 1 | 1 | **247** |
| Dementia | 309 | 235 | 263 | 236 | **1,043** |
| **Total** | **552** | **237** | **264** | **237** | **1,290** |

> Control coverage for fluency/recall/sentence is extremely sparse — these tasks are effectively Dementia-only in this download.

### CHAT transcripts (`Pitt_transcripts/`)

1,290 `.cha` files — exactly one per audio file, same group/task breakdown:

| Group | cookie | fluency | recall | sentence | **Total** |
|-------|-------:|--------:|-------:|---------:|---------:|
| Control | 243 | 2 | 1 | 1 | **247** |
| Dementia | 309 | 235 | 263 | 236 | **1,043** |
| **Total** | **552** | **237** | **264** | **237** | **1,290** |

### Noise-removed WAV files

| Location | WAV count |
|----------|----------:|
| Control/cookie/0wav | 242 |
| Dementia/cookie/0wav | 309 |
| 0extra/Control_noise | 242 |
| 0extra/Dementia_noise | 309 |

WAV files exist only for the cookie task. Each signal WAV has a matching noise reference file.

---

## Disk Usage

| Directory | Size | Files |
|-----------|-----:|------:|
| `data/dementia/pitt/` | 13 GB | ~2,400 |
| └─ `Dementia/` | 12 GB | 1,352 |
| &nbsp;&nbsp;&nbsp;&nbsp;└─ `recall/` | 8.9 GB | 263 |
| &nbsp;&nbsp;&nbsp;&nbsp;└─ `cookie/` | 2.1 GB | 618 |
| &nbsp;&nbsp;&nbsp;&nbsp;└─ `fluency/` | 529 MB | 235 |
| &nbsp;&nbsp;&nbsp;&nbsp;└─ `sentence/` | 454 MB | 236 |
| └─ `Control/` | 1.4 GB | 489 |
| └─ `0extra/` | 68 MB | 551 |

> Dementia/recall is unexpectedly large (~33.8 MB/file avg) despite being MP3-only — likely longer recording sessions.

---

## Participant Metadata — `PItt-data.xlsx`

**Source:** `data/dementia/PItt-data.xlsx`  
**Sheets:** `readme` (key), `data` (510 rows × 108 columns)

### Demographics (baseline)

| Metric | Value |
|--------|-------|
| Total participants | 510 |
| Female | 310 (61%) |
| Male | 200 (39%) |
| Entry age range | 45–91 |
| Mean entry age | 69.2 years |
| Mean MMSE at entry | 20.7 / 30 |

### Diagnostic groups (`groupdx` column)

| Code | Diagnosis | N |
|------|-----------|--:|
| 1 / 100 | Probable AD | 248 |
| 2 / 200 | Possible AD + comorbidity | 98 |
| 3 / 300 | Vascular dementia | 12 |
| 4 | Other dementia (incl. Parkinson: 420, 430; Lewy bodies: 596) | 12 |
| 5 | Subject reports problems, none diagnosed | 15 |
| 6 / 7 | Mild Cognitive Impairment (MCI) | 19 |
| 8 / 800 | Control (healthy) | 106 |

MCI subtypes — code 610/611: multi-domain MCI; 720/740: memory-only MCI; 730: psychiatric (anxiety/depression); 770: cerebrovascular disease without clinical stroke.

### Clinical assessment columns

| Column(s) | Instrument | Notes |
|-----------|------------|-------|
| `mms` / `mmse#` | Mini Mental State Exam | 0–30; tracked per visit |
| `cdrfs` / `cdr#` | Clinical Dementia Rating Scale | Functional staging |
| `blessed` / `bless#` | Blessed Dementia Scale | |
| `hamilton` / `hamilt#` | Hamilton Depression Rating Scale | |
| `htotal`, `hmtotal` / `hrs#` | Hachinski Ischemic Scale (total + modified) | |
| `mattis` / `mattis#` | Mattis Dementia Rating Scale | |
| `nyu` / `nyu#` | NYU Parkinson's Rating Scale | |
| `basedx` | Diagnosis at baseline (visit 0) | |
| `dx1`–`dx3` | Diagnosis at visits 1–3 | |
| `changedx` / `intertdx` | Date of diagnosis change / interim change | |
| `curdx1`–`curdx4` | All diagnoses at last visit | |
| `lastdate` | Date of last contact (data through 2000) | |
| `autopsy results` / `autopsy date` | Post-mortem confirmation | |

### Linking spreadsheet to `.cha` files

From the xlsx readme:
> The first 3 digits of the `.cha` filename correspond to `id` (column A).  
> The digit after the dash is the **visit index** (0 = baseline), which maps to the spreadsheet visit columns as: file `001-0.cha` = visit 1 on the spreadsheet, `001-1.cha` = visit 2, etc.

The `@ID` tier in each `.cha` file encodes: `eng|UPMC|PAR|age|gender|Dx|{fileid}|Participant|MMSE|`

Dx labels used in CHAT files vs spreadsheet codes:

| CHAT `@ID` label | Spreadsheet code |
|-----------------|-----------------|
| `ProbableAD` | 101 |
| `PossibleAD` | 200-range |
| `MCI` | 600 |
| `Memory` | 720, 740 |
| `Vascular` | 300 |
| `Control` | 821 |
| `Other` | 500, 540 |

> Age in `@ID` is exact only at baseline (visit 0). Later visits use approximated ages (±months to a year) because only first-visit age and subsequent visit dates are recorded.

---

## CHAT Transcript Format

**Reference:** `Pitt_transcripts/CHAT.pdf` (official TalkBank CHAT manual), `Clin-CLAN.docx` (clinician guide by Ratner, Brundage, Fromm; updated October 2023)

### File structure

Every `.cha` file has three sections:

**1. Headers** — appear before any utterances:

```text
@UTF8
@PID:       11312/a-00090633-0
@Begin
@Languages: eng
@Participants: PAR Participant, INV Investigator
@ID:        eng|Pitt|PAR|58;|female|Control||Participant|30||
@ID:        eng|Pitt|INV|||||Investigator|||
@Media:     002-0, audio
@G:         Cookie
```

Key header fields:
- `@Participants` — three-letter speaker code, name/ID, standard role
- `@ID` — pipe-delimited: language|corpus|speaker|age|sex|diagnosis|fileid|role|MMSE
- `@Media` — audio filename (no extension) and media type
- `@G` — task name (Cookie, Fluency, Recall, Sentence)

**2. Main lines** — prefixed with `*SPEAKER:` followed by a tab:

```text
*PAR:   the boy is taking cookies . 3754_5640
*INV:   can you tell me more ?
```

- Timestamps `start_end` (milliseconds) are appended when audio-linked
- One utterance per line; ends with `.`, `!`, or `?`
- `xxx` = unintelligible word; `(.)` = short pause; `(..)` = medium; `(...)` = long
- Retracing: `[/]` = repetition, `[//]` = revision; `[: word]` = target form
- Fillers: `&-uh`, `&-um`, etc.
- Post-codes: `[+ gram]` = grammatically correct, `[+ exc]` = exclamation

**3. Dependent tiers** — prefixed with `%tier:`, added by CLAN programs:

```text
%mor:   det|the-Def-Art noun|boy aux|be-Fin-Ind-Pres-S3 verb|take-Part-Pres-S noun|cookie-Plur .
%gra:   1|2|DET 2|4|NSUBJ 3|4|AUX 4|0|ROOT 5|4|OBJ 6|4|PUNCT
```

- `%mor` — morphosyntactic tags (auto-generated by CLAN's MOR program)
- `%gra` — dependency grammar (auto-generated by CLAN's GRA program)
- `%com` — free-text comment tier
- `%pho` — phonological coding

### Sample `.cha` excerpt (Control/cookie/002-0.cha)

```text
@ID: eng|Pitt|PAR|58;|female|Control||Participant|30||
*PAR: the scene is in the kitchen . 3754_5640
*PAR: a boy is trying to get cookies outta a jar . 12138_17928
*PAR: &-uh the little girl is reacting to his falling . 19223_23097
*INV: +< that's great .
*PAR: kitchen's very tidy . [+ gram] 39125_40315
@End
```

---

## CLAN Analysis Tool

**Reference:** `pitt/CLAN.pdf` (official manual), `Clin-CLAN.docx` (clinician guide)

CLAN (Computerized Language ANalysis) is part of the CHILDES ecosystem. It reads `.cha` files and computes linguistic measures automatically.

### Key CLAN programs

| Program | Output |
|---------|--------|
| `MOR` | Morphosyntactic tagging → inserts `%mor` tier |
| `GRA` | Dependency grammar → inserts `%gra` tier |
| `MLU` | Mean Length of Utterance (words or morphemes) |
| `FREQ` / `VOCD` | Type-token ratio, lexical diversity |
| `DSS` | Developmental Sentence Score |
| `FLUCALC` | Fluency: counts disfluency types, % stuttered words |
| `CHECK` | Validates CHAT file format before analysis |

### Audio-transcript linkage

CLAN links each transcript line to its audio segment via millisecond timestamps embedded at line ends (e.g., `3754_5640`). Pressing F4 on any bullet replays that utterance segment. The `@Media` header must exactly match the audio filename.

### Download and setup

- Download CLAN: `http://dali.talkbank.org/clan/`
- Screencasts: `https://media.talkbank.org/open/screencasts/index.html`
- Full manuals: `https://talkbank.org/`

---

## Filename Convention

**Audio MP3 and CHAT `.cha` files:**
```text
{speaker_id:03d}-{visit_index}.mp3  /  .cha
```
Examples: `001-0.mp3` (speaker 001, baseline), `010-3.mp3` (speaker 010, visit 3+1=4th visit)

**Noise-removed WAV files (`cookie/0wav/`):**
```text
{speaker_id:03d}-{visit_index}c.wav
```
Example: `002-0c.wav` (`c` = cookie task)

**Noise reference WAV files (`0extra/*_noise/`):**
```text
{speaker_id:03d}-{visit_index}c-n.wav
```
Example: `002-0c-n.wav` (matches speech file `002-0c.wav`)

**Visit index mapping (file → spreadsheet):**

| File suffix | Spreadsheet visit column |
|:-----------:|:------------------------:|
| `-0` | visit 1 (baseline) |
| `-1` | visit 2 |
| `-2` | visit 3 |
| `-3` | visit 4 |
| `-4` | visit 5 |

---

## Speaker Statistics

### Cookie Theft task (primary task)

| Metric | Control | Dementia |
|--------|--------:|---------:|
| Unique speakers | 99 | 194 |
| Total recordings | 243 | 309 |
| Avg sessions / speaker | 2.45 | 1.59 |
| Min sessions | 1 | 1 |
| Max sessions | 5 | 5 |

### Session distribution (Cookie Theft)

| Session index | Control | Dementia |
|:---:|--------:|---------:|
| 0 (baseline) | 71 | 170 |
| 1 | 67 | 77 |
| 2 | 53 | 39 |
| 3 | 36 | 17 |
| 4 | 6 | 6 |

Dementia subjects are more likely to have only one session (55% of their cookie recordings are session 0). Control subjects are more evenly distributed across visits.

### Other tasks (Dementia only)

| Task | Unique speakers | Total recordings |
|------|----------------:|-----------------:|
| fluency | 161 | 235 |
| recall | 176 | 263 |
| sentence | 159 | 236 |

---

## Noise Removal

**Scripts:** `pitt/0extra/noiseRemoval_ADReSS/`

| File | Purpose |
|------|---------|
| `noiseEnhacement.m` | Top-level script; takes noise WAV + speech WAV paths, calls `SSBOll79.m` |
| `SSBOll79.m` | Spectral subtraction implementation (Boll 1979) |
| `ccNoiseFiles.mat` | MATLAB list of Control noise filenames |
| `cdNoiseFiles.mat` | MATLAB list of Dementia noise filenames |
| `noiseRemoval_ADReSS.zip` | Full packaged archive |

**References:**
- Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction. *IEEE Transactions on acoustics, speech, and signal processing*, 27(2), 113–120.
- Implementation also at: https://uk.mathworks.com/matlabcentral/fileexchange/7675-boll-spectral-subtraction

Pre-processed noise-removed WAVs are already in `Control/cookie/0wav/` and `Dementia/cookie/0wav/`. No need to re-run MATLAB unless working on non-cookie tasks.

---

## Data Processing Pipeline

### 1. Download

See [download_data.md](download_data.md) for `wget` instructions. Requires a TalkBank account and session cookie (`talkbank_cookies.txt`).

### 2. Pair audio with transcripts

Each `.mp3` in `pitt/{Group}/{task}/` has a matching `.cha` in `Pitt_transcripts/{Group}/{task}/` with the same stem. The `@Media` header in the `.cha` file references the audio filename.

---

## Known Data Issues

| Issue | Detail |
|-------|--------|
| Control task sparsity | fluency/recall/sentence for Control have 2/1/1 files; multi-task training should use Dementia only or expect class imbalance |
| No CHAT transcripts in `pitt/` | `.cha` files live in `Pitt_transcripts/` not alongside audio — paths must be specified separately |
| Session 0 dominance in Dementia | 55% of Dementia cookie recordings are from the first visit; models may overfit to baseline presentations |
| Dementia/recall size | 263 MP3 files consume 8.9 GB (~33.8 MB/file), suggesting significantly longer recordings than other tasks |
| Age approximation in `@ID` | Only baseline age is exact; later visit ages are approximated from visit dates (±months to a year) |
| Spreadsheet visit offset | File `-0` = spreadsheet visit 1; file `-1` = visit 2, etc. — off-by-one when joining on visit number |
