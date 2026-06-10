# CHAT Parsing

This document explains every CHAT annotation type encountered in the Pitt Corpus, the decision made for the current **audio-fidelity pre-training** goal, and what to change if the goal shifts to **disfluency-aware dementia detection**.

The guiding principle for pre-training is: **the target text must match what is audible in the recording**. Anything the transcriber added that has no acoustic correlate is stripped. Anything spoken is kept.

---

## 1. Timestamps

| Raw | Current output | Regex |
|-----|---------------|-------|
| `\x15NNN_NNN\x15` (CLAN bullet) | removed | `_CONTROL_RE` |
| `NNN_NNN` at line end | removed | `_TIMESTAMP_RE` |

**Why removed:** Audio-linkage bookkeeping added by the CLAN editor. No speech content.

**For dementia detection:** No change needed. Timestamps are tooling artefacts in both goals.

---

## 2. `[: target]` — Transcriber normalisation annotation

| Raw | Current output |
|-----|---------------|
| `outta [: out of]` | `outta` |
| `hafta [: have to]` | `hafta` |
| `gonna [: going to]` | `gonna` |

**Why keep original word:** The speaker said "outta". The audio model must learn to map that acoustic signal to "outta", not "out of". Using the target form would misalign audio and text.

**For dementia detection:** Still no change. Non-standard pronunciations and catenatives are not dementia markers — they are regional/register variation. The original spoken form is the right label in both goals.

---

## 3. Retraces — `<attempt> [/]` and `<attempt> [//]`

| Code | Meaning | Raw | Current output |
|------|---------|-----|---------------|
| `[/]` | Repetition — speaker repeated the same thing | `<in the> [/] in the kitchen` | `in the in the kitchen` |
| `[//]` | Revision — speaker abandoned and corrected | `<a child is going> [//] a boy is going` | `a child is going a boy is going` |

**Why keep both attempts:** The audio contains both the abandoned attempt and the correction. Dropping the attempt would produce a target that doesn't match the audio, giving the model a contradictory gradient.

**For dementia detection:** This is the highest-value annotation type to reconsider. Retracing frequency and type are established dementia markers. Two options:

- **Special tokens:** `<rep> in the </rep> in the kitchen` and `<rev> a child is going </rev> a boy is going` — lets the model learn to label disfluency type. Requires adding tokens to the vocabulary and enough data to train them reliably (~550 cookie recordings is marginal).
- **Separate classification head:** Keep the text as-is for the ASR head; add a token-level disfluency-type label as a second output. Cleaner separation of concerns.

The special-token approach is only worth pursuing once the base transcription is solid.

---

## 4. Bare `[/]` repetitions without `< >`

| Raw | Current output |
|-----|---------------|
| `she [/] she has a plate` | `she she has a plate` |
| `action [/] action [/] action` | `action action action` |

**Why:** `_SCOPE_RE` strips the `[/]` markers and leaves the repeated words. The audio has the words twice (or more), so the doubled text is the correct target.

**For dementia detection:** Same as §3 — if labelling retraces, bare `[/]` should get the same `<rep>` treatment. The challenge is that without `< >`, the boundary of the retrace is ambiguous (only the immediately preceding word is conventionally implied). For now, keeping doubled text is correct.

---

## 5. `&=event` — Non-verbal event codes

| Raw | Current output |
|-----|---------------|
| `&=laughs` | removed |
| `&=coughs` | removed |
| `&=head:shake` | removed |
| `&=clears:throat` | removed |

**Why removed:** These are transcriber annotations of audible but non-lexical sounds. There is no corresponding text to output for them. The model is trained to transcribe words; it cannot and should not output `&=laughs`.

**For dementia detection:** Some events — particularly `&=clears:throat`, `&=sighs`, `&=laughs` — could be relevant paralinguistic markers. If a future model is trained to produce a richer output (e.g. a structured disfluency report), these could be preserved as tokens. For an ASR-based pipeline this is out of scope.

---

## 6. `&+word` — Phonological fragments

| Raw | Current output |
|-----|---------------|
| `&+dri` | removed |
| `&+sh` | removed |
| `&+sau` | removed |

**Why removed:** `&+` marks a partial utterance the transcriber could not resolve to a word. Like `&=event`, there is no valid text representation. Handled by the same `_EVENT_RE = re.compile(r"&[=+]\S+")`.

**For dementia detection:** High-frequency phonological fragments are a potential marker of word-finding difficulties. If labelling is added, fragments could become `<fragment>` tokens. For now, removal is correct.

---

## 7. `&-word` — Lexical fillers

| Raw | Current output |
|-----|---------------|
| `&-uh` | `uh` |
| `&-um` | `um` |
| `&-er` | `er` |

**Why keep the word:** `&-` marks a spoken disfluency filler. "uh" and "um" are actually in the audio. Stripping them would misalign text and audio. The `&-` prefix is CHAT notation, not speech.

**For dementia detection:** `uh`/`um` rate and placement are established dementia markers. Keeping them as plain words already preserves this signal for any downstream analysis of the transcript. No change needed — this decision serves both goals.

---

## 8. Trailing partial-word elisions — `word(letters)`

| Raw | Current output |
|-----|---------------|
| `tryin(g)` | `trying` |
| `runnin(g)` | `running` |
| `goin(g)` | `going` |
| `word()` | `word` |

**Why expand:** The transcriber writes the canonical form and parenthesises the dropped sound. Expanding gives the conventional written form of the word, which is what a standard ASR model would output. The acoustic signal is the reduced pronunciation, but the text target is the canonical word — consistent with how Qwen2-Audio was pre-trained.

**For dementia detection:** Phonological reduction (dropping word endings) may be a marker, but capturing it requires phonetic-level analysis, not just transcript text. The canonical word form is the right text target in both goals.

---

## 9. Leading partial-word elisions — `(letters)word`

| Raw | Current output |
|-----|---------------|
| `(be)cause` | `cause` |
| `(a)bout` | `bout` |

**Why keep only the spoken part:** The speaker said "cause", not "because". The parenthesised prefix is what was omitted. Keeping `(be)cause` in the text would put characters in the target that have no acoustic correlate. The spoken portion — the part after the parentheses — is the correct text.

**For dementia detection:** No change needed. The omitted prefix is a phonological reduction, not a lexical substitution error. The spoken form is the right label either way.

---

## 10. `[scope]` annotations — Post-codes and error markers

| Annotation | Meaning | Current output |
|------------|---------|---------------|
| `[+ gram]` | Grammatically well-formed | removed |
| `[+ exc]` | Exclamation judgement | removed |
| `[*]` | Transcriber error flag | removed |
| `[?]` | Uncertain transcription | removed |
| `[=! action]` | Paralinguistic description | removed |

**Why removed:** All are transcriber metalinguistic labels applied after the fact. None have acoustic correlates. The words they annotate are kept; only the brackets are stripped.

**For dementia detection:** `[*]` (grammatical error) and `[?]` (uncertain transcription) could be useful labels. A future structured output could flag the preceding word as erroneous or uncertain. Not worth adding until the base model transcribes well enough that these distinctions matter.

---

## 11. Pause markers

| Raw | Current output (`keep_pause_tokens=True`) | Output (`keep_pause_tokens=False`) |
|-----|------------------------------------------|------------------------------------|
| `(.)` | `<short_pause>` | *(removed)* |
| `(..)` | `<medium_pause>` | *(removed)* |
| `(...)` | `<long_pause>` | *(removed)* |

**Why keep as tokens:** Pause placement and duration are among the strongest acoustic and linguistic markers of dementia. A model that learns to output `<short_pause>` at the right moments is learning something clinically meaningful about the speech. The `keep_pause_tokens=False` flag is available for ablation experiments.

**For dementia detection:** Pause tokens are already the right representation. A natural extension is to distinguish filled pauses (`&-uh` in the middle of a phrase) from unfilled pauses `(.)` — the current parser already keeps both, so any downstream classifier can use both signals from the transcript text.

---

## 12. Utterance-overlap and continuation markers

| Marker | Meaning | Current output |
|--------|---------|---------------|
| `+<` | Utterance overlaps with prior | prefix stripped, content kept |
| `+>` | Utterance continues after overlap | prefix stripped, content kept |
| `+,` | Lazy continuation | prefix stripped, content kept |
| `+^` | Quick uptake | prefix stripped, content kept |
| `+...` | Trailing off / utterance incomplete | removed entirely |
| `+/.` | Interrupted | removed entirely |
| `+//.` | Self-interrupted | removed entirely |

**Why:** Overlap markers prefix real utterances — the speech is in the audio. The marker is CHAT notation. Trailing-off and interruption markers have no following text content to keep.

**For dementia detection:** Self-interruption (`+//.`) is related to revision (`[//]`). If revision gets special tokens, self-interruption could too. Not a priority until §3 is resolved.

---

## 13. Terminal punctuation

| Raw | Current output |
|-----|---------------|
| `the boy is falling .` | `the boy is falling.` |
| `you want more ?` | `you want more?` |
| `great !` | `great!` |

**Why keep and reattach:** Qwen2-Audio is pre-trained to output punctuation. Stripping it from fine-tuning targets would create a distribution mismatch and train the model against its own pre-trained behaviour. The CHAT convention of writing `word .` with a preceding space is collapsed to `word.` to match standard ASR output format.

**For dementia detection:** No change needed. Sentence-final punctuation is the same in both goals.

---

## 14. Speaker filtering (`INV` vs `PAR`)

Only `*PAR:` turns are kept. `*INV:` (investigator), `*FAM:` (family member), and any other speaker codes are discarded.

**Why:** The model is trained on patient speech. Investigator prompts are not part of the transcription task and would confuse the audio-text alignment (the investigator's audio segment would have no corresponding text in the target, or vice versa).

**For dementia detection:** No change. The dementia signal lives entirely in the patient's speech.

---

## 15. `xxx` — Unintelligible speech

| Raw   | Current output     |
|-------|--------------------|
| `xxx` | `<unintelligible>` |

**Why token not removal:** The audio contains a sound the transcriber could not resolve to words. Removing `xxx` would shorten the target relative to the audio, giving the model a misaligned gradient. A dedicated token teaches the model to output something for the unintelligible segment rather than silently skipping it.

**For dementia detection:** `<unintelligible>` frequency is a direct severity marker — patients with more advanced dementia produce more unintelligible speech. The token already preserves this signal for any downstream classifier.

---

## Summary table

| Annotation | Pre-training decision | Dementia-detection reconsideration |
|------------|----------------------|------------------------------------|
| Timestamps | Remove | — |
| `[: target]` | Keep original spoken word | — |
| `<...> [/]` `[//]` retraces | Keep both attempts as plain text | Add `<rep>`/`<rev>` tokens |
| Bare `[/]` repetitions | Keep doubled words | Add `<rep>` token before repeated word |
| `&=event` non-verbal | Remove | Consider `<event>` token for paralinguistics |
| `&+word` fragments | Remove | Consider `<fragment>` token |
| `&-word` fillers | Keep word (`uh`, `um`) | Already useful; no change |
| Trailing elisions `word(s)` | Expand to canonical form | — |
| Leading elisions `(s)word` | Keep spoken part only | — |
| `[scope]` post-codes | Remove | `[*]` → error flag; `[?]` → uncertainty flag |
| Pause markers `(.)` `(..)` `(...)` | `<short/medium/long_pause>` tokens | Already optimal; consider duration quantisation |
| Overlap markers `+<` `+>` | Strip prefix, keep content | — |
| Trailing-off `+...` `+/.` `+//.` | Remove | `+//.` → could flag as self-interruption |
| Terminal punctuation | Keep, reattach to word | — |
| Speaker filter | PAR only | — |
| `xxx` unintelligible | `<unintelligible>` token | Already optimal for both goals |
