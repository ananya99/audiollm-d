"""Frame-level CTC label vocabulary and rasterization (Phase 2.1).

The auxiliary CTC head is supervised with frame-level *acoustic-event* labels, one label
per 40 ms audio frame (25 fps, matching the post-pooling Qwen2-Audio encoder grid). Labels
are derived from the per-clip MFA alignment JSONs produced upstream — each a list of
``{"word", "start", "end", ["type": "silence"]}`` segments in seconds. We do **not** re-run
MFA; we rasterize those segments onto the frame grid.

Two artefacts come out of an alignment:

  * a dense ``frame_labels`` array (one class id per frame) used for the 2.4 visualization
    ground-truth and the data-loader frame-count assertion, and
  * the run-length-collapsed symbol sequence (``collapse_runs``) that is the *target* fed to
    ``torch.nn.functional.ctc_loss`` — CTC does its own frame alignment internally, so it wants
    the collapsed event sequence, not the dense labels.

``blank`` is a real vocab index (the last one) reserved for CTC's internal blank; it never
appears in ``frame_labels`` or in a collapsed target.

Documented simplifications (the alignment JSONs are noisy; per the ticket we pick a
deterministic rule rather than chase every edge case):

  * **Pause class comes from the transcript token, not the re-measured gap duration.** A
    ``[long-pause]`` segment is labelled ``long-pause`` even if its measured gap is 0.7 s. This
    keeps the CTC pause labels *consistent with the LM head's targets* (the DoD metric is the LM
    head's bracket-pause F1), which matters more for an auxiliary objective than acoustic-duration
    purity.
  * **Pause frames render only where there is real acoustic silence.** The upstream alignment
    gives each annotated pause a measured ``[start, end)``; CHAT pauses with no acoustic gap come
    through at ~0 s duration and therefore produce *no* frames — annotation noise is filtered by
    acoustic reality.
  * **Unannotated / absorbed silences fall to ``speech``.** The upstream "make-continuous" step
    folded un-annotated inter-word silence into the preceding word, so only transcript-annotated
    pauses survive as silence segments; everything else is ``speech``. This too is consistent with
    the LM head, which emits no pause token at those positions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# The disfluency bracket tokens, as they appear in the alignment JSON "word" field.
_SHORT_PAUSE = "[short-pause]"
_MEDIUM_PAUSE = "[medium-pause]"
_LONG_PAUSE = "[long-pause]"
_UNCLEAR = "[unclear-speech]"
_LAUGHTER = "[laughter]"
_NON_VERBAL = "[non-verbal]"
_UH = "[uh]"
_UM = "[um]"

SPEECH_CLASS = "speech"
BLANK_CLASS = "blank"

# Bracket token -> candidate acoustic-event class. A bracket only becomes its own class if that
# class is present in the configured vocab; otherwise it folds into ``speech`` (see
# ``CtcVocab.class_for_word``). ``[uh]``/``[um]`` are deliberately absent here: they are spoken
# filler, handled by the LM head, and always map to ``speech`` in the CTC head (per the ticket).
_BRACKET_TO_CLASS: dict[str, str] = {
    _SHORT_PAUSE: "short-pause",
    _MEDIUM_PAUSE: "medium-pause",
    _LONG_PAUSE: "long-pause",
    _UNCLEAR: "unclear-speech",
    _LAUGHTER: "laughter",
    _NON_VERBAL: "non-verbal",
}

# Full default vocab (user decision): all bracket-event classes that cleared the Phase 1.1
# frequency audit, plus ``speech``. ``blank`` is appended automatically as the last index.
DEFAULT_EVENT_CLASSES: list[str] = [
    SPEECH_CLASS,
    "short-pause",
    "medium-pause",
    "long-pause",
    "unclear-speech",
    "laughter",
    "non-verbal",
]

FRAME_SECONDS = 0.04  # 40 ms per frame == 25 fps, the post-pooling encoder grid.


@dataclass(frozen=True)
class CtcVocab:
    """Ordered CTC class vocabulary with ``blank`` as the final index.

    ``classes`` are the emittable acoustic-event symbols (``speech`` first by convention);
    ``blank`` is implicit and equals ``len(classes)``. ``class_to_index`` covers the event
    classes only — ``blank`` is never a target/frame label.
    """

    classes: tuple[str, ...]

    def __post_init__(self) -> None:
        if SPEECH_CLASS not in self.classes:
            raise ValueError(f"CTC vocab must include {SPEECH_CLASS!r}; got {self.classes}")
        if BLANK_CLASS in self.classes:
            raise ValueError("Do not list 'blank' in classes; it is appended automatically.")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError(f"Duplicate CTC classes: {self.classes}")

    @classmethod
    def default(cls) -> "CtcVocab":
        return cls(tuple(DEFAULT_EVENT_CLASSES))

    @classmethod
    def from_classes(cls, classes: list[str] | tuple[str, ...] | None) -> "CtcVocab":
        if not classes:
            return cls.default()
        return cls(tuple(classes))

    @property
    def blank_index(self) -> int:
        return len(self.classes)

    @property
    def vocab_size(self) -> int:
        """Number of CTC logits = event classes + 1 blank."""
        return len(self.classes) + 1

    @property
    def class_to_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.classes)}

    @property
    def index_to_class(self) -> dict[int, str]:
        mapping = {idx: name for idx, name in enumerate(self.classes)}
        mapping[self.blank_index] = BLANK_CLASS
        return mapping

    def class_for_word(self, word: str) -> int:
        """Map an alignment ``word`` token to its CTC class index.

        Pause/event brackets map to their class when that class is in the vocab, else to
        ``speech``. ``[uh]``/``[um]`` and all ordinary words map to ``speech``.
        """
        index = self.class_to_index
        if word in _BRACKET_TO_CLASS:
            cls_name = _BRACKET_TO_CLASS[word]
            return index.get(cls_name, index[SPEECH_CLASS])
        return index[SPEECH_CLASS]

    def to_json(self) -> dict[str, Any]:
        return {
            "classes": list(self.classes),
            "blank_class": BLANK_CLASS,
            "blank_index": self.blank_index,
            "vocab_size": self.vocab_size,
            "frame_seconds": FRAME_SECONDS,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "CtcVocab":
        return cls(tuple(payload["classes"]))


def n_frames_for_duration(duration_seconds: float) -> int:
    """Frame count for a clip: ``ceil(duration / 0.04)`` (the 2.1 acceptance length)."""
    return int(math.ceil(round(duration_seconds, 6) / FRAME_SECONDS))


def rasterize_alignment(alignment: list[dict[str, Any]], n_frames: int, vocab: CtcVocab):
    """Rasterize an MFA alignment into dense per-frame class ids of length ``n_frames``.

    Each frame ``f`` is labelled by the segment covering its centre time ``(f + 0.5) * 0.04``.
    Frames not covered by any segment — the small inter-segment alignment gaps (~270 of 826
    clips have at least one) and any trailing tail after the last aligned word — default to
    ``speech``. This is the documented deterministic simplification: residual MFA gaps are
    treated as speech rather than guessed-at pauses.

    Returns an ``int8`` numpy array.
    """
    import numpy as np

    speech_id = vocab.class_to_index[SPEECH_CLASS]
    labels = np.full(n_frames, speech_id, dtype=np.int8)
    if n_frames == 0:
        return labels

    for seg in alignment:
        cls_id = vocab.class_for_word(seg["word"])
        if cls_id == speech_id:
            continue  # already the default; skip the fill
        # Frame indices whose centre falls in [start, end): centre = (f+0.5)*0.04.
        start_f = int(math.ceil(seg["start"] / FRAME_SECONDS - 0.5))
        end_f = int(math.ceil(seg["end"] / FRAME_SECONDS - 0.5))
        start_f = max(start_f, 0)
        end_f = min(end_f, n_frames)
        if end_f > start_f:
            labels[start_f:end_f] = cls_id
    return labels


def collapse_runs(frame_labels) -> list[int]:
    """Run-length collapse dense frame labels into the CTC target symbol sequence.

    ``[0,0,0,2,2,0]`` -> ``[0,2,0]``. The result is the sequence of acoustic events (incl.
    ``speech`` runs) in order; ``blank`` is never present. This is what ``F.ctc_loss`` wants
    as ``targets``.
    """
    out: list[int] = []
    prev: int | None = None
    for value in frame_labels:
        value = int(value)
        if value != prev:
            out.append(value)
            prev = value
    return out
