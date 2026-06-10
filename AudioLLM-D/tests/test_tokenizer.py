"""Standalone tokenizer tests — no model download required."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from audiollm_d.special_tokens import SHORT_PAUSE_TOKEN, SPECIAL_TOKENS


@pytest.fixture()
def tokenizer() -> PreTrainedTokenizerFast:
    """Minimal fast tokenizer with a near-empty vocab to test token registration."""
    base = Tokenizer(WordLevel(vocab={"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    base.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=base,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )


def test_add_all_special_tokens_returns_correct_count(tokenizer: PreTrainedTokenizerFast) -> None:
    added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    assert added == len(SPECIAL_TOKENS)


def test_each_token_encodes_as_single_id(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    for tok in SPECIAL_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        assert len(ids) == 1, f"{tok!r} produced {len(ids)} ids, expected 1"


def test_round_trip_decode(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    for tok in SPECIAL_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert decoded == tok, f"Round-trip failed: {tok!r} → {decoded!r}"


def test_add_special_tokens_is_idempotent(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    added_again = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    assert added_again == 0


def test_token_ids_are_unique(tokenizer: PreTrainedTokenizerFast) -> None:
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    assert len(ids) == len(set(ids)), "Two special tokens share the same ID"


def test_vocab_size_grows_by_token_count(tokenizer: PreTrainedTokenizerFast) -> None:
    vocab_before = tokenizer.vocab_size
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    assert len(tokenizer) == vocab_before + len(SPECIAL_TOKENS)


def test_sentence_with_special_tokens(tokenizer: PreTrainedTokenizerFast) -> None:
    # Realistic dementia transcription: pause, filler, and non-verbal event mid-sentence.
    sentence = f"the boy is {SHORT_PAUSE_TOKEN} taking um the cookies and [laughter]"
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    ids = tokenizer.encode(sentence, add_special_tokens=False)

    pause_id = tokenizer.convert_tokens_to_ids(SHORT_PAUSE_TOKEN)
    laugh_id = tokenizer.convert_tokens_to_ids("[laughter]")

    # Each special token must appear exactly once in the encoded output.
    assert ids.count(pause_id) == 1, "short pause token missing from encoded sentence"
    assert ids.count(laugh_id) == 1, "[laughter] token missing from encoded sentence"

    # Total token count equals word count (whitespace-split), confirming no fragmentation.
    assert len(ids) == len(sentence.split())
