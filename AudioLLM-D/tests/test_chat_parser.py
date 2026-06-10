from pathlib import Path

from audiollm_d.chat_parser import (
    clean_chat_utterance,
    extract_timed_speaker_utterances,
    parse_id_tier,
    read_chat_transcript,
)


# ---------------------------------------------------------------------------
# parse_id_tier
# ---------------------------------------------------------------------------

def test_parse_id_tier_extracts_participant_metadata() -> None:
    metadata = parse_id_tier("@ID: eng|Pitt|PAR|57;|male|ProbableAD||Participant|18||")

    assert metadata["speaker"] == "PAR"
    assert metadata["age"] == "57"
    assert metadata["sex"] == "male"
    assert metadata["diagnosis"] == "ProbableAD"
    assert metadata["mmse"] == "18"


def test_clean_normalizes_known_event_codes() -> None:
    assert clean_chat_utterance("&=laughs not good &=chuckles") == "[laughter] not good [laughter]"


def test_clean_maps_non_laughter_events_to_non_verbal() -> None:
    assert clean_chat_utterance("&=sighs &=head:shake &=made:up:event") == "[non-verbal] [non-verbal] [non-verbal]"


def test_clean_removes_scope_annotations() -> None:
    assert clean_chat_utterance("kitchen's very tidy . [+ gram]") == "kitchen's very tidy."


def test_clean_keeps_original_word_drops_target_annotation() -> None:
    assert clean_chat_utterance("cookies outta [: out of] a jar") == "cookies outta a jar"


def test_clean_keeps_retrace_attempt() -> None:
    assert clean_chat_utterance("is <in the> [/] in the kitchen") == "is in the in the kitchen"


def test_clean_keeps_revision_attempt() -> None:
    assert clean_chat_utterance("<a child is going> [//] a boy is going") == "a child is going a boy is going"


def test_clean_strips_filler_prefix() -> None:
    assert clean_chat_utterance("&uh the little girl is reacting") == "[uh] the little girl is reacting"


def test_clean_expands_partial_words() -> None:
    assert clean_chat_utterance("the boy is tryin(g) to get cookies") == "the boy is trying to get cookies"


def test_clean_strips_overlap_markers() -> None:
    assert clean_chat_utterance("+< the mother's in a short sleeve dress") == "the mother's in a short sleeve dress"


def test_clean_keeps_terminal_punctuation() -> None:
    assert clean_chat_utterance("the window is open .") == "the window is open."
    assert clean_chat_utterance("you want more ?") == "you want more?"
    assert clean_chat_utterance("great !") == "great!"


def test_clean_preserves_phonological_fragments() -> None:
    assert clean_chat_utterance("a &+dri &uh a walkway away") == "a dri [uh] a walkway away"


def test_clean_expands_leading_elisions() -> None:
    assert clean_chat_utterance("(be)cause the water is running") == "cause the water is running"


def test_clean_replaces_ipa_word_with_xxx() -> None:
    # kɪtʃə˞@u contains IPA characters; [: kitchen] is the target annotation (stripped),
    # [* p:n-ret] and [//] are scope/retrace annotations (stripped).
    # The IPA token has no allowed characters, then final token normalization
    # maps the unintelligible marker to the current canonical token.
    result = clean_chat_utterance(
        "in her kɪtʃə˞@u [: kitchen] [* p:n-ret] [//] kitchen doing some work I suppose ."
    )
    assert result == "in her [unclear-speech] kitchen doing some work I suppose."


def test_clean_letter_name_sequence_joined_with_hyphens() -> None:
    assert clean_chat_utterance("c@l u@l p@l") == "c-u-p"


def test_clean_letter_name_long_sequence() -> None:
    assert clean_chat_utterance("c@l o@l o@l k@l i@l e@l .") == "c-o-o-k-i-e."


def test_clean_other_at_codes_strip_marker() -> None:
    assert clean_chat_utterance("she's boing@o outside .") == "she's boing outside."
    assert clean_chat_utterance("x@n is not a word .") == "x is not a word."


def test_clean_preserves_pause_tokens() -> None:
    assert (
        clean_chat_utterance("the boy (.) took (..) the cookies .")
        == "the boy [short-pause] took [medium-pause] the cookies."
    )


def test_clean_normalizes_trailing_utterance_pause() -> None:
    assert (
        clean_chat_utterance("I think +. it's a kitchen +...")
        == "I think [short-pause] it's a kitchen [long-pause]"
    )


def test_clean_omits_standalone_plus_without_breaking_pause_marker() -> None:
    assert clean_chat_utterance("I think + +...") == "I think [long-pause]"


def test_clean_normalizes_pause_token_before_punctuation() -> None:
    assert clean_chat_utterance("is she washing (..)?") == "is she washing [medium-pause]?"


def test_clean_preserves_unintelligible_marker() -> None:
    assert (
        clean_chat_utterance("the boy is xxx on the xxx")
        == "the boy is [unclear-speech] on the [unclear-speech]"
    )


def test_clean_unintelligible_standalone() -> None:
    assert clean_chat_utterance("xxx") == "[unclear-speech]"


# ---------------------------------------------------------------------------
# read_chat_transcript
# ---------------------------------------------------------------------------

def test_read_chat_transcript_keeps_only_participant_lines(tmp_path: Path) -> None:
    cha = tmp_path / "001.cha"
    cha.write_text(
        "\n".join([
            "@ID: eng|Pitt|PAR|70;|female|Control||Participant|29||",
            "*INV:\ttell me what you see .",
            "*PAR:\tuh the boy is (.) taking cookies .",
            "%mor:\tignored dependent tier",
            "*PAR:\tand the mother xxx is washing dishes .",
        ]),
        encoding="utf-8",
    )

    transcript = read_chat_transcript(cha)

    assert transcript.participant_id == "001"
    assert transcript.metadata["diagnosis"] == "Control"
    assert (
        transcript.text
        == "uh the boy is [short-pause] taking cookies. "
        "and the mother [unclear-speech] is washing dishes."
    )

def test_read_extracts_metadata_fields(tmp_path: Path) -> None:
    cha = tmp_path / "003.cha"
    cha.write_text(
        "\n".join([
            "@ID: eng|Pitt|PAR|65;|male|ProbableAD||Participant|18||",
            "*PAR:\tthe boy is falling .",
        ]),
        encoding="utf-8",
    )

    assert read_chat_transcript(cha).metadata == {
        "language": "eng",
        "corpus": "Pitt",
        "speaker": "PAR",
        "age": "65",
        "sex": "male",
        "diagnosis": "ProbableAD",
        "role": "Participant",
        "mmse": "18",
    }


def test_extract_timed_speaker_utterances_returns_cleaned_alignment(tmp_path: Path) -> None:
    cha = tmp_path / "004.cha"
    cha.write_text(
        "\n".join(
            [
                "*INV:\tignored investigator line . \x15100_500\x15",
                "*PAR:\tuh the boy is (.) taking cookies . \x15500_2500\x15",
                "*PAR:\tand the mother xxx is washing dishes . \x152500_4200\x15",
            ]
        ),
        encoding="utf-8",
    )

    utterances = extract_timed_speaker_utterances(cha, speaker="PAR")
    assert [(u.start_ms, u.end_ms) for u in utterances] == [(500, 2500), (2500, 4200)]
    assert [u.text for u in utterances] == [
        "uh the boy is [short-pause] taking cookies.",
        "and the mother [unclear-speech] is washing dishes.",
    ]
