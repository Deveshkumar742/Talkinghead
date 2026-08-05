"""Tests for script segmentation.

Segmentation is where a subtle bug would be least visible and most annoying: a
bad split produces an odd pause or a drifting voice, neither of which looks like
a crash. So the boundary behaviour is pinned down here.
"""

from __future__ import annotations

import pytest

from talkinghead.script_prep import (
    PreparedScript,
    normalize_text,
    prepare_script,
    split_sentences,
)


class TestNormalizeText:
    def test_collapses_runs_of_spaces_and_newlines(self):
        assert normalize_text("hello    world\nagain") == "hello world again"

    def test_preserves_paragraph_breaks(self):
        assert normalize_text("one\n\ntwo") == "one\n\ntwo"

    def test_collapses_multiple_blank_lines_to_one_break(self):
        assert normalize_text("one\n\n\n\ntwo") == "one\n\ntwo"

    def test_replaces_typographic_quotes(self):
        # Copy-paste from docs and email is the normal input path here, so smart
        # quotes are the common case rather than an edge case.
        assert normalize_text("“quoted” and ‘single’") == (
            '"quoted" and \'single\''
        )

    def test_replaces_dashes_and_ellipsis(self):
        assert normalize_text("a—b") == "a - b"
        assert normalize_text("a–b") == "a-b"
        assert normalize_text("wait…") == "wait..."

    def test_handles_crlf(self):
        assert normalize_text("one\r\n\r\ntwo") == "one\n\ntwo"

    def test_empty_input_yields_empty(self):
        assert normalize_text("   \n\n  ") == ""


class TestSplitSentences:
    def test_splits_on_terminal_punctuation(self):
        assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_does_not_split_on_known_abbreviation(self):
        assert split_sentences("Dr. Smith spoke.") == ["Dr. Smith spoke."]

    def test_does_not_split_on_single_initial(self):
        assert split_sentences("J. Smith arrived.") == ["J. Smith arrived."]

    def test_does_not_split_on_eg(self):
        assert split_sentences("Use it, e.g. here. Then stop.") == [
            "Use it, e.g. here.",
            "Then stop.",
        ]

    def test_keeps_trailing_quote_with_its_sentence(self):
        result = split_sentences('He said "go." Then left.')
        assert len(result) == 2
        assert result[1] == "Then left."

    def test_decimal_numbers_do_not_split(self):
        # No space after the period, so the sentence regex should not fire.
        assert split_sentences("It costs 3.50 today.") == ["It costs 3.50 today."]


class TestPrepareScript:
    def test_rejects_empty_script(self):
        with pytest.raises(ValueError, match="empty after normalization"):
            prepare_script("   \n\n  ")

    def test_indices_are_sequential_from_zero(self):
        prepared = prepare_script("One. Two. Three.")
        assert [s.index for s in prepared.segments] == [0, 1, 2]

    def test_final_segment_has_no_trailing_pause(self):
        # A trailing pause would leave the video holding on a silent frame.
        prepared = prepare_script("One. Two.")
        assert prepared.segments[-1].pause_after_ms == 0

    def test_sentence_pause_between_sentences(self):
        prepared = prepare_script("One. Two.", sentence_pause_ms=200)
        assert prepared.segments[0].pause_after_ms == 200

    def test_paragraph_pause_is_longer_than_sentence_pause(self):
        prepared = prepare_script(
            "One.\n\nTwo.", sentence_pause_ms=180, paragraph_pause_ms=450
        )
        assert prepared.segments[0].pause_after_ms == 450
        assert prepared.segments[0].is_paragraph_end is True

    def test_respects_max_chars(self):
        long_text = ", ".join(f"clause number {i}" for i in range(40)) + "."
        prepared = prepare_script(long_text, max_chars=80)
        assert len(prepared.segments) > 1
        # Clause-boundary splitting can leave one piece slightly over when a
        # single clause exceeds the cap; the hard-split path guarantees the rest.
        assert all(len(s.text) <= 100 for s in prepared.segments)

    def test_splits_unpunctuated_long_sentence_by_words(self):
        text = " ".join(["word"] * 200)
        prepared = prepare_script(text, max_chars=100)
        assert len(prepared.segments) > 1
        assert all(len(s.text) <= 100 for s in prepared.segments)

    def test_no_word_is_lost_when_splitting(self):
        text = " ".join(f"w{i}" for i in range(300))
        prepared = prepare_script(text, max_chars=60)
        rejoined = " ".join(s.text for s in prepared.segments)
        assert rejoined.split() == text.split()

    def test_mid_sentence_split_gets_no_pause(self):
        # A pause inside a sentence would be audible as a stumble.
        text = " ".join(["word"] * 100)
        prepared = prepare_script(text, max_chars=50)
        assert all(s.pause_after_ms == 0 for s in prepared.segments)

    def test_no_empty_segments(self):
        prepared = prepare_script("One.\n\n\n\nTwo.   \n\n  Three.")
        assert all(s.text.strip() for s in prepared.segments)

    def test_script_hash_is_stable(self):
        a = prepare_script("Same text here.")
        b = prepare_script("Same text here.")
        assert a.script_hash == b.script_hash

    def test_script_hash_changes_with_content(self):
        a = prepare_script("One version.")
        b = prepare_script("Another version.")
        assert a.script_hash != b.script_hash

    def test_script_hash_ignores_whitespace_noise(self):
        # Reformatting a script should not invalidate an expensive render.
        a = prepare_script("One. Two.")
        b = prepare_script("One.    Two.\n")
        assert a.script_hash == b.script_hash

    def test_segment_cache_key_tracks_its_own_text(self):
        # Editing one sentence must invalidate only that segment.
        a = prepare_script("First. Second. Third.")
        b = prepare_script("First. CHANGED. Third.")
        assert a.segments[0].cache_key() == b.segments[0].cache_key()
        assert a.segments[1].cache_key() != b.segments[1].cache_key()
        assert a.segments[2].cache_key() == b.segments[2].cache_key()

    def test_slug_is_zero_padded_for_sortability(self):
        prepared = prepare_script(" ".join("Sentence number %d." % i for i in range(12)))
        assert prepared.segments[0].slug == "seg_000"
        assert prepared.segments[11].slug == "seg_011"

    def test_duration_estimate_includes_pauses(self):
        no_pause = prepare_script("One. Two.", sentence_pause_ms=0)
        with_pause = prepare_script("One. Two.", sentence_pause_ms=1000)
        assert with_pause.estimated_duration_s() > no_pause.estimated_duration_s()

    def test_len_reports_segment_count(self):
        prepared = prepare_script("One. Two. Three.")
        assert len(prepared) == 3 == len(prepared.segments)


class TestRealSampleScript:
    """The fixed script used for every quality comparison must segment sanely."""

    def test_sample_script_segments_reasonably(self, sample_script_text):
        prepared = prepare_script(sample_script_text)
        assert 4 <= len(prepared) <= 20
        assert all(len(s.text) <= 300 for s in prepared.segments)
        # Three paragraphs, so two paragraph boundaries.
        assert sum(s.is_paragraph_end for s in prepared.segments) == 2
        assert 15 <= prepared.estimated_duration_s() <= 60
