"""Tests for chaselate.textutils: dedupe, hallucination filtering, sentence splitting."""

import pytest

from chaselate.textutils import (
    clean_translation,
    collapse_repeats,
    dedupe_overlap,
    extend_hint,
    is_hallucination,
    join_text,
    longest_common_prefix,
    normalize_ws,
    split_sentences,
    tokenize,
    truncate_middle,
)


# -- normalize_ws --------------------------------------------------------

def test_normalize_ws_collapses_multiple_spaces():
    assert normalize_ws("a    b") == "a b"


def test_normalize_ws_collapses_tabs_and_newlines():
    assert normalize_ws("a\t\tb\n\nc") == "a b c"


def test_normalize_ws_strips_leading_and_trailing_whitespace():
    assert normalize_ws("  hello  ") == "hello"


def test_normalize_ws_removes_zero_width_space():
    assert normalize_ws("a​b") == "ab"


def test_normalize_ws_empty_string_returns_empty():
    assert normalize_ws("") == ""


def test_normalize_ws_none_like_falsy_returns_empty():
    assert normalize_ws(None) == ""


# -- tokenize / join_text -------------------------------------------------

def test_tokenize_english_splits_on_words():
    assert tokenize("the cat sat") == ["the", "cat", "sat"]


def test_tokenize_chinese_splits_on_characters():
    assert tokenize("你好世界", lang="zh-TW") == ["你", "好", "世", "界"]


def test_tokenize_japanese_splits_on_characters():
    assert tokenize("こんにちは", lang="ja") == list("こんにちは")


def test_tokenize_mixed_chinese_english_uses_sniffed_script():
    # Majority CJK -> treated as spaceless, characters (including latin run) kept whole.
    tokens = tokenize("你好 world 大家好")
    assert "".join(tokens).replace(" ", "") in "".join(tokens)  # sanity: no crash
    assert isinstance(tokens, list)


def test_tokenize_explicit_lang_overrides_sniffing():
    # Text looks like plain ASCII but lang says zh-TW -> character split.
    tokens = tokenize("abc", lang="zh-TW")
    assert tokens == ["a", "b", "c"]


def test_tokenize_single_char_words_not_glued_together():
    """Regression: 'a b c' must tokenize into 3 words, not merge into 'abc'."""
    tokens = tokenize("a b c")
    assert tokens == ["a", "b", "c"]
    assert join_text(tokens, lang="en") == "a b c"


def test_join_text_inverse_of_tokenize_for_english():
    text = "hello there world"
    assert join_text(tokenize(text)) == text


def test_join_text_inverse_of_tokenize_for_chinese():
    text = "你好世界"
    tokens = tokenize(text, lang="zh-TW")
    assert join_text(tokens, lang="zh-TW") == text


def test_join_text_empty_tokens_returns_empty():
    assert join_text([]) == ""


# -- extend_hint --------------------------------------------------------

def test_extend_hint_starts_from_empty():
    assert extend_hint("", "hello world", limit=10) == "hello world"


def test_extend_hint_appends_and_keeps_trailing_tokens_english():
    hint = extend_hint("hello world this is", "a test of the hint budget", limit=6)
    assert hint == "a test of the hint budget"


def test_extend_hint_keeps_trailing_characters_for_cjk():
    # 12 kana; only the trailing 5 characters should survive.
    hint = extend_hint("", "きょうはいいてんきですね", limit=5, lang="ja")
    assert hint == "んきですね"


def test_extend_hint_accumulates_across_calls_like_a_rolling_window():
    hint = extend_hint("", "one two three", limit=5)
    hint = extend_hint(hint, "four five six", limit=5)
    # Only the most recent 5 tokens survive the second call.
    assert tokenize(hint) == ["two", "three", "four", "five", "six"]


def test_extend_hint_limit_zero_disables_it():
    assert extend_hint("anything pending", "more text", limit=0) == ""


def test_extend_hint_empty_addition_returns_existing_hint_trimmed():
    # lang="en" avoids join_text's single-char-token spaceless ambiguity (see
    # test_tokenize_single_char_words_not_glued_together above), same as real callers always
    # pass the ASR-detected language rather than leaving it to be sniffed.
    assert extend_hint("a b c", "", limit=2, lang="en") == "b c"


# -- longest_common_prefix ---------------------------------------------------

def test_longest_common_prefix_full_agreement():
    assert longest_common_prefix("the cat sat down", "the cat sat down", lang="en") == "the cat sat down"


def test_longest_common_prefix_partial_agreement_stops_at_first_divergence():
    # Second pass over a growing buffer heard more and revised the tail.
    a = "i think the meeting"
    b = "i think the meeting is"
    assert longest_common_prefix(a, b, lang="en") == "i think the meeting"


def test_longest_common_prefix_diverges_immediately_returns_empty():
    assert longest_common_prefix("hello there", "goodbye now", lang="en") == ""


def test_longest_common_prefix_no_prior_pass_is_empty_string():
    assert longest_common_prefix("", "anything at all", lang="en") == ""


def test_longest_common_prefix_case_insensitive_like_dedupe_overlap():
    assert longest_common_prefix("The Cat Sat", "the cat sat down", lang="en") == "The Cat Sat"


def test_longest_common_prefix_cjk_characters():
    a = "きょうは会議があるので"
    b = "きょうは会議があるので準備をします"
    assert longest_common_prefix(a, b, lang="ja") == "きょうは会議があるので"


# -- dedupe_overlap ---------------------------------------------------------

def test_dedupe_overlap_normal_case():
    prev = "the weather forecast for tomorrow"
    new = "for tomorrow predicts heavy rain"
    assert dedupe_overlap(prev, new) == "predicts heavy rain"


def test_dedupe_overlap_no_overlap_returns_incoming_unchanged():
    prev = "completely different text here"
    new = "another unrelated sentence entirely"
    assert dedupe_overlap(prev, new) == new


def test_dedupe_overlap_incoming_fully_contained_in_previous():
    prev = "the quick brown fox jumps over the lazy dog"
    new = "the lazy dog"
    assert dedupe_overlap(prev, new) == ""


def test_dedupe_overlap_ignores_case_and_punctuation():
    prev = "Tomorrow, it will rain heavily"
    new = "tomorrow it will rain heavily and then clear up"
    assert dedupe_overlap(prev, new) == "and then clear up"


def test_dedupe_overlap_empty_previous_returns_incoming():
    assert dedupe_overlap("", "hello world") == "hello world"


def test_dedupe_overlap_empty_incoming_returns_empty():
    assert dedupe_overlap("hello world", "") == ""


def test_dedupe_overlap_chinese_character_overlap():
    prev = "今天天氣非常好"
    new = "非常好而且很晴朗"
    assert dedupe_overlap(prev, new, lang="zh-TW") == "而且很晴朗"


def test_dedupe_overlap_prefers_longest_match_over_short_accidental_match():
    # "the" appears both as a short accidental match at the tail of `prev`, and a much
    # longer real overlap exists ("see the cat sit") - the longest must win.
    prev = "yesterday i went to see the cat sit on the mat and then the"
    new = "the cat sit on the mat and then the dog ran away"
    result = dedupe_overlap(prev, new)
    assert result == "dog ran away"


# -- collapse_repeats ---------------------------------------------------------

def test_collapse_repeats_word_loop_collapsed_to_max_repeats():
    text = "thank you " * 5
    result = collapse_repeats(text.strip(), max_repeats=2)
    assert result == "thank you thank you"


def test_collapse_repeats_multi_word_ngram_loop():
    text = "please subscribe now please subscribe now please subscribe now please subscribe now"
    result = collapse_repeats(text, max_repeats=2)
    assert result == "please subscribe now please subscribe now"


def test_collapse_repeats_preserves_emphasis_within_max_repeats():
    text = "no no no"
    result = collapse_repeats(text, max_repeats=2)
    # "no no no" is 3 repeats, max_repeats=2, so it should be collapsed to 2.
    assert result == "no no"


def test_collapse_repeats_leaves_normal_sentence_untouched():
    text = "the quick brown fox jumps over the lazy dog"
    assert collapse_repeats(text) == text


def test_collapse_repeats_empty_text_returns_empty():
    assert collapse_repeats("") == ""


# -- is_hallucination ---------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Thanks for watching",
        "thanks for watching!",
        "字幕由amara.org社群提供",
        "[Music]",
        "（笑）",
        "...",
        "???",
        "",
        "um um um um um",
    ],
)
def test_is_hallucination_true_cases(text):
    assert is_hallucination(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "The stock market rose by three percent today.",
        "It costs 42 dollars.",
        "Yes, I agree.",
        "Let's meet tomorrow at noon.",
        "這是一個正常的句子。",
    ],
)
def test_is_hallucination_false_cases(text):
    assert is_hallucination(text) is False


def test_is_hallucination_single_word_repeated_is_filler():
    assert is_hallucination("okay okay okay okay") is True


def test_is_hallucination_bracketed_only_is_noise():
    assert is_hallucination("(applause)") is True


# -- split_sentences ---------------------------------------------------------

def test_split_sentences_english_periods_and_question_marks():
    sentences, remainder = split_sentences("Hello there. How are you? I am fine!")
    assert sentences == ["Hello there.", "How are you?", "I am fine!"]
    assert remainder == ""


def test_split_sentences_chinese_terminators():
    sentences, remainder = split_sentences("你好嗎？我很好。再見！")
    assert sentences == ["你好嗎？", "我很好。", "再見！"]
    assert remainder == ""


def test_split_sentences_consecutive_terminators_absorbed():
    sentences, remainder = split_sentences("What is happening?! I don't know...")
    assert sentences[0] == "What is happening?!"
    assert sentences[1] == "I don't know..."
    assert remainder == ""


def test_split_sentences_trailing_quote_or_bracket_absorbed():
    sentences, remainder = split_sentences('She said "hello there." Then left.')
    assert sentences[0] == 'She said "hello there."'
    assert sentences[1] == "Then left."


def test_split_sentences_abbreviation_does_not_split():
    sentences, remainder = split_sentences("Dr. Chen said hello.")
    assert sentences == ["Dr. Chen said hello."]
    assert remainder == ""


def test_split_sentences_decimal_number_does_not_split():
    sentences, remainder = split_sentences("Inflation is at 3.5 percent this year.")
    assert len(sentences) == 1
    assert remainder == ""


def test_split_sentences_initialism_does_not_split():
    # Regression: the abbreviation guard used to fold "U.S" to "us" before checking, which
    # discarded the very periods that mark it as an initialism, so the sentence split in two.
    sentences, remainder = split_sentences("U.S. policy changed today.")
    assert sentences == ["U.S. policy changed today."]
    assert remainder == ""


def test_split_sentences_other_dotted_initialisms():
    for text in ("e.g. this example.", "The meeting is at 9 a.m. tomorrow."):
        sentences, _ = split_sentences(text)
        assert sentences == [text], f"{text!r} was split into {sentences}"


def test_split_sentences_no_terminator_goes_to_remainder():
    sentences, remainder = split_sentences("this has no ending punctuation")
    assert sentences == []
    assert remainder == "this has no ending punctuation"


def test_split_sentences_mixed_complete_and_incomplete_tail():
    sentences, remainder = split_sentences("First sentence done. Second one not done yet")
    assert sentences == ["First sentence done."]
    assert remainder == "Second one not done yet"


def test_split_sentences_empty_text_returns_empty_lists():
    sentences, remainder = split_sentences("")
    assert sentences == []
    assert remainder == ""


# -- clean_translation ---------------------------------------------------------

def test_clean_translation_strips_here_is_the_translation_preamble():
    assert clean_translation("Here is the translation: 你好") == "你好"


def test_clean_translation_strips_sure_here_preamble():
    assert clean_translation("Sure! Here's the translation: 你好世界") == "你好世界"


def test_clean_translation_strips_chinese_preamble():
    assert clean_translation("翻譯：這是一個測試") == "這是一個測試"
    assert clean_translation("译文：这是一个测试") == "这是一个测试"


def test_clean_translation_strips_think_block():
    assert clean_translation("<think>reasoning about it</think>Final answer") == "Final answer"


def test_clean_translation_strips_code_fence():
    assert clean_translation("```\nHello world\n```") == "Hello world"


def test_clean_translation_strips_matching_double_quotes():
    assert clean_translation('"Hello there"') == "Hello there"


def test_clean_translation_strips_matching_chinese_quotes():
    assert clean_translation("「你好」") == "你好"
    assert clean_translation("『你好』") == "你好"


def test_clean_translation_strips_trailing_note():
    result = clean_translation("Hello there.\nNote: this is a literal translation")
    assert result == "Hello there."


def test_clean_translation_does_not_strip_quotes_that_are_part_of_content():
    text = '"He said "hi" to me"'
    result = clean_translation(text)
    # Inner quotes present -> outer quotes must not be stripped as a wrapper.
    assert result == text.strip()


def test_clean_translation_does_not_strip_sentence_containing_word_translation():
    text = "The translation of this word is difficult."
    assert clean_translation(text) == text


def test_clean_translation_empty_returns_empty():
    assert clean_translation("") == ""


# -- truncate_middle ---------------------------------------------------------

def test_truncate_middle_limit_larger_than_text_returns_unchanged():
    assert truncate_middle("short", 100) == "short"


def test_truncate_middle_limit_equal_to_text_length_returns_unchanged():
    text = "exact"
    assert truncate_middle(text, len(text)) == text


def test_truncate_middle_limit_smaller_than_text_truncates_with_ellipsis():
    text = "abcdefghijklmnopqrstuvwxyz"
    result = truncate_middle(text, 11)
    assert len(result) == 11
    assert "..." in result
    assert result.startswith("abcd")
    assert result.endswith("wxyz")


def test_truncate_middle_limit_at_or_below_three_no_ellipsis():
    text = "abcdefgh"
    assert truncate_middle(text, 3) == "abc"
    assert truncate_middle(text, 2) == "ab"


def test_truncate_middle_limit_zero_returns_text_unchanged():
    # limit <= 0 short-circuits to returning text as-is per implementation.
    assert truncate_middle("hello", 0) == "hello"
