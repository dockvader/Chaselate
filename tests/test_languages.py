"""Tests for chaselate.languages: code lookup, Whisper aliasing, spaceless scripts."""

from chaselate import languages
from chaselate.languages import LANGUAGES, display_name, get, is_spaceless, whisper_code


# -- get ---------------------------------------------------------------

def test_get_exact_code():
    lang = get("en")
    assert lang is not None
    assert lang.code == "en"


def test_get_is_case_insensitive():
    assert get("zh-TW") == get("zh-tw") == get("ZH_TW")


def test_get_underscore_variant_resolves():
    assert get("zh_TW") == get("zh-TW")


def test_get_region_suffix_falls_back_to_base_language():
    assert get("en-US") == get("en")
    assert get("pt-BR") == get("pt")


def test_get_auto_returns_none():
    assert get("auto") is None


def test_get_unknown_code_returns_none():
    assert get("xx-unknown") is None


def test_get_empty_string_returns_none():
    assert get("") is None


def test_get_none_returns_none():
    assert get(None) is None


# -- traditional vs simplified chinese must be distinct ------------------

def test_traditional_and_simplified_chinese_are_different_languages():
    tw = get("zh-TW")
    cn = get("zh-CN")
    assert tw is not None and cn is not None
    assert tw.english == "Traditional Chinese"
    assert cn.english == "Simplified Chinese"
    assert tw != cn
    assert tw.english != cn.english


# -- whisper_code ---------------------------------------------------------

def test_whisper_code_maps_chinese_variants_to_bare_zh():
    assert whisper_code("zh-TW") == "zh"
    assert whisper_code("zh-CN") == "zh"
    assert whisper_code("zh-HK") == "zh"
    assert whisper_code("yue") == "zh"


def test_whisper_code_auto_returns_none():
    assert whisper_code("auto") is None


def test_whisper_code_plain_english_returns_en():
    assert whisper_code("en") == "en"


def test_whisper_code_none_returns_none():
    assert whisper_code(None) is None


# -- is_spaceless ---------------------------------------------------------

def test_is_spaceless_true_for_cjk_and_related_scripts():
    for code in ("zh-TW", "zh-CN", "ja", "th", "km"):
        assert is_spaceless(code) is True, code


def test_is_spaceless_false_for_spaced_scripts():
    for code in ("en", "fr", "de"):
        assert is_spaceless(code) is False, code


# -- display_name ---------------------------------------------------------

def test_display_name_auto_is_auto_detect():
    assert display_name("auto") == "Auto-detect"


def test_display_name_english_shows_just_english():
    # native == english for English -> no duplicated "(English)".
    assert display_name("en") == "English"


def test_display_name_chinese_shows_both_native_and_english():
    name = display_name("zh-TW")
    assert "繁體中文" in name
    assert "Traditional Chinese" in name


# -- data integrity ---------------------------------------------------------

def test_no_duplicate_language_codes():
    codes = [lang.code.lower() for lang in LANGUAGES]
    assert len(codes) == len(set(codes))


def test_every_language_entry_has_nonempty_fields():
    for lang in LANGUAGES:
        assert lang.code
        assert lang.english
        assert lang.native
