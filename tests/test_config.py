"""Tests for chaselate.config: serialisation, tolerant loading, deep copy."""

import json

from chaselate.config import AppConfig


# -- round trip -------------------------------------------------------------

def test_round_trip_to_dict_from_dict():
    cfg = AppConfig()
    restored = AppConfig.from_dict(cfg.to_dict())
    assert restored == cfg


def test_round_trip_save_and_load_utf8(tmp_path):
    path = tmp_path / "config.json"
    cfg = AppConfig()
    cfg.translate.extra_instructions = "請使用正式用語，並保留專有名詞。"
    cfg.save(path)

    loaded = AppConfig.load(path)
    assert loaded.translate.extra_instructions == cfg.translate.extra_instructions
    assert loaded == cfg

    # Confirm the file itself is valid UTF-8 JSON with the Chinese text intact.
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["translate"]["extra_instructions"] == cfg.translate.extra_instructions


# -- tolerant loading ---------------------------------------------------------

def test_unknown_keys_are_ignored():
    data = AppConfig().to_dict()
    data["totally_unknown_key"] = "surprise"
    data["ui"]["another_unknown"] = 123
    cfg = AppConfig.from_dict(data)
    assert cfg == AppConfig()


def test_missing_keys_keep_defaults():
    cfg = AppConfig.from_dict({"ui": {"font_size": 30}})
    default = AppConfig()
    assert cfg.ui.font_size == 30
    assert cfg.ui.opacity == default.ui.opacity
    assert cfg.audio == default.audio


def test_wrong_type_string_for_float_field_keeps_default():
    cfg = AppConfig.from_dict({"ui": {"opacity": "abc"}})
    assert cfg.ui.opacity == AppConfig().ui.opacity


def test_numeric_string_for_int_field_is_coerced():
    cfg = AppConfig.from_dict({"ui": {"font_size": "24"}})
    assert cfg.ui.font_size == 24
    assert isinstance(cfg.ui.font_size, int)


def test_string_yes_for_bool_field_is_coerced_true():
    cfg = AppConfig.from_dict({"ui": {"always_on_top": "yes"}})
    assert cfg.ui.always_on_top is True


def test_string_no_like_value_for_bool_field_is_coerced_false():
    cfg = AppConfig.from_dict({"ui": {"always_on_top": "no"}})
    assert cfg.ui.always_on_top is False


def test_malformed_json_file_returns_defaults(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{ this is not valid json !!!", encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg == AppConfig()


def test_missing_file_returns_defaults(tmp_path):
    path = tmp_path / "does_not_exist.json"
    cfg = AppConfig.load(path)
    assert cfg == AppConfig()


# -- copy ---------------------------------------------------------------

def test_copy_is_a_deep_copy():
    cfg = AppConfig()
    dup = cfg.copy()
    dup.audio.gain = 5.0
    assert cfg.audio.gain != 5.0
    assert dup.audio.gain == 5.0
