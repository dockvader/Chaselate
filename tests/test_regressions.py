"""Regression tests for defects found in review.

Each test here corresponds to a bug that shipped once. They are grouped in one file rather
than scattered so the cost of each past mistake stays visible.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time

import pytest

import chaselate  # noqa: F401 - pins the system C++ runtime; must precede PyQt5
from chaselate.config import AppConfig
from chaselate.textutils import is_hallucination, split_sentences


# -- sentence splitting ------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        # An interpunct separates the parts of a transliterated name in Chinese. Treating it
        # as a terminator cut names in half and translated each fragment separately.
        ("他見了唐納·川普。", 1),
        ("約翰·史密斯說了話。", 1),
        # A semicolon joins clauses; splitting there loses the other half's context.
        ("First part; second part.", 1),
        # Multi-part abbreviations: the first period has nothing behind it to recognise, so
        # the splitter needs to look ahead as well.
        ("Ph.D. holders left. Others stayed.", 2),
        ("She has a Ph.D. in physics. He does not.", 2),
        # Folding "U.S" to "us" discarded the periods that identify it as an initialism.
        ("U.S. policy changed today.", 1),
        ("The U.S. and the U.K. agreed. Then they signed.", 2),
        ("The meeting is at 9 a.m. tomorrow.", 1),
        ("Version 1.2.3 shipped.", 1),
        # Genuine boundaries must still split.
        ("他走了。她留下。", 2),
        ("Mr. Smith met Mrs. Jones. They talked.", 2),
        ("Hello there. How are you? Fine!", 3),
    ],
)
def test_sentence_count(text, expected):
    sentences, _ = split_sentences(text)
    assert len(sentences) == expected, f"{text!r} -> {sentences}"


# -- hallucination filter ----------------------------------------------------


@pytest.mark.parametrize(
    "text", ["Okay.", "Thanks.", "So.", "You.", "Yeah.", "Bye.", "The end.", "Yes, I agree."]
)
def test_meaningful_short_replies_are_not_filler(text):
    # These are all things Whisper emits over silence, but they are also all plausible
    # one-word utterances. Dropping them silently loses real speech; the confidence and
    # no-speech-probability thresholds in asr.py handle the silence case instead.
    assert is_hallucination(text) is False


@pytest.mark.parametrize(
    "text",
    ["Hmm.", "Uh.", "um", "[Music]", "（笑）", "...", "Thanks for watching!", "謝謝觀看"],
)
def test_non_lexical_and_stock_phrases_are_still_filler(text):
    assert is_hallucination(text) is True


# -- config loading ----------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        # int() raises OverflowError, not ValueError, on infinity -- and json accepts both
        # spellings. This escaped the loader's except clause and aborted startup.
        '{"ui": {"font_size": 1e400}}',
        '{"ui": {"font_size": Infinity}}',
        '{"ui": {"font_size": -Infinity}}',
        '{"ui": {"opacity": NaN}}',
        '{"ui": {"opacity": Infinity}}',
        # A scalar where a section belongs used to be assigned over the dataclass, so the
        # failure surfaced much later as an AttributeError during UI construction.
        '{"ui": "dark"}',
        '{"audio": [1, 2, 3]}',
        '{"translate": 42}',
        '{"asr": null}',
        # Malformed or wrongly-shaped documents.
        "{not json",
        "[1, 2, 3]",
        '"hello"',
        "null",
    ],
)
def test_bad_config_never_blocks_startup(tmp_path, body):
    path = tmp_path / "config.json"
    path.write_text(body, encoding="utf-8")
    cfg = AppConfig.load(path)
    # Every section must still be a real dataclass with usable values.
    assert isinstance(cfg.ui.font_size, int)
    assert math.isfinite(cfg.ui.opacity)
    assert isinstance(cfg.audio.gain, float)
    assert isinstance(cfg.translate.model, str)
    assert isinstance(cfg.asr.beam_size, int)


def test_missing_config_file_returns_defaults(tmp_path):
    cfg = AppConfig.load(tmp_path / "does-not-exist.json")
    assert cfg.ui.font_size == AppConfig().ui.font_size


def test_valid_values_still_load(tmp_path):
    # The tolerance above must not swallow legitimate settings.
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "ui": {"font_size": "24", "always_on_top": "yes", "opacity": 0.5},
                "translate": {"target_lang": "ja", "extra_instructions": "保持敬語"},
            }
        ),
        encoding="utf-8",
    )
    cfg = AppConfig.load(path)
    assert cfg.ui.font_size == 24 and type(cfg.ui.font_size) is int
    assert cfg.ui.always_on_top is True
    assert cfg.ui.opacity == 0.5
    assert cfg.translate.target_lang == "ja"
    assert cfg.translate.extra_instructions == "保持敬語"


# -- pipeline worker lifecycle -----------------------------------------------


def test_restart_does_not_revive_workers_from_the_previous_run():
    """A worker that outlives stop()'s join must not consume from the next run's queues.

    stop() joins with a timeout, so a thread blocked in a slow HTTP read can still be alive
    afterwards. Reusing one Event across runs meant the next start() cleared it and woke that
    thread up, at which point it read the *new* queue and translated sentences a second time.
    Workers now receive their run's event and queues as arguments, so a survivor keeps
    pointing at the old, already-set ones.
    """
    from chaselate.pipeline import Pipeline

    # Drive the loop directly rather than standing up audio and models.
    pipeline = Pipeline.__new__(Pipeline)
    consumed: list = []
    running = threading.Event()

    old_stop = threading.Event()
    old_queue: "queue.Queue" = queue.Queue()

    def fake_translate_one(utterance, stop):
        running.set()
        consumed.append(utterance)
        # Outlive the join: keep working after stop() has given up waiting.
        time.sleep(1.5)

    pipeline._translate_one = fake_translate_one  # type: ignore[method-assign]

    worker = threading.Thread(
        target=Pipeline._translate_loop, args=(pipeline, old_stop, old_queue), daemon=True
    )
    worker.start()
    old_queue.put("old-work")
    assert running.wait(2.0), "worker never picked up the first item"

    # What stop() does, including the join timing out.
    old_stop.set()
    worker.join(timeout=0.2)
    assert worker.is_alive(), "test needs the worker to still be running here"

    # What start() does: a fresh event and fresh queues.
    new_stop = threading.Event()
    new_queue: "queue.Queue" = queue.Queue()
    new_queue.put("new-work")

    # The survivor must not touch new_queue, and must exit on its own.
    worker.join(timeout=4.0)
    assert not worker.is_alive(), "survivor did not exit once its own stop event was set"
    assert consumed == ["old-work"], f"survivor stole work from the next run: {consumed}"
    assert new_queue.qsize() == 1, "the next run's queue was drained by the old worker"
    assert not new_stop.is_set()
