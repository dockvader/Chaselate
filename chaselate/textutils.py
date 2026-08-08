"""Text plumbing between Whisper and the translator.

Three problems live here:

* Segments overlap on purpose (VAD padding plus a pre-roll), so consecutive
  transcriptions repeat words at the boundary -> :func:`dedupe_overlap`.
* Whisper invents stock phrases over music and silence, and loops when it loses the
  thread -> :func:`is_hallucination` and :func:`collapse_repeats`.
* Instruction-tuned models wrap answers in preambles, quotes and code fences even when
  told not to -> :func:`clean_translation`.

Everything is pure and side-effect free so it can be unit tested without audio, a GPU or
a running Ollama.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Sequence, Tuple

from .languages import is_spaceless

__all__ = [
    "normalize_ws",
    "tokenize",
    "dedupe_overlap",
    "collapse_repeats",
    "is_hallucination",
    "split_sentences",
    "clean_translation",
    "join_text",
    "truncate_middle",
    "extend_hint",
    "longest_common_prefix",
]

_WS_RE = re.compile(r"\s+")

# Codepoints in scripts that do not separate words with spaces. Used to decide whether
# to compare word tokens or character tokens.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # kana
    (0x3400, 0x4DBF),  # CJK ext A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xF900, 0xFAFF),  # compatibility ideographs
    (0xAC00, 0xD7AF),  # hangul syllables
    (0x0E00, 0x0E7F),  # thai
    (0x1780, 0x17FF),  # khmer
    (0x1000, 0x109F),  # myanmar
    (0x20000, 0x2FA1F),  # CJK ext B+
)

# Sentence terminators, Latin and full-width alike. Kept separate from the abbreviation
# guard below because CJK terminators never appear in abbreviations.
#
# Notably absent:
#   * the interpuncts U+00B7 and U+30FB, which in Chinese separate the parts of a
#     transliterated personal name ("唐納·川普"). Treating them as terminators split names in
#     half and sent each fragment off as its own translation request.
#   * semicolons, which join two clauses rather than ending one. Splitting there would
#     translate each half without the other for context.
_TERMINATORS = ".!?。！？…‥"
_CLOSERS = "\"'”’）)]】」』》〉"

# "Dr." must not end a sentence. Only the ambiguous ASCII period needs this.
_ABBREVIATIONS = frozenset(
    """mr mrs ms dr prof sr jr st vs etc eg ie approx dept est fig no vol
    inc ltd corp co univ gen col sgt capt lt rev hon gov sen rep
    a b c d e f g h i j k l m n o p q r s t u v w x y z""".split()
)

_SENTENCE_SPLIT_MIN_CHARS = 2

# Dotted abbreviations: "U.S.", "e.g.", "a.m.", "Ph.D.". Every period in these is internal to
# the token, so none of them ends a sentence. Matched against the raw fragment with its final
# period already removed, and before case folding, which would strip the periods that make
# "U.S." distinguishable from the word "us".
#
# Segments are allowed up to three letters so "Ph.D." works; requiring exactly one split it
# into "Ph." and "D. ...". The trailing group is optional because the final period has been
# removed by the caller.
_INITIALISM_RE = re.compile(r"^(?:[A-Za-z]{1,3}\.)+[A-Za-z]{0,3}$")


def _is_spaceless_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _CJK_RANGES)


def normalize_ws(text: str) -> str:
    """Collapse runs of whitespace and trim. Never returns ``None``."""
    if not text:
        return ""
    return _WS_RE.sub(" ", text.replace("​", "")).strip()


def _fold(token: str) -> str:
    """Comparison key: case-folded, accent-stripped, punctuation-free."""
    token = unicodedata.normalize("NFKD", token.casefold())
    return "".join(
        ch for ch in token if not unicodedata.combining(ch) and (ch.isalnum() or ch == "'")
    )


def _looks_spaceless(text: str) -> bool:
    """True when most letters are in a script that does not use spaces."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    dense = sum(1 for ch in letters if _is_spaceless_char(ch))
    return dense * 2 > len(letters)


def _resolve_spaceless(text: str, lang: Optional[str]) -> bool:
    """Decide once whether to treat ``text`` as a spaceless script.

    The language hint wins when it names a known language; otherwise the script is
    sniffed from the text so ``auto`` source detection still gets sensible tokens.
    """
    if lang:
        from .languages import get as _get_lang

        if _get_lang(lang) is not None:
            return is_spaceless(lang)
    return _looks_spaceless(text)


def _split(text: str, spaceless: bool) -> List[str]:
    if spaceless:
        return [ch for ch in text if not ch.isspace()]
    return text.split(" ")


def _join(tokens: Sequence[str], spaceless: bool) -> str:
    return "".join(tokens) if spaceless else " ".join(tokens)


def tokenize(text: str, lang: Optional[str] = None) -> List[str]:
    """Split into comparison units: words for spaced scripts, characters otherwise."""
    text = normalize_ws(text)
    if not text:
        return []
    return _split(text, _resolve_spaceless(text, lang))


def extend_hint(hint: str, addition: str, limit: int, lang: Optional[str] = None) -> str:
    """Append ``addition`` to a rolling ``hint``, kept to its trailing ``limit`` tokens.

    Built for feeding Whisper's ``initial_prompt`` with recently recognised speech so
    terminology and phrasing stay consistent across segments -- the "last N words as prompt"
    trick from Whisper-Streaming's LocalAgreement paper (Macháček et al. 2023). Tokens are
    words for spaced scripts and characters for spaceless ones (see :func:`tokenize`), so the
    same numeric budget is a comparable amount of context regardless of script.
    """
    if limit <= 0:
        return ""
    combined = normalize_ws(f"{hint} {addition}" if hint else addition)
    if not combined:
        return ""
    tokens = tokenize(combined, lang)
    if len(tokens) > limit:
        tokens = tokens[-limit:]
    return join_text(tokens, lang)


def join_text(tokens: Sequence[str], lang: Optional[str] = None) -> str:
    """Inverse of :func:`tokenize` for the token style that language uses.

    Without a usable language hint the token shape is the only signal available, so
    all-single-character input is assumed to have come from a spaceless script.
    """
    if not tokens:
        return ""
    if lang:
        from .languages import get as _get_lang

        if _get_lang(lang) is not None:
            return _join(tokens, is_spaceless(lang))
    return _join(tokens, all(len(t) == 1 for t in tokens))


def longest_common_prefix(a: str, b: str, lang: Optional[str] = None) -> str:
    """Longest run of leading tokens shared by ``a`` and ``b``.

    This is LocalAgreement-2's commit rule (Whisper-Streaming / Macháček et al. 2023):
    re-transcribing the same, still-growing audio twice in a row and keeping only the prefix
    both passes agree on is a far more reliable "is this settled yet" signal than trusting a
    single pass, or a fixed pause length, blindly. Comparison is on folded tokens (see
    :func:`dedupe_overlap`), so case/accent differences between passes do not break agreement.
    """
    tokens_a = tokenize(a, lang)
    tokens_b = tokenize(b, lang)
    n = 0
    for ta, tb in zip(tokens_a, tokens_b):
        if _fold(ta) != _fold(tb):
            break
        n += 1
    return join_text(tokens_a[:n], lang)


def dedupe_overlap(
    previous: str,
    incoming: str,
    lang: Optional[str] = None,
    max_overlap_tokens: int = 40,
    min_overlap_tokens: int = 1,
) -> str:
    """Strip the leading part of ``incoming`` that repeats the tail of ``previous``.

    Returns the genuinely new text, or ``""`` when ``incoming`` adds nothing. The match
    is on folded tokens, so "Tomorrow," and "tomorrow" count as the same word.

    Prefers the *longest* overlap: a short accidental match ("the") would otherwise win
    over the real boundary and leave duplicated words on screen.
    """
    incoming = normalize_ws(incoming)
    if not incoming:
        return ""
    previous = normalize_ws(previous)
    if not previous:
        return incoming

    # Resolve the script from the incoming text, which is the text being rebuilt.
    spaceless = _resolve_spaceless(incoming, lang)
    prev_tokens = _split(previous, spaceless)
    new_tokens = _split(incoming, spaceless)
    if not prev_tokens or not new_tokens:
        return incoming

    prev_folded = [_fold(t) for t in prev_tokens]
    new_folded = [_fold(t) for t in new_tokens]

    limit = min(len(prev_folded), len(new_folded), max_overlap_tokens)
    best = 0
    for k in range(limit, min_overlap_tokens - 1, -1):
        tail = prev_folded[-k:]
        head = new_folded[:k]
        # Ignore matches made only of punctuation, which fold to empty strings.
        if not any(tail) or not any(head):
            continue
        if tail == head:
            best = k
            break

    if best == 0:
        return incoming
    remainder = new_tokens[best:]
    if not remainder:
        return ""
    return _join(remainder, spaceless)


def collapse_repeats(text: str, lang: Optional[str] = None, max_repeats: int = 2) -> str:
    """Collapse a token n-gram repeated more than ``max_repeats`` times in a row.

    Whisper degenerates into loops ("thank you thank you thank you ...") when the audio
    is music or noise. Keeps the first ``max_repeats`` copies so genuine emphasis
    ("no no no") survives.
    """
    text = normalize_ws(text)
    spaceless = _resolve_spaceless(text, lang)
    tokens = _split(text, spaceless)
    if len(tokens) < 2:
        return text

    folded = [_fold(t) for t in tokens]
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        collapsed = False
        # Longer patterns first: a 3-gram loop also contains 1-gram repeats.
        for size in range(min(8, (n - i) // 2), 0, -1):
            pattern = folded[i : i + size]
            if not any(pattern):
                continue
            count = 1
            j = i + size
            while j + size <= n and folded[j : j + size] == pattern:
                count += 1
                j += size
            if count > max_repeats:
                out.extend(tokens[i : i + size * max_repeats])
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1

    return _join(out, spaceless)


# Phrases Whisper emits over music, applause and silence. Matched against the folded,
# punctuation-free form of the whole segment, so only near-exact junk is dropped.
_HALLUCINATION_PHRASES = (
    "thanks for watching",
    "thank you for watching",
    "thanks for watching and dont forget to subscribe",
    "please subscribe to my channel",
    "please subscribe",
    "like and subscribe",
    "dont forget to subscribe",
    "subtitles by the amaraorg community",
    "subtitles by amaraorg",
    "amaraorg",
    "subtitles created by",
    "subtitled by",
    "transcription by castingwordscom",
    "translated by",
    "copyright fbi",
    # Non-lexical fillers only. Words that carry meaning are deliberately *not* listed, even
    # though Whisper does emit them over silence: "Okay.", "Thanks.", "So.", "You.", "Yeah."
    # and "Bye." are all plausible one-word utterances, and dropping them silently loses real
    # speech. The confidence and no-speech-probability thresholds in asr.py handle the
    # silence case without discarding legitimate short replies.
    "hmm",
    "mm",
    "mmm",
    "uh",
    "uhh",
    "um",
    "umm",
    "ah",
    "ahh",
    "eh",
    "oh",
    "hm",
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録お願いします",
    "最後までご視聴いただきありがとうございます",
    "字幕製作",
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "請不吝點贊",
    "訂閱轉發打賞支持明鏡與點點欄目",
    "由wwwyoutubecom提供",
    "謝謝觀看",
    "謝謝大家",
    "感謝您的觀看",
    "感謝觀看",
    "小編",
    "字幕由amaraorg社群提供",
    "продолжение следует",
    "субтитры сделал",
    "спасибо за просмотр",
    "подписывайтесь на канал",
    "gracias por ver",
    "suscribete al canal",
    "untertitel von",
    "vielen dank fur das zuschauen",
    "merci davoir regarde",
    "sottotitoli",
    "구독과 좋아요",
    "시청해주셔서 감사합니다",
    "mbc 뉴스",
)

_HALLUCINATION_SET = frozenset(
    "".join(_fold(t) for t in tokenize(p)) for p in _HALLUCINATION_PHRASES
)

# Segments made only of these are noise annotations, not speech.
_BRACKETED_ONLY_RE = re.compile(
    r"^[\s]*(?:[\(\[\{（【]{1}[^\)\]\}）】]*[\)\]\}）】]{1}[\s]*)+$"
)
_PUNCT_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def is_hallucination(text: str, lang: Optional[str] = None) -> bool:
    """True when the segment is almost certainly Whisper filler rather than speech."""
    text = normalize_ws(text)
    if not text:
        return True
    if _PUNCT_ONLY_RE.match(text):
        return True
    if _BRACKETED_ONLY_RE.match(text):
        # "[Music]", "(applause)", "（笑）"
        return True

    key = "".join(_fold(t) for t in tokenize(text, lang))
    if not key:
        return True
    if key in _HALLUCINATION_SET:
        return True

    # A single token repeated to fill the segment is a loop, not content.
    tokens = tokenize(text, lang)
    if len(tokens) >= 4:
        folded = {_fold(t) for t in tokens if _fold(t)}
        if len(folded) == 1:
            return True
    return False


_TOKEN_CHARS = re.compile(r"[A-Za-z0-9.]")
_MORE_ABBREV_AHEAD = re.compile(r"^[A-Za-z]{1,3}\.")


def _period_is_internal(text: str, index: int) -> bool:
    """True when the period at ``index`` is followed by more of the same abbreviation.

    ``_ends_with_abbreviation`` looks only backwards, which is blind to the *first* period of a
    multi-part abbreviation: at the "Ph." in "Ph.D." there is nothing yet to recognise. This
    peeks at what follows within the same whitespace-delimited token, so "Ph.D. holders left."
    is one sentence rather than three.
    """
    end = index + 1
    limit = len(text)
    while end < limit and _TOKEN_CHARS.match(text[end]):
        end += 1
    return bool(_MORE_ABBREV_AHEAD.match(text[index + 1 : end]))


def _ends_with_abbreviation(chunk: str) -> bool:
    """True if ``chunk`` ends in a known abbreviation, so its period is not a full stop."""
    if not chunk.endswith("."):
        return False
    word = re.split(r"[\s ]", chunk[:-1])[-1] if chunk[:-1] else ""
    # Dotted initialisms must be checked before folding, which strips the periods that
    # distinguish "U.S." from the word "us".
    if _INITIALISM_RE.match(word):
        return True
    word = _fold(word)
    if not word:
        return False
    if word in _ABBREVIATIONS:
        return True
    # A bare initial, as in "J. Smith".
    if len(word) == 1:
        return True
    # A decimal point: "3.5 percent".
    if word.isdigit():
        return True
    return False


def split_sentences(text: str, lang: Optional[str] = None) -> Tuple[List[str], str]:
    """Split into complete sentences plus whatever trails after the last terminator.

    Returns ``(sentences, remainder)``. The remainder is text the speaker has not
    finished yet; the caller normally holds it back until more audio arrives, then feeds
    it in again prefixed to the next chunk.
    """
    text = normalize_ws(text)
    if not text:
        return [], ""

    sentences: List[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in _TERMINATORS:
            end = i + 1
            # Absorb runs of terminators ("?!", "...") and any closing quote/bracket.
            while end < n and text[end] in _TERMINATORS:
                end += 1
            while end < n and text[end] in _CLOSERS:
                end += 1
            chunk = text[start:end].strip()
            if ch == "." and (
                _period_is_internal(text, i)
                or _ends_with_abbreviation(text[start:i + 1].strip())
            ):
                i = end
                continue
            if len(chunk) >= _SENTENCE_SPLIT_MIN_CHARS:
                sentences.append(chunk)
                start = end
                i = end
                continue
        i += 1

    remainder = text[start:].strip()
    return sentences, remainder


# Preambles instruction-tuned models add despite "output only the translation".
_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"(?:here(?:'s| is)|this is)\s+(?:the\s+)?(?:\w+\s+)?translation\s*[:\-]?"
    r"|(?:the\s+)?translat(?:ion|ed(?:\s+text)?)\s*[:\-]"
    r"|sure[,!.]?\s*(?:here.*?[:\-])?"
    r"|certainly[,!.]?\s*(?:here.*?[:\-])?"
    r"|of course[,!.]?\s*(?:here.*?[:\-])?"
    r"|翻譯\s*[:：]|翻译\s*[:：]|譯文\s*[:：]|译文\s*[:：]"
    r"|以下是.*?翻譯\s*[:：]?|以下是.*?翻译\s*[:：]?"
    r"|翻訳\s*[:：]|번역\s*[:：]"
    r")\s*",
    re.IGNORECASE,
)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?(.*?)\n?```\s*$", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_TRAILING_NOTE_RE = re.compile(
    r"\n\s*(?:\(|\[)?(?:note|explanation|literal(?:ly)?|alternative|注|說明|说明|注釋|注释)"
    r"\s*[:：].*$",
    re.IGNORECASE | re.DOTALL,
)

_MATCHING_QUOTES = (('"', '"'), ("'", "'"), ("“", "”"), ("「", "」"), ("『", "』"), ("«", "»"))


def clean_translation(text: str) -> str:
    """Strip preambles, reasoning blocks, code fences, wrapping quotes and trailing notes."""
    if not text:
        return ""

    text = _THINK_RE.sub("", text).strip()

    fence = _FENCE_RE.match(text)
    if fence:
        text = fence.group(1).strip()

    # A model can stack "Sure!" and "Here is the translation:" - strip until stable.
    for _ in range(3):
        stripped = _PREAMBLE_RE.sub("", text, count=1).strip()
        if stripped == text:
            break
        text = stripped

    text = _TRAILING_NOTE_RE.sub("", text).strip()

    for open_q, close_q in _MATCHING_QUOTES:
        if len(text) >= 2 and text.startswith(open_q) and text.endswith(close_q):
            inner = text[1:-1].strip()
            # Only unwrap when the quotes really are a wrapper, not part of the content.
            if open_q not in inner and close_q not in inner:
                text = inner
                break

    return normalize_ws(text)


def truncate_middle(text: str, limit: int) -> str:
    """Shorten to ``limit`` characters, keeping both ends (for log lines and tooltips)."""
    if limit <= 0 or len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    head = (limit - 3) // 2
    tail = limit - 3 - head
    return text[:head] + "..." + (text[-tail:] if tail else "")
