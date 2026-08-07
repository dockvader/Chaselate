"""Language table shared by the ASR side (Whisper codes) and the prompt side (names).

TranslateGemma's system prompt wants a human-readable language name next to the code
("professional English (en) to Japanese (ja) translator"), so every entry carries both
an English name and the endonym used for the UI.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional

AUTO = "auto"


class Language(NamedTuple):
    code: str
    english: str
    native: str


# Whisper's language set restricted to what TranslateGemma covers, plus the handful of
# extra codes Whisper commonly auto-detects. Ordered roughly by expected usage so the
# combo boxes put the likely picks near the top.
LANGUAGES: List[Language] = [
    # Traditional and Simplified are separate entries on purpose. Collapsing both to "zh"
    # makes the prompt say "Chinese", and the model then answers in Simplified regardless
    # of what the user picked -- a wrong answer, not a cosmetic one.
    Language("zh-TW", "Traditional Chinese", "繁體中文"),
    Language("zh-CN", "Simplified Chinese", "简体中文"),
    Language("zh-HK", "Traditional Chinese (Hong Kong)", "繁體中文（香港）"),
    Language("zh", "Chinese", "中文"),
    Language("yue", "Cantonese", "粵語"),
    Language("en", "English", "English"),
    Language("ja", "Japanese", "日本語"),
    Language("ko", "Korean", "한국어"),
    Language("es", "Spanish", "Español"),
    Language("fr", "French", "Français"),
    Language("de", "German", "Deutsch"),
    Language("ru", "Russian", "Русский"),
    Language("pt", "Portuguese", "Português"),
    Language("it", "Italian", "Italiano"),
    Language("ar", "Arabic", "العربية"),
    Language("hi", "Hindi", "हिन्दी"),
    Language("id", "Indonesian", "Bahasa Indonesia"),
    Language("th", "Thai", "ไทย"),
    Language("vi", "Vietnamese", "Tiếng Việt"),
    Language("tr", "Turkish", "Türkçe"),
    Language("pl", "Polish", "Polski"),
    Language("nl", "Dutch", "Nederlands"),
    Language("uk", "Ukrainian", "Українська"),
    Language("cs", "Czech", "Čeština"),
    Language("sv", "Swedish", "Svenska"),
    Language("da", "Danish", "Dansk"),
    Language("fi", "Finnish", "Suomi"),
    Language("no", "Norwegian", "Norsk"),
    Language("el", "Greek", "Ελληνικά"),
    Language("he", "Hebrew", "עברית"),
    Language("hu", "Hungarian", "Magyar"),
    Language("ro", "Romanian", "Română"),
    Language("bg", "Bulgarian", "Български"),
    Language("hr", "Croatian", "Hrvatski"),
    Language("sr", "Serbian", "Српски"),
    Language("sk", "Slovak", "Slovenčina"),
    Language("sl", "Slovenian", "Slovenščina"),
    Language("lt", "Lithuanian", "Lietuvių"),
    Language("lv", "Latvian", "Latviešu"),
    Language("et", "Estonian", "Eesti"),
    Language("fa", "Persian", "فارسی"),
    Language("ur", "Urdu", "اردو"),
    Language("bn", "Bengali", "বাংলা"),
    Language("ta", "Tamil", "தமிழ்"),
    Language("te", "Telugu", "తెలుగు"),
    Language("ml", "Malayalam", "മലയാളം"),
    Language("mr", "Marathi", "मराठी"),
    Language("gu", "Gujarati", "ગુજરાતી"),
    Language("kn", "Kannada", "ಕನ್ನಡ"),
    Language("pa", "Punjabi", "ਪੰਜਾਬੀ"),
    Language("ms", "Malay", "Bahasa Melayu"),
    Language("tl", "Tagalog", "Tagalog"),
    Language("sw", "Swahili", "Kiswahili"),
    Language("af", "Afrikaans", "Afrikaans"),
    Language("ca", "Catalan", "Català"),
    Language("gl", "Galician", "Galego"),
    Language("eu", "Basque", "Euskara"),
    Language("is", "Icelandic", "Íslenska"),
    Language("hy", "Armenian", "Հայերեն"),
    Language("ka", "Georgian", "ქართული"),
    Language("az", "Azerbaijani", "Azərbaycan"),
    Language("kk", "Kazakh", "Қазақ"),
    Language("uz", "Uzbek", "Oʻzbek"),
    Language("mn", "Mongolian", "Монгол"),
    Language("ne", "Nepali", "नेपाली"),
    Language("si", "Sinhala", "සිංහල"),
    Language("km", "Khmer", "ខ្មែរ"),
    Language("lo", "Lao", "ລາວ"),
    Language("my", "Burmese", "ဗမာ"),
    Language("am", "Amharic", "አማርኛ"),
]

# Keyed on the case-folded code so ``zh-TW``, ``zh-tw`` and ``ZH_TW`` all resolve.
_BY_CODE: Dict[str, Language] = {lang.code.lower(): lang for lang in LANGUAGES}

# Codes where Whisper's tokenizer has no matching token, so the audio is fed under a
# broader code while the translation prompt keeps the specific one. Whisper transcribes
# Mandarin audio as "zh" whether the user wants Traditional or Simplified output, and has
# no Cantonese token at all.
_WHISPER_ALIASES: Dict[str, str] = {
    "zh-tw": "zh",
    "zh-cn": "zh",
    "zh-hk": "zh",
    "yue": "zh",
}

# Scripts without inter-word spaces: sentence splitting and word-overlap dedup both
# need to know when whitespace is not a token boundary.
SPACELESS_CODES = frozenset(
    {"zh", "zh-tw", "zh-cn", "zh-hk", "yue", "ja", "th", "lo", "my", "km"}
)


def get(code: Optional[str]) -> Optional[Language]:
    """Look up a language, tolerating region suffixes such as ``zh-TW`` or ``en_US``."""
    if not code:
        return None
    code = code.strip().lower().replace("_", "-")
    if code == AUTO:
        return None
    if code in _BY_CODE:
        return _BY_CODE[code]
    base = code.split("-", 1)[0]
    return _BY_CODE.get(base)


def english_name(code: Optional[str], fallback: str = "the source language") -> str:
    lang = get(code)
    return lang.english if lang else fallback


def display_name(code: Optional[str]) -> str:
    """Label used in the UI: ``中文 (Chinese)``, or ``Auto-detect`` for ``auto``."""
    if not code or code == AUTO:
        return "Auto-detect"
    lang = get(code)
    if not lang:
        return code
    if lang.native == lang.english:
        return lang.english
    return f"{lang.native} ({lang.english})"


def whisper_code(code: Optional[str]) -> Optional[str]:
    """Language code to hand to faster-whisper; ``None`` means let Whisper detect."""
    if not code or code == AUTO:
        return None
    lang = get(code)
    if not lang:
        return None
    key = lang.code.lower()
    return _WHISPER_ALIASES.get(key, key)


def is_spaceless(code: Optional[str]) -> bool:
    lang = get(code)
    return bool(lang and lang.code.lower() in SPACELESS_CODES)
