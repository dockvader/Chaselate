"""Translation through a local Ollama server.

Everything stays on the machine: no API key, no network egress. The default model is
`translategemma <https://ollama.com/library/translategemma>`_ (Gemma 3 fine-tuned for
translation across 55 languages), but any model the local server has pulled can be selected
at runtime -- :meth:`OllamaClient.list_models` is what populates the picker.

Prompt shape follows TranslateGemma's published format exactly, including the naming of both
the language and its code, because that is what the model was tuned on. Preceding sentence
pairs are supplied as prior chat turns rather than being pasted into the text, which is the
difference between translating a sentence in isolation and translating it as part of a
conversation -- pronouns and dropped subjects resolve correctly only with that history.
Setting ``context_sentences`` to 0 reduces this to the exact single-turn official prompt.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urljoin

import requests

from .config import TranslateConfig
from .languages import english_name, get as get_language
from .textutils import clean_translation, normalize_ws, truncate_middle

log = logging.getLogger(__name__)

#: Default when the source language is unknown (Whisper auto-detect had no opinion).
UNKNOWN_SOURCE = "the source language"

DEFAULT_MODEL = "translategemma:latest"
#: Substring identifying models tuned specifically for translation, which need no coaxing.
TRANSLATION_MODEL_HINTS = ("translategemma", "madlad", "nllb", "opus-mt", "seamless")


class OllamaError(RuntimeError):
    """Any failure talking to the Ollama server."""


class OllamaUnavailable(OllamaError):
    """The server is not reachable at all (not started, or wrong port)."""


class OllamaModelMissing(OllamaError):
    """The server is up but the requested model has not been pulled."""


class TranslationCancelled(OllamaError):
    """The caller asked to abandon this translation."""


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size: int = 0
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""
    modified_at: str = ""

    @property
    def size_gb(self) -> float:
        return self.size / 1e9 if self.size else 0.0

    @property
    def label(self) -> str:
        bits = [self.name]
        detail = " ".join(p for p in (self.parameter_size, self.quantization) if p)
        if detail:
            bits.append(f"({detail})")
        if self.size:
            bits.append(f"{self.size_gb:.1f} GB")
        return "  ".join(bits)

    @property
    def is_translation_specialist(self) -> bool:
        lowered = self.name.casefold()
        return any(hint in lowered for hint in TRANSLATION_MODEL_HINTS)


@dataclass
class ContextPair:
    """One previously translated sentence, replayed as a chat turn for continuity."""

    source: str
    translation: str


def build_instruction(
    source_lang: Optional[str],
    target_lang: str,
    text: str,
    extra_instructions: str = "",
) -> str:
    """TranslateGemma's official prompt, filled in.

    The two blank lines before the text are part of the published template; the model
    treats them as the boundary between instruction and payload.
    """
    target = get_language(target_lang)
    target_name = target.english if target else target_lang
    target_code = target.code if target else target_lang

    source = get_language(source_lang) if source_lang else None
    if source:
        source_name = source.english
        source_desc = f"{source_name} ({source.code})"
    else:
        source_name = UNKNOWN_SOURCE
        source_desc = source_name

    lines = [
        f"You are a professional {source_desc} to {target_name} ({target_code}) "
        f"translator. Your goal is to accurately convey the meaning and nuances of the "
        f"original {source_name} text while adhering to {target_name} grammar, vocabulary, "
        f"and cultural sensitivities.",
        f"Produce only the {target_name} translation, without any additional explanations "
        f"or commentary.",
    ]
    extra = normalize_ws(extra_instructions)
    if extra:
        lines.append(extra)
    lines.append(
        f"Please translate the following {source_name} text into {target_name}:"
    )
    return "\n".join(lines) + "\n\n\n" + text


def build_messages(
    source_lang: Optional[str],
    target_lang: str,
    text: str,
    context: Sequence[ContextPair] = (),
    extra_instructions: str = "",
) -> List[dict]:
    """Chat history: prior pairs as user/assistant turns, then the sentence to translate."""
    messages: List[dict] = []
    for pair in context:
        if not pair.source or not pair.translation:
            continue
        messages.append(
            {
                "role": "user",
                "content": build_instruction(
                    source_lang, target_lang, pair.source, extra_instructions
                ),
            }
        )
        messages.append({"role": "assistant", "content": pair.translation})
    messages.append(
        {
            "role": "user",
            "content": build_instruction(source_lang, target_lang, text, extra_instructions),
        }
    )
    return messages


class OllamaClient:
    """Blocking client for the local Ollama HTTP API.

    Call from a worker thread. One :class:`requests.Session` is reused so the TCP connection
    stays warm between utterances, which matters when translating every few seconds.
    """

    def __init__(self, config: Optional[TranslateConfig] = None):
        self.config = config or TranslateConfig()
        self._session = requests.Session()

    @property
    def base_url(self) -> str:
        url = (self.config.base_url or "http://127.0.0.1:11434").strip()
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url.rstrip("/") + "/"

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path.lstrip("/"))

    def close(self) -> None:
        try:
            self._session.close()
        except Exception:  # noqa: BLE001
            pass

    # -- server introspection ---------------------------------------------

    def version(self, timeout: float = 3.0) -> str:
        """Server version string. Raises :class:`OllamaUnavailable` when unreachable."""
        try:
            resp = self._session.get(self._url("api/version"), timeout=timeout)
            resp.raise_for_status()
            return str(resp.json().get("version", "")) or "unknown"
        except requests.RequestException as exc:
            raise OllamaUnavailable(self._unreachable_message(exc)) from exc
        except ValueError as exc:
            raise OllamaError(f"unexpected response from {self.base_url}: {exc}") from exc

    def health(self, timeout: float = 3.0) -> Tuple[bool, str]:
        """``(ok, message)`` -- never raises, for polling from the UI."""
        try:
            return True, f"Ollama {self.version(timeout)}"
        except OllamaError as exc:
            return False, str(exc)

    def _unreachable_message(self, exc: Exception) -> str:
        return (
            f"Cannot reach Ollama at {self.base_url} ({type(exc).__name__}). "
            "Start it with 'ollama serve', or correct the address in Settings."
        )

    def list_models(self, timeout: float = 10.0) -> List[OllamaModel]:
        """Models available locally, translation specialists first then alphabetical."""
        try:
            resp = self._session.get(self._url("api/tags"), timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise OllamaUnavailable(self._unreachable_message(exc)) from exc
        except ValueError as exc:
            raise OllamaError(f"malformed model list from Ollama: {exc}") from exc

        models: List[OllamaModel] = []
        for entry in payload.get("models") or []:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name") or entry.get("model") or ""
            if not name:
                continue
            details = entry.get("details") or {}
            models.append(
                OllamaModel(
                    name=str(name),
                    size=int(entry.get("size") or 0),
                    family=str(details.get("family") or ""),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization=str(details.get("quantization_level") or ""),
                    modified_at=str(entry.get("modified_at") or ""),
                )
            )
        models.sort(key=lambda m: (not m.is_translation_specialist, m.name.casefold()))
        return models

    def unload_model(self, model: Optional[str] = None) -> None:
        """Ask Ollama to evict the model from VRAM (``keep_alive: 0``)."""
        name = model or self.config.model
        if not name:
            return
        try:
            self._session.post(
                self._url("api/chat"),
                json={"model": name, "messages": [], "keep_alive": 0},
                timeout=5.0,
            )
        except requests.RequestException as exc:
            log.debug("unload request failed: %s", exc)

    # -- translation -------------------------------------------------------

    def translate(
        self,
        text: str,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        context: Sequence[ContextPair] = (),
        on_delta: Optional[Callable[[str], None]] = None,
        cancel: Optional[threading.Event] = None,
    ) -> str:
        """Translate one sentence or utterance and return the cleaned result.

        ``on_delta`` receives the *cumulative* text each time tokens arrive, so a caller can
        render progressively without tracking its own buffer. Raises
        :class:`TranslationCancelled` if ``cancel`` is set mid-stream.
        """
        text = normalize_ws(text)
        if not text:
            return ""

        cfg = self.config
        target = target_lang or cfg.target_lang
        limit = max(0, int(cfg.context_sentences))
        used_context = list(context)[-limit:] if limit else []

        payload = {
            "model": cfg.model,
            "messages": build_messages(
                source_lang, target, text, used_context, cfg.extra_instructions
            ),
            "stream": bool(cfg.stream),
            "keep_alive": cfg.keep_alive,
            "options": {
                "temperature": float(cfg.temperature),
                "num_predict": int(cfg.num_predict),
            },
        }

        log.debug("translating %r -> %s via %s", truncate_middle(text, 80), target, cfg.model)
        if payload["stream"]:
            raw = self._stream_chat(payload, on_delta, cancel)
        else:
            raw = self._blocking_chat(payload, cancel)
        return clean_translation(raw)

    def _request(self, payload: dict, stream: bool):
        try:
            resp = self._session.post(
                self._url("api/chat"),
                json=payload,
                stream=stream,
                timeout=(5.0, float(self.config.request_timeout)),
            )
        except requests.RequestException as exc:
            raise OllamaUnavailable(self._unreachable_message(exc)) from exc

        if resp.status_code == 404:
            model = payload.get("model", "?")
            resp.close()
            raise OllamaModelMissing(
                f"Model '{model}' is not installed. Run:  ollama pull {model}"
            )
        if resp.status_code >= 400:
            detail = ""
            try:
                detail = str(resp.json().get("error", ""))
            except ValueError:
                detail = (resp.text or "")[:300]
            resp.close()
            raise OllamaError(f"Ollama returned {resp.status_code}: {detail}")
        return resp

    def _blocking_chat(self, payload: dict, cancel: Optional[threading.Event]) -> str:
        resp = self._request(payload, stream=False)
        try:
            data = resp.json()
        except ValueError as exc:
            raise OllamaError(f"malformed response from Ollama: {exc}") from exc
        finally:
            resp.close()
        if cancel is not None and cancel.is_set():
            raise TranslationCancelled("translation cancelled")
        if isinstance(data, dict):
            if data.get("error"):
                raise OllamaError(str(data["error"]))
            return str((data.get("message") or {}).get("content") or "")
        return ""

    def _stream_chat(
        self,
        payload: dict,
        on_delta: Optional[Callable[[str], None]],
        cancel: Optional[threading.Event],
    ) -> str:
        resp = self._request(payload, stream=True)
        chunks: List[str] = []
        try:
            for line in resp.iter_lines(decode_unicode=False):
                if cancel is not None and cancel.is_set():
                    raise TranslationCancelled("translation cancelled")
                if not line:
                    continue
                try:
                    event = json.loads(line.decode("utf-8", errors="replace"))
                except ValueError:
                    log.debug("skipping unparsable stream line: %r", line[:120])
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("error"):
                    raise OllamaError(str(event["error"]))
                piece = (event.get("message") or {}).get("content") or ""
                if piece:
                    chunks.append(piece)
                    if on_delta is not None:
                        # Hand over the cleaned cumulative text so partial preambles do not
                        # flash on screen before the real translation arrives.
                        try:
                            on_delta(clean_translation("".join(chunks)))
                        except Exception:  # noqa: BLE001
                            log.exception("on_delta callback raised")
                if event.get("done"):
                    break
        except requests.RequestException as exc:
            raise OllamaError(f"translation stream failed: {exc}") from exc
        finally:
            resp.close()
        return "".join(chunks)
