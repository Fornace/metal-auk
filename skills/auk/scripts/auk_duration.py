"""Target-length math, ported from the Prompt Enhancer.

AuK renders a fixed span: hand it the wrong length and the result is truncated or
padded speech. Upstream solves this in `pe.py` with formulas that need no model:
a byte-weighted speech rate for text, a per-word rate for spoken phrases, and an
edit-delta scaling for replacements, insertions and deletions. The constants come
from `pe.config.yaml` at run time, so a config update is picked up rather than
copied.

Everything here is local. Where upstream asks for a source transcript (to scale an
edit by how much speech is added or removed), the transcript comes from Orukeet
through `auk_verify`. If no transcript can be obtained the base length is kept and
the caller says so.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml

AUK_HOME = Path(os.environ.get("AUK_HOME", Path.home() / "works/repos/AuK"))
CONFIG = Path(
    os.environ.get("AUK_PE_CONFIG", AUK_HOME / "src/auk/infer/pe.config.yaml")
)

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
EN_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")


class DurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _config() -> dict:
    if not CONFIG.is_file():
        raise DurationError(f"prompt enhancer config not found: {CONFIG}")
    cfg = yaml.safe_load(CONFIG.read_text())
    if not isinstance(cfg, dict) or "runtime" not in cfg:
        raise DurationError(f"unexpected shape in {CONFIG}")
    return cfg


def _duration_cfg() -> dict:
    return _config()["runtime"]["duration"]


def _rate(language: str | None) -> float:
    table = _duration_cfg()["seconds_per_utf8_byte"]
    lang = language if language in table else "en"
    return float(table[lang])


def _f5() -> dict:
    return _duration_cfg()["f5"]


@lru_cache(maxsize=1)
def nonverbal_rules() -> list[dict]:
    return _config()["tasks"]["nonverbal_edit"]["duration"]["nonverbal_adjustments"]


@lru_cache(maxsize=1)
def nonverbal_default() -> dict:
    return _config()["tasks"]["nonverbal_edit"]["duration"]["default_nonverbal_adjustment"]


def is_chinese(text: str) -> bool:
    return bool(CJK.search(text or ""))


def utf8_bytes(text: str | None) -> int:
    return len((text or "").encode("utf-8")) if (text or "").strip() else 0


def byte_weight(text: str | None, language: str | None = None) -> float:
    """Seconds of speech implied by the text, weighted per character's script.

    Mirrors pe.py: a character's language is its own script when it has one, else the
    preceding script, else the following one, else the declared language. Mixed
    scripts therefore get a proportional rate rather than a single global one.
    """
    value = text or ""
    if not value.strip():
        return 0.0
    fallback = language or ("zh" if is_chinese(value) else "en")

    script = [
        "zh" if CJK.fullmatch(ch) else ("en" if ch.isascii() and ch.isalpha() else None)
        for ch in value
    ]
    following: list[str | None] = [None] * len(value)
    ahead: str | None = None
    for i in range(len(value) - 1, -1, -1):
        if script[i]:
            ahead = script[i]
        following[i] = ahead

    weight = 0.0
    previous: str | None = None
    for i, ch in enumerate(value):
        lang = script[i]
        if lang is None:
            lang = previous or following[i] or fallback
        else:
            previous = lang
        weight += len(ch.encode("utf-8")) * _rate(lang)
    return weight


def _local_speed(text: str | None) -> float:
    """Short utterances are stretched by the model, so upstream slows the estimate."""
    cfg = _f5()
    return (
        float(cfg["short_text_speed"])
        if utf8_bytes(text) < int(cfg["short_text_byte_threshold"])
        else 1.0
    )


def _frames(seconds: float) -> float:
    cfg = _f5()
    rate, hop = int(cfg["sample_rate"]), int(cfg["hop_length"])
    return int(seconds * rate / hop) * hop / rate


def tts_seconds(text: str, language: str | None = None) -> float:
    """Length for text-only synthesis, in the codec's frame grid."""
    weight = byte_weight(text, language)
    if weight <= 0:
        raise DurationError("empty text: nothing to estimate a length from")
    return _frames(weight / _local_speed(text))


def clone_seconds(reference_seconds: float, reference_text: str, target_text: str,
                  language: str | None = None) -> float:
    """Length for a cloned or edited read: scale the reference by relative text weight."""
    target_weight = byte_weight(target_text, language)
    reference_weight = byte_weight(reference_text, language)
    if target_weight <= 0 or reference_weight <= 0:
        raise DurationError("empty text on one side: cannot scale a reference length")
    return _frames(reference_seconds * target_weight / reference_weight / _local_speed(target_text))


def spoken_seconds(text: str | None, language: str | None = None) -> float:
    """How long a phrase takes to say: 0.21 s per Chinese character, 0.30 s per word."""
    value = text or ""
    seconds = len(CJK.findall(value)) * 0.21 + len(EN_WORD.findall(value)) * 0.30
    if seconds > 0:
        return seconds
    units = len(value.split()) if (language or "en") == "en" else len(re.findall(r"\S", value))
    return units * (0.30 if (language or "en") == "en" else 0.21)


CONTENT_SLOTS = {
    "insert-before": ("text", None),
    "insert-after": ("text", None),
    "delete": (None, "target"),
    "delete-before": (None, "target"),
    "delete-after": (None, "target"),
    "replace": ("new", "old"),
    "lyric": ("new", "old"),
}

# The caller (auk_lengths) supplies exactly these slot names, taken from the preset
# parameters. A slot name outside this set can never be filled, and an unfilled slot
# silently drops a phrase from the length arithmetic: `replace` said "orig" against a
# preset that calls the parameter "old", so the replaced phrase was never subtracted and
# a shortening edit was given a 1.19x longer target, which the model filled by repeating
# the new phrase. Import-time check, because a quiet fallback here is what caused that.
SLOT_KEYS = {"new", "old", "text", "target"}
_unknown_slots = sorted(
    {slot for slots in CONTENT_SLOTS.values() for slot in slots if slot and slot not in SLOT_KEYS}
)
if _unknown_slots:
    raise DurationError(
        f"CONTENT_SLOTS names slots the caller cannot supply: {_unknown_slots}. "
        f"Known slots: {sorted(SLOT_KEYS)}"
    )


def edit_seconds(
    action: str,
    params: dict,
    base_seconds: float,
    source_transcript: str | None,
    language: str | None = None,
) -> tuple[float, str]:
    """Length for a content edit. Returns (seconds, how it was derived)."""
    try:
        add_slot, drop_slot = CONTENT_SLOTS[action]
    except KeyError:
        return base_seconds, "source length (no length rule for this edit)"

    if action == "replace":
        # A replace re-times inside the span it already occupies. Scaling the target made
        # the model misbehave on a real 13.3 s clip: a 1.19x target repeated the new phrase
        # twice, a 0.96x target garbled the tail, and the same edits at the source length
        # came back clean. Substituting words does not change how much audio there is;
        # inserting and deleting words does, which is why those still scale below.
        return base_seconds, "source length (a replace re-times inside the same span)"

    if source_transcript:
        original = spoken_seconds(source_transcript, language)
        if original > 0:
            edited = original
            if add_slot:
                edited += spoken_seconds(params.get(add_slot), language)
            if drop_slot:
                edited -= spoken_seconds(params.get(drop_slot), language)
            scale = max(0.05, edited) / original
            return base_seconds * scale, f"edit delta on transcript, {scale:.2f}x"

    if add_slot and drop_slot:
        original = spoken_seconds(params.get(drop_slot), language)
        if original > 0:
            scale = spoken_seconds(params.get(add_slot), language) / original
            return base_seconds * scale, f"phrase ratio {scale:.2f}x, no transcript"

    return base_seconds, "source length (no transcript to measure the edit against)"


def nonverbal_delta(subtype: str, sound: str) -> tuple[float, str]:
    """Seconds to add or remove for a nonverbal event, by event family."""
    adding = not str(subtype).startswith("delete")
    haystack = str(sound or "").casefold()
    for rule in nonverbal_rules():
        if any(str(k).casefold() in haystack for k in rule.get("keywords", [])):
            delta = float(rule["add"] if adding else rule["delete"])
            return delta, f"event family {rule['keywords'][0]}"
    delta = float(nonverbal_default()["add" if adding else "delete"])
    return delta, "default event adjustment"


def emotion_multiplier(emotion: str) -> tuple[float, str]:
    """Emotion edits change how long the same words take: sad speech runs 1.22x."""
    rule = _config()["tasks"]["emotion_edit"]["duration"]
    multipliers = rule["emotion_multipliers"]
    key = str(emotion or "").strip().casefold()
    if key in multipliers:
        return float(multipliers[key]), f"emotion {key} runs {float(multipliers[key]):.2f}x"
    fallback = float(rule["default_emotion_multiplier"])
    return fallback, f"emotion {key or 'unspecified'} uses the {fallback:.2f}x default"


def transcript(path: str) -> tuple[str | None, str]:
    """Local transcript of a clip, for the length rules that need one."""
    try:
        import auk_verify

        text, engine, _ = auk_verify.transcribe(path)
        return (text, f"{engine}") if text else (None, "orukeet returned nothing")
    except Exception as exc:  # the length still has a fallback, and the caller says which
        return None, f"{type(exc).__name__}: {exc}"