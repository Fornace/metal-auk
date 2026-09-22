"""Instruction templates for every AuK capability.

AuK is instruction-driven: the same weights do TTS, editable speech, enhancement
and separation depending only on the wording. Getting that wording right is the
difference between a task working and producing mush, so the exact templates
validated by the upstream cookbook live here rather than being retyped per call.

Templates that upstream ships only in Mandarin stay in Mandarin: de-accent, whisper
conversion, separation and the head/tail nonverbal placements are Chinese
instructions in the release, and English phrasings of them were never validated.
Parameter values are snapped to the trained buckets in auk_values.
"""

from __future__ import annotations

from dataclasses import dataclass

import auk_restoration as restoration
import auk_values as values

# Instruction language: AuK reads both. Keep upstream's validated wording.
DEACCENT = "请把方言腔改成标准普通话发音,音色维持一致。"
WHISPER = "用小声耳语的方式把这段话说出来。"
WHISPER_NORMAL = "把这段耳语转换成正常说话的声音。"
ENHANCE = "Preserve all speakers, remove noise and reverberation, and output clean speech of the same length."
MUSIC_SEPARATE = "Keep the clean singing voice, drop all other audio."

CLEANUPS = ("", "denoise", "dereverb", "denoise_dereverb")


@dataclass(frozen=True)
class Preset:
    name: str
    needs_audio: bool
    needs_seconds: bool
    summary: str
    build: object  # callable(**params) -> str


def _say(text: str, **_kw) -> str:
    return f"Say the following with the same voice: '{text}'"


def _instruct(description: str, text: str, **_kw) -> str:
    return f'Based on the following description: "{description}", generate speech content "{text}".'


def _content(old: str, new: str, **_kw) -> str:
    return f"Replace '{old}' with '{new}'."


def _content_insert(text: str, anchor: str, side: str = "after", **_kw) -> str:
    return f"Add '{text}' {side} '{anchor}'."


def _content_delete(target: str, anchor: str = "", side: str = "after", **_kw) -> str:
    if anchor:
        return f"Remove '{target}' {side} '{anchor}'."
    return f"Remove '{target}'."


def _lyric(old: str, new: str, **_kw) -> str:
    return f'Change "{old}" to "{new}" in the vocal recording.'


def _pitch(semitones: float = 2.0, **_kw) -> str:
    snap = values.semitones(semitones)
    direction = "Raise" if snap.value >= 0 else "Lower"
    return f"{direction} the pitch by {abs(snap.value):g} semitones."


def _speed(factor: float = 1.5, **_kw) -> str:
    return f"Adjust the speech speed to {values.speed(factor).value:g}x."


def _volume(db: float = 10.0, **_kw) -> str:
    snap = values.gain_db(db)
    direction = "Increase" if snap.value >= 0 else "Decrease"
    return f"{direction} the volume by {abs(snap.value):g} dB."


def _emotion(emotion: str = "happy", **_kw) -> str:
    return f"Change the emotion to {values.emotion(emotion).value}."


def _timbre(timbre: str, **_kw) -> str:
    return f'Keep the spoken content unchanged and change the timbre to: "{timbre}".'


def _deaccent(**_kw) -> str:
    return DEACCENT


def _whisper(**_kw) -> str:
    return WHISPER


def _whisper_normal(**_kw) -> str:
    return WHISPER_NORMAL


def _nonverbal_add(sound: str = "cough", anchor: str = "", where: str = "", **_kw) -> str:
    """Add an event. Upstream validates anchored adds in English and the two
    positional ones (head, tail) only in Mandarin."""
    event = values.nonverbal(sound).value
    if where in ("head", "tail"):
        spot = "开头" if where == "head" else "结尾"
        return f"在语音{spot}增加{values.nonverbal_name(event, 'zh')}。"
    if anchor:
        side = "before" if where == "before" else "after"
        return f'Add a {values.nonverbal_name(event, "en")} {side} "{anchor}".'
    return f"Add a {values.nonverbal_name(event, 'en')} to the audio."


def _nonverbal_remove(sound: str = "cough", anchor: str = "", where: str = "", **_kw) -> str:
    event = values.nonverbal(sound).value
    if where in ("head", "tail"):
        spot = "开头" if where == "head" else "结尾"
        return f"将语音{spot}的{values.nonverbal_name(event, 'zh')}删除。"
    if anchor:
        side = "before" if where == "before" else "after"
        return f'Remove the {values.nonverbal_name(event, "en")} {side} "{anchor}" from the speech.'
    return f"Remove all the {values.nonverbal_name(event, 'en')} from the audio."


def _enhance(cleanup: str = "", **_kw) -> str:
    """The cookbook's own enhancement templates. Its quality-restoration wording is
    the one `restore` uses via the per-effect library in pe.config.yaml."""
    mode = values.cleanup(cleanup).value if cleanup else ""
    if mode == "denoise":
        return "Remove only the background noise, preserve everything else, and output audio of the same length."
    if mode == "dereverb":
        return "Remove only the room reverberation, preserve everything else, and output audio of the same length."
    return ENHANCE


def _quality(cleanup: str = "", **_kw) -> str:
    cleanup = cleanup or ""
    if cleanup not in restoration.BANDWIDTH:
        raise PresetError(f"quality cleanup must be one of {CLEANUPS}, got {cleanup!r}")
    return restoration.BANDWIDTH[cleanup]


def _restore(effect: str = "_generic", cleanup: str = "", **_kw) -> str:
    effect = (effect or "_generic").lower()
    cleanup = cleanup or ""
    if effect not in restoration.RESTORE:
        raise PresetError(
            f"restore effect must be one of {', '.join(sorted(restoration.RESTORE))}, got {effect!r}"
        )
    variants = restoration.RESTORE[effect]
    if cleanup not in variants:
        raise PresetError(
            f"restore effect '{effect}' supports cleanup {sorted(variants)}, got {cleanup!r}"
        )
    return variants[cleanup]


def _separate(cue: str = "", ordinal: str = "", n: int = 0, cleanup: str = "", **_kw) -> str:
    """Upstream validates separation instructions in Mandarin only: `by_content`
    names a phrase the target speaker says, `by_order` keeps the n-th speaker to start
    talking. Both templates are copied from pe.config.yaml."""
    if cue:
        return f"请只保留说'{cue}'的人，去掉其他说话人，输出等长纯净人声。"
    index = int(n or 0)
    if not index and ordinal:
        words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
        index = words.get(str(ordinal).casefold(), 0)
    if not index:
        raise PresetError(
            "separate needs --cue \"words the target speaker says\" or --ordinal first|second|--n 2"
        )
    chinese = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五"}.get(index, str(index))
    return f"这段音频中只保留第{chinese}个开始说话的人对应的语音，去掉其余说话人。"


def _music_separate(mode: str = "singing", **_kw) -> str:
    key = str(mode or "singing").casefold()
    if key in ("all", "all_human_voices", "speech_and_singing", "voices"):
        return "Keep all human voices, speech and singing alike, drop everything else."
    return MUSIC_SEPARATE


PRESETS: dict[str, Preset] = {
    "say": Preset("say", True, False, "Zero-shot TTS in the reference voice", _say),
    "instruct": Preset("instruct", False, True, "Text-only TTS from a voice description", _instruct),
    "content": Preset("content", True, False, "Replace one spoken phrase with another", _content),
    "content-insert": Preset("content-insert", True, False, "Insert a phrase before or after an anchor", _content_insert),
    "content-delete": Preset("content-delete", True, False, "Remove a phrase, optionally next to an anchor", _content_delete),
    "lyric": Preset("lyric", True, False, "Replace a phrase inside singing", _lyric),
    "pitch": Preset("pitch", True, False, "Shift pitch in semitones", _pitch),
    "speed": Preset("speed", True, False, "Change speaking rate", _speed),
    "volume": Preset("volume", True, False, "Change level in dB", _volume),
    "emotion": Preset("emotion", True, False, "Change expressed emotion", _emotion),
    "timbre": Preset("timbre", True, False, "Change voice timbre, keep the words", _timbre),
    "deaccent": Preset("deaccent", True, False, "Mandarin: dialect accent to standard", _deaccent),
    "whisper": Preset("whisper", True, False, "Mandarin: convert to whispered speech", _whisper),
    "whisper-normal": Preset("whisper-normal", True, False, "Mandarin: whispered speech back to normal", _whisper_normal),
    "nonverbal-add": Preset("nonverbal-add", True, False, "Insert a breath, laugh, cough or sigh", _nonverbal_add),
    "nonverbal-remove": Preset("nonverbal-remove", True, False, "Strip an event from the audio", _nonverbal_remove),
    "enhance": Preset("enhance", True, False, "Denoise and dereverberate", _enhance),
    "quality": Preset("quality", True, False, "Raise quality: bandwidth extension, add the missing highs", _quality),
    "restore": Preset("restore", True, False, "Raise quality: remove telephone, megaphone, muffled, clipped or dropped audio", _restore),
    "separate": Preset("separate", True, False, "Keep one speaker, drop the rest", _separate),
    "music-separate": Preset("music-separate", True, False, "Extract singing, or all human voices", _music_separate),
}

# AuK accepts these; anything else is a typo worth catching before a 7 s model call.
PARAMS = {
    "say": ("text",),
    "instruct": ("description", "text"),
    "content": ("old", "new"),
    "content-insert": ("text", "anchor", "side"),
    "content-delete": ("target", "anchor", "side"),
    "lyric": ("old", "new"),
    "pitch": ("semitones",),
    "speed": ("factor",),
    "volume": ("db",),
    "emotion": ("emotion",),
    "timbre": ("timbre",),
    "deaccent": (),
    "whisper": (),
    "whisper-normal": (),
    "nonverbal-add": ("sound", "anchor", "where"),
    "nonverbal-remove": ("sound", "anchor", "where"),
    "enhance": ("cleanup",),
    "quality": ("cleanup",),
    "restore": ("effect", "cleanup"),
    "separate": ("cue", "ordinal", "n", "cleanup"),
    "music-separate": ("mode",),
}

# Values snapped to trained buckets, and the note explaining any snap. Kept next to
# the presets so a new preset cannot silently skip its vocabulary.
SNAPPED = {
    "pitch": {"semitones": values.semitones},
    "speed": {"factor": values.speed},
    "volume": {"db": values.gain_db},
    "emotion": {"emotion": values.emotion},
    "restore": {"effect": values.effect, "cleanup": values.cleanup},
    "quality": {"cleanup": values.cleanup},
    "enhance": {"cleanup": values.cleanup},
    "separate": {"cleanup": values.cleanup},
    "nonverbal-add": {"sound": values.nonverbal},
    "nonverbal-remove": {"sound": values.nonverbal},
}

SIDES = ("before", "after")
NONVERBAL_PLACES = ("", "before", "after", "head", "tail")


class PresetError(ValueError):
    pass


def build_instruction(preset: str, params: dict, notes: list[str] | None = None) -> str:
    """Render the instruction for `preset`, rejecting unknown or missing parameters.

    Values outside the trained buckets are snapped, and every snap is appended to
    `notes` so the receipt shows what was actually asked of the model.
    """
    if preset not in PRESETS:
        raise PresetError(f"unknown preset '{preset}'. Known: {', '.join(sorted(PRESETS))}")

    params = dict(params)
    allowed = set(PARAMS[preset])
    unexpected = set(params) - allowed
    if unexpected:
        raise PresetError(f"preset '{preset}' takes {allowed or 'no parameters'}, got unexpected {sorted(unexpected)}")

    for name, snap_fn in SNAPPED.get(preset, {}).items():
        if params.get(name) in (None, ""):
            continue
        try:
            snapped = snap_fn(params[name])
        except ValueError as exc:
            raise PresetError(str(exc)) from exc
        params[name] = snapped.value
        if snapped.note and notes is not None:
            notes.append(snapped.note)

    if preset in ("nonverbal-add", "nonverbal-remove") and params.get("where"):
        where = str(params["where"]).casefold()
        if where not in NONVERBAL_PLACES:
            raise PresetError(f"where must be one of {NONVERBAL_PLACES}, got {where!r}")
        params["where"] = where
    if params.get("side"):
        side = str(params["side"]).casefold()
        if side not in SIDES:
            raise PresetError(f"side must be one of {SIDES}, got {side!r}")
        params["side"] = side

    try:
        return PRESETS[preset].build(**params)
    except TypeError as exc:
        raise PresetError(f"preset '{preset}' needs {allowed}: {exc}") from exc