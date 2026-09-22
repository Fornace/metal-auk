"""Trained value vocabularies.

AuK was trained on discrete buckets, not a continuous dial. `pe.config.yaml`
enumerates them: speed 0.5/0.75/1.25/1.5/2.0, pitch 1-3 semitones either way,
volume 5/10/15 dB, eight emotions, a fixed event list for nonverbals, six effect
families. Asking for 1.4x or 7 dB is off-distribution and can return the source
audio unchanged.

Values are therefore snapped to the nearest bucket and the snap is reported, so a
request is never silently rounded. Strings are resolved through the same aliases
upstream accepts, so "phone" reaches the telephone template and "cheerful" reaches
happy.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Snap:
    value: object
    note: str | None = None


SPEEDS = (0.5, 0.75, 1.25, 1.5, 2.0)
SEMITONES = (-3, -2, -1, 1, 2, 3)
GAINS_DB = (-15, -10, -5, 5, 10, 15)

EMOTIONS = {
    "happy": ("happy", "cheerful", "joyful", "glad", "happiness", "开心"),
    "sad": ("sad", "sadness", "unhappy", "sorrowful", "悲伤"),
    "angry": ("angry", "anger", "mad", "furious", "愤怒"),
    "calm": ("calm", "neutral", "relaxed", "平静"),
    "fearful": ("fearful", "afraid", "scared", "fear", "恐惧"),
    "surprised": ("surprised", "surprise", "astonished", "惊讶"),
    "excited": ("excited", "excitement", "thrilled", "兴奋"),
    "disgusted": ("disgusted", "disgust", "厌恶"),
}

# Effect families from runtime.effect_aliases, with the strings upstream accepts.
EFFECTS = {
    "telephone": ("telephone", "phone", "tel", "电话"),
    "megaphone": ("megaphone", "loudspeaker", "bullhorn", "pa", "扩音器", "喇叭"),
    "underwater": ("underwater", "muffled", "muffle", "muddy", "hollow", "水下", "闷"),
    "clipping": ("clipping", "clipped", "clip", "distorted", "削波", "破音", "爆音"),
    "dropout": ("dropout", "drop-out", "cutoff", "cut-off", "packet loss", "丢包", "瞬断", "脱落"),
    "dc_offset": ("dc_offset", "dc offset", "dc", "offset", "直流偏置"),
    "_generic": ("generic", "unknown", "auto", "restore", "quality"),
}

CLEANUP = {
    "": ("", "none", "nothing", "off"),
    "denoise": ("denoise", "denoising", "noise", "noise reduction", "去噪", "降噪", "去底噪", "去杂音"),
    "dereverb": ("dereverb", "dereverberate", "reverb", "reverberation", "echo", "去混响", "去回声", "消除混响"),
    "denoise_dereverb": (
        "denoise_dereverb", "both", "noise and reverb", "denoise and dereverb",
        "去噪去混响", "去噪与去混响",
    ),
}

# nonverbal_events: canonical key -> (English name, Chinese name).
NONVERBAL = {
    "breath": ("breath", "呼吸声"),
    "laugh": ("laughter", "笑声"),
    "sigh": ("sigh", "叹气声"),
    "cough": ("cough", "咳嗽声"),
    "throatclearing": ("throat clearing", "清嗓声"),
    "sniff": ("sniff", "吸鼻声"),
    "tsk": ("tsk", "咂舌声"),
    "uhm": ("uhm", "呃的语气词"),
    "pause": ("pause", "停顿"),
    "elongation": ("elongation", "拉长音"),
    "surprise-oh": ("surprise oh", "哦的惊讶声"),
    "surprise-ah": ("surprise ah", "啊的惊讶声"),
    "surprise-wa": ("surprise wa", "哇的惊讶声"),
    "question-ei": ("question ei", "诶的疑问声"),
    "confirmation-en": ("confirmation hum", "嗯的应答声"),
}

NONVERBAL_ALIASES = {
    "laughter": "laugh", "laughing": "laugh", "chuckle": "laugh", "giggle": "laugh",
    "sighing": "sigh", "coughing": "cough", "clearing throat": "throatclearing",
    "throat-clearing": "throatclearing", "throat": "throatclearing",
    "sniffing": "sniff", "sniffle": "sniff", "smack": "tsk", "smacking": "tsk",
    "breathing": "breath", "panting": "breath", "inhale": "breath", "exhale": "breath",
    "um": "uhm", "erm": "uhm", "thinking": "uhm", "hmm": "uhm",
    "silence": "pause", "hesitation": "pause",
}


def _nearest(value: float, options: tuple) -> tuple[float, str | None]:
    best = min(options, key=lambda option: abs(option - value))
    if abs(best - value) < 1e-9:
        return best, None
    return best, f"{value:g} is not a trained step; using {best:g}"


def speed(factor: float) -> Snap:
    value, note = _nearest(float(factor), SPEEDS)
    return Snap(value, note and f"speed: {note}")


def semitones(count: float) -> Snap:
    value, note = _nearest(float(count), SEMITONES)
    return Snap(value, note and f"pitch: {note}")


def gain_db(db: float) -> Snap:
    value, note = _nearest(float(db), GAINS_DB)
    return Snap(value, note and f"volume: {note}")


def _lookup(table: dict, raw: str, label: str, allow_unknown: bool = False) -> Snap:
    text = str(raw or "").strip().casefold().replace("-", "_").replace(" ", "_")
    for canonical, aliases in table.items():
        if text == canonical or text in {str(a).casefold().replace(" ", "_") for a in aliases}:
            return Snap(canonical)
    if allow_unknown:
        return Snap(raw, f"{label} '{raw}' is not in the trained list; passing it through")
    known = ", ".join(sorted(k for k in table if not k.startswith("_")))
    raise ValueError(f"unknown {label} '{raw}'. Trained values: {known}")


def emotion(name: str) -> Snap:
    return _lookup(EMOTIONS, name, "emotion")


def effect(name: str) -> Snap:
    return _lookup(EFFECTS, name, "effect")


def cleanup(name: str) -> Snap:
    return _lookup(CLEANUP, name or "", "cleanup")


def nonverbal(sound: str) -> Snap:
    """Return the canonical event key. Unknown sounds pass through with a warning:
    upstream's event list is longer in Chinese than in English, so a caller may know
    something the English aliases lack, and the template still carries it."""
    name = str(sound or "").strip()
    key = name.casefold().replace("-", " ").strip()
    if key in NONVERBAL:
        return Snap(key)
    if key in NONVERBAL_ALIASES:
        return Snap(NONVERBAL_ALIASES[key])
    for canonical, (en, zh) in NONVERBAL.items():
        if key in {en.casefold(), zh}:
            return Snap(canonical)
    spoken = NONVERBAL.get(key)
    if spoken:
        return Snap(spoken[0])
    return Snap(name, f"nonverbal '{sound}' is not in the trained event list; passing it through")


def nonverbal_name(key: str, language: str) -> str:
    en, zh = NONVERBAL.get(key, (key, key))
    return zh if language == "zh" else en