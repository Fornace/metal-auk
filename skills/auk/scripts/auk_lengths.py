"""Target length and source health for a job.

Split out of the CLI: picking the span AuK should render, and reporting what the source
audio looks like before a cleanup instruction touches it. Both are upstream's own rules
rather than guesses, so they live where a test can call them directly.
"""

from __future__ import annotations

import auk_duration
import auk_verify


def _duration(path: str) -> float:
    import soundfile as sf

    info = sf.info(path)
    return info.frames / info.samplerate


_transcripts: dict[str, tuple[str | None, str]] = {}


def _source_transcript(path: str, notes: list[str]) -> str | None:
    """Local transcript of a clip, cached per path. Logged once, not per use."""
    if path not in _transcripts:
        _transcripts[path] = auk_duration.transcript(path)
        text, how = _transcripts[path]
        notes.append(f"source transcript via {how}" if text else f"no source transcript ({how})")
    return _transcripts[path][0]


def health_note(task: str, audio: str | None, notes: list[str]) -> None:
    """Report the source's room and noise floor, and flag the level caveat.

    A cleanup instruction is not idempotent: it re-renders the voice at the level its
    training favoured, which on a clean clip came out 21 dB quieter. There is no reliable
    way to tell from the source alone whether enhancement will help, so the runner reports
    the measurements and the standing caveat instead of guessing.
    """
    if not audio or task not in ("enhance", "quality", "restore"):
        return
    import auk_verify

    room, _ = auk_verify.tail_db(audio)
    floor = auk_verify.noise_floor_db(audio)
    notes.append(
        f"source health: room {room:.0f} dB below voice, floor {floor:.0f} dBFS"
        "; cleanup instructions do not preserve level, so consider --match-level"
    )


def resolve_length(
    task: str,
    params: dict,
    audio: str | None,
    seconds: float | None,
    notes: list[str],
) -> float | None:
    """Pick the target span. AuK renders a fixed length, so a wrong one truncates or
    pads speech. Every rule below is upstream's own duration strategy, computed
    locally; --seconds always wins."""
    if seconds:
        return seconds
    base = _duration(audio) if audio else None

    if task == "say":
        target = str(params.get("text") or "")
        reference_text = _source_transcript(audio, notes) if audio else None
        if base and reference_text and target:
            value = auk_duration.clone_seconds(base, reference_text, target)
            notes.append(f"length from reference wording ratio: {value:.2f}s")
            return value
        if base:
            notes.append(f"length kept at the reference {base:.2f}s: no transcript to scale from")
            return base
        return None

    if task == "instruct":
        value = auk_duration.tts_seconds(str(params.get("text") or ""))
        notes.append(f"length from text rate: {value:.2f}s")
        return value

    if base is None:
        return None

    if task == "speed":
        factor = float(params.get("factor") or 1.5)
        value = base / factor
        notes.append(f"speed {factor:g}x: target length {value:.2f}s")
        return value

    if task in ("content", "lyric", "content-insert", "content-delete"):
        source_text = _source_transcript(audio, notes)
        side = params.get("side") or "after"
        action = {"content": "replace", "lyric": "lyric"}.get(task)
        if action is None:
            # the presets carry the side as a word: before/after the anchor
            if task == "content-insert":
                action = f"insert-{side}"
            else:
                action = f"delete-{side}" if params.get("anchor") else "delete"
        slots = {"new": params.get("new"), "old": params.get("old"),
                 "text": params.get("text"), "target": params.get("target")}
        value, how = auk_duration.edit_seconds(action, slots, base, source_text)
        notes.append(f"length from {how}: {value:.2f}s")
        return value

    if task in ("nonverbal-add", "nonverbal-remove"):
        delta, how = auk_duration.nonverbal_delta(task, params.get("sound") or "")
        value = max(0.1, base + delta)
        notes.append(f"length {delta:+.2f}s for the event ({how}): {value:.2f}s")
        return value

    if task == "emotion":
        multiplier, how = auk_duration.emotion_multiplier(str(params.get("emotion") or ""))
        value = base * multiplier
        notes.append(f"length {how}: {value:.2f}s")
        return value

    notes.append(f"length kept equal to the source {base:.2f}s")
    return base


