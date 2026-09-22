"""Verify AuK output by transcribing it and comparing against expectations.

Primary engine: Orukeet (oruk/orukeet, r3, q8 GGUF) on Metal — released 2026-09-09,
a Parakeet TDT 0.6B v3 finetune that beats Parakeet on 61/74 public splits
(LibriSpeech test-clean 1.46% WER) across 25 European languages, and ships an
official Metal runtime for Apple Silicon. It transcribes a 4 s clip in ~0.15 s
warm on an M5 Max, versus ~2 s for mlx-whisper turbo.

Orukeet has no Chinese. For zh the verifier falls back to mlx-whisper
(mlx-community/whisper-large-v3-turbo, downloaded on first use).

Expected text is compared word-wise, case and punctuation insensitive, because
ASR punctuation varies. The score is word accuracy (1 - WER).
"""

from __future__ import annotations

import json
import math
import os
import re
import statistics
import time
from pathlib import Path

AUK_ROOT = Path(os.path.expanduser(os.environ.get("AUK_HOME") or "~/works/repos/AuK"))
ORUKEET_INSTALL = Path(
    os.path.expanduser(os.environ.get("AUK_ORUKEET") or "~/works/models/orukeet/installation.json")
)

_orukeet = None


def _norm(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _word_accuracy(expected: str, got: str) -> tuple[float, list[str]]:
    """Return (accuracy, substitutions+insertions+deletions summary)."""
    exp, act = _norm(expected).split(), _norm(got).split()
    if not exp:
        return (1.0, []) if not act else (0.0, [])
    # Levenshtein on words
    prev = list(range(len(act) + 1))
    for i, e in enumerate(exp, 1):
        cur = [i]
        for j, a in enumerate(act, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (e != a)))
        prev = cur
    dist = prev[-1]
    return 1 - dist / len(exp), []


def transcribe(path: str, lang: str = "en") -> tuple[str, str, float]:
    """Transcribe with the best available engine. Returns (text, engine, seconds)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    if lang == "zh":
        return _transcribe_whisper(path, lang)
    return _transcribe_orukeet(path)


def _transcribe_orukeet(path: str) -> tuple[str, str, float]:
    global _orukeet
    t0 = time.time()
    if _orukeet is None:
        if not ORUKEET_INSTALL.is_file():
            # Loud on purpose: a missing verifier must not quietly become a whisper run with
            # different accuracy characteristics. Say what to do instead.
            raise SystemExit(
                f"Orukeet is not installed at {ORUKEET_INSTALL}. "
                f"Install it (see the skill's docs/INSTALL.md) and point AUK_ORUKEET at its "
                f"installation.json, or verify Chinese with --lang zh, which uses whisper."
            )
        from orukeet import Orukeet

        inst = json.loads(ORUKEET_INSTALL.read_text())
        _orukeet = Orukeet(
            model=inst["model"], runtime=inst["runtime"], device=inst.get("device", "metal")
        )
    text = _orukeet.transcribe(path)["text"]
    return text, "orukeet-r3-q8-metal", time.time() - t0


def _transcribe_whisper(path: str, lang: str) -> tuple[str, str, float]:
    import mlx_whisper

    t0 = time.time()
    r = mlx_whisper.transcribe(
        path, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language=lang
    )
    return r["text"], "whisper-large-v3-turbo-mlx", time.time() - t0


def _dbfs(path: str) -> tuple[float, float]:
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    peak = float(abs(wav).max()) if wav.size else 0.0
    rms = float(statistics.fmean(wav**2) ** 0.5) if wav.size else 0.0
    return 20 * math.log10(peak + 1e-12), 20 * math.log10(rms + 1e-12)


def voiced_fraction(path: str) -> float:
    """Share of 20 ms frames above 12% of the loudest frame.

    Transcription cannot tell whether speaker separation worked: removing one side of a
    conversation drops roughly half the timeline to silence while the other speaker keeps
    most of the words, so the transcript still reads like the full dialogue. This metric
    does show it. Source clips run 70-90% voiced, a good separation renders 25-45%.
    """
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    frame = max(int(0.02 * sr), 1)
    usable = len(wav) // frame * frame
    if not usable:
        return 0.0
    frames = wav[:usable].reshape(-1, frame)
    energies = (frames**2).mean(axis=1) ** 0.5
    threshold = max(float(energies.max()) * 0.12, 1e-4)
    return float((energies > threshold).mean())


def hf_fraction(path: str, cutoff: int = 4000) -> float:
    """Share of spectral energy above `cutoff`.

    The bandwidth-extension and effect-removal templates exist to put back highs that
    were cut or masked. This is the measurement that shows they did: a telephone-band
    clip downsampled to 8 kHz has essentially nothing above 4 kHz, and a restored
    wideband version has several percent.
    """
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if wav.size == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(wav * np.hanning(len(wav)))) ** 2
    freqs = np.fft.rfftfreq(len(wav), 1 / sr)
    total = float(spectrum.sum())
    return float(spectrum[freqs >= cutoff].sum() / total) if total > 0 else 0.0


def _frames(path: str, hop_ms: float = 5.0):
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    hop = max(int(sr * hop_ms / 1000), 1)
    usable = len(wav) // hop * hop
    if not usable:
        return np.zeros(0), np.zeros(0), sr
    frames = wav[:usable].reshape(-1, hop)
    energies = (frames**2).mean(axis=1) ** 0.5
    times = np.arange(len(energies)) * hop / sr
    return energies, times, sr


def noise_floor_db(path: str) -> float:
    """Level of the quietest tenth of the timeline, in dBFS.

    Denoising should push this down while leaving speech where it was. It is the
    measurement that shows a denoise instruction did something.
    """
    import numpy as np

    energies, _, _ = _frames(path)
    if energies.size == 0:
        return -120.0
    quiet = max(float(np.percentile(energies, 10)), 1e-6)  # clamp: true silence is -120
    return float(20 * math.log10(quiet))


def tail_db(path: str) -> tuple[float, float]:
    """(relative, absolute) level of the room after speech stops, in dB.

    Reverb is defined by how loud the room stays once you stop talking, so that is what
    this measures: the level 100 to 250 ms after each offset, as dB below the voice and
    as absolute dBFS. A reverberant recording keeps the room 10 to 20 dB under the voice;
    a cleaned one drops it to the silence floor. The relative figure saturates once the
    output gates its gaps, which is why the absolute one is returned too.

    Transcripts cannot see any of this, so without these two numbers an enhancement
    instruction cannot be told apart from one that did nothing.
    """
    import numpy as np

    energies, _, _ = _frames(path, hop_ms=5.0)
    if energies.size < 60:
        return 0.0, -120.0
    env = np.convolve(20 * np.log10(np.maximum(energies, 1e-12)), np.ones(3) / 3, mode="same")
    floor = float(np.percentile(env, 10))
    peak = float(np.percentile(env, 95))
    if peak - floor < 15:
        return 0.0, floor

    gate = min(peak - 3.0, floor + 12.0)
    start, end = int(0.100 / 0.005), int(0.250 / 0.005)
    relative: list[float] = []
    absolute: list[float] = []
    for k in range(1, len(env) - end):
        if env[k] < gate or env[k + 1] >= gate:
            continue
        tail = float(np.median(env[k + start:k + end]))
        relative.append(env[k] - tail)
        absolute.append(tail)
    if not relative:
        return 0.0, floor
    return float(np.median(relative)), float(np.median(absolute))


def speech_duty(path: str, above_db: float = 8.0) -> float:
    """Share of frames that sit more than `above_db` over the noise floor.

    Reverb smears speech into the gaps, so this rises. Cleaning the room back out
    lowers it.
    """
    import numpy as np

    energies, _, _ = _frames(path)
    if energies.size == 0:
        return 0.0
    floor = max(float(np.percentile(energies, 10)), 1e-12)
    threshold = floor * (10 ** (above_db / 20))
    return float((energies > threshold).mean())


def verify(args) -> bool:
    text, engine, dt = transcribe(args.audio, args.lang)
    result = {
        "audio": os.path.abspath(args.audio),
        "engine": engine,
        "transcribe_seconds": round(dt, 2),
        "transcript": text,
        "lang": args.lang,
    }

    if args.expect:
        acc, _ = _word_accuracy(args.expect, text)
        result["expected"] = args.expect
        result["word_accuracy"] = round(acc, 4)
        result["pass"] = acc >= 0.75  # a cloned voice must say the words; style differs
    else:
        result["pass"] = len(text.strip()) > 0

    if args.source:
        import soundfile as sf

        src_db, src_rms = _dbfs(args.source)
        out_db, out_rms = _dbfs(args.audio)
        src_info, out_info = sf.info(args.source), sf.info(args.audio)
        src_tail_rel, src_tail_abs = tail_db(args.source)
        out_tail_rel, out_tail_abs = tail_db(args.audio)
        result["delta"] = {
            "peak_dbfs": round(out_db - src_db, 2),
            "rms_dbfs": round(out_rms - src_rms, 2),
            "seconds": round(
                out_info.frames / out_info.samplerate - src_info.frames / src_info.samplerate, 2
            ),
            "voiced_fraction": round(voiced_fraction(args.audio), 3),
            "voiced_fraction_source": round(voiced_fraction(args.source), 3),
            "hf_fraction": round(hf_fraction(args.audio), 4),
            "hf_fraction_source": round(hf_fraction(args.source), 4),
            "noise_floor_db": round(noise_floor_db(args.audio), 1),
            "noise_floor_db_source": round(noise_floor_db(args.source), 1),
            "room_db_below_voice": round(out_tail_rel, 1),
            "room_db_below_voice_source": round(src_tail_rel, 1),
            "room_dbfs": round(out_tail_abs, 1),
            "room_dbfs_source": round(src_tail_abs, 1),
            "speech_duty": round(speech_duty(args.audio), 3),
            "speech_duty_source": round(speech_duty(args.source), 3),
        }
    else:
        floor = round(noise_floor_db(args.audio), 1)
        rel, absolute = tail_db(args.audio)
        result["voiced_fraction"] = round(voiced_fraction(args.audio), 3)
        result["hf_fraction"] = round(hf_fraction(args.audio), 4)
        result["noise_floor_db"] = floor
        result["room_db_below_voice"] = round(rel, 1)
        result["room_dbfs"] = round(absolute, 1)
        result["speech_duty"] = round(speech_duty(args.audio), 3)

    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
    else:
        verdict = "PASS" if result["pass"] else "FAIL"
        line = f"{verdict} | {engine} {result['transcribe_seconds']}s | {text!r}"
        if "word_accuracy" in result:
            line += f" | accuracy {result['word_accuracy']:.0%}"
        if "delta" in result:
            d = result["delta"]
            line += (
                f" | rms {d['rms_dbfs']:+.1f} dB, len {d['seconds']:+.2f} s"
                f"\n     floor {d['noise_floor_db_source']:.0f} -> {d['noise_floor_db']:.0f} dBFS"
                f", room {d['room_db_below_voice_source']:.0f} -> {d['room_db_below_voice']:.0f} dB below voice"
                f" ({d['room_dbfs_source']:.0f} -> {d['room_dbfs']:.0f} dBFS)"
                f", voiced {d['voiced_fraction_source']:.0%} -> {d['voiced_fraction']:.0%}"
                f", highs {d['hf_fraction_source']:.2%} -> {d['hf_fraction']:.2%}"
            )
        else:
            line += f" | voiced {result['voiced_fraction']:.0%}"
        print(line)
    return result["pass"]
