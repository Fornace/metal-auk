"""AuK MLX engine wrapper: warm engine cache and per-job receipts.

Wraps upstream's `auk_mlx.infer.AukMLX` (branch feat/mlx-apple-silicon) so that:

  * one engine survives across jobs instead of reloading 5.7 GB of weights each
    call (cold load is ~10 s, so batch and daemon use both matter),
  * every job returns a receipt with the exact settings, timings, RTF and audio
    statistics, so a claim about a run is checkable after the fact,
  * text-only jobs fail early with a clear message when no length was given.

Inference runs entirely on MLX. torch is present only because the HF Omni
processor and qwen_omni_utils import it for their image and video branches, which
AuK does not use.
"""

from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

INSTALL_ROOT = os.environ.get("AUK_HOME", os.path.expanduser("~/works/repos/AuK"))
DEFAULT_WEIGHTS = os.path.join(INSTALL_ROOT, "ckpts", "mlx-8bit")

def peak_rss_gb() -> float:
    """Peak resident set size of this process in GB (macOS reports bytes)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024**3) if sys.platform == "darwin" else raw / (1024**2)


@dataclass
class Settings:
    variant: str = "flash"
    bits: int | None = 8
    group_size: int = 64
    sequential: bool = False
    weights: str = DEFAULT_WEIGHTS
    seed: int | None = 0
    nfe: int = 32
    cfg: float = 2.0
    sway: float = -1.0
    match_level: bool = False
    cache_limit_gb: float = 4.0

    def config_path(self) -> str:
        name = "config_flash.yaml" if self.variant == "flash" else "config_base.yaml"
        return os.path.join(self.weights, name)

    def qwen_path(self) -> str:
        return os.path.join(self.weights, "qwen")

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "bits": self.bits,
            "group_size": self.group_size,
            "sequential": self.sequential,
            "match_level": self.match_level,
            "cache_limit_gb": self.cache_limit_gb,
            "weights": self.weights,
            "seed": self.seed,
            "nfe": self.nfe if self.variant == "base" else 4,
            "cfg": self.cfg if self.variant == "base" else 0.0,
            "sway": self.sway if self.variant == "base" else None,
        }

    def key(self) -> tuple:
        return (self.variant, self.bits, self.group_size, self.sequential, self.weights)


@dataclass
class Receipt:
    task: str
    instruction: str
    output: str
    source_audio: str | None
    gen_seconds: float
    audio_seconds: float
    load_seconds: float
    generate_seconds: float
    rtf: float
    peak_rss_gb: float
    settings: dict
    stats: dict
    warm: bool
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class EngineCache:
    """One AukMLX per (variant, bits, group_size, sequential, weights)."""

    def __init__(self) -> None:
        self._engines: dict[tuple, Any] = {}
        self.tuning: str | None = None

    def apply_runtime_tuning(self, settings: Settings) -> str:
        """Bound MLX's buffer cache. This is the single biggest win on this machine.

        MLX keeps freed buffers in an allocator cache and grows it without limit by
        default. Measured here on one 14 s enhancement: the cache reached 31.6 GB, macOS
        compressed and swapped 2.9 GB, and the job took 21.2 s. Holding the cache at
        4 GB instead left zero swapping and ran the same job in 7.5 s, and a 2.2 s
        clone went from 4.9-7.3 s to 2.5-3.4 s, on a machine under more load. The cap
        costs nothing measurable: short jobs reuse the same buffers either way.
        """
        import mlx.core as mx

        if self.tuning:
            return self.tuning
        env = os.environ.get("AUK_MLX_CACHE_LIMIT")
        limit_gb = float(env) if env else settings.cache_limit_gb
        if limit_gb <= 0:
            self.tuning = "MLX cache unbounded (--cache-limit 0): expect swapping on long jobs"
            return self.tuning
        mx.set_cache_limit(int(limit_gb * 1e9))
        self.tuning = f"MLX cache limited to {limit_gb:.1f} GB"
        return self.tuning

    def get(self, settings: Settings) -> tuple[Any, float, bool]:
        key = settings.key()
        self.apply_runtime_tuning(settings)
        if key in self._engines:
            return self._engines[key], 0.0, True

        sys.path.insert(0, os.path.join(INSTALL_ROOT, "src"))
        from auk_mlx.infer import AukMLX

        t0 = time.time()
        engine = AukMLX(
            settings.weights,
            settings.config_path(),
            settings.qwen_path(),
            bits=settings.bits,
            group_size=settings.group_size,
            sequential=settings.sequential,
        )
        load = time.time() - t0
        self._engines[key] = engine
        return engine, load, False

    def drop(self, settings: Settings | None = None) -> None:
        if settings is None:
            self._engines.clear()
        else:
            self._engines.pop(settings.key(), None)


def _stats(wav: Any, sr: int) -> dict:
    import numpy as np

    wav = np.asarray(wav, dtype=np.float32)
    rms = float(np.sqrt((wav**2).mean())) if wav.size else 0.0
    peak = float(np.abs(wav).max()) if wav.size else 0.0
    return {
        "sample_rate": sr,
        "samples": int(wav.size),
        "seconds": round(wav.size / sr, 4) if sr else 0.0,
        "rms": round(rms, 6),
        "peak": round(peak, 6),
        "dbfs_peak": round(20 * float(np.log10(peak + 1e-12)), 2),
        "clipped_fraction": round(float((np.abs(wav) >= 0.999).mean()), 6) if wav.size else 0.0,
    }


def speech_rms(wav: Any, sr: int) -> float:
    """RMS of the loud frames only: the voice, not the gaps.

    File-wide RMS cannot compare an enhancement input with its output, because the model
    gates silence to exact zero and that alone moves the number by tens of dB. Measuring
    the loud half of the frames reports the level of the speech itself.
    """
    import numpy as np

    wav = np.asarray(wav, dtype=np.float32)
    frame = max(int(0.02 * sr), 1)
    usable = len(wav) // frame * frame
    if not usable:
        return 0.0
    frames = wav[:usable].reshape(-1, frame)
    energies = (frames**2).mean(axis=1) ** 0.5
    loud = frames[energies >= float(np.percentile(energies, 60))]
    return float(np.sqrt((loud**2).mean())) if loud.size else 0.0


def match_level(wav: Any, sr: int, source_audio: str) -> tuple[Any, str]:
    """Scale the result so its speech sits at the source's speech level.

    Enhancement lands at whatever level the model's training favoured: on an already
    clean clip the voice came back 21 dB quieter, which is not something a caller asked
    for. With `--match-level` the output is brought to the source's speech RMS, capped at
    24 dB of correction and peak limited, and the receipt records the gain applied.
    """
    import numpy as np
    import soundfile as sf

    source, source_rate = sf.read(source_audio, dtype="float32")
    if source.ndim > 1:
        source = source.mean(axis=1)
    target = speech_rms(source, source_rate)
    current = speech_rms(wav, sr)
    if target <= 0 or current <= 0:
        return wav, "level match skipped: one side has no measurable speech"

    gain_db = 20 * float(np.log10(target / current))
    capped = max(-24.0, min(24.0, gain_db))
    out = np.asarray(wav, dtype=np.float32) * (10 ** (capped / 20))
    peak = float(np.abs(out).max()) if out.size else 0.0
    note = f"level matched to source speech: {capped:+.1f} dB"
    if abs(gain_db) > 24:
        note += f" (capped from {gain_db:+.1f} dB)"
    if peak > 0.99:
        out = out * (0.99 / peak)
        note += f", peak limited by {-20 * float(np.log10(0.99 / peak)):.1f} dB"
    return out, note


def run_task(
    cache: EngineCache,
    task: str,
    instruction: str,
    output: str,
    settings: Settings,
    source_audio: str | None = None,
    gen_seconds: float | None = None,
    notes: list[str] | None = None,
) -> Receipt:
    """Generate into `output`, returning a receipt. Raises on any real failure."""
    import soundfile as sf

    sys.path.insert(0, os.path.join(INSTALL_ROOT, "src"))
    from auk_mlx.infer import GenerateOptions

    if source_audio is not None:
        if not os.path.isfile(source_audio):
            raise FileNotFoundError(f"source audio not found: {source_audio}")
        if gen_seconds is None:
            info = sf.info(source_audio)
            gen_seconds = info.frames / info.samplerate
    if gen_seconds is None:
        raise ValueError(
            "this task needs a length: pass --seconds (no source audio or text to derive one from)"
        )

    engine, load_seconds, warm = cache.get(settings)
    if cache.tuning:
        notes = list(notes or []) + [cache.tuning]

    opts = GenerateOptions(gen_seconds=float(gen_seconds), seed=settings.seed)
    if settings.variant == "base":
        opts.nfe = settings.nfe
        opts.cfg_strength = settings.cfg
        opts.sway_sampling_coef = settings.sway
    # Flash overrides nfe/cfg/sway internally: 4 fixed steps, guidance off.

    t0 = time.time()
    wav, sr = engine.generate(instruction, audio_path=source_audio, opts=opts)
    generate_seconds = time.time() - t0

    if settings.match_level and source_audio:
        wav, level_note = match_level(wav, sr, source_audio)
        notes = list(notes or []) + [level_note]

    parent = os.path.dirname(os.path.abspath(output))
    if parent:
        os.makedirs(parent, exist_ok=True)
    sf.write(output, wav, sr)

    seconds = len(wav) / sr
    return Receipt(
        task=task,
        instruction=instruction,
        output=os.path.abspath(output),
        source_audio=os.path.abspath(source_audio) if source_audio else None,
        gen_seconds=float(gen_seconds),
        audio_seconds=round(seconds, 4),
        load_seconds=round(load_seconds, 2),
        generate_seconds=round(generate_seconds, 2),
        rtf=round(generate_seconds / seconds, 3) if seconds else 0.0,
        peak_rss_gb=round(peak_rss_gb(), 2),
        settings=settings.as_dict(),
        stats=_stats(wav, sr),
        warm=warm,
        notes=list(notes or []),
    )


def doctor(weights: str = DEFAULT_WEIGHTS) -> int:
    """Check weights, Python deps and helper engines. Returns process exit code."""
    import importlib.util
    import shutil

    checks: list[tuple[str, bool, str]] = []

    def want(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    try:
        import mlx.core as mx

        limit_gb = Settings().cache_limit_gb
        mx.set_cache_limit(int(limit_gb * 1e9))
        want("mlx cache bounded", True,
             f"{limit_gb:.1f} GB default; unbounded swapped 13.5 GB and took 26-28 s "
             f"where 4 GB took 7.5 s on a 14 s enhancement")
    except Exception as exc:  # a real failure: MLX could not bound its allocator cache
        want("mlx cache bounded", False, str(exc))

    venv = os.path.join(INSTALL_ROOT, ".venv/bin/python")
    want("AuK repo", os.path.isdir(os.path.join(INSTALL_ROOT, "src/auk_mlx")), INSTALL_ROOT)
    want("AuK venv", os.path.isfile(venv), venv)

    expect = [
        "config_flash.yaml", "config_base.yaml",
        os.path.join("qwen", "chat_template.json"),
    ]
    missing = [f for f in expect if not os.path.isfile(os.path.join(weights, f))]
    safetensors = [f for f in os.listdir(weights) if f.endswith(".safetensors")] if os.path.isdir(weights) else []
    want("weights complete", not missing and len(safetensors) >= 2,
         f"{weights} ({len(safetensors)} safetensors)" + (f"; missing {missing}" if missing else ""))

    for mod in ("mlx", "transformers", "scipy", "soundfile", "qwen_omni_utils", "orukeet", "mlx_whisper",
                "openai", "tencentcloud", "yaml"):
        want(f"dep {mod}", importlib.util.find_spec(mod) is not None)

    want("prompt enhancer", os.path.isfile(os.path.join(INSTALL_ROOT, "src/auk/infer/pe.py")))
    skills = Path(__file__).resolve().parent
    for name in ("auk.py", "auk_tasks.py", "auk_values.py", "auk_duration.py",
                 "auk_lengths.py", "auk_restoration.py", "auk_verify.py", "auk_pe.py",
                 "auk_profile.py"):
        want(f"runner {name}", (skills / name).is_file())
    want("brain config", Path(os.environ.get(
        "AUK_PE_CONFIG", os.path.join(INSTALL_ROOT, "src/auk/infer/pe.config.yaml"))).is_file())

    orukeet_install = Path.home() / "works/models/orukeet/installation.json"
    want("orukeet weights", orukeet_install.is_file(), str(orukeet_install))

    print(f"AuK doctor | weights {weights}")
    bad = 0
    for name, ok, detail in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")
        bad += not ok
    print(f"{len(checks) - bad}/{len(checks)} ok")
    return 1 if bad else 0
