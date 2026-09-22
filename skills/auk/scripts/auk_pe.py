"""Prompt Enhancer: the quality front end.

AuK's Prompt Enhancer (upstream `src/auk/infer/pe.py`) takes a free-form request
("make this whisper sound normal", "say this in my voice"), classifies the task,
derives the target duration, preprocesses the audio that AuK needs, and emits a
validated model instruction. Feeding it a half-formed instruction is the
difference between an edit that lands and one that silently does nothing.

Two upstream assumptions are replaced with what exists on this machine:

* ASR. Upstream tries Tencent Cloud recording-file ASR, then SenseVoiceSmall on
  CPU (a torch model download). Here an Orukeet adapter satisfies the same
  `ASRProvider` protocol: local, Metal, no cloud credentials, no extra weights.
* LLM. Upstream reads LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME. Those are
  filled from ~/.agent_credentials/envs/fornace-llm.env (FORNACE_LLM_*), which
  is an OpenAI-compatible endpoint. The default model is the cheapest capable
  tier, fornace-flash, because PE's work is short natural-language classification
  and rewriting.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

LLM_ENV = Path.home() / ".agent_credentials/envs/fornace-llm.env"
PE_PATH = Path(os.environ.get("AUK_HOME", Path.home() / "works/repos/AuK")) / "src/auk/infer/pe.py"

_module = None


def _read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        out[key.strip()] = value.strip().strip("'\"")
    return out


def llm_config(model: str | None = None) -> tuple[str, str, str]:
    """Return (api_key, base_url, model). Explicit environment wins over the file."""
    file_env = _read_env_file(LLM_ENV)
    key = os.environ.get("LLM_API_KEY") or file_env.get("FORNACE_LLM_API_KEY", "")
    base = os.environ.get("LLM_BASE_URL") or file_env.get("FORNACE_LLM_BASE_URL", "")
    name = model or os.environ.get("LLM_MODEL_NAME") or "fornace-flash"
    if not key or not base:
        raise SystemExit(
            f"no OpenAI-compatible LLM credentials: set LLM_API_KEY and LLM_BASE_URL "
            f"or fill {LLM_ENV}"
        )
    return key, base.rstrip("/"), name


def load_pe():
    """Import upstream pe.py as a module. It pulls torch, torchaudio, openai and
    the Tencent SDK at import time even when neither cloud path is used."""
    global _module
    if _module is None:
        if not PE_PATH.is_file():
            raise SystemExit(f"prompt enhancer not found: {PE_PATH}")
        spec = importlib.util.spec_from_file_location("auk_pe_upstream", PE_PATH)
        _module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _module  # dataclasses resolve their module by name
        spec.loader.exec_module(_module)
    return _module


def build_enhancer(model: str | None = None):
    pe = load_pe()

    class OrukeetASR:
        """Local ASRProvider over the Orukeet verifier (Metal, 25 languages)."""

        def transcribe(self, audio_path: str):
            import auk_verify

            text, engine, _ = auk_verify.transcribe(audio_path)
            return pe.ASRCall(
                model=engine, text=text or None, language=None, raw_response=None,
                error=None if text else "orukeet returned an empty transcript",
            )

    key, base, name = llm_config(model)
    return pe.PromptEnhancer(
        llm_api_key=key, llm_base_url=base, llm_model=name, asr_provider=OrukeetASR()
    )


def _rebalance(prepared: str | None, original: str | None) -> tuple[str | None, str | None]:
    """Undo upstream's broken whisper level target, visibly.

    pe.config.yaml ships `whisper.to_normal_target_rms: 0.000707945784384138`, which is
    10 ** (-63/20): the dBFS value pasted into an RMS field. PE therefore attenuates a
    whisper clip to -63 dBFS and AuK dutifully renders near-silence. When the prepared
    audio is more than 6 dB away from the source level we restore the source level and
    say so in the receipt. Nothing else about the preprocessing is touched.
    """
    if not prepared or not original or prepared == original:
        return prepared, None
    import math
    import tempfile

    import soundfile as sf

    def mono_rms(path: str) -> float:
        wav, _ = sf.read(path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return float(math.sqrt(float((wav**2).mean())))

    src_rms, prep_rms = mono_rms(original), mono_rms(prepared)
    if src_rms <= 0 or prep_rms <= 0:
        return prepared, None
    delta_db = 20 * math.log10(prep_rms / src_rms)
    if abs(delta_db) <= 6.0:
        return prepared, None

    wav, sr = sf.read(prepared, dtype="float32")
    gain = src_rms / prep_rms
    out = wav * gain
    peak = float(abs(out).max())
    if peak > 0.99:
        out *= 0.99 / peak
    handle, path = tempfile.mkstemp(prefix="auk_pe_rebalanced_", suffix=".wav")
    os.close(handle)
    sf.write(path, out, sr, subtype="FLOAT")
    note = (
        f"prepared clip was {delta_db:+.1f} dB off source level "
        f"(upstream whisper level target is -63 dBFS); restored source level"
    )
    return path, note


def enhance(args, run_fn) -> None:
    """Prepare the instruction, print it, and optionally run it through the engine."""
    import json

    pe = load_pe()
    source = args.audio or args.ref
    prepared = build_enhancer(getattr(args, "llm_model", None)).prepare(
        args.instruction, source, target_duration=args.target_duration
    )
    try:
        if getattr(args, "json", False):
            print(json.dumps({k: v for k, v in pe.asdict(prepared).items()}, ensure_ascii=False, indent=2))
        else:
            task = prepared.task_type + (f" / {prepared.operation_subtype}" if prepared.operation_subtype else "")
            print(f"task       {task}")
            print(f"language   {prepared.language}   length {prepared.gen_seconds:.2f}s ({prepared.duration_source})")
            print(f"audio      {prepared.audio or 'none (text-only)'}")
            print(f"instruction {prepared.instruction}")
            if prepared.reasoning:
                print(f"reasoning  {prepared.reasoning.strip()[:400]}")

        if getattr(args, "run", False):
            run_audio, note = _rebalance(prepared.audio, source)
            notes = [f"prompt enhancer: {prepared.task_type}",
                     f"length from {prepared.duration_source}"]
            if note:
                notes.append(note)
            try:
                run_fn(prepared.instruction, run_audio, prepared.gen_seconds, notes)
            finally:
                if note and run_audio != prepared.audio:
                    os.unlink(run_audio)
    finally:
        prepared.cleanup()