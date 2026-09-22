"""Where the seconds go: per-stage timing for one AuK job on this machine.

MLX dispatches asynchronously, so a naive `time.time()` around a call measures the
Python dispatch, not the work. Every stage here is followed by an `mx.eval` on its
result, which is the barrier that makes the number real.

The profiler runs the skill's own `run_task` path and wraps the engine's stage methods
around it, so what it times is the code that ships, not a copy of it.

    python auk_profile.py --task say --ref ref.wav --seconds 2.5
    python auk_profile.py --task enhance --audio bad.wav --variant base
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auk_engine as engine_mod  # noqa: E402


def _barrier(value) -> None:
    """Make the GPU work real before the clock is read."""
    import mlx.core as mx

    if isinstance(value, (list, tuple)):
        mx.eval(*[v for v in value if hasattr(v, "shape")])
    elif hasattr(value, "shape"):
        mx.eval(value)


def profile(args: argparse.Namespace) -> dict:
    import auk_tasks
    import mlx.core as mx

    settings = engine_mod.Settings(
        variant=args.variant,
        bits=None if args.bits == 0 else args.bits,
        group_size=args.group_size,
        weights=os.path.abspath(args.weights),
        seed=0 if args.seed < 0 else args.seed,
    )
    cache = engine_mod.EngineCache()

    t0 = time.time()
    engine, load_seconds, warm = cache.get(settings)
    build_seconds = time.time() - t0

    stages: dict[str, float] = {}

    def timed(name, attr_owner, attr):
        original = getattr(attr_owner, attr)

        def wrapper(*a, **k):
            start = time.time()
            result = original(*a, **k)
            _barrier(result)
            stages[name] = round(stages.get(name, 0.0) + time.time() - start, 3)
            return result

        setattr(attr_owner, attr, wrapper)

    timed("vae_encode", engine, "encode_audio")
    timed("text_encode", engine, "encode_text")
    timed("sample", engine, "sample")
    timed("vae_decode", engine.vae, "decode")

    params = json.loads(args.params) if args.params else {}
    try:
        instruction = auk_tasks.build_instruction(args.task, params)
    except Exception as exc:  # an unknown preset or a missing parameter: say so, do not guess
        raise SystemExit(f"cannot build instruction for {args.task!r}: {exc}")

    t1 = time.time()
    receipt = engine_mod.run_task(
        cache,
        args.task,
        instruction,
        args.output,
        settings,
        source_audio=args.audio or args.ref,
        gen_seconds=args.seconds,
    )
    wall = time.time() - t1

    steps = 4 if settings.variant == "flash" else settings.nfe
    cached = stages.get("sample", 0.0)
    return {
        "task": args.task,
        "variant": settings.variant,
        "bits": settings.bits,
        "group_size": settings.group_size,
        "audio_seconds": receipt.audio_seconds,
        "engine_warm": warm,
        "load_seconds": round(load_seconds, 2),
        "build_seconds": round(build_seconds, 2),
        "generate_seconds": receipt.generate_seconds,
        "rtf": receipt.rtf,
        "rtf_excluding_load": round(receipt.generate_seconds / receipt.audio_seconds, 3)
        if receipt.audio_seconds
        else None,
        "steps": steps,
        "sample_per_step_seconds": round(cached / steps, 3) if steps else None,
        "stages": stages,
        "unattributed_seconds": round(receipt.generate_seconds - sum(stages.values()), 3),
        "wall_with_overheads_seconds": round(wall, 2),
        "peak_rss_gb": receipt.peak_rss_gb,
        "mlx_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "mlx_cache_gb": round(mx.get_cache_memory() / 1e9, 2),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", default="say")
    p.add_argument("--params", default="", help="JSON params for the task, as auk_tasks.render expects")
    p.add_argument("--audio", default=None, help="source audio for edit tasks")
    p.add_argument("--ref", default=None, help="reference voice for cloning tasks")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--output", default="/tmp/auk-profile.wav")
    p.add_argument("--variant", choices=["flash", "base"], default="flash")
    p.add_argument("--bits", type=int, default=8, help="0 = fp32")
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--weights", default=engine_mod.DEFAULT_WEIGHTS)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(profile(args), indent=2))


if __name__ == "__main__":
    main()