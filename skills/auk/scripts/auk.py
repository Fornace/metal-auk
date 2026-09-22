"""auk: run AuK speech generation and editing on Apple Silicon (MLX, 8-bit default).

    auk say --text "Hello there" --ref voice.wav
    auk instruct --description "a calm male voice" --text "The train leaves at six." --seconds 4
    auk edit --preset pitch --audio in.wav --semitones 2
    auk edit --preset content --audio in.wav --old "cat videos" --new "interpretive dance"
    auk run --instruction "<raw AuK instruction>" --audio in.wav --seconds 5
    auk batch --jobs jobs.jsonl --outdir out/
    auk verify --audio out.wav --expect "Hello there"
    auk doctor
    auk env

Every command prints a one-line result and can emit a JSON receipt with --json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import auk_engine as engine_mod  # noqa: E402
import auk_lengths  # noqa: E402
import auk_tasks  # noqa: E402


def add_settings_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--variant", choices=["flash", "base"], default="flash",
                   help="flash: 4-step distilled, fast (default). base: 32-step, higher edit fidelity.")
    p.add_argument("--bits", type=int, choices=[0, 4, 8], default=8,
                   help="quantization for DiT and Thinker; 0 means fp32. 8 is the validated default.")
    p.add_argument("--cache-limit", type=float, default=4.0, metavar="GB",
                   help="bound MLX's buffer cache in GB (default 4, 0 = unbounded). Left unbounded "
                        "it grows to tens of GB, makes macOS compress and swap, and costs about "
                        "half the throughput on long jobs.")
    p.add_argument("--match-level", action="store_true",
                   help="scale the result to the source's speech level: enhancement lands at "
                        "whatever level the model favours, up to 21 dB off on clean input.")
    p.add_argument("--sequential", action="store_true",
                   help="hold one stack in memory at a time: lower peak memory, slower per job.")
    p.add_argument("--seed", type=int, default=0, help="0 is deterministic across runs; -1 for random.")
    p.add_argument("--nfe", type=int, default=32, help="base variant only: ODE steps.")
    p.add_argument("--cfg", type=float, default=2.0, help="base variant only: guidance strength.")
    p.add_argument("--sway", type=float, default=-1.0, help="base variant only: sway sampling coefficient.")
    p.add_argument("--weights", default=engine_mod.DEFAULT_WEIGHTS, help="directory of MLX weights.")
    p.add_argument("--group-size", type=int, default=64, help="quantization group size.")
    p.add_argument("--json", action="store_true", help="print the receipt as JSON.")
    p.add_argument("--quiet", action="store_true", help="print only the output path.")


def settings_from(args: argparse.Namespace) -> engine_mod.Settings:
    return engine_mod.Settings(
        variant=args.variant,
        bits=None if args.bits == 0 else args.bits,
        group_size=args.group_size,
        sequential=args.sequential,
        weights=os.path.abspath(args.weights),
        seed=None if args.seed < 0 else args.seed,
        nfe=args.nfe,
        cfg=args.cfg,
        sway=args.sway,
        match_level=args.match_level,
        cache_limit_gb=args.cache_limit,
    )


def default_output(task: str, out_arg: str | None) -> str:
    if out_arg:
        return out_arg
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return os.path.join("generated", "auk", f"{task}-{stamp}.wav")


def emit(receipt: engine_mod.Receipt, args: argparse.Namespace) -> None:
    if args.quiet:
        print(receipt.output)
        return
    if args.json:
        print(json.dumps(receipt.as_dict(), indent=2))
        return
    print(f"wrote {receipt.output}")
    print(
        f"  {receipt.task} | {receipt.audio_seconds:.2f}s audio | gen {receipt.generate_seconds:.2f}s | "
        f"RTF {receipt.rtf:.2f} | load {receipt.load_seconds:.2f}s | peak RSS {receipt.peak_rss_gb:.2f} GB"
    )
    s = receipt.stats
    print(f"  peak {s['dbfs_peak']:.1f} dBFS | rms {s['rms']:.4f} | clipped {s['clipped_fraction'] * 100:.3f}%")
    for note in receipt.notes:
        print(f"  note: {note}")


def requires(cache: engine_mod.EngineCache, args: argparse.Namespace, task: str, instruction: str,
             audio: str | None, seconds: float | None, notes: list[str] | None = None) -> None:
    settings = settings_from(args)
    receipt = engine_mod.run_task(
        cache, task, instruction, default_output(task, args.output), settings,
        source_audio=audio, gen_seconds=seconds, notes=notes,
    )
    emit(receipt, args)


def cmd_say(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    if not args.ref:
        sys.exit("say needs --ref <reference.wav> (AuK clones the voice in that clip). For text-only use `auk instruct`.")
    notes: list[str] = []
    instr = auk_tasks.build_instruction("say", {"text": args.text}, notes)
    seconds = auk_lengths.resolve_length("say", {"text": args.text}, args.ref, args.seconds, notes)
    requires(cache, args, "say", instr, args.ref, seconds, notes)


def cmd_instruct(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    notes: list[str] = []
    params = {"description": args.description, "text": args.text}
    instr = auk_tasks.build_instruction("instruct", params, notes)
    seconds = auk_lengths.resolve_length("instruct", params, None, args.seconds, notes)
    requires(cache, args, "instruct", instr, None, seconds, notes)


def cmd_edit(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    if not args.audio:
        sys.exit("edit needs --audio <source.wav>")
    notes: list[str] = []
    if args.instruction:
        instr, task = args.instruction, args.preset or "edit"
        params = {}
    else:
        if not args.preset:
            sys.exit("edit needs --preset <name> or --instruction '<raw instruction>'")
        params = {k: v for k, v in json.loads(args.params or "{}").items()}
        for name in auk_tasks.PARAMS[args.preset]:
            if getattr(args, name, None) is not None:
                params[name] = getattr(args, name)
        try:
            instr = auk_tasks.build_instruction(args.preset, params, notes)
        except auk_tasks.PresetError as exc:
            sys.exit(str(exc))
        task = args.preset

    seconds = auk_lengths.resolve_length(task, params, args.audio, args.seconds, notes)
    auk_lengths.health_note(task, args.audio, notes)
    requires(cache, args, task, instr, args.audio, seconds, notes)


def cmd_run(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    requires(cache, args, "run", args.instruction, args.audio, args.seconds)


def cmd_batch(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    """Each JSONL line: {preset, params, audio, seconds, output} or {instruction, audio, seconds, output}."""
    if not os.path.isfile(args.jobs):
        sys.exit(f"no such jobs file: {args.jobs}")
    outdir = args.outdir or os.path.join("generated", "auk", "batch")
    os.makedirs(outdir, exist_ok=True)

    receipts, failures = [], 0
    with open(args.jobs) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            job = json.loads(line)
            try:
                notes: list[str] = []
                if job.get("instruction"):
                    instr = job["instruction"]
                    task = job.get("task", "run")
                    params = {}
                else:
                    task = job["preset"]
                    params = job.get("params", {})
                    instr = auk_tasks.build_instruction(task, params, notes)
                output = job.get("output") or os.path.join(outdir, f"{lineno:03d}-{task}.wav")
                seconds = auk_lengths.resolve_length(task, params, job.get("audio"), job.get("seconds"), notes)
                auk_lengths.health_note(task, job.get("audio"), notes)
                receipt = engine_mod.run_task(
                    cache, task, instr, output, settings_from(args),
                    source_audio=job.get("audio"), gen_seconds=seconds, notes=notes,
                )
                receipts.append(receipt)
                print(f"ok   [{lineno:3d}] {task:18s} {receipt.audio_seconds:5.2f}s in {receipt.generate_seconds:6.2f}s -> {receipt.output}")
            except Exception as exc:  # one bad job must not kill the batch, but it must be visible
                failures += 1
                receipts.append({"line": lineno, "task": job.get("preset") or job.get("task"), "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAIL [{lineno:3d}] {job.get('preset') or job.get('task')}: {type(exc).__name__}: {exc}")

    if args.json:
        print(json.dumps({"jobs": len(receipts), "failures": failures, "receipts": receipts}, indent=2))
    else:
        print(f"\n{len(receipts) - failures}/{len(receipts)} ok")
    if failures:
        sys.exit(1)


def cmd_verify(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    import auk_verify

    sys.exit(0 if auk_verify.verify(args) else 2)


def cmd_doctor(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    sys.exit(engine_mod.doctor(args.weights))


def cmd_env(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    import auk_config

    sys.exit(auk_config.report(args.json))


def cmd_pe(cache: engine_mod.EngineCache, args: argparse.Namespace) -> None:
    import auk_pe

    def run_fn(instruction: str, audio: str | None, seconds: float, notes: list[str]) -> None:
        requires(cache, args, "pe", instruction, audio, seconds, notes)

    auk_pe.enhance(args, run_fn)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="auk", description="AuK speech generation and editing on Apple Silicon")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("say", help="zero-shot TTS in a reference voice")
    s.add_argument("--text", required=True)
    s.add_argument("--ref", required=True, help="reference voice clip, any rate or channel count")
    s.add_argument("--seconds", type=float, default=None, help="output length; defaults to the reference length")
    s.add_argument("-o", "--output")
    add_settings_flags(s)
    s.set_defaults(func=cmd_say)

    s = sub.add_parser("instruct", help="text-only TTS from a voice description")
    s.add_argument("--description", required=True, help="voice description, e.g. 'a warm male voice speaking clear English'")
    s.add_argument("--text", required=True)
    s.add_argument("--seconds", type=float, required=False, default=None)
    s.add_argument("-o", "--output")
    add_settings_flags(s)
    s.set_defaults(func=cmd_instruct)

    s = sub.add_parser("edit", help="edit speech, singing, emotion, level or pitch")
    s.add_argument("--audio", required=True, help="source clip")
    s.add_argument("--preset", choices=sorted(auk_tasks.PRESETS), help="edit type")
    s.add_argument("--instruction", help="raw AuK instruction, bypasses presets")
    s.add_argument("--params", help='JSON object of preset parameters, e.g. \'{"old":"a","new":"b"}\'')
    s.add_argument("--seconds", type=float, default=None, help="target length; computed from the task when omitted")
    s.add_argument("-o", "--output")
    for name in ("text", "old", "new", "target", "semitones", "factor", "db", "emotion", "timbre", "sound", "anchor", "side", "where", "mode", "cue", "ordinal", "effect", "cleanup"):
        s.add_argument(f"--{name}", default=None)
    s.add_argument("--n", type=int, default=None, help="speaker number for separate, 1 is the first to talk")
    add_settings_flags(s)
    s.set_defaults(func=cmd_edit)

    s = sub.add_parser("run", help="raw instruction escape hatch")
    s.add_argument("--instruction", required=True)
    s.add_argument("--audio", default=None)
    s.add_argument("--seconds", type=float, default=None)
    s.add_argument("-o", "--output")
    add_settings_flags(s)
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("batch", help="many jobs, one model load")
    s.add_argument("--jobs", required=True, help="JSONL file of jobs")
    s.add_argument("--outdir", default=None)
    add_settings_flags(s)
    s.set_defaults(func=cmd_batch)

    s = sub.add_parser("verify", help="transcribe and measure a result")
    s.add_argument("--audio", required=True)
    s.add_argument("--expect", default=None, help="expected transcript, checked by word accuracy")
    s.add_argument("--lang", default="en", choices=["en", "zh"])
    s.add_argument("--source", default=None, help="original clip, to measure level and duration change")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_verify)

    s = sub.add_parser("pe", help="prompt enhancer: free-form ask into a validated instruction")
    s.add_argument("--instruction", required=True, help="what you want, in plain words")
    s.add_argument("--audio", default=None, help="clip to edit; omit for text or voice-clone work")
    s.add_argument("--ref", default=None, help="reference voice clip for cloning")
    s.add_argument("--target-duration", type=float, default=None, help="override the predicted length")
    s.add_argument("--llm-model", default=None, help="OpenAI-compatible model, default fornace-flash")
    s.add_argument("--run", action="store_true", help="run the prepared job immediately")
    s.add_argument("--seconds", type=float, default=None, help=argparse.SUPPRESS)
    s.add_argument("-o", "--output")
    add_settings_flags(s)
    s.set_defaults(func=cmd_pe)

    s = sub.add_parser("doctor", help="check weights, deps and store layout")
    s.add_argument("--weights", default=engine_mod.DEFAULT_WEIGHTS)
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("env", help="print resolved paths and configuration, exit 1 if one is missing")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_env)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cache = engine_mod.EngineCache()
    args.func(cache, args)


if __name__ == "__main__":
    main()