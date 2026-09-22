---
name: auk
description: "Generate and edit speech natively on Apple Silicon with AuK (Tencent) via MLX 8-bit: voice-clone and instruct TTS, content insert delete and replace, lyric edits, pitch, speed, volume, emotion, timbre, nonverbal events, Mandarin de-accent and whisper conversion, denoise, quality restoration, speaker separation and vocal isolation, with Orukeet ASR verification."
---

# AuK on Apple Silicon

AuK (Tencent-Hunyuan, branch `feat/mlx-apple-silicon`, MLX backend) generates and edits
speech in one instruction-driven model. Everything runs locally on the GPU of an
Apple Silicon Mac; 8-bit quantization is the validated default.

Install lives at `~/works/repos/AuK` (repo, venv, weights). Weights:
`ckpts/mlx-8bit` (smcleod/AuK-MLX-8bit, 7.7 GB, flash + base + thinker).
See [docs/INSTALL.md](docs/INSTALL.md) before touching the environment.

This skill is published at [Fornace/metal-auk](https://github.com/Fornace/metal-auk)
(clone to `~/works/repos/metal-auk`, symlink `skills/auk` into place) with the measured
tuning in its README, and as the Hugging Face card
[fornace-ml/metal-auk](https://huggingface.co/fornace-ml/metal-auk).

## Run it

```bash
AU=~/works/repos/AuK/.venv/bin/python
SK=~/.pi/agent/skills/auk/scripts/auk.py

$AU $SK say --text "Hello from the runner." --ref ~/works/repos/AuK/assets/demo-input-audio/zero-shot-tts/ref.wav
$AU $SK verify --audio generated/auk/say-*.wav --expect "Hello from the runner"
```

All commands: `say instruct edit run pe batch verify doctor`. `--json` prints a full
receipt (timings, RTF, peak RSS, dbfs). Outputs default to `generated/auk/`.
Speed on this machine lives in `scripts/auk_profile.py`, which times each stage behind
an `mx.eval` barrier; the measured defaults are the 4 GB MLX cache cap and the flash
variant.

## Every preset

All 17 upstream task types are reachable. Presets are the validated wording; a near
miss is a silent no-op, so prefer them over hand-written instructions.

| Preset | What it does | Key parameters |
|---|---|---|
| `say` | clone the voice in a reference clip | `--text --ref` |
| `instruct` | text-only TTS from a voice description | `--description --text` |
| `content` | replace a spoken phrase | `--old --new` |
| `content-insert` | insert a phrase next to an anchor | `--text --anchor --side before\\|after` |
| `content-delete` | remove a phrase, optionally anchored | `--target [--anchor --side]` |
| `lyric` | replace a phrase inside singing | `--old --new` |
| `pitch` | shift pitch | `--semitones +-1\\|2\\|3` |
| `speed` | change speaking rate | `--factor 0.5\\|0.75\\|1.25\\|1.5\\|2` |
| `volume` | change level | `--db +-5\\|10\\|15` |
| `emotion` | change expressed emotion | `--emotion` (8 values) |
| `timbre` | change voice timbre, keep the words | `--timbre` |
| `deaccent` | Mandarin dialect to standard | |
| `whisper` / `whisper-normal` | Mandarin whisper in either direction | |
| `nonverbal-add` | insert breath, laugh, cough, sigh | `--sound --where head\\|tail\\|before\\|after --anchor` |
| `nonverbal-remove` | strip an event | `--sound [--where --anchor]` |
| `enhance` | denoise, dereverb, or both. First choice for reverb and noisy recordings | `--cleanup denoise\\|dereverb` |
| `quality` | bandwidth extension, add the missing highs | `--cleanup` |
| `restore` | remove telephone, megaphone, muffled, clipped, dropped or DC-offset audio | `--effect --cleanup` |
| `separate` | keep one speaker | `--cue` or `--ordinal`/`--n` |
| `music-separate` | singing only, or all human voices | `--mode singing\\|all` |

## Values are snapped to trained buckets

AuK learned discrete buckets, not a continuous dial, and an off-distribution number
can return the source audio unchanged. The runner snaps to the nearest trained step
and says so in the receipt: `--factor 1.4` becomes 1.5, `--db 7` becomes 5 dB,
`--semitones 5` becomes 3. Emotions are limited to happy, sad, angry, calm, fearful,
surprised, excited, disgusted, with common synonyms resolved (`cheerful` reaches
happy). Unknown ones are rejected before a model call. Effects accept upstream's
aliases, so `--effect phone` reaches the telephone template.

## Lengths are computed, not guessed

AuK renders a fixed span: too short truncates speech, too long pads it. Upstream's
own duration rules are ported in `auk_duration.py`, so `--seconds` is optional
everywhere and only needed to override.

| Task | Length |
|---|---|
| `say` | reference length scaled by wording, from a local Orukeet transcript of the reference |
| `instruct` | text rate: 0.0656 s per byte of English, 0.0803 per byte of Chinese, on the codec's frame grid |
| `content`, `content-insert`, `content-delete`, `lyric` | source length scaled by how much speech the edit adds or removes |
| `speed` | source length divided by the factor |
| `emotion` | source length times the emotion multiplier: sad 1.22x, fearful 1.16x, others 1.06x |
| `nonverbal-add` / `nonverbal-remove` | source length plus the event delta: laugh, sigh and cough +0.75 s, breath +0.35 s, tsk and sniff +0.5 s |
| everything else | source length unchanged |

## Two quality dials

1. **Variant.** `--variant base` is AuK, the 32-step model upstream describes as the
   high-quality one; `--variant flash` is AuK-Flash, the 4-step distill used by default
   because it is roughly twice as fast at equal quality on generation. Reach for base
   when a flash edit misses structure rather than wording (pitch, heavy edits).
2. **Prompt Enhancer.** `auk pe --instruction "..." [--audio clip] [--run]` runs
   upstream `pe.py`: it classifies the task, derives the target duration, preprocesses
   the audio AuK needs, and rewrites a loose ask into a valid instruction plus length.
   A half-formed instruction is the usual reason an edit does nothing. ASR is local
   Orukeet and the LLM is the fornace gateway (`fornace-flash` by default, `--llm-model`
   to change). Add `--run` to generate immediately in the same process.

```bash
$AU $SK pe --instruction "make this whisper sound normal, same speaker, same length" \
  --audio ~/works/repos/AuK/assets/demo-input-audio/whisper/wh-w2n-zh-input.wav --run
```

## Defaults that matter

| Setting | Default | Why |
|---|---|---|
| bits | 8 | validated quality/speed point; fp16 doubles memory for marginal gain |
| variant | flash | 4-step distill, RTF 0.6-1.3 warm; `--variant base` is the 32-step high-quality model |
| seed | 0 | deterministic reruns; `--seed -1` for variety |
| length | computed per task | see the length table; `--seconds` overrides |
| level | left as the model returns it | `--match-level` restores the source speech level; enhancement can land 21 dB low |
| cache limit | 4 GB | bounds MLX's allocator cache, which otherwise grows past 27 GB and makes macOS swap; see the speed section |
| prompt enhancer | pe.py, fornace-flash, local Orukeet ASR | rewritten instruction plus derived length |
| sampling | flash fixes its own 4 steps | `--nfe --cfg --sway` apply to base only |

## Quirks learned from testing (2026-09-22, M5 Max, 8-bit)

- Python 3.10 venv breaks: newest scipy wheel for 3.10 is Mach-O invalid on macOS 27.
  Use Python 3.12 (the venv already does).
- `qwen_omni_utils` has undeclared imports: `audioread` and `av` must be installed.
- jinja2 is required by the chat template and missing from upstream deps.
- torch is required only so the HF Omni processor and qwen_omni_utils import; inference is MLX.
- `content` edits only replace words actually spoken; check the transcript first
  (`verify --audio source.wav` without `--expect`).
- `speed` needs the shrunken target length or you get an 11 s "faster" clip that is
  the same length with mangled words. The runner sets it automatically.
- `separate` uses the Mandarin template because that is the only one upstream ships.
  Measured on an 88%-voiced two-person clip: 27% voiced after separation in Mandarin,
  42-46% with an English instruction, so the Mandarin wording separates materially
  better. Judge separation with `verify --source`, which reports the voiced fraction.
  Do not judge it by transcript: removing one side of a conversation leaves the other
  speaker's words intact, so the transcript still reads like the full dialogue.
- `pe` whisper conversion arrives 37 dB too quiet: `pe.config.yaml` ships
  `to_normal_target_rms: 0.000707945784384138`, which is 10^(-63/20), a dBFS value in an
  RMS field. The runner restores the source level and notes it in the receipt.
- Whisper conversion and de-accent instructions are Mandarin; English phrasings
  are unvalidated. Verify zh output with `verify --lang zh` (whisper-turbo fallback).
- Installed `openai` and the Tencent ASR SDK are import-time requirements of `pe.py`,
  not features in use: the Tencent and SenseVoice paths are both replaced by Orukeet.
- Text-only `instruct` TTS gets a length from the text rate, so `--seconds` is no
  longer required; pass it only to override.
- `volume +10 dB` measures about +7 dB rms; treat volume asks as approximate.
- Peak RSS 3-7 GB per engine, and the MLX cache cap keeps the process footprint near
  2-5 GB instead of 30 GB. flash + base held together fits in 48 GB; on smaller Macs pass
  `--sequential` or run one variant per process.
- The first jobs in a fresh process are slow. Warm steady state for a 2.2 s clone is
  0.7-0.8 s, but early jobs in the same process measured 5.6, 5.2 and 3.2 s. Metal
  compiles kernels per shape on first use, so a one-shot job pays seconds that a batch
  pays once. Measure medians over repeats, never a single job.
- MLX's allocator cache is unbounded by default and that is the single worst thing about
  running this on a Mac. See the speed section: it is now capped at 4 GB.
- `content` replace keeps the source length, deliberately. A replace re-times inside the span
  it already occupies. On a 13.3 s clip, a 1.19x target made the model say the new phrase
  twice, a 0.96x target garbled the tail, and the same edit at the source length came back
  clean in both the shortening and the lengthening direction.
- `content-insert` is the fragile edit. On that same 13.3 s clip a six-word insert was clean
  at 14.5 and 15.5 s but garbled at the 16.3 s the arithmetic asks for; a fifteen-word insert
  was imperfect at 21.6, 18.1, 16.6 and 15.3 s. Trim with `--seconds` and check the
  transcript rather than trusting the derived length for a long insert.
- Length rules name their slots from the preset parameters. A slot name the caller cannot
  supply is now a hard error at import, because one silently mismatched name ("orig" against
  a parameter called "old") meant a replaced phrase was never subtracted and a shortening
  edit was given a longer target. That is the bug that produced the repeat above.

## Raising audio quality: what is measured

The `quality` and `restore` instructions were tested against a control, since a
restoration that does nothing is invisible in a transcript. Energy above the source's
own Nyquist can only come from the model, so a plain resample is the baseline.

On a real 8 kHz recording (`se-zh-1-input.wav`, so nothing above 4 kHz exists):

| Output | above 4 kHz | above 6 kHz |
|---|---|---|
| plain resample to 24 kHz (control) | 0.0053% | 0.0000% |
| `quality` bandwidth extension | 0.0258% | 0.0140% |
| `enhance` denoise and dereverb | 0.0000% | 0.0000% |
| `restore --effect telephone` | 0.0001% | 0.0001% |

What that means in practice:

- `quality` is the one that adds high frequencies, about 5x the control above 4 kHz and
  the only variant producing anything above 6 kHz. The gain is real and modest.
- `restore --effect telephone` removes the telephone coloration but does not add highs,
  and on a synthetically band-limited file it dulls the voice (spectral centroid 632 Hz
  to 307 Hz). Use it to strip a coloration, not to recover bandwidth.
- `enhance` leaves the spectrum alone by design. It is for noise and reverb.
- Neither variant rebuilds a wideband signal from an aggressively filtered one: on a
  real 24 kHz clip brickwalled at 3.4 kHz, `quality` left the 0.56% of energy above
  3.4 kHz at 0.59% on base and 0.48% on flash. Set expectations at "clearer", not
  "restored".

## Bad audio: which instruction

Reverb and a noisy low-quality recording are the common case, and they are what
`enhance` is for. Order of preference:

1. `enhance` when the recording sounds like a room, a phone, a laptop mic or a noisy
   space. It is the instruction that removes noise and reverberation together.
2. `quality` when the problem is missing high frequencies, not noise: a genuinely
   narrowband source. It is the one that generates content above the input's Nyquist.
3. `restore --effect` when you can name the coloration: telephone, megaphone, underwater
   or muffled, clipping, packet dropouts, DC offset. Use `--effect generic` for a
   low-quality recording with no single cause.

Measured on a 14 s clean recording degraded with a 0.6 s convolution reverb, pink noise
and a 6.5 kHz low pass, flash variant, 8 bit:

| | input | after `enhance` |
|---|---|---|
| noise floor | -89 dBFS | -120 dBFS |
| room level after speech stops | 11 dB below the voice, -87 dBFS | 43 dB below, exact digital silence |
| transcript | 100% word accuracy | 100% word accuracy |
| speech level | -39.4 dBFS | -41.2 dBFS |

On a real 8 kHz noisy recording from the release (`se-zh-1-input.wav`): noise floor
-32 dBFS to -120 dBFS, speech level -23.2 to -29.7 dBFS, content preserved.

Specialization between the instructions is weaker than the names suggest. On that same
degraded clip, `enhance` and `enhance --cleanup dereverb` returned the same audio
(correlation 0.999), and `enhance --cleanup denoise` matched `quality` (0.997) while
differing from both (0.06). Pick `enhance` and move on rather than tuning the cleanup
word. `--variant base` correlates 0.92 with flash at 4x the time (29.6 s against 6.8 s
for 14 s), so flash is the right default.

Two behaviours to expect:

- **Gaps are gated to exact zero.** 33 to 39% of output frames are digital silence in
  every enhancement test, room tone included. That is why `verify` reports the room
  level inside speech offsets, and why file RMS is a misleading way to compare input and
  output.
- **Level is not preserved.** On audio that was already clean, `enhance` returned the
  voice 21.3 dB quieter and 7x brighter above 4 kHz: enhancement is not idempotent and it
  damages clean input. Only run it on audio that needs it, and pass `--match-level` to
  bring the result back to the source speech level. The receipt records the gain, for
  example `level matched to source speech: +21.3 dB, peak limited by 5.2 dB`.

## Speed on Apple Silicon: what was measured

Everything below was measured on this machine with the profiler that ships with the skill
(`python scripts/auk_profile.py --task ...`), which times each stage behind an `mx.eval`
barrier so the numbers are real GPU time rather than dispatch time.

**The allocator cache is the one real lever.** MLX keeps freed buffers and grows that
cache without limit. On a 14 s enhancement the cache reached 27 to 37 GB, macOS compressed
and swapped 3.4 GB per job, and the job took 26 to 28 s. Capping the cache at 4 GB in the
same process, interleaved with the uncapped runs so the machine's other load cancels out:

| MLX cache | 14 s enhance, four runs | swapped |
|---|---|---|
| unbounded | 26.2, 28.2, 28.1, 28.7 s | 13.5 GB |
| 2 GB | 8.4, 21.4, 23.5, 26.6 s | 0.9 GB |
| **4 GB (default)** | **7.4, 7.7, 7.7, 7.5 s** | **0.04 GB** |

3.7x faster and the spread collapses from 2.5 s to 0.04 s. 2 GB is too tight to reuse
buffers, so 4 GB is the default; `--cache-limit GB` changes it and `--cache-limit 0`
returns the unbounded behaviour, as does `AUK_MLX_CACHE_LIMIT`. Short jobs are unaffected:
interleaved 2.2 s clones measured 7.9 s capped against 11.1 s uncapped at their best.

These runs happened at load average 21 from other sessions on this machine, which is why
the absolute seconds look bad and why every comparison above is interleaved within one
process. On a quiet machine expect the capped numbers.

**Where the time goes, per stage** (2.2 s clone, warm): Thinker 0.28 s, ELMo fusion 6 ms,
DiT 0.24 s, VAE decode 0.18 s. Per DiT step, 61 ms for a 1.75 GB quantised model with 288
text tokens, which matches the compute ceiling of the GPU measured at 39 TFLOPS for a
4096-cubed bf16 matmul. The DiT is not the problem.

**The VAE decode is the long-clip cost**, roughly RTF 0.09 at 2 s and 0.33 at 14 s, so a
14 s render is dominated by decoding, not by the model.

**Checked and found already optimal, so do not spend time here:**

- The port already uses `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm`,
  `mx.fast.rope` and `mx.fast.layer_norm`. Hand-rolled attention would have been the
  classic slow path; there is none to fix.
- The audio tower slices to the unpadded mel length, so the 30000-frame padded tensor from
  the HF processor never reaches the GPU (995 real frames out of 30000).
- Compute dtype changes nothing. bf16 quantised matmul measured 0.71 ms against 0.66 ms
  for fp32 on a 128x4096x4096 layer, and a whole DiT step measured 22.9 ms in bf16 against
  22.7 ms in fp32.
- `mx.compile` on the VAE decode is slower, not faster: 6.8 s against 4.7 s for 14 s.
- bf16 weights plus bf16 latents in the VAE: 1.1x on a 2 s decode and no better on 14 s,
  with a correlation of 0.996 against fp32. Not worth the quality risk.

## Measured on this machine (48 GB M5 Max, 8-bit, warm engine)

RTF is generation time over audio length, load excluded, measured with the 4 GB cache cap
and a warm engine. Early jobs in a fresh process are several times slower than these
numbers, and this machine's other load moved absolute timings by up to 4x between runs, so
read these as order of magnitude.

| Task | RTF | Note |
|---|---|---|
| say 6 s clone | 1.3 | perfect transcript |
| instruct 4.5 s | 1.3 | 100% word accuracy |
| content swap 7 s | 0.73 | flash, real phrase |
| speed 1.5x | 1.04 | exact transcript after length fix |
| volume +10 dB | ~7.6 | rms delta +7.2 dB |
| pitch +3 st | 2.0 | base variant, words preserved |
| cue separation 28 s | 0.58 | 27% voiced against 88% source, Mandarin template |
| enhance 14 s reverb+noise | 0.48 | floor -89 to -120 dBFS, room to digital silence, transcript intact |
| enhance 14 s, base variant | 2.11 | same result as flash at 4x the time |
| enhance 8 kHz noisy real clip | 1.79 | floor -32 to -120 dBFS |
| content insert 7 s | 0.41 | phrase landed, length 7.01 to 8.14 s |
| content delete 7 s | 0.54 | phrase removed, length 7.01 to 4.80 s |
| laugh at head 5.5 s | 1.35 | length +0.76 s, the trained +0.75 event delta |
| bandwidth extension 5 s | 0.29 | above 4 kHz 0.0053% control to 0.0258% |
| prompt enhancer + whisper fix 8.6 s | 0.39 | rms within 1 dB of source |
| Orukeet verify 4 s clip | 0.04 | 0.15 s, Metal |
| 2.2 s clone, warm steady state | 0.32 | 0.7-0.8 s per job once compiled; first jobs in a process cost 5 s |

## Verify engine

Orukeet r3 q8 (oruk/orukeet, released 2026-09-09, Parakeet TDT 0.6B v3 finetune,
25 European languages, beats Parakeet on 61/74 splits, LibriSpeech test-clean
1.46% WER) on Metal, installed at `~/works/models/orukeet`. It replaced
mlx-whisper as the verifier: 13x faster per clip and more accurate. Chinese falls
back to whisper-large-v3-turbo (first zh call downloads 1.6 GB).

`verify` compares transcripts word-wise (case and punctuation insensitive) and
passes at 75% word accuracy. With `--source` it also reports what a transcript cannot
show, which is the only way to tell a working cleaning instruction from a no-op:

| Metric | Reads | Use it for |
|---|---|---|
| `room_db_below_voice`, `room_dbfs` | level 100-250 ms after each speech offset | reverb: 11 dB below the voice in, 43 dB and silent out |
| `noise_floor_db` | quietest tenth of the timeline | denoise: -89 dBFS in, -120 out |
| `voiced_fraction` | share of active frames | separation: 88% in, 27% after |
| `hf_fraction` | energy above 4 kHz | bandwidth extension, against a plain-resample control |
| `rms_dbfs`, `seconds` | whole-file level and length | level and length drift |

## Files

- `scripts/auk.py` CLI
- `scripts/auk_engine.py` engine cache, receipts, doctor
- `scripts/auk_tasks.py` validated instruction templates for every preset
- `scripts/auk_restoration.py` bandwidth-extension and effect-removal library
- `scripts/auk_values.py` trained value vocabularies and snapping
- `scripts/auk_duration.py` upstream length rules, computed locally
- `scripts/auk_lengths.py` target length and source health for a job
- `scripts/auk_verify.py` Orukeet plus whisper-zh verification, level, room, noise, voiced and high-frequency metrics
- `scripts/auk_profile.py` per-stage timing for one job, behind `mx.eval` barriers
- `scripts/auk_pe.py` prompt enhancer front end, local ASR, gateway LLM, level repair
- `docs/INSTALL.md` full environment recipe
