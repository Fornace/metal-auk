# metal-auk

AuK speech generation, editing and restoration on Apple Silicon, over MLX. A Python
CLI with 21 task presets, plus a pi skill that teaches an agent to use it.

AuK is Tencent's 1.5B speech foundation model, MIT licensed, released 2026-09-09.
Upstream ships an official MLX backend on the `feat/mlx-apple-silicon` branch, and
this repo is the runner around it: the presets, the length arithmetic, the value
snapping, the verification, and the tuning that makes it behave on a Mac.

## What it does

```
auk say      --text "Hello there" --ref voice.wav        voice clone in 0.8 s warm
auk instruct --description "calm male voice" --text ...  voice from a description
auk edit     --preset enhance --audio bad.wav            denoise and dereverb
auk edit     --preset quality --audio narrowband.wav     bandwidth extension
auk edit     --preset restore --audio phone.wav --effect telephone
auk batch    --jobs jobs.jsonl                           many jobs, one model load
auk verify   --audio out.wav --source in.wav             transcript and measurements
auk pe       --instruction "clean up this noisy recording" --audio bad.wav
auk doctor                                              26 install checks
```

The 21 presets cover the whole upstream task surface: content replace, insert and
delete, whisper conversion in both directions, emotion, pitch, speed, volume,
timbre, de-accent, casual speech, nonverbal insert and remove, lyric and vocal
extraction, music separation, speaker separation by order or by cue, plus the
quality family (`enhance`, `quality`, `restore`).

Values snap to the buckets AuK was actually trained on. Speed 1.4 becomes 1.5,
pitch 4 semitones becomes 3, volume 12 dB becomes 10, and the receipt prints the
snap, because an off-distribution value is silently ignored by the model.

Lengths are computed from upstream's own duration rules, ported and tested, so
`--seconds` is an override rather than a requirement.

## Install

See [docs/INSTALL.md](skills/auk/docs/INSTALL.md). Short version: clone Tencent's
MLX branch, pull 8-bit weights, build a Python 3.12 venv, add the dependencies upstream
forgot to declare (jinja2, audioread, av, plus the Prompt Enhancer's openai import), then
run `auk doctor`.

The skill is meant to live at `~/.pi/agent/skills/auk`, which can be a symlink to
`skills/auk` in this repo. Nothing in it depends on pi: the CLI runs standalone.

## Measured on an M5 Max, 48 GB

Every number below was measured on this machine, at machine load average 10 to 22
from unrelated work, so comparisons are interleaved within one process and medians
over repeats. The absolute seconds are worse than a quiet machine would give.

### Bound MLX's allocator cache, it is the one real lever

MLX keeps freed buffers and grows that cache without limit. On one 14 s
enhancement the cache passed 27 GB, macOS compressed and swapped, and the job took
26 to 28 s. Holding the cache at 4 GB, interleaved with the uncapped runs in the
same process:

| MLX cache | 14 s enhance, four runs each | memory swapped |
|---|---|---|
| unbounded | 26.2, 28.2, 28.1, 28.7 s | 13.5 GB |
| 2 GB | 8.4, 21.4, 23.5, 26.6 s | 0.9 GB |
| 4 GB, the default | 7.4, 7.7, 7.7, 7.5 s | 0.04 GB |

3.7x faster, and the spread collapses from 2.5 s to 0.04 s. 2 GB is too tight to
reuse buffers. `--cache-limit GB` changes it, `AUK_MLX_CACHE_LIMIT` sets it per
environment, and 0 restores the unbounded behaviour. As a side effect the process
footprint drops from about 30 GB to 2 to 5 GB.

### Cold kernels, warm jobs

Warm steady state for a 2.2 s voice clone is 0.7 to 0.8 s. The first jobs in a
fresh process measured 5.6, 5.2 and 3.2 s, because Metal compiles each kernel per
shape on first use. Keep one process alive for a batch, and time medians rather
than single jobs.

Per stage, warm, for that clone: Thinker 0.28 s, layer fusion 6 ms, DiT 0.24 s,
VAE decode 0.18 s. The VAE decode is the cost on long clips, roughly RTF 0.09 at
2 s against 0.33 at 14 s.

### Checked and already optimal, so do not spend time here

- The port uses `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm`,
  `mx.fast.rope` and `mx.fast.layer_norm`. There is no slow hand-rolled path.
- The audio tower slices the 30000 frame padded mel down to the 995 real frames
  before the GPU sees it.
- Compute dtype is not the lever. bf16 quantized matmul measured 0.71 ms against
  0.66 ms for fp32 on a 128x4096x4096 layer, and a whole DiT step measured 22.9 ms
  in bf16 against 22.7 ms in fp32.
- `mx.compile` on the VAE decode is slower: 6.8 s against 4.7 s for a 14 s clip.
- bf16 weights and latents in the VAE give 1.1x on a 2 s decode and nothing at
  14 s, at a correlation of 0.996. Not worth the quality risk.

A DiT step costs 61 ms for 1.75B parameters at 288 text tokens, which matches the
39 TFLOPS this GPU sustains on a 4096 cubed bf16 matmul. The DiT is not the
bottleneck.

## Restoring bad audio

`enhance` is the instruction for the common case, a recording with room and noise.
On a clean 14 s clip degraded with a 0.6 s convolution reverb, pink noise and a
6.5 kHz low pass:

| | input | after `enhance` |
|---|---|---|
| noise floor | -89 dBFS | -120 dBFS |
| room level after speech stops | 11 dB below the voice | 43 dB below, exact digital silence |
| transcript | 100% | 100% |
| speech level | -39.4 dBFS | -41.2 dBFS |

On a real 8 kHz noisy recording from the release: floor -32 to -120 dBFS, speech
level -23.2 to -29.7 dBFS, content preserved.

Two things to know. Cleaning instructions gate 33 to 39% of output frames to exact
digital silence, so file RMS is a misleading way to compare input and output. And
they are not idempotent: on audio that was already clean, `enhance` returned the
voice 21.3 dB quieter and 7 times brighter above 4 kHz. Run it on audio that needs
it, and pass `--match-level` to bring the result back to the source speech level.

The instruction names promise more specialization than exists. On the same
degraded clip `enhance` and `enhance --cleanup dereverb` returned identical audio
(correlation 0.999), and `--cleanup denoise` matched `quality` (0.997) while
differing from both (0.06). Pick `enhance` and move on.

Bandwidth extension was verified against a control rather than assumed: on a true
8 kHz source, `quality` produced about 5 times the control's energy above 4 kHz,
where the control is a plain upsample of the same input.

## Verification, because a transcript cannot judge an edit

`auk verify` transcribes with Orukeet (Parakeet TDT 0.6B v3 finetune, q8 GGUF on
Metal, 0.15 s per clip against about 2 s for whisper-large-v3-turbo) and, with
`--source`, reports what text cannot show:

- `noise_floor_db`, the quietest tenth of the timeline, for denoise
- `room_db_below_voice` and `room_dbfs`, the level 100 to 250 ms after each speech
  offset, for reverb
- `voiced_fraction`, the share of active frames, for separation
- `hf_fraction`, energy above 4 kHz, for bandwidth extension

Chinese keeps a whisper fallback, since Orukeet has no Chinese.

## Bugs found upstream, and fixed here

- `pe.config.yaml` ships `whisper.to_normal_target_rms: 0.000707945784384138`,
  which is exactly 10^(-63/20), a dBFS value pasted into an RMS field. Every
  whisper-to-normal job through the Prompt Enhancer rendered at -63 dBFS. The
  runner detects the value, compensates, and prints the correction in the receipt.
- A content replace should not change the clip length. Scaling the target by the
  phrase ratio made the model repeat the new phrase at 1.19x and garble the tail
  at 0.96x on a real 13.3 s clip. At the source length the same edits came back
  clean in both directions. The length rule now says so, in the code.

## The skill

`skills/auk/SKILL.md` is the agent-facing document, and it is where the quirks
live in the form that prevents them: which instruction to reach for, what each
measurement means, and every trap this test pass uncovered. `auk doctor` runs 26
checks and fails loudly rather than falling back.

## Credit and license

This repo is the runner and its measurements. The model and the MLX backend are
Tencent's:

- [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK), MIT, branch
  `feat/mlx-apple-silicon`
- Thinker and audio tower derive from Qwen2.5-Omni-3B, which carries Qwen's
  research license
- 8-bit MLX weights: [smcleod/AuK-MLX-8bit](https://huggingface.co/smcleod/AuK-MLX-8bit)
- Official weights: tencent/AuK and tencent/AuK-Flash on Hugging Face
- Verifier: [Orukeet](https://github.com/Oruk-AI/orukeet)

Code in this repo is MIT, see [LICENSE](LICENSE). No model weights are hosted
here.