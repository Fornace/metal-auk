---
license: mit
base_model:
  - tencent/AuK
  - tencent/AuK-Flash
library_name: mlx
tags:
  - mlx
  - audio
  - text-to-speech
  - speech-editing
  - speech-restoration
  - apple-silicon
  - metal
  - macos
---

# metal-auk: AuK on Apple Silicon

How to run [Tencent AuK](https://huggingface.co/tencent/AuK) on a Mac, with the tuning
measured on an M5 Max and the quirk list that comes with it.

AuK is a 1.5B speech foundation model for generation and editing, MIT licensed,
released 2026-09-09. Upstream ships an official MLX backend on the
`feat/mlx-apple-silicon` branch. What is missing upstream is not the backend but the
runner: 21 task presets, length arithmetic, value handling, verification, and the MLX
settings that keep the machine from swapping.

The runner is [Fornace/metal-auk](https://github.com/Fornace/metal-auk), a Python CLI
plus a pi skill. It runs standalone, with no pi dependency.

## Run it

Fastest route: paste the prompt in
[docs/INSTALL-PROMPT.md](https://github.com/Fornace/metal-auk/blob/main/skills/auk/docs/INSTALL-PROMPT.md)
into an agent on the Mac you want to run on. It installs the model, runner, weights and
verifier, then proves the install by generating and transcribing a clip. By hand:

```bash
git clone https://github.com/Fornace/metal-auk ~/works/repos/metal-auk
git clone -b feat/mlx-apple-silicon https://github.com/Tencent-Hunyuan/AuK.git ~/works/repos/AuK
cd ~/works/repos/AuK
hf download smcleod/AuK-MLX-8bit --local-dir ckpts/mlx-8bit   # 7.7 GB, complete bundle

uv venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[mlx]"
uv pip install --python .venv/bin/python jinja2 audioread av mlx-whisper openai \
  tencentcloud-sdk-python-asr                                  # undeclared upstream deps

.venv/bin/python ~/works/repos/metal-auk/skills/auk/scripts/auk.py doctor
.venv/bin/python ~/works/repos/metal-auk/skills/auk/scripts/auk.py env
.venv/bin/python ~/works/repos/metal-auk/skills/auk/scripts/auk.py \
  say --text "Hello there." --ref voice.wav -o out.wav
```

Python 3.10 fails on macOS 27: the newest scipy wheel for 3.10 is Mach-O invalid there.
Use 3.12.

## What was measured

### Bound MLX's allocator cache, it is the one real lever

MLX keeps freed buffers and grows that cache without limit. On one 14 s enhancement the
cache passed 27 GB and macOS compressed and swapped. Interleaved runs in a single
process, four runs each:

| MLX cache | 14 s enhance | memory swapped |
|---|---|---|
| unbounded | 26.2, 28.2, 28.1, 28.7 s | 13.5 GB |
| 2 GB | 8.4, 21.4, 23.5, 26.6 s | 0.9 GB |
| 4 GB, the default | 7.4, 7.7, 7.7, 7.5 s | 0.04 GB |

3.7x faster, and the spread collapses from 2.5 s to 0.04 s. Process footprint drops from
about 30 GB to 2 to 5 GB. Set it with `--cache-limit GB` or `AUK_MLX_CACHE_LIMIT`.

### Cold kernels, warm jobs

Warm steady state for a 2.2 s voice clone is 0.7 to 0.8 s. The first jobs in a fresh
process measured 5.6, 5.2 and 3.2 s, because Metal compiles kernels per shape on first
use. Warm per stage: Thinker 0.28 s, layer fusion 6 ms, DiT 0.24 s, VAE decode 0.18 s.
The VAE decode is the long-clip cost, roughly RTF 0.09 at 2 s against 0.33 at 14 s.

A DiT step costs 61 ms for 1.75B parameters at 288 text tokens, which matches the
39 TFLOPS this GPU sustains on a 4096 cubed bf16 matmul. The DiT is not the bottleneck.

### Checked and already optimal

- The port already uses `mx.fast.scaled_dot_product_attention`, `mx.fast.rms_norm`,
  `mx.fast.rope` and `mx.fast.layer_norm`.
- The audio tower slices the 30000 frame padded mel to the 995 real frames before the GPU.
- bf16 is not the lever: 0.71 ms against 0.66 ms for a quantized 128x4096x4096 matmul,
  and 22.9 ms against 22.7 ms for a whole DiT step.
- `mx.compile` on the VAE decode is slower, 6.8 s against 4.7 s for 14 s.
- bf16 VAE weights and latents: 1.1x at 2 s, nothing at 14 s, correlation 0.996.

### Quantization

8-bit group 64 is the right default. The 32-step `base` variant correlates 0.92 with the
4-step `flash` distill on the same input at 4 times the cost, so flash is the default and
`--variant base` is for the restoration family, where it measurably helps. Upstream
measured 8-bit against fp32 at a waveform correlation of 0.989 with identical
transcripts, and 4-bit degrades Chinese.

## Restoring bad audio

`enhance` is the instruction for the common case, a recording with room and noise. On a
clean 14 s clip degraded with a 0.6 s convolution reverb, pink noise and a 6.5 kHz low
pass:

| | input | after `enhance` |
|---|---|---|
| noise floor | -89 dBFS | -120 dBFS |
| room after speech stops | 11 dB below the voice | 43 dB below, exact digital silence |
| transcript | 100% | 100% |

Cleaning instructions gate 33 to 39% of output frames to exact digital silence, so file
RMS is a misleading way to compare input and output. They are also not idempotent: on
audio that was already clean, `enhance` returned the voice 21.3 dB quieter and 7 times
brighter above 4 kHz. Run it on audio that needs it and pass `--match-level`.

The instruction names promise more specialization than exists: `enhance` and
`enhance --cleanup dereverb` returned identical audio (correlation 0.999), and
`--cleanup denoise` matched `quality` (0.997) while differing from both (0.06).

## Verification

`auk verify` transcribes with Orukeet (Parakeet TDT 0.6B v3 finetune, q8 GGUF on Metal,
0.15 s per clip against about 2 s for whisper-large-v3-turbo) and reports what a
transcript cannot show: noise floor, room level after each speech offset, voiced frame
fraction for separation, and energy above 4 kHz for bandwidth extension. Chinese keeps a
whisper fallback.

## Two upstream bugs

- `pe.config.yaml` ships `whisper.to_normal_target_rms: 0.000707945784384138`, which is
  exactly 10^(-63/20), a dBFS value pasted into an RMS field. Every whisper-to-normal job
  through the Prompt Enhancer rendered at -63 dBFS. The runner detects it, compensates,
  and prints the correction.
- A content replace should not change clip length. Scaling the target by the phrase ratio
  made the model repeat the new phrase at 1.19x and garble the tail at 0.96x on a real
  13.3 s clip. At the source length the same edits came back clean.

## Weights

This repo hosts no tensors. The complete runnable bundles today:

| bundle | precision | complete |
|---|---|---|
| `smcleod/AuK-MLX-8bit` | 8-bit, both variants, plus fusion and processor files | yes |
| `tencent/AuK`, `tencent/AuK-Flash` | source, needs the MLX conversion step | yes |
| `vanch007/AuK-Base-MLX`, `AuK-Flash-MLX` | full precision DiT and VAE | no Thinker, no fusion files |

A complete full-precision MLX bundle, which would make the quality path practical without
a source conversion, is not published. Ask in the GitHub repo if you want it.

## License

The runner and these measurements are MIT. AuK and its MLX backend are Tencent's, MIT.
The Thinker and audio tower derive from Qwen2.5-Omni-3B, which carries Qwen's research
license. Orukeet, used for verification, is its own project.