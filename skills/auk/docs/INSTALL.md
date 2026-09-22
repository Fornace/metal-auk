# AuK MLX install (Apple Silicon, validated 2026-09-22)

Machine this was built on: M5 Max, 48 GB, macOS 27. Adapt paths if `AUK_HOME`
differs; the skill reads `AUK_HOME` (default `~/works/repos/AuK`).

## 1. Repo and weights

```bash
git clone https://github.com/Fornace/metal-auk ~/works/repos/metal-auk
git clone -b feat/mlx-apple-silicon https://github.com/Tencent-Hunyuan/AuK.git ~/works/repos/AuK
cd ~/works/repos/AuK
hf download smcleod/AuK-MLX-8bit --local-dir ckpts/mlx-8bit   # 7.7 GB
```

The skill itself lives in the metal-auk repo at `skills/auk`, and is normally
symlinked into place:

```bash
ln -s ~/works/repos/metal-auk/skills/auk ~/.pi/agent/skills/auk
```

Nothing below depends on pi: `scripts/auk.py` runs standalone.

`ckpts/mlx-8bit` must contain `config_flash.yaml`, `config_base.yaml`,
`qwen/` (thinker) and the DiT safetensors for both variants.

## 2. Python 3.12 venv, not 3.10

The newest scipy wheel published for 3.10 is Mach-O invalid on macOS 27
(`__thread_bss` validation). Python 3.12 gets scipy 1.18.1 which loads.

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -e ".[mlx]"
```

Upstream undeclared requirements that must be added by hand:

```bash
uv pip install --python .venv/bin/python jinja2 audioread av mlx-whisper \
  openai tencentcloud-sdk-python-asr
```

- jinja2: the Qwen chat template renderer, imported lazily, absent from deps.
- audioread, av: imported by qwen_omni_utils at processor call time.
- mlx-whisper: Chinese verification fallback only.
- openai, tencentcloud-sdk-python-asr: import-time requirements of the Prompt
  Enhancer. The Tencent ASR client is never called here; `auk_pe.py` injects a local
  Orukeet adapter through the same `ASRProvider` protocol.

torch/torchvision/torchaudio arrive via `.[mlx]`; they are import requirements of
the HF Omni processor, not the inference path. Inference is MLX.

## 3. Orukeet verifier (replaces whisper for 25 languages)

```bash
uv pip install --python .venv/bin/python \
  https://github.com/Oruk-AI/orukeet/releases/download/v0.1.1/orukeet-0.1.1-py3-none-any.whl
.venv/bin/orukeet install --device auto --cache ~/works/models/orukeet \
  --output ~/works/models/orukeet/installation.json
```

`--device auto` picks Metal on Apple Silicon. The installer downloads the q8 GGUF
(714 MB, sha-pinned) and a native runtime, and writes `installation.json`;
`auk_verify.py` reads that file. No PyPI release yet: the wheel comes from GitHub.

## 4. Prompt Enhancer LLM credentials

`auk pe` reads `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL_NAME`, falling back to
`~/.agent_credentials/envs/fornace-llm.env` (`FORNACE_LLM_API_KEY`,
`FORNACE_LLM_BASE_URL`) with model `fornace-flash`. No Tencent credentials are
needed: the cloud ASR path is replaced by local Orukeet.

## 5. Check

```bash
$AUK_HOME/.venv/bin/python ~/.pi/agent/skills/auk/scripts/auk.py doctor
```

26 checks green on a healthy install. One of them is `mlx cache bounded`: the runner
caps MLX's allocator cache at 4 GB, because left unbounded it grows past 27 GB and makes
macOS compress and swap, which costs about 3.7x on a 14 s job. Change it with
`--cache-limit GB` per run or `AUK_MLX_CACHE_LIMIT` per environment; `0` restores the
unbounded behaviour. See the speed section in SKILL.md for the measured numbers.

## Update notes

- Upstream branch moves: `git -C ~/works/repos/AuK pull` on
  `feat/mlx-apple-silicon`, then re-run doctor. The runner only imports
  `auk_mlx.infer` (AukMLX, GenerateOptions); API drift shows up as an ImportError
  at first engine load, loudly.
- Orukeet: re-run `orukeet install` after upgrading the wheel; the receipt pins
  absolute paths.
