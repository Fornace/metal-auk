# Delegated install: paste this into your agent

Copy everything in the fenced block below into your agent (pi, or any coding agent with
shell access) on the Mac you want to run AuK on. It installs AuK over MLX, the runner,
the weights and the Orukeet verifier, then proves the install by generating and
transcribing a clip. Nothing in it depends on a specific home directory: everything the
runner needs resolves from environment variables, and the prompt ends by recording the
choices it made.

Expect about 10 GB of downloads (7.7 GB of weights, 714 MB of verifier, plus the
environment) and 20 minutes on a fast connection.

````text
Install AuK for Apple Silicon on this Mac, using the runner published at
https://github.com/Fornace/metal-auk. Work from that repo's skills/auk/docs/INSTALL.md,
which is the contract; where the doc and this machine disagree, the machine wins and you
say so.

Recon first, then tell me what you found before installing anything:
- confirm Apple Silicon and report the chip and RAM (`sysctl -n machdep.cpu.brand_string`,
  `sysctl -n hw.memsize`). AuK runs comfortably on 16 GB with one model variant resident,
  32 GB is comfortable with both, and the runner caps MLX's allocator cache at 4 GB.
- report free disk (`df -h ~`). This needs about 20 GB.
- confirm python 3.12 and uv are available (`python3.12 -V`, `uv --version`) and the
  Hugging Face CLI can reach the hub (`hf --version`). Install what is missing.
- if ~/works/repos/AuK or ~/works/repos/metal-auk already exists, stop and ask whether to
  reuse or replace it.

Then install, with these defaults, and export AUK_HOME if you put it anywhere else:
- runner:   git clone https://github.com/Fornace/metal-auk ~/works/repos/metal-auk
- upstream: git clone -b feat/mlx-apple-silicon https://github.com/Tencent-Hunyuan/AuK.git ~/works/repos/AuK
- weights:  cd ~/works/repos/AuK && hf download smcleod/AuK-MLX-8bit --local-dir ckpts/mlx-8bit
- venv:     cd ~/works/repos/AuK && uv venv --python 3.12 && uv pip install --python .venv/bin/python -e ".[mlx]"
- upstream forgot to declare: uv pip install --python .venv/bin/python jinja2 audioread av
  mlx-whisper openai tencentcloud-sdk-python-asr
- verifier: uv pip install --python .venv/bin/python
  https://github.com/Oruk-AI/orukeet/releases/download/v0.1.1/orukeet-0.1.1-py3-none-any.whl
  then .venv/bin/orukeet install --device auto --cache ~/works/models/orukeet
  --output ~/works/models/orukeet/installation.json
- skill:    ln -s ~/works/repos/metal-auk/skills/auk ~/.pi/agent/skills/auk

Do NOT install Python 3.10. The newest scipy wheel for 3.10 is Mach-O invalid on macOS 27,
which is why the venv is 3.12.

The runner finds everything from environment variables, so tell me which of these to put in
my shell profile and print the exact lines:
  AUK_HOME        install root, default ~/works/repos/AuK
  AUK_WEIGHTS     weights directory, default $AUK_HOME/ckpts/mlx-8bit
  AUK_ORUKEET     Orukeet installation.json, default ~/works/models/orukeet/installation.json
  AUK_PE_CONFIG   pe.config.yaml, default $AUK_HOME/src/auk/infer/pe.config.yaml
  AUK_MLX_CACHE_LIMIT   GB cap on MLX's allocator cache, default 4
  LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME   any OpenAI-compatible endpoint, for `auk pe` only

Verify, do not assume. Run both of these and show me the raw output:
  $AUK_HOME/.venv/bin/python ~/works/repos/metal-auk/skills/auk/scripts/auk.py doctor
  $AUK_HOME/.venv/bin/python ~/works/repos/metal-auk/skills/auk/scripts/auk.py env
Every check must be green. doctor distinguishes required failures from optional gaps on
purpose: a missing Orukeet is a required failure, because it backs `verify` and the clone
length rule, while a missing enhancer endpoint is only a gap. Fix required failures before
going on, and never paper over one.

Then the smoke test, which is the only real proof:
  $AUK_HOME/.venv/bin/python .../auk.py say --text "The install works on this machine." \
    --ref $AUK_HOME/assets/demo-input-audio/zero-shot-tts/ref.wav -o /tmp/auk-install-check.wav
  $AUK_HOME/.venv/bin/python .../auk.py verify --audio /tmp/auk-install-check.wav \
    --expect "The install works on this machine"
The verify line must report PASS with high word accuracy and name orukeet-r3-q8-metal as the
engine. If it names whisper instead, the Orukeet wiring is wrong even though the words match,
so fix it rather than reporting success.

Finally, report in this shape: the chip and RAM, every path you installed, the doctor
output, the smoke test line with its timing, which environment variables I should add to my
profile, and anything that failed or that you had to work around. Do not report success for
a step you did not see pass.
````

## What it needs beyond the repo

Two things are yours to decide, and the prompt stops for both rather than inventing an
answer:

- Where the install lives. The defaults put the runner and the model in `~/works/repos`
  and the verifier in `~/works/models`. Any other layout works, since every path comes from
  an environment variable.
- The Prompt Enhancer's text model. `auk pe` needs an OpenAI-compatible endpoint and key.
  Everything else, including all generation, editing, restoration and verification, runs
  fully local without one.

## Verified on a machine that was not this one

The runner was checked against the four states an install can be in, so an agent does not
have to guess what a failure means:

| state | doctor | behaviour |
|---|---|---|
| complete install | every check green, exit 0 | jobs run |
| AuK present, Orukeet absent | required failure, exit 1 | `verify` stops with the exact fix; `say --seconds` and `verify --lang zh` still work |
| weights moved elsewhere | `AUK_WEIGHTS` resolves them, no code change | jobs run |
| nothing installed | required failures naming each missing path, exit 1 | nothing runs, nothing pretends to |