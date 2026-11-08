"""Resolved configuration: where the install, the weights and the helper engines live.

Nothing here is machine specific. Every path comes from an environment variable with the
layout this project uses as the default, so an install can sit anywhere and an agent can
see what a run will actually use before paying for a model load.

    AUK_HOME              install root: the mac-models release (default ~/Models/auk-current, with
                          `python`, `src` and `weights`) or an upstream clone (`.venv`, `src`, `ckpts`)
    AUK_WEIGHTS           weights directory, default $AUK_HOME/weights, else $AUK_HOME/ckpts/mlx-8bit
    AUK_ORUKEET_BIN       Orukeet CLI used by `verify` (default ~/Models/orukeet-current/bin/orukeet,
                          mac-models `./install.sh orukeet`)
    AUK_ORUKEET           only where that CLI is absent: the pip wheel's installation.json,
                          default $AUK_HOME/orukeet/installation.json
    AUK_PE_CONFIG         pe.config.yaml used by `pe`
    AUK_MLX_CACHE_LIMIT   GB cap on MLX's allocator cache, default 4
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME   OpenAI-compatible endpoint for `pe`

`auk env` prints the resolved values, marks what is missing, and exits 1 when a required
one is absent, so it works as a gate. Optional pieces (Orukeet, the enhancer's endpoint)
are reported as optional: a missing them does not fail the command, because generation
does not need them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_HOME = "~/Models/auk-current"
WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"


def _expand(value: str | None, default: str) -> str:
    return os.path.abspath(os.path.expanduser(value or default))


def install_root() -> str:
    return _expand(os.environ.get("AUK_HOME"), DEFAULT_HOME)


def weights_dir(root: str | None = None) -> str:
    root = root or install_root()
    default = os.path.join(root, "weights")  # the release layout
    if not os.path.isdir(default):
        default = os.path.join(root, "ckpts/mlx-8bit")  # an upstream clone
    return _expand(os.environ.get("AUK_WEIGHTS"), default)


def python_path(root: str | None = None) -> str:
    """The interpreter of the install: the release's `python`, else an upstream clone's venv."""
    root = root or install_root()
    release = os.path.join(root, "python")
    return release if os.path.isfile(release) else os.path.join(root, ".venv/bin/python")


def orukeet_bin() -> str:
    return _expand(os.environ.get("AUK_ORUKEET_BIN"), "~/Models/orukeet-current/bin/orukeet")


def orukeet_install() -> str:
    return _expand(os.environ.get("AUK_ORUKEET"), os.path.join(install_root(), "orukeet/installation.json"))


def orukeet_verifier() -> str | None:
    """The Orukeet engine `verify` will use: the native CLI, else the pip wheel's installation.json."""
    if os.access(orukeet_bin(), os.X_OK):
        return orukeet_bin()
    return orukeet_install() if os.path.isfile(orukeet_install()) else None


def pe_config(root: str | None = None) -> str:
    return _expand(
        os.environ.get("AUK_PE_CONFIG"), os.path.join(root or install_root(), "src/auk/infer/pe.config.yaml")
    )


def cache_limit_gb() -> float:
    return float(os.environ.get("AUK_MLX_CACHE_LIMIT", "4"))


def llm() -> dict[str, str | bool]:
    """The enhancer's endpoint, without ever echoing the key itself."""
    from auk_pe import LLM_ENV, _read_env_file

    file_env = _read_env_file(LLM_ENV)
    key = os.environ.get("LLM_API_KEY") or file_env.get("FORNACE_LLM_API_KEY", "")
    base = os.environ.get("LLM_BASE_URL") or file_env.get("FORNACE_LLM_BASE_URL", "")
    model = os.environ.get("LLM_MODEL_NAME") or "fornace-flash"
    return {
        "base_url": base,
        "key_set": bool(key),
        "model": model,
        "source": "environment" if os.environ.get("LLM_API_KEY") else ("file" if key else "unset"),
    }


def resolve() -> dict:
    """Everything a run needs, with existence checks and what to set when it is absent."""
    root = install_root()
    weights = weights_dir(root)
    python = python_path(root)
    orukeet = orukeet_verifier()
    pe = pe_config(root)
    variants = [
        name
        for name in ("config_flash.yaml", "config_base.yaml")
        if os.path.isfile(os.path.join(weights, name))
    ]
    return {
        "install_root": {"path": root, "exists": os.path.isdir(root), "required": True,
                         "set_with": "AUK_HOME"},
        "python": {"path": python, "exists": os.path.isfile(python), "required": True,
                   "set_with": "install the release (mac-models install.sh auk) or create <install_root>/.venv"},
        "weights": {"path": weights, "exists": os.path.isdir(weights), "required": True,
                    "variants": variants, "set_with": "AUK_WEIGHTS"},
        "orukeet": {"path": orukeet or orukeet_bin(), "exists": orukeet is not None, "required": True,
                    "set_with": "AUK_ORUKEET_BIN",
                    "used_by": "verify, and the clone length rule: a say without --seconds "
                               "transcribes the reference clip"},
        "pe_config": {"path": pe, "exists": os.path.isfile(pe), "required": False,
                      "set_with": "AUK_PE_CONFIG", "used_by": "pe"},
        "mlx_cache_limit_gb": {"value": cache_limit_gb(), "required": False,
                               "set_with": "AUK_MLX_CACHE_LIMIT"},
        "llm": {**llm(), "required": False, "set_with": "LLM_BASE_URL, LLM_API_KEY",
                "used_by": "pe"},
        "whisper_fallback": {"value": WHISPER_REPO, "required": False, "used_by": "verify --lang zh"},
    }


def _line(name: str, entry: dict, numeric: bool = False) -> str:
    if numeric:
        return f"  {name:22s} {entry['value']}"
    mark = "ok " if entry.get("exists") else ("MISS" if entry.get("required") else "opt ")
    detail = entry["path"]
    extra = ""
    if name == "weights" and entry.get("exists"):
        extra = f"  [{', '.join(entry['variants']) or 'no variant configs'}]"
    if not entry.get("exists") and not entry.get("required"):
        extra = f"  optional, set {entry['set_with']}"
    return f"  {mark} {name:22s} {detail}{extra}"


def report(as_json: bool) -> int:
    cfg = resolve()
    missing = [k for k, v in cfg.items() if isinstance(v, dict) and v.get("required") and not v.get("exists")]
    if as_json:
        print(json.dumps({**cfg, "ok": not missing, "missing": missing}, indent=2))
        return 1 if missing else 0

    print("resolved configuration")
    for key in ("install_root", "python", "weights", "orukeet", "pe_config"):
        print(_line(key, cfg[key]))
    print(_line("mlx_cache_limit_gb", cfg["mlx_cache_limit_gb"], numeric=True))
    llm_cfg = cfg["llm"]
    state = f"{llm_cfg['base_url']} model={llm_cfg['model']} key={'set' if llm_cfg['key_set'] else 'unset'}"
    print(f"  {'ok ' if llm_cfg['key_set'] and llm_cfg['base_url'] else 'opt '} {'llm (pe)':22s} {state}")
    print()
    print("export these to make the choices permanent:")
    print(f"  export AUK_HOME={cfg['install_root']['path']}")
    print(f"  export AUK_WEIGHTS={cfg['weights']['path']}" if os.environ.get("AUK_WEIGHTS") else
          "  # AUK_WEIGHTS defaults to $AUK_HOME/weights, else $AUK_HOME/ckpts/mlx-8bit")
    if not cfg["orukeet"]["exists"]:
        print(f"  export AUK_ORUKEET_BIN={cfg['orukeet']['path']}  # after installing Orukeet")
    if not (llm_cfg["key_set"] and llm_cfg["base_url"]):
        print("  export LLM_BASE_URL=<openai-compatible endpoint>  # for `auk pe` only")
        print("  export LLM_API_KEY=<key>")
    if missing:
        print()
        print(f"required and missing: {', '.join(missing)}. Fix those before running a job.")
    return 1 if missing else 0