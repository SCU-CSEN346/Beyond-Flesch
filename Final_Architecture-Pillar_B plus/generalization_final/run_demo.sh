#!/usr/bin/env bash
# Launches the Gradio demo (live text classification).
# Loads Qwen-2.5-7B via vLLM (~30-60 sec) then serves at http://127.0.0.1:7860.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/hf_user/hub}
mkdir -p "$HF_HUB_CACHE/.locks"

PY=${PY:-python}
$PY demo/app.py
