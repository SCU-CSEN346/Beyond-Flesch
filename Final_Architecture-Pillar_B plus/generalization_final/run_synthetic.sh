#!/usr/bin/env bash
# Synthetic OOD evaluation: generate 150 balanced texts via Llama-3.1, judge them,
# extract Qwen features, evaluate against the existing multi-task bundle.
# Assumes models in outputs/models/ already exist (run run_all.sh first if not).
# Wall-clock: ~10-15 min.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/hf_user/hub}
mkdir -p "$HF_HUB_CACHE/.locks"

PY=${PY:-python}

echo "[run] (1/4) Generate 150 synthetic texts (balanced 50/50/50) + judge"
$PY scripts/step3i_generate_synthetic.py

echo "[run] (2/4) Static features"
$PY scripts/step3d_static_features.py --splits ood_synthetic --input-dir outputs/splits

echo "[run] (3/4) Qwen prompt features"
$PY scripts/step5d_inference_generalization.py \
    --model qwen2.5-7b \
    --splits ood_synthetic \
    --multi-dir "$(pwd)/outputs/splits"

echo "[run] (4/4) Evaluate Gombert multi-task bundle on synthetic"
$PY scripts/step9g_eval_synthetic.py \
    --bundle-pattern 'per_llm__llm_judge__qwen2.5-7b__seed*_gradereg.pkl' \
    --llms qwen2.5-7b \
    --split ood_synthetic

echo "[run] DONE. See outputs/eval_reports/ood_synthetic_results.json"
