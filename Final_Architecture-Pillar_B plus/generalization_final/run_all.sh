#!/usr/bin/env bash
# Full pipeline from raw curated dataset to trained models + figures.
# ~4-6 hours GPU end-to-end (mostly step3f LLM-judging + step5d prompt extraction).
set -euo pipefail
cd "$(dirname "$0")"

# Use a user-writable HF cache. Skip if your default cache is fine.
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HOME/.cache/hf_user/hub}
mkdir -p "$HF_HUB_CACHE/.locks"

PY=${PY:-python}

echo "[run] (1/8) Pull OneStop out of train/val/test -> outputs/splits/"
$PY scripts/step3e_split_holdout.py

echo "[run] (2/8) Static features for all 6 splits"
$PY scripts/step3d_static_features.py \
    --input-dir outputs/splits \
    --splits train val test ood_onestop ood_race-middle ood_race-high

echo "[run] (3/8) Llama-3.1-8B applies unified rubric to every row (~30-60 min)"
$PY scripts/step3f_llm_judge.py

echo "[run] (4/8) Per-corpus original-vs-judge diagnostic"
$PY scripts/step3g_label_disagreement.py

echo "[run] (5/8) Build share-ready clean dataset"
$PY scripts/step3h_build_clean_dataset.py

echo "[run] (6/8) Extract 63 prompt features per LLM (4 LLMs, ~30 min each)"
for LLM in qwen2.5-7b mistral-7b gemma-7b llama2-7b; do
  $PY scripts/step5d_inference_generalization.py \
      --model "$LLM" \
      --splits train val test ood_onestop ood_race-middle ood_race-high
done

echo "[run] (7/8) Train COMBO classifier (4 sweep cells)"
$PY scripts/step8l_combo_classifier.py --config all_llms --label-source llm_judge
$PY scripts/step8l_combo_classifier.py --config all_llms --label-source original
$PY scripts/step8l_combo_classifier.py --config per_llm  --label-source llm_judge
$PY scripts/step8l_combo_classifier.py --config per_llm  --label-source original
# Gombert multi-task head variant (the 0.940 result):
$PY scripts/step8l_combo_classifier.py --config per_llm --label-source llm_judge \
    --llms qwen2.5-7b --use-grade-reg

echo "[run] (8/8) Per-corpus calibration sweep + figures"
$PY scripts/step9f_calibration_sweep.py
$PY scripts/plot_results.py

echo "[run] DONE. See outputs/{models, eval_reports, figures, clean_dataset}."
