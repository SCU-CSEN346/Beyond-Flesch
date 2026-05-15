#!/usr/bin/env bash
# Sequential runner for the LLM-as-a-Judge pipeline.
# Run from anywhere; this script cd's to its own folder first.

set -euo pipefail
cd "$(dirname "$0")"

PY=/home/tanmayv/my_repos/text_difficulty_classifier/.venv/bin/python

echo "[run] (1/7) Pull OneStop out of train/val/test -> outputs/splits/"
$PY scripts/step3e_split_holdout.py

echo "[run] (2/7) Static features for all 6 splits (~5 min CPU)"
$PY scripts/step3d_static_features.py \
    --input-dir outputs/splits \
    --splits train val test ood_onestop ood_race-middle ood_race-high

echo "[run] (3/7) LLM-as-judge with Llama-3.1-8B (~30-90 min GPU)"
$PY scripts/step3f_llm_judge.py

echo "[run] (4/8) Diagnose label disagreement -> Fig 1 data"
$PY scripts/step3g_label_disagreement.py

echo "[run] (5/8) Build share-ready clean dataset (original + judge labels)"
$PY scripts/step3h_build_clean_dataset.py

echo "[run] (6/8) Train COMBO classifier in 4 sweep cells (~3-5 hr GPU):"
$PY scripts/step8l_combo_classifier.py --config all_llms --label-source llm_judge
$PY scripts/step8l_combo_classifier.py --config all_llms --label-source original
$PY scripts/step8l_combo_classifier.py --config per_llm  --label-source llm_judge
$PY scripts/step8l_combo_classifier.py --config per_llm  --label-source original

echo "[run] (7/8) K-shot per-corpus calibration"
$PY scripts/step9f_calibration_sweep.py

echo "[run] (8/8) Generate Figs 1-4"
$PY scripts/plot_results.py

echo "[run] DONE. See outputs/figures/, outputs/eval_reports/, outputs/clean_dataset/."
