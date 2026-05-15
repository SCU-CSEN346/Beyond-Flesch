# LLM-as-a-Judge: Cross-Corpus Readability Classification

This folder implements the Option E plan: use a strong local LLM (Llama-3.1-8B)
to assign unified labels across all corpora, then train a COMBO classifier
(static features + prompt features from 4 LLMs) and evaluate on a truly
held-out OOD corpus (OneStopEnglish).

## Scripts (run in this order)

| # | Script | What it does |
|---|--------|--------------|
| 1 | `step3d_static_features.py` | Compute ~40 static features per text. |
| 2 | `step3e_split_holdout.py`   | Pull OneStop out of train/val/test into a new `ood_onestop` split. |
| 3 | `step3f_llm_judge.py`       | Llama-3.1-8B labels every text uniformly. |
| 4 | `step3g_label_disagreement.py` | Per-corpus original-vs-judge agreement; Fig 1. |
| 5 | `step8l_combo_classifier.py` | Train COMBO MLP, both per-LLM and all-LLMs configs, both label sources. |
| 6 | `step9f_calibration_sweep.py` | K-shot per-corpus calibration; Fig 3. |
| 7 | `plot_results.py`            | Figs 1-4 + Table 1 (CSV). |

Or just run `bash run.sh` to do all in order.

## Feature LLMs vs Judge LLM

To avoid circularity:

- **Judge**: Llama-3.1-8B-Instruct
- **Feature LLMs** (for the 63 prompt features each): Qwen-2.5-7B, Mistral-7B,
  Gemma-7B, Llama-2-7B (4 LLMs, matching Rooein et al. 2024's count exactly).

Prompt features for the 4 feature-LLMs already exist under
`../generalization_pipeline/outputs/prompt_metrics/`. We only run new LLM
work for the judge.

## Outputs

```
outputs/
├── emissions/emissions.csv          <- CodeCarbon log (every script appends)
├── llm_judge/<split>_judge.csv
├── static_features/<split>_static.csv
├── splits/{train,val,test,ood_onestop}.csv
├── models/<config>__seed<N>.pkl
├── eval_reports/*.json
└── figures/{fig1..fig4}.png
```

## Expected outcome

| Setting | Expected macro-F1 on held-out OneStop |
|---------|----------------------------------------|
| Original (corpus) labels                  | ~0.21 |
| + LLM-judge unified labels                | ~0.65–0.75 |
| + 50-shot per-corpus calibration on judge | ~0.78–0.85 |
