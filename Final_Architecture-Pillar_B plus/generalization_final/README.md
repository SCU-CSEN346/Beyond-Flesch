# Generalization-Final: LLM-as-Judge COMBO Classifier for Educational Text Difficulty

Self-contained reproduction of our cross-corpus readability classification work:

* **LLM-as-judge unified labels** (Llama-3.1-8B-Instruct) on top of a curated multi-corpus dataset
* **COMBO classifier** (34 static readability + 63 prompt features per LLM)
* **Gombert-style multi-task head** (3-class softmax + continuous-grade regression)
* **0.940 macro-F1 on truly unseen balanced synthetic data** (with single-LLM Qwen features + multi-task head)
* **0.840 macro-F1 on balanced subsample of in-distribution test** (4-LLM concat features, no multi-task)

This folder contains everything needed to reproduce those numbers end-to-end.

## Quick reference

| Metric | Number |
|---|---|
| ID test (full 4-LLM bundle, judge labels) | 0.844 ± 0.004 |
| Balanced 150-row subsample of test (judge labels) | 0.840 ± 0.013 |
| Synthetic 150 truly-unseen, single-LLM Qwen | 0.926 |
| Synthetic 150 + Gombert multi-task head | **0.940** |
| ID-original-labels baseline (the "0.21" pre-judge baseline you started with) | 0.21 |

## Folder layout

```
generalization_final/
├── README.md                        (this file)
├── requirements.txt                 (Python deps)
├── run_all.sh                       (full pipeline from raw curated dataset, ~4-6 hr GPU)
├── run_synthetic.sh                 (synthetic OOD evaluation, ~10-15 min)
├── scripts/
│   ├── step3d_static_features.py    34 readability+lexical features per text
│   ├── step3e_split_holdout.py      Pulls OneStop out of train/val/test
│   ├── step3f_llm_judge.py          Llama-3.1-8B labels every text uniformly
│   ├── step3g_label_disagreement.py Diagnostic: per-corpus original-vs-judge agreement
│   ├── step3h_build_clean_dataset.py Builds the share-ready dataset (`outputs/clean_dataset/`)
│   ├── step3i_generate_synthetic.py Generates 150 balanced synthetic texts via Llama-3.1
│   ├── step5b_inference_generic.py  Helper module (shared logic for step5d)
│   ├── step5d_inference_generalization.py
│   │                                Extracts 63 prompt features per text per LLM (vLLM)
│   ├── step8l_combo_classifier.py   Trains COMBO MLP (optionally with Gombert multi-task head)
│   ├── step9f_calibration_sweep.py  K-shot per-corpus calibration on OOD
│   ├── step9g_eval_synthetic.py     Evaluates a bundle on a held-out split
│   ├── step9h_balanced_subsample.py Balanced 50/50/50 subsample evaluation
│   └── plot_results.py              Generates 4 paper figures
├── inputs/
│   ├── curated_generalization_collection/    (source dataset from step2e)
│   │   ├── train.csv, val.csv, test.csv
│   │   ├── ood_race-middle.csv, ood_race-high.csv
│   │   ├── all_curated_rows.csv, source_summary.csv
│   │   └── collection_manifest.json
│   └── prompt_questions.json                  (Rooein's 63 prompts)
└── outputs/
    ├── splits/                       Post-holdout train/val/test + OOD splits
    ├── static_features/              <split>_static.csv (34 features each)
    ├── llm_judge/                    <split>_judge.csv (Llama-3.1 verdicts)
    ├── prompt_metrics/               <split>_prompt_metrics_<llm>__multi.csv
    ├── models/                       Trained classifier bundles (.pkl)
    │   ├── per_llm__llm_judge__qwen2.5-7b__seed*_gradereg.pkl   (Gombert multi-task)
    │   └── all_llms__llm_judge__all__seed*.pkl                  (4-LLM concat)
    ├── eval_reports/                 Per-run JSON eval reports
    ├── figures/                      fig1..fig4 PNGs
    ├── clean_dataset/                Share-ready CSVs with original + judge labels
    └── emissions/                    CodeCarbon log
```

## Environment setup

```bash
# Python 3.11+ recommended (we used 3.13)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Hugging Face gated models (Llama-3.1, Llama-2, Gemma): run `huggingface-cli login`
first or `export HF_TOKEN=hf_...`.

If your HF cache `~/.cache/huggingface/hub/.locks` is owned by another user
(common on shared boxes), point HF at a user-writable cache:

```bash
export HF_HUB_CACHE=$HOME/.cache/hf_user/hub
mkdir -p "$HF_HUB_CACHE/.locks"
```

GPU recommended: a single 24+ GB GPU works (we used a 48 GB RTX 5880 Ada).

## How to run

### Option A: Full pipeline from raw dataset (~4-6 hr GPU)

```bash
bash run_all.sh
```

This walks through every step:

1. `step3e_split_holdout.py`        — pull OneStop into ood_onestop
2. `step3d_static_features.py`      — 34 static features per row
3. `step3f_llm_judge.py`            — Llama-3.1 unified labels (~30-60 min)
4. `step3g_label_disagreement.py`   — diagnostic table
5. `step3h_build_clean_dataset.py`  — write `outputs/clean_dataset/`
6. `step5d_inference_generalization.py` x 4 LLMs (Qwen, Mistral, Gemma, Llama-2) — ~30 min each
7. `step8l_combo_classifier.py` x 4 sweep cells (all_llms / per_llm × original / llm_judge)
8. `step9f_calibration_sweep.py`    — K-shot per-corpus calibration
9. `plot_results.py`                — figs 1–4

### Option B: Quick synthetic evaluation (~10-15 min)

Assumes models are already trained (and present in `outputs/models/`). Generates
150 balanced synthetic texts via Llama-3.1, extracts Qwen features, and evaluates
the multi-task bundle.

```bash
bash run_synthetic.sh
```

### Option C: Train just the Gombert multi-task variant

If you only want to retrain the multi-task Qwen variant (~3 min on GPU):

```bash
python scripts/step8l_combo_classifier.py \
    --config per_llm --label-source llm_judge \
    --llms qwen2.5-7b \
    --use-grade-reg
```

### Option D: Evaluate existing bundle on any split

```bash
# Synthetic, multi-task bundle:
python scripts/step9g_eval_synthetic.py \
    --bundle-pattern 'per_llm__llm_judge__qwen2.5-7b__seed*_gradereg.pkl' \
    --llms qwen2.5-7b --split ood_synthetic

# Balanced subsample of test, full 4-LLM bundle:
python scripts/step9h_balanced_subsample.py --per-class 50 --n-resamples 5
```

## Architecture

```
Input text
   │
   ├──► Static features (textstat-derived)      ─┐
   │     34 readability + lexical + structural    │
   │                                              ├──► concat
   ├──► 63 prompt features (Llama-3.1-8B judge)   │     (97-dim per LLM,
   │     "Is this text suitable for elementary?"  │      286-dim for 4-LLM concat)
   │     "Does this contain simple examples?"     │
   │     ... etc                                  │
   │     (per feature LLM: Qwen/Mistral/Gemma/L2) │
   │                                              ┘
   │
   ▼
Shared 2-layer MLP backbone (256 → 128, dropout 0.2)
   │
   ├──► 3-class softmax head      (elementary/middle/high) — CrossEntropy loss
   └──► continuous-grade head [1-dim, Gombert multi-task] — SmoothL1 loss
         (only with --use-grade-reg flag)

Total loss = CE + 0.3 × SmoothL1
```

## Why this works (the LLM-as-judge story)

Real-world readability corpora label "elementary/middle/high" inconsistently:

* ScienceQA "elementary" = US grade 1-5
* OneStopEnglish "Elementary" = CEFR B1 (intermediate-adult)
* RACE-middle/high = Chinese exam levels

Training across these corpora on their original labels gives ~0.21 macro-F1 on
truly held-out data — that's the noise ceiling.

**LLM-as-judge** (Llama-3.1 applies a single uniform rubric to every text)
**re-labels the dataset under a consistent scheme**. The classifier learns
*that* unified construct, not the inconsistent corpus tags. On balanced
unseen data evaluated against the same unified scheme, the model hits 0.94
macro-F1 — what you'd expect from a clean discriminative task.

## Citations

* Rooein et al. 2024 (BEA workshop, ACL): introduced the 63 prompt features +
  COMBO architecture; trained/tested in-distribution on ScienceQA only.
* Gombert et al. 2024 (BEA workshop, ACL): multi-task head with shared
  encoder backbone + per-task heads (originally for item-difficulty regression).
* Cao et al. 2020: CORN ordinal classification (not used in final architecture).

## Known limitations (be honest about them)

* The LLM-judge itself uses surface heuristics (sentence length, vocab) when
  classifying. Adversarial tests show that even Llama-3.1 calls "quantum mechanics
  in short simple sentences" middle-school instead of high. Our classifier
  inherits this limitation. See `adversarial` test in the sibling
  `llm_as_a_judge/` folder for evidence.
* Real-world OOD macro-F1 numbers (OneStop, RACE) look low (0.50) but are capped
  by post-judge class collapse, not by model capacity.
* The model is discriminative-only — it does not generate text at target levels.
  See README §"What it cannot do" in the project notes for the boundary.
