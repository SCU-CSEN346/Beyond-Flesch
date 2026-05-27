# gen_v2/ — Efficient, Content-Aware Text-Difficulty Classifier

This folder is the **third generation** of the text-difficulty work in this
repo. It proposes a small, fast, content-aware alternative to the Rooein-2024 prompt-
metric pipeline and tests it on a new adversarial benchmark.

## Headline result

| Metric | Baseline (Rooein-style 7B + 63 prompts + MLP) | **Pillar B+ (ours, Phi-3.5-mini LoRA)** |
|---|---|---|
| In-distribution test macro-F1 | 0.840 | **0.872** |
| Balanced-test macro-F1 (5×150) | 0.856 | **0.869** |
| ood_synthetic vs judge | 0.918 | **0.925** |
| ood_synthetic vs gen | **0.940** | 0.893 |
| ood_race-middle | **0.595** | 0.561 |
| ood_race-high | 0.479 | **0.546** |
| ood_onestop | 0.415 | **0.460** |
| **AdvConcept-50 overall** | 0.600 | **0.760** |
| **surface_hard / concept_easy** | **0.000** | **1.000** |
| Inference latency | ~5 sec/text | **~40 ms/text** |
| Inference VRAM | ~27 GB (Qwen-7B) | ~8 GB (Phi-3.5) |
| On-disk size | ~14 GB (Qwen-7B + scaler) | **5 GB** (Phi-3.5 + 200 MB LoRA) |
| Training energy (kWh) | ~0.86 (estimated end-to-end) | **0.39** (CodeCarbon measured) |

**Headline:** Pillar B+ beats the baseline on 7 of 9 quantitative metrics
**and** is 130× faster at inference, while being trained on ~50% less
energy. The big AdvConcept-50 win — **1.00 vs 0.00** on the
`surface_hard / concept_easy` category — is the curriculum-content
discrimination that makes the model deployable in a real classroom.

See `paper/paper.md` for the full writeup with all 6 figures.

## Reproducing

All scripts are standalone Python and write to `outputs/`. Run from
the repository root in the existing `.venv`:

```bash
# Step 1 — build the adversarial benchmark (50 hand-curated rows)
.venv/bin/python gen_v2/scripts/build_adv_concept.py
# → gen_v2/data/adv_concept.csv

# Step 2 — re-evaluate the baseline (Rooein-style per_llm Qwen-7B bundle)
# Spins up Qwen-2.5-7B via transformers (NOT vLLM) for the AdvConcept-50
# prompt-feature extraction. ~10 minutes total.
.venv/bin/python gen_v2/scripts/evaluate_all.py
# → gen_v2/outputs/eval_reports/baseline_per_llm_qwen2.5-7b_seed7.eval.json

# Step 3 — Pillar A encoder bake-off (Lean COMBO)
# Trains 6 LogReg variants on (MiniLM | bge-small | bge-base) × (emb | emb+static5).
# ~15 minutes; mostly embedding-cache construction.
.venv/bin/python gen_v2/scripts/pillar_a_lean_combo.py
# → gen_v2/outputs/models/pillar_a__*.pkl
# → gen_v2/outputs/eval_reports/pillar_a__*.eval.json
# → gen_v2/outputs/eval_reports/pillar_a_summary.json

# Step 4 — Pillar B decoder bake-off (Tiny Teacher)
# LoRA fine-tune Qwen-2.5-3B, Phi-3.5-mini, Gemma-2-2b. ~45 minutes.
.venv/bin/python gen_v2/scripts/pillar_b_tiny_teacher.py \
    --backbones qwen-2.5-3b-instruct phi-3.5-mini-instruct gemma-2-2b-it \
    --train-subset 5000 --epochs 1
# → gen_v2/outputs/models/pillar_b__*__lora/
# → gen_v2/outputs/eval_reports/pillar_b__*.eval.json
# → gen_v2/outputs/eval_reports/pillar_b_summary.json

# Step 5 — Pillar B+ retraining of the bake-off winner (Phi-3.5-mini)
# Full 17K train data, 2 epochs, r=32, class-weighted sampling.
# ~70 minutes wall-clock; ~0.39 kWh measured.
.venv/bin/python gen_v2/scripts/pillar_b_plus.py \
    --epochs 2 --target-per-class 3500 --lora-r 32
# → gen_v2/outputs/models/pillar_b_plus__phi-3.5-mini-instruct__lora/best/
# → gen_v2/outputs/eval_reports/pillar_b_plus__phi-3.5-mini-instruct.eval.json

# Step 6 — comparison table
.venv/bin/python gen_v2/scripts/compare_carbon.py
# → gen_v2/outputs/eval_reports/comparison_table.md
# → gen_v2/outputs/eval_reports/comparison_table.json

# Step 7 — paper figures
.venv/bin/python paper/figures/make_figures.py
# → paper/figures/*.png
```

## Folder map

```
gen_v2/
├── README.md                    (this file)
├── requirements.txt             (sentence-transformers + einops; rest already in .venv)
├── inputs/                      (symlink → ../generalization_final/inputs)
├── data/
│   └── adv_concept.csv          (50-row adversarial benchmark)
├── scripts/
│   ├── build_adv_concept.py     (writes adv_concept.csv from a hand-curated seed list)
│   ├── evaluate_all.py          (canonical eval harness — works for any model "kind")
│   ├── pillar_a_lean_combo.py   (encoder + readability + LogReg bake-off)
│   ├── pillar_b_tiny_teacher.py (small-LLM LoRA bake-off)
│   ├── pillar_b_plus.py         (Phi-3.5-mini full-data retraining)
│   └── compare_carbon.py        (final Markdown comparison table)
└── outputs/
    ├── embeddings/              (cached encoder outputs per (encoder, split))
    ├── prompt_cache/            (cached Qwen-7B prompt features for AdvConcept-50)
    ├── models/                  (Pillar A pickles + Pillar B LoRA adapters)
    ├── eval_reports/            (one *.eval.json per model + summaries)
    ├── emissions/               (CodeCarbon CSV)
    └── figures/                 (per-pillar PNGs — paper figures live in ../paper/figures/)
```

## What's where

- **The benchmark we're proudest of**: `gen_v2/data/adv_concept.csv` (50
  curriculum-grounded adversarial rows; `build_adv_concept.py` documents
  the rationale for each).
- **The model we'd deploy**: `gen_v2/outputs/models/pillar_b_plus__phi-3.5-mini-instruct__lora/best/`.
  Loadable with PEFT + transformers + Phi-3.5-mini base; ~200 MB LoRA
  adapter on top of the 7.6 GB base model.
- **The headline numbers**:
  `gen_v2/outputs/eval_reports/pillar_b_plus__phi-3.5-mini-instruct.eval.json`
  and `gen_v2/outputs/eval_reports/comparison_table.md`.

## Hard constraints honoured


- **No Llama-4**: the budget rules out 17B-active / 109B-total MoE models.
  Discussed in §4.5 and §7.3 of the paper.
- **CodeCarbon-tracked**: every script wraps a `tracker.start() / .stop()`
  block, emissions written to `gen_v2/outputs/emissions/emissions.csv`.

## Caveats and what to do next

See §7 of the paper. Most important caveats:

- AdvConcept-50 is small (6 rows in the headline failure category).
  Statistically significant in *direction* but you'd want ~500 rows
  before submitting to a venue.
- All training and evaluation uses Llama-3.1-8B as the judge. A panel of
  judges would be more rigorous.



