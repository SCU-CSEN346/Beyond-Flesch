# Beyond-Flesch

Cross-subject generalization of transformer-based difficulty classification for educational texts.

**Team:** Beyond Flesch
**Course project — Submission 2**

---

## Project option

**Option 1** — We're reproducing a paper whose code was not publicly released: *Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts* (Rooein et al., 2024, BEA Workshop). As an extension we're also adapting the architecture of Gombert et al. (2024), whose code [is available on GitHub](https://github.com/SGombert/edutec-bea-shared-task-2024) — we build on that reference implementation rather than reimplement it. Rooein remains the primary reproduction target and the primary baseline for our results.

---

## What this is

Rooein et al. (2024) classify educational texts into three grade buckets — elementary, middle, high — using a mix of static readability metrics and prompt-based metrics derived from LLMs. Their results on ScienceQA are strong (0.95 macro-F1 with the combined pipeline), but they flag generalizability as an open problem: the same classifier might not hold up on topics or subjects it wasn't trained on, and ScienceQA is the only dataset of its kind they could evaluate on.

That's where we're going. On top of the reproduction, we're adapting the scalar-mixed transformer with rational regression heads from Gombert et al. (2024) — originally built for biomedical item difficulty — to this task, and testing how well it holds up when we hold out entire ScienceQA subjects from training. S1 covers the intro, related work, and the data + metrics pipeline. The model work starts in S2.

**Links**
- Paper (Overleaf): https://www.overleaf.com/project/69d7dc6cf65cac573d60c762
- HF Dataset: https://huggingface.co/datasets/nlpscu/Beyond-Flesch
- HF Model: In Progress

---

## Repository structure

```
logistic-regression/   Full Rooein et al. (2024) reproduction pipeline:
                       dataset preprocessing, 46 static readability metrics,
                       63 LLM-generated prompt metrics across multiple LLMs,
                       and (in S2) the final logistic-regression classifier.

transformers/          Our extension: Gombert et al. (2024) scalar-mixed
                       transformer architecture adapted to ScienceQA,
                       with LoRA (in progress).
```

Each approach has its own notebooks and outputs.

---

## Installation

```bash
git clone <repo-url>
cd Beyond-Flesch
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

We pinned the package versions from Appendix B of the Rooein paper so `textstat` and `nltk` don't drift on us between runs. Main ones: `nltk==3.8.1`, `pandas==2.2.0`, `textstat==0.7.3`, `spacy==3.7.4`, `scikit-learn==1.4.0`, `transformers==4.38.0`, `bitsandbytes==0.42.0`. Full list in `requirements.txt`.

A GPU is needed for Step 5 (LLM inference). Everything else runs on CPU. We developed in Colab — if you're running locally, the Drive-mounting cells no-op and fall back to the current directory.

---

## How to run it

Notebooks live in `logistic-regression/notebooks/` and are meant to be run in order. Each one writes to `logistic-regression/outputs/` so the next can pick up where it left off. Config constants from the paper live in `logistic-regression/outputs/config.json`.

| Step | What it does | Runtime |
|------|-------------|---------|
| 1 | Environment setup, config constants pulled from the paper | ~5 min |
| 2 | Load ScienceQA, filter image questions, bucket grades, dedupe, sample 1,516 per level, 80/20 split | ~5 min |
| 3 | Static metric extraction (46 metrics from Appendix C) | ~10 min |
| 4 | Prompt template generation (63 prompts from Appendix A) | ~1 min |
| 5 | LLM inference — runs all 63 prompts against all 4,548 texts, one notebook per model | ~10 hours per model on A100 |
| 6 | Zero-shot / few-shot Gemma-7B baselines (100-example eval subset) | ~20-40 min (GPU) |
| 7 | Multinomial logistic regression (STATIC / PROMPT / COMBO) + SelectKBest tuning | ~5-10 min |
| 8 | Evaluation: macro-F1, per-class report, 1,000-sample bootstrap significance tests | ~5-10 min |
| 9 | Feature importance analysis (univariate F-tests) + ranking tables | ~2-5 min |
| 10 | Out-of-domain generalization (OneStopEnglish, UniversalCEFR) | coming in S2 |
| 11 | Reproduce paper-style result tables + save final artifacts | ~2-5 min |

Steps 1–4 all live in `step1_to_4_pipeline.ipynb`. Step 5 is templated across four LLMs (Llama-2-7B, Llama-2-13B, Mistral-7B, Gemma-7B) — one notebook each, meant to be run in parallel Colab tabs. It's set up to be resumable: if Colab disconnects (and it will), just re-run the cell and it picks up from the last checkpoint. Progress saves every 50 texts to `outputs/prompt_metrics/{model}_progress.csv`.

---

## What the outputs look like

**After Step 2:**
- `logistic-regression/data/train.csv` — 3,638 rows, balanced across the three grade buckets
- `logistic-regression/data/test.csv` — 910 rows, same balance
- Columns: `question`, `choices`, `solution`, `lecture`, `full_text`, `text_question`, `text_solution`, `text_lecture`, `education_level`, `grade`, `subject`, `topic`, `category`

**After Step 3:**
- `logistic-regression/outputs/static_metrics/{train,test}_static_metrics.csv` — 46 feature columns per Appendix C (readability scores, POS counts, syntactic features, WordNet polysemy)
- `logistic-regression/outputs/static_metrics/static_metrics_distribution.png` — per-metric distribution plot

**After Step 4:**
- `logistic-regression/outputs/prompt_metrics/prompt_questions.json` — the 63 prompts grouped by grade level
- `logistic-regression/outputs/prompt_metrics/prompt_definitions.json` — metadata for each prompt

**After Step 5 (per LLM):**
- `logistic-regression/outputs/prompt_metrics/{train,test}_prompt_metrics_{model}.csv` — 63 prompt-metric columns, each a yes/no score from that LLM
- `logistic-regression/outputs/prompt_metrics/{model}_progress.csv` — resumable progress state

**After Modeling & Evaluation :**
- `logistic-regression/results/tables/table1_model_f1_scores.csv` — macro-F1 + per-class F1 summary for STATIC/PROMPT/COMBO
- `logistic-regression/results/tables/table2_per_class_results.csv` — detailed class-wise precision/recall/F1
- `logistic-regression/results/tables/table3_static_feature_importance.csv` — ranked static features from univariate F-tests
- `logistic-regression/results/tables/table4_prompt_feature_importance.csv` — ranked prompt features from univariate F-tests
- `logistic-regression/results/tables/table5_zeroshot_fewshot_baselines.csv` — zero-shot/few-shot Gemma-7B baseline scores
- `logistic-regression/results/models/bootstrap_significance.json` — bootstrap significance test outputs (`n=1000`, p-values, significance flags)
- `logistic-regression/results/models/full_results.json` — serialized model outputs and evaluation artifacts
- `logistic-regression/results/final_results.json` — consolidated summary of key experiment outputs
- `logistic-regression/results/figures/{fig1_confusion_matrices,fig2_model_comparison,fig3_feature_importance}.png` — main analysis figures
- `logistic-regression/results/logs/run_log.json` — run metadata and execution log

---

## What's done so far

| LLM | Status |
|-----|--------|
| Gemma-7B-IT | ✅ complete (all 4,548 texts) |
| Mistral-7B-Instruct-v0.2 | ✅ complete (all 4,548 texts)|
| Llama-2-7B-Chat | ✅ complete (all 4,548 texts) |
| Llama-2-13B-Chat |✅ complete (all 4,548 texts)|

---

## Part 2 status (Modeling, Evaluation, Analysis)

- Baselines (Gemma-7B): zero-shot macro-F1 = **0.1654**, few-shot macro-F1 = **0.1856** (`results/tables/table5_zeroshot_fewshot_baselines.csv`)
- Logistic regression + feature selection:
  - STATIC (`k=35`) macro-F1 = **0.7778**
  - PROMPT (`k=50`) macro-F1 = **0.6813**
  - COMBO (`k=70`) macro-F1 = **0.7894** (best)
- Bootstrap significance (`n=1000`):
  - COMBO vs STATIC: **+0.0116** macro-F1, **p=0.031**, significant at alpha=0.05
  - PROMPT vs STATIC: **-0.0964**, **p=1.0**, not significant
- Out-of-domain generalization:
  - **Queued for next phase** (including OneStopEnglish and UniversalCEFR evaluation)
- Final artifacts saved under `logistic-regression/results/`:
  - `tables/` (model F1s, per-class metrics, static/prompt feature rankings, baseline table)
  - `figures/` (confusion matrices, model comparison, feature importance)
  - `models/` (bootstrap significance + full serialized results)
  - `logs/` (run log)

---

## Who did what

**Shrishti — Rooein pipeline reproduction (`logistic-regression/`, Steps 1–5)**
Got the environment set up with the pinned versions from Appendix B. Built the full ScienceQA preprocessing pipeline: pulling from `derek-thomas/ScienceQA`, dropping image-based items, collapsing the 12 K-12 grades into three buckets per Section 4.1, deduplicating, sampling 1,516 per level with seed 42 to get the 4,548-text balanced subset, and doing the 80/20 stratified split. Implemented the 46 static metrics from Appendix C and generated the 63 prompt templates from Appendix A. Ran Step 5 for Gemma-7B-IT, Mistral-7B-Instruct-v0.2, and Llama-2-7B-Chat (all 4,548 texts each), with 8-bit quantization and resumable progress saving. Smruthi ran Llama-2-13B-Chat. All four LLM inference runs from the paper are now complete.

**Smruthi — Gombert transformer adaptation (`transformers/`)**
Identified that the allennlp and rational_activations packages are incompatible with modern PyTorch versions and cannot be installed. To resolve this, extracted the ScalarMix module directly from the AllenNLP repository and refactored it into a standalone, dependency-free module for our project. Also adapted the codebase from Gombert et al., which was originally designed for USMLE-style regression tasks, to work with the ScienceQA dataset. This involved writing data parsing and preprocessing code for ScienceQA including grade-level labeling and class balancing, converting the model from regression to classification by replacing MSE loss with cross-entropy loss, simplifying the architecture from two regression heads with two ScalarMix modules down to a single classification head with one ScalarMix, and replacing the Rational activation function with ReLU since the original Rational implementation was unavailable.

**Maneesha — Modeling, Evaluation & Analysis (`logistic-regression/`, Steps 6–11)**
Built the full modeling and evaluation pipeline on top of the Step 1-5 feature matrices. Implemented multinomial logistic regression with `SelectKBest` feature selection across three configurations (STATIC/PROMPT/COMBO), ran macro-F1 and per-class evaluation, and added 1,000-sample bootstrap significance testing. Results show COMBO improves over STATIC by +0.0116 macro-F1 and is statistically significant (p=0.031), while PROMPT-only underperforms. Added zero-shot/few-shot Gemma-7B baselines and performed univariate F-test feature importance analysis. Out-of-domain generalization experiments (including OneStopEnglish) are queued for the next phase. Packaged final results into reproducible tables, figures, model JSON outputs, and logs under `logistic-regression/results/`.

---

## References

- Rooein, D., Röttger, P., Shaitarova, A., & Hovy, D. (2024). *Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts.* BEA Workshop, ACL 2024.
- Gombert, S., Menzel, L., Di Mitri, D., & Drachsler, H. (2024). *Predicting Item Difficulty and Item Response Time with Scalar-mixed Transformer Encoder Models and Rational Network Regression Heads.* BEA Workshop, ACL 2024. [Code on GitHub](https://github.com/SGombert/edutec-bea-shared-task-2024)

## License

TBD — leaning toward Apache 2.0 to match the Gombert reference code.
