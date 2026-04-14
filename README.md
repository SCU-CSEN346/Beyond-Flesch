# Beyond-Flesch

Cross-subject generalization of transformer-based difficulty classification for educational texts.

**Team:** Beyond Flesch
**Course project — Submission 1**

---

## Project option

**Option 1** — We're reproducing a paper whose code was not publicly released: *Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts* (Rooein et al., 2024, BEA Workshop). As an extension we're also adapting the architecture of Gombert et al. (2024), whose code [is available on GitHub](https://github.com/SGombert/edutec-bea-shared-task-2024) — we build on that reference implementation rather than reimplement it. Rooein remains the primary reproduction target and the primary baseline for our results.

---

## What this is

Rooein et al. (2024) classify educational texts into three grade buckets — elementary, middle, high — using a mix of static readability metrics and prompt-based metrics derived from LLMs. Their results on ScienceQA are strong (0.95 macro-F1 with the combined pipeline), but they flag generalizability as an open problem: the same classifier might not hold up on topics or subjects it wasn't trained on, and ScienceQA is the only dataset of its kind they could evaluate on.

That's where we're going. On top of the reproduction, we're adapting the scalar-mixed transformer with rational regression heads from Gombert et al. (2024) — originally built for biomedical item difficulty — to this task, and testing how well it holds up when we hold out entire ScienceQA subjects from training. S1 covers the intro, related work, and the data + metrics pipeline. The model work starts in S2.

**Links**
- Paper (Overleaf): 
- HF Dataset: 
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
| 6+ | Regression training, evaluation, cross-subject experiments | *coming in S2–S3* |

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

---

## What's done so far (S1)

| LLM | Status |
|-----|--------|
| Gemma-7B-IT | ✅ complete (all 4,548 texts) |
| Mistral-7B-Instruct-v0.2 | 🟡 ~50% (2,250 / 4,548) — resuming in S2 |
| Llama-2-7B-Chat | queued for S2 |
| Llama-2-13B-Chat | queued for S2 |

---

## Who did what

**Shrishti — Rooein pipeline reproduction (`logistic-regression/`, Steps 1–5)**
Got the environment set up with the pinned versions from Appendix B. Built the full ScienceQA preprocessing pipeline: pulling from `derek-thomas/ScienceQA`, dropping image-based items, collapsing the 12 K-12 grades into three buckets per Section 4.1, deduplicating, sampling 1,516 per level with seed 42 to get the 4,548-text balanced subset, and doing the 80/20 stratified split. Implemented the 46 static metrics from Appendix C and generated the 63 prompt templates from Appendix A. Ran Step 5 end-to-end for Gemma-7B-IT (all 4,548 texts) and got Mistral-7B-Instruct to ~50% before the S1 cutoff, both with 8-bit quantization and resumable progress saving. Remaining two LLM runs (Llama-2-7B, Llama-2-13B) queued for S2.

**[Smruthi] — Gombert transformer adaptation (`transformers/`)**
[TODO: fill in S1 contribution]**

**[Maneesha] — [TODO: fill in S1 contribution]**

---

## References

- Rooein, D., Röttger, P., Shaitarova, A., & Hovy, D. (2024). *Beyond Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of Educational Texts.* BEA Workshop, ACL 2024.
- Gombert, S., Menzel, L., Di Mitri, D., & Drachsler, H. (2024). *Predicting Item Difficulty and Item Response Time with Scalar-mixed Transformer Encoder Models and Rational Network Regression Heads.* BEA Workshop, ACL 2024. [Code on GitHub](https://github.com/SGombert/edutec-bea-shared-task-2024)

## License

TBD — leaning toward Apache 2.0 to match the Gombert reference code.