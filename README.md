# Beyond-Flesch

Cross-corpus, concept-aware K-12 text difficulty classification for educational
texts.

**Team:** Beyond Flesch

**Project path:** ACL paper reproduction and extension without released code

**Final submission status:** Rooein reproduction, generalization tests using transformers, DANNs, synthetic dataset generation, LLM-as-a-judge relabeling, and final Pillar B+ model are included.

## Project Overview

This project starts from Rooein et al. (2024), *Beyond Flesch-Kincaid:
Prompt-based Metrics Improve Difficulty Classification of Educational Texts*.
The paper classifies educational text into three curriculum levels:

- `elementary`
- `middle`
- `high`

Rooein et al. combine static readability features with prompt-based LLM features:
instead of asking an LLM to directly classify a passage, they ask many yes/no
questions about the passage and use the answers as features for a classifier.
Their code was not released, so the first part of this repository is a
from-scratch reproduction of that pipeline.

Our final research question goes beyond reproduction:

> Can we build a K-12 text-difficulty classifier that generalizes across
> corpora, distinguishes curriculum concept difficulty from surface readability,
> and is cheap enough to deploy?

The final answer in this repo is **Pillar B+**: a Phi-3.5-mini LoRA model that
predicts curriculum level directly in one forward pass. It is faster and cheaper
than the prompt-metric pipeline, and it fixes the main failure mode we found:
surface-readable text with hard concepts, and wordy text with easy concepts.

## Links

- Paper draft: https://www.overleaf.com/project/69d7dc6cf65cac573d60c762
- Hugging Face dataset: https://huggingface.co/datasets/nlpscu/Beyond-Flesch
- Primary reproduction target: Rooein et al. (2024), BEA Workshop
- Architecture inspiration: Gombert et al. (2024), BEA Workshop

## Headline Results

### Final Model: Pillar B+

The final model lives in `Final_Architecture-Pillar_B plus/`.

| Metric | Rooein-style baseline | Pillar B+ |
|---|---:|---:|
| In-distribution test macro-F1 | 0.840 | **0.872** |
| Balanced-test macro-F1 | 0.856 | **0.869** |
| OOD OneStop macro-F1 | 0.415 | **0.460** |
| OOD RACE-high macro-F1 | 0.479 | **0.546** |
| AdvConcept-50 overall score | 0.600 | **0.760** |
| Surface-hard / concept-easy AdvConcept score | 0.000 | **1.000** |
| Median inference latency | about 5,000 ms/text | **about 39 ms/text** |
| Training energy | about 2.29 kWh estimated | **0.387 kWh measured** |

Pillar B+ is a LoRA-fine-tuned `microsoft/Phi-3.5-mini-instruct` model:

- 3.8B frozen base parameters
- about 50M trainable LoRA parameters
- LoRA rank `r=32`, alpha `64`, dropout `0.05`
- trained for 2 epochs on 16,849 multi-corpus rows
- class-weighted sampling to reduce majority-class bias
- measured training wall time: about 66 minutes on one GPU

The key result is not just the macro-F1 gain. The important finding is that the
final model handles cases where surface readability and curriculum concept level
disagree.

### Rooein Reproduction: ScienceQA

The original reproduction and feature pipeline live in `logistic-regression/`.
The best ScienceQA COMBO result uses Mistral-7B prompt metrics.

| Prompt metric source | STATIC macro-F1 | PROMPT macro-F1 | COMBO macro-F1 |
|---|---:|---:|---:|
| Gemma-7B | 0.8005 | 0.6836 | 0.8290 |
| Llama-2-7B | 0.8005 | 0.7045 | 0.8368 |
| Mistral-7B | 0.8005 | 0.7431 | **0.8416** |
| Llama-2-13B | 0.8005 | 0.7595 | 0.8405 |
| Qwen2.5-7B | 0.8005 | **0.7634** | 0.8394 |

For the Mistral run, COMBO vs STATIC was significant in the saved bootstrap test:
`p = 0.031`, improvement `+0.0412` macro-F1.

### Frozen OOD Transfer vs In-domain OOD Training

The ScienceQA-trained Rooein-style model does not transfer cleanly to
OneStopEnglish:

| OneStopEnglish setting | STATIC macro-F1 | PROMPT macro-F1 | COMBO macro-F1 |
|---|---:|---:|---:|
| Frozen ScienceQA model | 0.1802 | **0.2496** | 0.1707 |
| Retrained on OneStop split | 0.9130 | 0.7829 | **0.9649** |

This is one of the main motivations for the final project direction: in-domain
scores can look strong, but label conventions and corpus style can dominate
cross-corpus transfer.

### LLM-as-a-Judge Generalization Pipeline

The `llm_as_a_judge/` folder builds a cleaner multi-corpus experiment. It keeps
both the original corpus labels and a unified Llama-3.1-8B judge label.

| Label source | Test macro-F1 | OneStop OOD | RACE-middle OOD | RACE-high OOD |
|---|---:|---:|---:|---:|
| Original corpus labels | 0.8374 | 0.2923 | 0.1912 | 0.1815 |
| LLM-judge labels | **0.8491** | **0.5246** | **0.6643** | **0.4959** |

The judge labels are not perfect, but they make cross-corpus labels more
consistent. The saved clean dataset contains 22,612 rows and is available under
`llm_as_a_judge/outputs/clean_dataset/`.

### Other Experimental Branches

- `transformers/` contains the Gombert-inspired transformer experiments,
  including ELECTRA + ScalarMix and hierarchical classifiers. Saved
  in-domain results reach about 0.89 macro-F1.
- `DomainAdversialNN/` contains domain-adversarial ELECTRA experiments.
  The saved result CSV reports macro-F1 of 0.8255 on CoQA OOD and 0.8797 on
  CommonLit OOD.
- `synthetic-data/` contains synthetic QA generation and train-on-synthetic
  experiments. These were diagnostic: synthetic training often learned style
  shortcuts and did not solve generalization by itself.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- logistic-regression/
|   |-- notebooks/              # Rooein reproduction and Step*.ipynb pipelines
|   |-- data/                   # ScienceQA train/test split
|   |-- outputs/                # static metrics and prompt metrics
|   |-- results/                # in-domain model results and artifacts
|   `-- ood/                    # OneStopEnglish OOD prep/evaluation
|-- llm_as_a_judge/
|   |-- scripts/                # judge labeling, features, COMBO classifier
|   `-- outputs/                # clean dataset, eval reports, figures, models
|-- Final_Architecture-Pillar_B plus/
|   |-- scripts/                # Pillar A, Pillar B, Pillar B+ experiments
|   |-- data/                   # AdvConcept-50 and probe data
|   |-- outputs/                # final eval reports, figures, emissions
|   `-- demo/                   # Gradio demo for the final model
|-- transformers/               # Gombert-style transformer adaptation experiments
|-- DomainAdversialNN/          # domain-adversarial experiments
`-- synthetic-data/             # synthetic data generation and evaluation
```

## Installation

The original Rooein reproduction uses the top-level requirements:

```bash
git clone <repo-url>
cd Beyond-Flesch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Additional folders have their own requirements:

```bash
pip install -r "Final_Architecture-Pillar_B plus/requirements.txt"
pip install -r llm_as_a_judge/requirements.txt
```

GPU notes:

- Rooein Step 5 prompt inference needs a GPU.
- `llm_as_a_judge/scripts/step3f_llm_judge.py` needs a GPU for
  Llama-3.1-8B inference.
- Pillar B/Pillar B+ fine-tuning needs a CUDA GPU.
- Some notebooks and scripts were developed in Colab or on a remote GPU box.
  If rerunning them locally, update hardcoded local paths or virtualenv paths
  before launching.

## How to Run

### 1. Rooein Reproduction

Start with the notebooks in `logistic-regression/notebooks/`.

| Step | File or stage | Purpose |
|---|---|---|
| 1-4 | `step1_to_4_pipeline.ipynb` | setup, ScienceQA preprocessing, static metrics, prompt templates |
| 5 | `step5_inference_gemma_7b.ipynb`, `Step5_Inference_Mistral_7b-2.ipynb`, `Step5_Inference_Llama2_7b-2.ipynb` | prompt-metric inference |
| 6 | `Step6_Baselines_ZeroShot_FewShot.ipynb` | zero-shot and few-shot baselines |
| 7 | `Step7_Modeling_Evaluation.ipynb` | logistic regression, feature selection, final ScienceQA evaluation |
| 8+ | `Step8*`, `Step9*` notebooks | multi-corpus, OOD, and hybrid experiments |

Main output locations:

- `logistic-regression/data/train.csv`
- `logistic-regression/data/test.csv`
- `logistic-regression/outputs/static_metrics/`
- `logistic-regression/outputs/prompt_metrics/`
- `logistic-regression/results/by_prompt_model/{model}/`


## 2. Train-on-Synthetic / Test-on-Real (Multi-LLM)

Generates synthetic grade-level Q/A training data from three LLM APIs (Claude, DeepSeek, Mistral), then trains a classifier on each synthetic set and evaluates it against three real corpora. The generation scripts live under `synthetic_data/`; the cross-corpus training and evaluation runs from `TrainOnSynthetic_TestOnReal_MultiLLM.ipynb`.

First generate one synthetic dataset per LLM. Each script splits the work across several API calls (rotating through five subject areas to reduce overlap), balances samples across the three grade levels, deduplicates on question text, and writes a CSV. Set the relevant API key as an environment variable (or paste it into `API_KEY`) before running:

```
cd synthetic_generation
export ANTHROPIC_API_KEY="sk-..."   && python claude-generation.py     # -> synthetic_claude_qa.csv
export DEEPSEEK_API_KEY="sk-..."    && python deepseek-generation.py   # -> synthetic_deepseek_qa.csv
export MISTRAL_API_KEY="..."        && python mistral-generation.py    # -> synthetic_mistral_qa.csv
```

Then open and run `TrainOnSynthetic_TestOnReal_MultiLLM.ipynb` (Colab/GPU). For each synthetic dataset it trains an ELECTRA-large scalar-mix classifier (80/20 stratified split for held-in monitoring, 3 epochs), then evaluates the trained model against three real corpora — ScienceQA, OneStopEnglish, and RACE (middle + high) — writing per-epoch losses and per-corpus classification reports to a log per run.

Main inputs/outputs:

* Synthetic CSVs: `synthetic_{claude,deepseek,mistral}_qa.csv`
* Real test corpora: ScienceQA and OneStopEnglish (loaded from HuggingFace), RACE (`ood_race-middle.csv`, `ood_race-high.csv`)
* Per-run logs: `electra_<llm>_full_log.txt` and a per-tag results CSV


### 3. LLM-as-a-Judge Pipeline

The full pipeline is implemented as scripts under `llm_as_a_judge/scripts/`.
The included `run.sh` documents the intended order, but it contains a
machine-specific Python path. For a fresh checkout, either update `PY` in
`run.sh` or run the scripts with your active environment:

```bash
cd llm_as_a_judge
python scripts/step3e_split_holdout.py
python scripts/step3d_static_features.py --input-dir outputs/splits --splits train val test ood_onestop ood_race-middle ood_race-high
python scripts/step3f_llm_judge.py
python scripts/step3g_label_disagreement.py
python scripts/step3h_build_clean_dataset.py
python scripts/step8l_combo_classifier.py --config all_llms --label-source llm_judge
python scripts/step9f_calibration_sweep.py
python scripts/plot_results.py
```

Main output locations:

- `llm_as_a_judge/outputs/clean_dataset/`
- `llm_as_a_judge/outputs/eval_reports/`
- `llm_as_a_judge/outputs/figures/`
- `llm_as_a_judge/outputs/models/`

### 4. Final Pillar B+ Architecture

The final architecture and its saved reports are in
`Final_Architecture-Pillar_B plus/`.

Useful entry points:

```bash
cd "Final_Architecture-Pillar_B plus"
python scripts/build_adv_concept.py
python scripts/pillar_a_lean_combo.py
python scripts/pillar_b_tiny_teacher.py --backbones qwen-2.5-3b-instruct phi-3.5-mini-instruct gemma-2-2b-it --train-subset 5000 --epochs 1
python scripts/pillar_b_plus.py --epochs 2 --target-per-class 3500 --lora-r 32
python scripts/compare_carbon.py
```

Saved final reports:

- `Final_Architecture-Pillar_B plus/PILLAR_B_PLUS.md`
- `Final_Architecture-Pillar_B plus/outputs/eval_reports/comparison_table.md`
- `Final_Architecture-Pillar_B plus/outputs/eval_reports/pillar_b_plus__phi-3.5-mini-instruct.eval.json`
- `Final_Architecture-Pillar_B plus/outputs/emissions/emissions.csv`
- `Final_Architecture-Pillar_B plus/outputs/figures/`

## Data and Artifacts

Important datasets and generated artifacts:

- `logistic-regression/data/train.csv` and `test.csv`: balanced ScienceQA
  split with 4,548 total examples.
- `logistic-regression/outputs/static_metrics/`: 46 static readability and
  linguistic features.
- `logistic-regression/outputs/prompt_metrics/`: 63 prompt metrics per LLM.
- `llm_as_a_judge/outputs/clean_dataset/`: multi-corpus dataset with both
  original labels and judge labels.
- `synthetic-data/synthetic_{claude,deepseek,mistral}_qa.csv`: synthetic
  QA datasets generated to test whether LLM-created examples improve
  generalization.
- `Final_Architecture-Pillar_B plus/data/adv_concept.csv`: AdvConcept-50.
- `Final_Architecture-Pillar_B plus/outputs/eval_reports/`: final model
  comparison reports.

Large model weights are not stored directly in this repository. The final model
uses the Hugging Face base model `microsoft/Phi-3.5-mini-instruct` plus the
LoRA adapter produced by the Pillar B+ script.

## Team Contributions

### Shrishti Srivastava

Shrishti's work includes:

- `Final_Architecture-Pillar_B plus/`
  - Built the final Pillar B+ direction, focused on curriculum concept
    difficulty instead of only surface readability.
  - Created AdvConcept-50 and organized the final comparison reports, figures,
    emissions, and model-card documentation.
- `llm_as_a_judge/`
  - Built the LLM-as-a-judge pipeline, produced the clean dataset with original
    and judge labels, and evaluated judge-label vs original-label performance.
- `logistic-regression/`
  - Implemented the Rooein reproduction through LLM inference: ScienceQA
    preprocessing, static metrics, prompt definitions, and Step 5 prompt-metric
    inference.
  - Worked on later generalization/OOD Step notebooks, including multi-corpus
    loading, generic feature extraction, newer-model prompt inference, DeBERTa
    and hybrid variants, and per-corpus OOD breakdowns.

### Maneesha Prasanna

TODO: Add final contribution summary here.


### Smruthi Danda

Smruthi's work includes:

- `transformers/`
  - Implemented transformer-based difficulty classification on ScienceQA: text
    preprocessing and class balancing, a ScalarMix-pooled ScoringModel with a
    learnable Rational-activation head variant, and a hierarchical two-stage
    (elementary-vs-rest, then middle-vs-high) classifier on DeBERTa-v3-large.
  - Tested on different models like deberta, but only attached the code that
    produced the best results.
- `synthetic-data/`
  - Created synthetic QA datasets with Claude, DeepSeek, and Mistral.
  - Evaluated train-on-synthetic/test-on-real behavior to test whether generated
    data alone could improve OOD generalization.

## Caveats and Future Work

- `AdvConcept-50` is intentionally diagnostic but small. A stronger version
  should scale it to `AdvConcept-500` and validate labels with K-12 teachers.
- The LLM-as-a-judge pipeline currently depends on one judge model. A
  multi-judge panel would reduce single-model label bias.
- Some OOD sets have different label definitions and class distributions, so
  macro-F1 comparisons should be interpreted as generalization evidence rather
  than direct leaderboard scores.
- A promising next step is to keep the same lightweight model direction while
  adding domain-adversarial training. The DANN experiments performed well on
  held-out CoQA and CommonLit, so combining Pillar B+ efficiency with
  domain-invariant representations is worth testing.
- Several scripts were run on remote GPU infrastructure. Saved artifacts are
  included, but fresh reruns may require path cleanup and model downloads.



## References

- Rooein, D., Rottger, P., Shaitarova, A., & Hovy, D. (2024). *Beyond
  Flesch-Kincaid: Prompt-based Metrics Improve Difficulty Classification of
  Educational Texts.* BEA Workshop, ACL 2024.
- Gombert, S., Menzel, L., Di Mitri, D., & Drachsler, H. (2024). *Predicting
  Item Difficulty and Item Response Time with Scalar-mixed Transformer Encoder
  Models and Rational Network Regression Heads.* BEA Workshop, ACL 2024.
- Hu, E. J., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language
  Models.* arXiv:2106.09685.
- Lacoste, A., Luccioni, A., Schmidt, V., & Dandres, T. (2019). *Quantifying
  the carbon emissions of machine learning.* arXiv:1910.09700.
- Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O.,
  Clark, P., & Kalyan, A. (2022). *Learn to explain: Multimodal reasoning via
  thought chains for science question answering.* NeurIPS.
- Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT:
  Pre-training of deep bidirectional transformers for language understanding.*
  NAACL.
- Vajjala, S., & Lucic, I. (2018). *OneStopEnglish corpus: A new corpus for
  automatic readability assessment and text simplification.* BEA Workshop.
- Anthropic. (2025). *Claude Code documentation.*
  https://docs.anthropic.com/en/docs/claude-code/overview
- OpenAI. (2025). *Codex.*
  https://openai.com/codex
