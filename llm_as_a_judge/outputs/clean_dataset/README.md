# Clean Dataset — LLM-as-a-Judge labels

This folder contains the labeled dataset produced by the Option E pipeline.
Each row carries BOTH the corpus's original label AND the unified label
assigned by Llama-3.1-8B-Instruct, so you can:

  * train a classifier on `education_level_judge` (recommended; this is
    what gives 0.70+ macro-F1 cross-corpus), or
  * train on `education_level_original` if you want to reproduce the
    noisy baseline (~0.21 macro-F1 cross-corpus).

## Splits

| File                       | Description                                              |
|----------------------------|----------------------------------------------------------|
| train.csv                  | Training pool after pulling OneStop out                  |
| val.csv                    | Validation split (early-stopping in our experiments)     |
| test.csv                   | In-distribution test                                     |
| ood_onestop.csv            | Held-out OneStopEnglish (multi-class, primary OOD)       |
| ood_race-middle.csv        | RACE-middle passages (single-class with original labels) |
| ood_race-high.csv          | RACE-high passages (single-class with original labels)   |
| all_rows.csv               | All splits concatenated, with a `split` column           |

## Columns

| Column                       | What it is                                              |
|------------------------------|---------------------------------------------------------|
| split                        | Which split the row is in (e.g. `train`, `ood_onestop`) |
| orig_split                   | Source split in the curated dataset                     |
| orig_idx                     | Row index in the original curated split CSV             |
| source_dataset               | Source corpus (scienceqa, clear, onestop, race, ...)    |
| subject                      | Subject domain when known (math, sci, ...) — optional   |
| raw_label                    | Original numeric label when available — optional        |
| full_text                    | The text being classified                               |
| education_level_original     | `{elementary, middle, high}` from the source corpus     |
| education_level_judge        | `{elementary, middle, high}` from the Llama-3.1 judge   |
| judge_raw_response           | Verbatim string the judge returned (for audit)          |

## Label encoding

When you load these CSVs and train a model, map the labels:

    {'elementary': 0, 'middle': 1, 'high': 2}

This matches our experiment code (`step8l_combo_classifier.py`).

## Provenance

* Curated dataset built by `step2e_curated_generalization_collection.py`.
* OneStop pulled out by `step3e_split_holdout.py`.
* LLM-judge labels produced by `step3f_llm_judge.py` using
  `meta-llama/Llama-3.1-8B-Instruct` with greedy decoding and a single
  fixed rubric prompt applied uniformly across all corpora.
* See `manifest.json` for row counts and per-split label distributions.
