"""
Step 3h — Build the share-ready clean dataset.

For every split in outputs/splits/ that has a matching judge file in
outputs/llm_judge/, produce a single tidy CSV with both the original corpus
label and the LLM-judge unified label side-by-side.

This is the artifact to share with teammates. Each CSV row carries everything
they need to either (a) reproduce our experiments or (b) train their own
classifier on the LLM-judge unified labels.

Output layout:
    outputs/clean_dataset/
        train.csv          (training pool after pulling OneStop out)
        val.csv
        test.csv
        ood_onestop.csv    (held-out OneStopEnglish)
        ood_race-middle.csv
        ood_race-high.csv
        all_rows.csv       (concatenation of all splits, with `split` column)
        README.md          (column docs)
        manifest.json      (row counts + label distributions per split)

Per-row columns:
    split, orig_split, orig_idx, source_dataset, subject (optional),
    raw_label (optional), full_text,
    education_level_original,           <- corpus's label
    education_level_judge,              <- Llama-3.1-8B unified judge label
    judge_raw_response                  <- raw text the judge returned
"""

import argparse
import json
import os
import sys

import pandas as pd
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
SPLITS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'splits')
JUDGE_DIR    = os.path.join(_PROJECT_DIR, 'outputs', 'llm_judge')
OUT_DIR      = os.path.join(_PROJECT_DIR, 'outputs', 'clean_dataset')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

# Stable column order in the share-ready files.
CORE_COLS = ['split', 'orig_split', 'orig_idx', 'source_dataset',
             'subject', 'raw_label', 'full_text',
             'education_level_original', 'education_level_judge',
             'judge_raw_response']

README_TXT = """# Clean Dataset — LLM-as-a-Judge labels

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
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-dir', default=SPLITS_DIR)
    ap.add_argument('--judge-dir',  default=JUDGE_DIR)
    ap.add_argument('--out-dir',    default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir,  exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step3h_build_clean_dataset',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        per_split = {}
        for fname in sorted(os.listdir(args.splits_dir)):
            if not fname.endswith('.csv'):
                continue
            split = fname.replace('.csv', '')
            split_csv = os.path.join(args.splits_dir, fname)
            judge_csv = os.path.join(args.judge_dir, f'{split}_judge.csv')
            if not os.path.exists(judge_csv):
                print(f"[step3h] skip {split}: no judge CSV at {judge_csv}")
                continue
            sdf = pd.read_csv(split_csv)
            jdf = pd.read_csv(judge_csv)
            if len(sdf) != len(jdf):
                sys.exit(f"[step3h] {split}: split rows {len(sdf)} != judge rows {len(jdf)}")
            out = pd.DataFrame()
            out['split']                    = [split] * len(sdf)
            out['orig_split']               = sdf.get('orig_split', split)
            out['orig_idx']                 = sdf.get('orig_idx', range(len(sdf)))
            out['source_dataset']           = sdf['source_dataset'].astype(str)
            out['subject']                  = sdf['subject'].astype(str) if 'subject' in sdf.columns else ''
            out['raw_label']                = sdf['raw_label'] if 'raw_label' in sdf.columns else ''
            out['full_text']                = sdf['full_text'].astype(str)
            out['education_level_original'] = sdf['education_level'].astype(str)
            out['education_level_judge']    = jdf['llm_judge_label'].astype(str)
            out['judge_raw_response']       = jdf['judge_raw_response'].astype(str)
            out = out[CORE_COLS]
            out_path = os.path.join(args.out_dir, f'{split}.csv')
            out.to_csv(out_path, index=False)
            per_split[split] = out
            agree = (out['education_level_original'] == out['education_level_judge']).mean()
            print(f"[step3h] {split}: {len(out)} rows, "
                  f"original-vs-judge agreement={agree:.3f}, "
                  f"judge_dist={dict(out['education_level_judge'].value_counts())}")

        if per_split:
            all_rows = pd.concat(per_split.values(), ignore_index=True)
            all_rows.to_csv(os.path.join(args.out_dir, 'all_rows.csv'), index=False)
            print(f"[step3h] all_rows.csv: {len(all_rows)} rows")

            manifest = {
                'n_total': int(len(all_rows)),
                'overall_agreement': float((all_rows['education_level_original'] ==
                                            all_rows['education_level_judge']).mean()),
                'splits': {split: {
                    'n_rows': int(len(df)),
                    'original_dist': dict(df['education_level_original'].value_counts()),
                    'judge_dist':    dict(df['education_level_judge'].value_counts()),
                    'agreement':     float((df['education_level_original'] ==
                                            df['education_level_judge']).mean()),
                } for split, df in per_split.items()},
            }
            with open(os.path.join(args.out_dir, 'manifest.json'), 'w') as fh:
                json.dump(manifest, fh, indent=2, default=str)

            with open(os.path.join(args.out_dir, 'README.md'), 'w') as fh:
                fh.write(README_TXT)
            print(f"[step3h] wrote manifest.json + README.md to {args.out_dir}")
        else:
            sys.exit("[step3h] no splits processed — did step3f run?")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
