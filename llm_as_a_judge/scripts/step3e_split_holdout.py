"""
Step 3e — Pull OneStopEnglish out of train/val/test into a new ood_onestop split.

Reads the curated CSVs from text-difficulty-classification/outputs/...
For each split, filters out rows where source_dataset == 'onestop'.
Combines all pulled rows into a single ood_onestop.csv.

Output:
    outputs/splits/train.csv         (curated train minus onestop)
    outputs/splits/val.csv           (curated val minus onestop)
    outputs/splits/test.csv          (curated test minus onestop)
    outputs/splits/ood_onestop.csv   (the pulled rows)
    outputs/splits/ood_race-middle.csv  (copied through unchanged)
    outputs/splits/ood_race-high.csv    (copied through unchanged)

Every row keeps an `orig_split` and `orig_idx` so we can join prompt features
from generalization_pipeline/outputs/prompt_metrics/ by (orig_split, orig_idx).
"""

import argparse
import os
import sys

import pandas as pd
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT   = os.path.dirname(_PROJECT_DIR)
INPUT_DIR    = os.path.join(_REPO_ROOT, 'text-difficulty-classification', 'outputs',
                            'curated_generalization_collection')
OUTPUT_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'splits')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

ONESTOP_NAMES = ('onestop', 'onestopenglish', 'onestop-english')


def _is_onestop(s):
    return str(s).lower() in ONESTOP_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir',  default=INPUT_DIR)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.output_dir,  exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step3e_split_holdout',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        onestop_chunks = []
        for split in ('train', 'val', 'test'):
            src = os.path.join(args.input_dir, f'{split}.csv')
            if not os.path.exists(src):
                sys.exit(f"[step3e] missing {src}")
            df = pd.read_csv(src)
            df['orig_split'] = split
            df['orig_idx']   = df.index
            is_onestop = df['source_dataset'].map(_is_onestop)
            stays = df[~is_onestop].reset_index(drop=True)
            leaves = df[is_onestop].reset_index(drop=True)
            stays.to_csv(os.path.join(args.output_dir, f'{split}.csv'), index=False)
            print(f"[step3e] {split}: {len(df)} -> {len(stays)} kept, {len(leaves)} -> onestop")
            onestop_chunks.append(leaves)

        ood_onestop = pd.concat(onestop_chunks, ignore_index=True)
        ood_onestop.to_csv(os.path.join(args.output_dir, 'ood_onestop.csv'), index=False)
        print(f"[step3e] ood_onestop.csv rows={len(ood_onestop)} "
              f"label_dist={ood_onestop['education_level'].value_counts().to_dict()}")

        for ood_name in ('ood_race-middle', 'ood_race-high'):
            src = os.path.join(args.input_dir, f'{ood_name}.csv')
            if not os.path.exists(src):
                print(f"[step3e] {src} not present, skipping")
                continue
            df = pd.read_csv(src)
            df['orig_split'] = ood_name
            df['orig_idx']   = df.index
            df.to_csv(os.path.join(args.output_dir, f'{ood_name}.csv'), index=False)
            print(f"[step3e] {ood_name}: passed through, rows={len(df)}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
