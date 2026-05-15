"""
Step 3g — Label disagreement diagnostic.

Reads outputs/llm_judge/<split>_judge.csv files. Aggregates:
  * per-corpus original-vs-judge confusion (3x3 stacked-bar friendly form)
  * agreement rate per corpus
  * overall agreement
  * class distribution shift per corpus

Writes:
    outputs/eval_reports/label_disagreement_by_corpus.csv
    outputs/eval_reports/label_disagreement_summary.json

This is the diagnosis evidence for the paper: it documents the cross-corpus
label inconsistency that motivates using an LLM-as-judge in the first place.
"""

import argparse
import json
import os

import pandas as pd
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
JUDGE_DIR    = os.path.join(_PROJECT_DIR, 'outputs', 'llm_judge')
OUT_DIR      = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

LABELS = ['elementary', 'middle', 'high']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-dir', default=JUDGE_DIR)
    ap.add_argument('--out-dir',   default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step3g_label_disagreement',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        frames = []
        for f in sorted(os.listdir(args.judge_dir)):
            if not f.endswith('_judge.csv'):
                continue
            d = pd.read_csv(os.path.join(args.judge_dir, f))
            d['split'] = f.replace('_judge.csv', '')
            frames.append(d)
        if not frames:
            raise SystemExit(f"[step3g] no judge CSVs in {args.judge_dir}")
        all_df = pd.concat(frames, ignore_index=True)

        # Restrict to known labels (drop edge cases just in case).
        all_df = all_df[all_df['education_level'].isin(LABELS) &
                        all_df['llm_judge_label'].isin(LABELS)].copy()

        rows = []
        # Use single-column groupby (string, not list) for compatibility with
        # pandas < 2.0 where tuple-keys aren't returned for single columns.
        for corpus, grp in all_df.groupby('source_dataset'):
            n = len(grp)
            agree = (grp['education_level'] == grp['llm_judge_label']).mean()
            orig_dist  = grp['education_level'].value_counts(normalize=True)
            judge_dist = grp['llm_judge_label'].value_counts(normalize=True)
            row = {'source_dataset': corpus, 'n_rows': n, 'agreement': float(agree)}
            for lbl in LABELS:
                row[f'orig_pct_{lbl}']  = float(orig_dist.get(lbl, 0.0))
                row[f'judge_pct_{lbl}'] = float(judge_dist.get(lbl, 0.0))
            for orig in LABELS:
                for judge in LABELS:
                    mask = (grp['education_level'] == orig) & (grp['llm_judge_label'] == judge)
                    row[f'conf_{orig}_to_{judge}'] = float(mask.sum() / max(n, 1))
            rows.append(row)
        summary = pd.DataFrame(rows).sort_values('source_dataset').reset_index(drop=True)
        out_csv = os.path.join(args.out_dir, 'label_disagreement_by_corpus.csv')
        summary.to_csv(out_csv, index=False)

        overall = {
            'overall_agreement': float((all_df['education_level'] == all_df['llm_judge_label']).mean()),
            'n_total': int(len(all_df)),
            'orig_dist':  {str(k): int(v) for k, v in all_df['education_level'].value_counts().items()},
            'judge_dist': {str(k): int(v) for k, v in all_df['llm_judge_label'].value_counts().items()},
        }
        with open(os.path.join(args.out_dir, 'label_disagreement_summary.json'), 'w') as fh:
            json.dump(overall, fh, indent=2, default=str)

        print(f"[step3g] wrote {out_csv}")
        print(f"[step3g] overall original-vs-judge agreement: {overall['overall_agreement']:.3f}")
        print(summary[['source_dataset', 'n_rows', 'agreement']].to_string(index=False))
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
