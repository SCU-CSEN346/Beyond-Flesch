"""
Step 9f — Per-corpus K-shot calibration sweep.

For each OOD split that has a trained model's logits/labels in
outputs/models/<tag>.pkl:

    For K in {0, 10, 50, 100} (and 'full' as oracle):
        Repeat with `n_resamples` seeds:
            Sample K labeled examples uniformly from that OOD split.
            Fit a multinomial LogisticRegression on (logits) -> labels.
            Apply to remaining rows. Compute macro-F1.
    Aggregate (mean +/- std).

Writes outputs/eval_reports/calibration_<tag>.json with rows
    (split, K, mean_f1, std_f1, n_resamples).

Picks the best model bundle automatically (by val_f1) unless --pkl is passed.
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
from codecarbon import EmissionsTracker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
MODELS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'models')
EVAL_DIR     = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

LABEL_MAP   = {'elementary': 0, 'middle': 1, 'high': 2}


def _best_bundle():
    """Pick the (config=all_llms, label_source=llm_judge) bundle with the
    highest val F1 across seeds. That's the production candidate."""
    best, best_f1 = None, -1.0
    for p in sorted(glob.glob(os.path.join(MODELS_DIR, 'all_llms__llm_judge__*.pkl'))):
        with open(p, 'rb') as fh:
            b = pickle.load(fh)
        f1 = b['report']['best_val_f1_macro']
        if f1 > best_f1:
            best, best_f1 = p, f1
    if best is None:
        sys.exit("[step9f] no all_llms__llm_judge__*.pkl bundles found in outputs/models/. "
                 "Run step8l first.")
    print(f"[step9f] picked bundle {best}  val_f1={best_f1:.3f}")
    return best


def _ood_labels(ood, label_source):
    """Re-load OOD labels for a given split, matching the bundle's label_source
    so we calibrate/evaluate against the *same* label scheme the model was
    trained on. Returns int64 array or None if unavailable."""
    import pandas as pd
    if label_source == 'llm_judge':
        path = os.path.join(_PROJECT_DIR, 'outputs', 'llm_judge', f'{ood}_judge.csv')
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        col = 'llm_judge_label'
    elif label_source == 'original':
        path = os.path.join(_PROJECT_DIR, 'outputs', 'splits', f'{ood}.csv')
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        col = 'education_level'
    else:
        return None
    return df[col].map(LABEL_MAP).astype('Int64').dropna().astype(np.int64).to_numpy()


def calibrate(logits, y, ks=(0, 10, 50, 100), n_resamples=5, seed=42):
    """For each K, sample K labeled rows, fit LR on logits, eval on the rest.
    Returns a list of dicts: {K, mean_f1, std_f1, n_resamples}."""
    rng = np.random.default_rng(seed)
    n = len(y)
    results = []
    for K in ks:
        if K == 0:
            # No calibration: argmax of logits directly.
            pred = logits.argmax(-1)
            f1 = f1_score(y, pred, average='macro', labels=[0, 1, 2], zero_division=0)
            results.append({'K': 0, 'mean_f1': float(f1), 'std_f1': 0.0, 'n_resamples': 1})
            continue
        if K >= n:
            results.append({'K': K, 'mean_f1': None, 'std_f1': None, 'n_resamples': 0,
                            'note': f'K ({K}) >= n ({n}), skipped'})
            continue
        f1s = []
        for r in range(n_resamples):
            idx = np.arange(n)
            rng.shuffle(idx)
            cal_idx, eval_idx = idx[:K], idx[K:]
            if len(np.unique(y[cal_idx])) < 2:
                continue  # LR needs at least 2 classes in calibration set
            try:
                lr = LogisticRegression(max_iter=2000)
                lr.fit(logits[cal_idx], y[cal_idx])
                pred = lr.predict(logits[eval_idx])
                f1 = f1_score(y[eval_idx], pred, average='macro',
                              labels=[0, 1, 2], zero_division=0)
                f1s.append(float(f1))
            except Exception:
                continue
        if f1s:
            results.append({'K': K, 'mean_f1': float(np.mean(f1s)),
                            'std_f1': float(np.std(f1s)), 'n_resamples': len(f1s)})
        else:
            results.append({'K': K, 'mean_f1': None, 'std_f1': None, 'n_resamples': 0})

    # 'Full' oracle: train LR on 80% of OOD, test on 20%.
    f1s = []
    for r in range(n_resamples):
        idx = np.arange(n)
        rng.shuffle(idx)
        split = int(0.8 * n)
        if len(np.unique(y[idx[:split]])) < 2:
            continue
        lr = LogisticRegression(max_iter=2000)
        lr.fit(logits[idx[:split]], y[idx[:split]])
        pred = lr.predict(logits[idx[split:]])
        f1s.append(float(f1_score(y[idx[split:]], pred, average='macro',
                                  labels=[0, 1, 2], zero_division=0)))
    if f1s:
        results.append({'K': 'full_80pct_oracle', 'mean_f1': float(np.mean(f1s)),
                        'std_f1': float(np.std(f1s)), 'n_resamples': len(f1s)})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=None,
                    help='Step8l bundle. Defaults to best all_llms__llm_judge by val F1.')
    ap.add_argument('--ks', nargs='+', type=int, default=[0, 10, 50, 100])
    ap.add_argument('--n-resamples', type=int, default=5)
    args = ap.parse_args()
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step9f_calibration_sweep',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        pkl_path = args.pkl or _best_bundle()
        with open(pkl_path, 'rb') as fh:
            bundle = pickle.load(fh)

        ood_logits = bundle['ood_logits']
        label_source = bundle['report']['label_source']
        out = {'pkl': pkl_path,
               'config': bundle['report']['config'],
               'label_source': label_source,
               'seed': bundle['report']['seed'],
               'sweep_ks': args.ks,
               'by_ood': {}}

        for ood, logits in ood_logits.items():
            y = _ood_labels(ood, label_source)
            if y is None or len(y) != len(logits):
                print(f"[step9f] skipping {ood}: y unavailable or length mismatch "
                      f"(y={None if y is None else len(y)} logits={len(logits)})")
                continue
            print(f"\n[step9f] {ood}  n={len(y)}  class_dist={np.bincount(y, minlength=3).tolist()}")
            out['by_ood'][ood] = calibrate(logits, y, ks=tuple(args.ks),
                                           n_resamples=args.n_resamples)
            for r in out['by_ood'][ood]:
                print(f"  K={r['K']:<8}  mean_f1={r['mean_f1']}  std={r['std_f1']}  "
                      f"n_resamples={r['n_resamples']}")

        out_path = os.path.join(EVAL_DIR, 'calibration_sweep.json')
        with open(out_path, 'w') as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[step9f] wrote {out_path}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
