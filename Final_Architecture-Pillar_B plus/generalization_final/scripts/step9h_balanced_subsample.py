"""
Step 9h — Balanced 150-row subsample evaluation.

Sample 50 rows per judge-label class from the existing test split (150 rows
total, balanced) and evaluate the trained COMBO bundle on this clean
balanced subset. This is *not* OOD — it's an in-distribution slice — but it
removes the class-imbalance artifact that has been driving the macro-F1
ceiling so we can see what the model actually does when classes are equal.

All features are already on disk (static + 4 LLM prompt features). No model
inference needed except the MLP forward pass.

Output: outputs/eval_reports/balanced_subsample_results.json
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score)

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
SPLITS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'splits')
STATIC_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'static_features')
JUDGE_DIR    = os.path.join(_PROJECT_DIR, 'outputs', 'llm_judge')
MODELS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'models')
EVAL_DIR     = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
PROMPT_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'prompt_metrics')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

LABEL_MAP    = {'elementary': 0, 'middle': 1, 'high': 2}
LABEL_NAMES  = ['elementary', 'middle', 'high']
FEATURE_LLMS = ['qwen2.5-7b', 'mistral-7b', 'gemma-7b', 'llama2-7b']
SPLIT        = 'test'      # subsample from the in-distribution test split


def _pick_best_bundle():
    best_path, best_f1 = None, -1.0
    pattern = os.path.join(MODELS_DIR, 'all_llms__llm_judge__all__seed*.pkl')
    for p in sorted(glob.glob(pattern)):
        with open(p, 'rb') as fh:
            b = pickle.load(fh)
        f1 = b['report']['best_val_f1_macro']
        if f1 > best_f1:
            best_path, best_f1 = p, f1
    if best_path is None:
        sys.exit(f"[step9h] no bundles matching {pattern}")
    print(f"[step9h] using bundle {os.path.basename(best_path)}  val_f1={best_f1:.3f}")
    return best_path


def _load_full_features():
    # Static features are row-aligned to splits/<SPLIT>.csv (post-OneStop pull-out).
    sdf = pd.read_csv(os.path.join(STATIC_DIR, f'{SPLIT}_static.csv'))
    stat_cols = sorted(c for c in sdf.columns if c.startswith('stat_'))
    static = sdf[stat_cols].astype(np.float32).to_numpy()
    n = len(static)

    # Prompt CSVs were generated against the ORIGINAL curated test (pre-OneStop
    # pull-out), so we look them up by orig_idx (the position in that original
    # CSV). The same pattern step8l uses.
    split_df = pd.read_csv(os.path.join(SPLITS_DIR, f'{SPLIT}.csv'))
    if len(split_df) != n:
        sys.exit(f"[step9h] split rows {len(split_df)} != static rows {n}")
    orig_idx = split_df['orig_idx'].astype(np.int64).to_numpy()

    parts = [static]
    for llm in FEATURE_LLMS:
        path = os.path.join(PROMPT_DIR, f'{SPLIT}_prompt_metrics_{llm}__multi.csv')
        if not os.path.exists(path):
            sys.exit(f"[step9h] missing {path}")
        pdf = pd.read_csv(path)
        cols = sorted([c for c in pdf.columns if c.startswith('prompt_')],
                      key=lambda s: int(s.split('_')[1]))
        arr = pdf[cols].astype(np.float32).to_numpy()
        # Impute hedge (-1) -> col mean BEFORE selecting rows
        for j in range(arr.shape[1]):
            col = arr[:, j]
            if (col < 0).any():
                valid = col[col >= 0]
                col[col < 0] = float(valid.mean()) if len(valid) else 0.0
        # Select rows by orig_idx
        if orig_idx.max() >= len(arr):
            sys.exit(f"[step9h] {llm}: orig_idx max={orig_idx.max()} >= prompt rows={len(arr)}")
        parts.append(arr[orig_idx])
    return np.concatenate(parts, axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=50,
                    help='Number of rows to sample per class (default 50 -> 150 total)')
    ap.add_argument('--n-resamples', type=int, default=5,
                    help='Number of random balanced subsamples to draw')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--pkl', default=None)
    args = ap.parse_args()
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step9h_balanced_subsample',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        from step8l_combo_classifier import ComboMLP

        pkl_path = args.pkl or _pick_best_bundle()
        with open(pkl_path, 'rb') as fh:
            bundle = pickle.load(fh)
        scaler = bundle['scaler']
        in_dim = bundle['feature_dim']

        # ---- Build all-test features once ----
        X_all = _load_full_features()
        if X_all.shape[1] != in_dim:
            sys.exit(f"[step9h] feature-dim mismatch: built {X_all.shape[1]}, "
                     f"bundle expects {in_dim}")
        X_all_s = scaler.transform(X_all).astype(np.float32)

        # ---- Load judge labels for the FULL test split ----
        judge_df = pd.read_csv(os.path.join(JUDGE_DIR, f'{SPLIT}_judge.csv'))
        y_all = judge_df['llm_judge_label'].map(LABEL_MAP).astype(np.int64).to_numpy()
        n_total = len(y_all)
        class_counts = np.bincount(y_all, minlength=3)
        print(f"[step9h] full {SPLIT} split: n={n_total}, class_dist={class_counts.tolist()}")

        # ---- Per-class sample sizes ----
        K = args.per_class
        sample_per_class = {i: min(K, int(class_counts[i])) for i in range(3)}
        actual_per_class = sample_per_class.copy()
        if min(sample_per_class.values()) < K:
            print(f"[step9h] WARNING: requested {K}/class, but actual minimum is "
                  f"{min(sample_per_class.values())} (class {min(sample_per_class, key=sample_per_class.get)})")
        print(f"[step9h] sampling: {sample_per_class}")

        # ---- Reconstruct model from bundle ----
        use_grade_reg = bool(bundle.get('use_grade_reg', False))
        model = ComboMLP(in_dim=in_dim, use_grade_reg=use_grade_reg)
        model.load_state_dict(bundle['model_state'])
        model.eval()

        # ---- Multiple resamples for std-dev ----
        rng = np.random.default_rng(args.seed)
        all_results = []
        for r in range(args.n_resamples):
            chosen = []
            for cls, n_pick in sample_per_class.items():
                pool = np.where(y_all == cls)[0]
                pick = rng.choice(pool, size=n_pick, replace=False)
                chosen.append(pick)
            idx = np.concatenate(chosen)
            rng.shuffle(idx)

            X_sub = X_all_s[idx]
            y_sub = y_all[idx]
            with torch.no_grad():
                cls_logits, _ = model(torch.tensor(X_sub, dtype=torch.float32))
                logits = cls_logits.numpy()
            pred = logits.argmax(-1)

            all_results.append({
                'resample': r,
                'n': int(len(y_sub)),
                'class_dist': np.bincount(y_sub, minlength=3).tolist(),
                'f1_macro':    float(f1_score(y_sub, pred, average='macro',
                                              labels=[0, 1, 2], zero_division=0)),
                'f1_per_class': f1_score(y_sub, pred, average=None,
                                         labels=[0, 1, 2], zero_division=0).tolist(),
                'accuracy':    float(accuracy_score(y_sub, pred)),
                'qwk':         float(cohen_kappa_score(y_sub, pred, weights='quadratic')),
                'conf_matrix': confusion_matrix(y_sub, pred, labels=[0, 1, 2]).tolist(),
                'pred_counts': np.bincount(pred, minlength=3).tolist(),
            })

        # Aggregate
        f1_arr   = np.array([r['f1_macro']    for r in all_results])
        acc_arr  = np.array([r['accuracy']    for r in all_results])
        qwk_arr  = np.array([r['qwk']         for r in all_results])
        pc_arr   = np.array([r['f1_per_class'] for r in all_results])

        agg = {
            'pkl': pkl_path,
            'split_source': SPLIT,
            'per_class_requested': K,
            'per_class_actual': sample_per_class,
            'n_resamples': args.n_resamples,
            'mean_f1_macro':    float(f1_arr.mean()),
            'std_f1_macro':     float(f1_arr.std()),
            'mean_accuracy':    float(acc_arr.mean()),
            'std_accuracy':     float(acc_arr.std()),
            'mean_qwk':         float(qwk_arr.mean()),
            'std_qwk':          float(qwk_arr.std()),
            'mean_f1_per_class': pc_arr.mean(axis=0).tolist(),
            'std_f1_per_class':  pc_arr.std(axis=0).tolist(),
            'per_resample': all_results,
        }
        out = os.path.join(EVAL_DIR, 'balanced_subsample_results.json')
        with open(out, 'w') as fh:
            json.dump(agg, fh, indent=2)
        print(f"\n[step9h] wrote {out}")

        # Headline print
        print("\n=== Balanced subsample (judge labels, in-distribution test) ===")
        print(f"  bundle:              {os.path.basename(pkl_path)}")
        print(f"  per-class samples:   {sample_per_class}  x {args.n_resamples} resamples")
        print(f"  macro-F1:            {agg['mean_f1_macro']:.3f} ± {agg['std_f1_macro']:.3f}")
        print(f"  accuracy:            {agg['mean_accuracy']:.3f} ± {agg['std_accuracy']:.3f}")
        print(f"  QWK:                 {agg['mean_qwk']:.3f} ± {agg['std_qwk']:.3f}")
        print(f"  per-class F1 mean:   "
              f"elem={agg['mean_f1_per_class'][0]:.3f}  "
              f"mid={agg['mean_f1_per_class'][1]:.3f}  "
              f"high={agg['mean_f1_per_class'][2]:.3f}")
        print(f"  example conf-matrix (resample 0): {all_results[0]['conf_matrix']}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
