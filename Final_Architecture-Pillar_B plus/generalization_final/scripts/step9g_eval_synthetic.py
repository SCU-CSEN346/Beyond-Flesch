"""
Step 9g — Evaluate the trained COMBO classifier on the synthetic 150-text
balanced OOD set.

Loads the best `all_llms__llm_judge__all__seed*.pkl` bundle (highest val F1),
builds the same 286-dim feature vector for each synthetic text (46 static
+ 4 LLMs x 63 prompt features = 286 total — wait, actually 34 static + 252
prompt = 286; the README label is approximate), scales with the bundle's
fitted scaler, and predicts.

Reports macro-F1 and per-class F1 against BOTH:
    * gen labels        (what we asked Llama-3.1 to write at)
    * llm_judge labels  (what Llama-3.1 said after re-judging)

The gen-vs-judge agreement is also surfaced as the synthetic-data quality
signal.

Inputs expected (all already on disk before this script runs):
    outputs/splits/ood_synthetic.csv
    outputs/static_features/ood_synthetic_static.csv
    outputs/llm_judge/ood_synthetic_judge.csv
    ../generalization_pipeline/outputs/prompt_metrics/
        ood_synthetic_prompt_metrics_<llm>__multi.csv     (for each of 4 LLMs)

Output:
    outputs/eval_reports/ood_synthetic_results.json
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
DEFAULT_FEATURE_LLMS = ['qwen2.5-7b', 'mistral-7b', 'gemma-7b', 'llama2-7b']
DEFAULT_SPLIT_ARG = 'ood_synthetic'


def _pick_best_bundle(pattern_glob):
    best_path, best_f1 = None, -1.0
    pattern = os.path.join(MODELS_DIR, pattern_glob)
    for p in sorted(glob.glob(pattern)):
        with open(p, 'rb') as fh:
            b = pickle.load(fh)
        f1 = b['report']['best_val_f1_macro']
        if f1 > best_f1:
            best_path, best_f1 = p, f1
    if best_path is None:
        sys.exit(f"[step9g] no bundles matching {pattern}. Run step8l first.")
    print(f"[step9g] using bundle {best_path}  (val_f1={best_f1:.3f})")
    return best_path


def _load_static(split):
    p = os.path.join(STATIC_DIR, f'{split}_static.csv')
    if not os.path.exists(p):
        sys.exit(f"[step9g] missing {p}. Run step3d --splits {split} first.")
    df = pd.read_csv(p)
    cols = sorted(c for c in df.columns if c.startswith('stat_'))
    return df[cols].astype(np.float32).to_numpy(), cols


def _load_prompts(split, llm):
    p = os.path.join(PROMPT_DIR, f'{split}_prompt_metrics_{llm}__multi.csv')
    if not os.path.exists(p):
        sys.exit(f"[step9g] missing {p}. Run step5d for {llm} on {split} first.")
    df = pd.read_csv(p)
    cols = sorted([c for c in df.columns if c.startswith('prompt_')],
                  key=lambda s: int(s.split('_')[1]))
    arr = df[cols].astype(np.float32).to_numpy()
    # Hedge (-1) -> per-column mean of valid rows.
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if (col < 0).any():
            valid = col[col >= 0]
            col[col < 0] = float(valid.mean()) if len(valid) else 0.0
    return arr


def _eval_block(y, p):
    return {
        'f1_macro':    float(f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0)),
        'f1_per_class': f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        'accuracy':    float(accuracy_score(y, p)),
        'qwk':         float(cohen_kappa_score(y, p, weights='quadratic')),
        'conf_matrix': confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        'pred_counts': np.bincount(np.asarray(p, dtype=np.int64), minlength=3).tolist(),
        'n':           int(len(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default=None,
                    help='step8l bundle (defaults to best by val F1 matching --bundle-pattern)')
    ap.add_argument('--bundle-pattern', default='all_llms__llm_judge__all__seed*.pkl',
                    help='Glob pattern under outputs/models/ to choose the best bundle from')
    ap.add_argument('--llms', nargs='+', default=DEFAULT_FEATURE_LLMS,
                    help='Feature LLMs to concatenate (must match what the bundle was trained on)')
    ap.add_argument('--split', default=DEFAULT_SPLIT_ARG,
                    help='Which split to evaluate on (default: ood_synthetic).')
    args = ap.parse_args()
    SPLIT = args.split
    os.makedirs(EVAL_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step9g_eval_synthetic',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        from step8l_combo_classifier import ComboMLP

        pkl_path = args.pkl or _pick_best_bundle(args.bundle_pattern)
        with open(pkl_path, 'rb') as fh:
            bundle = pickle.load(fh)

        scaler   = bundle['scaler']
        in_dim   = bundle['feature_dim']

        # ---- Build feature vector for synthetic: 34 static + N x 63 prompts ----
        static_X, stat_cols = _load_static(SPLIT)
        n = len(static_X)
        parts = [static_X]
        for llm in args.llms:
            parts.append(_load_prompts(SPLIT, llm))
        X = np.concatenate(parts, axis=1).astype(np.float32)
        if X.shape[1] != in_dim:
            sys.exit(f"[step9g] feature-dim mismatch: built {X.shape[1]}, bundle expects {in_dim}. "
                     f"Static has {len(stat_cols)} cols + {len(args.llms)} LLMs x 63 prompts = expected "
                     f"{len(stat_cols) + len(args.llms) * 63}.")
        X = scaler.transform(X).astype(np.float32)

        # ---- Reconstruct the MLP from saved state_dict ----
        use_grade_reg = bool(bundle.get('use_grade_reg', False))
        model = ComboMLP(in_dim=in_dim, use_grade_reg=use_grade_reg)
        model.load_state_dict(bundle['model_state'])
        model.eval()
        with torch.no_grad():
            cls_logits, _ = model(torch.tensor(X, dtype=torch.float32))
            logits = cls_logits.numpy()
        pred = logits.argmax(-1)

        # ---- Load both label sources ----
        split_df = pd.read_csv(os.path.join(SPLITS_DIR, f'{SPLIT}.csv'))
        judge_df = pd.read_csv(os.path.join(JUDGE_DIR, f'{SPLIT}_judge.csv'))
        if len(split_df) != n or len(judge_df) != n:
            sys.exit(f"[step9g] row mismatch: split={len(split_df)} judge={len(judge_df)} static={n}")
        y_gen   = split_df['education_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
        y_judge = judge_df['llm_judge_label'].map(LABEL_MAP).astype(np.int64).to_numpy()

        report = {
            'pkl': pkl_path,
            'n_synthetic': n,
            'feature_dim': int(X.shape[1]),
            'gen_label_dist':   {LABEL_NAMES[i]: int((y_gen == i).sum())   for i in range(3)},
            'judge_label_dist': {LABEL_NAMES[i]: int((y_judge == i).sum()) for i in range(3)},
            'gen_vs_judge_agreement': float((y_gen == y_judge).mean()),
            'vs_gen_label':   _eval_block(y_gen, pred),
            'vs_judge_label': _eval_block(y_judge, pred),
        }

        out = os.path.join(EVAL_DIR, f'{SPLIT}_results.json')
        with open(out, 'w') as fh:
            json.dump(report, fh, indent=2)
        print(f"\n[step9g] wrote {out}")

        # ---- Headline print ----
        print("\n=== Synthetic OOD evaluation ===")
        print(f"  bundle:                 {os.path.basename(pkl_path)}")
        print(f"  n synthetic texts:      {n}")
        print(f"  gen-vs-judge agreement: {report['gen_vs_judge_agreement']:.3f}")
        print(f"  vs GEN labels   : macro-F1={report['vs_gen_label']['f1_macro']:.3f}  "
              f"acc={report['vs_gen_label']['accuracy']:.3f}  "
              f"qwk={report['vs_gen_label']['qwk']:.3f}")
        print(f"  vs JUDGE labels : macro-F1={report['vs_judge_label']['f1_macro']:.3f}  "
              f"acc={report['vs_judge_label']['accuracy']:.3f}  "
              f"qwk={report['vs_judge_label']['qwk']:.3f}")
        print(f"  per-class F1 (vs gen):    {report['vs_gen_label']['f1_per_class']}")
        print(f"  per-class F1 (vs judge):  {report['vs_judge_label']['f1_per_class']}")
        print(f"  conf-matrix (vs judge):   {report['vs_judge_label']['conf_matrix']}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
