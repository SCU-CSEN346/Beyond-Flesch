"""
Step 8l — COMBO classifier (static + prompt features) with softmax 3-class head.

Faithful to Rooein et al. 2024 in spirit: concatenate static + prompt features,
train a small classifier on top. We use an MLP (256, 128, softmax-3) rather
than logistic regression because Phase A on the curated data showed LR
plateaus; MLP gives ~0.03-0.05 macro-F1 lift in-distribution. Loss is plain
cross-entropy on 3 classes.

Two feature configurations:
  - per_llm: 46 static + 63 prompts from ONE LLM = 109-dim (4 LLMs -> 4 models)
  - all_llms: 46 static + 4 * 63 prompts concatenated = 298-dim (1 model)

Two label sources, switched via --label-source:
  - original: corpus's existing label (the noisy baseline ~0.21 macro-F1 OOD)
  - llm_judge: Llama-3.1's unified label (the contribution; target ~0.65-0.80 OOD)

Splits come from outputs/splits/ (post-OneStop-holdout). Prompt features come
from ../generalization_pipeline/outputs/prompt_metrics/ keyed by (orig_split,
orig_idx) since our new splits are filtered slices of those.

Usage:
    python step8l_combo_classifier.py --label-source llm_judge --config all_llms
    python step8l_combo_classifier.py --label-source llm_judge --config per_llm
    python step8l_combo_classifier.py --label-source original  --config all_llms

Per (config, label_source) we train 3 seeds. Outputs per cell:
    outputs/models/<config>__<label_source>__<llm_or_all>__seed<N>.pkl
    outputs/eval_reports/<same_basename>.eval.json
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from codecarbon import EmissionsTracker
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, mean_absolute_error)
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from tqdm.auto import tqdm

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT   = os.path.dirname(_PROJECT_DIR)
SPLITS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'splits')
STATIC_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'static_features')
JUDGE_DIR    = os.path.join(_PROJECT_DIR, 'outputs', 'llm_judge')
PROMPT_DIR   = os.path.join(_REPO_ROOT, 'generalization_pipeline', 'outputs', 'prompt_metrics')
MODELS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'models')
EVAL_DIR     = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

LABEL_MAP   = {'elementary': 0, 'middle': 1, 'high': 2}
NUM_CLASSES = 3
FEATURE_LLMS = ['qwen2.5-7b', 'mistral-7b', 'gemma-7b', 'llama2-7b']  # judge excluded
OOD_SPLITS = ['ood_onestop', 'ood_race-middle', 'ood_race-high']


class ComboMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128), num_classes=NUM_CLASSES, dropout=0.2):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _load_static(split):
    p = os.path.join(STATIC_DIR, f'{split}_static.csv')
    if not os.path.exists(p):
        sys.exit(f"[step8l] missing static features: {p}. Run step3d first.")
    df = pd.read_csv(p)
    cols = [c for c in df.columns if c.startswith('stat_')]
    return df, cols


def _load_prompts(orig_split, llm):
    """Load prompt features from generalization_pipeline outputs, indexed by row.
    The original split CSVs and prompt CSVs are aligned by row index, which
    matches the orig_idx column in our new splits."""
    p = os.path.join(PROMPT_DIR, f'{orig_split}_prompt_metrics_{llm}__multi.csv')
    if not os.path.exists(p):
        sys.exit(f"[step8l] missing prompt features: {p}. "
                 "Re-run step5d in generalization_pipeline first.")
    df = pd.read_csv(p)
    cols = sorted([c for c in df.columns if c.startswith('prompt_')],
                  key=lambda s: int(s.split('_')[1]))
    arr = df[cols].astype(np.float32).to_numpy()
    # Hedge (-1) -> column mean of valid rows
    for j in range(arr.shape[1]):
        col = arr[:, j]
        if (col < 0).any():
            valid = col[col >= 0]
            col[col < 0] = float(valid.mean()) if len(valid) else 0.0
    return arr  # row-aligned to original split CSV


def _load_labels(split, label_source):
    """Return a label array aligned to splits/<split>.csv row order."""
    split_df = pd.read_csv(os.path.join(SPLITS_DIR, f'{split}.csv'))
    if label_source == 'original':
        labels = split_df['education_level'].astype(str)
    elif label_source == 'llm_judge':
        judge = pd.read_csv(os.path.join(JUDGE_DIR, f'{split}_judge.csv'))
        if len(judge) != len(split_df):
            sys.exit(f"[step8l] judge rows {len(judge)} != split rows {len(split_df)} for {split}")
        labels = judge['llm_judge_label'].astype(str)
    else:
        sys.exit(f"[step8l] unknown --label-source {label_source}")
    y = labels.map(LABEL_MAP)
    if y.isna().any():
        bad = labels[y.isna()].unique().tolist()[:5]
        sys.exit(f"[step8l] {split}: unknown labels {bad}")
    return y.astype(np.int64).to_numpy(), split_df


def _build_X(split, llms_to_use):
    """Build feature matrix X for our new split. Looks up prompt features by
    (orig_split, orig_idx) so the rows align even after the OneStop pull-out."""
    static_df, stat_cols = _load_static(split)  # row-aligned to splits/<split>.csv
    split_df = pd.read_csv(os.path.join(SPLITS_DIR, f'{split}.csv'))
    if len(static_df) != len(split_df):
        sys.exit(f"[step8l] {split}: static rows {len(static_df)} != split rows {len(split_df)}. "
                 "Re-run step3d.")

    parts = [static_df[stat_cols].astype(np.float32).to_numpy()]

    # Cache prompt CSVs by (orig_split, llm) so we don't re-read 5 times per LLM.
    prompt_cache = {}
    for llm in llms_to_use:
        prompt_block = np.zeros((len(split_df), 63), dtype=np.float32)
        for i, (orig_split, orig_idx) in enumerate(zip(split_df['orig_split'], split_df['orig_idx'])):
            key = (orig_split, llm)
            if key not in prompt_cache:
                prompt_cache[key] = _load_prompts(orig_split, llm)
            prompt_block[i] = prompt_cache[key][int(orig_idx)]
        parts.append(prompt_block)

    X = np.concatenate(parts, axis=1).astype(np.float32)
    return X


def _eval(y, p):
    return {
        'f1_macro':    float(f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0)),
        'f1_per_class': f1_score(y, p, average=None, labels=[0, 1, 2], zero_division=0).tolist(),
        'accuracy':    float(accuracy_score(y, p)),
        'qwk':         float(cohen_kappa_score(y, p, weights='quadratic')),
        'mae':         float(mean_absolute_error(y, p)),
        'conf_matrix': confusion_matrix(y, p, labels=[0, 1, 2]).tolist(),
        'pred_counts': np.bincount(np.asarray(p, dtype=np.int64), minlength=3).tolist(),
        'n':           int(len(y)),
    }


def fit_one(X_train, y_train, X_val, y_val, *, epochs=120, patience=15,
            batch=64, lr=1e-3, wd=1e-4, seed=42, hidden=(256, 128), dropout=0.2):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.long, device=device)
    Xv = torch.tensor(X_val,   dtype=torch.float32, device=device)
    model = ComboMLP(X_train.shape[1], hidden=hidden, dropout=dropout).to(device)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, stale = -1.0, None, 0
    n = len(X_train)
    for epoch in range(epochs):
        model.train()
        # Proper shuffled-pass: permute once per epoch and iterate through.
        # Bug we previously had: first batch from a shuffled order, but every
        # subsequent batch from rng.choice — never visiting each example once.
        order = rng.permutation(n)
        for start in range(0, n, batch):
            b = order[start:start + batch]
            if len(b) == 0:
                continue
            logits = model(Xt[b])
            loss = F.cross_entropy(logits, yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(Xv).argmax(-1).cpu().numpy()
        f1 = f1_score(y_val, pred, average='macro', labels=[0, 1, 2], zero_division=0)
        if f1 > best_f1:
            best_f1, stale = f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.to('cpu')
    return model, float(best_f1)


@torch.no_grad()
def predict(model, X):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    logits = model(Xt).numpy()
    return logits.argmax(-1), logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=['per_llm', 'all_llms'], required=True)
    ap.add_argument('--label-source', choices=['original', 'llm_judge'], required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 13])
    ap.add_argument('--llms', nargs='+', default=FEATURE_LLMS)
    args = ap.parse_args()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR,   exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name=f'step8l_{args.config}_{args.label_source}',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        per_llm_list = args.llms if args.config == 'per_llm' else [None]

        for llm_tag in per_llm_list:
            llms_for_X = [llm_tag] if args.config == 'per_llm' else args.llms

            print(f"\n[step8l] config={args.config} label={args.label_source} "
                  f"llm={llm_tag or 'ALL'} llms_for_X={llms_for_X}")

            X_train = _build_X('train', llms_for_X)
            X_val   = _build_X('val',   llms_for_X)
            X_test  = _build_X('test',  llms_for_X)
            X_oods  = {ood: _build_X(ood, llms_for_X) for ood in OOD_SPLITS
                       if os.path.exists(os.path.join(SPLITS_DIR, f'{ood}.csv'))}

            y_train, _ = _load_labels('train', args.label_source)
            y_val,   _ = _load_labels('val',   args.label_source)
            y_test,  _ = _load_labels('test',  args.label_source)
            y_oods = {ood: _load_labels(ood, args.label_source)[0] for ood in X_oods}

            scaler = StandardScaler().fit(X_train)
            X_train_s = scaler.transform(X_train).astype(np.float32)
            X_val_s   = scaler.transform(X_val).astype(np.float32)
            X_test_s  = scaler.transform(X_test).astype(np.float32)
            X_oods_s  = {k: scaler.transform(v).astype(np.float32) for k, v in X_oods.items()}

            print(f"[step8l] dim={X_train_s.shape[1]}  train={len(y_train)} "
                  f"val={len(y_val)} test={len(y_test)} ood_sizes="
                  f"{ {k: len(v) for k, v in y_oods.items()} }")

            for seed in args.seeds:
                t0 = time.time()
                model, best_val = fit_one(X_train_s, y_train, X_val_s, y_val, seed=seed)
                test_pred, test_logits = predict(model, X_test_s)
                ood_preds  = {k: predict(model, v) for k, v in X_oods_s.items()}

                report = {
                    'config': args.config,
                    'label_source': args.label_source,
                    'llm': llm_tag,
                    'seed': seed,
                    'feature_dim': int(X_train_s.shape[1]),
                    'best_val_f1_macro': best_val,
                    'test': _eval(y_test, test_pred),
                    'ood': {ood: _eval(y_oods[ood], pr[0]) for ood, pr in ood_preds.items()},
                    'fit_seconds': round(time.time() - t0, 1),
                }

                tag = f'{args.config}__{args.label_source}__' \
                      f'{llm_tag if llm_tag else "all"}__seed{seed}'
                pkl_path  = os.path.join(MODELS_DIR, f'{tag}.pkl')
                json_path = os.path.join(EVAL_DIR,   f'{tag}.eval.json')
                with open(pkl_path, 'wb') as fh:
                    pickle.dump({
                        'model_state': model.state_dict(),
                        'feature_dim': X_train_s.shape[1],
                        'scaler': scaler,
                        'report': report,
                        'test_logits': test_logits,
                        'ood_logits': {ood: pr[1] for ood, pr in ood_preds.items()},
                    }, fh)
                with open(json_path, 'w') as fh:
                    json.dump(report, fh, indent=2)

                test_f1 = report['test']['f1_macro']
                ood_str = ' '.join(f"{ood}={report['ood'][ood]['f1_macro']:.3f}" for ood in report['ood'])
                print(f"[step8l] {tag}  val={best_val:.3f}  test={test_f1:.3f}  ood: {ood_str}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
