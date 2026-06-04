"""
plot_results.py — generate the 4 paper figures.

Fig 1: per-corpus original-vs-judge label agreement.
Fig 2: macro-F1 progression bar chart on held-out OneStop.
Fig 3: K-shot calibration curve.
Fig 4: 3x3 confusion matrix on held-out OneStop (best config).

Reads from outputs/eval_reports/ and outputs/models/. Writes PNGs to
outputs/figures/.
"""

import glob
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
EVAL_DIR     = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
MODELS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'models')
FIG_DIR      = os.path.join(_PROJECT_DIR, 'outputs', 'figures')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

LABEL_NAMES = ['elementary', 'middle', 'high']


def fig1_disagreement():
    csv = os.path.join(EVAL_DIR, 'label_disagreement_by_corpus.csv')
    if not os.path.exists(csv):
        print('[plot] skip Fig 1, missing label_disagreement_by_corpus.csv')
        return
    df = pd.read_csv(csv).sort_values('agreement').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 3.5), dpi=150)
    ax.barh(df['source_dataset'], df['agreement'], color='#3a7bd5')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Agreement: original corpus label vs LLM-judge label')
    ax.set_title('Per-corpus label inconsistency (lower = more disagreement)')
    for i, v in enumerate(df['agreement']):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_label_disagreement.png')
    plt.savefig(path); plt.close()
    print(f'[plot] wrote {path}')


def _aggregate_f1(label_source, config, llm_or_all, splits=('ood_onestop',)):
    f1s_by_split = {s: [] for s in splits}
    pattern = os.path.join(EVAL_DIR, f'{config}__{label_source}__{llm_or_all}__seed*.eval.json')
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            d = json.load(fh)
        for s in splits:
            if s in d.get('ood', {}):
                f1s_by_split[s].append(d['ood'][s]['f1_macro'])
    return {s: (float(np.mean(v)), float(np.std(v)), len(v))
            for s, v in f1s_by_split.items() if v}


def fig2_macroF1_progression():
    onestop_orig  = _aggregate_f1('original',  'all_llms', 'all', ('ood_onestop',))
    onestop_judge = _aggregate_f1('llm_judge', 'all_llms', 'all', ('ood_onestop',))

    calib_path = os.path.join(EVAL_DIR, 'calibration_sweep.json')
    judge_50shot = None
    judge_oracle = None
    if os.path.exists(calib_path):
        with open(calib_path) as fh:
            c = json.load(fh)
        if 'ood_onestop' in c.get('by_ood', {}):
            for r in c['by_ood']['ood_onestop']:
                if r['K'] == 50 and r['mean_f1'] is not None:
                    judge_50shot = (r['mean_f1'], r['std_f1'])
                if r['K'] == 'full_80pct_oracle' and r['mean_f1'] is not None:
                    judge_oracle = (r['mean_f1'], r['std_f1'])

    bars = []
    if 'ood_onestop' in onestop_orig:
        m, s, _ = onestop_orig['ood_onestop']
        bars.append(('Original labels\n(baseline)', m, s))
    if 'ood_onestop' in onestop_judge:
        m, s, _ = onestop_judge['ood_onestop']
        bars.append(('+ LLM-judge\nunified labels', m, s))
    if judge_50shot is not None:
        bars.append(('+ 50-shot\ncalibration', judge_50shot[0], judge_50shot[1]))
    if judge_oracle is not None:
        bars.append(('Oracle:\n80% target\ntraining', judge_oracle[0], judge_oracle[1]))

    if not bars:
        print('[plot] skip Fig 2, no eval reports yet')
        return
    labels = [b[0] for b in bars]
    means  = [b[1] for b in bars]
    stds   = [b[2] for b in bars]
    colors = ['#cccccc', '#3a7bd5', '#5fa055', '#888888'][:len(bars)]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    pos = np.arange(len(bars))
    ax.bar(pos, means, yerr=stds, color=colors, capsize=4, edgecolor='black')
    ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.axhline(0.8, color='red', linestyle='--', linewidth=0.7, alpha=0.6)
    ax.text(len(bars) - 0.5, 0.81, 'target 0.80', color='red', fontsize=8, ha='right')
    ax.set_ylabel('Macro-F1 on held-out OneStopEnglish')
    ax.set_title('Cross-corpus generalization: progression of macro-F1')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.02, f"{m:.2f}", ha='center', fontsize=9)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_macroF1_progression.png')
    plt.savefig(path); plt.close()
    print(f'[plot] wrote {path}')


def fig3_calibration_curve():
    calib_path = os.path.join(EVAL_DIR, 'calibration_sweep.json')
    if not os.path.exists(calib_path):
        print('[plot] skip Fig 3, missing calibration_sweep.json')
        return
    with open(calib_path) as fh:
        c = json.load(fh)
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
    for ood, rows in c.get('by_ood', {}).items():
        K_vals, means, stds = [], [], []
        for r in rows:
            if not isinstance(r['K'], int) or r['mean_f1'] is None:
                continue
            K_vals.append(r['K']); means.append(r['mean_f1']); stds.append(r['std_f1'] or 0.0)
        if K_vals:
            ax.errorbar(K_vals, means, yerr=stds, marker='o', capsize=3, label=ood)
    ax.set_xlabel('K (labeled examples held out for calibration)')
    ax.set_ylabel('Macro-F1 on remaining held-out')
    ax.set_title('Per-corpus K-shot calibration')
    ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_calibration_curve.png')
    plt.savefig(path); plt.close()
    print(f'[plot] wrote {path}')


def fig4_confusion_matrix():
    best = sorted(glob.glob(os.path.join(EVAL_DIR, 'all_llms__llm_judge__all__seed*.eval.json')))
    if not best:
        print('[plot] skip Fig 4, no all_llms__llm_judge bundle yet')
        return
    # Pick the one with highest OOD onestop F1.
    best_path, best_score = None, -1.0
    for f in best:
        with open(f) as fh:
            d = json.load(fh)
        score = d.get('ood', {}).get('ood_onestop', {}).get('f1_macro', -1.0)
        if score > best_score:
            best_score, best_path = score, f
    if best_path is None:
        return
    with open(best_path) as fh:
        d = json.load(fh)
    cm = np.array(d['ood']['ood_onestop']['conf_matrix'])
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.2, 4), dpi=150)
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(LABEL_NAMES, fontsize=9)
    ax.set_yticks(range(3)); ax.set_yticklabels(LABEL_NAMES, fontsize=9)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True (LLM-judge)')
    ax.set_title(f'Held-out OneStop confusion (F1={best_score:.2f})')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha='center', va='center',
                    color='white' if norm[i, j] > 0.5 else 'black', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_confusion_matrix.png')
    plt.savefig(path); plt.close()
    print(f'[plot] wrote {path}')


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)
    tracker = EmissionsTracker(project_name='plot_results',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        fig1_disagreement()
        fig2_macroF1_progression()
        fig3_calibration_curve()
        fig4_confusion_matrix()
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
