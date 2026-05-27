"""
Plot the AdvConcept before/after comparison.

The figure compares zero-shot Phi-3.5-mini, Pillar B+, and the augmented
Pillar B+ variant on the core metrics used in the presentation.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
REPORT_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
FIG_DIR      = os.path.join(_PROJECT_DIR, 'outputs', 'figures')


def _display_path(path):
    return os.path.relpath(path, _PROJECT_DIR)


BEFORE_KEY = 'zero_shot__phi-3.5-mini-instruct'
AFTER_KEY  = 'pillar_b_plus__phi-3.5-mini-instruct'
AUG_KEY    = 'pillar_b_plus_aug__phi-3.5-mini-instruct'

BEFORE_LABEL = 'BEFORE\nPhi-3.5-mini (zero-shot)'
AFTER_LABEL  = 'AFTER\nPhi-3.5-mini + LoRA (Pillar B+)'
AUG_LABEL    = 'AFTER + AUG\nPillar B+ with counterfactual data'

# Presentation-friendly palette.
BEFORE_COLOR = '#E76F51'  # warm coral
AFTER_COLOR  = '#2A9D8F'  # teal
AUG_COLOR    = '#264653'  # deep blue-green


def _load_eval(name):
    path = os.path.join(REPORT_DIR, f'{name}.eval.json')
    with open(path) as f:
        return json.load(f)


def _extract_metrics(r):
    """Pull the charted metrics from an eval report."""
    if 'per_split' in r and 'adv_concept' in r:
        return {
            'In-dist\ntest F1':           r['per_split']['test']['f1_macro'],
            'AdvConcept-50\noverall F1':  r['adv_concept']['overall_f1_macro'],
            'Surface easy /\nConcept hard\n(accuracy)':
                r['adv_concept']['per_category_accuracy'].get(
                    'surface_easy_concept_hard', 0.0),
            'Surface hard /\nConcept easy\n(accuracy)':
                r['adv_concept']['per_category_accuracy'].get(
                    'surface_hard_concept_easy', 0.0),
            'Surface matches\nconcept\n(accuracy)':
                r['adv_concept']['per_category_accuracy'].get(
                    'surface_matches_concept', 0.0),
        }
    raise ValueError("unexpected eval JSON schema")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    before = _extract_metrics(_load_eval(BEFORE_KEY))
    after  = _extract_metrics(_load_eval(AFTER_KEY))
    aug    = _extract_metrics(_load_eval(AUG_KEY))

    metrics = list(before.keys())
    n = len(metrics)
    b_vals = [before[m] for m in metrics]
    a_vals = [after[m]  for m in metrics]
    g_vals = [aug[m]    for m in metrics]

    fig, ax = plt.subplots(figsize=(15, 6.5))
    x = np.arange(n)
    w = 0.27

    ax.bar(x - w, b_vals, w, label=BEFORE_LABEL,
            color=BEFORE_COLOR, edgecolor='black', linewidth=0.6)
    ax.bar(x,     a_vals, w, label=AFTER_LABEL,
            color=AFTER_COLOR,  edgecolor='black', linewidth=0.6)
    ax.bar(x + w, g_vals, w, label=AUG_LABEL,
            color=AUG_COLOR,    edgecolor='black', linewidth=0.6)

    # Value labels.
    for xi, v in zip(x - w, b_vals):
        ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=9,
                 color='#7c2d12', fontweight='bold')
    for xi, v in zip(x, a_vals):
        ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=9,
                 color='#0f5e54', fontweight='bold')
    for xi, v in zip(x + w, g_vals):
        ax.text(xi, v + 0.012, f'{v:.2f}', ha='center', fontsize=9,
                 color='#0f3a44', fontweight='bold')

    # Show the augmented-model delta above each metric group.
    for xi, (a, g) in enumerate(zip(a_vals, g_vals)):
        delta = g - a
        y_top = max(b_vals[xi], a, g) + 0.09
        color = '#16a34a' if delta > 0.01 else ('#dc2626' if delta < -0.01
                                                 else '#6b7280')
        ax.text(xi, y_top, f'Δ(AUG-Pillar B+) = {delta:+.2f}',
                 ha='center', fontsize=10, color=color, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor=color, linewidth=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(0, 1.30)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Phi-3.5-mini: zero-shot vs LoRA fine-tune vs '
                  'LoRA fine-tune + counterfactual augmentation',
                  fontsize=13, fontweight='bold', pad=18)
    ax.legend(loc='upper center', fontsize=10, framealpha=0.95,
               bbox_to_anchor=(0.5, -0.18), ncol=3)
    ax.grid(axis='y', linestyle=':', alpha=0.45)
    ax.set_axisbelow(True)

    # Footer caption below the legend.
    fig.text(0.5, -0.10,
              'Green Δ = improvement   Red Δ = regression   Grey Δ = no change   |   '
              'AdvConcept-50: hand-curated 50-row adversarial set where '
              'surface readability ≠ curriculum level',
              ha='center', fontsize=9, color='#374151', style='italic')

    fig.tight_layout()
    png = os.path.join(FIG_DIR, 'adv_concept_comparison.png')
    pdf = os.path.join(FIG_DIR, 'adv_concept_comparison.pdf')
    fig.savefig(png, dpi=300, bbox_inches='tight')
    fig.savefig(pdf,           bbox_inches='tight')
    print(f"[plot] wrote {_display_path(png)}")
    print(f"[plot] wrote {_display_path(pdf)}")


if __name__ == '__main__':
    main()
