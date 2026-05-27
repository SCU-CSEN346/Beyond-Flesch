"""
Evaluate the base models before LoRA fine-tuning.

These runs use the same prompt format and E/M/H logit readout as Pillar B+,
but with no adapter. That gives a clean before/after comparison for the slides.
"""

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from sklearn.metrics import accuracy_score, f1_score

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
from evaluate_all import (  # noqa: E402
    ADV_CSV, LABEL_MAP, LABEL_NAMES, _balanced_subsample, _eval_block,
)
from pillar_b_tiny_teacher import (  # noqa: E402
    CANONICAL_SPLITS, _find_letter_token_ids, _load_split, _measure_latency,
    _predict, REPORT_DIR, EMISSIONS_DIR,
)


def _display_path(path):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.relpath(path, project_dir)


# Short names match the saved eval report convention.
DEFAULT_BACKBONES = [
    ('phi-3.5-mini-instruct', 'microsoft/Phi-3.5-mini-instruct'),
    ('qwen-2.5-3b-instruct',  'Qwen/Qwen2.5-3B-Instruct'),
    ('gemma-2-2b-it',         'google/gemma-2-2b-it'),
]


def evaluate_zero_shot(backbone_short, hf_id):
    """Run the full eval matrix with the base model only."""
    print(f"\n[zero_shot] ==== {backbone_short} ({hf_id}) ====")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model.eval()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[zero_shot]   total params: {n_total:,}")

    letter_ids = _find_letter_token_ids(tok)
    print(f"[zero_shot]   letter_ids = {letter_ids}")

    # Shared split list.
    per_split = {}
    eval_texts_for_latency = None
    for split in CANONICAL_SPLITS:
        texts, y = _load_split(split, 'judge')
        t0 = time.time()
        pred = _predict(model, tok, texts, letter_ids, batch_size=8)
        per_split[split] = _eval_block(y, pred)
        per_split[split]['eval_sec'] = round(time.time() - t0, 1)
        print(f"[zero_shot]   {split}: f1_macro = {per_split[split]['f1_macro']:.4f}")
        if split == 'test':
            eval_texts_for_latency = texts[:60]
            f1s = []
            for s_seed in range(5):
                _, y_bal, idx_bal = _balanced_subsample(texts, y, per_class=50,
                                                        seed=42 + s_seed)
                f1s.append(float(f1_score(y_bal, pred[idx_bal], average='macro',
                                           labels=[0, 1, 2], zero_division=0)))
            per_split['balanced_test'] = {
                'f1_macro_mean':     float(np.mean(f1s)),
                'f1_macro_std':      float(np.std(f1s)),
                'f1_macro_per_seed': f1s,
                'n_resamples': 5, 'per_class_n': 50,
            }
            print(f"[zero_shot]   balanced_test = "
                  f"{per_split['balanced_test']['f1_macro_mean']:.4f}")

    # AdvConcept evaluation.
    adv_df = pd.read_csv(ADV_CSV)
    adv_texts = adv_df['text'].astype(str).tolist()
    adv_y = adv_df['true_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
    pred_adv = _predict(model, tok, adv_texts, letter_ids, batch_size=8)
    adv_block = _eval_block(adv_y, pred_adv)
    cat_acc = {cat: float(accuracy_score(
        adv_y[(adv_df['category'] == cat).to_numpy()],
        pred_adv[(adv_df['category'] == cat).to_numpy()]))
        for cat in adv_df['category'].unique()}
    per_row = [
        {'idx': int(i), 'text': r['text'], 'true_level': r['true_level'],
         'predicted_level': LABEL_NAMES[int(pred_adv[i])], 'category': r['category'],
         'correct': bool(LABEL_NAMES[int(pred_adv[i])] == r['true_level'])}
        for i, r in adv_df.iterrows()
    ]
    adv_report = {
        'overall_accuracy':      adv_block['accuracy'],
        'overall_f1_macro':      adv_block['f1_macro'],
        'per_category_accuracy': cat_acc,
        'conf_matrix':           adv_block['conf_matrix'],
        'per_row':               per_row,
    }
    print(f"[zero_shot]   adv_concept = "
          f"acc {adv_block['accuracy']:.4f} / f1 {adv_block['f1_macro']:.4f}")
    for cat, acc in cat_acc.items():
        print(f"[zero_shot]     {cat}: {acc:.4f}")

    latency = _measure_latency(model, tok, eval_texts_for_latency, letter_ids)

    report = {
        'kind':           'zero_shot',
        'name':           f'zero_shot__{backbone_short}',
        'backbone':       backbone_short,
        'hf_id':          hf_id,
        'n_params_total': n_total,
        'n_params_trained': 0,
        'per_split':      per_split,
        'adv_concept':    adv_report,
        'latency':        latency,
    }

    # Free memory before the next backbone.
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', type=str, default=None,
                    help='comma-separated short-name filter; default: all three')
    args = ap.parse_args()

    backbones = DEFAULT_BACKBONES
    if args.backbones:
        wanted = set(s.strip() for s in args.backbones.split(','))
        backbones = [(s, h) for (s, h) in DEFAULT_BACKBONES if s in wanted]
        if not backbones:
            raise ValueError(f"No matching backbones in {wanted}. "
                              f"Available: {[s for s,_ in DEFAULT_BACKBONES]}")

    os.makedirs(REPORT_DIR,   exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='zero_shot_baselines',
                                output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        summary = []
        for short, hf_id in backbones:
            report = evaluate_zero_shot(short, hf_id)
            out_path = os.path.join(REPORT_DIR, f'zero_shot__{short}.eval.json')
            with open(out_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"[zero_shot] wrote {_display_path(out_path)}")
            summary.append({
                'backbone':     short,
                'test_f1':      report['per_split']['test']['f1_macro'],
                'bal_test_f1':  report['per_split']['balanced_test']['f1_macro_mean'],
                'adv_acc':      report['adv_concept']['overall_accuracy'],
                'adv_f1':       report['adv_concept']['overall_f1_macro'],
                'latency_ms':   report['latency']['median_ms'],
            })
    finally:
        kwh = tracker.stop()
        print(f"[zero_shot] emissions kWh = {kwh}")

    print("\n[zero_shot] ============== SUMMARY ==============")
    print(f"{'backbone':30s}  {'test_f1':>8s}  {'bal_test':>8s}  "
          f"{'adv_acc':>8s}  {'adv_f1':>8s}  {'lat_ms':>7s}")
    for row in summary:
        print(f"{row['backbone']:30s}  {row['test_f1']:8.4f}  "
              f"{row['bal_test_f1']:8.4f}  {row['adv_acc']:8.4f}  "
              f"{row['adv_f1']:8.4f}  {row['latency_ms']:7.1f}")


if __name__ == '__main__':
    main()
