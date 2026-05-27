"""
Train the "clean labels" Pillar B ablation.

This run drops the LLM-judge labels and trains only on corpora with more direct
curriculum labels. It is useful as a sanity check: does removing surface-tagged
data help, or does it remove useful diversity?

The LoRA recipe stays the same as Pillar B+ so the label/data choice is the
main thing being tested.
"""

import argparse
import gc
import json
import math
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
    GF_OUT, ADV_CSV, LABEL_MAP, LABEL_NAMES, _balanced_subsample, _eval_block,
)
from pillar_b_tiny_teacher import (  # noqa: E402
    INSTRUCTION, build_prompt, _find_letter_token_ids,
    _predict, _eval_quick, _measure_latency, CANONICAL_SPLITS,
    MODEL_DIR, REPORT_DIR, EMISSIONS_DIR,
)

BACKBONE_SHORT = 'phi-3.5-mini-instruct'
BACKBONE_HF    = 'microsoft/Phi-3.5-mini-instruct'
KEEP_CORPORA   = {'scienceqa', 'openbookqa', 'grade-aware', 'clear'}


def _display_path(path):
    return os.path.relpath(path, _PROJECT_DIR)


def _load_clean_split(split_name):
    """Load curriculum-grounded rows and their original labels."""
    sdf = pd.read_csv(os.path.join(GF_OUT, 'splits', f'{split_name}.csv'))
    before = len(sdf)
    sdf = sdf[sdf['source_dataset'].isin(KEEP_CORPORA)].copy()
    texts_col = 'full_text' if 'full_text' in sdf.columns else 'text'
    texts = sdf[texts_col].astype(str).tolist()
    y = sdf['education_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
    print(f"[pillar_b_clean] {split_name}: {len(sdf)}/{before} rows after corpus filter")
    print(f"[pillar_b_clean]   labels: {dict(zip(LABEL_NAMES, np.bincount(y, minlength=3).tolist()))}")
    return texts, y, sdf


def train(epochs, batch_size, grad_accum, lr, lora_r, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    print(f"[pillar_b_clean] loading {BACKBONE_HF} ...")
    tok = AutoTokenizer.from_pretrained(BACKBONE_HF)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BACKBONE_HF, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    for targets in (
        ['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'],
        ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    ):
        try:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05,
                target_modules=targets, bias='none',
            )
            model = get_peft_model(model, lora_cfg)
            print(f"[pillar_b_clean] LoRA targets = {targets}")
            break
        except ValueError:
            continue
    model.print_trainable_parameters()

    # Load the filtered training data.
    train_texts, train_y, _ = _load_clean_split('train')
    val_texts,   val_y,   _ = _load_clean_split('val')

    letter_ids = _find_letter_token_ids(tok)
    print(f"[pillar_b_clean] letter_ids = {letter_ids}")
    max_length = 1280

    def _tokenize(text, y):
        prompt = build_prompt(text)
        prompt_ids = tok(prompt, add_special_tokens=False)['input_ids']
        if len(prompt_ids) > max_length - 2:
            prompt_ids = prompt_ids[-(max_length - 2):]
        letter = ['E', 'M', 'H'][y]
        ans_ids = tok(f' {letter}', add_special_tokens=False)['input_ids']
        return prompt_ids + ans_ids, [-100] * len(prompt_ids) + ans_ids

    print(f"[pillar_b_clean] tokenizing ...")
    train_items = [_tokenize(t, y) for t, y in zip(train_texts, train_y)]
    val_items   = [_tokenize(t, y) for t, y in zip(val_texts,   val_y)]
    print(f"[pillar_b_clean]   train={len(train_items)}  val={len(val_items)}")

    steps_per_epoch = math.ceil(len(train_items) / batch_size / grad_accum)
    total_opt = steps_per_epoch * epochs
    warmup = max(1, int(0.05 * total_opt))
    def _sched(s):
        if s < warmup: return s / max(1, warmup)
        p = (s - warmup) / max(1, total_opt - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=0.0)
    sched = LambdaLR(opt, _sched)
    print(f"[pillar_b_clean] {total_opt} opt-steps × {epochs} epochs, warmup={warmup}")

    pad = tok.pad_token_id
    device = next(model.parameters()).device

    def _collate(batch):
        max_len = max(len(b[0]) for b in batch)
        inp = torch.tensor([b[0] + [pad] * (max_len - len(b[0])) for b in batch],
                            dtype=torch.long, device=device)
        lab = torch.tensor([b[1] + [-100] * (max_len - len(b[1])) for b in batch],
                            dtype=torch.long, device=device)
        msk = torch.tensor([[1] * len(b[0]) + [0] * (max_len - len(b[0])) for b in batch],
                            dtype=torch.long, device=device)
        return inp, lab, msk

    eval_every = max(1, total_opt // 6)
    rng = np.random.default_rng(42)
    best_val_f1 = -1.0
    opt_step = 0
    accum_loss = 0.0
    t0 = time.time()
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(train_items)).tolist()
        for i in range(0, len(train_items), batch_size):
            batch = [train_items[order[k]] for k in range(i, min(i + batch_size, len(train_items)))]
            if not batch:
                continue
            inp, lab, msk = _collate(batch)
            out = model(input_ids=inp, labels=lab, attention_mask=msk)
            loss = out.loss / grad_accum
            loss.backward()
            accum_loss += loss.item()
            if ((i // batch_size) + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                opt_step += 1
                if opt_step % 10 == 0:
                    elapsed = time.time() - t0
                    rate = opt_step / max(elapsed, 1.0)
                    eta = (total_opt - opt_step) / max(rate, 1e-6)
                    print(f"[pillar_b_clean]   step {opt_step}/{total_opt}  "
                          f"loss={accum_loss:.3f}  lr={sched.get_last_lr()[0]:.2e}  "
                          f"{rate:.2f}step/s  eta={eta/60:.1f}min")
                accum_loss = 0.0
                if opt_step % eval_every == 0:
                    vf1 = _eval_quick(model, tok, val_items[:256], letter_ids, batch_size=8)
                    if vf1 > best_val_f1:
                        best_val_f1 = vf1
                        os.makedirs(out_dir, exist_ok=True)
                        model.save_pretrained(os.path.join(out_dir, 'best'))
                        tok.save_pretrained(os.path.join(out_dir, 'best'))
                        print(f"[pillar_b_clean]   val_f1@step{opt_step} = {vf1:.4f}  (saved best)")
                    else:
                        print(f"[pillar_b_clean]   val_f1@step{opt_step} = {vf1:.4f}  (best {best_val_f1:.4f})")
                    model.train()

    final_vf1 = _eval_quick(model, tok, val_items, letter_ids, batch_size=8)
    print(f"[pillar_b_clean] final full-val F1 = {final_vf1:.4f}")
    if final_vf1 > best_val_f1:
        best_val_f1 = final_vf1
        model.save_pretrained(os.path.join(out_dir, 'best'))
        tok.save_pretrained(os.path.join(out_dir, 'best'))
        print(f"[pillar_b_clean]   saved best (final)")
    print(f"[pillar_b_clean] train wall-clock: {(time.time() - t0)/60:.1f} min")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return best_val_f1, time.time() - t0


def evaluate(out_dir, train_sec, best_val_f1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n[pillar_b_clean] reloading best adapter for eval ...")
    tok = AutoTokenizer.from_pretrained(BACKBONE_HF)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BACKBONE_HF, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(base, os.path.join(out_dir, 'best')).to('cuda').eval()
    letter_ids = _find_letter_token_ids(tok)

    # Evaluate on full splits so this ablation remains comparable cross-domain.
    eval_splits = {}
    for s in CANONICAL_SPLITS:
        sdf = pd.read_csv(os.path.join(GF_OUT, 'splits', f'{s}.csv'))
        texts_col = 'full_text' if 'full_text' in sdf.columns else 'text'
        texts = sdf[texts_col].astype(str).tolist()
        y = sdf['education_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
        eval_splits[s] = (texts, y)
        print(f"[pillar_b_clean]   eval {s}: n={len(texts)} (vs ORIGINAL labels)")

    per_split = {}
    for split, (texts, y) in eval_splits.items():
        pred = _predict(model, tok, texts, letter_ids, batch_size=8)
        per_split[split] = _eval_block(y, pred)
        print(f"[pillar_b_clean]   {split}: f1_macro = {per_split[split]['f1_macro']:.4f}")
        if split == 'test':
            # Balanced subsample with 5 seeds, 50/class
            f1s = []
            for seed in range(5):
                _, y_bal, idx_bal = _balanced_subsample(texts, y, per_class=50,
                                                        seed=42 + seed)
                f1s.append(float(f1_score(y_bal, pred[idx_bal], average='macro',
                                           labels=[0, 1, 2], zero_division=0)))
            per_split['balanced_test'] = {
                'f1_macro_mean': float(np.mean(f1s)),
                'f1_macro_std':  float(np.std(f1s)),
                'f1_macro_per_seed': f1s,
                'n_resamples': 5, 'per_class_n': 50,
            }
            print(f"[pillar_b_clean]   balanced_test = {per_split['balanced_test']['f1_macro_mean']:.4f}")
            # Also report test on JUST the curriculum-grounded corpora
            sdf = pd.read_csv(os.path.join(GF_OUT, 'splits', 'test.csv'))
            mask = sdf['source_dataset'].isin(KEEP_CORPORA).to_numpy()
            y_clean = y[mask]
            pred_clean = pred[mask]
            per_split['test_clean_corpora_only'] = _eval_block(y_clean, pred_clean)
            print(f"[pillar_b_clean]   test (clean corpora only): "
                  f"f1_macro = {per_split['test_clean_corpora_only']['f1_macro']:.4f}  "
                  f"n={len(y_clean)}")

    # AdvConcept-50
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
        {'idx': int(r['idx']), 'text': r['text'], 'true_level': r['true_level'],
         'predicted_level': LABEL_NAMES[int(pred_adv[i])], 'category': r['category'],
         'correct': bool(LABEL_NAMES[int(pred_adv[i])] == r['true_level'])}
        for i, r in adv_df.iterrows()
    ]
    adv_report = {
        'overall_accuracy': adv_block['accuracy'],
        'overall_f1_macro': adv_block['f1_macro'],
        'per_category_accuracy': cat_acc,
        'conf_matrix': adv_block['conf_matrix'],
        'per_row': per_row,
    }
    print(f"[pillar_b_clean]   adv_concept overall = {adv_block['accuracy']:.4f}")
    for cat, acc in cat_acc.items():
        print(f"[pillar_b_clean]     {cat}: {acc:.4f}")

    latency = _measure_latency(model, tok, eval_splits['test'][0][:60], letter_ids)
    n_total = sum(p.numel() for p in model.parameters())
    for n, p in model.named_parameters():
        if 'lora' in n.lower():
            p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = {
        'kind':           'pillar_b_clean',
        'name':           f'pillar_b_clean__{BACKBONE_SHORT}',
        'backbone_short': BACKBONE_SHORT,
        'backbone_hf':    BACKBONE_HF,
        'n_params_total': int(n_total),
        'n_params_trainable_lora': int(n_trainable),
        'best_val_f1':    best_val_f1,
        'train_sec':      round(train_sec, 1),
        'training_corpora': sorted(KEEP_CORPORA),
        'label_source':   'original (no LLM-judge)',
        'per_split':      per_split,
        'adv_concept':    adv_report,
        'latency':        latency,
    }
    out_path = os.path.join(REPORT_DIR, f'pillar_b_clean__{BACKBONE_SHORT}.eval.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"[pillar_b_clean] wrote {_display_path(out_path)}")
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--lora-r', type=int, default=32)
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)
    out_dir = os.path.join(MODEL_DIR, f'pillar_b_clean__{BACKBONE_SHORT}__lora')

    tracker = EmissionsTracker(project_name='pillar_b_clean',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        best_val_f1, train_sec = train(args.epochs, args.batch_size,
                                         args.grad_accum, args.lr,
                                         args.lora_r, out_dir)
        rep = evaluate(out_dir, train_sec, best_val_f1)
        synth_gen = rep['per_split'].get('ood_synthetic', {}).get('f1_macro', 0.0)
        adv = rep['adv_concept']['overall_accuracy']
        print(f"\n[pillar_b_clean] FINAL — synth_vs_gen F1 = {synth_gen:.4f}  "
              f"adv_concept = {adv:.4f}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
