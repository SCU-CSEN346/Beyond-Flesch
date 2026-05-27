"""
Train the hybrid-label Pillar B ablation.

For curriculum-tagged corpora, this run uses the original label. For the rest,
it keeps the LLM-judge label. The goal is to see whether we can keep corpus
diversity while reducing some judge-label surface bias.

The training recipe matches Pillar B+ so the label strategy is the main change.
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
from pillar_b_plus import _class_weighted_indices  # noqa: E402

BACKBONE_SHORT = 'phi-3.5-mini-instruct'
BACKBONE_HF    = 'microsoft/Phi-3.5-mini-instruct'
# Curriculum-tagged corpora — for these we trust the original label.
# Everything else uses the LLM-judge label.
CURRICULUM_CORPORA = {'scienceqa', 'openbookqa', 'grade-aware'}


def _display_path(path):
    return os.path.relpath(path, _PROJECT_DIR)


def _load_hybrid(split):
    """Load a split and apply the hybrid label rule."""
    s = pd.read_csv(os.path.join(GF_OUT, 'splits', f'{split}.csv'))
    j = pd.read_csv(os.path.join(GF_OUT, 'llm_judge', f'{split}_judge.csv'))
    assert len(s) == len(j), f'{split}: row mismatch'
    s = s.reset_index(drop=True)
    s['llm_judge_label'] = j['llm_judge_label'].values
    s['hybrid_label'] = np.where(
        s['source_dataset'].isin(CURRICULUM_CORPORA),
        s['education_level'],
        s['llm_judge_label'],
    )
    texts_col = 'full_text' if 'full_text' in s.columns else 'text'
    texts = s[texts_col].astype(str).tolist()
    y = s['hybrid_label'].map(LABEL_MAP).astype(np.int64).to_numpy()
    return texts, y, s


def train(epochs, batch_size, grad_accum, lr, lora_r, target_per_class, out_dir):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    print(f"[pillar_b_hybrid] loading {BACKBONE_HF} ...")
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
            print(f"[pillar_b_hybrid] LoRA targets = {targets}")
            break
        except ValueError:
            continue
    model.print_trainable_parameters()

    train_texts, train_y, _ = _load_hybrid('train')
    val_texts,   val_y,   _ = _load_hybrid('val')
    print(f"[pillar_b_hybrid] train={len(train_texts)} val={len(val_texts)}")
    print(f"[pillar_b_hybrid] train hybrid label dist: "
          f"{dict(zip(LABEL_NAMES, np.bincount(train_y, minlength=3).tolist()))}")

    # Same class-weighted sampling recipe as Pillar B+.
    epoch_indices = []
    for ep in range(epochs):
        idx = _class_weighted_indices(train_y, target_per_class=target_per_class,
                                       seed=42 + ep)
        from collections import Counter
        cnt = Counter(train_y[idx].tolist())
        print(f"[pillar_b_hybrid]   epoch {ep} sample: {dict(cnt)}  (total {len(idx)})")
        epoch_indices.append(idx)

    letter_ids = _find_letter_token_ids(tok)
    print(f"[pillar_b_hybrid] letter_ids = {letter_ids}")
    max_length = 1280

    def _tokenize(text, y):
        prompt = build_prompt(text)
        prompt_ids = tok(prompt, add_special_tokens=False)['input_ids']
        if len(prompt_ids) > max_length - 2:
            prompt_ids = prompt_ids[-(max_length - 2):]
        letter = ['E', 'M', 'H'][y]
        ans_ids = tok(f' {letter}', add_special_tokens=False)['input_ids']
        return prompt_ids + ans_ids, [-100] * len(prompt_ids) + ans_ids

    print("[pillar_b_hybrid] tokenizing val ...")
    val_items = [_tokenize(t, y) for t, y in zip(val_texts, val_y)]
    print("[pillar_b_hybrid] tokenizing per-epoch sampled train ...")
    epoch_train_items = []
    for ep, idx in enumerate(epoch_indices):
        items = [_tokenize(train_texts[i], train_y[i]) for i in idx]
        epoch_train_items.append(items)
        print(f"[pillar_b_hybrid]   epoch {ep}: tokenized {len(items)}")

    total_micro = sum(math.ceil(len(items) / batch_size) for items in epoch_train_items)
    total_opt = math.ceil(total_micro / grad_accum)
    warmup = max(1, int(0.05 * total_opt))
    def _sched(s):
        if s < warmup: return s / max(1, warmup)
        p = (s - warmup) / max(1, total_opt - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=0.0)
    sched = LambdaLR(opt, _sched)
    print(f"[pillar_b_hybrid] {total_opt} opt-steps, warmup={warmup}")

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
    best_val_f1 = -1.0
    opt_step = 0
    accum_loss = 0.0
    t0 = time.time()
    model.train()
    for ep, items in enumerate(epoch_train_items):
        rng = np.random.default_rng(42 + ep)
        order = rng.permutation(len(items)).tolist()
        for i in range(0, len(items), batch_size):
            batch = [items[order[k]] for k in range(i, min(i + batch_size, len(items)))]
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
                    print(f"[pillar_b_hybrid]   step {opt_step}/{total_opt}  "
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
                        print(f"[pillar_b_hybrid]   val_f1@step{opt_step} = {vf1:.4f}  (saved best)")
                    else:
                        print(f"[pillar_b_hybrid]   val_f1@step{opt_step} = {vf1:.4f}  (best {best_val_f1:.4f})")
                    model.train()

    final_vf1 = _eval_quick(model, tok, val_items, letter_ids, batch_size=8)
    print(f"[pillar_b_hybrid] final full-val F1 = {final_vf1:.4f}")
    if final_vf1 > best_val_f1:
        best_val_f1 = final_vf1
        model.save_pretrained(os.path.join(out_dir, 'best'))
        tok.save_pretrained(os.path.join(out_dir, 'best'))
        print(f"[pillar_b_hybrid]   saved best (final)")
    print(f"[pillar_b_hybrid] train wall-clock: {(time.time() - t0)/60:.1f} min")
    del model
    gc.collect(); torch.cuda.empty_cache()
    return best_val_f1, time.time() - t0


def evaluate(out_dir, train_sec, best_val_f1):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n[pillar_b_hybrid] reloading best adapter for eval ...")
    tok = AutoTokenizer.from_pretrained(BACKBONE_HF)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BACKBONE_HF, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model = PeftModel.from_pretrained(base, os.path.join(out_dir, 'best')).to('cuda').eval()
    letter_ids = _find_letter_token_ids(tok)

    # Use the same eval matrix as Pillar B+.
    eval_splits = {}
    for s in CANONICAL_SPLITS:
        sdf = pd.read_csv(os.path.join(GF_OUT, 'splits', f'{s}.csv'))
        texts_col = 'full_text' if 'full_text' in sdf.columns else 'text'
        texts = sdf[texts_col].astype(str).tolist()
        jdf = pd.read_csv(os.path.join(GF_OUT, 'llm_judge', f'{s}_judge.csv'))
        y = jdf['llm_judge_label'].map(LABEL_MAP).astype(np.int64).to_numpy()
        eval_splits[s] = (texts, y)
        print(f"[pillar_b_hybrid]   eval {s}: n={len(texts)}")

    per_split = {}
    for split, (texts, y) in eval_splits.items():
        pred = _predict(model, tok, texts, letter_ids, batch_size=8)
        per_split[split] = _eval_block(y, pred)
        print(f"[pillar_b_hybrid]   {split}: f1_macro = {per_split[split]['f1_macro']:.4f}")
        if split == 'test':
            f1s = []
            for seed in range(5):
                _, y_bal, idx_bal = _balanced_subsample(texts, y, per_class=50,
                                                        seed=42 + seed)
                f1s.append(float(f1_score(y_bal, pred[idx_bal], average='macro',
                                           labels=[0,1,2], zero_division=0)))
            per_split['balanced_test'] = {
                'f1_macro_mean': float(np.mean(f1s)),
                'f1_macro_std': float(np.std(f1s)),
                'f1_macro_per_seed': f1s,
                'n_resamples': 5, 'per_class_n': 50,
            }
            print(f"[pillar_b_hybrid]   balanced_test = {per_split['balanced_test']['f1_macro_mean']:.4f}")
        if split == 'ood_synthetic':
            split_csv = os.path.join(GF_OUT, 'splits', 'ood_synthetic.csv')
            sdf = pd.read_csv(split_csv)
            y_gen = sdf['education_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
            per_split['ood_synthetic_vs_gen'] = _eval_block(y_gen, pred)
            print(f"[pillar_b_hybrid]   ood_synthetic_vs_gen = "
                  f"{per_split['ood_synthetic_vs_gen']['f1_macro']:.4f}")

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
    print(f"[pillar_b_hybrid]   adv_concept overall = {adv_block['accuracy']:.4f}")
    for cat, acc in cat_acc.items():
        print(f"[pillar_b_hybrid]     {cat}: {acc:.4f}")

    latency = _measure_latency(model, tok, eval_splits['test'][0][:60], letter_ids)
    n_total = sum(p.numel() for p in model.parameters())
    for n, p in model.named_parameters():
        if 'lora' in n.lower():
            p.requires_grad_(True)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    report = {
        'kind':           'pillar_b_hybrid',
        'name':           f'pillar_b_hybrid__{BACKBONE_SHORT}',
        'backbone_short': BACKBONE_SHORT,
        'backbone_hf':    BACKBONE_HF,
        'n_params_total': int(n_total),
        'n_params_trainable_lora': int(n_trainable),
        'best_val_f1':    best_val_f1,
        'train_sec':      round(train_sec, 1),
        'label_source':   'hybrid (original for ScienceQA/OpenBookQA/grade-aware, judge for others)',
        'curriculum_corpora': sorted(CURRICULUM_CORPORA),
        'per_split':      per_split,
        'adv_concept':    adv_report,
        'latency':        latency,
    }
    out_path = os.path.join(REPORT_DIR, f'pillar_b_hybrid__{BACKBONE_SHORT}.eval.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"[pillar_b_hybrid] wrote {_display_path(out_path)}")
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
    ap.add_argument('--target-per-class', type=int, default=3500)
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)
    out_dir = os.path.join(MODEL_DIR, f'pillar_b_hybrid__{BACKBONE_SHORT}__lora')

    tracker = EmissionsTracker(project_name='pillar_b_hybrid',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        best_val_f1, train_sec = train(args.epochs, args.batch_size,
                                         args.grad_accum, args.lr,
                                         args.lora_r, args.target_per_class,
                                         out_dir)
        rep = evaluate(out_dir, train_sec, best_val_f1)
        synth_gen = rep['per_split'].get('ood_synthetic_vs_gen', {}).get('f1_macro', 0.0)
        adv = rep['adv_concept']['overall_accuracy']
        adv_hard = rep['adv_concept']['per_category_accuracy'].get('surface_hard_concept_easy', 0)
        print(f"\n[pillar_b_hybrid] FINAL — synth_vs_gen = {synth_gen:.4f}  "
              f"adv = {adv:.4f}  surface_hard_concept_easy = {adv_hard:.4f}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
