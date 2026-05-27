"""
Run the Pillar B small-LLM bake-off.

Each backbone is LoRA-tuned to answer with one curriculum label: E, M, or H.
At inference time we read the next-token logits for those letters instead of
generating free-form text.

This script is the quick comparison run that chose the Phi-3.5-mini backbone
for the final Pillar B+ recipe.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
from evaluate_all import (  # noqa: E402
    GF_OUT, ADV_CSV, LABEL_MAP, LABEL_NAMES, _balanced_subsample, _eval_block,
)

MODEL_DIR  = os.path.join(_PROJECT_DIR, 'outputs', 'models')
REPORT_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')


def _display_path(path):
    return os.path.relpath(path, _PROJECT_DIR)


# Short name, Hugging Face id, rough parameter count in billions.
BACKBONES = [
    ('llama-3.2-3b-instruct', 'meta-llama/Llama-3.2-3B-Instruct',  3.2),
    ('qwen-2.5-3b-instruct',  'Qwen/Qwen2.5-3B-Instruct',          3.1),
    ('phi-3.5-mini-instruct', 'microsoft/Phi-3.5-mini-instruct',   3.8),
    ('gemma-2-2b-it',         'google/gemma-2-2b-it',              2.6),
]

INSTRUCTION = (
    "Read the following text and classify it by the curriculum grade level "
    "required to understand its CONCEPTS (not just its reading complexity). "
    "Answer with only one letter: E for elementary school (US grades 1-5), "
    "M for middle school (US grades 6-8), H for high school (US grades 9-12)."
)
LETTER_FOR_LEVEL = {'elementary': 'E', 'middle': 'M', 'high': 'H'}
LEVEL_FOR_LETTER = {v: k for k, v in LETTER_FOR_LEVEL.items()}
LETTER_INDEX = {'E': 0, 'M': 1, 'H': 2}

MAX_TEXT_TOKENS = 1024   # text portion truncation
CANONICAL_SPLITS = ['test', 'ood_synthetic', 'ood_race-middle',
                    'ood_race-high', 'ood_onestop']


def build_prompt(text):
    return f"{INSTRUCTION}\n\nText: {text}\n\nAnswer:"


def _load_split(split_name, label_source='judge'):
    split_csv = os.path.join(GF_OUT, 'splits', f'{split_name}.csv')
    sdf = pd.read_csv(split_csv)
    texts_col = 'text' if 'text' in sdf.columns else 'full_text'
    texts = sdf[texts_col].astype(str).tolist()
    if label_source == 'judge':
        jdf = pd.read_csv(os.path.join(GF_OUT, 'llm_judge', f'{split_name}_judge.csv'))
        y = jdf['llm_judge_label'].map(LABEL_MAP).astype(np.int64).to_numpy()
    elif label_source == 'gen':
        y = sdf['education_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
    else:
        raise ValueError(label_source)
    return texts, y


def _find_letter_token_ids(tok):
    """Find one usable token id for each E/M/H label."""
    out = {}
    for letter in 'EMH':
        candidates = [f' {letter}', letter]
        chosen = None
        for c in candidates:
            ids = tok(c, add_special_tokens=False)['input_ids']
            if len(ids) == 1:
                chosen = ids[0]
                break
        if chosen is None:
            # Fallback for tokenizers that split the label.
            ids = tok(f' {letter}', add_special_tokens=False)['input_ids']
            chosen = ids[0]
            print(f"[pillar_b] WARNING {letter} is multi-token in this tokenizer "
                  f"(ids={ids}); using first token id {chosen}")
        out[letter] = chosen
    return out  # e.g. {'E': 468, 'M': 469, 'H': 470}


def _train_one_backbone(backbone_short, hf_id, tokenizer, model,
                        train_texts, train_y, val_texts, val_y,
                        out_adapter_dir, epochs=1, lr=2e-4,
                        batch_size=4, grad_accum=4, max_length=1280):
    """Fine-tune one backbone and save the best validation checkpoint."""
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    device = next(model.parameters()).device
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Start with the common projection names.
    lora_targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj']

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=lora_targets, bias='none',
    )
    try:
        model = get_peft_model(model, lora_cfg)
    except ValueError as e:
        # Some backbones use different projection names.
        print(f"[pillar_b] LoRA target mismatch ({e}). Retrying with fallback targets.")
        for fallback in (['qkv_proj', 'o_proj', 'gate_up_proj', 'down_proj'],
                          ['Wqkv', 'out_proj', 'fc1', 'fc2'],
                          ['q_proj', 'k_proj', 'v_proj']):
            try:
                lora_cfg = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=16, lora_alpha=32, lora_dropout=0.05,
                    target_modules=fallback, bias='none',
                )
                model = get_peft_model(model, lora_cfg)
                print(f"[pillar_b]   succeeded with target_modules={fallback}")
                break
            except ValueError:
                continue
        else:
            raise
    model.print_trainable_parameters()

    # Pre-tokenize so the training loop only batches tensors.
    print(f"[pillar_b] pre-tokenizing train ({len(train_texts)}) ...")
    letter_ids = _find_letter_token_ids(tokenizer)
    print(f"[pillar_b]   letter_ids = {letter_ids}")

    def _tokenize_pair(text, y):
        prompt = build_prompt(text)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
        # Keep the tail where the instruction answer lives.
        if len(prompt_ids) > max_length - 2:
            # keep the tail of the prompt (more relevant: "Answer:")
            prompt_ids = prompt_ids[-(max_length - 2):]
        # Construct the answer letter ids - we use the leading-space form.
        letter = ['E', 'M', 'H'][y]
        ans_ids = tokenizer(f' {letter}', add_special_tokens=False)['input_ids']
        full = prompt_ids + ans_ids
        labels = [-100] * len(prompt_ids) + ans_ids
        return full, labels

    train_items = [_tokenize_pair(t, y) for t, y in zip(train_texts, train_y)]
    val_items   = [_tokenize_pair(t, y) for t, y in zip(val_texts,   val_y)]
    print(f"[pillar_b]   train={len(train_items)} val={len(val_items)}")

    def _collate(batch):
        max_len = max(len(b[0]) for b in batch)
        pad = tokenizer.pad_token_id
        inp = torch.tensor([b[0] + [pad] * (max_len - len(b[0])) for b in batch],
                           dtype=torch.long, device=device)
        lab = torch.tensor([b[1] + [-100] * (max_len - len(b[1])) for b in batch],
                           dtype=torch.long, device=device)
        msk = torch.tensor([[1] * len(b[0]) + [0] * (max_len - len(b[0])) for b in batch],
                           dtype=torch.long, device=device)
        return inp, lab, msk

    opt = AdamW([p for p in model.parameters() if p.requires_grad],
                lr=lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train_items) / batch_size / grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup = max(1, int(0.05 * total_steps))
    def _sched(s):
        if s < warmup:
            return s / max(1, warmup)
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    sched = LambdaLR(opt, _sched)

    rng = np.random.default_rng(42)
    t0 = time.time()
    best_val_f1, best_step = -1.0, -1
    eval_every_micro = max(1, steps_per_epoch * grad_accum // 4)  # 4 evals per epoch
    print(f"[pillar_b] training: {total_steps} opt-steps "
          f"({steps_per_epoch}/epoch × {epochs} epochs), eval every {eval_every_micro} micro-steps")

    opt_step = 0
    accum_loss = 0.0
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(train_items)).tolist()
        for i in range(0, len(train_items), batch_size):
            batch_idx = order[i:i + batch_size]
            batch = [train_items[k] for k in batch_idx]
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
                    eta = (total_steps - opt_step) / max(rate, 1e-6)
                    print(f"[pillar_b]   step {opt_step}/{total_steps}  "
                          f"loss={accum_loss:.3f}  lr={sched.get_last_lr()[0]:.2e}  "
                          f"{rate:.2f}step/s  eta={eta/60:.1f}min")
                accum_loss = 0.0
            # Eval on val (subset for speed)
            if (i // batch_size) % eval_every_micro == 0 and i > 0:
                vf1 = _eval_quick(model, tokenizer, val_items[:128], letter_ids, batch_size)
                if vf1 > best_val_f1:
                    best_val_f1, best_step = vf1, opt_step
                    os.makedirs(out_adapter_dir, exist_ok=True)
                    model.save_pretrained(os.path.join(out_adapter_dir, 'best'))
                    tokenizer.save_pretrained(os.path.join(out_adapter_dir, 'best'))
                    print(f"[pillar_b]   val_f1@step{opt_step} = {vf1:.4f}  (saved best)")
                else:
                    print(f"[pillar_b]   val_f1@step{opt_step} = {vf1:.4f}  (best {best_val_f1:.4f})")
                model.train()

    # Final eval on full val set
    final_vf1 = _eval_quick(model, tokenizer, val_items, letter_ids, batch_size)
    print(f"[pillar_b] final val_f1 (full) = {final_vf1:.4f}")
    if final_vf1 > best_val_f1:
        best_val_f1 = final_vf1
        model.save_pretrained(os.path.join(out_adapter_dir, 'best'))
        tokenizer.save_pretrained(os.path.join(out_adapter_dir, 'best'))
        print(f"[pillar_b]   saved best (final)")
    print(f"[pillar_b] train wall-clock: {(time.time() - t0)/60:.1f} min")
    return best_val_f1


@torch.no_grad()
def _eval_quick(model, tokenizer, items, letter_ids, batch_size=8):
    """Compute macro-F1 on a list of pre-tokenized items where labels has -100
    everywhere except at the answer position. We just predict the answer-letter
    via logit comparison."""
    device = next(model.parameters()).device
    model.eval()
    pad = tokenizer.pad_token_id
    e_id, m_id, h_id = letter_ids['E'], letter_ids['M'], letter_ids['H']
    preds, golds = [], []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        # Use LEFT padding so answer position is always at -1.
        max_len = max(len(b[0]) - 1 for b in batch)  # all but the final letter
        inp_ids = []
        masks = []
        for b in batch:
            prompt_only = b[0][:-1]  # drop the answer letter
            pad_len = max_len - len(prompt_only)
            inp_ids.append([pad] * pad_len + prompt_only)
            masks.append([0] * pad_len + [1] * len(prompt_only))
            # ground truth label = the letter that was masked
            gold_id = b[0][-1]
            if gold_id == e_id: golds.append(0)
            elif gold_id == m_id: golds.append(1)
            elif gold_id == h_id: golds.append(2)
            else:
                # fallback: find which letter best matches
                # take label from `labels` (last non-(-100) entry)
                for x in reversed(b[1]):
                    if x != -100:
                        if x == e_id: golds.append(0)
                        elif x == m_id: golds.append(1)
                        else: golds.append(2)
                        break
                else:
                    golds.append(0)
        inp_t  = torch.tensor(inp_ids, dtype=torch.long, device=device)
        mask_t = torch.tensor(masks,   dtype=torch.long, device=device)
        out = model(input_ids=inp_t, attention_mask=mask_t, use_cache=False)
        last_logits = out.logits[:, -1, :]
        e_l = last_logits[:, e_id]
        m_l = last_logits[:, m_id]
        h_l = last_logits[:, h_id]
        stacked = torch.stack([e_l, m_l, h_l], dim=-1)
        preds.extend(stacked.argmax(dim=-1).cpu().tolist())
    return float(f1_score(golds, preds, average='macro', labels=[0, 1, 2],
                           zero_division=0))


@torch.no_grad()
def _predict(model, tokenizer, texts, letter_ids, batch_size=8, max_length=1280):
    """Predict class labels for a list of texts via logit comparison.
    Returns ndarray of int class indices."""
    device = next(model.parameters()).device
    model.eval()
    tokenizer.padding_side = 'left'
    pad = tokenizer.pad_token_id
    e_id, m_id, h_id = letter_ids['E'], letter_ids['M'], letter_ids['H']
    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        prompts = [build_prompt(t) for t in batch_texts]
        enc = tokenizer(prompts, padding=True, return_tensors='pt',
                         truncation=True, max_length=max_length).to(device)
        out = model(**enc, use_cache=False)
        last_logits = out.logits[:, -1, :]
        e_l = last_logits[:, e_id]
        m_l = last_logits[:, m_id]
        h_l = last_logits[:, h_id]
        stacked = torch.stack([e_l, m_l, h_l], dim=-1)
        preds.extend(stacked.argmax(dim=-1).cpu().tolist())
    return np.array(preds, dtype=np.int64)


def _measure_latency(model, tokenizer, sample_texts, letter_ids,
                     n_warmup=5, n_timed=30, max_length=1280):
    device = next(model.parameters()).device
    model.eval()
    tokenizer.padding_side = 'left'
    times = []
    for t in sample_texts[:n_warmup]:
        _ = _predict(model, tokenizer, [t], letter_ids, batch_size=1,
                      max_length=max_length)
    for t in sample_texts[n_warmup:n_warmup + n_timed]:
        t0 = time.perf_counter()
        _ = _predict(model, tokenizer, [t], letter_ids, batch_size=1,
                      max_length=max_length)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        'median_ms': float(np.median(times) * 1000),
        'p90_ms':    float(np.quantile(times, 0.9) * 1000),
        'n_warmup':  n_warmup, 'n_timed': n_timed,
        'device':    device.type,
    }


def evaluate_one_backbone(backbone_short, hf_id, train_data, val_data,
                           eval_splits, adv_data, train_subset_size,
                           epochs, batch_size, grad_accum, lr):
    """Train + eval one (backbone). Returns the report dict + frees the model."""
    print(f"\n[pillar_b] ==== {backbone_short} ({hf_id}) ====")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    model.gradient_checkpointing_enable()
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[pillar_b]   total params: {n_total:,}")

    train_texts, train_y = train_data
    val_texts, val_y     = val_data

    # Subsample train if specified.
    if train_subset_size is not None and train_subset_size < len(train_texts):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_texts), size=train_subset_size, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_y     = train_y[idx]
        print(f"[pillar_b]   train subsampled to {len(train_texts)}")

    adapter_dir = os.path.join(MODEL_DIR, f'pillar_b__{backbone_short}__lora')
    os.makedirs(adapter_dir, exist_ok=True)

    t0_train = time.time()
    best_val_f1 = _train_one_backbone(
        backbone_short, hf_id, tok, model,
        train_texts, train_y, val_texts, val_y,
        out_adapter_dir=adapter_dir, epochs=epochs, lr=lr,
        batch_size=batch_size, grad_accum=grad_accum)
    train_sec = time.time() - t0_train
    print(f"[pillar_b]   total train time = {train_sec/60:.1f} min")

    # Reload the best validation checkpoint before evaluation.
    print(f"[pillar_b]   reloading best LoRA adapter for eval ...")
    del model
    gc.collect(); torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(
        hf_id, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, os.path.join(adapter_dir, 'best')).to('cuda')
    model.eval()

    letter_ids = _find_letter_token_ids(tok)

    # Full eval matrix
    per_split = {}
    for split, (texts, y) in eval_splits.items():
        t0 = time.time()
        pred = _predict(model, tok, texts, letter_ids, batch_size=8)
        per_split[split] = _eval_block(y, pred)
        per_split[split]['eval_sec'] = round(time.time() - t0, 1)
        print(f"[pillar_b]   {split}: f1_macro = {per_split[split]['f1_macro']:.4f}")
        if split == 'test':
            f1s = []
            for s in range(5):
                _, y_bal, idx_bal = _balanced_subsample(texts, y, per_class=50,
                                                        seed=42 + s)
                f1s.append(float(f1_score(y_bal, pred[idx_bal], average='macro',
                                           labels=[0, 1, 2], zero_division=0)))
            per_split['balanced_test'] = {
                'f1_macro_mean': float(np.mean(f1s)),
                'f1_macro_std':  float(np.std(f1s)),
                'f1_macro_per_seed': f1s,
                'n_resamples': 5, 'per_class_n': 50,
            }
            print(f"[pillar_b]   balanced_test: "
                  f"{per_split['balanced_test']['f1_macro_mean']:.4f}")
        if split == 'ood_synthetic':
            # Also evaluate vs gen labels (paper headline)
            split_csv = os.path.join(GF_OUT, 'splits', 'ood_synthetic.csv')
            sdf = pd.read_csv(split_csv)
            y_gen = sdf['education_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
            per_split['ood_synthetic_vs_gen'] = _eval_block(y_gen, pred)
            print(f"[pillar_b]   ood_synthetic_vs_gen: f1_macro = "
                  f"{per_split['ood_synthetic_vs_gen']['f1_macro']:.4f}")

    # AdvConcept-50
    adv_texts, adv_y, adv_df = adv_data
    pred_adv = _predict(model, tok, adv_texts, letter_ids, batch_size=8)
    adv_block = _eval_block(adv_y, pred_adv)
    cat_acc = {}
    for cat in adv_df['category'].unique():
        mask = (adv_df['category'] == cat).to_numpy()
        cat_acc[cat] = float(accuracy_score(adv_y[mask], pred_adv[mask]))
    per_row = []
    for i, row in adv_df.iterrows():
        per_row.append({
            'idx':         int(row['idx']),
            'text':        row['text'],
            'true_level':  row['true_level'],
            'predicted_level': LABEL_NAMES[int(pred_adv[i])],
            'category':    row['category'],
            'correct':     bool(LABEL_NAMES[int(pred_adv[i])] == row['true_level']),
        })
    adv_report = {
        'overall_accuracy': adv_block['accuracy'],
        'overall_f1_macro': adv_block['f1_macro'],
        'per_category_accuracy': cat_acc,
        'conf_matrix': adv_block['conf_matrix'],
        'per_row': per_row,
    }
    print(f"[pillar_b]   adv_concept: acc = {adv_block['accuracy']:.4f}")
    for cat, acc in cat_acc.items():
        print(f"[pillar_b]     {cat}: {acc:.4f}")

    # Latency
    latency = _measure_latency(model, tok, eval_splits['test'][0][:60], letter_ids)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = {
        'kind':           'pillar_b',
        'name':           f'pillar_b__{backbone_short}',
        'backbone_short': backbone_short,
        'backbone_hf':    hf_id,
        'n_params_total': int(n_total),
        'n_params_trainable_lora': int(n_trainable),
        'epochs':         epochs,
        'batch_size':     batch_size,
        'grad_accum':     grad_accum,
        'lr':             lr,
        'train_sec':      round(train_sec, 1),
        'best_val_f1':    best_val_f1,
        'per_split':      per_split,
        'adv_concept':    adv_report,
        'latency':        latency,
    }
    out_path = os.path.join(REPORT_DIR, f'pillar_b__{backbone_short}.eval.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"[pillar_b]   wrote {_display_path(out_path)}")

    # Free GPU
    del model, base
    gc.collect(); torch.cuda.empty_cache()
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backbones', nargs='+',
                    default=[b[0] for b in BACKBONES],
                    help='Subset of: ' + ', '.join(b[0] for b in BACKBONES))
    ap.add_argument('--train-subset', type=int, default=None,
                    help='Limit train rows (default: all 17K)')
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='pillar_b_tiny_teacher',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        # Load all data once.
        train_texts, train_y = _load_split('train', 'judge')
        val_texts,   val_y   = _load_split('val',   'judge')
        print(f"[pillar_b] loaded train={len(train_texts)} val={len(val_texts)}")
        print(f"[pillar_b] train dist: "
              f"{dict(zip(LABEL_NAMES, np.bincount(train_y, minlength=3).tolist()))}")

        eval_splits = {}
        for s in CANONICAL_SPLITS:
            tx, y = _load_split(s, 'judge')
            eval_splits[s] = (tx, y)
            print(f"[pillar_b]   eval split {s}: n={len(tx)}")

        adv_df = pd.read_csv(ADV_CSV)
        adv_texts = adv_df['text'].astype(str).tolist()
        adv_y = adv_df['true_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
        adv_data = (adv_texts, adv_y, adv_df)

        # Bake-off loop
        all_reports = []
        for short, hf_id, _est in BACKBONES:
            if short not in args.backbones:
                continue
            try:
                rep = evaluate_one_backbone(
                    short, hf_id,
                    train_data=(train_texts, train_y),
                    val_data=(val_texts, val_y),
                    eval_splits=eval_splits,
                    adv_data=adv_data,
                    train_subset_size=args.train_subset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    grad_accum=args.grad_accum,
                    lr=args.lr,
                )
                all_reports.append(rep)
            except Exception as e:
                import traceback
                print(f"[pillar_b] FAILED {short}: {e}")
                traceback.print_exc()
                all_reports.append({
                    'kind': 'pillar_b', 'name': f'pillar_b__{short}',
                    'backbone_short': short, 'backbone_hf': hf_id,
                    'error': str(e),
                })

        # Summary
        def _key(r):
            return r.get('per_split', {}).get('ood_synthetic_vs_gen', {}).get('f1_macro', 0.0)
        winner = max(all_reports, key=_key) if all_reports else None
        summary = {
            'kind': 'pillar_b_summary',
            'n_candidates': len(all_reports),
            'winner_by_synthetic_vs_gen':
                {'name': winner['name'],
                 'f1_synthetic_vs_gen': _key(winner)} if winner else None,
            'candidates': [
                {
                    'name': r.get('name'),
                    'backbone_short': r.get('backbone_short'),
                    'n_params_total': r.get('n_params_total'),
                    'n_params_trainable_lora': r.get('n_params_trainable_lora'),
                    'train_sec': r.get('train_sec'),
                    'val_f1': r.get('best_val_f1'),
                    'test_f1':                  r.get('per_split', {}).get('test', {}).get('f1_macro'),
                    'balanced_test_f1_mean':    r.get('per_split', {}).get('balanced_test', {}).get('f1_macro_mean'),
                    'synthetic_judge_f1':       r.get('per_split', {}).get('ood_synthetic', {}).get('f1_macro'),
                    'synthetic_gen_f1':         r.get('per_split', {}).get('ood_synthetic_vs_gen', {}).get('f1_macro'),
                    'race_middle_f1':           r.get('per_split', {}).get('ood_race-middle', {}).get('f1_macro'),
                    'race_high_f1':             r.get('per_split', {}).get('ood_race-high', {}).get('f1_macro'),
                    'onestop_f1':               r.get('per_split', {}).get('ood_onestop', {}).get('f1_macro'),
                    'adv_concept_overall':      r.get('adv_concept', {}).get('overall_accuracy'),
                    'adv_per_category':         r.get('adv_concept', {}).get('per_category_accuracy'),
                    'latency_median_ms':        r.get('latency', {}).get('median_ms'),
                    'error':                    r.get('error'),
                }
                for r in all_reports
            ],
        }
        with open(os.path.join(REPORT_DIR, 'pillar_b_summary.json'), 'w') as fh:
            json.dump(summary, fh, indent=2)
        if winner:
            print(f"\n[pillar_b] WINNER: {winner['name']} "
                  f"(synthetic_vs_gen F1 = {_key(winner):.4f})")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
