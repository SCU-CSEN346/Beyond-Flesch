"""
Shared evaluation code for the gen_v2 experiments.

The point of this file is consistency: the Rooein-style baseline, Pillar A,
Pillar B, and Pillar B+ all go through the same splits and metric code. That
keeps the comparisons honest across normal test data, OOD sets, balanced
resamples, and AdvConcept-50.

For the baseline, a few texts need fresh prompt features. We build those once
with Qwen-2.5-7B and cache them for later runs.
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from math import log2

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)

import textstat
import torch

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT   = os.path.dirname(_PROJECT_DIR)

GF_DIR       = os.path.join(_REPO_ROOT, 'generalization_final')
GF_OUT       = os.path.join(GF_DIR, 'outputs')
ADV_CSV      = os.path.join(_PROJECT_DIR, 'data', 'adv_concept.csv')
REPORT_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')
PROMPT_CACHE_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'prompt_cache')

LABEL_MAP   = {'elementary': 0, 'middle': 1, 'high': 2}
LABEL_NAMES = ['elementary', 'middle', 'high']

def _display_path(path):
    return os.path.relpath(path, _REPO_ROOT)

# Load the baseline model class from the original pipeline code.
sys.path.insert(0, os.path.join(GF_DIR, 'scripts'))
from step8l_combo_classifier import ComboMLP  # noqa: E402


# Static feature extraction. Keep this aligned with the original baseline so
# cached and freshly-computed features mean the same thing.

READABILITY = [
    'flesch_reading_ease', 'flesch_kincaid_grade', 'smog_index',
    'coleman_liau_index', 'automated_readability_index',
    'dale_chall_readability_score', 'difficult_words',
    'linsear_write_formula', 'gunning_fog', 'text_standard',
    'spache_readability', 'mcalpine_eflaw',
]
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _safe(fn, *args, **kwargs):
    try:
        v = fn(*args, **kwargs)
        if isinstance(v, str):
            m = re.search(r'\d+', v)
            return float(m.group(0)) if m else float('nan')
        return float(v)
    except Exception:
        return float('nan')


def _word_entropy(words):
    if not words:
        return 0.0
    c = Counter(w.lower() for w in words)
    n = sum(c.values())
    return -sum((k / n) * log2(k / n) for k in c.values())


def static_features(text):
    text = (text or '').strip()
    if not text:
        return {}
    feats = {}
    for name in READABILITY:
        feats[f'stat_{name}'] = _safe(getattr(textstat, name), text)
    words = WORD_RE.findall(text)
    sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]
    n_words = len(words)
    n_sents = max(len(sentences), 1)
    feats['stat_char_count']     = len(text)
    feats['stat_word_count']     = n_words
    feats['stat_unique_words']   = len(set(w.lower() for w in words))
    feats['stat_ttr']            = feats['stat_unique_words'] / n_words if n_words else 0.0
    feats['stat_avg_word_len']   = float(np.mean([len(w) for w in words])) if words else 0.0
    feats['stat_long_word_ratio'] = sum(1 for w in words if len(w) > 6) / n_words if n_words else 0.0
    feats['stat_sentence_count'] = len(sentences)
    feats['stat_avg_sent_len']   = n_words / n_sents
    feats['stat_max_sent_len']   = max((len(WORD_RE.findall(s)) for s in sentences), default=0)
    feats['stat_word_entropy']   = _word_entropy(words)
    feats['stat_n_question']     = text.count('?')
    feats['stat_n_exclaim']      = text.count('!')
    feats['stat_n_comma']        = text.count(',')
    feats['stat_n_semicolon']    = text.count(';')
    feats['stat_n_paren']        = text.count('(') + text.count(')')
    feats['stat_n_quote']        = text.count('"')
    feats['stat_n_paragraphs']   = text.count('\n\n') + 1
    feats['stat_n_newlines']     = text.count('\n')
    lower = text.lower()
    feats['stat_has_because']    = float(' because' in lower)
    feats['stat_has_therefore']  = float('therefore' in lower)
    feats['stat_has_example']    = float('example' in lower or 'e.g.' in lower)
    feats['stat_has_multiple_choice'] = float(any(m in lower for m in [' (a)', ' (b)', ' (c)']))
    return feats


# Prompt feature extraction for texts without cached prompt metrics.

PROMPT_QUESTIONS_JSON = os.path.join(GF_DIR, 'inputs', 'prompt_questions.json')
FEATURE_LLM = 'Qwen/Qwen2.5-7B-Instruct'
MAX_MODEL_LEN = 2048


def build_prompt_raw(text, question):
    return (
        f"Read the following text and answer the question with only 'yes' or 'no'.\n\n"
        f"Text: {text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def parse_yes_no(response):
    r = response.strip().lower()
    r = re.sub(r'^[\*_`>\-\s"\']+', '', r)
    head = r[:30]
    if head.startswith('yes'):
        return 1
    if head.startswith('no'):
        return 0
    has_yes = bool(re.search(r'\byes\b', head))
    has_no  = bool(re.search(r'\bno\b',  head))
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return -1


_LLM_STATE = {}  # singleton: (model, tokenizer)


def _get_llm():
    """Load Qwen-2.5-7B only when prompt features are needed."""
    if _LLM_STATE:
        return _LLM_STATE['model'], _LLM_STATE['tokenizer']
    os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/hf_user/hub'))
    print(f"[evaluate_all] loading {FEATURE_LLM} via transformers ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(FEATURE_LLM)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        FEATURE_LLM, dtype=torch.bfloat16, attn_implementation='sdpa'
    ).to('cuda').eval()
    _LLM_STATE['model'] = model
    _LLM_STATE['tokenizer'] = tok
    print(f"[evaluate_all] model loaded on GPU")
    return model, tok


def _yes_no_token_ids(tok):
    """Return token ids for common yes/no variants."""
    candidates_yes = ['yes', ' yes', 'Yes', ' Yes', 'YES', ' YES']
    candidates_no  = ['no', ' no', 'No', ' No', 'NO', ' NO']
    yes_ids, no_ids = set(), set()
    for w in candidates_yes:
        ids = tok(w, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            yes_ids.add(ids[0])
    for w in candidates_no:
        ids = tok(w, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            no_ids.add(ids[0])
    return sorted(yes_ids), sorted(no_ids)


@torch.no_grad()
def extract_prompt_features(texts, cache_key=None, batch_size=16,
                             max_input_tokens=1024):
    """Run the 63 yes/no prompt metrics and return a binary feature matrix."""
    if cache_key:
        os.makedirs(PROMPT_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(PROMPT_CACHE_DIR, f'{cache_key}.csv')
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
            cols = sorted([c for c in df.columns if c.startswith('prompt_')],
                          key=lambda s: int(s.split('_')[1]))
            arr = df[cols].astype(np.float32).to_numpy()
            if len(arr) == len(texts) and arr.shape[1] == 63:
                print(f"[evaluate_all] using cached prompt features {_display_path(cache_path)}")
                return arr
            print(f"[evaluate_all] cache size mismatch ({arr.shape} vs {len(texts)} texts) - re-extracting")

    with open(PROMPT_QUESTIONS_JSON) as fh:
        questions = json.load(fh)
    assert len(questions) == 63, f"expected 63 prompts, got {len(questions)}"

    model, tok = _get_llm()
    yes_ids, no_ids = _yes_no_token_ids(tok)
    print(f"[evaluate_all] yes_token_ids={yes_ids}  no_token_ids={no_ids}")
    yes_t = torch.tensor(yes_ids, dtype=torch.long, device=model.device)
    no_t  = torch.tensor(no_ids,  dtype=torch.long, device=model.device)
    # Leave room for the question and answer marker.
    overhead = len(tok(build_prompt_raw('', questions[0]),
                       add_special_tokens=False)['input_ids'])
    max_text_tokens = max_input_tokens - overhead - 16

    # Build all text/question prompt strings.
    n = len(texts)
    all_prompts = []
    for raw_text in texts:
        text = (raw_text or '').strip()
        ids = tok(text, add_special_tokens=False)['input_ids']
        if len(ids) > max_text_tokens:
            text = tok.decode(ids[:max_text_tokens], skip_special_tokens=True)
        for q in questions:
            all_prompts.append(build_prompt_raw(text, q))
    print(f"[evaluate_all] {len(all_prompts)} prompts to score ({n} texts × 63 questions)")

    # Left padding keeps the answer position aligned at the end.
    tok.padding_side = 'left'
    preds = np.zeros(len(all_prompts), dtype=np.int8)
    t0 = time.time()
    for start in range(0, len(all_prompts), batch_size):
        batch = all_prompts[start:start + batch_size]
        enc = tok(batch, padding=True, return_tensors='pt',
                  truncation=True, max_length=max_input_tokens).to(model.device)
        out = model(**enc, use_cache=False)
        # Next-token logits after "Answer:".
        last_logits = out.logits[:, -1, :]  # (B, V)
        yes_score = last_logits[:, yes_t].max(dim=-1).values
        no_score  = last_logits[:, no_t].max(dim=-1).values
        preds_batch = (yes_score > no_score).cpu().numpy().astype(np.int8)
        preds[start:start + len(batch)] = preds_batch
        done = start + len(batch)
        if done % (batch_size * 10) == 0 or done == len(all_prompts):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-3)
            eta = (len(all_prompts) - done) / max(rate, 1e-6)
            print(f"[evaluate_all] scored {done}/{len(all_prompts)}  "
                  f"({rate:.0f}/sec, eta {eta:.0f}s)")

    arr = preds.reshape(n, 63).astype(np.float32)

    if cache_key:
        cols = [f'prompt_{j}' for j in range(63)]
        out_df = pd.DataFrame(arr, columns=cols)
        out_df.insert(0, 'text', texts)
        out_df.to_csv(cache_path, index=False)
        print(f"[evaluate_all] cached prompt features to {_display_path(cache_path)}")
    return arr


# Evaluation helpers.

def _eval_block(y_true, y_pred):
    return {
        'f1_macro':    float(f1_score(y_true, y_pred, average='macro',
                                       labels=[0, 1, 2], zero_division=0)),
        'f1_per_class': f1_score(y_true, y_pred, average=None,
                                  labels=[0, 1, 2], zero_division=0).tolist(),
        'accuracy':    float(accuracy_score(y_true, y_pred)),
        'qwk':         float(cohen_kappa_score(y_true, y_pred, weights='quadratic')),
        'conf_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        'pred_counts': np.bincount(np.asarray(y_pred, dtype=np.int64),
                                    minlength=3).tolist(),
        'n':           int(len(y_true)),
    }


def _balanced_subsample(texts, y, per_class=50, seed=42):
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(3):
        pool = np.where(y == c)[0]
        if len(pool) < per_class:
            idx.append(pool)
        else:
            idx.append(rng.choice(pool, size=per_class, replace=False))
    idx = np.concatenate(idx)
    return [texts[i] for i in idx], y[idx], idx


# Baseline harness for the saved Rooein-style Qwen bundle.

class BaselineHarness:
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as fh:
            bundle = pickle.load(fh)
        self.pkl_path = pkl_path
        self.scaler = bundle['scaler']
        self.feature_dim = bundle['feature_dim']
        self.use_grade_reg = bool(bundle.get('use_grade_reg', False))
        self.model = ComboMLP(in_dim=self.feature_dim,
                              use_grade_reg=self.use_grade_reg)
        self.model.load_state_dict(bundle['model_state'])
        self.model.eval()
        self.best_val_f1 = bundle.get('report', {}).get('best_val_f1_macro', None)
        # Static feature order is alphabetical (matches step3d).
        # Determine the static dim from the bundle.
        # per_llm bundles use 34 static + 63 prompts = 97.
        # all_llms use 34 + 4*63 = 286.
        n_prompts = 63
        self.n_static = self.feature_dim - n_prompts
        if self.n_static != 34:
            # Not a per_llm bundle; we only handle per_llm in this harness.
            raise ValueError(
                f"BaselineHarness only supports per_llm bundles "
                f"(34 static + 63 prompts = 97). Bundle has feature_dim={self.feature_dim}.")

        # Pin static feature names from generalization_final's saved JSON.
        fname_json = os.path.join(GF_OUT, 'static_features', 'feature_names.json')
        with open(fname_json) as fh:
            self.static_feature_names = sorted(json.load(fh))
        assert len(self.static_feature_names) == self.n_static, (
            f"static_feature_names has {len(self.static_feature_names)}, "
            f"expected {self.n_static}")

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    # ----- feature construction -----

    def features_from_cached(self, split_name):
        """Build baseline features for a saved split."""
        sdf = pd.read_csv(os.path.join(GF_OUT, 'static_features',
                                       f'{split_name}_static.csv'))
        stat_cols = sorted(c for c in sdf.columns if c.startswith('stat_'))
        static = sdf[stat_cols].astype(np.float32).to_numpy()

        split_df = pd.read_csv(os.path.join(GF_OUT, 'splits', f'{split_name}.csv'))
        if len(split_df) != len(static):
            sys.exit(f"[evaluate_all] {split_name}: static {len(static)} vs split {len(split_df)}")

        # Cache prompt CSVs by original split so each file is read once.
        prompt_cache = {}
        prompt_block = np.zeros((len(split_df), 63), dtype=np.float32)
        for i, (orig_split, orig_idx) in enumerate(zip(split_df['orig_split'],
                                                        split_df['orig_idx'])):
            if orig_split not in prompt_cache:
                p = os.path.join(GF_OUT, 'prompt_metrics',
                                 f'{orig_split}_prompt_metrics_qwen2.5-7b__multi.csv')
                pdf = pd.read_csv(p)
                cols = sorted([c for c in pdf.columns if c.startswith('prompt_')],
                              key=lambda s: int(s.split('_')[1]))
                arr = pdf[cols].astype(np.float32).to_numpy()
                # Replace uncertain prompt outputs with the column mean.
                for j in range(arr.shape[1]):
                    col = arr[:, j]
                    if (col < 0).any():
                        valid = col[col >= 0]
                        col[col < 0] = float(valid.mean()) if len(valid) else 0.0
                prompt_cache[orig_split] = arr
            prompt_block[i] = prompt_cache[orig_split][int(orig_idx)]
        return np.concatenate([static, prompt_block], axis=1).astype(np.float32)

    def features_fresh(self, texts, cache_key=None):
        """Build baseline features for ad-hoc text."""
        n = len(texts)
        static = np.zeros((n, self.n_static), dtype=np.float32)
        for i, t in enumerate(texts):
            f = static_features(t)
            for j, name in enumerate(self.static_feature_names):
                v = f.get(name, 0.0)
                static[i, j] = 0.0 if (v is None or np.isnan(v)) else v
        # Prompt features.
        prompts = extract_prompt_features(texts, cache_key=cache_key)
        # Match the imputation used for cached features.
        for j in range(prompts.shape[1]):
            col = prompts[:, j]
            if (col < 0).any():
                valid = col[col >= 0]
                col[col < 0] = float(valid.mean()) if len(valid) else 0.0
        return np.concatenate([static, prompts], axis=1).astype(np.float32)

    # ----- prediction -----

    def predict(self, X):
        X_s = self.scaler.transform(X).astype(np.float32)
        with torch.no_grad():
            logits, _ = self.model(torch.tensor(X_s, dtype=torch.float32))
        return logits.argmax(-1).cpu().numpy()


# Canonical split runner.

CANONICAL_SPLITS = {
    'test':              ('test',           'judge'),
    'ood_synthetic':     ('ood_synthetic',  'judge'),
    'ood_race-middle':   ('ood_race-middle','judge'),
    'ood_race-high':     ('ood_race-high',  'judge'),
    'ood_onestop':       ('ood_onestop',    'judge'),
}


def _load_labels(split_name, label_source='judge'):
    """Return labels and texts for one saved split."""
    if label_source == 'judge':
        judge_path = os.path.join(GF_OUT, 'llm_judge', f'{split_name}_judge.csv')
        if not os.path.exists(judge_path):
            return None, None
        df = pd.read_csv(judge_path)
        y = df['llm_judge_label'].map(LABEL_MAP).astype(np.int64).to_numpy()
    elif label_source == 'gen':
        split_path = os.path.join(GF_OUT, 'splits', f'{split_name}.csv')
        df = pd.read_csv(split_path)
        y = df['education_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
    else:
        raise ValueError(f"unknown label_source {label_source}")
    # Load the split CSV to get the text column.
    split_path = os.path.join(GF_OUT, 'splits', f'{split_name}.csv')
    sdf = pd.read_csv(split_path)
    texts = sdf['text'].tolist() if 'text' in sdf.columns else sdf['full_text'].tolist()
    return y, texts


def evaluate_baseline(pkl_path, include_adv=True, n_balanced_resamples=5,
                       balanced_seed_base=42):
    """Run the full eval matrix against a per_llm bundle. Returns the report dict."""
    h = BaselineHarness(pkl_path)
    report = {
        'kind': 'baseline',
        'name': os.path.basename(pkl_path),
        'model_meta': {
            'pkl': _display_path(pkl_path),
            'feature_dim': h.feature_dim,
            'n_params':    h.n_params(),
            'best_val_f1_macro': h.best_val_f1,
            'kind_detail': 'per_llm Qwen-7B + 34 static + 63 prompts → MLP-256-128',
        },
        'per_split': {},
    }

    for split_name, (gf_split, label_src) in CANONICAL_SPLITS.items():
        y, texts = _load_labels(gf_split, label_source=label_src)
        if y is None:
            print(f"[evaluate_all] skipping {split_name}: no labels file")
            continue
        X = h.features_from_cached(gf_split)
        if X.shape[0] != len(y):
            sys.exit(f"[evaluate_all] {split_name}: row mismatch X={X.shape[0]} y={len(y)}")
        pred = h.predict(X)
        report['per_split'][split_name] = _eval_block(y, pred)
        print(f"[evaluate_all]   {split_name}: macro-F1 = {report['per_split'][split_name]['f1_macro']:.4f}")

        if split_name == 'test':
            # Five balanced resamples of the test set.
            f1s = []
            for s in range(n_balanced_resamples):
                _, y_bal, idx_bal = _balanced_subsample(texts, y, per_class=50,
                                                        seed=balanced_seed_base + s)
                pred_bal = pred[idx_bal]
                f1s.append(float(f1_score(y_bal, pred_bal, average='macro',
                                           labels=[0, 1, 2], zero_division=0)))
            report['per_split']['balanced_test'] = {
                'f1_macro_mean': float(np.mean(f1s)),
                'f1_macro_std':  float(np.std(f1s)),
                'f1_macro_per_seed': f1s,
                'n_resamples': n_balanced_resamples,
                'per_class_n': 50,
            }
            print(f"[evaluate_all]   balanced_test (mean of {n_balanced_resamples}): "
                  f"{report['per_split']['balanced_test']['f1_macro_mean']:.4f}")

    # Also score synthetic data against the requested generation label.
    y_gen, texts = _load_labels('ood_synthetic', label_source='gen')
    if y_gen is not None:
        X = h.features_from_cached('ood_synthetic')
        pred = h.predict(X)
        report['per_split']['ood_synthetic_vs_gen'] = _eval_block(y_gen, pred)
        print(f"[evaluate_all]   ood_synthetic_vs_gen: macro-F1 = "
              f"{report['per_split']['ood_synthetic_vs_gen']['f1_macro']:.4f}")

    if include_adv:
        adv_df = pd.read_csv(ADV_CSV)
        adv_texts = adv_df['text'].tolist()
        y_adv = adv_df['true_level'].map(LABEL_MAP).astype(np.int64).to_numpy()
        X_adv = h.features_fresh(adv_texts, cache_key='adv_concept_qwen2.5-7b')
        pred_adv = h.predict(X_adv)
        adv_block = _eval_block(y_adv, pred_adv)
        # Per-category accuracy.
        cat_acc = {}
        for cat in adv_df['category'].unique():
            mask = (adv_df['category'] == cat).to_numpy()
            cat_acc[cat] = float(accuracy_score(y_adv[mask], pred_adv[mask]))
        # Per-row table for error analysis.
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
        report['adv_concept'] = {
            'overall_accuracy':       adv_block['accuracy'],
            'overall_f1_macro':       adv_block['f1_macro'],
            'per_category_accuracy':  cat_acc,
            'conf_matrix':            adv_block['conf_matrix'],
            'pred_counts':            adv_block['pred_counts'],
            'per_row':                per_row,
        }
        print(f"[evaluate_all]   adv_concept overall accuracy = {adv_block['accuracy']:.4f}")
        for cat, acc in cat_acc.items():
            print(f"[evaluate_all]     {cat}: {acc:.4f}")

    return report


# CLI.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=os.path.join(GF_OUT, 'models',
                    'per_llm__llm_judge__qwen2.5-7b__seed7_gradereg.pkl'))
    ap.add_argument('--out-name', default='baseline_per_llm_qwen2.5-7b_seed7')
    ap.add_argument('--skip-adv', action='store_true')
    args = ap.parse_args()
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='evaluate_all_baseline',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        report = evaluate_baseline(args.bundle, include_adv=not args.skip_adv)
    finally:
        tracker.stop()

    out_path = os.path.join(REPORT_DIR, f'{args.out_name}.eval.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[evaluate_all] wrote {_display_path(out_path)}")


if __name__ == '__main__':
    main()
