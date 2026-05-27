"""
Run the Pillar A encoder bake-off.

Pillar A is the cheap baseline: sentence-transformer embeddings, optionally
plus five readability features, followed by logistic regression. It tests how
far we can get without the LLM prompt-feature loop.

Each encoder is evaluated twice: embedding-only, then embedding plus the small
readability feature set.
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
from codecarbon import EmissionsTracker
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT   = os.path.dirname(_PROJECT_DIR)
sys.path.insert(0, _SCRIPT_DIR)
from evaluate_all import (  # noqa: E402
    static_features, GF_OUT, ADV_CSV, LABEL_MAP, LABEL_NAMES,
    _balanced_subsample, _eval_block,
)

EMB_DIR     = os.path.join(_PROJECT_DIR, 'outputs', 'embeddings')
MODEL_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'models')
REPORT_DIR  = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')


def _display_path(path):
    return os.path.relpath(path, _PROJECT_DIR)


ENCODERS = [
    ('all-MiniLM-L6-v2',
        'sentence-transformers/all-MiniLM-L6-v2',  22, 384),
    ('bge-small-en-v1.5',
        'BAAI/bge-small-en-v1.5',                  33, 384),
    ('bge-base-en-v1.5',
        'BAAI/bge-base-en-v1.5',                  109, 768),
]

# Five cheap readability features that were strong in the Rooein analysis.
STATIC_TOP5 = [
    'stat_gunning_fog',
    'stat_coleman_liau_index',
    'stat_flesch_reading_ease',
    'stat_automated_readability_index',
    'stat_unique_words',
]

# Same split list used by the shared evaluator.
CANONICAL_SPLITS = ['test', 'ood_synthetic', 'ood_race-middle',
                    'ood_race-high', 'ood_onestop']


def _load_split_texts_and_labels(split_name, label_source='judge'):
    """Load texts and labels for one split."""
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
    assert len(texts) == len(y), f"{split_name}: texts {len(texts)} vs labels {len(y)}"
    return texts, y


def _embed(encoder_hf_id, encoder_short, texts, split_tag,
            device='cuda', batch_size=64):
    """Encode texts and cache the embedding matrix."""
    os.makedirs(EMB_DIR, exist_ok=True)
    cache = os.path.join(EMB_DIR, f'{encoder_short}__{split_tag}.npy')
    if os.path.exists(cache):
        arr = np.load(cache)
        if len(arr) == len(texts):
            print(f"[pillar_a] cached embeddings: {_display_path(cache)}  shape={arr.shape}")
            return arr
        print(f"[pillar_a] cache size mismatch ({len(arr)} vs {len(texts)}) - re-embedding")

    from sentence_transformers import SentenceTransformer
    print(f"[pillar_a] loading encoder {encoder_hf_id}")
    model = SentenceTransformer(encoder_hf_id, device=device)
    t0 = time.time()
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=False,
                       convert_to_numpy=True, normalize_embeddings=True)
    dt = time.time() - t0
    print(f"[pillar_a] embedded {len(texts)} texts in {dt:.1f}s  ({len(texts)/dt:.0f}/s)  "
          f"shape={emb.shape}  -> {_display_path(cache)}")
    np.save(cache, emb.astype(np.float32))
    return emb.astype(np.float32)


def _readability_features(texts):
    """Compute the small readability feature set."""
    out = np.zeros((len(texts), len(STATIC_TOP5)), dtype=np.float32)
    for i, t in enumerate(texts):
        f = static_features(t)
        for j, name in enumerate(STATIC_TOP5):
            v = f.get(name, 0.0)
            out[i, j] = 0.0 if (v is None or np.isnan(v)) else v
    return out


def _train_logreg(X_tr_s, y_tr, X_va_s, y_va, seed=42):
    """Train the classifier and return validation macro-F1."""
    model = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0,
                               random_state=seed)
    model.fit(X_tr_s, y_tr)
    val_pred = model.predict(X_va_s)
    val_f1 = float(f1_score(y_va, val_pred, average='macro', labels=[0, 1, 2],
                             zero_division=0))
    return model, val_f1


def _measure_inference_latency(encoder_hf_id, encoder_short, sample_texts,
                                clf, scaler, with_static, n_warmup=5, n_timed=50):
    """Measure encode-plus-predict latency."""
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(encoder_hf_id, device=device)
    times = []
    # warmup
    for t in sample_texts[:n_warmup]:
        emb = model.encode([t], show_progress_bar=False, convert_to_numpy=True,
                           normalize_embeddings=True)
        if with_static:
            stat = _readability_features([t])
            X = np.concatenate([emb, stat], axis=1)
        else:
            X = emb
        Xs = scaler.transform(X)
        _ = clf.predict(Xs)
    # timed
    for t in sample_texts[n_warmup:n_warmup + n_timed]:
        t0 = time.perf_counter()
        emb = model.encode([t], show_progress_bar=False, convert_to_numpy=True,
                           normalize_embeddings=True)
        if with_static:
            stat = _readability_features([t])
            X = np.concatenate([emb, stat], axis=1)
        else:
            X = emb
        Xs = scaler.transform(X)
        _ = clf.predict(Xs)
        times.append(time.perf_counter() - t0)
    return {
        'median_ms': float(np.median(times) * 1000),
        'p90_ms':    float(np.quantile(times, 0.9) * 1000),
        'n_warmup':  n_warmup,
        'n_timed':   n_timed,
        'device':    device,
    }


def evaluate_one_combo(encoder_short, encoder_hf_id, with_static, all_data,
                       seed=42):
    """Train + evaluate one (encoder, variant) combination.

    `all_data`: dict with keys 'train', 'val', and one per canonical eval split,
                each mapping to (X_emb, X_static, y).
    Returns a dict ready to dump to JSON.
    """
    variant = 'emb_plus_static5' if with_static else 'emb_only'
    print(f"\n[pillar_a] === {encoder_short} :: {variant} ===")

    def _stack(name):
        emb, stat, y = all_data[name]
        X = np.concatenate([emb, stat], axis=1) if with_static else emb
        return X, y

    X_tr, y_tr = _stack('train')
    X_va, y_va = _stack('val')

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)

    t0 = time.time()
    clf, val_f1 = _train_logreg(X_tr_s, y_tr, X_va_s, y_va, seed=seed)
    fit_sec = time.time() - t0
    print(f"[pillar_a]   fit in {fit_sec:.1f}s, val_f1 = {val_f1:.4f}")

    # Per-split eval
    per_split = {'val': _eval_block(y_va, clf.predict(X_va_s))}
    per_split['val']['f1_macro_train'] = val_f1  # already same

    for split in CANONICAL_SPLITS:
        if split not in all_data:
            continue
        X, y = _stack(split)
        Xs = scaler.transform(X).astype(np.float32)
        pred = clf.predict(Xs)
        per_split[split] = _eval_block(y, pred)
        # Synthetic vs gen labels — same predictions, different ground truth.
        if split == 'ood_synthetic':
            y_g = all_data['ood_synthetic_gen'][2]  # the (emb, stat, y) tuple's labels
            per_split['ood_synthetic_vs_gen'] = _eval_block(y_g, pred)
        print(f"[pillar_a]   {split}: f1_macro = {per_split[split]['f1_macro']:.4f}")
        if split == 'test':
            # balanced subsample (5 seeds, 50 per-class)
            texts_t = all_data['__texts__']['test']
            f1s = []
            for s in range(5):
                _, y_bal, idx_bal = _balanced_subsample(texts_t, y,
                                                        per_class=50,
                                                        seed=42 + s)
                f1s.append(float(f1_score(y_bal, pred[idx_bal], average='macro',
                                           labels=[0, 1, 2], zero_division=0)))
            per_split['balanced_test'] = {
                'f1_macro_mean': float(np.mean(f1s)),
                'f1_macro_std':  float(np.std(f1s)),
                'f1_macro_per_seed': f1s,
                'n_resamples': 5, 'per_class_n': 50,
            }
            print(f"[pillar_a]   balanced_test (5 × 150): "
                  f"{per_split['balanced_test']['f1_macro_mean']:.4f}")

    # AdvConcept-50
    X_adv, y_adv = _stack('adv_concept')
    Xs_adv = scaler.transform(X_adv).astype(np.float32)
    pred_adv = clf.predict(Xs_adv)
    adv_block = _eval_block(y_adv, pred_adv)
    adv_df = pd.read_csv(ADV_CSV)
    cat_acc = {}
    for cat in adv_df['category'].unique():
        mask = (adv_df['category'] == cat).to_numpy()
        cat_acc[cat] = float(accuracy_score(y_adv[mask], pred_adv[mask]))
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
        'overall_accuracy':      adv_block['accuracy'],
        'overall_f1_macro':      adv_block['f1_macro'],
        'per_category_accuracy': cat_acc,
        'conf_matrix':           adv_block['conf_matrix'],
        'per_row':               per_row,
    }
    print(f"[pillar_a]   adv_concept: acc = {adv_block['accuracy']:.4f}")
    for cat, acc in cat_acc.items():
        print(f"[pillar_a]     {cat}: {acc:.4f}")

    # Save artifacts
    name = f'pillar_a__{encoder_short}__{variant}'
    pkl_path = os.path.join(MODEL_DIR, f'{name}.pkl')
    with open(pkl_path, 'wb') as fh:
        pickle.dump({
            'kind':            'pillar_a',
            'encoder_short':   encoder_short,
            'encoder_hf_id':   encoder_hf_id,
            'variant':         variant,
            'with_static':     with_static,
            'static_features': STATIC_TOP5 if with_static else [],
            'scaler':          scaler,
            'clf':             clf,
            'n_params_clf':    int(clf.coef_.size + clf.intercept_.size),
            'val_f1':          val_f1,
            'fit_sec':         fit_sec,
        }, fh)
    print(f"[pillar_a]   saved {_display_path(pkl_path)}")

    # Latency probe (on test set sample)
    latency = _measure_inference_latency(
        encoder_hf_id, encoder_short,
        all_data['__texts__']['test'][:60],  # 5 warmup + 50 timed → need at least 55
        clf, scaler, with_static)

    report = {
        'kind':           'pillar_a',
        'name':           name,
        'encoder_short':  encoder_short,
        'encoder_hf_id':  encoder_hf_id,
        'variant':        variant,
        'with_static':    with_static,
        'static_features': STATIC_TOP5 if with_static else [],
        'feature_dim':    int(X_tr.shape[1]),
        'n_params_clf':   int(clf.coef_.size + clf.intercept_.size),
        'val_f1':         val_f1,
        'fit_sec':        fit_sec,
        'per_split':      per_split,
        'adv_concept':    adv_report,
        'latency':        latency,
    }
    out_path = os.path.join(REPORT_DIR, f'{name}.eval.json')
    with open(out_path, 'w') as fh:
        json.dump(report, fh, indent=2)
    print(f"[pillar_a]   wrote {_display_path(out_path)}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoders', nargs='+', default=[e[0] for e in ENCODERS],
                    help='Encoder short names to run (subset of ' +
                         ', '.join(e[0] for e in ENCODERS) + ')')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='pillar_a_lean_combo',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        # Load all texts + labels + readability features once (they're encoder-independent).
        print("[pillar_a] loading texts + labels + readability for all splits ...")
        texts = {}
        labels = {}
        for split in ['train', 'val'] + CANONICAL_SPLITS:
            tx, y = _load_split_texts_and_labels(split, label_source='judge')
            texts[split], labels[split] = tx, y
        # Synthetic with gen labels (paper headline target)
        _, y_gen = _load_split_texts_and_labels('ood_synthetic', label_source='gen')
        labels['ood_synthetic_gen'] = y_gen

        # AdvConcept-50
        adv_df = pd.read_csv(ADV_CSV)
        texts['adv_concept'] = adv_df['text'].astype(str).tolist()
        labels['adv_concept'] = adv_df['true_level'].map(LABEL_MAP).astype(np.int64).to_numpy()

        readability = {sp: _readability_features(t) for sp, t in texts.items()}
        for sp, r in readability.items():
            print(f"[pillar_a]   readability[{sp}] shape={r.shape}")

        # Encoder bake-off
        all_reports = []
        for short, hf_id, n_params, dim in ENCODERS:
            if short not in args.encoders:
                continue
            print(f"\n[pillar_a] === embedding pass: {short} ({n_params}M, {dim}-dim) ===")
            embs = {sp: _embed(hf_id, short, t, sp) for sp, t in texts.items()}

            # Bundle the data for evaluate_one_combo
            all_data = {sp: (embs[sp], readability[sp], labels[sp])
                         for sp in texts.keys()}
            all_data['ood_synthetic_gen'] = (embs['ood_synthetic'],
                                              readability['ood_synthetic'],
                                              labels['ood_synthetic_gen'])
            all_data['__texts__'] = texts

            for with_static in (False, True):
                rep = evaluate_one_combo(short, hf_id, with_static, all_data,
                                          seed=args.seed)
                rep['encoder_size_m'] = n_params
                rep['encoder_dim'] = dim
                all_reports.append(rep)

        # Pick winner by synthetic-OOD vs_gen F1 (the headline target)
        def _key(r):
            return r['per_split'].get('ood_synthetic_vs_gen', {}).get('f1_macro', 0.0)
        winner = max(all_reports, key=_key) if all_reports else None
        summary = {
            'kind': 'pillar_a_summary',
            'n_candidates': len(all_reports),
            'winner_by_synthetic_vs_gen':
                {'name': winner['name'],
                 'f1_synthetic_vs_gen': _key(winner)} if winner else None,
            'candidates': [
                {
                    'name': r['name'],
                    'encoder_short': r['encoder_short'],
                    'variant': r['variant'],
                    'encoder_size_m': r['encoder_size_m'],
                    'feature_dim': r['feature_dim'],
                    'val_f1': r['val_f1'],
                    'test_f1': r['per_split'].get('test', {}).get('f1_macro'),
                    'balanced_test_f1_mean': r['per_split'].get('balanced_test', {}).get('f1_macro_mean'),
                    'synthetic_judge_f1': r['per_split'].get('ood_synthetic', {}).get('f1_macro'),
                    'synthetic_gen_f1': r['per_split'].get('ood_synthetic_vs_gen', {}).get('f1_macro'),
                    'race_middle_f1': r['per_split'].get('ood_race-middle', {}).get('f1_macro'),
                    'race_high_f1': r['per_split'].get('ood_race-high', {}).get('f1_macro'),
                    'onestop_f1': r['per_split'].get('ood_onestop', {}).get('f1_macro'),
                    'adv_concept_overall': r['adv_concept']['overall_accuracy'],
                    'adv_per_category': r['adv_concept']['per_category_accuracy'],
                    'latency_median_ms': r['latency']['median_ms'],
                }
                for r in all_reports
            ],
        }
        out_path = os.path.join(REPORT_DIR, 'pillar_a_summary.json')
        with open(out_path, 'w') as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n[pillar_a] wrote {_display_path(out_path)}")
        if winner:
            print(f"[pillar_a] WINNER: {winner['name']} "
                  f"(synthetic_vs_gen F1 = {_key(winner):.4f})")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
