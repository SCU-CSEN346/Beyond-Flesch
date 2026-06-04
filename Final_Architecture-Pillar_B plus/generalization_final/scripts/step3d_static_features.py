"""
Step 3d — Static (readability + lexical + structural) features.

Computes ~40 features per row, matching the Rooein et al. 2024 STATIC feature
set as closely as we can with the textstat library + a few manual counts.

For every split CSV in the curated dataset, writes
    outputs/static_features/<split>_static.csv
with columns:
    text_idx, stat_0, stat_1, ..., stat_K
plus a sidecar list of feature names in
    outputs/static_features/feature_names.json
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from math import log2

import numpy as np
import pandas as pd
import textstat
from codecarbon import EmissionsTracker
from tqdm.auto import tqdm

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
INPUT_DIR    = os.path.join(_PROJECT_DIR, 'inputs', 'curated_generalization_collection')
OUTPUT_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'static_features')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

# Readability indices from textstat. One feature per index.
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
            # text_standard returns "7th and 8th grade" — take the first number.
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
    # 12 readability indices via textstat.
    for name in READABILITY:
        feats[f'stat_{name}'] = _safe(getattr(textstat, name), text)

    # Manual lexical / structural features.
    words = WORD_RE.findall(text)
    sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]
    n_words = len(words)
    n_sents = max(len(sentences), 1)

    feats['stat_char_count']        = len(text)
    feats['stat_word_count']        = n_words
    feats['stat_unique_words']      = len(set(w.lower() for w in words))
    feats['stat_ttr']               = feats['stat_unique_words'] / n_words if n_words else 0.0
    feats['stat_avg_word_len']      = float(np.mean([len(w) for w in words])) if words else 0.0
    feats['stat_long_word_ratio']   = sum(1 for w in words if len(w) > 6) / n_words if n_words else 0.0
    feats['stat_sentence_count']    = len(sentences)
    feats['stat_avg_sent_len']      = n_words / n_sents
    feats['stat_max_sent_len']      = max((len(WORD_RE.findall(s)) for s in sentences), default=0)
    feats['stat_word_entropy']      = _word_entropy(words)

    feats['stat_n_question']        = text.count('?')
    feats['stat_n_exclaim']         = text.count('!')
    feats['stat_n_comma']           = text.count(',')
    feats['stat_n_semicolon']       = text.count(';')
    feats['stat_n_paren']           = text.count('(') + text.count(')')
    feats['stat_n_quote']           = text.count('"')

    feats['stat_n_paragraphs']      = text.count('\n\n') + 1
    feats['stat_n_newlines']        = text.count('\n')

    lower = text.lower()
    feats['stat_has_because']       = float(' because' in lower)
    feats['stat_has_therefore']     = float('therefore' in lower)
    feats['stat_has_example']       = float('example' in lower or 'e.g.' in lower)
    feats['stat_has_multiple_choice'] = float(any(m in lower for m in [' (a)', ' (b)', ' (c)']))

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir',  default=INPUT_DIR)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    ap.add_argument('--splits', nargs='+',
                    default=['train', 'val', 'test', 'ood_race-middle', 'ood_race-high'])
    args = ap.parse_args()
    os.makedirs(args.output_dir,  exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step3d_static_features',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        feature_names = None
        for split in args.splits:
            csv_in = os.path.join(args.input_dir, f'{split}.csv')
            if not os.path.exists(csv_in):
                sys.exit(f"[step3d] missing input {csv_in}")
            df = pd.read_csv(csv_in)
            rows = []
            for i, text in enumerate(tqdm(df['full_text'].astype(str).tolist(),
                                          desc=f'static/{split}')):
                feats = static_features(text)
                feats['text_idx'] = i
                rows.append(feats)
            out = pd.DataFrame(rows).fillna(0.0)
            # Stable column order: text_idx first, then alphabetical feature names.
            cols = ['text_idx'] + sorted(c for c in out.columns if c != 'text_idx')
            out = out[cols]
            if feature_names is None:
                feature_names = [c for c in cols if c.startswith('stat_')]
            out_path = os.path.join(args.output_dir, f'{split}_static.csv')
            out.to_csv(out_path, index=False)
            print(f"[step3d] wrote {out_path}: rows={len(out)} features={len(feature_names)}")

        with open(os.path.join(args.output_dir, 'feature_names.json'), 'w') as fh:
            json.dump(feature_names, fh, indent=2)
        print(f"[step3d] wrote feature_names.json ({len(feature_names)} names)")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
