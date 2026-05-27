"""
Generate concept-preserving paraphrases for data augmentation.

For a stratified sample of training rows, the generator rewrites the text while
keeping the underlying concept and curriculum level fixed. The saved label is
kept as-is; this script changes wording, not supervision.

The goal is to give Pillar B+ more surface variation around the same concepts.
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR     = os.path.join(_PROJECT_DIR, 'data')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

sys.path.insert(0, _SCRIPT_DIR)
from evaluate_all import GF_OUT, LABEL_MAP, LABEL_NAMES  # noqa: E402

GEN_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_MODEL_LEN = 2048

# Surface-variation directions, assigned randomly per row.
DIRECTIONS = {
    'wordier': (
        "Make the rewrite noticeably LONGER and MORE elaborate (more descriptive "
        "phrases, more sentences) while preserving the exact same concept and "
        "curriculum level."
    ),
    'concise': (
        "Make the rewrite noticeably SHORTER and more direct (drop filler phrases, "
        "combine sentences) while preserving the exact same concept and curriculum "
        "level."
    ),
    'simpler_vocab': (
        "Keep the same meaning but use SIMPLER everyday vocabulary throughout, "
        "while preserving the exact same concept and curriculum level."
    ),
    'complex_vocab': (
        "Keep the same meaning but use MORE ADVANCED vocabulary throughout, "
        "while preserving the exact same concept and curriculum level."
    ),
}


def build_prompt(text, direction_instruction):
    return (
        "You will rewrite a piece of text used in a school reading curriculum. "
        "Rewrite it so the underlying concept (what is being taught) and "
        "the curriculum grade level required to understand it stay EXACTLY "
        "the same, but vary the surface form.\n\n"
        f"Specifically: {direction_instruction}\n\n"
        "Original text:\n"
        f"{text}\n\n"
        "Output ONLY the rewritten text. Do not include any preamble, "
        "explanation, label, quote marks, or markdown."
    )


def _load_train():
    """Load train rows with their judge labels."""
    train_csv = pd.read_csv(os.path.join(GF_OUT, 'splits', 'train.csv'))
    judge_csv = pd.read_csv(os.path.join(GF_OUT, 'llm_judge', 'train_judge.csv'))
    text_col = 'text' if 'text' in train_csv.columns else 'full_text'
    df = pd.DataFrame({
        'orig_idx':         np.arange(len(train_csv)),
        'text':             train_csv[text_col].astype(str).values,
        'llm_judge_label':  judge_csv['llm_judge_label'].astype(str).values,
    })
    return df


def _stratified_subsample(df, n_total, seed=42):
    """Sample rows while keeping the class mix close to the original."""
    rng = np.random.default_rng(seed)
    labels = df['llm_judge_label'].unique()
    parts = []
    for lbl in labels:
        sub = df[df['llm_judge_label'] == lbl]
        share = int(round(n_total * len(sub) / len(df)))
        share = min(share, len(sub))
        if share == 0:
            continue
        idx = rng.choice(len(sub), size=share, replace=False)
        parts.append(sub.iloc[idx])
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=2000,
                    help='number of paraphrases to generate')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gpu-mem-frac', type=float, default=0.55)
    ap.add_argument('--max-tokens', type=int, default=400)
    ap.add_argument('--temperature', type=float, default=0.8)
    args = ap.parse_args()

    os.makedirs(DATA_DIR,      exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    df = _load_train()
    print(f"[gen_cf] loaded {len(df)} train rows. "
          f"class dist: {dict(df['llm_judge_label'].value_counts())}")

    sub = _stratified_subsample(df, args.n, seed=args.seed)
    print(f"[gen_cf] sampled {len(sub)} rows. "
          f"class dist: {dict(sub['llm_judge_label'].value_counts())}")

    rng = random.Random(args.seed)
    direction_keys = list(DIRECTIONS.keys())
    sub = sub.copy()
    sub['direction'] = [rng.choice(direction_keys) for _ in range(len(sub))]

    prompts = [build_prompt(t, DIRECTIONS[d])
               for t, d in zip(sub['text'].tolist(), sub['direction'].tolist())]

    tracker = EmissionsTracker(project_name='generate_counterfactuals',
                                output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"[gen_cf] loading {GEN_MODEL} via vLLM ...")
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
        llm = LLM(model=GEN_MODEL, dtype='bfloat16',
                  enable_prefix_caching=True,
                  gpu_memory_utilization=args.gpu_mem_frac,
                  max_model_len=MAX_MODEL_LEN,
                  trust_remote_code=True,
                  seed=args.seed)
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{'role': 'user', 'content': p}],
                tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
        sp = SamplingParams(temperature=args.temperature, top_p=0.95,
                             max_tokens=args.max_tokens, seed=args.seed)

        t0 = time.time()
        print(f"[gen_cf] generating {len(chat_prompts)} paraphrases ...")
        outs = llm.generate(chat_prompts, sp, use_tqdm=True)
        wall = time.time() - t0
        print(f"[gen_cf] wall-clock {wall/60:.1f} min  "
              f"({len(outs)/max(wall,1e-3):.1f} req/s)")

        paraphrases = [o.outputs[0].text.strip() for o in outs]
    finally:
        kwh = tracker.stop()
        print(f"[gen_cf] emissions kWh = {kwh}")

    out = pd.DataFrame({
        'orig_idx':        sub['orig_idx'].values,
        'text':            paraphrases,
        'llm_judge_label': sub['llm_judge_label'].values,
        'direction':       sub['direction'].values,
        'gen_seed':        args.seed,
    })

    # Drop obvious failures.
    pre = len(out)
    out = out[out['text'].str.len() >= 20].reset_index(drop=True)
    print(f"[gen_cf] dropped {pre - len(out)} failed/empty paraphrases. "
          f"final n = {len(out)}")
    print(f"[gen_cf] class dist (after drop): "
          f"{dict(out['llm_judge_label'].value_counts())}")
    print(f"[gen_cf] direction dist: "
          f"{dict(out['direction'].value_counts())}")

    out_csv = os.path.join(DATA_DIR, 'counterfactuals.csv')
    out.to_csv(out_csv, index=False)
    print(f"[gen_cf] wrote {os.path.relpath(out_csv, _PROJECT_DIR)}")


if __name__ == '__main__':
    main()
