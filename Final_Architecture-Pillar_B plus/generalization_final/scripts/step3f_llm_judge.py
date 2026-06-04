"""
Step 3f — LLM-as-judge.

Uses Llama-3.1-8B-Instruct (via vLLM, bf16) to classify every text into one of
{elementary, middle, high} with a single uniform rubric. This replaces the
per-corpus original labels with a label set that is internally consistent
across corpora — the central methodological move of the LLM-as-judge approach.

We read each split CSV from outputs/splits/ (the post-holdout splits) and
write outputs/llm_judge/<split>_judge.csv with columns:
    orig_split, orig_idx, source_dataset, education_level (= original label),
    llm_judge_label, judge_raw_response

Notes
-----
* Llama-3.1-8B is gated; you must have `huggingface-cli login` set up.
* We use vLLM's chat template via apply_chat_template — Llama-3.1 expects it.
* Greedy decoding, max_tokens=8 (one word + slack).
* Parse: first word of stripped output, lowercased, mapped via prefix:
    'elem' -> elementary, 'mid' -> middle, 'high' -> high.
* On parse failure: marked as 'elementary' to mirror Rooein et al. 2024's
  invalid-response handling (so the row is still usable).
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
from codecarbon import EmissionsTracker
from tqdm.auto import tqdm

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
SPLITS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'splits')
OUTPUT_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'llm_judge')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

JUDGE_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
JUDGE_TAG   = 'llama3.1-8b'

RUBRIC = (
    "Classify the following text by its target reader's US education level.\n"
    "Choose exactly one of:\n"
    "- elementary  (US grades 1-5, simple vocabulary, short sentences)\n"
    "- middle      (US grades 6-8)\n"
    "- high        (US grades 9-12, advanced vocabulary, complex ideas)\n\n"
    "Text:\n{text}\n\n"
    "Reply with one word only: elementary, middle, or high."
)


def _parse_label(raw):
    s = str(raw).strip().lower()
    s = re.sub(r'^[^a-z]+', '', s)  # strip punctuation / quotes / markdown
    if s.startswith('elem'): return 'elementary'
    if s.startswith('mid'):  return 'middle'
    if s.startswith('high'): return 'high'
    return 'elementary'  # invalid -> default (mirrors Rooein 2024 handling)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-dir', default=SPLITS_DIR)
    ap.add_argument('--output-dir', default=OUTPUT_DIR)
    ap.add_argument('--splits', nargs='+',
                    default=['train', 'val', 'test',
                             'ood_onestop', 'ood_race-middle', 'ood_race-high'])
    ap.add_argument('--max-model-len', type=int, default=2048)
    ap.add_argument('--max-tokens', type=int, default=8)
    ap.add_argument('--gpu-mem-frac', type=float, default=0.90)
    args = ap.parse_args()
    os.makedirs(args.output_dir,  exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step3f_llm_judge',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"[step3f] loading judge model {JUDGE_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
        llm = LLM(model=JUDGE_MODEL, dtype='bfloat16',
                  enable_prefix_caching=True,
                  gpu_memory_utilization=args.gpu_mem_frac,
                  max_model_len=args.max_model_len,
                  trust_remote_code=True)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

        # Pre-truncate text portion to fit in the model's context window.
        # Compute exact overhead: render an empty-text version of the full
        # chat template (with role markers, eot tokens, generation prompt)
        # and tokenize that — accounts for chat-template overhead exactly,
        # not just the rubric text.
        empty_msgs   = [{'role': 'user', 'content': RUBRIC.format(text='')}]
        empty_prompt = tokenizer.apply_chat_template(empty_msgs, tokenize=False,
                                                     add_generation_prompt=True)
        overhead_tokens = len(tokenizer(empty_prompt,
                                        add_special_tokens=False)['input_ids'])
        budget = args.max_model_len - overhead_tokens - args.max_tokens - 8
        if budget < 64:
            sys.exit(f"[step3f] computed text budget {budget} is too small; "
                     f"overhead={overhead_tokens} max_new={args.max_tokens}")
        print(f"[step3f] chat-template overhead={overhead_tokens} "
              f"-> max_text_tokens={budget}")

        def build_prompt(text):
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            if len(ids) > budget:
                text = tokenizer.decode(ids[:budget], skip_special_tokens=True)
            msgs = [{'role': 'user', 'content': RUBRIC.format(text=text)}]
            return tokenizer.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True)

        for split in args.splits:
            csv_in = os.path.join(args.splits_dir, f'{split}.csv')
            if not os.path.exists(csv_in):
                print(f"[step3f] skip {split} (missing {csv_in})")
                continue
            out_path = os.path.join(args.output_dir, f'{split}_judge.csv')
            if os.path.exists(out_path):
                done = pd.read_csv(out_path)
                if len(done) == len(pd.read_csv(csv_in)):
                    print(f"[step3f] {split}: already complete ({len(done)} rows), skipping")
                    continue

            df = pd.read_csv(csv_in)
            prompts = [build_prompt(str(t)) for t in df['full_text'].astype(str).tolist()]
            print(f"[step3f] {split}: judging {len(prompts)} texts")
            outputs = llm.generate(prompts, sampling, use_tqdm=True)
            raws = [o.outputs[0].text for o in outputs]
            labels = [_parse_label(r) for r in raws]

            rec = pd.DataFrame({
                'orig_split':      df['orig_split'] if 'orig_split' in df.columns else split,
                'orig_idx':        df['orig_idx']   if 'orig_idx'   in df.columns else df.index,
                'source_dataset':  df['source_dataset'].astype(str),
                'education_level': df['education_level'].astype(str),  # original
                'llm_judge_label': labels,
                'judge_raw_response': raws,
            })
            rec.to_csv(out_path, index=False)
            agree = (rec['education_level'] == rec['llm_judge_label']).mean()
            dist = dict(rec['llm_judge_label'].value_counts())
            print(f"[step3f] {split}: wrote {out_path}; "
                  f"agreement-with-original={agree:.3f}; judge_dist={dist}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
