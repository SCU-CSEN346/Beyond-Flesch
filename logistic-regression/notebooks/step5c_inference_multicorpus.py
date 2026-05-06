"""
Step 5c — Multi-corpus prompt inference (extends Step 5b)

Runs prompt feature extraction on the unified train/val/test/OOD splits
generated in Step 2b.

Key differences from Step 5b:
- Reads inputs from `outputs/multi_corpus/<split>.csv` (not `data/train.csv`)
- Optionally adds 10 domain-neutral prompts from
  `outputs/prompt_metrics/extra_generic_prompts.json` to improve generalization
  beyond ScienceQA-style prompts

Outputs:
- One CSV per split per model:
  `outputs/prompt_metrics/<split>_prompt_metrics_<model>__multi.csv`

Usage:
- Base 63 prompts (paper setting):
    python step5c_inference_multicorpus.py --model qwen2.5-7b --split train

- With 10 extra generic prompts (~16% more compute):
    python step5c_inference_multicorpus.py --model qwen2.5-7b --split train --extra

- OOD split:
    python step5c_inference_multicorpus.py --model qwen2.5-7b --split ood_onestop

- All splits:
    for s in train val test ood_onestop ood_clear ood_race; do
        python step5c_inference_multicorpus.py --model qwen2.5-7b --split $s --extra
    done
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = os.environ.get('PROJECT_ROOT', os.path.dirname(os.path.abspath(__file__)))
MULTI_DIR    = os.path.join(PROJECT_ROOT, 'outputs', 'multi_corpus')
PROMPT_DIR   = os.path.join(PROJECT_ROOT, 'outputs', 'prompt_metrics')
EXTRA_PATH   = os.path.join(PROMPT_DIR, 'extra_generic_prompts.json')
PAPER_PATH   = os.path.join(PROMPT_DIR, 'prompt_questions.json')


def _load_prompts(use_extra):
    """Return paper's 63 + optional 10 generic prompts. Order matters because
    column names are positional (`prompt_0` ... `prompt_N`)."""
    with open(PAPER_PATH) as fh:
        prompts = list(json.load(fh))
    if use_extra:
        if not os.path.exists(EXTRA_PATH):
            sys.exit(f"[step5c] {EXTRA_PATH} missing. Was --extra requested without "
                     f"the JSON file?")
        with open(EXTRA_PATH) as fh:
            prompts.extend(json.load(fh))
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='Model key from step5b_inference_generic.MODEL_REGISTRY.')
    ap.add_argument('--split', required=True,
                    help='CSV basename in outputs/multi_corpus/ (without .csv).')
    ap.add_argument('--extra', action='store_true',
                    help='Append the 10 generic domain-neutral prompts.')
    ap.add_argument('--chat', action='store_true')
    ap.add_argument('--quant', choices=['int8', 'int4', 'bf16', 'fp16'], default='int8')
    if any(a.endswith('.json') or a.startswith('-f') for a in sys.argv[1:]):
        sys.exit("[step5c] running under Jupyter — set sys.argv before main().")
    args = ap.parse_args()

    # Late imports — heavy.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from step5b_inference_generic import (
        MODEL_REGISTRY, build_prompt_raw, build_prompt_chat,
        get_llm_responses_batch, parse_yes_no, pick_batch_size,
        _safe_gpu_mem_gb, _bf16_supported, _size_b_from_key,
    )

    if args.model not in MODEL_REGISTRY:
        sys.exit(f"[step5c] unknown model {args.model}. "
                 f"Available: {list(MODEL_REGISTRY)}")
    csv_in = os.path.join(MULTI_DIR, f'{args.split}.csv')
    if not os.path.exists(csv_in):
        sys.exit(f"[step5c] {csv_in} missing. Run Step 2c first.")

    suffix = '__multi_extra' if args.extra else '__multi'
    out_path = os.path.join(PROMPT_DIR,
                            f'{args.split}_prompt_metrics_{args.model}{suffix}.csv')
    # Skip if already complete — avoids re-doing every prompt on every re-run
    # of the bash for-loop the user wraps around step5c.
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        df_check = pd.read_csv(csv_in)
        if len(existing) == len(df_check):
            print(f"[step5c] {out_path} already complete ({len(existing)} rows) — skipping.")
            return
        else:
            print(f"[step5c] {out_path} exists but row count mismatch "
                  f"({len(existing)} vs {len(df_check)}) — re-running.")

    df = pd.read_csv(csv_in)
    prompts_list = _load_prompts(args.extra)
    _, gated = MODEL_REGISTRY[args.model]
    if gated:
        print(f"[step5c] WARNING — {args.model} is gated. "
              f"Run `huggingface-cli login` first if you haven't.")
    print(f"[step5c] split={args.split} model={args.model} "
          f"template={'chat' if args.chat else 'raw'} quant={args.quant} "
          f"rows={len(df)} prompts={len(prompts_list)} "
          f"(extra={args.extra}) calls={len(df)*len(prompts_list):,}")

    gpu_mem_gb = _safe_gpu_mem_gb()
    if gpu_mem_gb == 0:
        sys.exit("[step5c] No CUDA GPU detected.")

    # Model load.
    model_id, _ = MODEL_REGISTRY[args.model]
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16
    if args.quant == 'int8':
        load_kwargs = {'quantization_config': BitsAndBytesConfig(load_in_8bit=True)}
    elif args.quant == 'int4':
        load_kwargs = {'quantization_config':
                       BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=compute_dtype)}
    elif args.quant == 'fp16':
        load_kwargs = {'torch_dtype': torch.float16}
    else:
        load_kwargs = {'torch_dtype': compute_dtype}
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map='auto', trust_remote_code=True, **load_kwargs)
    model.eval()

    BATCH = pick_batch_size(gpu_mem_gb, _size_b_from_key(args.model))
    build = (lambda t, q: build_prompt_chat(tokenizer, t, q)) if args.chat else build_prompt_raw
    MAX_NEW = 20 if args.chat else 10

    progress_path = out_path.replace('.csv', '_progress.csv')
    sidecar_path  = out_path.replace('.csv', '_progress.meta.json')
    current_meta = {'model': args.model, 'split': args.split,
                    'template': 'chat' if args.chat else 'raw',
                    'quant': args.quant, 'extra': bool(args.extra)}
    if os.path.exists(progress_path):
        existing = pd.read_csv(progress_path)
        start_idx = len(existing)
        results = existing.to_dict('records')
        # Refuse to mix templates / quant on resume — silent CSV corruption.
        if os.path.exists(sidecar_path):
            with open(sidecar_path) as fh:
                prior = json.load(fh)
            for k, v in current_meta.items():
                if prior.get(k) != v:
                    sys.exit(f"[step5c] Resume aborted: {k} changed "
                             f"({prior.get(k)} -> {v}). Delete {progress_path} "
                             f"to start fresh, or rerun with original {k}.")
        print(f"[step5c] RESUMING from {start_idx}/{len(df)}")
    else:
        start_idx, results = 0, []
    with open(sidecar_path, 'w') as fh:
        json.dump(current_meta, fh)

    SAVE_EVERY = 50
    t0 = time.time()
    done = 0
    for text_idx in tqdm(range(start_idx, len(df)), desc=f'{args.model}/{args.split}'):
        text = str(df.iloc[text_idx]['full_text'])
        rec = {'text_idx': text_idx,
               'education_level': df.iloc[text_idx]['education_level'],
               'source_dataset':  df.iloc[text_idx].get('source_dataset', ''),
               'domain':          df.iloc[text_idx].get('domain', '')}
        prompts = [build(text, q) for q in prompts_list]
        for bs in range(0, len(prompts), BATCH):
            chunk = prompts[bs:bs+BATCH]
            resps = get_llm_responses_batch(tokenizer, model, chunk, max_new_tokens=MAX_NEW)
            for j, r in enumerate(resps):
                rec[f'prompt_{bs+j}'] = parse_yes_no(r)
        results.append(rec)
        done += 1
        if (text_idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            elapsed = max(time.time() - t0, 1e-6)
            rate = max(done / elapsed, 1e-6)
            eta_min = (len(df) - start_idx - done) / rate / 60
            print(f"  ckpt {text_idx+1}/{len(df)}  {rate:.2f}/s  eta~{eta_min:.0f}min")

    pd.DataFrame(results).to_csv(out_path, index=False)
    # Clean up sidecars on successful completion.
    for p in (progress_path, sidecar_path):
        if os.path.exists(p):
            os.remove(p)
    print(f"[step5c] wrote {out_path}")


if __name__ == '__main__':
    main()
