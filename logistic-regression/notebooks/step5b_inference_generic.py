"""
Step 5b — Generic LLM inference (newer models, post-paper)

Same loop as step5_gemma_inference.py / step5_inference_mistral_7b.py but
parameterised so we can swap in newer instruction-tuned LLMs without copying
the whole script.

Pick a model with the MODEL env var, or edit DEFAULT_MODEL_KEY below.

"""


import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSIFICATION_ROOT = os.environ.get('CLASSIFICATION_ROOT', os.path.join(_SCRIPT_DIR, 'text-difficulty-classification'))
DATA_DIR     = os.path.join(CLASSIFICATION_ROOT, 'data')
PROMPT_DIR   = os.path.join(CLASSIFICATION_ROOT, 'outputs', 'prompt_metrics')
os.makedirs(PROMPT_DIR, exist_ok=True)



MODEL_REGISTRY = {
    # Paper-era (kept for re-runs / sanity)
    'llama2-7b':   ('meta-llama/Llama-2-7b-chat-hf',          True),
    'llama2-13b':  ('meta-llama/Llama-2-13b-chat-hf',         True),
    'mistral-7b':  ('mistralai/Mistral-7B-Instruct-v0.2',     False),
    'gemma-7b':    ('google/gemma-7b-it',                     True),

    # Extension (newer)
    'llama3.1-8b': ('meta-llama/Llama-3.1-8B-Instruct',       True),
    'llama3.3-70b':('meta-llama/Llama-3.3-70B-Instruct',      True),
    'qwen2.5-7b':  ('Qwen/Qwen2.5-7B-Instruct',               False),
    'qwen2.5-32b': ('Qwen/Qwen2.5-32B-Instruct',              False),
    'gemma2-9b':   ('google/gemma-2-9b-it',                   True),
    'mistral-nemo':('mistralai/Mistral-Nemo-Instruct-2407',   False),
    'phi-4-mini':  ('microsoft/Phi-3.5-mini-instruct',        False),
    'deepseek-v2-lite':('deepseek-ai/DeepSeek-V2-Lite-Chat',  False),
}

DEFAULT_MODEL_KEY = os.environ.get('MODEL', 'qwen2.5-7b')


def parse_yes_no(response):
    """Three-way: 1=yes, 0=no, -1=hedge/refusal/unparseable.

    Step 6 / Step 7 will impute -1 -> column mean (or drop the row), so a model
    that refuses 30% of the time doesn't look like a model with 30% no-rate.

    Strips common chat-model decorations (markdown bold, bullets) before
    matching. Avoids the "no doubt" / "no problem" false-negative by also
    requiring the 'no' to be a word-boundary match.
    """
    import re as _re
    r = response.strip().lower()
    # Strip leading markdown / quote / bullet noise.
    r = _re.sub(r'^[\*_`>\-\s"\']+', '', r)
    head = r[:30]
    if head.startswith('yes'):                       return 1
    if head.startswith('no'):                        return 0
    has_yes = bool(_re.search(r'\byes\b', head))
    has_no  = bool(_re.search(r'\bno\b',  head))
    if has_yes and not has_no: return 1
    if has_no  and not has_yes: return 0
    return -1


def build_prompt_raw(text, question):
    """Paper template — byte-identical to original Step 5."""
    return (
        f"Read the following text and answer the question with only 'yes' or 'no'.\n\n"
        f"Text: {text}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def build_prompt_chat(tokenizer, text, question):
    """Chat-template variant for instruction-tuned models that expect it."""
    msgs = [
        {'role': 'user',
         'content': (f"Read the following text and answer the question with only 'yes' or 'no'.\n\n"
                     f"Text: {text}\n\nQuestion: {question}")}
    ]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


@torch.no_grad()
def _batch_generate(tokenizer, model, prompts, max_new_tokens=10):
    tokenizer.padding_side = 'left'
    inputs = tokenizer(prompts, return_tensors='pt', truncation=True,
                       max_length=2048, padding=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             temperature=None, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    in_len = inputs['input_ids'].shape[1]
    resps = [tokenizer.decode(outputs[i][in_len:], skip_special_tokens=True)
             for i in range(len(prompts))]
    del inputs, outputs
    torch.cuda.empty_cache()
    return resps


@torch.no_grad()
def get_llm_responses_batch(tokenizer, model, prompts, max_new_tokens=10):
    try:
        return _batch_generate(tokenizer, model, prompts, max_new_tokens)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return [_batch_generate(tokenizer, model, [p], max_new_tokens)[0]
                for p in prompts]


def pick_batch_size(gpu_mem_gb, model_size_b):
    """Heuristic. Larger model = smaller batch."""
    if gpu_mem_gb > 70:                       # A100 80GB / H100
        return 32 if model_size_b < 13 else 8
    if gpu_mem_gb > 35:                       # A100 40GB
        return 16 if model_size_b < 13 else 4
    if gpu_mem_gb > 20:                       # A10G / L4 24GB
        return 8 if model_size_b < 13 else 2
    if gpu_mem_gb > 12:                       # T4 16GB
        return 4 if model_size_b < 9 else 1
    return 1


def _size_b_from_key(key):
    """Map model registry key -> approx parameter count in B for batch sizing.
    Falls back to 7 if unknown — safe under-estimate."""
    table = {
        'phi-4-mini': 3.8, 'deepseek-v2-lite': 16,  # MoE, total params count for VRAM
        'mistral-7b': 7, 'llama2-7b': 7, 'qwen2.5-7b': 7, 'gemma-7b': 7,
        'llama3.1-8b': 8, 'gemma2-9b': 9,
        'mistral-nemo': 12, 'llama2-13b': 13,
        'qwen2.5-32b': 32, 'llama3.3-70b': 70,
    }
    return table.get(key, 7)


def _safe_gpu_mem_gb():
    """Return GPU memory in GB, or 0 if no CUDA. Avoids the crash that
    line `torch.cuda.get_device_properties(0)` causes on CPU-only runtimes."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / 1e9


def _bf16_supported():
    """T4 / V100 / Turing report False; A100 / A10G / Ampere+ report True."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8   # Ampere = sm_80+


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEFAULT_MODEL_KEY,
                    choices=list(MODEL_REGISTRY))
    ap.add_argument('--chat', action='store_true',
                    help='Use chat template instead of raw paper template.')
    ap.add_argument('--quant', choices=['int8', 'int4', 'bf16', 'fp16'], default='int8')
    # In Colab notebook, sys.argv carries the kernel-launcher's flags
    # (e.g. `-f /tmp/kernel-uuid.json`). Detect that and fall back to defaults
    # so callers don't have to override sys.argv manually.
    if any(a.endswith('.json') or a.startswith('-f') for a in sys.argv[1:]):
        args = ap.parse_args([])
        print("[step5b] running under Jupyter — using defaults (override via "
              "sys.argv = ['step5b', '--model', '...'] before main()).")
    else:
        args = ap.parse_args()

    key = args.model
    model_id, gated = MODEL_REGISTRY[key]
    print(f"[step5b] model={key}  hf_id={model_id}  template={'chat' if args.chat else 'raw'}  quant={args.quant}")
    if gated:
        print("[step5b] gated — `huggingface-cli login` first if you haven't.")

    df_train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    df_test  = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    df_all   = pd.concat([df_train, df_test], ignore_index=True)
    df_all['split'] = ['train']*len(df_train) + ['test']*len(df_test)

    with open(os.path.join(PROMPT_DIR, 'prompt_questions.json')) as fh:
        ALL_PROMPTS = json.load(fh)
    print(f"[step5b] texts={len(df_all)} prompts={len(ALL_PROMPTS)} calls={len(df_all)*len(ALL_PROMPTS):,}")

    # ---- model load ----
    
    compute_dtype = torch.bfloat16 if _bf16_supported() else torch.float16
    if args.quant == 'int8':
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs = {'quantization_config': qcfg}
    elif args.quant == 'int4':
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute_dtype)
        load_kwargs = {'quantization_config': qcfg}
    elif args.quant == 'fp16':
        load_kwargs = {'torch_dtype': torch.float16}
    else:  # bf16
        load_kwargs = {'torch_dtype': compute_dtype}

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto', **load_kwargs)
    model.eval()

    gpu_mem_gb = _safe_gpu_mem_gb()
    if gpu_mem_gb == 0:
        sys.exit("[step5b] No CUDA GPU detected — inference needs one.")
    size_b = _size_b_from_key(key)
    BATCH_SIZE = pick_batch_size(gpu_mem_gb, size_b)
    print(f"[step5b] gpu={gpu_mem_gb:.1f}GB size~{size_b}B → BATCH_SIZE={BATCH_SIZE}")

    build_prompt = (lambda t, q: build_prompt_chat(tokenizer, t, q)) if args.chat else build_prompt_raw
    
    MAX_NEW = 20 if args.chat else 10

    # ---- resume ----
    progress_path = os.path.join(PROMPT_DIR, f'{key}_progress.csv')
    sidecar_path  = os.path.join(PROMPT_DIR, f'{key}_progress.meta.json')
    current_meta = {'model': key,
                    'template': 'chat' if args.chat else 'raw',
                    'quant':    args.quant}
    if os.path.exists(progress_path):
        existing = pd.read_csv(progress_path)
        start_idx = len(existing)
        results = existing.to_dict('records')
        if os.path.exists(sidecar_path):
            with open(sidecar_path) as fh:
                prior_meta = json.load(fh)
            for k_, v_ in current_meta.items():
                if prior_meta.get(k_) != v_:
                    sys.exit(f"[step5b] Resume aborted: {k_} changed "
                             f"({prior_meta.get(k_)} -> {v_}). Delete "
                             f"{progress_path} to start fresh, or rerun with "
                             f"the original {k_} value.")
        print(f"[step5b] RESUMING from {start_idx}/{len(df_all)}")
    else:
        start_idx, results = 0, []
        print(f"[step5b] starting fresh: 0/{len(df_all)}")
    with open(sidecar_path, 'w') as fh:
        json.dump(current_meta, fh)

    # ---- loop ----
    SAVE_EVERY = 50
    t0 = time.time()
    done = 0
    for text_idx in tqdm(range(start_idx, len(df_all)), desc=key):
        row = df_all.iloc[text_idx]
        text = str(row['full_text'])
        row_results = {'text_idx': text_idx, 'split': row['split']}
        prompts = [build_prompt(text, q) for q in ALL_PROMPTS]
        for bs in range(0, len(prompts), BATCH_SIZE):
            chunk = prompts[bs:bs+BATCH_SIZE]
            resps = get_llm_responses_batch(tokenizer, model, chunk, max_new_tokens=MAX_NEW)
            for j, r in enumerate(resps):
                row_results[f'prompt_{bs+j}'] = parse_yes_no(r)
        results.append(row_results)
        done += 1
        if (text_idx + 1) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv(progress_path, index=False)
            elapsed = max(time.time() - t0, 1e-6)
            rate = max(done / elapsed, 1e-6)
            eta_min = (len(df_all) - start_idx - done) / rate / 60
            print(f"  ckpt {text_idx+1}/{len(df_all)}  {rate:.2f}/s  eta~{eta_min:.0f}min")

    df_out = pd.DataFrame(results)
    df_out.to_csv(progress_path, index=False)

    # ---- split + save ----
    prompt_cols = [c for c in df_out.columns if c.startswith('prompt_')]
    train_mask = df_out['split'] == 'train'
    test_mask  = df_out['split'] == 'test'
    train_p = df_out.loc[train_mask, prompt_cols].reset_index(drop=True)
    test_p  = df_out.loc[test_mask,  prompt_cols].reset_index(drop=True)
    train_p['education_level'] = df_train['education_level'].values
    test_p['education_level']  = df_test['education_level'].values
    train_p.to_csv(os.path.join(PROMPT_DIR, f'train_prompt_metrics_{key}.csv'), index=False)
    test_p.to_csv(os.path.join(PROMPT_DIR,  f'test_prompt_metrics_{key}.csv'),  index=False)
    print(f"[step5b] {key} DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
