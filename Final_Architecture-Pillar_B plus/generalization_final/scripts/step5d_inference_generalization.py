"""
Step 5d — Prompt inference for curated/generalization split sets

Wraps step5b_inference_generic so we can run prompt features on the separate
Step 2d/2e train/val/test/ood_* CSVs without touching Step 5c outputs.

    1. Reads texts from outputs/curated_generalization_collection/<split>.csv by default.
    2. Optionally appends 10 domain-neutral prompts from
       outputs/prompt_metrics_curated_generalization/extra_generic_prompts.json — these address the
       roadmap's "ScienceQA topic prompts don't generalize" concern.

Outputs (one CSV per split per model):
    outputs/prompt_metrics_curated_generalization/<split>_prompt_metrics_<model>__multi.csv

USAGE
-----
    # paper-faithful 63 prompts only, on curated train.csv
    python step5d_inference_generalization.py --model qwen2.5-7b --split train

    # append the 10 generic prompts (73 total) — costs ~16% more compute
    python step5d_inference_generalization.py --model qwen2.5-7b --split train --extra

    # OOD set
    python step5d_inference_generalization.py --model qwen2.5-7b --split ood_race-middle

    # all current generalization splits, loading the LLM once
    python step5d_inference_generalization.py --model qwen2.5-7b \
        --splits train val test ood_race-middle ood_race-high
"""

import argparse
import hashlib
import json
import os
import sys
import time

import pandas as pd
from codecarbon import EmissionsTracker
from tqdm.auto import tqdm

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)   # generalization_final/
# step5b lives next to us in scripts/, so make sure scripts/ is on sys.path.
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
MULTI_DIR    = os.environ.get('MULTI_DIR',
                              os.path.join(_PROJECT_DIR, 'inputs', 'curated_generalization_collection'))
PROMPT_DIR   = os.environ.get('PROMPT_DIR',
                              os.path.join(_PROJECT_DIR, 'outputs', 'prompt_metrics'))
EXTRA_PATH   = os.path.join(PROMPT_DIR, 'extra_generic_prompts.json')
PAPER_PATH   = os.path.join(_PROJECT_DIR, 'inputs', 'prompt_questions.json')
os.makedirs(PROMPT_DIR, exist_ok=True)


def _load_prompts(use_extra):
    """Return paper's 63 + optional 10 generic prompts. Order matters because
    column names are positional (`prompt_0` ... `prompt_N`)."""
    if not os.path.exists(PAPER_PATH):
        sys.exit(f"[step5d] missing prompt file: {PAPER_PATH}. Run the Part 1 "
                 "prompt-question generation cell first, or copy "
                 "prompt_questions.json into outputs/prompt_metrics/.")
    with open(PAPER_PATH) as fh:
        prompts = list(json.load(fh))
    if use_extra:
        if not os.path.exists(EXTRA_PATH):
            sys.exit(f"[step5d] {EXTRA_PATH} missing. Was --extra requested without "
                     f"the JSON file?")
        with open(EXTRA_PATH) as fh:
            prompts.extend(json.load(fh))
    return prompts


def _prompt_fingerprint(prompts):
    payload = json.dumps(prompts, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()


def _input_fingerprint(df):
    cols = [c for c in ['full_text', 'education_level', 'source_dataset',
                        'domain', 'split_group'] if c in df.columns]
    if not cols:
        return ''
    payload = df[cols].fillna('').astype(str)
    row_hashes = pd.util.hash_pandas_object(payload, index=False).values.tobytes()
    return hashlib.sha1(row_hashes).hexdigest()


def _meta_matches(prior, current):
    keys = ['model', 'split', 'template', 'engine', 'quant', 'extra', 'n_prompts',
            'prompt_hash', 'input_hash', 'input_rows']
    return all(prior.get(k) == current.get(k) for k in keys)


def main():
    global MULTI_DIR, PROMPT_DIR, EXTRA_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='Model key from step5b_inference_generic.MODEL_REGISTRY.')
    ap.add_argument('--multi-dir', default=MULTI_DIR,
                    help='Input split directory from Step 2e curated collection, or Step 2d if overridden.')
    ap.add_argument('--prompt-dir', default=PROMPT_DIR,
                    help='Output prompt feature directory.')
    ap.add_argument('--split', default=None,
                    help='CSV basename in the input split directory (without .csv).')
    ap.add_argument('--splits', nargs='+', default=None,
                    help='Multiple split basenames. Loads the LLM once, then processes each split.')
    ap.add_argument('--extra', action='store_true',
                    help='Append the 10 generic domain-neutral prompts.')
    ap.add_argument('--chat', action='store_true')
    ap.add_argument('--engine', choices=['vllm', 'hf'], default='vllm',
                    help="Inference backend. 'vllm' uses prefix caching across the 63 prompts "
                         "that share the same text prefix — typically 30-60x faster than 'hf' "
                         "on this workload. 'hf' is the original transformers path.")
    ap.add_argument('--quant', choices=['int8', 'int4', 'bf16', 'fp16'], default='bf16',
                    help="Only used when --engine=hf. vLLM always runs bf16.")
    if any(a.endswith('.json') or a.startswith('-f') for a in sys.argv[1:]):
        sys.exit("[step5d] running under Jupyter — set sys.argv before main().")
    args = ap.parse_args()
    MULTI_DIR = args.multi_dir
    PROMPT_DIR = args.prompt_dir
    EXTRA_PATH = os.path.join(PROMPT_DIR, 'extra_generic_prompts.json')
    os.makedirs(PROMPT_DIR, exist_ok=True)
    if args.split and args.splits:
        sys.exit("[step5d] pass either --split or --splits, not both.")
    split_list = args.splits or ([args.split] if args.split else None)
    if not split_list:
        sys.exit("[step5d] pass --split <name> or --splits <name> [<name> ...].")

    # Late imports — heavy.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from step5b_inference_generic import (
        MODEL_REGISTRY, build_prompt_raw, build_prompt_chat,
        get_llm_responses_batch, parse_yes_no, pick_batch_size,
        _safe_gpu_mem_gb, _bf16_supported, _size_b_from_key,
    )

    if args.model not in MODEL_REGISTRY:
        sys.exit(f"[step5d] unknown model {args.model}. "
                 f"Available: {list(MODEL_REGISTRY)}")
    split_paths = {}
    for split in split_list:
        csv_in = os.path.join(MULTI_DIR, f'{split}.csv')
        if not os.path.exists(csv_in):
            sys.exit(f"[step5d] {csv_in} missing. Run Step 2e first, or pass the correct --multi-dir.")
        split_paths[split] = csv_in

    prompts_list = _load_prompts(args.extra)
    prompt_hash = _prompt_fingerprint(prompts_list)
    _, gated = MODEL_REGISTRY[args.model]
    if gated:
        print(f"[step5d] WARNING — {args.model} is gated. "
              f"Run `huggingface-cli login` first if you haven't.")
    print(f"[step5d] multi_dir={MULTI_DIR} prompt_dir={PROMPT_DIR}")
    print(f"[step5d] splits={split_list} model={args.model} "
          f"template={'chat' if args.chat else 'raw'} engine={args.engine} "
          f"quant={args.quant if args.engine == 'hf' else 'bf16(vllm)'} "
          f"prompts={len(prompts_list)} extra={args.extra}")

    gpu_mem_gb = _safe_gpu_mem_gb()
    if gpu_mem_gb == 0:
        sys.exit("[step5d] No CUDA GPU detected.")

    # Model load.
    model_id, _ = MODEL_REGISTRY[args.model]
    # trust_remote_code lets newer / community models (Phi-3.5, DeepSeek-V2)
    # ship custom modeling code. Safe for the registry models we list.
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Raw prompts begin with the (long) text and end with the (short) question,
    # so all 63 prompts for a given text share a long literal prefix. vLLM's
    # automatic prefix caching reuses the prefill KV-cache across them — the
    # key reason this path is ~30-60x faster on the per-row 63-call workload.
    MAX_NEW = 8 if args.chat else 3
    if args.engine == 'vllm':
        from vllm import LLM, SamplingParams
        llm = LLM(model=model_id, dtype='bfloat16', enable_prefix_caching=True,
                  gpu_memory_utilization=0.90, max_model_len=2048,
                  trust_remote_code=True)
        sampling = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
        model = None
        BATCH = None
    else:
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
        llm = None
        sampling = None

    build = (lambda t, q: build_prompt_chat(tokenizer, t, q)) if args.chat else build_prompt_raw

    # Pre-truncate texts so total prompt + max_new fits in vLLM's context window.
    # Llama-2 has only 2048 tokens of total context; on long-text rows the
    # prompt (instruction + text + question) can exceed it and vLLM aborts.
    # Compute the worst-case overhead (instruction template + longest question
    # in tokens) once, then truncate `text` per row to leave room.
    _max_question_overhead = 0
    for _q in prompts_list:
        _full = build('', _q)
        _max_question_overhead = max(
            _max_question_overhead,
            len(tokenizer(_full, add_special_tokens=False)['input_ids']))
    _safety = 16
    MAX_TEXT_TOKENS = 2048 - _max_question_overhead - MAX_NEW - _safety
    if MAX_TEXT_TOKENS < 64:
        sys.exit(f"[step5d] computed max_text_tokens={MAX_TEXT_TOKENS} is too small. "
                 f"prompt overhead={_max_question_overhead}, max_new={MAX_NEW}.")
    print(f"[step5d] prompt overhead={_max_question_overhead} max_new={MAX_NEW} "
          f"-> max_text_tokens={MAX_TEXT_TOKENS}")

    def process_split(split, csv_in):
        suffix = '__multi_extra' if args.extra else '__multi'
        out_path = os.path.join(PROMPT_DIR,
                                f'{split}_prompt_metrics_{args.model}{suffix}.csv')
        df = pd.read_csv(csv_in)
        if 'full_text' not in df.columns or 'education_level' not in df.columns:
            sys.exit(f"[step5d] {csv_in} must contain full_text and education_level columns.")
        out_meta_path = out_path.replace('.csv', '.meta.json')
        current_meta = {'model': args.model, 'split': split,
                        'template': 'chat' if args.chat else 'raw',
                        'engine': args.engine,
                        'quant': 'bf16' if args.engine == 'vllm' else args.quant,
                        'extra': bool(args.extra),
                        'n_prompts': len(prompts_list),
                        'prompt_hash': prompt_hash,
                        'input_hash': _input_fingerprint(df),
                        'input_rows': len(df),
                        'input_path': os.path.abspath(csv_in)}
        if os.path.exists(out_path):
            existing = pd.read_csv(out_path)
            prior_meta = {}
            if os.path.exists(out_meta_path):
                with open(out_meta_path) as fh:
                    prior_meta = json.load(fh)
            if len(existing) == len(df) and _meta_matches(prior_meta, current_meta):
                print(f"[step5d] {out_path} already complete ({len(existing)} rows) — skipping.")
                return
            if len(existing) == len(df):
                sys.exit(f"[step5d] {out_path} exists with matching row count but "
                         "different/missing metadata. This usually means the input "
                         "dataset or prompt set changed. Delete that CSV to regenerate.")
            print(f"[step5d] {out_path} exists but row count mismatch "
                  f"({len(existing)} vs {len(df)}) — re-running.")

        print(f"[step5d] split={split} rows={len(df)} "
              f"calls={len(df) * len(prompts_list):,}")
        progress_path = out_path.replace('.csv', '_progress.csv')
        sidecar_path  = out_path.replace('.csv', '_progress.meta.json')
        if os.path.exists(progress_path):
            existing = pd.read_csv(progress_path)
            start_idx = len(existing)
            if start_idx > len(df):
                sys.exit(f"[step5d] {progress_path} has {start_idx} rows but "
                         f"input has {len(df)} rows. Delete progress files and rerun.")
            results = existing.to_dict('records')
            if not os.path.exists(sidecar_path):
                sys.exit(f"[step5d] {progress_path} exists without {sidecar_path}. "
                         "Delete the progress CSV and rerun to avoid stale resume.")
            with open(sidecar_path) as fh:
                prior = json.load(fh)
            for k, v in current_meta.items():
                if prior.get(k) != v:
                    sys.exit(f"[step5d] Resume aborted: {k} changed "
                             f"({prior.get(k)} -> {v}). Delete {progress_path} "
                             f"to start fresh, or rerun with original {k}.")
            print(f"[step5d] RESUMING {split} from {start_idx}/{len(df)}")
        else:
            start_idx, results = 0, []
        with open(sidecar_path, 'w') as fh:
            json.dump(current_meta, fh)

        SAVE_EVERY = 50
        t0 = time.time()
        done = 0
        for text_idx in tqdm(range(start_idx, len(df)), desc=f'{args.model}/{split}'):
            text = str(df.iloc[text_idx]['full_text'])
            # Truncate any text whose token count alone would push the prompt
            # past vLLM's 2048-token window. Only fires on long rows (RACE
            # passages, some agentlans-readability rows).
            _ids = tokenizer(text, add_special_tokens=False)['input_ids']
            if len(_ids) > MAX_TEXT_TOKENS:
                _ids = _ids[:MAX_TEXT_TOKENS]
                text = tokenizer.decode(_ids, skip_special_tokens=True)
            rec = {'text_idx': text_idx,
                   'education_level': df.iloc[text_idx]['education_level'],
                   'source_dataset':  df.iloc[text_idx].get('source_dataset', ''),
                   'domain':          df.iloc[text_idx].get('domain', '')}
            prompts = [build(text, q) for q in prompts_list]
            if args.engine == 'vllm':
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                for j, out in enumerate(outputs):
                    rec[f'prompt_{j}'] = parse_yes_no(out.outputs[0].text)
            else:
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
        with open(out_meta_path, 'w') as fh:
            json.dump(current_meta, fh, indent=2)
        for p in (progress_path, sidecar_path):
            if os.path.exists(p):
                os.remove(p)
        print(f"[step5d] wrote {out_path}")

    for split, csv_in in split_paths.items():
        process_split(split, csv_in)


if __name__ == '__main__':
    _emissions_dir = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')
    os.makedirs(_emissions_dir, exist_ok=True)
    _tracker = EmissionsTracker(project_name='step5d_inference_generalization',
                                output_dir=_emissions_dir, log_level='warning')
    _tracker.start()
    try:
        main()
    finally:
        _tracker.stop()
