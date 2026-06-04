"""
Step 3i — Generate a balanced 150-text synthetic OOD set.

Llama-3.1-8B-Instruct is asked to write ~100 words on each of 10 topics x 5
variations x 3 levels = 150 texts, then re-judge them with the same rubric
used in step3f. Texts where the judge disagrees with the generation level
are flagged but still saved (so the paper can report the consistency rate).

Outputs:
    outputs/splits/ood_synthetic.csv          (the synthetic test split)
    outputs/llm_judge/ood_synthetic_judge.csv (judge re-label per row)

Each row in ood_synthetic.csv carries:
    full_text, education_level (gen label), source_dataset='synthetic',
    domain=topic, orig_split='ood_synthetic', orig_idx=row index,
    gen_topic, gen_variation
"""

import argparse
import os
import re

import pandas as pd
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
SPLITS_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'splits')
JUDGE_DIR    = os.path.join(_PROJECT_DIR, 'outputs', 'llm_judge')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

GEN_MODEL  = 'meta-llama/Llama-3.1-8B-Instruct'
MAX_MODEL_LEN = 2048
TOPICS = {
    'elementary': [
        'my pet dog', 'a rainy day', 'my family',
        'a school day', 'animals at the zoo', 'my favorite food',
        'playing with friends', 'a soccer game',
        'the four seasons of the year', 'my hobby',
    ],
    'middle': [
        'photosynthesis in plants', 'the water cycle on Earth',
        'an overview of World War II', 'ecosystems and food webs',
        "Newton's three laws of motion", 'the geography of Asia',
        'the Renaissance in Europe', 'climate change basics',
        'an introduction to algebra', 'ancient Egyptian civilization',
    ],
    'high': [
        'quantum mechanics and wave-particle duality',
        'macroeconomic policy and fiscal theory',
        'constitutional law and judicial review',
        'integral calculus and its applications',
        'genetics and CRISPR gene editing',
        'existential philosophy and Sartre',
        'linear algebra and eigenvalues',
        'geopolitics in the post-Cold War era',
        'Shakespearean tragedy and dramatic structure',
        'organic chemistry and reaction mechanisms',
    ],
}
N_VARIATIONS  = 5
GRADE_RANGE   = {'elementary': '1-5', 'middle': '6-8', 'high': '9-12'}
STYLE         = {
    'elementary': 'simple words, short sentences, and concrete examples a child can picture',
    'middle':     'moderate vocabulary, multi-clause sentences, and age-appropriate explanations',
    'high':       'advanced vocabulary, complex sentence structures, and abstract reasoning',
}

GEN_RUBRIC = (
    "You are writing an educational passage for {level} students (US grades {range}).\n"
    "Write approximately 100 words on the topic: \"{topic}\".\n"
    "Use {style}.\n"
    "Output ONLY the passage. No title, no introductory phrase, no quotation marks, "
    "no labels. Just the passage text.\n"
    "Variation #{variation}: take a slightly different angle on the topic than other variations."
)

JUDGE_RUBRIC = (
    "Classify the following text by its target reader's US education level.\n"
    "Choose exactly one of:\n"
    "- elementary  (US grades 1-5, simple vocabulary, short sentences)\n"
    "- middle      (US grades 6-8)\n"
    "- high        (US grades 9-12, advanced vocabulary, complex ideas)\n\n"
    "Text:\n{text}\n\n"
    "Reply with one word only: elementary, middle, or high."
)


def _parse_judge(raw):
    s = str(raw).strip().lower()
    s = re.sub(r'^[^a-z]+', '', s)
    if s.startswith('elem'): return 'elementary'
    if s.startswith('mid'):  return 'middle'
    if s.startswith('high'): return 'high'
    return 'elementary'   # invalid -> default (Rooein-2024 style)


def _clean_passage(raw):
    """Strip common LLM preamble + trailing fluff. Keep the passage itself."""
    t = str(raw).strip()
    # Remove leading "Here is..." / "Passage:" patterns.
    t = re.sub(r'^(here\s+is.*?:?\s*\n)', '', t, flags=re.IGNORECASE)
    t = re.sub(r'^(passage|text)\s*:\s*\n?', '', t, flags=re.IGNORECASE)
    t = t.strip('"\'`* \n\t')
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-tokens-gen',  type=int, default=200)
    ap.add_argument('--max-tokens-judge', type=int, default=8)
    ap.add_argument('--gpu-mem-frac', type=float, default=0.55)  # fit in ~26GB with leftover workers
    args = ap.parse_args()
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(JUDGE_DIR,  exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(project_name='step3i_generate_synthetic',
                               output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        print(f"[step3i] loading {GEN_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
        llm = LLM(model=GEN_MODEL, dtype='bfloat16',
                  enable_prefix_caching=True,
                  gpu_memory_utilization=args.gpu_mem_frac,
                  max_model_len=MAX_MODEL_LEN,
                  trust_remote_code=True,
                  seed=args.seed)

        gen_sampling = SamplingParams(temperature=0.8, top_p=0.95,
                                      max_tokens=args.max_tokens_gen,
                                      seed=args.seed)
        judge_sampling = SamplingParams(temperature=0.0,
                                        max_tokens=args.max_tokens_judge)

        # ---- Build all 150 generation prompts ----
        records = []
        for level, topics in TOPICS.items():
            assert len(topics) == 10, f"{level}: expected 10 topics, got {len(topics)}"
            for topic in topics:
                for v in range(N_VARIATIONS):
                    msgs = [{'role': 'user',
                             'content': GEN_RUBRIC.format(
                                 level=level, range=GRADE_RANGE[level],
                                 topic=topic, style=STYLE[level], variation=v+1)}]
                    prompt = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                    records.append({'level': level, 'topic': topic,
                                    'variation': v, 'gen_prompt': prompt})
        assert len(records) == 150, f"expected 150 records, got {len(records)}"

        # ---- Generate all 150 in one batch ----
        print(f"[step3i] generating {len(records)} synthetic texts")
        gen_outputs = llm.generate([r['gen_prompt'] for r in records],
                                   gen_sampling, use_tqdm=True)
        for r, o in zip(records, gen_outputs):
            r['text_raw'] = o.outputs[0].text
            r['full_text'] = _clean_passage(o.outputs[0].text)

        # ---- Re-judge all 150 ----
        print(f"[step3i] re-judging {len(records)} synthetic texts")
        judge_prompts = []
        for r in records:
            msgs = [{'role': 'user',
                     'content': JUDGE_RUBRIC.format(text=r['full_text'])}]
            judge_prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True))
        judge_outputs = llm.generate(judge_prompts, judge_sampling, use_tqdm=True)
        for r, o in zip(records, judge_outputs):
            raw = o.outputs[0].text
            r['judge_raw'] = raw
            r['llm_judge_label'] = _parse_judge(raw)

        # ---- Build outputs ----
        df = pd.DataFrame(records)
        df['orig_split']     = 'ood_synthetic'
        df['orig_idx']       = range(len(df))
        df['source_dataset'] = 'synthetic'
        df['domain']         = df['topic']
        df['education_level'] = df['level']   # generation-anchored label
        df['agreement'] = (df['education_level'] == df['llm_judge_label']).astype(int)

        splits_cols = ['full_text', 'education_level', 'source_dataset',
                       'domain', 'orig_split', 'orig_idx',
                       'topic', 'variation']
        df[splits_cols].to_csv(os.path.join(SPLITS_DIR, 'ood_synthetic.csv'), index=False)
        print(f"[step3i] wrote {os.path.join(SPLITS_DIR, 'ood_synthetic.csv')} "
              f"({len(df)} rows, label dist={dict(df['education_level'].value_counts())})")

        judge_cols = ['orig_split', 'orig_idx', 'source_dataset', 'education_level',
                      'llm_judge_label', 'judge_raw']
        judge_df = df[judge_cols].rename(columns={'judge_raw': 'judge_raw_response'})
        judge_df.to_csv(os.path.join(JUDGE_DIR, 'ood_synthetic_judge.csv'), index=False)

        agree = df['agreement'].mean()
        judge_dist = dict(df['llm_judge_label'].value_counts())
        print(f"[step3i] gen-vs-judge agreement = {agree:.3f}  judge_dist={judge_dist}")
    finally:
        tracker.stop()


if __name__ == '__main__':
    main()
