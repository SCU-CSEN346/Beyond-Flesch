"""
Live demo for the COMBO + Gombert multi-task text-difficulty classifier.

Usage:
    cd generalization_final
    python demo/app.py

Then open http://127.0.0.1:7860 in a browser.

Pipeline per text:
    1. compute 34 static features (textstat-based, ~50 ms CPU)
    2. compute 63 prompt features via Qwen-2.5-7B (~5-10 sec GPU)
    3. concat static + prompts (97-dim)
    4. scale with the saved StandardScaler
    5. forward through the trained MLP (3-class softmax + grade head)
    6. show predicted level, per-class probabilities, predicted continuous grade

The bundle loaded by default is the Gombert multi-task per_llm Qwen variant
(macro-F1 0.940 on 150 truly-unseen synthetic OOD texts).
"""

import glob
import json
import os
import pickle
import re
import sys
from collections import Counter
from math import log2

import gradio as gr
import numpy as np
import textstat
import torch
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_DIR, 'scripts'))
from step8l_combo_classifier import ComboMLP  # noqa: E402

os.environ.setdefault('HF_HUB_CACHE',
                      os.path.expanduser('~/.cache/hf_user/hub'))

PROMPT_QUESTIONS_JSON = os.path.join(_PROJECT_DIR, 'inputs', 'prompt_questions.json')
MODELS_DIR  = os.path.join(_PROJECT_DIR, 'outputs', 'models')
FEATURE_LLM = 'Qwen/Qwen2.5-7B-Instruct'
LABEL_NAMES = ['elementary', 'middle', 'high']
MAX_MODEL_LEN = 2048


# ---------- static feature extraction (inlined from step3d) ----------

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


def static_feature_vector(text, feature_names):
    feats = static_features(text)
    arr = np.array([feats.get(n, 0.0) for n in feature_names], dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ---------- prompt feature extraction via Qwen-2.5-7B ----------

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


# ---------- global state ----------

class DemoState:
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.sampling = None
        self.bundle = None
        self.model = None
        self.scaler = None
        self.feature_dim = None
        self.use_grade_reg = None
        self.prompt_questions = None
        self.static_feature_names = None
        self.max_text_tokens = None

    def is_ready(self):
        return self.llm is not None and self.model is not None


STATE = DemoState()


def _pick_best_bundle():
    pattern = os.path.join(MODELS_DIR, 'per_llm__llm_judge__qwen2.5-7b__seed*_gradereg.pkl')
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No bundle found matching {pattern}. "
            f"Train one first via: python scripts/step8l_combo_classifier.py "
            f"--config per_llm --label-source llm_judge --llms qwen2.5-7b --use-grade-reg")
    best, best_f1 = None, -1.0
    for p in candidates:
        with open(p, 'rb') as fh:
            b = pickle.load(fh)
        f1 = b['report']['best_val_f1_macro']
        if f1 > best_f1:
            best, best_f1 = p, f1
    return best, best_f1


def _load_static_feature_names():
    """Pull the 34 stat_* column names from the saved feature_names.json."""
    p = os.path.join(_PROJECT_DIR, 'outputs', 'static_features', 'feature_names.json')
    if not os.path.exists(p):
        # Fall back to inferring from static_features() on a sample text.
        sample = static_features("This is a sample text. It has two sentences.")
        return sorted(k for k in sample if k.startswith('stat_'))
    with open(p) as fh:
        return sorted(json.load(fh))


def initialize():
    """Heavy startup: load Qwen via vLLM, load bundle, set up state."""
    print("[demo] loading Qwen-2.5-7B via vLLM (this takes ~30-60 sec)...")
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    STATE.tokenizer = AutoTokenizer.from_pretrained(FEATURE_LLM)
    STATE.llm = LLM(model=FEATURE_LLM, dtype='bfloat16',
                    enable_prefix_caching=True,
                    gpu_memory_utilization=0.55,
                    max_model_len=MAX_MODEL_LEN,
                    trust_remote_code=True)
    STATE.sampling = SamplingParams(temperature=0.0, max_tokens=3)
    # max_text_tokens budget so prompt + answer fits
    overhead = len(STATE.tokenizer(
        build_prompt_raw('', 'Is this text suitable for elementary school students?'),
        add_special_tokens=False)['input_ids'])
    STATE.max_text_tokens = MAX_MODEL_LEN - overhead - 3 - 8

    print("[demo] loading 63 prompt questions...")
    with open(PROMPT_QUESTIONS_JSON) as fh:
        STATE.prompt_questions = json.load(fh)
    print(f"[demo] {len(STATE.prompt_questions)} prompts loaded")

    print("[demo] loading static feature names...")
    STATE.static_feature_names = _load_static_feature_names()
    print(f"[demo] {len(STATE.static_feature_names)} static features")

    print("[demo] loading trained model bundle...")
    pkl_path, best_f1 = _pick_best_bundle()
    with open(pkl_path, 'rb') as fh:
        STATE.bundle = pickle.load(fh)
    STATE.scaler = STATE.bundle['scaler']
    STATE.feature_dim = STATE.bundle['feature_dim']
    STATE.use_grade_reg = bool(STATE.bundle.get('use_grade_reg', False))
    STATE.model = ComboMLP(in_dim=STATE.feature_dim, use_grade_reg=STATE.use_grade_reg)
    STATE.model.load_state_dict(STATE.bundle['model_state'])
    STATE.model.eval()
    print(f"[demo] bundle = {os.path.basename(pkl_path)}  "
          f"feature_dim={STATE.feature_dim}  "
          f"multi_task={STATE.use_grade_reg}  "
          f"val_f1={best_f1:.3f}")


def predict(text):
    """Run the full pipeline on one text."""
    if not STATE.is_ready():
        return ({'error': 'model not yet loaded'}, None, '', '')
    text = (text or '').strip()
    if not text:
        return ({}, None, 'Enter some text to classify.', '')

    # Truncate text to fit in the prompt window.
    ids = STATE.tokenizer(text, add_special_tokens=False)['input_ids']
    if len(ids) > STATE.max_text_tokens:
        ids = ids[:STATE.max_text_tokens]
        text_for_prompts = STATE.tokenizer.decode(ids, skip_special_tokens=True)
    else:
        text_for_prompts = text

    # Static features.
    stat_vec = static_feature_vector(text, STATE.static_feature_names)

    # Prompt features: 63 yes/no queries via Qwen.
    prompts = [build_prompt_raw(text_for_prompts, q) for q in STATE.prompt_questions]
    outputs = STATE.llm.generate(prompts, STATE.sampling, use_tqdm=False)
    prompt_vec = np.array([parse_yes_no(o.outputs[0].text) for o in outputs],
                          dtype=np.float32)
    # Impute -1 (hedge / unparseable) with the SAME per-column mean used at
    # training time. step8l replaces -1 with the column mean of valid rows,
    # then StandardScaler centers everything to that mean. So scaler.mean_ for
    # each prompt column IS exactly the training-time imputation value.
    # (Earlier we used 0.5 as a "neutral" guess, which mis-aligned the model
    #  on any prompt where Qwen hedged - the bug that made the demo classify
    #  worse than the offline step9g eval on the same texts.)
    n_static = STATE.feature_dim - len(prompt_vec)
    for j in range(len(prompt_vec)):
        if prompt_vec[j] < 0:
            prompt_vec[j] = float(STATE.scaler.mean_[n_static + j])

    # Concat + scale.
    X = np.concatenate([stat_vec, prompt_vec]).astype(np.float32).reshape(1, -1)
    if X.shape[1] != STATE.feature_dim:
        return ({}, None,
                f"Feature dim mismatch: built {X.shape[1]}, "
                f"bundle expects {STATE.feature_dim}", '')
    X_s = STATE.scaler.transform(X).astype(np.float32)

    # Forward.
    with torch.no_grad():
        cls_logits, grade_pred = STATE.model(torch.tensor(X_s, dtype=torch.float32))
        probs = torch.softmax(cls_logits, dim=-1).numpy()[0]
        pred_idx = int(probs.argmax())
    pred_label = LABEL_NAMES[pred_idx]

    probs_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(3)}
    if grade_pred is not None:
        grade = float(grade_pred.numpy()[0])
        grade_str = f"~ US grade {grade:.1f}"
    else:
        grade = None
        grade_str = ""

    # Show a small subset of prompts that fired most strongly per class.
    detail = (f"Prediction: **{pred_label}**  ({100 * probs[pred_idx]:.1f}%)\n\n"
              f"Per-class probabilities:\n"
              f"  - elementary: {100 * probs[0]:5.1f}%\n"
              f"  - middle:     {100 * probs[1]:5.1f}%\n"
              f"  - high:       {100 * probs[2]:5.1f}%\n\n"
              f"Continuous grade prediction: {grade_str}\n\n"
              f"Static signals: FK-grade = {stat_vec[STATE.static_feature_names.index('stat_flesch_kincaid_grade')]:.1f}  "
              f"avg-sent-len = {stat_vec[STATE.static_feature_names.index('stat_avg_sent_len')]:.1f}  "
              f"long-word-ratio = {stat_vec[STATE.static_feature_names.index('stat_long_word_ratio')]:.2f}")

    return probs_dict, grade, detail, f"{sum(1 for v in prompt_vec if v == 1)} of 63 prompts answered 'yes'"


EXAMPLES = [
    ["The cat sat on the mat. It was a sunny day. The cat saw a bird. The bird flew away. The cat was sad."],
    ["Photosynthesis is the process by which plants convert sunlight into chemical energy. Chlorophyll molecules absorb light, primarily in the red and blue wavelengths. The captured energy drives the synthesis of glucose from carbon dioxide and water, releasing oxygen as a byproduct."],
    ["The principle of stare decisis dictates that courts must adhere to precedential rulings established by higher tribunals within the same jurisdiction. This doctrine, fundamental to common law systems, promotes legal stability while permitting the gradual evolution of jurisprudence through carefully reasoned departures from established interpretations."],
    ["Yesterday I went to the store with my mom. We bought apples and bread. I picked the red apples because they are my favorite. Then we walked home together."],
]


def build_ui():
    with gr.Blocks(title="Text Difficulty Classifier") as demo:
        gr.Markdown("# Text Difficulty Classifier — Live Demo")
        gr.Markdown(
            "COMBO classifier (34 static features + 63 Qwen-2.5-7B prompt features) "
            "with **Gombert multi-task head**. Trained on LLM-judge-unified labels. "
            "Synthetic-OOD macro-F1 = **0.940**, QWK = **0.957**. "
            "First prediction takes ~30-60 sec while vLLM warms up; subsequent predictions ~5-10 sec each."
        )
        with gr.Row():
            with gr.Column(scale=1):
                txt = gr.Textbox(
                    lines=12, max_lines=20,
                    label="Paste text to classify",
                    placeholder="The water cycle describes how water moves continuously through the Earth's atmosphere..."
                )
                btn = gr.Button("Classify", variant='primary')
                gr.Examples(examples=EXAMPLES, inputs=[txt], label="Example texts")
            with gr.Column(scale=1):
                lab = gr.Label(label="Class probabilities", num_top_classes=3)
                grade = gr.Number(label="Predicted continuous grade (US K-12 scale)",
                                  precision=2, interactive=False)
                detail = gr.Markdown(label="Details")
                meta = gr.Markdown(label="Prompt-feature summary")
        btn.click(predict, inputs=[txt], outputs=[lab, grade, detail, meta])
    return demo


if __name__ == '__main__':
    # CodeCarbon spans the entire demo session — startup (model load) +
    # however long the UI runs. Writes to outputs/emissions/emissions.csv
    # on graceful shutdown (Ctrl-C). For per-prediction tracking, see the
    # _predict_with_tracking wrapper noted below — disabled by default
    # since starting/stopping a tracker on each call has overhead.
    emissions_dir = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')
    os.makedirs(emissions_dir, exist_ok=True)
    tracker = EmissionsTracker(project_name='demo_app_session',
                               output_dir=emissions_dir, log_level='warning')
    tracker.start()
    try:
        initialize()
        ui = build_ui()
        print("[demo] launching UI on http://127.0.0.1:7860")
        ui.launch(server_name='0.0.0.0', server_port=7860, share=False)
    finally:
        tracker.stop()
