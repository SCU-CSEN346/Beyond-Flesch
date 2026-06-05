from __future__ import annotations

import json
import os
import re
from typing import Any

import gradio as gr
import joblib
import nltk
import numpy as np
import pandas as pd
import requests
import spacy
import textstat
from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, wordnet


# ── Runtime setup ─────────────────────────────────────────────────────────────
for package in [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "stopwords",
]:
    nltk.download(package, quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as spacy_download

    spacy_download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# ── Rooein-style logistic regression artifacts ───────────────────────────────
BASE_DIR = os.path.dirname(__file__)
PROMPT_MODEL_KEY = "qwen2.5-7b"
PROMPT_MODEL_ID = os.environ.get(
    "TOGETHER_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct-Turbo"
)
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"
TOGETHER_TIMEOUT_SECONDS = int(os.environ.get("TOGETHER_TIMEOUT_SECONDS", "120"))
TOGETHER_PROMPT_MODE = os.environ.get("TOGETHER_PROMPT_MODE", "chunked").lower()
TOGETHER_CHUNK_COUNT = int(os.environ.get("TOGETHER_CHUNK_COUNT", "3"))
ARTIFACTS_DIR = os.path.join(
    BASE_DIR, "results", "by_prompt_model", PROMPT_MODEL_KEY, "artifacts"
)
PROMPT_QUESTIONS_PATH = os.path.join(BASE_DIR, "outputs", "prompt_metrics", "prompt_questions.json")
SCHEMA_PATH = os.path.join(ARTIFACTS_DIR, f"feature_schema_{PROMPT_MODEL_KEY}.json")

with open(SCHEMA_PATH, "r", encoding="utf-8") as schema_file:
    FEATURE_SCHEMA: dict[str, Any] = json.load(schema_file)

STATIC_COLUMNS: list[str] = FEATURE_SCHEMA["static_columns"]
PROMPT_COLUMNS: list[str] = FEATURE_SCHEMA["prompt_columns"]
COMBO_COLUMNS: list[str] = FEATURE_SCHEMA["combo_columns"]
LABELS = ["elementary", "middle", "high"]

with open(PROMPT_QUESTIONS_PATH, "r", encoding="utf-8") as prompt_file:
    PROMPT_QUESTIONS: list[str] = json.load(prompt_file)

selector_static = joblib.load(os.path.join(ARTIFACTS_DIR, f"selector_static_{PROMPT_MODEL_KEY}.joblib"))
scaler_static = joblib.load(os.path.join(ARTIFACTS_DIR, f"scaler_static_{PROMPT_MODEL_KEY}.joblib"))
model_static = joblib.load(os.path.join(ARTIFACTS_DIR, f"model_static_{PROMPT_MODEL_KEY}.joblib"))

selector_combo = joblib.load(os.path.join(ARTIFACTS_DIR, f"selector_combo_{PROMPT_MODEL_KEY}.joblib"))
scaler_combo = joblib.load(os.path.join(ARTIFACTS_DIR, f"scaler_combo_{PROMPT_MODEL_KEY}.joblib"))
model_combo = joblib.load(os.path.join(ARTIFACTS_DIR, f"model_combo_{PROMPT_MODEL_KEY}.joblib"))

STOP_WORDS = set(stopwords.words("english"))
AUXILIARY_VERBS = {
    "am",
    "are",
    "be",
    "been",
    "being",
    "can",
    "could",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "having",
    "is",
    "may",
    "might",
    "must",
    "shall",
    "should",
    "was",
    "were",
    "will",
    "would",
}
NEGATION_WORDS = {"no", "not", "never", "neither", "nobody", "nothing", "nowhere", "nor"}
VOWEL_RE = re.compile(r"[aeiou]", re.IGNORECASE)


# ── Feature extraction ───────────────────────────────────────────────────────
def count_syllables(word: str) -> int:
    count = len(VOWEL_RE.findall(word.lower().rstrip("e")))
    return max(count, 1)


def is_complex_word(word: str) -> bool:
    return count_syllables(word) >= 3


def word_sense_count(word: str, pos: str | None = None) -> int:
    synsets = wordnet.synsets(word, pos=pos) if pos else wordnet.synsets(word)
    return len(synsets)


def wordnet_depth(synsets: list[Any]) -> float:
    depths = []
    for synset in synsets:
        try:
            depths.append(synset.min_depth())
        except Exception:
            continue
    return float(np.mean(depths)) if depths else 0.0


def mean_or_zero(values: list[float] | list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def compute_static_features(text: str) -> dict[str, float]:
    text = str(text or "").strip()
    if not text:
        return {column: 0.0 for column in STATIC_COLUMNS}

    words_alpha = [word for word in word_tokenize(text) if word.isalpha()]
    words_lower = [word.lower() for word in words_alpha]
    sentences = sent_tokenize(text)
    tagged_words = pos_tag(words_alpha)
    doc = nlp(text)

    nouns = [word for word, tag in tagged_words if tag.startswith("NN")]
    verbs = [word for word, tag in tagged_words if tag.startswith("VB")]
    adjectives = [word for word, tag in tagged_words if tag.startswith("JJ")]
    adverbs = [word for word, tag in tagged_words if tag.startswith("RB")]

    word_count = len(words_alpha)
    text_length = len(text)
    word_lengths = [len(word) for word in words_alpha]
    syllables = [count_syllables(word) for word in words_alpha]

    content_words = set(word.lower() for word in nouns + verbs + adjectives + adverbs)
    content_no_stop = [word for word in words_lower if word not in STOP_WORDS]
    unique_lower = set(words_lower)

    noun_chunks = list(doc.noun_chunks)
    sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]

    passive_heads: set[int] = set()
    active_heads: set[int] = set()
    agentless_passive = 0
    for token in doc:
        if token.dep_ == "nsubjpass":
            passive_heads.add(token.head.i)
            if not any(child.dep_ == "agent" for child in token.head.children):
                agentless_passive += 1
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            active_heads.add(token.head.i)

    active_count = len(active_heads)
    passive_count = len(passive_heads)
    voice_total = active_count + passive_count

    words_before_main_verb = []
    for sentence in doc.sents:
        for index, token in enumerate(sentence):
            if token.pos_ == "VERB" and token.dep_ in {"ROOT", "ccomp", "advcl"}:
                words_before_main_verb.append(index)
                break

    non_aux_verbs = [word for word in verbs if word.lower() not in AUXILIARY_VERBS]
    noun_synsets = [
        synset
        for word in set(nouns)
        for synset in wordnet.synsets(word.lower(), pos=wordnet.NOUN)
    ]
    verb_synsets = [
        synset
        for word in set(verbs)
        for synset in wordnet.synsets(word.lower(), pos=wordnet.VERB)
    ]

    features = {
        "n_words_q": 0.0,
        "n_words_a_solution": 0.0,
        "n_words_a_lecture": 0.0,
        "Text_Length": float(text_length),
        "Word_Count": float(word_count),
        "Nouns": float(len(nouns)),
        "Verbs": float(len(verbs)),
        "Adjectives": float(len(adjectives)),
        "Adverbs": float(len(adverbs)),
        "Num_Numbers": float(sum(character.isdigit() for character in text)),
        "Num_Commas": float(text.count(",")),
        "Num_Complex_Words": float(sum(is_complex_word(word) for word in words_alpha)),
        "Num_Unique_Words": float(len(unique_lower)),
        "Num_Content_Words": float(sum(word in content_words for word in words_lower)),
        "Num_Content_Words_No_Stopwords": float(len(content_no_stop)),
        "Word_Length_Syllables": mean_or_zero(syllables),
        "Avg_Sentence_Length": mean_or_zero(sentence_lengths),
        "Num_Prepositional_Phrases": float(sum(token.pos_ == "ADP" for token in doc)),
        "Num_Negated_Words_Stem": float(sum(token.dep_ == "neg" for token in doc)),
        "Num_Negated_Words_Lead_In": float(sum(word in NEGATION_WORDS for word in words_lower)),
        "Num_Main_Noun_Phrases": float(len(noun_chunks)),
        "Avg_Main_NP_Length": mean_or_zero([len(chunk) for chunk in noun_chunks]),
        "Num_Verb_Phrases": float(sum(token.pos_ == "VERB" for token in doc)),
        "Prop_Active_Voice_Verbs": active_count / voice_total if voice_total else 1.0,
        "Prop_Passive_Voice_Verbs": passive_count / voice_total if voice_total else 0.0,
        "Ratio_Active_to_Passive_Verbs": (
            active_count / passive_count if passive_count else float(active_count)
        ),
        "Num_Words_Before_Main_Verb": mean_or_zero(words_before_main_verb),
        "Num_Agentless_Passive_Constructions": float(agentless_passive),
        "Word_Length_Std_Dev": float(np.std(word_lengths)) if word_lengths else 0.0,
        "Num_Polysemic_Words": float(sum(word_sense_count(word) > 1 for word in unique_lower)),
        "Num_Word_Senses": float(sum(word_sense_count(word) for word in unique_lower)),
        "Num_Word_Senses_For_Content_Words": float(
            sum(word_sense_count(word) for word in set(content_no_stop))
        ),
        "Num_Word_Senses_For_Nouns": float(
            sum(word_sense_count(word.lower(), wordnet.NOUN) for word in set(nouns))
        ),
        "Num_Word_Senses_For_Verbs": float(
            sum(word_sense_count(word.lower(), wordnet.VERB) for word in set(verbs))
        ),
        "Num_Word_Senses_For_Non_Auxiliary_Verbs": float(
            sum(word_sense_count(word.lower(), wordnet.VERB) for word in set(non_aux_verbs))
        ),
        "Num_Word_Senses_For_Adjectives": float(
            sum(word_sense_count(word.lower(), wordnet.ADJ) for word in set(adjectives))
        ),
        "Num_Word_Senses_For_Adverbs": float(
            sum(word_sense_count(word.lower(), wordnet.ADV) for word in set(adverbs))
        ),
        "Distance_To_Root_Nouns": wordnet_depth(noun_synsets),
        "Distance_To_Root_Verbs": wordnet_depth(verb_synsets),
        "flesch_kincaid_grade": float(textstat.flesch_kincaid_grade(text)),
        "flesch_kincaid_ease": float(textstat.flesch_reading_ease(text)),
        "coleman_liau_index": float(textstat.coleman_liau_index(text)),
        "automated_readability_index": float(textstat.automated_readability_index(text)),
        "smog_index": float(textstat.smog_index(text)),
        "gunning_fog": float(textstat.gunning_fog(text)),
        "traenkle_bailer_index": (
            0.449 * mean_or_zero(sentence_lengths) + 0.103 * mean_or_zero(word_lengths) - 0.651
        ),
    }
    return {column: float(features.get(column, 0.0)) for column in STATIC_COLUMNS}


# ── Prediction ───────────────────────────────────────────────────────────────
def empty_prediction() -> dict[str, float]:
    return {label: 0.0 for label in LABELS}


def format_prediction(classes: list[str], probabilities: np.ndarray) -> dict[str, float]:
    prediction = empty_prediction()
    for class_name, probability in zip(classes, probabilities):
        prediction[str(class_name)] = float(probability)
    return prediction


def build_prompt_raw(text: str, question: str) -> str:
    return (
        "Read the following text and answer the question with only 'yes' or 'no'.\n\n"
        f"Text: {text}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def parse_yes_no(response: str) -> int:
    response_head = re.sub(r'^[\*_`>\-\s"\']+', "", response.strip().lower())[:30]
    if response_head.startswith("yes"):
        return 1
    if response_head.startswith("no"):
        return 0

    has_yes = bool(re.search(r"\byes\b", response_head))
    has_no = bool(re.search(r"\bno\b", response_head))
    if has_yes and not has_no:
        return 1
    if has_no and not has_yes:
        return 0
    return -1


def together_api_key() -> str:
    api_key = os.environ.get("TOGETHER_API_KEY", "").strip()
    if not api_key:
        raise gr.Error(
            "Missing TOGETHER_API_KEY. Add it as a Hugging Face Space secret before using "
            "the combo model."
        )
    return api_key


def call_together(prompt: str, max_tokens: int, json_mode: bool = False) -> str:
    request_body: dict[str, Any] = {
        "model": PROMPT_MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if json_mode:
        request_body["response_format"] = {"type": "json_object"}

    response = requests.post(
        TOGETHER_API_URL,
        headers={
            "Authorization": f"Bearer {together_api_key()}",
            "Content-Type": "application/json",
        },
        json=request_body,
        timeout=TOGETHER_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise gr.Error(f"Together API request failed: {response.text[:500]}") from exc

    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def coerce_prompt_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and int(value) in {-1, 0, 1}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "yes", "true", "y"}:
            return 1
        if normalized in {"0", "no", "false", "n"}:
            return 0
        if normalized in {"-1", "unknown", "unclear", "maybe"}:
            return -1
        return parse_yes_no(normalized)
    return -1


def normalize_prompt_values(values: list[Any], expected_count: int) -> list[int]:
    if len(values) < expected_count:
        values = values + [-1] * (expected_count - len(values))
    return [coerce_prompt_value(value) for value in values[:expected_count]]


def extract_json_candidate(response: str) -> Any | None:
    cleaned = response.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    for candidate in [cleaned]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    object_match = re.search(r"\{[\s\S]*\}", cleaned)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass

    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def values_from_json_payload(payload: Any, expected_count: int) -> list[Any] | None:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        return None

    for key in ("scores", "answers", "prompt_scores", "prompt_features", "values"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    keyed_values = []
    for index in range(expected_count):
        for key in (f"prompt_{index}", str(index), index):
            if key in payload:
                keyed_values.append(payload[key])
                break
        else:
            return None
    return keyed_values


def parse_indexed_prompt_values(response: str, expected_count: int) -> list[int] | None:
    keyed_matches = re.findall(
        r"(?:prompt_)?(\d{1,2})\s*[:=]\s*(yes|no|true|false|-?1|0)",
        response,
        flags=re.IGNORECASE,
    )
    if len(keyed_matches) >= expected_count:
        values = [-1] * expected_count
        for index_text, value in keyed_matches:
            index = int(index_text)
            if 0 <= index < expected_count:
                values[index] = coerce_prompt_value(value)
        return values

    loose_values = re.findall(r"\b(?:yes|no|true|false|-?1|0)\b", response, flags=re.IGNORECASE)
    if len(loose_values) >= expected_count:
        return normalize_prompt_values(loose_values, expected_count)

    return None


def parse_prompt_feature_json(response: str, expected_count: int = len(PROMPT_COLUMNS)) -> list[int]:
    payload = extract_json_candidate(response)
    values = values_from_json_payload(payload, expected_count)
    if values is not None:
        return normalize_prompt_values(values, expected_count)

    indexed_values = parse_indexed_prompt_values(response, expected_count)
    if indexed_values is not None:
        return indexed_values

    raise gr.Error(
        "Together did not return parseable prompt scores. Try again, or set "
        "TOGETHER_PROMPT_MODE=exact for the slower fallback."
    )


def build_single_call_prompt(text: str) -> str:
    numbered_questions = "\n".join(
        f"{index}. {question}" for index, question in enumerate(PROMPT_QUESTIONS)
    )
    return (
        "Read the text and answer each numbered question.\n"
        "Return only a valid JSON object with exactly one key named \"scores\".\n"
        "The value of \"scores\" must be an array of exactly 63 integers in the same order as the questions.\n"
        "Use 1 for yes, 0 for no, and -1 only if the answer is unclear.\n\n"
        f"Text:\n{text}\n\n"
        f"Questions:\n{numbered_questions}\n\n"
        "Return format: {\"scores\": [1, 0, -1, ...]}"
    )


def generate_prompt_features_single_call(text: str) -> dict[str, float]:
    response = call_together(build_single_call_prompt(text), max_tokens=512, json_mode=True)
    values = parse_prompt_feature_json(response)
    return {column: float(values[index]) for index, column in enumerate(PROMPT_COLUMNS)}


def build_chunk_call_prompt(text: str, questions: list[str]) -> str:
    numbered_questions = "\n".join(
        f"{index}. {question}" for index, question in enumerate(questions)
    )
    return (
        "Read the text and answer each numbered question.\n"
        "Return only a valid JSON object with exactly one key named \"scores\".\n"
        f"The value of \"scores\" must be an array of exactly {len(questions)} integers "
        "in the same order as the questions.\n"
        "Use 1 for yes, 0 for no, and -1 only if the answer is unclear.\n\n"
        f"Text:\n{text}\n\n"
        f"Questions:\n{numbered_questions}\n\n"
        "Return format: {\"scores\": [1, 0, -1, ...]}"
    )


def prompt_question_chunks() -> list[list[str]]:
    chunk_size = max(1, int(np.ceil(len(PROMPT_QUESTIONS) / TOGETHER_CHUNK_COUNT)))
    return [
        PROMPT_QUESTIONS[start : start + chunk_size]
        for start in range(0, len(PROMPT_QUESTIONS), chunk_size)
    ]


def generate_prompt_features_chunked(text: str) -> dict[str, float]:
    values: list[int] = []
    for questions in prompt_question_chunks():
        response = call_together(
            build_chunk_call_prompt(text, questions),
            max_tokens=192,
            json_mode=True,
        )
        values.extend(parse_prompt_feature_json(response, expected_count=len(questions)))

    values = normalize_prompt_values(values, len(PROMPT_COLUMNS))
    return {column: float(values[index]) for index, column in enumerate(PROMPT_COLUMNS)}


def generate_prompt_features_exact_calls(text: str) -> dict[str, float]:
    values = []
    for question in PROMPT_QUESTIONS:
        response = call_together(build_prompt_raw(text, question), max_tokens=10)
        values.append(parse_yes_no(response))
    return {column: float(values[index]) for index, column in enumerate(PROMPT_COLUMNS)}


def generate_prompt_features(text: str) -> dict[str, float]:
    if TOGETHER_PROMPT_MODE == "exact":
        return generate_prompt_features_exact_calls(text)
    if TOGETHER_PROMPT_MODE == "single":
        return generate_prompt_features_single_call(text)
    return generate_prompt_features_chunked(text)


def predict_static_from_features(features: dict[str, float]) -> dict[str, float]:
    row = pd.DataFrame([features], columns=STATIC_COLUMNS).fillna(0.0)
    selected = selector_static.transform(row.values)
    scaled = scaler_static.transform(selected)
    probabilities = model_static.predict_proba(scaled)[0]
    return format_prediction(list(model_static.classes_), probabilities)


def predict_combo_from_features(
    static_features: dict[str, float], prompt_features: dict[str, float]
) -> dict[str, float]:
    row = pd.DataFrame([{**static_features, **prompt_features}], columns=COMBO_COLUMNS).fillna(0.0)
    selected = selector_combo.transform(row.values)
    scaled = scaler_combo.transform(selected)
    probabilities = model_combo.predict_proba(scaled)[0]
    return format_prediction(list(model_combo.classes_), probabilities)


def classify_combo(text: str) -> dict[str, float]:
    text = str(text or "").strip()
    if not text:
        return empty_prediction()

    features = compute_static_features(text)
    prompt_features = generate_prompt_features(text)
    return predict_combo_from_features(features, prompt_features)


# ── Gradio UI ────────────────────────────────────────────────────────────────
EXAMPLES = [
    ["The cat sat on the mat. It was warm and cozy."],
    ["Plants use sunlight to make food through a process called photosynthesis."],
    ["Climate change affects ecosystems, agriculture, and public health across the globe."],
    ["The mitochondria produces ATP via cellular respiration, using glucose and oxygen."],
    ["The legislature ratified the constitutional amendment after prolonged bipartisan negotiations."],
    ["Quantum entanglement describes a phenomenon where particles remain correlated regardless of distance."],
]

css = """
body, .gradio-container, .dark .gradio-container, .dark body {
    background-color: white !important;
    color: #1f2937 !important;
}
* {
    color: #1f2937 !important;
}
.gr-button-primary, button[variant="primary"] {
    color: white !important;
    background-color: #f97316 !important;
}
/* no hover highlight on examples */
.examples tbody tr:hover,
.examples tbody tr:hover td,
[class*="examples"] tr:hover,
[class*="examples"] tr:hover td,
[class*="gallery-item"]:hover {
    background-color: transparent !important;
    background: transparent !important;
    cursor: pointer;
}
/* force all blocks/panels to white regardless of theme */
.block, .panel, .form, [class*="block"], [class*="panel"] {
    background-color: white !important;
    border-color: #e5e7eb !important;
}
/* textbox always light */
textarea, input[type="text"], .scroll-hide {
    background-color: white !important;
    color: #1f2937 !important;
    border-color: #e5e7eb !important;
}
/* predicted level label always light */
[class*="label"], [class*="output"], .output-class, .bar {
    background-color: white !important;
    color: #1f2937 !important;
}
[class*="bar-wrap"], [class*="bar"] {
    background-color: #f3f4f6 !important;
}
[class*="bar-fill"] {
    background-color: #6366f1 !important;
}
/* predicted level highlight always white */
[class*="label"] [class*="selected"],
[class*="label"] [class*="choice"],
[class*="label"] [class*="highlight"],
[class*="output-class"] {
    background-color: white !important;
    color: #1f2937 !important;
}
[class*="column"] {
    background-color: white !important;
}
.float, [class*="float"] {
    background-color: white !important;
    color: #1f2937 !important;
}
"""

js = """
() => {
    const url = new URL(window.location.href);
    url.searchParams.set('__theme', 'light');
    if (window.location.href !== url.toString()) {
        window.location.replace(url.toString());
    }
}
"""

with gr.Blocks(
    title="Rooein Model Combo Reading Level Classifier",
    theme=gr.themes.Default(),
    css=css,
    js=js,
) as demo:
    gr.Markdown(
        """# 📚 Reading Level Classifier — Rooein Model Combo
Classify raw text into elementary, middle, or high school reading level."""
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=8,
                placeholder="Paste a sentence, paragraph, or passage here...",
                label="Input text",
            )
            with gr.Row():
                submit_btn = gr.Button("Classify", variant="primary")
                gr.ClearButton(text_input, value="Clear")

        with gr.Column(scale=1):
            gr.Markdown("### Rooein Model Combo")
            gr.Markdown("*Static features + 63 API-generated yes/no prompt features*")
            combo_output = gr.Label(num_top_classes=3, label="Predicted level")

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

    submit_btn.click(classify_combo, inputs=text_input, outputs=combo_output)
    text_input.submit(classify_combo, inputs=text_input, outputs=combo_output)

if __name__ == "__main__":
    demo.queue(max_size=10).launch(show_api=False)
