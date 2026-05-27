"""
Pillar B+ live demo

This demo classifies one piece of text into a curriculum level:
elementary, middle, or high school.

It uses Phi-3.5-mini with our trained LoRA adapter. Instead of asking the
model many prompt questions, it makes one forward pass and reads the scores
for E, M, and H.

Run this file from the repository root with the project environment active.

Then open:

    http://127.0.0.1:7862

Device choice is automatic:
CUDA is fastest, Apple MPS is slower, and CPU works but can take longer.
"""

import math
import os
import sys
import time

import gradio as gr
import torch
from codecarbon import EmissionsTracker

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT = os.path.dirname(_PROJECT_DIR)

sys.path.insert(0, os.path.join(_PROJECT_DIR, "scripts"))
from pillar_b_tiny_teacher import build_prompt, _find_letter_token_ids  # noqa: E402

os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/hf_user/hub"))

BACKBONE = "microsoft/Phi-3.5-mini-instruct"
ADAPTER = os.path.join(
    _PROJECT_DIR,
    "outputs",
    "models",
    "pillar_b_plus__phi-3.5-mini-instruct__lora",
    "best",
)
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, "outputs", "emissions")

LABEL_NAMES = ["elementary", "middle", "high"]
GRADE_BAND = {
    "elementary": "US grades 1-5",
    "middle": "US grades 6-8",
    "high": "US grades 9-12",
}


class DemoState:
    model = None
    tokenizer = None
    letter_ids = None
    device = None
    ready = False


STATE = DemoState()


def _pick_device_and_dtype():
    """Pick the best available device for this machine."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16

    return torch.device("cpu"), torch.float32


def initialize():
    """Load the base model and attach our trained LoRA adapter."""
    device, dtype = _pick_device_and_dtype()
    print(f"[demo] Loading {BACKBONE} on {device}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    if not os.path.isdir(ADAPTER):
        raise FileNotFoundError(
            f"LoRA adapter not found at {ADAPTER}. "
            "Train it first with: python gen_v2/scripts/pillar_b_plus.py"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        BACKBONE,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(device)

    model = PeftModel.from_pretrained(base_model, ADAPTER).to(device).eval()

    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.letter_ids = _find_letter_token_ids(tokenizer)
    STATE.device = next(model.parameters()).device
    STATE.ready = True

    print(f"[demo] Ready on {STATE.device}. Letter token IDs: {STATE.letter_ids}")


@torch.no_grad()
def _predict_one(text):
    """
    Classify one text.

    The model sees the text inside our fixed instruction prompt. Then we look
    only at the next-token scores for E, M, and H.
    """
    prompt = build_prompt(text)

    encoded = STATE.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1280,
    ).to(STATE.device)

    start = time.perf_counter()
    output = STATE.model(**encoded, use_cache=False)

    if STATE.device.type == "cuda":
        torch.cuda.synchronize()
    elif STATE.device.type == "mps":
        torch.mps.synchronize()

    latency_ms = (time.perf_counter() - start) * 1000

    last_token_logits = output.logits[0, -1, :]

    e_logit = float(last_token_logits[STATE.letter_ids["E"]])
    m_logit = float(last_token_logits[STATE.letter_ids["M"]])
    h_logit = float(last_token_logits[STATE.letter_ids["H"]])

    # Stable softmax over the three label scores.
    max_logit = max(e_logit, m_logit, h_logit)
    e_score = math.exp(e_logit - max_logit)
    m_score = math.exp(m_logit - max_logit)
    h_score = math.exp(h_logit - max_logit)

    total = e_score + m_score + h_score

    probs = {
        "elementary": e_score / total,
        "middle": m_score / total,
        "high": h_score / total,
    }

    return probs, latency_ms


def classify(text):
    if not STATE.ready:
        return {}, "Model is still loading. Try again in a few seconds.", ""

    text = (text or "").strip()

    if not text:
        return {}, "Paste some text to classify.", ""

    probs, latency_ms = _predict_one(text)

    label = max(probs, key=probs.get)
    confidence = probs[label]

    detail = (
        f"### Prediction: **{label}** ({GRADE_BAND[label]})\n\n"
        f"**Confidence:** {confidence * 100:.1f}%\n\n"
        f"| level | probability |\n"
        f"| --- | --- |\n"
        f"| elementary | {probs['elementary'] * 100:5.1f}% |\n"
        f"| middle | {probs['middle'] * 100:5.1f}% |\n"
        f"| high | {probs['high'] * 100:5.1f}% |\n\n"
        f"**Inference latency:** {latency_ms:.1f} ms\n\n"
        f"**Text length:** {len(text)} characters / {len(text.split())} words"
    )

    return probs, detail, f"{latency_ms:.1f} ms"


EXAMPLES = [
    ["What is mitosis?"],
    ["What is photosynthesis?"],
    ["Define angular momentum."],
    ["Calculate the derivative of x squared."],
    ["Use Cramer's rule."],
    ["Why is the sky blue?"],
    ["What is fertilization?"],
    [
        "The small brown dog ran across the bright green grass in the sunny park "
        "while his happy owner cheerfully waved at him from the wooden bench under "
        "the tall leafy tree, and then the dog stopped suddenly because he saw a "
        "beautiful red ball lying on the soft grass nearby."
    ],
    [
        "Yesterday afternoon during the bright sunny weather, the curious little "
        "girl named Sarah carefully picked up the colorful round apple from the "
        "green wicker basket on the wooden kitchen table and then thoughtfully "
        "placed it inside her favorite blue lunchbox."
    ],
    ["Two plus two equals four."],
    ["Solve for x: 3x + 7 = 22."],
    [
        "Photosynthesis involves the light-dependent reactions in the thylakoid "
        "membrane, where photosystems II and I drive electron transport to generate "
        "ATP and NADPH, which are subsequently consumed by the Calvin-Benson cycle "
        "in the stroma to fix CO2 into G3P."
    ],
]


def build_ui():
    with gr.Blocks(title="Pillar B+ Text-Difficulty Classifier") as demo:
        gr.Markdown(
            "# Pillar B+ Text-Difficulty Classifier\n\n"
            "This demo predicts the curriculum level of a text: elementary, "
            "middle, or high school.\n\n"
            "The model is Phi-3.5-mini with a trained LoRA adapter. It does not "
            "run 63 separate prompt questions. It reads the text once, then scores "
            "the three answer choices: E, M, and H.\n\n"
            "The goal is to classify by the concept being taught, not only by how "
            "long or complicated the sentence looks."
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_box = gr.Textbox(
                    lines=8,
                    max_lines=20,
                    label="Paste text to classify",
                    placeholder=(
                        'Examples:\n'
                        '  "What is mitosis?"\n'
                        '  "Why is the sky blue?"\n'
                        '  "The small brown dog ran across the bright green grass..."'
                    ),
                )

                classify_button = gr.Button("Classify", variant="primary")

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=[text_box],
                    label="Try an example",
                )

            with gr.Column(scale=1):
                label_output = gr.Label(
                    label="Class probabilities",
                    num_top_classes=3,
                )
                detail_output = gr.Markdown()
                latency_output = gr.Textbox(
                    label="Latency",
                    interactive=False,
                    max_lines=1,
                )

        classify_button.click(
            classify,
            inputs=[text_box],
            outputs=[label_output, detail_output, latency_output],
        )

        gr.Markdown(
            "---\n\n"
            "### Model details\n\n"
            "- **Backbone:** `microsoft/Phi-3.5-mini-instruct`, frozen during fine-tuning.\n"
            "- **Adapter:** LoRA, trained for this curriculum-level classification task.\n"
            "- **Training data:** about 17K examples from 11 educational/readability corpora.\n"
            "- **Labels:** unified with an LLM judge so all corpora use the same three labels.\n"
            "- **Final task:** predict elementary, middle, or high school curriculum level.\n\n"
            "### Why these examples matter\n\n"
            "Some texts look easy because they are short, but the concept is hard. "
            "Other texts look hard because they are long, but the concept is simple. "
            "These examples test whether the model is reading the concept, not just "
            "reacting to sentence length.\n\n"
            "*Caveat:* very short questions with little context can still be ambiguous, "
            "even for humans."
        )

    return demo


if __name__ == "__main__":
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    tracker = EmissionsTracker(
        project_name="pillar_b_plus_demo",
        output_dir=EMISSIONS_DIR,
        log_level="warning",
    )

    tracker.start()

    try:
        initialize()
        ui = build_ui()

        print("[demo] Launching UI on http://127.0.0.1:7862")
        ui.launch(server_name="0.0.0.0", server_port=7862, share=False)

    finally:
        tracker.stop()
