from __future__ import annotations
from typing import Any, List, Tuple, Dict

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter, ParameterList
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
print(f"Loading on {device} …")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — DANN (ELECTRA-large + ScalarMix)
# ══════════════════════════════════════════════════════════════════════════════
DANN_MODEL_NAME = "google/electra-large-discriminator"
MAX_LEN         = 512

DANN_CKPT_PATH = hf_hub_download(
    repo_id="sdanda99/demo_models",
    filename="best_model.pt",
    token=os.environ.get("HF_TOKEN"),
)

label2id = {"elementary": 0, "middle": 1, "high": 2}
id2label  = {v: k for k, v in label2id.items()}


class ScalarMix(nn.Module):
    def __init__(self, mixture_size: int, trainable: bool = True) -> None:
        super().__init__()
        self.scalar_parameters = ParameterList(
            [Parameter(torch.zeros(1), requires_grad=trainable) for _ in range(mixture_size)]
        )
        self.gamma = Parameter(torch.ones(1), requires_grad=trainable)

    def forward(self, tensors: List[Tensor]) -> Tensor:
        w = F.softmax(torch.cat(list(self.scalar_parameters)), dim=0)
        w = torch.split(w, 1)
        return self.gamma * sum(weight * t for weight, t in zip(w, tensors))


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, lambda_: float) -> Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class DifficultyClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DomainClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_domains: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_domains),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ElectraScalarMixDANN(nn.Module):
    def __init__(self, model_name, num_classes, num_domains, head_in_dim=None, dropout=0.2):
        super().__init__()
        self.encoder    = AutoModel.from_pretrained(model_name)
        hidden          = int(self.encoder.config.hidden_size)
        n_layers        = int(self.encoder.config.num_hidden_layers) + 1
        self.scalar_mix = ScalarMix(n_layers)
        self.dropout    = nn.Dropout(dropout)
        in_dim          = head_in_dim if head_in_dim is not None else hidden
        self.difficulty_head = DifficultyClassifierHead(in_dim, num_classes, dropout)
        self.domain_head     = DomainClassifierHead(in_dim, num_domains, dropout)

    def encode_pooled(self, input_ids, attention_mask):
        out    = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        mixed  = self.dropout(self.scalar_mix(list(out.hidden_states)))
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (mixed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        expected = self.difficulty_head.net[0].in_features
        if pooled.shape[-1] < expected:
            pad    = torch.zeros(pooled.shape[0], expected - pooled.shape[-1], device=pooled.device)
            pooled = torch.cat([pooled, pad], dim=-1)
        return pooled

    def difficulty_logits_only(self, input_ids, attention_mask):
        return self.difficulty_head(self.encode_pooled(input_ids, attention_mask))


ckpt        = torch.load(DANN_CKPT_PATH, map_location=device)
domain2id   = ckpt.get("domain2id", {"default": 0})
head_in_dim = ckpt["state_dict"]["difficulty_head.net.0.weight"].shape[1]

dann_model = ElectraScalarMixDANN(
    DANN_MODEL_NAME, num_classes=3, num_domains=len(domain2id), head_in_dim=head_in_dim
).to(device)
dann_model.load_state_dict(ckpt["state_dict"])
dann_model.eval()

dann_tokenizer = AutoTokenizer.from_pretrained(DANN_MODEL_NAME)
print("DANN model ready.")


def predict_dann(text: str) -> Dict[str, float]:
    text = str(text).strip()
    if not text:
        return {"elementary": 0.0, "middle": 0.0, "high": 0.0}
    enc = dann_tokenizer(text, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = dann_model.difficulty_logits_only(enc["input_ids"], enc["attention_mask"])
        probs  = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return {id2label[i]: float(probs[i]) for i in range(len(probs))}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — Pillar B+ (Phi-3.5-mini + LoRA)
# ══════════════════════════════════════════════════════════════════════════════
LORA_BASE_MODEL   = "microsoft/Phi-3.5-mini-instruct"
LORA_ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "sdanda99/pillar-b-plus-lora")

lora_tok  = AutoTokenizer.from_pretrained(LORA_BASE_MODEL)
if lora_tok.pad_token_id is None:
    lora_tok.pad_token = lora_tok.eos_token
lora_tok.padding_side = "left"

lora_base  = AutoModelForCausalLM.from_pretrained(LORA_BASE_MODEL, torch_dtype=dtype, attn_implementation="sdpa").to(device)
lora_model = PeftModel.from_pretrained(lora_base, LORA_ADAPTER_REPO).to(device).eval()
print("LoRA model ready.")

INSTRUCTION = (
    "Read the following text and classify it by the curriculum grade level "
    "required to understand its CONCEPTS (not just its reading complexity). "
    "Answer with only one letter: E for elementary school (US grades 1-5), "
    "M for middle school (US grades 6-8), H for high school (US grades 9-12)."
)
LEVELS     = ["elementary", "middle", "high"]
GRADE_BAND = {"elementary": "US grades 1-5", "middle": "US grades 6-8", "high": "US grades 9-12"}


def _letter_ids(tokenizer):
    out = {}
    for letter in "EMH":
        for candidate in [f" {letter}", letter]:
            ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                out[letter] = ids[0]
                break
        else:
            out[letter] = tokenizer(f" {letter}", add_special_tokens=False)["input_ids"][0]
    return out

letter_ids = _letter_ids(lora_tok)


@torch.no_grad()
def predict_lora(text: str) -> Dict[str, float]:
    text = (text or "").strip()
    if not text:
        return {"elementary": 0.0, "middle": 0.0, "high": 0.0}
    prompt = f"{INSTRUCTION}\n\nText: {text}\n\nAnswer:"
    enc    = lora_tok(prompt, return_tensors="pt", truncation=True, max_length=1280).to(device)
    out    = lora_model(**enc, use_cache=False)
    last   = out.logits[0, -1, :]
    logits = torch.stack([last[letter_ids["E"]], last[letter_ids["M"]], last[letter_ids["H"]]]).float()
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    return {LEVELS[i]: float(probs[i]) for i in range(3)}


# ══════════════════════════════════════════════════════════════════════════════
# Combined inference
# ══════════════════════════════════════════════════════════════════════════════
def classify_both(text: str):
    if not (text or "").strip():
        empty = {"elementary": 0.0, "middle": 0.0, "high": 0.0}
        return empty, empty
    return predict_dann(text), predict_lora(text)


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
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

with gr.Blocks(title="Reading Level Classifier Comparison", theme=gr.themes.Default(), css=css, js=js) as demo:
    gr.Markdown(
        """# 📚 Reading Level Classifier — Model Comparison
Compare two approaches to text difficulty classification side by side."""
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
                clear_btn  = gr.ClearButton(text_input, value="Clear")

        with gr.Column(scale=1):
            gr.Markdown("### ELECTRA + ScalarMix + DANN")
            gr.Markdown("*Trained on CNN/DailyMail · OneStop · RACE*")
            dann_out = gr.Label(num_top_classes=3, label="Predicted level")

        with gr.Column(scale=1):
            gr.Markdown("### Phi-3.5-mini + LoRA (Pillar B+)")
            gr.Markdown("*Concept-level curriculum classifier*")
            lora_out = gr.Label(num_top_classes=3, label="Predicted level")

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="Try an example")

    submit_btn.click(classify_both, inputs=text_input, outputs=[dann_out, lora_out])
    text_input.submit(classify_both, inputs=text_input, outputs=[dann_out, lora_out])

demo.launch(show_api=False)