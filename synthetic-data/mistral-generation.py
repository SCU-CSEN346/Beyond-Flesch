"""Generate synthetic Q/A training data via the Mistral API and save to CSV.

Splits the work across multiple calls so the output never gets truncated.
Each call is steered toward a different subject area to reduce overlap
between calls, and exact-duplicate questions are dropped at the end.

Requires:
    pip install mistralai pandas
    export MISTRAL_API_KEY="..."   (or set API_KEY below)
"""
import json
import os
from mistralai.client import Mistral
import pandas as pd

N_SAMPLES = 1500
PER_CALL = 150
MODEL = "mistral-large-latest"   # swap to "mistral-large-latest" for higher quality
OUT_CSV = "synthetic_mistral_qa.csv"

API_KEY = os.environ.get("MISTRAL_API_KEY")  # or paste your key here

# Each call focuses on a different subject area to reduce overlap.
SUBJECT_FOCI = [
    "natural sciences (biology, ecology, earth science, weather, animals, plants)",
    "physical sciences and mathematics (physics, chemistry, arithmetic, geometry, algebra, statistics)",
    "social studies (world history, U.S. history, geography, civics, economics)",
    "language arts (vocabulary, grammar, reading comprehension, literature)",
    "applied and everyday topics (technology, health, sports, nutrition, current events, cultural traditions)",
]

PROMPT_TEMPLATE = """You are generating synthetic training data for a text-difficulty classifier.

Generate exactly {n} question-and-answer pairs, with {per_level} per
grade level (elementary, middle, high).

Subject focus for THIS batch: {subject}
Stay within this subject area for all {n} pairs in this batch.

Grade level style guide:
- elementary: Grades 1-5. Short sentences, concrete vocabulary, simple recall or one-step reasoning. Brief answers.
- middle: Grades 6-8. Grade-appropriate vocabulary, multi-step reasoning, comparing/classifying/applying concepts.
- high: Grades 9-12. Domain-specific terminology where natural, analysis/synthesis, substantive explanations.

Vary the specific sub-topic and question form within the subject area
(recall, comparison, application, analysis). Do not repeat questions.

Return your response as a single JSON object with one key "samples" whose
value is an array of exactly {n} objects. Each object in the array must
have exactly these keys:

  "question"    — string
  "answer"      — string
  "grade_level" — one of "elementary", "middle", "high"

Example shape: {{"samples": [{{"question": "...", "answer": "...", "grade_level": "elementary"}}, ...]}}
"""


def parse_samples(text: str) -> list[dict]:
    """JSON mode guarantees a valid JSON object; pull the 'samples' array out."""
    obj = json.loads(text)
    return obj.get("samples", obj if isinstance(obj, list) else [])


client = Mistral(api_key=API_KEY)
n_calls = (N_SAMPLES + PER_CALL - 1) // PER_CALL
all_rows: list[dict] = []

for i in range(n_calls):
    remaining = N_SAMPLES - len(all_rows)
    n_this_call = min(PER_CALL, remaining)
    n_this_call = (n_this_call // 3) * 3
    if n_this_call == 0:
        break

    subject = SUBJECT_FOCI[i % len(SUBJECT_FOCI)]
    prompt = PROMPT_TEMPLATE.format(
        n=n_this_call,
        per_level=n_this_call // 3,
        subject=subject,
    )
    print(f"\nCall {i + 1}/{n_calls} — {n_this_call} samples on: {subject}")
    print("Streaming…")

    chunks: list[str] = []
    usage_info = None
    stream = client.chat.stream(
        model=MODEL,
        max_tokens=8000,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
    )
    for event in stream:
        data = event.data
        if data.choices and data.choices[0].delta.content:
            piece = data.choices[0].delta.content
            chunks.append(piece)
            print(piece, end="", flush=True)
        if getattr(data, "usage", None):
            usage_info = data.usage

    text = "".join(chunks)
    if usage_info:
        print(f"\n  Usage: {usage_info.prompt_tokens} in / {usage_info.completion_tokens} out")
    else:
        print()

    try:
        items = parse_samples(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse failed: {e}. Skipping this call.")
        continue

    all_rows.extend(items)
    print(f"  Parsed {len(items)} rows. Total so far: {len(all_rows)}")

# Build DataFrame, deduplicate on question text, save.
df = pd.DataFrame(all_rows)
before = len(df)
df = df.drop_duplicates(subset="question").reset_index(drop=True)
print(f"\nDropped {before - len(df)} duplicate questions ({len(df)} unique remaining).")

df = df.head(N_SAMPLES)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUT_CSV}")
print(df["grade_level"].value_counts().to_dict())