"""Generate synthetic Q/A training data via the Claude API and save to CSV.

Splits the work across multiple calls so the output never gets truncated.
Each call is steered toward a different subject area to reduce overlap
between calls, and exact-duplicate questions are dropped at the end.
"""
import json
import anthropic
import pandas as pd

N_SAMPLES = 1500
PER_CALL = 300
MODEL = "claude-haiku-4-5-20251001"
OUT_CSV = "synthetic_claude_qa.csv"

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

Return your response as STRICT JSON — a single JSON array, nothing before or after it,
no markdown code fences. Each element must be an object with exactly these keys:

  "question"    — string
  "answer"      — string
  "grade_level" — one of "elementary", "middle", "high"

Begin the JSON array now:"""


def parse_json_array(text: str) -> list[dict]:
    """Forgiving parser — recovers complete objects if the array was truncated."""
    start = text.find("[")
    end = text.rfind("]")
    if end > start:
        candidate = text[start:end + 1]
    else:
        candidate = text[start:text.rfind("}") + 1] + "]"
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return json.loads(candidate[:candidate.rfind("},") + 1] + "]")


client = anthropic.Anthropic(api_key=API_KEY)
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
    with client.messages.stream(
        model=MODEL,
        max_tokens=32000,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            chunks.append(chunk)
            print(chunk, end="", flush=True)
        resp = stream.get_final_message()

    text = "".join(chunks)
    print(f"\n  Usage: {resp.usage.input_tokens} in / {resp.usage.output_tokens} out")

    items = parse_json_array(text)
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