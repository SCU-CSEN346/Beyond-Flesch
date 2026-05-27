"""
Probe whether the existing judge labels need a stronger second opinion.

This asks an external judge to label AdvConcept rows and a small sample from the
training set. The result is an agreement check, not a replacement label file.

Use --smoke first when checking credentials or rate limits.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR     = os.path.join(_PROJECT_DIR, 'data')
EMISSIONS_DIR = os.path.join(_PROJECT_DIR, 'outputs', 'emissions')

sys.path.insert(0, _SCRIPT_DIR)
from evaluate_all import GF_OUT, ADV_CSV  # noqa: E402

JUDGE_MODEL  = 'deepseek-ai/deepseek-v4-flash'  # DeepSeek family on NIM, more generous free-tier rate limits than v4-pro
NIM_BASE_URL = 'https://integrate.api.nvidia.com/v1'
THROTTLE_SEC = 2.0  # inter-request sleep to stay under NIM free-tier RPM limits

JUDGE_PROMPT = (
    "You are a K-12 curriculum specialist evaluating educational text.\n\n"
    "Read the text below and classify it by the curriculum grade level "
    "required to understand its CONCEPTS — NOT its reading complexity, "
    "vocabulary, or length.\n\n"
    "Categories:\n"
    "  E = elementary school (US grades 1-5)\n"
    "  M = middle school     (US grades 6-8)\n"
    "  H = high school       (US grades 9-12)\n\n"
    "A short question like \"What is mitosis?\" is M (middle-school biology) "
    "even though the words are simple. A long wordy paragraph about \"the "
    "cat on the mat\" is E (elementary) even though the surface is complex.\n\n"
    "Think briefly about what the underlying concept is and what grade-level "
    "standard it aligns with. Then, on the VERY LAST line of your response, "
    "output ONLY a single letter (no period, no other text): E, M, or H.\n\n"
    "TEXT:\n{text}\n\n"
    "Reasoning, then the letter on its own final line:"
)

LABEL_FROM_LETTER = {'E': 'elementary', 'M': 'middle', 'H': 'high'}


def _display_path(path):
    return os.path.relpath(path, _PROJECT_DIR)


def _load_nvidia_key():
    """Load the NVIDIA NIM key from the environment."""
    k = os.environ.get('NVIDIA_API_KEY') or os.environ.get('NGC_API_KEY')
    if k and k.startswith('nvapi-'):
        return k
    raise RuntimeError(
        "NVIDIA_API_KEY not set or doesn't look like a NIM key. "
        "Export it: `export NVIDIA_API_KEY=nvapi-...` then re-run.")


def _parse_letter(response_text):
    """Extract the final E/M/H label from a judge response."""
    if not response_text:
        return None
    lines = [l.strip() for l in response_text.strip().splitlines() if l.strip()]
    if not lines:
        return None
    last = lines[-1].rstrip('.').rstrip(':').rstrip(',').strip()
    if last in ('E', 'M', 'H'):
        return last
    # Last-token check on last line: "Answer: M" / "Final: H" / "**E**"
    m_last = re.search(r'(?:^|[\s\*])([EMH])\s*[\.\*]*\s*$', last)
    if m_last:
        return m_last.group(1)
    # Whole-response fallback: take the LAST standalone E/M/H token.
    # \b uses word boundaries (treats both letters AND digits as word
    # characters), so this won't false-match in "H1", "Grade-E", or "M-phase".
    matches = re.findall(r'\b([EMH])\b', response_text)
    if matches:
        return matches[-1]
    return None


def _judge_one(client, model, text, max_retries=5):
    """Call the judge once and retry transient API errors."""
    import openai
    delay = 5.0  # initial backoff (5, 10, 20, 40, 80 = 155s total)
    last_err = None
    for attempt in range(max_retries):
        try:
            msg = client.chat.completions.create(
                model=model,
                max_tokens=300,
                temperature=0.0,
                messages=[{'role': 'user',
                           'content': JUDGE_PROMPT.format(text=text[:4000])}],
                timeout=30.0,  # don't let a single hung NIM call stall the run
            )
            choice = msg.choices[0] if msg.choices else None
            raw = (choice.message.content or '') if choice else ''
            letter = _parse_letter(raw)
            usage = getattr(msg, 'usage', None)
            in_t  = getattr(usage, 'prompt_tokens',     0) if usage else 0
            out_t = getattr(usage, 'completion_tokens', 0) if usage else 0
            return letter, raw, in_t, out_t
        except (openai.AuthenticationError, openai.BadRequestError,
                openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise
        except (openai.RateLimitError, openai.APIConnectionError,
                openai.InternalServerError, openai.APITimeoutError) as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                print(f"[probe]    retry {attempt+1}/{max_retries} after "
                      f"{type(e).__name__}: sleeping {wait:.1f}s")
                time.sleep(wait)
            continue
        except openai.APIError as e:
            last_err = e
            if attempt < max_retries - 1:
                wait = delay * (2 ** attempt)
                print(f"[probe]    retry {attempt+1}/{max_retries} after "
                      f"APIError: sleeping {wait:.1f}s")
                time.sleep(wait)
            continue
    raise RuntimeError(f"NIM call failed after {max_retries} retries: "
                        f"{last_err!r}")


def _load_train_sample(n, seed=42):
    """Sample training rows with their existing judge labels."""
    tr = pd.read_csv(os.path.join(GF_OUT, 'splits', 'train.csv'))
    jd = pd.read_csv(os.path.join(GF_OUT, 'llm_judge', 'train_judge.csv'))
    if len(tr) != len(jd):
        raise RuntimeError(f"row mismatch: train.csv={len(tr)} vs "
                            f"train_judge.csv={len(jd)}")
    text_col = 'text' if 'text' in tr.columns else 'full_text'
    df = pd.DataFrame({
        'orig_idx':         np.arange(len(tr)),
        'text':             tr[text_col].astype(str).values,
        'llama_label':      jd['llm_judge_label'].astype(str).values,
    })
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


def _load_adv():
    df = pd.read_csv(ADV_CSV)
    return pd.DataFrame({
        'orig_idx':   np.arange(len(df)),
        'text':       df['text'].astype(str).values,
        'gold_label': df['true_level'].astype(str).values,
        'category':   df['category'].astype(str).values,
    })


CSV_COLUMNS = ['source', 'orig_idx', 'text', 'gold_label', 'llama_label',
                'category', 'judge_letter', 'judge_label', 'judge_raw']


def _open_writer(path):
    """Open a CSV writer and flush each row."""
    import csv
    fh = open(path, 'w', newline='', buffering=1)  # line-buffered
    w  = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    w.writeheader()
    return fh, w


def _emit(writer, fh, row):
    writer.writerow({k: row.get(k, '') for k in CSV_COLUMNS})
    fh.flush()
    os.fsync(fh.fileno())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=200,
                    help='train rows to sample for judge vs Llama comparison')
    ap.add_argument('--no-adv', action='store_true',
                    help='skip the AdvConcept-50 re-label half of the probe')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--model', type=str, default=JUDGE_MODEL)
    ap.add_argument('--smoke', action='store_true',
                    help='5-row smoke test (5 adv + 5 train), ~$0.01, ~1 min')
    args = ap.parse_args()

    os.makedirs(DATA_DIR,      exist_ok=True)
    os.makedirs(EMISSIONS_DIR, exist_ok=True)

    api_key = _load_nvidia_key()
    print(f"[probe] NVIDIA NIM key found (len={len(api_key)}), model={args.model}")
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=NIM_BASE_URL)

    suffix = '_smoke' if args.smoke else ''
    out_csv = os.path.join(DATA_DIR, f'judge_probe{suffix}.csv')
    out_txt = os.path.join(DATA_DIR, f'judge_probe_summary{suffix}.txt')

    fh, writer = _open_writer(out_csv)
    in_tok = out_tok = 0
    rows_for_summary = []

    tracker = EmissionsTracker(project_name='probe_judges',
                                output_dir=EMISSIONS_DIR, log_level='warning')
    tracker.start()
    try:
        # ---- Part 1: AdvConcept-50 (or first 5 of it in smoke mode) ----
        if not args.no_adv:
            adv = _load_adv()
            if args.smoke:
                adv = adv.head(5).reset_index(drop=True)
            print(f"[probe] judging AdvConcept-50 ({len(adv)} rows) ...")
            t0 = time.time()
            for i, r in adv.iterrows():
                letter, raw, it, ot = _judge_one(client, args.model, r['text'])
                in_tok += it; out_tok += ot
                time.sleep(THROTTLE_SEC)
                row = {
                    'source':          'adv_concept',
                    'orig_idx':        int(r['orig_idx']),
                    'text':            r['text'],
                    'gold_label':      r['gold_label'],
                    'llama_label':     '',
                    'category':        r['category'],
                    'judge_letter':   letter or '?',
                    'judge_label':    LABEL_FROM_LETTER.get(letter, 'unknown'),
                    'judge_raw':      raw,
                }
                _emit(writer, fh, row)
                rows_for_summary.append(row)
                if (i + 1) % 5 == 0 or args.smoke:
                    rate = (time.time() - t0) / (i + 1)
                    print(f"[probe]   adv {i+1}/{len(adv)}  "
                          f"({rate:.2f}s/row)  letter={letter or '?'}")
            print(f"[probe] adv took {(time.time()-t0)/60:.1f} min")

        # ---- Part 2: train sample vs Llama ----
        n_train = 5 if args.smoke else args.n_train
        tr = _load_train_sample(n_train, seed=args.seed)
        print(f"[probe] judging {len(tr)} train rows ...")
        t0 = time.time()
        for i, r in tr.iterrows():
            letter, raw, it, ot = _judge_one(client, args.model, r['text'])
            in_tok += it; out_tok += ot
            time.sleep(THROTTLE_SEC)
            row = {
                'source':          'train_sample',
                'orig_idx':        int(r['orig_idx']),
                'text':            r['text'],
                'gold_label':      '',
                'llama_label':     r['llama_label'],
                'category':        '',
                'judge_letter':   letter or '?',
                'judge_label':    LABEL_FROM_LETTER.get(letter, 'unknown'),
                'judge_raw':      raw,
            }
            _emit(writer, fh, row)
            rows_for_summary.append(row)
            if (i + 1) % 25 == 0 or args.smoke:
                rate = (time.time() - t0) / (i + 1)
                print(f"[probe]   train {i+1}/{len(tr)}  "
                      f"({rate:.2f}s/row)  letter={letter or '?'}")
        print(f"[probe] train took {(time.time()-t0)/60:.1f} min")
    finally:
        fh.close()
        kwh = tracker.stop()
        print(f"[probe] codecarbon kWh (local only) = {kwh}")
        print(f"[probe] wrote {_display_path(out_csv)}")

    # NVIDIA NIM free tier — no per-token cost.
    est_cost = 0.0
    rate_label = 'NIM free tier'
    print(f"[probe] token usage: {in_tok:,} in, {out_tok:,} out  "
          f"(free, {rate_label})")

    # ======== AGREEMENT SUMMARY ========
    df = pd.DataFrame(rows_for_summary)
    lines = []
    lines.append(f"PROBE SUMMARY ({args.model})\n" + "=" * 60)
    lines.append(f"Token usage: {in_tok:,} in / {out_tok:,} out  ~${est_cost:.2f}")

    adv_df = df[df['source'] == 'adv_concept']
    if len(adv_df) > 0:
        valid = adv_df[adv_df['judge_label'] != 'unknown']
        if len(valid) == 0:
            lines.append("\n(1) Judge vs AdvConcept-50 gold: NO PARSEABLE LABELS")
            adv_agree = 0.0
        else:
            adv_agree = float((valid['judge_label'] == valid['gold_label']).mean())
            n_correct = int(adv_agree * len(valid))
            lines.append(f"\n(1) Judge vs AdvConcept-50 curriculum gold")
            lines.append(f"    agreement: {adv_agree:.3f}  ({n_correct}/{len(valid)} "
                          f"parseable; {len(adv_df) - len(valid)} unparseable)")
            for cat in valid['category'].unique():
                sub = valid[valid['category'] == cat]
                ca = (sub['judge_label'] == sub['gold_label']).mean()
                lines.append(f"      {cat:30s}: {ca:.3f}  ({int(ca*len(sub))}/{len(sub)})")
    else:
        adv_agree = None

    tr_df = df[df['source'] == 'train_sample']
    if len(tr_df) > 0:
        valid = tr_df[tr_df['judge_label'] != 'unknown']
        if len(valid) == 0:
            lines.append("\n(2) Judge vs Llama-3.1-8B: NO PARSEABLE LABELS")
            llama_disagree = 0.0
        else:
            llama_agree = float((valid['judge_label'] == valid['llama_label']).mean())
            llama_disagree = 1.0 - llama_agree
            lines.append(f"\n(2) Judge vs Llama-3.1-8B on {len(valid)} train rows "
                          f"({len(tr_df) - len(valid)} unparseable)")
            lines.append(f"    agreement:    {llama_agree:.3f}  ({int(llama_agree*len(valid))}/{len(valid)})")
            lines.append(f"    disagreement: {llama_disagree:.3f}  ({int(llama_disagree*len(valid))}/{len(valid)})")
            labels = ['elementary', 'middle', 'high']
            lines.append(f"\n    Confusion (rows = Llama label, cols = Judge label):")
            lines.append(f"                Judge:         E      M      H")
            for ll in labels:
                row = valid[valid['llama_label'] == ll]
                cnt = {cl: int((row['judge_label'] == cl).sum()) for cl in labels}
                lines.append(f"      Llama={ll:11s} {cnt['elementary']:5d}  "
                              f"{cnt['middle']:5d}  {cnt['high']:5d}")
    else:
        llama_disagree = None

    # ======== DECISION ========
    lines.append("\n" + "=" * 60)
    lines.append("DECISION")
    lines.append("=" * 60)
    judge_ok = (adv_agree is not None) and (adv_agree >= 0.85)
    relabel_motivated = (llama_disagree is not None) and (llama_disagree > 0.15)
    if adv_agree is not None:
        lines.append(f"  Judge trustworthy (adv-agreement >= 0.85)?      "
                      f"{adv_agree:.3f}  ->  {'YES' if judge_ok else 'NO'}")
    if llama_disagree is not None:
        lines.append(f"  Re-label motivated (llama-disagreement > 0.15)? "
                      f"{llama_disagree:.3f}  ->  "
                      f"{'YES' if relabel_motivated else 'NO'}")
    if judge_ok and relabel_motivated:
        lines.append("\n  ==> PROCEED to full re-label of training data.")
    elif judge_ok and not relabel_motivated:
        lines.append("\n  ==> STOP. Judge is trusted, but agrees with Llama on most "
                      "training rows. The label-bias concern is overblown; the "
                      "current Pillar B+ labels are fine.")
    elif not judge_ok and adv_agree is not None:
        lines.append("\n  ==> STOP. The judge itself doesn't agree with the curriculum "
                      "gold reliably enough; this judge isn't trustworthy for the "
                      "task. Try a different judge or different prompt.")
    else:
        lines.append("\n  ==> INDETERMINATE. Not enough parseable labels to decide.")

    summary = "\n".join(lines)
    print("\n" + summary)
    with open(out_txt, 'w') as fh2:
        fh2.write(summary + "\n")
    print(f"\n[probe] wrote {_display_path(out_txt)}")


if __name__ == '__main__':
    main()
