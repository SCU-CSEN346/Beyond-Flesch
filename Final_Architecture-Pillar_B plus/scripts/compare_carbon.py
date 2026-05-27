"""
Build the comparison table used in the paper and slides.

This pulls the saved evaluation reports for the Rooein-style baseline, Pillar A,
Pillar B, and Pillar B+. It also joins in the CodeCarbon measurements where we
have them.

The JSON output keeps report paths relative to the repository so it can be
shared without exposing local machine paths.
"""

import csv
import glob
import json
import os
import sys

import pandas as pd

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
_REPO_ROOT   = os.path.dirname(_PROJECT_DIR)
GF_DIR       = os.path.join(_REPO_ROOT, 'generalization_final')
REPORT_DIR   = os.path.join(_PROJECT_DIR, 'outputs', 'eval_reports')


def _display_path(path):
    return os.path.relpath(path, _REPO_ROOT)


def _read_emissions(emissions_csv):
    """Read total kWh per CodeCarbon project name."""
    out = {}
    if not os.path.exists(emissions_csv):
        return out
    with open(emissions_csv) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            project = row.get('project_name', 'unknown')
            kwh = float(row.get('energy_consumed', 0) or 0)
            out[project] = out.get(project, 0.0) + kwh
    return out


def _load_eval_json(path, kind, name):
    """Load one eval report and keep only the numbers used downstream."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        d = json.load(fh)
    extract = {
        'kind': kind,
        'name': name,
        'json_path': os.path.relpath(path, _REPO_ROOT),
    }
    if kind == 'baseline':
        extract['n_params'] = d.get('model_meta', {}).get('n_params')
        per = d.get('per_split', {})
        adv = d.get('adv_concept', {})
    elif kind == 'pillar_a':
        per = d.get('per_split', {})
        adv = d.get('adv_concept', {})
        extract['n_params'] = d.get('feature_dim', 0)  # encoder + 1.2K classifier
        extract['n_params_encoder_m'] = d.get('encoder_size_m')
        extract['latency_ms'] = d.get('latency', {}).get('median_ms')
    elif kind in ('pillar_b', 'pillar_b_plus'):
        per = d.get('per_split', {})
        adv = d.get('adv_concept', {})
        extract['n_params'] = d.get('n_params_total')
        extract['n_params_trainable_lora'] = d.get('n_params_trainable_lora')
        extract['latency_ms'] = d.get('latency', {}).get('median_ms')
        extract['train_sec'] = d.get('train_sec')

    extract['test_f1']           = per.get('test', {}).get('f1_macro')
    extract['balanced_test_f1']  = per.get('balanced_test', {}).get('f1_macro_mean')
    extract['synth_judge_f1']    = per.get('ood_synthetic', {}).get('f1_macro')
    extract['synth_gen_f1']      = per.get('ood_synthetic_vs_gen', {}).get('f1_macro')
    extract['race_mid_f1']       = per.get('ood_race-middle', {}).get('f1_macro')
    extract['race_high_f1']      = per.get('ood_race-high', {}).get('f1_macro')
    extract['onestop_f1']        = per.get('ood_onestop', {}).get('f1_macro')
    extract['adv_overall']       = adv.get('overall_accuracy')
    extract['adv_per_category']  = adv.get('per_category_accuracy', {})
    return extract


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    rows = []

    # Rooein-style baseline.
    baseline_json = os.path.join(REPORT_DIR, 'baseline_per_llm_qwen2.5-7b_seed7.eval.json')
    b = _load_eval_json(baseline_json, 'baseline', 'baseline_qwen2.5-7b')
    if b:
        rows.append(b)

    # Encoder-based lean models.
    for f in sorted(glob.glob(os.path.join(REPORT_DIR, 'pillar_a__*.eval.json'))):
        name = os.path.basename(f).replace('.eval.json', '')
        r = _load_eval_json(f, 'pillar_a', name)
        if r:
            rows.append(r)

    # Small decoder-LLM bake-off.
    for f in sorted(glob.glob(os.path.join(REPORT_DIR, 'pillar_b__*.eval.json'))):
        name = os.path.basename(f).replace('.eval.json', '')
        r = _load_eval_json(f, 'pillar_b', name)
        if r:
            rows.append(r)

    # Final model.
    for f in sorted(glob.glob(os.path.join(REPORT_DIR, 'pillar_b_plus__*.eval.json'))):
        name = os.path.basename(f).replace('.eval.json', '')
        r = _load_eval_json(f, 'pillar_b_plus', name)
        if r:
            rows.append(r)

    # Carbon logs.
    emissions = {}
    for tag, csv_path in [
        ('generalization_final', os.path.join(GF_DIR, 'outputs', 'emissions', 'emissions.csv')),
        ('gen_v2',                os.path.join(_PROJECT_DIR, 'outputs', 'emissions', 'emissions.csv')),
    ]:
        for proj, kwh in _read_emissions(csv_path).items():
            emissions[proj] = kwh

    # Attach the relevant kWh number to each row where we have one.
    for r in rows:
        if r['kind'] == 'baseline':
            r['energy_kwh'] = None  # baseline training emissions weren't tracked end-to-end
        elif r['kind'] == 'pillar_a':
            r['energy_kwh'] = emissions.get('pillar_a_lean_combo', None)
        elif r['kind'] == 'pillar_b':
            r['energy_kwh'] = emissions.get('pillar_b_tiny_teacher', None)
        elif r['kind'] == 'pillar_b_plus':
            r['energy_kwh'] = emissions.get('pillar_b_plus', None)

    # Markdown table for quick reading.
    md = ['# gen_v2 Comparison Table', '']
    md.append('## Headline F1 scores')
    md.append('')
    md.append('| Model | val | test | bal-test | synth-judge | synth-gen | race-mid | race-high | onestop | adv | adv-srf-hard-easy | params | latency (ms) |')
    md.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        vals = []
        # Val is missing for some older reports.
        vals.append(_fmt(r.get('val_f1')))
        vals.append(_fmt(r['test_f1']))
        vals.append(_fmt(r['balanced_test_f1']))
        vals.append(_fmt(r['synth_judge_f1']))
        vals.append(_fmt(r['synth_gen_f1']))
        vals.append(_fmt(r['race_mid_f1']))
        vals.append(_fmt(r['race_high_f1']))
        vals.append(_fmt(r['onestop_f1']))
        vals.append(_fmt(r['adv_overall']))
        srf_hard_easy = r.get('adv_per_category', {}).get('surface_hard_concept_easy')
        vals.append(_fmt(srf_hard_easy))
        params = r.get('n_params')
        vals.append(_format_params(params))
        vals.append(_fmt_int(r.get('latency_ms')))
        md.append(f'| {r["name"]} | ' + ' | '.join(vals) + ' |')
    md.append('')

    md.append('## Energy consumption (CodeCarbon)')
    md.append('')
    md.append('| Project | kWh |')
    md.append('|---|---|')
    for proj, kwh in sorted(emissions.items()):
        md.append(f'| {proj} | {kwh:.5f} |')
    md.append('')

    md_path = os.path.join(REPORT_DIR, 'comparison_table.md')
    with open(md_path, 'w') as fh:
        fh.write('\n'.join(md))
    print(f"[compare_carbon] wrote {_display_path(md_path)}")

    # Machine-readable copy.
    json_path = os.path.join(REPORT_DIR, 'comparison_table.json')
    with open(json_path, 'w') as fh:
        json.dump({'models': rows, 'emissions_kwh': emissions}, fh, indent=2)
    print(f"[compare_carbon] wrote {_display_path(json_path)}")

    # Console summary.
    valid = [r for r in rows if r['adv_overall'] is not None]
    by_adv = max(valid, key=lambda r: r['adv_overall'])
    by_synth_gen = max(valid, key=lambda r: r['synth_gen_f1'] or 0)
    print()
    print(f"[compare_carbon] best on adv_concept:   {by_adv['name']} = {by_adv['adv_overall']:.4f}")
    print(f"[compare_carbon] best on synth_vs_gen:  {by_synth_gen['name']} = {by_synth_gen['synth_gen_f1']:.4f}")


def _fmt(v):
    if v is None:
        return '—'
    return f'{v:.3f}'


def _fmt_int(v):
    if v is None:
        return '—'
    return f'{v:.0f}'


def _format_params(n):
    if n is None:
        return '—'
    if n < 1e6:
        return f'{n:,}'
    if n < 1e9:
        return f'{n/1e6:.0f}M'
    return f'{n/1e9:.1f}B'


if __name__ == '__main__':
    main()
