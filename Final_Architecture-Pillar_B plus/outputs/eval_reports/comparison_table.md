# gen_v2 Comparison Table

## Headline F1 scores

| Model | val | test | bal-test | synth-judge | synth-gen | race-mid | race-high | onestop | adv | adv-srf-hard-easy | params | latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_qwen2.5-7b | — | 0.840 | 0.856 | 0.918 | 0.940 | 0.595 | 0.479 | 0.415 | 0.600 | 0.000 | 58,500 | — |
| pillar_a__all-MiniLM-L6-v2__emb_only | — | 0.782 | 0.767 | 0.770 | 0.743 | 0.435 | 0.458 | 0.395 | 0.440 | 0.500 | 384 | 2 |
| pillar_a__all-MiniLM-L6-v2__emb_plus_static5 | — | 0.808 | 0.787 | 0.703 | 0.677 | 0.475 | 0.474 | 0.424 | 0.360 | 0.000 | 389 | 3 |
| pillar_a__bge-base-en-v1.5__emb_only | — | 0.804 | 0.800 | 0.729 | 0.749 | 0.468 | 0.487 | 0.455 | 0.460 | 0.667 | 768 | 4 |
| pillar_a__bge-base-en-v1.5__emb_plus_static5 | — | 0.804 | 0.801 | 0.800 | 0.783 | 0.512 | 0.495 | 0.458 | 0.400 | 0.333 | 773 | 4 |
| pillar_a__bge-small-en-v1.5__emb_only | — | 0.799 | 0.790 | 0.785 | 0.805 | 0.502 | 0.480 | 0.421 | 0.500 | 0.500 | 384 | 3 |
| pillar_a__bge-small-en-v1.5__emb_plus_static5 | — | 0.806 | 0.811 | 0.801 | 0.821 | 0.560 | 0.499 | 0.483 | 0.520 | 0.167 | 389 | 4 |
| pillar_b__gemma-2-2b-it | — | 0.856 | 0.827 | 0.856 | 0.826 | 0.492 | 0.521 | 0.493 | 0.620 | 0.500 | 2.6B | 28 |
| pillar_b__phi-3.5-mini-instruct | — | 0.857 | 0.854 | 0.863 | 0.885 | 0.673 | 0.529 | 0.436 | 0.680 | 0.667 | 3.8B | 34 |
| pillar_b__qwen-2.5-3b-instruct | — | 0.857 | 0.834 | 0.860 | 0.879 | 0.524 | 0.537 | 0.518 | 0.660 | 0.000 | 3.1B | 33 |
| pillar_b_plus__phi-3.5-mini-instruct | — | 0.872 | 0.869 | 0.925 | 0.893 | 0.561 | 0.546 | 0.460 | 0.760 | 1.000 | 3.9B | 39 |

## Energy consumption (CodeCarbon)

| Project | kWh |
|---|---|
| build_adv_concept | 0.00002 |
| evaluate_all_baseline | 0.00318 |
| pillar_a_lean_combo | 0.02828 |
| pillar_b_plus | 0.38679 |
| pillar_b_tiny_teacher | 0.26857 |
| step9g_eval_synthetic | 0.00002 |
