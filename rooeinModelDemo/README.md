---
title: Rooein Model Combo Reading Level Classifier
emoji: 📚
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.0
python_version: "3.10"
app_file: app.py
pinned: false
---

# Rooein Model Combo Reading Level Classifier

Gradio demo for a Rooein-style logistic regression reading level classifier.

Input raw text and the app runs the Rooein model combo using static features
plus 63 API-generated yes/no prompt features.

The output returns probability scores for:

- elementary
- middle
- high

This Space can run on CPU because Qwen is called through the Together API. Add
`TOGETHER_API_KEY` as a Hugging Face Space secret before using the combo model.

By default, the app splits the 63 prompt questions into 3 Together API calls
to avoid large-request timeouts while keeping latency reasonable. To force one
large JSON request, set `TOGETHER_PROMPT_MODE=single`. To run one API call per
Rooein prompt instead, set `TOGETHER_PROMPT_MODE=exact`.

Required local files:

- `outputs/prompt_metrics/prompt_questions.json`
- `results/by_prompt_model/qwen2.5-7b/artifacts/feature_schema_qwen2.5-7b.json`
- `results/by_prompt_model/qwen2.5-7b/artifacts/selector_static_qwen2.5-7b.joblib`
- `results/by_prompt_model/qwen2.5-7b/artifacts/scaler_static_qwen2.5-7b.joblib`
- `results/by_prompt_model/qwen2.5-7b/artifacts/model_static_qwen2.5-7b.joblib`
- `results/by_prompt_model/qwen2.5-7b/artifacts/selector_combo_qwen2.5-7b.joblib`
- `results/by_prompt_model/qwen2.5-7b/artifacts/scaler_combo_qwen2.5-7b.joblib`
- `results/by_prompt_model/qwen2.5-7b/artifacts/model_combo_qwen2.5-7b.joblib`
