# Shlok — Evaluation Harness & Metrics

## What This Module Does
This module provides three things:
1. **Abstract model interface** (`base.py`) — defines the contract every model backend must follow
2. **Local HuggingFace backend** (`local_hf.py`) — runs Llama-3.1-8B-Instruct locally
3. **Metrics** (`metrics.py`) — all accuracy computation and confidence intervals for the project

---

## Files

### `src/models/base.py`
Abstract base class `LLMBackend`. Every model backend in this project must subclass this.

**What it defines:**
- `predict(prompt: str) -> dict` — must be implemented by every backend
- Return format is always:
```python
{
    "prediction":    "A" | "B" | "C" | "D" | None,
    "input_tokens":  int,
    "output_tokens": int,
    "latency_ms":    float
}
```
- `prediction` is `None` if the model output cannot be parsed
- **Never raises** under any circumstance

---

### `src/models/local_hf.py`
Implements `LLMBackend` for **Llama-3.1-8B-Instruct** via HuggingFace pipeline.

**How it works:**
- `__init__(model_id, device="cuda")` loads the model and tokenizer, sets `temperature=0` for deterministic decoding
- `predict(prompt)`:
  1. Tokenizes the prompt directly to count `input_tokens`
  2. Starts timer with `time.perf_counter()`
  3. Runs the HuggingFace pipeline
  4. Stops timer, computes `latency_ms`
  5. Counts `output_tokens` as (total tokens - input tokens)
  6. Calls `extract_choice()` from Meghna's `parse_response.py` to get A/B/C/D
  7. Returns the standard dict — never raises

> **Note:** Actual inference requires a GPU. Tests are fully mocked and run on any laptop.

---

### `src/metrics.py`
**Authoritative metrics module. No other module computes accuracy independently.**

**Functions:**

`accuracy(results) -> float`
- Input: list of result dicts with a `correct` boolean field
- Returns: fraction of correct predictions
- Raises `ValueError` if list is empty

`per_category_accuracy(results) -> dict[str, float]`
- Groups results by `category` field
- Returns accuracy per category: `{"single-doc": 0.6, "multi-doc": 0.4, ...}`

`bootstrap_ci(results, n_resamples=10000, ci=0.95) -> tuple[float, float]`
- Estimates confidence interval for accuracy via bootstrap resampling
- Fully vectorised with NumPy — no Python loop over resamples
- Returns `(lower, upper)` bounds, e.g. `(0.48, 0.62)` for a 95% CI

---

## Running Tests
Tests are fully mocked — no GPU or model weights needed.

```bash
export PYTHONPATH=.
pytest tests/test_shlok.py -v
```

Expected output: **19 passed**

---

## Dependencies
```bash
pip install pytest numpy transformers scikit-learn
```

---

## Integration Contract
Every result record this module consumes must follow this exact schema:

```json
{
    "instance_id":     "string",
    "category":        "string",
    "budget":          8192,
    "prompt_strategy": "DA",
    "model_id":        "meta-llama/Llama-3.1-8B-Instruct",
    "prediction":      "B",
    "ground_truth":    "A",
    "correct":         false,
    "input_tokens":    7841,
    "output_tokens":   3,
    "latency_ms":      4210.3
}
```

## Who Depends on This Module
| Person  | Needs                                        |
|---------|----------------------------------------------|
| Meghna  | `base.py` to write `e2a_pipeline.py`         |
| Meghna  | `local_hf.py` to run the 32K pilot           |
| Divyam  | `metrics.py` for plots                       |
| Rudrani | `metrics.py` to sample incorrect predictions |
