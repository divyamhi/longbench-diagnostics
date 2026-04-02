"""
e2a_pipeline.py
---------------
Orchestrates the two-stage Extract-Then-Answer (E2A) prompting pipeline.

Stage 1: build_extraction_prompt() → model.predict() → raw evidence string
Stage 2: build_answer_prompt()     → model.predict() → final A/B/C/D answer

The Stage 1 model output is used RAW as the evidence string — it is NOT
parsed through extract_choice(). Only the Stage 2 output is parsed into
a final answer letter.

Token counts and latency are summed across both calls so the result
record is directly comparable to a single Direct Answer call.

Design rules:
  - run_e2a() is the only public function.
  - Never raises; propagates None prediction if either stage fails.
  - Follows the shared result schema exactly — no extra or missing keys.
"""

try:
    from ..models.base import LLMBackend
except ImportError:
    LLMBackend = object  # fallback stub until base.py is available

from .e2a_stage1 import build_extraction_prompt, NO_EVIDENCE_SENTINEL
from .e2a_stage2 import build_answer_prompt


def run_e2a(instance: dict, model) -> dict:
    try:
        stage1_prompt = build_extraction_prompt(instance)
        stage1_result = model.predict(stage1_prompt)

        extracted_evidence = (
            stage1_result.get("raw_output")
            or stage1_result["prediction"]
            or NO_EVIDENCE_SENTINEL
        )

        stage2_prompt = build_answer_prompt(instance, extracted_evidence)
        stage2_result = model.predict(stage2_prompt)

        prediction = stage2_result["prediction"]
        ground_truth = instance["answer"]
        correct = (prediction == ground_truth)

        input_tokens = stage1_result["input_tokens"] + stage2_result["input_tokens"]
        output_tokens = stage1_result["output_tokens"] + stage2_result["output_tokens"]
        latency_ms = stage1_result["latency_ms"] + stage2_result["latency_ms"]

        return {
            "instance_id": instance["id"],
            "extracted_evidence": extracted_evidence,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "correct": correct,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
        }
    except Exception:
        return {
            "instance_id": instance.get("id", "unknown"),
            "extracted_evidence": "",
            "prediction": None,
            "ground_truth": instance.get("answer", ""),
            "correct": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0.0,
        }
