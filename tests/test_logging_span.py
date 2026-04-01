import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.logging_utils import InferenceLogger
from src.span_detect import find_relevant_span


def test_logging():
    test_file = "test_output.jsonl"

    if os.path.exists(test_file):
        os.remove(test_file)

    record = {
        "instance_id": "1",
        "category": "test",
        "budget": 8192,
        "prompt_strategy": "DA",
        "model_id": "test-model",
        "prediction": "A",
        "ground_truth": "B",
        "correct": False,
        "input_tokens": 100,
        "output_tokens": 5,
        "latency_ms": 10.5,
    }

    with InferenceLogger(test_file) as logger:
        logger.log(record)

    with open(test_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    assert len(lines) == 1

    loaded = json.loads(lines[0])
    assert loaded == record

    os.remove(test_file)


def test_missing_key():
    try:
        with InferenceLogger("temp.jsonl") as logger:
            logger.log({"instance_id": "1"})
    except KeyError:
        return

    assert False, "Expected KeyError"


def test_span_detection():
    instance = {
        "input": "Cats are animals. Dogs are mammals. Birds can fly high in the sky.",
        "question": "Which animals can fly?",
        "category": "single-document QA",
    }

    result = find_relevant_span(instance)

    assert "span_text" in result
    assert "sentence_index" in result
    assert "score" in result
    assert "flagged" in result
    assert "fly" in result["span_text"].lower()
    assert result["flagged"] is False


def test_invalid_category():
    instance = {
        "input": "Some text.",
        "question": "Some question?",
        "category": "multi-document QA",
    }

    try:
        find_relevant_span(instance)
    except ValueError:
        return

    assert False, "Expected ValueError"