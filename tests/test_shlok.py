"""
tests/test_shlok.py

Unit tests for:
  - src/models/base.py       (interface contract)
  - src/models/local_hf.py   (mocked — no GPU required)
  - src/metrics.py           (accuracy, per_category_accuracy, bootstrap_ci)

Run with:
    pytest tests/test_shlok.py -v
"""

import json
import time
import pytest
from unittest.mock import MagicMock, patch

# ---- metrics imports -------------------------------------------------------
from src.metrics import accuracy, per_category_accuracy, bootstrap_ci

# ---- base import -----------------------------------------------------------
from src.models.base import LLMBackend


# ===========================================================================
# Helpers
# ===========================================================================

def make_results(n_correct: int, n_total: int, category: str = "single-doc") -> list[dict]:
    """Build a minimal results list with n_correct correct entries."""
    results = []
    for i in range(n_total):
        results.append({
            "instance_id":     f"id_{i}",
            "category":        category,
            "budget":          8192,
            "prompt_strategy": "DA",
            "model_id":        "test-model",
            "prediction":      "A",
            "ground_truth":    "A" if i < n_correct else "B",
            "correct":         i < n_correct,
            "input_tokens":    100,
            "output_tokens":   1,
            "latency_ms":      200.0,
        })
    return results


# ===========================================================================
# Tests: metrics.py
# ===========================================================================

class TestAccuracy:
    def test_all_correct(self):
        results = make_results(100, 100)
        assert accuracy(results) == 1.0

    def test_none_correct(self):
        results = make_results(0, 100)
        assert accuracy(results) == 0.0

    def test_half_correct(self):
        results = make_results(50, 100)
        assert accuracy(results) == pytest.approx(0.5)

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            accuracy([])


class TestPerCategoryAccuracy:
    def test_single_category(self):
        results = make_results(40, 100, category="single-doc")
        cat_acc = per_category_accuracy(results)
        assert "single-doc" in cat_acc
        assert cat_acc["single-doc"] == pytest.approx(0.4)

    def test_multiple_categories(self):
        r1 = make_results(10, 10, category="single-doc")
        r2 = make_results(0,  10, category="multi-doc")
        cat_acc = per_category_accuracy(r1 + r2)
        assert cat_acc["single-doc"] == pytest.approx(1.0)
        assert cat_acc["multi-doc"]  == pytest.approx(0.0)


class TestBootstrapCI:
    def test_all_correct_ci(self):
        results = make_results(100, 100)
        lower, upper = bootstrap_ci(results, n_resamples=1000)
        assert lower == pytest.approx(1.0)
        assert upper == pytest.approx(1.0)

    def test_half_correct_ci_contains_point5(self):
        results = make_results(50, 100)
        lower, upper = bootstrap_ci(results, n_resamples=5000)
        assert lower <= 0.5 <= upper

    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])

    def test_returns_tuple_of_two_floats(self):
        results = make_results(30, 100)
        ci = bootstrap_ci(results, n_resamples=500)
        assert len(ci) == 2
        lower, upper = ci
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= upper


# ===========================================================================
# Tests: models/base.py
# ===========================================================================

class TestLLMBackendInterface:
    def test_cannot_instantiate_abstract_class(self):
        with pytest.raises(TypeError):
            LLMBackend()

    def test_concrete_subclass_must_implement_predict(self):
        """A concrete subclass that forgets predict() should still raise TypeError."""
        class Incomplete(LLMBackend):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_works(self):
        class DummyBackend(LLMBackend):
            def predict(self, prompt: str) -> dict:
                return {
                    "prediction":    "A",
                    "input_tokens":  5,
                    "output_tokens": 1,
                    "latency_ms":    10.0,
                }

        backend = DummyBackend()
        result = backend.predict("What is the answer?")
        assert result["prediction"] in {"A", "B", "C", "D", None}
        assert result["latency_ms"] > 0
        assert result["input_tokens"] > 0


# ===========================================================================
# Tests: models/local_hf.py  (mocked — no GPU / model weights required)
# ===========================================================================

class TestLocalHFBackend:
    """
    All HuggingFace calls are mocked so these tests run without a GPU
    or downloaded model weights.
    """

    def _make_backend(self, generated_text: str = "The answer is B."):
        """Build a LocalHFBackend with mocked tokenizer and pipeline."""
        from src.models.local_hf import LocalHFBackend

        with patch("src.models.local_hf.AutoTokenizer") as mock_tok_cls, \
             patch("src.models.local_hf.pipeline") as mock_pipe_cls:

            # Tokenizer mock: returns a fixed number of token IDs
            mock_tokenizer = MagicMock()
            mock_tokenizer.return_value = {"input_ids": [[1] * 50]}
            mock_tokenizer.side_effect = lambda text, **kw: {
                "input_ids": [[1] * max(1, len(text.split()))]
            }
            mock_tok_cls.from_pretrained.return_value = mock_tokenizer

            # Pipeline mock: returns a fixed generated text
            mock_pipeline_instance = MagicMock()
            mock_pipeline_instance.return_value = [{"generated_text": generated_text}]
            mock_pipe_cls.return_value = mock_pipeline_instance

            backend = LocalHFBackend.__new__(LocalHFBackend)
            backend.model_id  = "meta-llama/Llama-3.1-8B-Instruct"
            backend.tokenizer = mock_tokenizer
            backend.pipe      = mock_pipeline_instance

        return backend

    def test_prediction_is_valid_choice_or_none(self):
        backend = self._make_backend("The answer is B.")
        result  = backend.predict("Some prompt")
        assert result["prediction"] in {"A", "B", "C", "D", None}

    def test_latency_ms_positive(self):
        backend = self._make_backend("C")
        result  = backend.predict("Some prompt")
        assert result["latency_ms"] > 0

    def test_input_tokens_positive(self):
        backend = self._make_backend("A")
        result  = backend.predict("word1 word2 word3")
        assert result["input_tokens"] > 0

    def test_unparseable_output_returns_none(self):
        backend = self._make_backend("I have no idea.")
        result  = backend.predict("Some prompt")
        assert result["prediction"] is None

    def test_return_dict_has_required_keys(self):
        backend  = self._make_backend("B")
        result   = backend.predict("Some prompt")
        required = {"prediction", "input_tokens", "output_tokens", "latency_ms"}
        assert required.issubset(result.keys())

    def test_does_not_raise_on_model_error(self):
        """Even if the pipeline blows up, predict() must not raise."""
        from src.models.local_hf import LocalHFBackend

        backend = LocalHFBackend.__new__(LocalHFBackend)
        backend.model_id  = "test"
        backend.tokenizer = MagicMock(side_effect=lambda t, **k: {"input_ids": [[1, 2, 3]]})

        broken_pipe = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        backend.pipe = broken_pipe

        # Should NOT raise
        result = backend.predict("Some prompt")
        assert result["prediction"] is None
