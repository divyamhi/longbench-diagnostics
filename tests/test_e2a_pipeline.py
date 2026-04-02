import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompts.e2a_pipeline import run_e2a
from prompts.e2a_stage1 import NO_EVIDENCE_SENTINEL


class MockBackend:
    def __init__(self, stage1_raw=None, stage1_prediction="A", stage2_prediction="B",
                 input_tokens=100, output_tokens=5, latency_ms=200.0):
        self.stage1_raw = stage1_raw
        self.stage1_prediction = stage1_prediction
        self.stage2_prediction = stage2_prediction
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.calls = []

    def predict(self, prompt: str) -> dict:
        self.calls.append(prompt)
        if len(self.calls) == 1:
            result = {
                "prediction": self.stage1_prediction,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "latency_ms": self.latency_ms,
            }
            if self.stage1_raw is not None:
                result["raw_output"] = self.stage1_raw
            return result
        else:
            return {
                "prediction": self.stage2_prediction,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "latency_ms": self.latency_ms,
            }


def make_instance():
    return {
        "id": "test-001",
        "input": "The Eiffel Tower was built for the World's Fair and opened to the public that spring.",
        "question": "When was the Eiffel Tower completed?",
        "choices": ["1776", "1889", "1901", "1945"],
        "answer": "B",
        "category": "single-document QA",
    }


class TestReturnSchema(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()
        self.model = MockBackend()
        self.result = run_e2a(self.instance, self.model)

    def test_return_keys(self):
        expected = {"instance_id", "extracted_evidence", "prediction", "ground_truth",
                    "correct", "input_tokens", "output_tokens", "latency_ms"}
        self.assertEqual(set(self.result.keys()), expected)

    def test_instance_id(self):
        self.assertEqual(self.result["instance_id"], "test-001")

    def test_ground_truth(self):
        self.assertEqual(self.result["ground_truth"], "B")


class TestTwoModelCalls(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()
        self.model = MockBackend()
        self.result = run_e2a(self.instance, self.model)

    def test_two_model_calls(self):
        self.assertEqual(len(self.model.calls), 2)

    def test_stage1_prompt_has_no_choices(self):
        stage1_prompt = self.model.calls[0]
        for choice in self.instance["choices"]:
            self.assertNotIn(choice, stage1_prompt)

    def test_stage2_prompt_has_choices(self):
        stage2_prompt = self.model.calls[1]
        for choice in self.instance["choices"]:
            self.assertIn(choice, stage2_prompt)


class TestTokenAndLatencyAggregation(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()

    def test_input_tokens_summed(self):
        model = MockBackend(input_tokens=100)
        result = run_e2a(self.instance, model)
        self.assertEqual(result["input_tokens"], 200)

    def test_output_tokens_summed(self):
        model = MockBackend(output_tokens=5)
        result = run_e2a(self.instance, model)
        self.assertEqual(result["output_tokens"], 10)

    def test_latency_summed(self):
        model = MockBackend(latency_ms=200.0)
        result = run_e2a(self.instance, model)
        self.assertEqual(result["latency_ms"], 400.0)


class TestCorrectness(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()

    def test_correct_when_prediction_matches(self):
        model = MockBackend(stage2_prediction="B")
        result = run_e2a(self.instance, model)
        self.assertTrue(result["correct"])

    def test_incorrect_when_prediction_wrong(self):
        model = MockBackend(stage2_prediction="A")
        result = run_e2a(self.instance, model)
        self.assertFalse(result["correct"])

    def test_none_prediction_is_incorrect(self):
        model = MockBackend(stage2_prediction=None)
        result = run_e2a(self.instance, model)
        self.assertFalse(result["correct"])


class TestEvidenceHandling(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()

    def test_raw_output_used_as_evidence(self):
        model = MockBackend(stage1_raw="verbatim passage text")
        result = run_e2a(self.instance, model)
        self.assertEqual(result["extracted_evidence"], "verbatim passage text")
        self.assertIn("verbatim passage text", model.calls[1])

    def test_sentinel_used_when_no_raw_output(self):
        model = MockBackend(stage1_raw=None, stage1_prediction=None)
        result = run_e2a(self.instance, model)
        self.assertEqual(result["extracted_evidence"], NO_EVIDENCE_SENTINEL)


class TestErrorSafety(unittest.TestCase):

    def test_error_safety(self):
        class BrokenBackend:
            def predict(self, prompt):
                raise RuntimeError("backend failure")

        instance = make_instance()
        result = run_e2a(instance, BrokenBackend())
        self.assertIsNone(result["prediction"])
        self.assertFalse(result["correct"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
