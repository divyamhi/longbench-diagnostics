import sys, os, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompts.parse_response import extract_choice
from prompts.direct_answer import build_da_prompt, _DA_TEMPLATE
from prompts.e2a_stage1 import build_extraction_prompt, NO_EVIDENCE_SENTINEL
from prompts.e2a_stage2 import build_answer_prompt


def make_instance(context="The sky is blue due to Rayleigh scattering.",
                  question="Why is the sky blue?",
                  choices=None):
    if choices is None:
        choices = ["Dense cloud cover", "Atmospheric scattering", "Industrial smog", "Optical illusion"]
    return {
        "id": "test-001",
        "input": context,
        "question": question,
        "choices": choices,
        "answer": "B",
        "category": "single-document QA",
    }


class TestExtractChoice(unittest.TestCase):

    def test_simple_letter(self):
        self.assertEqual(extract_choice("B"), "B")

    def test_letter_in_sentence(self):
        self.assertEqual(extract_choice("The answer is B."), "B")

    def test_letter_in_parentheses(self):
        self.assertEqual(extract_choice("(C)"), "C")

    def test_letter_with_markdown_bold(self):
        self.assertEqual(extract_choice("**A**"), "A")

    def test_repeated_same_letter(self):
        self.assertEqual(extract_choice("I think the answer is definitely A. Yes, A."), "A")

    def test_lowercase_input(self):
        self.assertEqual(extract_choice("the answer is b"), "B")

    def test_answer_prefix(self):
        self.assertEqual(extract_choice("Answer: D"), "D")

    def test_two_distinct_letters(self):
        self.assertIsNone(extract_choice("A or B"))

    def test_three_distinct_letters(self):
        self.assertIsNone(extract_choice("Could be A, B, or C"))

    def test_empty_string(self):
        self.assertIsNone(extract_choice(""))

    def test_whitespace_only(self):
        self.assertIsNone(extract_choice("   "))

    def test_no_letter(self):
        self.assertIsNone(extract_choice("I don't know the answer."))

    def test_none_input(self):
        self.assertIsNone(extract_choice(None))

    def test_integer_input(self):
        self.assertIsNone(extract_choice(42))

    def test_letter_in_word_because(self):
        self.assertIsNone(extract_choice("because of this"))

    def test_letter_in_word_canada(self):
        self.assertIsNone(extract_choice("canada"))

    def test_letter_standalone_next_to_punctuation(self):
        self.assertEqual(extract_choice("Answer: C!"), "C")


class TestBuildDaPrompt(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()
        self.prompt = build_da_prompt(self.instance)

    def test_contains_instruction_phrase(self):
        self.assertIn("Answer with a single letter", self.prompt)

    def test_contains_all_choice_labels(self):
        for label in ("A.", "B.", "C.", "D."):
            self.assertIn(label, self.prompt)

    def test_contains_all_choice_texts(self):
        for choice in self.instance["choices"]:
            self.assertIn(choice, self.prompt)

    def test_context_appears_verbatim(self):
        self.assertIn(self.instance["input"], self.prompt)

    def test_question_appears(self):
        self.assertIn(self.instance["question"], self.prompt)

    def test_returns_string(self):
        self.assertIsInstance(self.prompt, str)

    def test_context_before_question(self):
        self.assertLess(
            self.prompt.index(self.instance["input"]),
            self.prompt.index(self.instance["question"])
        )

    def test_choices_in_order(self):
        positions = [self.prompt.index(c) for c in self.instance["choices"]]
        self.assertEqual(positions, sorted(positions))

    def test_missing_input_key_raises(self):
        instance = make_instance()
        del instance["input"]
        with self.assertRaises(KeyError):
            build_da_prompt(instance)

    def test_missing_question_key_raises(self):
        instance = make_instance()
        del instance["question"]
        with self.assertRaises(KeyError):
            build_da_prompt(instance)

    def test_too_few_choices_raises(self):
        instance = make_instance(choices=["Only one choice"])
        with self.assertRaises(IndexError):
            build_da_prompt(instance)

    def test_same_instance_same_output(self):
        self.assertEqual(build_da_prompt(self.instance), build_da_prompt(self.instance))


class TestBuildExtractionPrompt(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()
        self.prompt = build_extraction_prompt(self.instance)

    def test_contains_verbatim_instruction(self):
        self.assertIn("Copy them verbatim", self.prompt)

    def test_contains_sentinel_instruction(self):
        self.assertIn(NO_EVIDENCE_SENTINEL, self.prompt)

    def test_context_appears(self):
        self.assertIn(self.instance["input"], self.prompt)

    def test_question_appears(self):
        self.assertIn(self.instance["question"], self.prompt)

    def test_choices_excluded(self):
        for choice in self.instance["choices"]:
            self.assertNotIn(choice, self.prompt)

    def test_returns_string(self):
        self.assertIsInstance(self.prompt, str)

    def test_missing_input_raises(self):
        instance = make_instance()
        del instance["input"]
        with self.assertRaises(KeyError):
            build_extraction_prompt(instance)

    def test_missing_question_raises(self):
        instance = make_instance()
        del instance["question"]
        with self.assertRaises(KeyError):
            build_extraction_prompt(instance)

    def test_sentinel_is_string(self):
        self.assertIsInstance(NO_EVIDENCE_SENTINEL, str)

    def test_sentinel_is_nonempty(self):
        self.assertGreater(len(NO_EVIDENCE_SENTINEL), 0)


class TestBuildAnswerPrompt(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()
        self.evidence = "The sky is blue due to Rayleigh scattering."
        self.prompt = build_answer_prompt(self.instance, self.evidence)

    def test_contains_instruction_phrase(self):
        self.assertIn("Answer with a single letter", self.prompt)

    def test_contains_evidence_header(self):
        self.assertIn("Extracted evidence", self.prompt)

    def test_contains_evidence_text(self):
        self.assertIn(self.evidence, self.prompt)

    def test_contains_question(self):
        self.assertIn(self.instance["question"], self.prompt)

    def test_contains_all_choices(self):
        for choice in self.instance["choices"]:
            self.assertIn(choice, self.prompt)

    def test_contains_all_choice_labels(self):
        for label in ("A.", "B.", "C.", "D."):
            self.assertIn(label, self.prompt)

    def test_original_context_excluded(self):
        if self.instance["input"] != self.evidence:
            self.assertNotIn(self.instance["input"], self.prompt)

    def test_evidence_before_question(self):
        self.assertLess(
            self.prompt.index(self.evidence),
            self.prompt.index(self.instance["question"])
        )

    def test_sentinel_evidence_works(self):
        result = build_answer_prompt(self.instance, NO_EVIDENCE_SENTINEL)
        self.assertIn(NO_EVIDENCE_SENTINEL, result)
        self.assertIn("Answer with a single letter", result)

    def test_missing_question_raises(self):
        instance = make_instance()
        del instance["question"]
        with self.assertRaises(KeyError):
            build_answer_prompt(instance, self.evidence)

    def test_too_few_choices_raises(self):
        instance = make_instance(choices=["Only", "Two"])
        with self.assertRaises(IndexError):
            build_answer_prompt(instance, self.evidence)

    def test_non_string_evidence_raises(self):
        with self.assertRaises(TypeError):
            build_answer_prompt(self.instance, 42)

    def test_none_evidence_raises(self):
        with self.assertRaises(TypeError):
            build_answer_prompt(self.instance, None)

    def test_same_inputs_same_output(self):
        self.assertEqual(
            build_answer_prompt(self.instance, self.evidence),
            build_answer_prompt(self.instance, self.evidence)
        )


class TestPromptComparison(unittest.TestCase):

    def setUp(self):
        self.instance = make_instance()
        self.evidence = "Sky appears blue due to Rayleigh scattering of sunlight."
        self.da_prompt = build_da_prompt(self.instance)
        self.s2_prompt = build_answer_prompt(self.instance, self.evidence)

    def test_both_end_with_answer_instruction(self):
        ending = "Answer with a single letter: A, B, C, or D."
        self.assertTrue(self.da_prompt.endswith(ending))
        self.assertTrue(self.s2_prompt.endswith(ending))

    def test_da_contains_full_context(self):
        self.assertIn(self.instance["input"], self.da_prompt)

    def test_s2_does_not_contain_full_context(self):
        self.assertNotIn("instance['input']", self.s2_prompt)

    def test_s2_contains_evidence_not_in_da(self):
        self.assertIn(self.evidence, self.s2_prompt)
        if self.evidence != self.instance["input"]:
            self.assertNotIn(self.evidence, self.da_prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
