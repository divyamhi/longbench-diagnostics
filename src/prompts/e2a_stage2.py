_STAGE2_TEMPLATE = (
    "Using only the extracted evidence below, answer the multiple choice "
    "question.\n\n"
    "Extracted evidence:\n{extracted_evidence}\n\n"
    "Question: {question}\n\n"
    "A. {choice_a}\n"
    "B. {choice_b}\n"
    "C. {choice_c}\n"
    "D. {choice_d}\n\n"
    "Answer with a single letter: A, B, C, or D."
)


def build_answer_prompt(instance: dict, extracted_evidence: str) -> str:
    if not isinstance(extracted_evidence, str):
        raise TypeError(f"extracted_evidence must be a str, got {type(extracted_evidence).__name__}")
    return _STAGE2_TEMPLATE.format(
        extracted_evidence=extracted_evidence,
        question=instance["question"],
        choice_a=instance["choices"][0],
        choice_b=instance["choices"][1],
        choice_c=instance["choices"][2],
        choice_d=instance["choices"][3],
    )
