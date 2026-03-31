_STAGE1_TEMPLATE = (
    "Read the following context carefully.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Extract the most relevant sentences or passages from the context that "
    "would help answer this question. Copy them verbatim. "
    'If nothing is relevant, write "No relevant passage found."'
)

NO_EVIDENCE_SENTINEL = "No relevant passage found."


def build_extraction_prompt(instance: dict) -> str:
    return _STAGE1_TEMPLATE.format(
        context=instance["input"],
        question=instance["question"],
    )
