import json
import os
from pathlib import Path

RAW_DIR = Path("data/longbench_v2")
OUT_DIR = Path("data/processed_longbench")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP = {
    "narrativeqa": "single-document QA",
    "qasper": "single-document QA",
    "multifieldqa_en": "single-document QA",
    "hotpotqa": "multi-document QA",
    "2wikimqa": "multi-document QA",
    "musique": "multi-document QA",
}

def make_placeholder_choices(answer_text: str):
    return [
        answer_text,
        "None of the above",
        "Insufficient information",
        "Unknown",
    ]

def convert_file(path):
    with open(path, "r") as f:
        raw = json.load(f)

    dataset_name = path.stem.lower()
    category = CATEGORY_MAP.get(dataset_name, "single-document QA")

    processed = []

    for i, ex in enumerate(raw):
        example_id = ex.get("_id", f"{dataset_name}-{i}")
        input_text = ex.get("input", "")
        context_text = ex.get("context", "")

        full_input = f"{input_text}\n\n{context_text}".strip()

        answers = ex.get("answers", [])
        gold = answers[0] if answers else "A"

        item = {
            "id": str(example_id),
            "input": full_input,
            "question": input_text,
            "choices": make_placeholder_choices(str(gold)),
            "answer": "A",
            "category": category,
        }

        processed.append(item)

    out_path = OUT_DIR / path.name
    with open(out_path, "w") as f:
        json.dump(processed, f, indent=2)

    print(f"Converted {path.name}")

def main():
    files = list(RAW_DIR.glob("*.json"))

    if not files:
        print("No raw files found")
        return

    for f in files:
        convert_file(f)

if __name__ == "__main__":
    main()
